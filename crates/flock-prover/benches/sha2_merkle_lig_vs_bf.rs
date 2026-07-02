//! SHA-256 Merkle-path prove: BaseFold (`prove_merkle_path`) vs Ligerito
//! (`prove_merkle_path_ligerito`) head-to-head.
//!
//! Run twice for ST and MT:
//!   cargo bench --bench sha2_merkle_lig_vs_bf                       # MT (default)
//!   RAYON_NUM_THREADS=1 cargo bench --bench sha2_merkle_lig_vs_bf   # ST
//!
//! By default benches at m=31 (K=65536 SHA-256 compressions; K_LOG=15, so
//! m = 15 + 16). Override with `SHA2_K=<n_blocks>` (power of 2 ≥ 8; Ligerito
//! needs a registered config, i.e. m ≥ 22 → K ≥ 128).
//! `FLOCK_BENCH_RUNS=<n>` controls best-of-n (default 3).
//!
//! Set `MERKLE_TRACE=1` to print per-phase prover timing
//! (base_r1cs / fold_slots / shift_sumcheck / open_batch) for each backend.
//!
//! For each backend prints best-of-N prove time, peak memory, verify time,
//! and serialized size of `(commitment, proof)`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::sha2::{
    Compression, K_LOG, SHA256_IV, Sha256HybridSetup, min_n_blocks_log, sha256_compress,
};

// Peak-heap tracker (mirrors blake3_chain_lig_vs_bf.rs).
struct PeakAlloc;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(c, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CUR.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                let c = CUR.fetch_add(new - l.size(), Ordering::Relaxed) + (new - l.size());
                PEAK.fetch_max(c, Ordering::Relaxed);
            } else {
                CUR.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}
#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;
fn reset_peak() {
    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
}
fn peak_mb() -> f64 {
    PEAK.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn nx(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Build an honest SHA-256 Merkle path of `n` compressions (mirrors
/// `sha2_merkle_proof.rs::honest_merkle_path`). Returns `(blocks, leaf, root, b_bits)`.
fn honest_merkle_path(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8], Vec<bool>) {
    let mut rng = Rng::new(seed);
    let leaf: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
    let mut b_bits = vec![false; n];
    for bit in b_bits.iter_mut().skip(1) {
        *bit = rng.nx() & 1 == 1;
    }
    let mut blocks = Vec::with_capacity(n);
    let mut current = leaf;
    for i in 0..n {
        let sibling: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
        let mut m = [0u32; 16];
        if !b_bits[i] {
            m[..8].copy_from_slice(&current);
            m[8..].copy_from_slice(&sibling);
        } else {
            m[..8].copy_from_slice(&sibling);
            m[8..].copy_from_slice(&current);
        }
        blocks.push((SHA256_IV, m));
        current = sha256_compress(&SHA256_IV, &m);
    }
    let root = current;
    (blocks, leaf, root, b_bits)
}

fn fmt_ms(s: f64) -> String {
    let ms = s * 1000.0;
    if ms < 1.0 {
        format!("{:>8.2} µs", s * 1e6)
    } else if ms < 1000.0 {
        format!("{:>8.2} ms", ms)
    } else {
        format!("{:>8.2} s ", s)
    }
}

fn fmt_kb(b: usize) -> String {
    if b >= 1024 * 1024 {
        format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

fn bench_block(n_blocks: usize, n_runs: usize, threads_label: &str) {
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    let n_slots = 1usize << n_log;
    let witness_bytes = (1usize << m) / 8;

    println!(
        "\n=== K = {n_blocks:>6}  (m = {m}, slots = {n_slots}, witness = {} MB, {threads_label}) ===",
        witness_bytes >> 20
    );

    let setup = Sha256HybridSetup::new(n_blocks);
    let (blocks, leaf, root, b) = honest_merkle_path(n_blocks, 0xD15EA5E ^ n_blocks as u64);

    // ============ BaseFold (prove_merkle_path) ============
    {
        // Warm-up.
        let mut ch_p = FsChallenger::new(b"flock-merkle-bench-v0");
        let (p, _) = setup.prove_merkle_path(&blocks, &b, &mut ch_p);
        black_box(&p);

        let mut best = f64::INFINITY;
        for _ in 0..n_runs {
            let mut ch_p = FsChallenger::new(b"flock-merkle-bench-v0");
            let t0 = Instant::now();
            let (p, _) = setup.prove_merkle_path(&blocks, &b, &mut ch_p);
            best = best.min(t0.elapsed().as_secs_f64());
            black_box(&p);
        }

        // One canonical run for verify + size measurement.
        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-merkle-bench-v0");
        let (proof, commitment) = setup.prove_merkle_path(&blocks, &b, &mut ch_p);
        let peak_after_prove = peak_mb();
        let mut ch_v = FsChallenger::new(b"flock-merkle-bench-v0");
        setup
            .verify_merkle_path(&commitment, &proof, &leaf, &root, &b, &mut ch_v)
            .expect("bf merkle verify");
        let mut ch_v = FsChallenger::new(b"flock-merkle-bench-v0");
        let t0 = Instant::now();
        setup
            .verify_merkle_path(&commitment, &proof, &leaf, &root, &b, &mut ch_v)
            .expect("bf merkle verify");
        let verify_t = t0.elapsed().as_secs_f64();
        let size = bincode::serialize(&(&commitment, &proof))
            .expect("serialize bf merkle proof")
            .len();
        black_box(&proof);

        println!(
            "  BaseFold:  prove = {}   verify = {}   size = {}   peak = {:.2} MB",
            fmt_ms(best),
            fmt_ms(verify_t),
            fmt_kb(size),
            peak_after_prove,
        );
    }

    // ============ Ligerito (prove_merkle_path_ligerito) ============
    {
        let mut ch_p = FsChallenger::new(b"flock-merkle-bench-v0");
        let (p, _) = setup.prove_merkle_path_ligerito(&blocks, &b, &mut ch_p);
        black_box(&p);

        let mut best = f64::INFINITY;
        for _ in 0..n_runs {
            let mut ch_p = FsChallenger::new(b"flock-merkle-bench-v0");
            let t0 = Instant::now();
            let (p, _) = setup.prove_merkle_path_ligerito(&blocks, &b, &mut ch_p);
            best = best.min(t0.elapsed().as_secs_f64());
            black_box(&p);
        }

        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-merkle-bench-v0");
        let (proof, commitment) = setup.prove_merkle_path_ligerito(&blocks, &b, &mut ch_p);
        let peak_after_prove = peak_mb();
        let mut ch_v = FsChallenger::new(b"flock-merkle-bench-v0");
        let t0 = Instant::now();
        setup
            .verify_merkle_path_ligerito(&commitment, &proof, &leaf, &root, &b, &mut ch_v)
            .expect("lig merkle verify");
        let verify_t = t0.elapsed().as_secs_f64();
        let size = bincode::serialize(&(&commitment, &proof))
            .expect("serialize lig merkle proof")
            .len();
        black_box(&proof);

        println!(
            "  Ligerito:  prove = {}   verify = {}   size = {}   peak = {:.2} MB",
            fmt_ms(best),
            fmt_ms(verify_t),
            fmt_kb(size),
            peak_after_prove,
        );
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = rayon::current_num_threads();
    let label_owned = if threads == 1 {
        "ST".to_string()
    } else {
        format!("MT, {threads} threads")
    };
    let label = label_owned.as_str();

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!("SHA-256 prove_merkle_path: BaseFold vs Ligerito head-to-head — {label}");

    // Default K=65536 → m=31. Override with SHA2_K=<value> (power of 2 ≥ 128).
    let ks: Vec<usize> = match std::env::var("SHA2_K") {
        Ok(s) => s
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().expect("SHA2_K: integer K (n_blocks)"))
            .collect(),
        Err(_) => vec![65536],
    };
    let n_runs: usize = std::env::var("FLOCK_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    for &n in &ks {
        bench_block(n, n_runs, label);
    }
}
