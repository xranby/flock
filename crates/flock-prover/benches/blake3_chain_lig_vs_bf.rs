//! BLAKE3 hash-chain prove: BaseFold (`prove_chain_basefold`) vs Ligerito
//! (`prove_chain`) head-to-head.
//!
//! Run twice for ST and MT:
//!   cargo bench --bench blake3_chain_lig_vs_bf                       # MT (default)
//!   RAYON_NUM_THREADS=1 cargo bench --bench blake3_chain_lig_vs_bf   # ST
//!
//! By default benches at m=31 (K=131072 BLAKE3 compressions). Override with
//! `BLAKE3_K=<n_blocks>` (must be a power of 2 ≥ 8 — the chain protocol
//! requires n_blocks to exactly fill n_block_slots, no padding).
//! `FLOCK_BENCH_RUNS=<n>` controls best-of-n (default 3).
//!
//! Set `CHAIN_TRACE=1` to also print per-phase prover timing
//! (base_r1cs / fold_in_out / shift_sumcheck / open_batch) for each backend.
//!
//! For each backend prints best-of-N prove time, peak memory, verify time,
//! and serialized proof size (whole `ChainProofBundle`, incl. public endpoints).

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::proof_io::{ChainProofBundle, ChainProofBundleLigerito, HashKind};
use flock_prover::r1cs_hashes::blake3::{
    Blake3Setup, Compression, K_LOG, blake3_compress, cv_to_phys_bits, min_n_blocks_log,
};

// Peak-heap tracker (mirrors blake3_lig_vs_bf.rs).
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

/// Build an honest BLAKE3 chain of `n` compressions (mirrors
/// `blake3_chain_proof.rs::honest_chain`). Returns `(blocks, cv_0, cv_last)`.
fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
    let mut rng = Rng::new(seed);
    let mut cv: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
    let cv0 = cv;
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let m: [u32; 16] = std::array::from_fn(|_| rng.nx() as u32);
        let block: Compression = (cv, m, 0u64, 64u32, 0u32);
        blocks.push(block);
        let st = blake3_compress(&cv, &m, 0u64, 64u32, 0u32);
        cv = st[0..8].try_into().unwrap();
    }
    let cv_last = cv;
    (blocks, cv0, cv_last)
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

    let setup = Blake3Setup::new(n_blocks);
    // Honest chain (blocks[i+1].cv == compress(blocks[i])[0..8]); reused across
    // runs — the prove path is deterministic in the witness, so a single chain
    // is fine for best-of-n timing.
    let (blocks, cv_0, cv_last) = honest_chain(n_blocks, 0xC0FFEE_BEEF ^ n_blocks as u64);
    let cv_0_phys = cv_to_phys_bits(&cv_0);
    let cv_last_phys = cv_to_phys_bits(&cv_last);

    // ============ BaseFold (prove_chain_basefold) ============
    {
        // Warm-up.
        let mut ch_p = FsChallenger::new(b"flock-chain-bench-v0");
        let (p, _) = setup.prove_chain_basefold(&blocks, &mut ch_p);
        black_box(&p);

        let mut best = f64::INFINITY;
        for _ in 0..n_runs {
            let mut ch_p = FsChallenger::new(b"flock-chain-bench-v0");
            let t0 = Instant::now();
            let (p, _) = setup.prove_chain_basefold(&blocks, &mut ch_p);
            best = best.min(t0.elapsed().as_secs_f64());
            black_box(&p);
        }

        // One canonical run for verify + size measurement.
        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-chain-bench-v0");
        let (proof, commitment) = setup.prove_chain_basefold(&blocks, &mut ch_p);
        let peak_after_prove = peak_mb();
        let mut ch_v = FsChallenger::new(b"flock-chain-bench-v0");
        let t0 = Instant::now();
        setup
            .verify_chain_basefold(&commitment, &proof, &cv_0, &cv_last, &mut ch_v)
            .expect("bf chain verify");
        let verify_t = t0.elapsed().as_secs_f64();
        let bundle = ChainProofBundle {
            hash_kind: HashKind::Blake3,
            commitment,
            proof,
            cv_0_phys: cv_0_phys.clone(),
            cv_last_phys: cv_last_phys.clone(),
        };
        let size = bundle.to_bytes().len();
        black_box(&bundle);

        println!(
            "  BaseFold:  prove = {}   verify = {}   size = {}   peak = {:.2} MB",
            fmt_ms(best),
            fmt_ms(verify_t),
            fmt_kb(size),
            peak_after_prove,
        );
    }

    // ============ Ligerito (prove_chain) ============
    {
        let mut ch_p = FsChallenger::new(b"flock-chain-bench-v0");
        let (p, _) = setup.prove_chain(&blocks, &mut ch_p);
        black_box(&p);

        let mut best = f64::INFINITY;
        for _ in 0..n_runs {
            let mut ch_p = FsChallenger::new(b"flock-chain-bench-v0");
            let t0 = Instant::now();
            let (p, _) = setup.prove_chain(&blocks, &mut ch_p);
            best = best.min(t0.elapsed().as_secs_f64());
            black_box(&p);
        }

        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-chain-bench-v0");
        let (proof, commitment) = setup.prove_chain(&blocks, &mut ch_p);
        let peak_after_prove = peak_mb();
        let mut ch_v = FsChallenger::new(b"flock-chain-bench-v0");
        let t0 = Instant::now();
        setup
            .verify_chain(&commitment, &proof, &cv_0, &cv_last, &mut ch_v)
            .expect("lig chain verify");
        let verify_t = t0.elapsed().as_secs_f64();
        let bundle = ChainProofBundleLigerito {
            hash_kind: HashKind::Blake3,
            commitment,
            proof,
            cv_0_phys: cv_0_phys.clone(),
            cv_last_phys: cv_last_phys.clone(),
        };
        let size = bundle.to_bytes().len();
        black_box(&bundle);

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
    println!("BLAKE3 prove_chain: BaseFold vs Ligerito head-to-head — {label}");

    // Default K=131072 → m=31. Override with BLAKE3_K=<value> (power of 2 ≥ 8).
    let ks: Vec<usize> = match std::env::var("BLAKE3_K") {
        Ok(s) => s
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().expect("BLAKE3_K: integer K (n_blocks)"))
            .collect(),
        Err(_) => vec![131072],
    };
    let n_runs: usize = std::env::var("FLOCK_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    for &n in &ks {
        bench_block(n, n_runs, label);
    }
}
