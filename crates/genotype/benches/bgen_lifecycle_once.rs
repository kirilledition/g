use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use g_genotype::{BGEN_OWNED_SNAPSHOT_MAXIMUM_BYTE_COUNT, BgenReaderCore, ChunkStatisticsPolicy, Packed8Compatibility};

const VARIANT_CHUNK_SIZE: usize = 16_384;
const PACKED8_STATISTICS_POLICY: ChunkStatisticsPolicy =
    ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: true };

fn benchmark_bgen_path() -> PathBuf {
    std::env::var_os("GWAS_ENGINE_BGEN_BENCHMARK_PATH").map_or_else(
        || {
            std::env::var_os("GWAS_ENGINE_DATA_DIR")
                .map_or_else(|| PathBuf::from("data"), PathBuf::from)
                .join("1kg_chr22_full.bgen")
        },
        PathBuf::from,
    )
}

#[cfg(not(feature = "benchmark-positioned-source"))]
fn open_benchmark_reader(bgen_path: &Path) -> BgenReaderCore {
    BgenReaderCore::open(bgen_path).expect("native Rust BGEN lifecycle reader should open")
}

#[cfg(feature = "benchmark-positioned-source")]
fn open_benchmark_reader(bgen_path: &Path) -> BgenReaderCore {
    BgenReaderCore::open_positioned_for_benchmark(bgen_path)
        .expect("native Rust positioned BGEN lifecycle reader should open")
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn main() {
    let bgen_path = benchmark_bgen_path();
    let source_byte_count = std::fs::metadata(&bgen_path).expect("benchmark BGEN metadata should be available").len();
    let xdg_cache_home_configured = std::env::var_os("XDG_CACHE_HOME").is_some_and(|cache_home| !cache_home.is_empty());
    let primed_reopen = std::env::var_os("GWAS_ENGINE_BGEN_BENCHMARK_PRIMED_REOPEN").is_some();
    let priming_reader = primed_reopen.then(|| open_benchmark_reader(&bgen_path));
    drop(priming_reader);
    let lifecycle_start = Instant::now();

    let open_start = Instant::now();
    let reader = open_benchmark_reader(&bgen_path);
    let open_elapsed = open_start.elapsed();
    let sample_count = reader.sample_count();
    let variant_count = reader.variant_count();

    let preparation_start = Instant::now();
    let sample_indices: Vec<usize> = (0..sample_count).collect();
    let read_session = reader.read_session(&sample_indices).expect("full-sample BGEN read session should build");
    let compatibility = reader.packed8_compatibility_with_cache().expect("packed8 compatibility scan should succeed");
    assert_eq!(compatibility, Packed8Compatibility::Compatible, "lifecycle input must support packed8 delivery");
    let preparation_elapsed = preparation_start.elapsed();

    let scan_start = Instant::now();
    let compute_variant_count = VARIANT_CHUNK_SIZE.min(variant_count);
    let mut variant_start = 0_usize;
    let mut batch_count = 0_usize;
    let mut tail_variant_count = 0_usize;
    let mut full_batch_elapsed = Duration::ZERO;
    let mut tail_batch_elapsed = Duration::ZERO;
    while variant_start < variant_count {
        let variant_stop = variant_start.saturating_add(VARIANT_CHUNK_SIZE).min(variant_count);
        let batch_variant_count = variant_stop - variant_start;
        let batch_start = Instant::now();
        let batch = read_session
            .decode_variant_major_batch(
                variant_start,
                variant_stop,
                compute_variant_count,
                true,
                PACKED8_STATISTICS_POLICY,
            )
            .expect("full-scan packed8 BGEN delivery should succeed");
        std::hint::black_box(batch);
        let batch_elapsed = batch_start.elapsed();
        if batch_variant_count == VARIANT_CHUNK_SIZE {
            full_batch_elapsed += batch_elapsed;
        } else {
            tail_batch_elapsed += batch_elapsed;
            tail_variant_count = batch_variant_count;
        }
        batch_count += 1;
        variant_start = variant_stop;
    }
    let scan_elapsed = scan_start.elapsed();

    let finish_start = Instant::now();
    read_session.finish().expect("BGEN lifecycle session should finish against an unchanged source");
    let finish_elapsed = finish_start.elapsed();
    let lifecycle_elapsed = lifecycle_start.elapsed();

    println!("source_byte_count={source_byte_count}");
    println!("sample_count={sample_count}");
    println!("variant_count={variant_count}");
    println!("variant_chunk_size={VARIANT_CHUNK_SIZE}");
    println!("compute_variant_count={compute_variant_count}");
    println!("batch_count={batch_count}");
    println!("tail_variant_count={tail_variant_count}");
    println!("primed_reopen={primed_reopen}");
    println!("packed8_compatibility=compatible");
    println!("xdg_cache_home_configured={xdg_cache_home_configured}");
    let source_policy = if cfg!(feature = "benchmark-positioned-source")
        || source_byte_count > BGEN_OWNED_SNAPSHOT_MAXIMUM_BYTE_COUNT
    {
        "positioned"
    } else {
        "owned_snapshot"
    };
    println!("source_policy={source_policy}");
    println!("open_and_index_seconds={:.9}", seconds(open_elapsed));
    println!("preparation_seconds={:.9}", seconds(preparation_elapsed));
    println!("packed8_full_scan_seconds={:.9}", seconds(scan_elapsed));
    println!("packed8_full_batches_seconds={:.9}", seconds(full_batch_elapsed));
    println!("packed8_tail_batch_seconds={:.9}", seconds(tail_batch_elapsed));
    println!("session_finish_seconds={:.9}", seconds(finish_elapsed));
    println!("total_lifecycle_seconds={:.9}", seconds(lifecycle_elapsed));
}
