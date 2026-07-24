use std::path::PathBuf;
use std::time::Instant;

use g_genotype::{BGEN_OWNED_SNAPSHOT_MAXIMUM_BYTE_COUNT, BgenContentSelector, BgenOpenRequest, BgenReaderCore};
use g_genotype_contracts::{BgenContentSha256, BgenSnapshotResolution};
use sha2::{Digest, Sha256};

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

fn process_resident_set_size_kibibytes() -> u64 {
    let process_status = std::fs::read_to_string("/proc/self/status").expect("process status should be readable");
    let resident_set_size_line =
        process_status.lines().find(|line| line.starts_with("VmRSS:")).expect("process status should report VmRSS");
    resident_set_size_line
        .split_ascii_whitespace()
        .nth(1)
        .expect("VmRSS should contain a numeric value")
        .parse()
        .expect("VmRSS should fit uint64")
}

fn snapshot_cache_result(snapshot_cache_hit: bool) -> &'static str {
    if snapshot_cache_hit { "strong_canonical_hit" } else { "miss" }
}

fn benchmark_content_selector(bgen_path: &std::path::Path) -> BgenContentSelector {
    let source_bytes = std::fs::read(bgen_path).expect("benchmark BGEN should be readable for content selection");
    BgenContentSelector {
        content_sha256: BgenContentSha256::from_bytes(Sha256::digest(&source_bytes).into()),
        expected_byte_count: Some(u64::try_from(source_bytes.len()).expect("BGEN byte count should fit uint64")),
    }
}

fn main() {
    let bgen_path = benchmark_bgen_path();
    let source_byte_count = std::fs::metadata(&bgen_path).expect("benchmark BGEN metadata should be available").len();
    let snapshot_cache_applies = source_byte_count <= BGEN_OWNED_SNAPSHOT_MAXIMUM_BYTE_COUNT;
    assert!(snapshot_cache_applies, "content-selected open benchmark requires an owned-snapshot input");
    let content_selector = benchmark_content_selector(&bgen_path);
    let missing_cache_hit_locator = bgen_path.with_extension("selected-cache-hit-missing");
    assert!(!missing_cache_hit_locator.exists(), "cache-hit benchmark locator must remain absent");
    let resident_set_size_before_open_kibibytes = process_resident_set_size_kibibytes();

    let (first_open_elapsed, reopen_elapsed, sample_count, variant_count, first_open_cache_hit, reopen_cache_hit) = {
        let first_open_start = Instant::now();
        let first_reader = BgenReaderCore::open_request(BgenOpenRequest {
            locator: &bgen_path,
            content_selector: Some(content_selector),
        })
        .expect("first-process selected BGEN reader should open");
        let first_open_elapsed = first_open_start.elapsed();
        let first_open_cache_hit = first_reader.opened_from_process_snapshot_cache();
        assert_eq!(first_reader.source_provenance().resolution, BgenSnapshotResolution::CapturedFromLocator);
        std::hint::black_box((first_reader.sample_count(), first_reader.variant_count()));

        let reopen_start = Instant::now();
        let reopened_reader = BgenReaderCore::open_request(BgenOpenRequest {
            locator: &missing_cache_hit_locator,
            content_selector: Some(content_selector),
        })
        .expect("same-process selected BGEN reader should reopen without locator access");
        let reopen_elapsed = reopen_start.elapsed();
        let reopen_cache_hit = reopened_reader.opened_from_process_snapshot_cache();
        assert_eq!(reopened_reader.source_provenance().resolution, BgenSnapshotResolution::ProcessSnapshotCache);
        let sample_count = reopened_reader.sample_count();
        let variant_count = reopened_reader.variant_count();
        std::hint::black_box((sample_count, variant_count));
        (first_open_elapsed, reopen_elapsed, sample_count, variant_count, first_open_cache_hit, reopen_cache_hit)
    };
    let resident_set_size_after_reader_drop_kibibytes = process_resident_set_size_kibibytes();

    let post_drop_reopen_start = Instant::now();
    let post_drop_reopened_reader = BgenReaderCore::open_request(BgenOpenRequest {
        locator: &missing_cache_hit_locator,
        content_selector: Some(content_selector),
    })
    .expect("strongly cached selected BGEN reader should reopen after earlier readers drop");
    let post_drop_reopen_elapsed = post_drop_reopen_start.elapsed();
    let post_drop_reopen_cache_hit = post_drop_reopened_reader.opened_from_process_snapshot_cache();
    assert_eq!(post_drop_reopened_reader.sample_count(), sample_count);
    assert_eq!(post_drop_reopened_reader.variant_count(), variant_count);
    assert!(!first_open_cache_hit);
    assert!(reopen_cache_hit);
    assert!(post_drop_reopen_cache_hit);
    std::hint::black_box(&post_drop_reopened_reader);

    println!("bgen_source_byte_count={source_byte_count}");
    println!("bgen_content_sha256={}", content_selector.content_sha256);
    println!("snapshot_cache_applies={snapshot_cache_applies}");
    println!("first_process_open_cache={}", snapshot_cache_result(first_open_cache_hit));
    println!("same_process_reopen_cache={}", snapshot_cache_result(reopen_cache_hit));
    println!("post_drop_reopen_cache={}", snapshot_cache_result(post_drop_reopen_cache_hit));
    println!("first_process_locator_open_count=1");
    println!("first_process_snapshot_capture_byte_count={source_byte_count}");
    println!("first_process_content_hash_byte_count={source_byte_count}");
    println!("first_process_index_parse_count=1");
    println!("same_process_reopen_locator_open_count=0");
    println!("same_process_reopen_snapshot_capture_byte_count=0");
    println!("same_process_reopen_content_hash_byte_count=0");
    println!("same_process_reopen_index_parse_count=0");
    println!("post_drop_reopen_locator_open_count=0");
    println!("post_drop_reopen_snapshot_capture_byte_count=0");
    println!("post_drop_reopen_content_hash_byte_count=0");
    println!("post_drop_reopen_index_parse_count=0");
    println!("live_shared_snapshot_byte_count={}", if snapshot_cache_applies { source_byte_count } else { 0 });
    println!(
        "cache_retained_source_byte_count_without_live_reader={}",
        if snapshot_cache_applies { source_byte_count } else { 0 }
    );
    println!("cache_retains_parsed_index_and_metadata={snapshot_cache_applies}");
    println!("resident_set_size_before_open_kibibytes={resident_set_size_before_open_kibibytes}");
    println!("resident_set_size_after_reader_drop_kibibytes={resident_set_size_after_reader_drop_kibibytes}");
    println!(
        "resident_set_size_retained_delta_kibibytes={}",
        i128::from(resident_set_size_after_reader_drop_kibibytes) - i128::from(resident_set_size_before_open_kibibytes)
    );
    println!("first_process_open_and_index_seconds={:.9}", first_open_elapsed.as_secs_f64());
    println!("same_process_reopen_and_index_seconds={:.9}", reopen_elapsed.as_secs_f64());
    println!("post_drop_reopen_and_index_seconds={:.9}", post_drop_reopen_elapsed.as_secs_f64());
}
