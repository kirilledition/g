//! Linux process-residency probe for repeated BGEN opens.
//!
//! Run on a compute node with `cargo run --release -p g-genotype --example
//! bgen_index_memory -- /absolute/source.bgen`. RSS includes allocator-retained
//! pages, so eviction need not return its storage to the operating system.

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use g_genotype::BgenReaderCore;

fn report_snapshot(stage: &str) -> std::io::Result<()> {
    let status = fs::read_to_string("/proc/self/status")?;
    let resident_kibibytes = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:")?.split_whitespace().next()?.parse::<u64>().ok())
        .ok_or_else(|| std::io::Error::other("Linux process status has no resident-memory value"))?;
    let descriptor_count = fs::read_dir("/proc/self/fd")?.count();
    println!(
        "{{\"stage\":\"{stage}\",\"resident_bytes\":{},\"open_descriptor_count\":{descriptor_count}}}",
        resident_kibibytes * 1024,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source_path = PathBuf::from(std::env::args_os().nth(1).ok_or("Pass one BGEN source path")?);
    report_snapshot("before_first_open")?;
    let started_at = Instant::now();
    let reader = BgenReaderCore::open(&source_path)?;
    println!(
        "{{\"stage\":\"first_open\",\"elapsed_seconds\":{},\"variant_count\":{}}}",
        started_at.elapsed().as_secs_f64(),
        reader.variant_count(),
    );
    report_snapshot("first_reader_live")?;
    drop(reader);
    report_snapshot("probation_without_reader")?;

    // Report probation separately; neither the wait nor its fresh promotion
    // parse belongs to cached-reopen timing or retained-index snapshots.
    let probation_started_at = Instant::now();
    std::thread::sleep(Duration::from_secs(2));
    println!("{{\"stage\":\"probation_wait\",\"elapsed_seconds\":{}}}", probation_started_at.elapsed().as_secs_f64());
    let promotion_started_at = Instant::now();
    drop(BgenReaderCore::open(&source_path)?);
    println!("{{\"stage\":\"promotion_parse\",\"elapsed_seconds\":{}}}", promotion_started_at.elapsed().as_secs_f64());
    report_snapshot("cache_without_reader")?;

    let started_at = Instant::now();
    for _iteration in 0..100 {
        std::hint::black_box(BgenReaderCore::open(&source_path)?);
    }
    println!("{{\"stage\":\"100_reopens\",\"elapsed_seconds\":{}}}", started_at.elapsed().as_secs_f64());
    report_snapshot("after_repeated_opens")?;

    let eviction_path = std::env::temp_dir().join(format!("g-bgen-index-memory-{}.bgen", std::process::id()));
    let mut header = vec![0_u8; 24];
    header[0..4].copy_from_slice(&20_u32.to_le_bytes());
    header[4..8].copy_from_slice(&20_u32.to_le_bytes());
    header[16..20].copy_from_slice(b"bgen");
    header[20..24].copy_from_slice(&(2_u32 << 2).to_le_bytes());
    fs::write(&eviction_path, header)?;
    let eviction_result = BgenReaderCore::open(&eviction_path);
    fs::remove_file(&eviction_path)?;
    drop(eviction_result?);
    report_snapshot("after_eviction")?;
    Ok(())
}
