use std::env;
use std::path::Path;

use _core::genotype::bgen::BgenReaderCore;

fn main() {
    let bgen_path = Path::new("data/1kg_chr22_full.bgen");
    if !bgen_path.exists() {
        eprintln!("missing benchmark BGEN: {}", bgen_path.display());
        std::process::exit(1);
    }

    let mode_name = env::var("G_BGEN_SIMD").unwrap_or_else(|_| "auto".to_string());
    let chunk_size =
        env::var("G_BGEN_PROFILE_CHUNK_SIZE").ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(16_384);
    let reader = BgenReaderCore::open(bgen_path, true).expect("trusted BGEN reader should open");
    reader.validate_trusted_no_missing_diploid().expect("trusted validation should pass");
    let all_sample_indices: Vec<i64> = (0..reader.sample_count())
        .map(|sample_index| i64::try_from(sample_index).expect("sample index should fit i64"))
        .collect();
    reader.prepare_sample_selection(&all_sample_indices).expect("identity sample selection should prepare");

    let selected_variant_count = chunk_size.min(reader.variant_count());
    let mut output_values = vec![0.0_f32; reader.sample_count() * selected_variant_count];
    reader.reset_profile();
    let chunk_stats = reader
        .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
            0,
            selected_variant_count,
            output_values.as_mut_ptr() as usize,
            output_values.len(),
        )
        .expect("trusted variant-major read should succeed");
    let profile_snapshot = reader.profile_snapshot();
    println!(
        "mode={mode_name} chunk_size={selected_variant_count} variants={} decompression_ns={} probability_decode_ns={} output_write_ns={} variant_decode_count={} output_byte_count={} dosage_rows={}",
        selected_variant_count,
        profile_snapshot.decompression_ns,
        profile_snapshot.probability_decode_ns,
        profile_snapshot.output_write_ns,
        profile_snapshot.variant_decode_count,
        profile_snapshot.output_byte_count,
        chunk_stats.allele_one_frequency.len(),
    );
    std::hint::black_box(output_values);
}
