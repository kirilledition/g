use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use _core::genotype::bgen::{BgenReaderCore, set_bgen_row_major_direct_write_enabled};

const CHUNK_SIZES: [usize; 5] = [1024, 2048, 4096, 8192, 16384];

fn benchmark_bgen_path() -> PathBuf {
    std::env::var_os("GWAS_ENGINE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data"))
        .join("1kg_chr22_full.bgen")
}

fn prepare_full_sample_selection(reader: &BgenReaderCore) {
    let all_sample_indices: Vec<i64> = (0..reader.sample_count())
        .map(|sample_index| i64::try_from(sample_index).expect("sample index should fit i64"))
        .collect();
    reader
        .prepare_sample_selection(&all_sample_indices)
        .expect("prepared sample selection should succeed for benchmark input");
}

fn prepare_contiguous_prefix_sample_selection(reader: &BgenReaderCore, selected_sample_count: usize) {
    let sample_indices: Vec<i64> = (0..selected_sample_count)
        .map(|sample_index| i64::try_from(sample_index).expect("sample index should fit i64"))
        .collect();
    reader
        .prepare_sample_selection(&sample_indices)
        .expect("prepared contiguous sample selection should succeed for benchmark input");
}

fn prepare_strided_half_sample_selection(reader: &BgenReaderCore) -> usize {
    let sample_indices: Vec<i64> = (0..reader.sample_count())
        .step_by(2)
        .map(|sample_index| i64::try_from(sample_index).expect("sample index should fit i64"))
        .collect();
    let selected_sample_count = sample_indices.len();
    reader
        .prepare_sample_selection(&sample_indices)
        .expect("prepared non-contiguous sample selection should succeed for benchmark input");
    selected_sample_count
}

fn benchmark_preprocessed_variant_major_read(
    criterion: &mut Criterion,
    reader: &BgenReaderCore,
    group_name: &str,
    selected_sample_count: usize,
) {
    let mut variant_group = criterion.benchmark_group(group_name);
    for chunk_size in CHUNK_SIZES {
        let selected_variant_count = chunk_size.min(reader.variant_count());
        let mut output_buffer = vec![0.0_f32; selected_sample_count * selected_variant_count];
        variant_group.throughput(Throughput::Elements(
            u64::try_from(selected_variant_count).expect("variant count should fit u64"),
        ));
        variant_group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &selected_variant_count,
            |benchmark, selected_variant_count| {
                benchmark.iter(|| {
                    let chunk_stats = reader
                        .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                            0,
                            *selected_variant_count,
                            output_buffer.as_mut_ptr() as usize,
                            output_buffer.len(),
                        )
                        .expect("prepared native Rust variant-major BGEN read should succeed");
                    std::hint::black_box(chunk_stats.observation_count.len());
                });
            },
        );
    }
    variant_group.finish();
}

fn benchmark_preprocessed_variant_major_packed8_read(
    criterion: &mut Criterion,
    reader: &BgenReaderCore,
    group_name: &str,
    selected_sample_count: usize,
) {
    let mut variant_group = criterion.benchmark_group(group_name);
    for chunk_size in CHUNK_SIZES {
        let selected_variant_count = chunk_size.min(reader.variant_count());
        let mut output_buffer = vec![0_u8; selected_sample_count * selected_variant_count * 2];
        variant_group.throughput(Throughput::Elements(
            u64::try_from(selected_variant_count).expect("variant count should fit u64"),
        ));
        variant_group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &selected_variant_count,
            |benchmark, selected_variant_count| {
                benchmark.iter(|| {
                    let chunk_stats = reader
                        .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
                            0,
                            *selected_variant_count,
                            output_buffer.as_mut_ptr() as usize,
                            output_buffer.len(),
                        )
                        .expect("prepared native Rust packed8 BGEN read should succeed");
                    std::hint::black_box(chunk_stats.observation_count.len());
                });
            },
        );
    }
    variant_group.finish();
}

fn benchmark_row_major_direct_write_mode(
    criterion: &mut Criterion,
    reader: &BgenReaderCore,
    group_name: &str,
    direct_write_enabled: bool,
) {
    set_bgen_row_major_direct_write_enabled(direct_write_enabled);
    let mut variant_group = criterion.benchmark_group(group_name);
    for chunk_size in CHUNK_SIZES {
        let selected_variant_count = chunk_size.min(reader.variant_count());
        let mut output_buffer = vec![0.0_f32; reader.sample_count() * selected_variant_count];
        variant_group.throughput(Throughput::Elements(
            u64::try_from(selected_variant_count).expect("variant count should fit u64"),
        ));
        variant_group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &selected_variant_count,
            |benchmark, selected_variant_count| {
                benchmark.iter(|| {
                    reader
                        .read_dosage_f32_into_address_prepared(
                            0,
                            *selected_variant_count,
                            output_buffer.as_mut_ptr() as usize,
                            output_buffer.len(),
                        )
                        .expect("prepared native Rust row-major BGEN read should succeed");
                });
            },
        );
    }
    variant_group.finish();
    set_bgen_row_major_direct_write_enabled(false);
}

#[allow(clippy::too_many_lines)]
fn benchmark_native_bgen_read(criterion: &mut Criterion) {
    let bgen_path = benchmark_bgen_path();
    if !bgen_path.exists() {
        return;
    }

    let reader = BgenReaderCore::open(&bgen_path, false).expect("native Rust BGEN reader should open benchmark input");
    prepare_full_sample_selection(&reader);

    {
        let mut variant_group = criterion.benchmark_group("bgen_read_full_sample_variants");
        for chunk_size in CHUNK_SIZES {
            let selected_variant_count = chunk_size.min(reader.variant_count());
            let mut output_buffer = vec![0.0_f32; reader.sample_count() * selected_variant_count];
            variant_group.throughput(Throughput::Elements(
                u64::try_from(selected_variant_count).expect("variant count should fit u64"),
            ));
            variant_group.bench_with_input(
                BenchmarkId::from_parameter(chunk_size),
                &selected_variant_count,
                |benchmark, selected_variant_count| {
                    benchmark.iter(|| {
                        reader
                            .read_dosage_f32_into_address_prepared(
                                0,
                                *selected_variant_count,
                                output_buffer.as_mut_ptr() as usize,
                                output_buffer.len(),
                            )
                            .expect("prepared native Rust BGEN read should succeed");
                    });
                },
            );
        }
        variant_group.finish();
    }
    {
        let mut byte_group = criterion.benchmark_group("bgen_read_full_sample_bytes");
        for chunk_size in CHUNK_SIZES {
            let selected_variant_count = chunk_size.min(reader.variant_count());
            let mut output_buffer = vec![0.0_f32; reader.sample_count() * selected_variant_count];
            let output_byte_count = reader
                .sample_count()
                .checked_mul(selected_variant_count)
                .and_then(|value_count| value_count.checked_mul(std::mem::size_of::<f32>()))
                .expect("output byte count should fit usize");
            byte_group.throughput(Throughput::Bytes(
                u64::try_from(output_byte_count).expect("output byte count should fit u64"),
            ));
            byte_group.bench_with_input(
                BenchmarkId::from_parameter(chunk_size),
                &selected_variant_count,
                |benchmark, selected_variant_count| {
                    benchmark.iter(|| {
                        reader
                            .read_dosage_f32_into_address_prepared(
                                0,
                                *selected_variant_count,
                                output_buffer.as_mut_ptr() as usize,
                                output_buffer.len(),
                            )
                            .expect("prepared native Rust BGEN read should succeed");
                    });
                },
            );
        }
        byte_group.finish();
    }

    benchmark_preprocessed_variant_major_read(
        criterion,
        &reader,
        "bgen_preprocessed_variant_major_trusted_disabled",
        reader.sample_count(),
    );

    prepare_full_sample_selection(&reader);
    benchmark_row_major_direct_write_mode(criterion, &reader, "bgen_row_major_tile_copy", false);
    benchmark_row_major_direct_write_mode(criterion, &reader, "bgen_row_major_direct_write", true);

    let contiguous_subset_sample_count = reader.sample_count() / 2;
    prepare_contiguous_prefix_sample_selection(&reader, contiguous_subset_sample_count);
    benchmark_preprocessed_variant_major_read(
        criterion,
        &reader,
        "bgen_preprocessed_variant_major_contiguous_subset_trusted_disabled",
        contiguous_subset_sample_count,
    );

    let strided_subset_sample_count = prepare_strided_half_sample_selection(&reader);
    benchmark_preprocessed_variant_major_read(
        criterion,
        &reader,
        "bgen_preprocessed_variant_major_strided_subset_trusted_disabled",
        strided_subset_sample_count,
    );

    let trusted_reader =
        BgenReaderCore::open(&bgen_path, true).expect("trusted native Rust BGEN reader should open benchmark input");
    prepare_full_sample_selection(&trusted_reader);
    if trusted_reader.validate_trusted_no_missing_diploid().is_ok() {
        benchmark_preprocessed_variant_major_read(
            criterion,
            &trusted_reader,
            "bgen_preprocessed_variant_major_trusted_no_missing_diploid",
            trusted_reader.sample_count(),
        );
        benchmark_preprocessed_variant_major_packed8_read(
            criterion,
            &trusted_reader,
            "bgen_preprocessed_variant_major_packed8_trusted_no_missing_diploid",
            trusted_reader.sample_count(),
        );

        let contiguous_subset_sample_count = trusted_reader.sample_count() / 2;
        prepare_contiguous_prefix_sample_selection(&trusted_reader, contiguous_subset_sample_count);
        benchmark_preprocessed_variant_major_read(
            criterion,
            &trusted_reader,
            "bgen_preprocessed_variant_major_contiguous_subset_trusted_no_missing_diploid",
            contiguous_subset_sample_count,
        );
        benchmark_preprocessed_variant_major_packed8_read(
            criterion,
            &trusted_reader,
            "bgen_preprocessed_variant_major_packed8_contiguous_subset_trusted_no_missing_diploid",
            contiguous_subset_sample_count,
        );

        let strided_subset_sample_count = prepare_strided_half_sample_selection(&trusted_reader);
        benchmark_preprocessed_variant_major_read(
            criterion,
            &trusted_reader,
            "bgen_preprocessed_variant_major_strided_subset_trusted_no_missing_diploid",
            strided_subset_sample_count,
        );
        benchmark_preprocessed_variant_major_packed8_read(
            criterion,
            &trusted_reader,
            "bgen_preprocessed_variant_major_packed8_strided_subset_trusted_no_missing_diploid",
            strided_subset_sample_count,
        );
    }
}

criterion_group!(benches, benchmark_native_bgen_read);
criterion_main!(benches);
