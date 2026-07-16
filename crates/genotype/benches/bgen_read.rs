use std::collections::BTreeSet;
use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use g_genotype::{BgenReadSession, BgenReaderCore, ChunkStatisticsPolicy, Packed8Compatibility};

const CHUNK_SIZES: [usize; 5] = [1024, 2048, 4096, 8192, 16384];
const GPU_HOST_DELIVERY_CHUNK_SIZES: [usize; 2] = [8192, 16384];
const DOSAGE_STATISTICS_POLICY: ChunkStatisticsPolicy =
    ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: true, collect_sparse_candidate_mask: false };
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

fn full_sample_indices(reader: &BgenReaderCore) -> Vec<usize> {
    (0..reader.sample_count()).collect()
}

fn contiguous_prefix_sample_indices(reader: &BgenReaderCore) -> Vec<usize> {
    (0..reader.sample_count() / 2).collect()
}

fn strided_half_sample_indices(reader: &BgenReaderCore) -> Vec<usize> {
    (0..reader.sample_count()).step_by(2).collect()
}

fn benchmark_variant_major_read(
    criterion: &mut Criterion,
    reader: &BgenReaderCore,
    read_session: &BgenReadSession<'_>,
    group_name: &str,
    use_packed8: bool,
    statistics_policy: ChunkStatisticsPolicy,
) {
    let mut variant_group = criterion.benchmark_group(group_name);
    for chunk_size in CHUNK_SIZES {
        let selected_variant_count = chunk_size.min(reader.variant_count());
        variant_group.throughput(Throughput::Elements(
            u64::try_from(selected_variant_count).expect("variant count should fit u64"),
        ));
        variant_group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &selected_variant_count,
            |benchmark, selected_variant_count| {
                benchmark.iter(|| {
                    std::hint::black_box(
                        read_session
                            .decode_variant_major_batch(
                                0,
                                *selected_variant_count,
                                *selected_variant_count,
                                use_packed8,
                                statistics_policy,
                            )
                            .expect("native Rust variant-major BGEN read should succeed"),
                    );
                });
            },
        );
    }
    variant_group.finish();
}

fn benchmark_bgen_open(criterion: &mut Criterion) {
    let bgen_path = benchmark_bgen_path();
    criterion.bench_function("bgen_open_and_index", |benchmark| {
        benchmark.iter(|| {
            std::hint::black_box(
                BgenReaderCore::open(std::hint::black_box(&bgen_path))
                    .expect("native Rust BGEN reader should open benchmark input"),
            );
        });
    });
}

fn benchmark_gpu_host_delivery(criterion: &mut Criterion, reader: &BgenReaderCore, read_session: &BgenReadSession<'_>) {
    std::hint::black_box(read_session.compressed_packed8_transfer());
    let mut delivery_group = criterion.benchmark_group("bgen_gpu_host_delivery_full_samples");
    for chunk_size in GPU_HOST_DELIVERY_CHUNK_SIZES {
        let chunk_specs = reader
            .plan_chromosome_homogeneous_chunks(chunk_size, &BTreeSet::new())
            .expect("benchmark chunk plan should build");
        let Some(first_chunk_spec) = chunk_specs.first() else {
            delivery_group.finish();
            return;
        };
        let Ok(Some(compressed_layout)) = reader.plan_compressed_packed8_batch_layout(&chunk_specs) else {
            delivery_group.finish();
            return;
        };
        let variant_start = first_chunk_spec.variant_start_index;
        let variant_stop = first_chunk_spec.variant_stop_index;
        let selected_variant_count = variant_stop - variant_start;
        delivery_group.throughput(Throughput::Elements(
            u64::try_from(selected_variant_count).expect("variant count should fit u64"),
        ));
        delivery_group.bench_with_input(
            BenchmarkId::new("decoded_packed8", chunk_size),
            &selected_variant_count,
            |benchmark, selected_variant_count| {
                benchmark.iter(|| {
                    std::hint::black_box(
                        read_session
                            .decode_variant_major_batch(
                                variant_start,
                                variant_stop,
                                *selected_variant_count,
                                true,
                                PACKED8_STATISTICS_POLICY,
                            )
                            .expect("native packed8 BGEN read should succeed"),
                    );
                });
            },
        );
        delivery_group.bench_with_input(
            BenchmarkId::new("raw_deflate_pack", chunk_size),
            &selected_variant_count,
            |benchmark, _selected_variant_count| {
                benchmark.iter(|| {
                    std::hint::black_box(
                        read_session
                            .pack_compressed_packed8_batch(&compressed_layout, variant_start, variant_stop)
                            .expect("raw-DEFLATE BGEN packing should succeed"),
                    );
                });
            },
        );
    }
    delivery_group.finish();
}

#[allow(clippy::too_many_lines)]
fn benchmark_native_bgen_read(criterion: &mut Criterion) {
    let bgen_path = benchmark_bgen_path();
    let reader = BgenReaderCore::open(&bgen_path).expect("native Rust BGEN reader should open benchmark input");

    let full_sample_indices = full_sample_indices(&reader);
    let full_sample_session =
        reader.read_session(&full_sample_indices).expect("full-sample BGEN read session should build");
    benchmark_variant_major_read(
        criterion,
        &reader,
        &full_sample_session,
        "bgen_variant_major_dosage_full_samples",
        false,
        DOSAGE_STATISTICS_POLICY,
    );

    let contiguous_sample_indices = contiguous_prefix_sample_indices(&reader);
    let contiguous_sample_session =
        reader.read_session(&contiguous_sample_indices).expect("contiguous-subset BGEN read session should build");
    benchmark_variant_major_read(
        criterion,
        &reader,
        &contiguous_sample_session,
        "bgen_variant_major_dosage_contiguous_half",
        false,
        DOSAGE_STATISTICS_POLICY,
    );

    let strided_sample_indices = strided_half_sample_indices(&reader);
    let strided_sample_session =
        reader.read_session(&strided_sample_indices).expect("strided-subset BGEN read session should build");
    benchmark_variant_major_read(
        criterion,
        &reader,
        &strided_sample_session,
        "bgen_variant_major_dosage_strided_half",
        false,
        DOSAGE_STATISTICS_POLICY,
    );

    if reader.packed8_compatibility_with_cache().expect("packed8 compatibility scan should succeed")
        == Packed8Compatibility::Compatible
    {
        benchmark_gpu_host_delivery(criterion, &reader, &full_sample_session);
        benchmark_variant_major_read(
            criterion,
            &reader,
            &full_sample_session,
            "bgen_variant_major_packed8_full_samples",
            true,
            PACKED8_STATISTICS_POLICY,
        );
        benchmark_variant_major_read(
            criterion,
            &reader,
            &contiguous_sample_session,
            "bgen_variant_major_packed8_contiguous_half",
            true,
            PACKED8_STATISTICS_POLICY,
        );
        benchmark_variant_major_read(
            criterion,
            &reader,
            &strided_sample_session,
            "bgen_variant_major_packed8_strided_half",
            true,
            PACKED8_STATISTICS_POLICY,
        );
    }
}

criterion_group!(benches, benchmark_bgen_open, benchmark_native_bgen_read);
criterion_main!(benches);
