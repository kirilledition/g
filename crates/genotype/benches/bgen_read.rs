use std::collections::BTreeSet;
use std::path::PathBuf;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use g_genotype::{BgenReadSession, BgenReaderCore, ChunkStatisticsPolicy, Packed8Compatibility};

const CHUNK_SIZES: [usize; 5] = [1024, 2048, 4096, 8192, 16384];
const GPU_HOST_DELIVERY_CHUNK_SIZES: [usize; 2] = [8192, 16384];
const ACCESS_ORDER_BATCH_COUNT: usize = 16;
const DOSAGE_STATISTICS_POLICY: ChunkStatisticsPolicy =
    ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: true, collect_sparse_candidate_mask: false };
const PACKED8_STATISTICS_POLICY: ChunkStatisticsPolicy =
    ChunkStatisticsPolicy { retain_imputed_dosage_square_sum: false, collect_sparse_candidate_mask: true };

struct PackedBatchBenchmarkCase {
    benchmark_name: String,
    variant_start: usize,
    variant_stop: usize,
    fresh_storage: bool,
}

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
    let source_byte_count = std::fs::metadata(&bgen_path).expect("benchmark BGEN metadata should be available").len();
    let mut open_group = criterion.benchmark_group("bgen_open_and_index");
    open_group.throughput(Throughput::Bytes(source_byte_count));
    open_group.bench_function("sequential_index", |benchmark| {
        benchmark.iter(|| {
            std::hint::black_box(
                BgenReaderCore::open(std::hint::black_box(&bgen_path))
                    .expect("native Rust BGEN reader should open benchmark input"),
            );
        });
    });
    open_group.finish();
}

fn benchmark_packed_batch(
    delivery_group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    read_session: &BgenReadSession<'_>,
    compressed_layout: &g_genotype::CompressedPacked8BatchLayout,
    reader: &BgenReaderCore,
    full_sample_indices: &[usize],
    benchmark_case: &PackedBatchBenchmarkCase,
) {
    let transfer_byte_count = read_session
        .pack_compressed_packed8_batch(compressed_layout, benchmark_case.variant_start, benchmark_case.variant_stop)
        .expect("raw-DEFLATE BGEN packing should succeed")
        .raw_deflate_slab()
        .len();
    delivery_group
        .throughput(Throughput::Bytes(u64::try_from(transfer_byte_count).expect("transfer byte count should fit u64")));
    if benchmark_case.fresh_storage {
        delivery_group.bench_function(&benchmark_case.benchmark_name, |benchmark| {
            benchmark.iter_batched(
                || reader.read_session(full_sample_indices).expect("fresh full-sample BGEN read session should build"),
                |fresh_read_session| {
                    std::hint::black_box(
                        fresh_read_session
                            .pack_compressed_packed8_batch(
                                compressed_layout,
                                benchmark_case.variant_start,
                                benchmark_case.variant_stop,
                            )
                            .expect("fresh-storage raw-DEFLATE BGEN packing should succeed"),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    } else {
        delivery_group.bench_function(&benchmark_case.benchmark_name, |benchmark| {
            benchmark.iter(|| {
                std::hint::black_box(
                    read_session
                        .pack_compressed_packed8_batch(
                            compressed_layout,
                            benchmark_case.variant_start,
                            benchmark_case.variant_stop,
                        )
                        .expect("pooled raw-DEFLATE BGEN packing should succeed"),
                );
            });
        });
    }
}

fn permuted_chunk_indices(chunk_count: usize) -> Vec<usize> {
    let selected_chunk_count = chunk_count.min(ACCESS_ORDER_BATCH_COUNT);
    (0..selected_chunk_count).map(|index| (index * 17 + 7) % chunk_count).collect()
}

fn benchmark_packed_access_order(
    delivery_group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    read_session: &BgenReadSession<'_>,
    compressed_layout: &g_genotype::CompressedPacked8BatchLayout,
    chunk_specs: &[g_genotype::ChunkSpec],
) {
    let selected_chunk_count = chunk_specs.len().min(ACCESS_ORDER_BATCH_COUNT);
    let sequential_indices = (0..selected_chunk_count).collect::<Vec<_>>();
    let random_indices = permuted_chunk_indices(chunk_specs.len());
    let variant_count = sequential_indices
        .iter()
        .map(|chunk_index| {
            let chunk_spec = &chunk_specs[*chunk_index];
            chunk_spec.variant_stop_index - chunk_spec.variant_start_index
        })
        .sum::<usize>();
    delivery_group
        .throughput(Throughput::Elements(u64::try_from(variant_count).expect("variant count should fit u64")));
    for (benchmark_name, chunk_indices) in
        [("sequential_offsets", sequential_indices), ("random_offsets", random_indices)]
    {
        delivery_group.bench_function(benchmark_name, |benchmark| {
            benchmark.iter(|| {
                for chunk_index in &chunk_indices {
                    let chunk_spec = &chunk_specs[*chunk_index];
                    std::hint::black_box(
                        read_session
                            .pack_compressed_packed8_batch(
                                compressed_layout,
                                chunk_spec.variant_start_index,
                                chunk_spec.variant_stop_index,
                            )
                            .expect("raw-DEFLATE BGEN access-order packing should succeed"),
                    );
                }
            });
        });
    }
}

fn benchmark_gpu_host_delivery(
    criterion: &mut Criterion,
    reader: &BgenReaderCore,
    read_session: &BgenReadSession<'_>,
    full_sample_indices: &[usize],
) {
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
        benchmark_packed_batch(
            &mut delivery_group,
            read_session,
            &compressed_layout,
            reader,
            full_sample_indices,
            &PackedBatchBenchmarkCase {
                benchmark_name: format!("raw_deflate_pack/{chunk_size}"),
                variant_start,
                variant_stop,
                fresh_storage: false,
            },
        );
        benchmark_packed_batch(
            &mut delivery_group,
            read_session,
            &compressed_layout,
            reader,
            full_sample_indices,
            &PackedBatchBenchmarkCase {
                benchmark_name: format!("raw_deflate_pack_fresh_storage/{chunk_size}"),
                variant_start,
                variant_stop,
                fresh_storage: true,
            },
        );
        if let Some(tail_chunk_spec) = chunk_specs.last()
            && tail_chunk_spec.variant_stop_index - tail_chunk_spec.variant_start_index < chunk_size
        {
            benchmark_packed_batch(
                &mut delivery_group,
                read_session,
                &compressed_layout,
                reader,
                full_sample_indices,
                &PackedBatchBenchmarkCase {
                    benchmark_name: format!("raw_deflate_pack_tail/{chunk_size}"),
                    variant_start: tail_chunk_spec.variant_start_index,
                    variant_stop: tail_chunk_spec.variant_stop_index,
                    fresh_storage: false,
                },
            );
            benchmark_packed_batch(
                &mut delivery_group,
                read_session,
                &compressed_layout,
                reader,
                full_sample_indices,
                &PackedBatchBenchmarkCase {
                    benchmark_name: format!("raw_deflate_pack_tail_fresh_storage/{chunk_size}"),
                    variant_start: tail_chunk_spec.variant_start_index,
                    variant_stop: tail_chunk_spec.variant_stop_index,
                    fresh_storage: true,
                },
            );
        }
        if chunk_size == 16_384 {
            benchmark_packed_access_order(&mut delivery_group, read_session, &compressed_layout, &chunk_specs);
        }
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
        benchmark_gpu_host_delivery(criterion, &reader, &full_sample_session, &full_sample_indices);
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
