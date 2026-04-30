use std::path::Path;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};

use _core::bgen::BgenReaderCore;

const CHUNK_SIZES: [usize; 5] = [1024, 2048, 4096, 8192, 16384];

fn benchmark_native_bgen_read(criterion: &mut Criterion) {
    let bgen_path = Path::new("data/1kg_chr22_full.bgen");
    if !bgen_path.exists() {
        return;
    }

    let reader = BgenReaderCore::open(bgen_path, false).expect("native Rust BGEN reader should open benchmark input");
    let all_sample_indices: Vec<i64> = (0..reader.sample_count())
        .map(|sample_index| i64::try_from(sample_index).expect("sample index should fit i64"))
        .collect();
    reader
        .prepare_sample_selection(&all_sample_indices)
        .expect("prepared sample selection should succeed for benchmark input");

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
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = benchmark_native_bgen_read
}
criterion_main!(benches);
