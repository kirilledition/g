use std::fmt::Write;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use g_genotype_contracts::{VariantMetadataColumns, VariantMetadataStore};
use g_output::NativeVariantMetadataHandle;

const BENCHMARK_ROW_COUNTS: [usize; 2] = [16_384, 9_343];

fn metadata_columns(row_count: usize) -> VariantMetadataColumns {
    let dictionary: Box<[Arc<str>]> = ["22", "A", "C"].map(Arc::<str>::from).into();
    let mut identifier_text = String::new();
    let mut identifier_offsets = vec![0_u32];
    for variant_index in 0..row_count {
        write!(identifier_text, "22:variant-{variant_index}-β").expect("writing a String succeeds");
        identifier_offsets.push(u32::try_from(identifier_text.len()).expect("benchmark identifiers fit uint32"));
    }
    let store = Arc::new(
        VariantMetadataStore::from_parts(
            dictionary,
            vec![0_u32; row_count].into_boxed_slice(),
            identifier_text.into_boxed_str(),
            identifier_offsets.into_boxed_slice(),
            (0..row_count)
                .map(|index| 100_i64 + i64::try_from(index).expect("benchmark position fits int64"))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            vec![1_u32; row_count].into_boxed_slice(),
            vec![2_u32; row_count].into_boxed_slice(),
        )
        .expect("benchmark metadata store should satisfy its invariants"),
    );
    VariantMetadataColumns::new(store, 0..row_count).expect("benchmark metadata range should be valid")
}

fn benchmark_metadata_handles(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("metadata_handles");
    for row_count in BENCHMARK_ROW_COUNTS {
        let metadata = metadata_columns(row_count);
        group.throughput(Throughput::Elements(u64::try_from(row_count).expect("row count fits uint64")));
        group.bench_with_input(
            BenchmarkId::new("two_independent_groups", row_count),
            &metadata,
            |bencher, metadata| {
                bencher.iter(|| {
                    let first_group = NativeVariantMetadataHandle::try_new(std::hint::black_box(metadata))
                        .expect("benchmark metadata is valid");
                    let second_group = NativeVariantMetadataHandle::try_new(std::hint::black_box(metadata))
                        .expect("benchmark metadata is valid");
                    std::hint::black_box(first_group);
                    std::hint::black_box(second_group);
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("two_shared_groups", row_count), &metadata, |bencher, metadata| {
            bencher.iter(|| {
                let first_group = NativeVariantMetadataHandle::try_new(std::hint::black_box(metadata))
                    .expect("benchmark metadata is valid");
                let second_group = first_group.clone();
                std::hint::black_box(first_group);
                std::hint::black_box(second_group);
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(30);
    targets = benchmark_metadata_handles
}
criterion_main!(benches);
