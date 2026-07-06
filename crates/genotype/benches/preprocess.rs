use criterion::{Criterion, criterion_group, criterion_main};
use g_genotype as native_genotype;
use std::hint;

fn dense_variant_major_dosages(selected_variant_count: usize, selected_sample_count: usize) -> Vec<f32> {
    let mut dosage_values = Vec::with_capacity(selected_variant_count * selected_sample_count);
    for variant_index in 0..selected_variant_count {
        for sample_index in 0..selected_sample_count {
            let raw_value = ((variant_index * 31) + (sample_index * 17)) % 511;
            let raw_value_u16 = u16::try_from(raw_value).expect("synthetic dosage value should fit u16");
            dosage_values.push(f32::from(raw_value_u16) / 255.0_f32);
        }
    }
    dosage_values
}

fn benchmark_variant_major_summary(criterion: &mut Criterion) {
    let selected_variant_count = 64_usize;
    let mut group = criterion.benchmark_group("preprocess_variant_major_summary");
    for selected_sample_count in [1024_usize, 2048, 4096, 8192, 16384] {
        let dosage_values = dense_variant_major_dosages(selected_variant_count, selected_sample_count);
        group.throughput(criterion::Throughput::Elements((selected_variant_count * selected_sample_count) as u64));
        group.bench_function(selected_sample_count.to_string(), |bencher| {
            bencher.iter(|| {
                hint::black_box(
                    native_genotype::summarize_variant_major_dosage_matrix(
                        hint::black_box(&dosage_values),
                        selected_sample_count,
                        selected_variant_count,
                    )
                    .expect("variant-major summary should compute"),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_variant_major_summary);
criterion_main!(benches);
