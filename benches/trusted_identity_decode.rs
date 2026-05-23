use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use _core::genotype::bgen::benchmark_decode_trusted_identity_mode;

const SAMPLE_COUNTS: [usize; 3] = [10_000, 100_000, 500_000];
const RAW_DOSAGE_ALTERNATING_VALUES: [i32; 8] = [0, 1, 127, 128, 255, 382, 383, 510];
const RAW_DOSAGE_RARE_VARIANT_LIKE_VALUES: [i32; 9] = [0, 0, 0, 0, 0, 1, 2, 255, 510];

struct ProbabilityPattern {
    name: &'static str,
    bytes: Vec<u8>,
}

fn probability_pair_for_raw_dosage(raw_dosage_integer: i32) -> [u8; 2] {
    let reference_probability_units = 510_i32 - raw_dosage_integer;
    [
        u8::try_from(reference_probability_units / 2).expect("homozygous reference probability should fit u8"),
        u8::try_from(reference_probability_units % 2).expect("heterozygous probability should fit u8"),
    ]
}

fn repeated_raw_probability_bytes(sample_count: usize, raw_dosage_integers: &[i32]) -> Vec<u8> {
    let mut probabilities = Vec::with_capacity(sample_count * 2);
    for sample_index in 0..sample_count {
        probabilities.extend_from_slice(&probability_pair_for_raw_dosage(
            raw_dosage_integers[sample_index % raw_dosage_integers.len()],
        ));
    }
    probabilities
}

fn deterministic_random_valid_probability_bytes(sample_count: usize) -> Vec<u8> {
    let mut generator_state = 0x9E37_79B9_u32;
    let mut probabilities = Vec::with_capacity(sample_count * 2);
    for _ in 0..sample_count {
        generator_state = generator_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let homozygous_reference_probability_byte =
            u8::try_from(generator_state & 0xFF).expect("masked byte should fit u8");
        generator_state = generator_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let heterozygous_limit = 255_u16 - u16::from(homozygous_reference_probability_byte);
        let heterozygous_probability_byte = u8::try_from((generator_state & 0xFF) as u16 % (heterozygous_limit + 1))
            .expect("heterozygous probability should fit u8");
        probabilities.push(homozygous_reference_probability_byte);
        probabilities.push(heterozygous_probability_byte);
    }
    probabilities
}

fn probability_patterns(sample_count: usize) -> Vec<ProbabilityPattern> {
    vec![
        ProbabilityPattern { name: "all_dosage_zero", bytes: repeated_raw_probability_bytes(sample_count, &[0]) },
        ProbabilityPattern { name: "all_dosage_two", bytes: repeated_raw_probability_bytes(sample_count, &[510]) },
        ProbabilityPattern {
            name: "alternating_raw",
            bytes: repeated_raw_probability_bytes(sample_count, &RAW_DOSAGE_ALTERNATING_VALUES),
        },
        ProbabilityPattern {
            name: "rare_variant_like",
            bytes: repeated_raw_probability_bytes(sample_count, &RAW_DOSAGE_RARE_VARIANT_LIKE_VALUES),
        },
        ProbabilityPattern {
            name: "deterministic_random_valid",
            bytes: deterministic_random_valid_probability_bytes(sample_count),
        },
    ]
}

fn trusted_identity_decode_modes() -> Vec<&'static str> {
    let mut modes = vec!["lookup", "raw_scalar", "auto"];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            modes.push("raw_avx2");
        }
    }
    modes
}

fn benchmark_trusted_identity_decode(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("trusted_identity_decode");
    for sample_count in SAMPLE_COUNTS {
        for probability_pattern in probability_patterns(sample_count) {
            for decode_mode in trusted_identity_decode_modes() {
                let mut output_values = vec![0.0_f32; sample_count];
                let input_byte_count = probability_pattern.bytes.len();
                let output_byte_count =
                    sample_count.checked_mul(std::mem::size_of::<f32>()).expect("output byte count should fit usize");
                benchmark_group.throughput(Throughput::Bytes(
                    u64::try_from(input_byte_count + output_byte_count).expect("throughput byte count should fit u64"),
                ));
                benchmark_group.bench_with_input(
                    BenchmarkId::new(format!("{decode_mode}/{}", probability_pattern.name), sample_count),
                    &probability_pattern.bytes,
                    |benchmark, packed_probability_bytes| {
                        benchmark.iter(|| {
                            let checksum = benchmark_decode_trusted_identity_mode(
                                decode_mode,
                                packed_probability_bytes,
                                &mut output_values,
                            );
                            std::hint::black_box(checksum);
                            std::hint::black_box(&output_values);
                        });
                    },
                );
            }
        }
    }
    benchmark_group.finish();
}

criterion_group!(benches, benchmark_trusted_identity_decode);
criterion_main!(benches);
