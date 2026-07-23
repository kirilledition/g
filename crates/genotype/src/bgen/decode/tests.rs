use std::mem::MaybeUninit;
use std::ptr::NonNull;

use super::super::CompressionType;
use super::super::metadata::VariantRecord;
use super::super::sample_selection::build_sample_selection;
use super::super::source::BgenByteWindow;
use super::probability::PackedProbabilityReader;
use super::*;

#[derive(Debug)]
struct DecodedTile {
    output_values: Vec<f32>,
    dosage_sum: Vec<f32>,
    dosage_square_sum: Vec<f32>,
    observation_count: Vec<i32>,
    zero_count: Option<Vec<i32>>,
    homozygous_alternate_count: Option<Vec<i32>>,
}

#[derive(Clone, Copy, Debug)]
struct EncodedProbabilityBlock<'block> {
    probability_payload: &'block [u8],
    declared_uncompressed_block_length: usize,
}

fn test_variant_record_at(probability_payload_offset: usize, probability_payload_length: usize) -> VariantRecord {
    test_variant_record_with_declared_length(
        probability_payload_offset,
        probability_payload_length,
        probability_payload_length,
    )
}

fn test_variant_record_with_declared_length(
    probability_payload_offset: usize,
    probability_payload_length: usize,
    declared_uncompressed_block_length: usize,
) -> VariantRecord {
    VariantRecord {
        probability_payload_offset: u64::try_from(probability_payload_offset)
            .expect("test probability payload offset should fit u64"),
        probability_payload_length: u32::try_from(probability_payload_length)
            .expect("test probability payload length should fit u32"),
        declared_uncompressed_block_length: u32::try_from(declared_uncompressed_block_length)
            .expect("test probability block length should fit u32"),
    }
}

fn pack_probabilities(probabilities: &[u32], bit_count: u8) -> Vec<u8> {
    let mut packed_bytes = vec![0_u8; (probabilities.len() * usize::from(bit_count)).div_ceil(8)];
    let mut bit_offset = 0_usize;
    for probability in probabilities {
        for bit_index in 0..usize::from(bit_count) {
            if ((probability >> bit_index) & 1) != 0 {
                let output_bit = bit_offset + bit_index;
                packed_bytes[output_bit / 8] |= 1 << (output_bit % 8);
            }
        }
        bit_offset += usize::from(bit_count);
    }
    packed_bytes
}

fn probability_block(
    sample_count: u32,
    ploidy: &[u8],
    phased_flag: u8,
    probability_bit_count: u8,
    probabilities: &[u32],
) -> Vec<u8> {
    let mut block = Vec::new();
    block.extend_from_slice(&sample_count.to_le_bytes());
    block.extend_from_slice(&2_u16.to_le_bytes());
    block.push(2);
    block.push(2);
    block.extend_from_slice(ploidy);
    block.push(phased_flag);
    block.push(probability_bit_count);
    block.extend(pack_probabilities(probabilities, probability_bit_count));
    block
}

fn probability_bit_count_offset(sample_count: usize) -> usize {
    4 + 2 + 1 + 1 + sample_count + 1
}

fn dosage_output_with_sentinels(value_count: usize) -> Vec<MaybeUninit<f32>> {
    vec![MaybeUninit::new(f32::NAN); value_count]
}

fn initialized_f32_values(values: Vec<MaybeUninit<f32>>) -> Vec<f32> {
    values
        .into_iter()
        .map(|value| {
            // SAFETY: every test output slot is initialized with a NaN sentinel
            // before decode, independently of whether the decoder writes it.
            unsafe { value.assume_init() }
        })
        .collect()
}

fn decode_blocks(
    probability_blocks: &[Vec<u8>],
    sample_count: usize,
    sample_indices: &[usize],
    collect_sparse_candidate_counts: bool,
) -> Result<DecodedTile, VariantDecodeFailure> {
    let encoded_probability_blocks = probability_blocks
        .iter()
        .map(|probability_block| EncodedProbabilityBlock {
            probability_payload: probability_block,
            declared_uncompressed_block_length: probability_block.len(),
        })
        .collect::<Vec<_>>();
    decode_encoded_blocks(
        &encoded_probability_blocks,
        CompressionType::None,
        sample_count,
        sample_indices,
        collect_sparse_candidate_counts,
    )
}

fn decode_encoded_blocks(
    probability_blocks: &[EncodedProbabilityBlock<'_>],
    compression_type: CompressionType,
    sample_count: usize,
    sample_indices: &[usize],
    collect_sparse_candidate_counts: bool,
) -> Result<DecodedTile, VariantDecodeFailure> {
    let sample_selection = build_sample_selection(sample_count, sample_indices).expect("sample selection should build");
    let mut source_bytes = Vec::new();
    let mut variant_records = Vec::with_capacity(probability_blocks.len());
    for probability_block in probability_blocks {
        let probability_payload_offset = source_bytes.len();
        source_bytes.extend_from_slice(probability_block.probability_payload);
        variant_records.push(test_variant_record_with_declared_length(
            probability_payload_offset,
            probability_block.probability_payload.len(),
            probability_block.declared_uncompressed_block_length,
        ));
    }

    let output_value_count = probability_blocks.len() * sample_indices.len();
    let mut output_values = dosage_output_with_sentinels(output_value_count);
    let mut dosage_sum = vec![0.0_f32; probability_blocks.len()];
    let mut dosage_square_sum = vec![0.0_f32; probability_blocks.len()];
    let mut observation_count = vec![0_i32; probability_blocks.len()];
    let mut zero_count = collect_sparse_candidate_counts.then(|| vec![0_i32; probability_blocks.len()]);
    let mut homozygous_alternate_count = collect_sparse_candidate_counts.then(|| vec![0_i32; probability_blocks.len()]);
    let mut thread_scratch = ThreadScratch::default();
    {
        let sparse_candidate_counts = match (zero_count.as_mut(), homozygous_alternate_count.as_mut()) {
            (Some(zero_count), Some(homozygous_alternate_count)) => {
                Some(VariantMajorSparseCandidateCountsMut { zero_count, homozygous_alternate_count })
            }
            (None, None) => None,
            _ => unreachable!("sparse candidate count buffers are allocated together"),
        };
        let mut tile_stats = VariantMajorTileStatsMut {
            dosage_sum: &mut dosage_sum,
            dosage_square_sum: &mut dosage_square_sum,
            observation_count: &mut observation_count,
            sparse_candidate_counts,
        };
        decode_variant_major_dosage_tile(
            VariantMajorTileDecodeRequest {
                source_window: BgenByteWindow::from_bytes(&source_bytes),
                compression_type,
                sample_count,
                sample_selection: &sample_selection,
                variant_records: &variant_records,
                tile_variant_start_index: 0,
            },
            &mut output_values,
            &mut tile_stats,
            &mut thread_scratch,
        )?;
    }

    Ok(DecodedTile {
        output_values: initialized_f32_values(output_values),
        dosage_sum,
        dosage_square_sum,
        observation_count,
        zero_count,
        homozygous_alternate_count,
    })
}

fn compress_zlib(source: &[u8]) -> Vec<u8> {
    // SAFETY: compression level one is supported and allocation has no caller
    // invariants beyond a valid level.
    let compressor_pointer = unsafe { libdeflate_sys::libdeflate_alloc_compressor(1) };
    let compressor = NonNull::new(compressor_pointer).expect("test zlib compressor should allocate");
    // SAFETY: `compressor` remains live until the matching free below.
    let compressed_capacity =
        unsafe { libdeflate_sys::libdeflate_zlib_compress_bound(compressor.as_ptr(), source.len()) };
    let mut compressed_payload = vec![0_u8; compressed_capacity];
    // SAFETY: the source and destination pointers are valid for their supplied
    // lengths, and the destination uses libdeflate's stated upper bound.
    let compressed_length = unsafe {
        libdeflate_sys::libdeflate_zlib_compress(
            compressor.as_ptr(),
            source.as_ptr().cast(),
            source.len(),
            compressed_payload.as_mut_ptr().cast(),
            compressed_payload.len(),
        )
    };
    // SAFETY: the compressor was allocated above and is freed exactly once.
    unsafe { libdeflate_sys::libdeflate_free_compressor(compressor.as_ptr()) };
    assert_ne!(compressed_length, 0, "test probability block should compress");
    compressed_payload.truncate(compressed_length);
    compressed_payload
}

fn compressed_decode_error_message(
    probability_payload: &[u8],
    compression_type: CompressionType,
    declared_uncompressed_block_length: usize,
) -> String {
    let encoded_probability_block = EncodedProbabilityBlock { probability_payload, declared_uncompressed_block_length };
    match decode_encoded_blocks(&[encoded_probability_block], compression_type, 3, &[0, 1, 2], false) {
        Ok(_) => panic!("invalid compressed probability block should fail"),
        Err(failure) => failure.source.to_string(),
    }
}

fn assert_three_sample_hard_call_decode(decoded: &DecodedTile) {
    let expected_output_values = [0.0_f32, 1.0, 2.0];
    assert_eq!(decoded.output_values.len(), expected_output_values.len());
    for (observed_value, expected_value) in decoded.output_values.iter().zip(expected_output_values) {
        assert!((observed_value - expected_value).abs() < 1.0e-6);
    }
    assert!((decoded.dosage_sum[0] - 3.0).abs() < 1.0e-6);
    assert!((decoded.dosage_square_sum[0] - 5.0).abs() < 1.0e-6);
    assert_eq!(decoded.observation_count, vec![3]);
}

fn decode_error_message(probability_block: Vec<u8>, sample_count: usize) -> String {
    let sample_indices = (0..sample_count).collect::<Vec<_>>();
    match decode_blocks(&[probability_block], sample_count, &sample_indices, false) {
        Ok(_) => panic!("invalid probability block should fail"),
        Err(failure) => failure.source.to_string(),
    }
}

#[test]
fn packed_probability_reader_reads_supported_widths_and_reports_truncation() {
    let test_cases: [(u8, &[u32]); 6] = [
        (1, &[1, 0, 1, 1, 0, 0, 1, 0, 1]),
        (2, &[1, 2, 3, 0, 3]),
        (4, &[0, 15, 8, 7, 1]),
        (8, &[0, 255, 17]),
        (16, &[0, 65_535, 0xBEEF]),
        (32, &[0, u32::MAX, 0x1234_5678]),
    ];

    for (bit_count, expected_probabilities) in test_cases {
        let packed_probabilities = pack_probabilities(expected_probabilities, bit_count);
        let mut reader = PackedProbabilityReader::new(&packed_probabilities);
        for expected_probability in expected_probabilities {
            assert_eq!(
                reader.read_probability(bit_count).expect("packed probability should decode"),
                *expected_probability
            );
        }
    }

    let cross_byte_test_cases: [(u8, &[u8], &[u32]); 5] = [
        (3, &[0xD5, 0x31], &[5, 2, 7, 0, 3]),
        (5, &[0x71, 0x7C, 0x90, 0x00], &[17, 3, 31, 0, 9]),
        (7, &[0xC1, 0xC1, 0x1F, 0xA0, 0x02], &[65, 3, 127, 0, 42]),
        (9, &[0x01, 0x07, 0xFC, 0x07, 0xA0, 0x12], &[257, 3, 511, 0, 298]),
        (
            31,
            &[
                0x01, 0x00, 0x00, 0xC0, 0x01, 0x00, 0x00, 0xC0, 0xFF, 0xFF, 0xFF, 0x1F, 0x00, 0x00, 0x00, 0x80, 0x67,
                0x45, 0x23, 0x01,
            ],
            &[0x4000_0001, 3, 0x7FFF_FFFF, 0, 0x1234_5678],
        ),
    ];
    for (bit_count, packed_probabilities, expected_probabilities) in cross_byte_test_cases {
        let mut reader = PackedProbabilityReader::new(packed_probabilities);
        for expected_probability in expected_probabilities {
            assert_eq!(
                reader.read_probability(bit_count).expect("cross-byte packed probability should decode"),
                *expected_probability
            );
        }
        assert!(reader.has_only_zero_padding());
    }

    let mut nonzero_padding_reader = PackedProbabilityReader::new(&[0xD5, 0xB1]);
    for _ in 0..5 {
        nonzero_padding_reader.read_probability(3).expect("cross-byte packed probability should decode");
    }
    assert!(!nonzero_padding_reader.has_only_zero_padding());

    let packed_probabilities = pack_probabilities(&[1, 2, 3, 0], 2);
    let mut reader = PackedProbabilityReader::new(&packed_probabilities);
    for _ in 0..4 {
        reader.read_probability(2).expect("byte-aligned probability should decode");
    }
    assert!(reader.read_probability(2).expect_err("truncated stream").to_string().contains("ended"));
}

#[test]
fn compressed_variant_major_decode_accepts_valid_zlib_and_zstandard_payloads() {
    let probability_block = probability_block(3, &[2, 2, 2], 0, 8, &[255, 0, 0, 255, 0, 0]);
    let zlib_payload = compress_zlib(&probability_block);
    let zstandard_payload =
        zstd::bulk::compress(&probability_block, 1).expect("test probability block should compress");

    for (compression_type, probability_payload) in
        [(CompressionType::Zlib, zlib_payload.as_slice()), (CompressionType::Zstandard, zstandard_payload.as_slice())]
    {
        let encoded_probability_block = EncodedProbabilityBlock {
            probability_payload,
            declared_uncompressed_block_length: probability_block.len(),
        };
        let decoded = decode_encoded_blocks(&[encoded_probability_block], compression_type, 3, &[0, 1, 2], false)
            .unwrap_or_else(|failure| {
                panic!("{compression_type:?} probability block should decode: {}", failure.source)
            });
        assert_three_sample_hard_call_decode(&decoded);
    }
}

#[test]
fn zlib_variant_major_decode_rejects_corruption_truncation_trailing_input_and_wrong_lengths() {
    let probability_block = probability_block(3, &[2, 2, 2], 0, 8, &[255, 0, 0, 255, 0, 0]);
    let compressed_payload = compress_zlib(&probability_block);

    let mut corrupted_payload = compressed_payload.clone();
    *corrupted_payload.last_mut().expect("zlib payload should contain an Adler checksum") ^= 0x01;
    assert!(
        compressed_decode_error_message(&corrupted_payload, CompressionType::Zlib, probability_block.len())
            .contains("invalid zlib data")
    );

    let mut truncated_payload = compressed_payload.clone();
    truncated_payload.pop();
    assert!(
        compressed_decode_error_message(&truncated_payload, CompressionType::Zlib, probability_block.len())
            .contains("invalid zlib data")
    );

    let mut payload_with_trailing_input = compressed_payload.clone();
    payload_with_trailing_input.push(0xA5);
    assert!(
        compressed_decode_error_message(&payload_with_trailing_input, CompressionType::Zlib, probability_block.len(),)
            .contains("consumed")
    );

    assert!(
        compressed_decode_error_message(&compressed_payload, CompressionType::Zlib, probability_block.len() - 1)
            .contains("exceeds its declared uncompressed length")
    );
    assert!(
        compressed_decode_error_message(&compressed_payload, CompressionType::Zlib, probability_block.len() + 1)
            .contains("incomplete output block")
    );
}

#[test]
fn zstandard_variant_major_decode_rejects_corruption_truncation_trailing_input_and_wrong_lengths() {
    let probability_block = probability_block(3, &[2, 2, 2], 0, 8, &[255, 0, 0, 255, 0, 0]);
    let compressed_payload =
        zstd::bulk::compress(&probability_block, 1).expect("test probability block should compress");

    let mut corrupted_payload = compressed_payload.clone();
    corrupted_payload[0] ^= 0x01;
    assert!(
        compressed_decode_error_message(&corrupted_payload, CompressionType::Zstandard, probability_block.len())
            .contains("invalid data")
    );

    let mut truncated_payload = compressed_payload.clone();
    truncated_payload.pop();
    assert!(
        compressed_decode_error_message(&truncated_payload, CompressionType::Zstandard, probability_block.len())
            .contains("invalid data")
    );

    let mut payload_with_trailing_input = compressed_payload.clone();
    payload_with_trailing_input.push(0xA5);
    assert!(
        compressed_decode_error_message(
            &payload_with_trailing_input,
            CompressionType::Zstandard,
            probability_block.len(),
        )
        .contains("invalid data")
    );

    assert!(
        compressed_decode_error_message(&compressed_payload, CompressionType::Zstandard, probability_block.len() - 1,)
            .contains("invalid data")
    );
    let oversized_length_error =
        compressed_decode_error_message(&compressed_payload, CompressionType::Zstandard, probability_block.len() + 1);
    assert!(oversized_length_error.contains("expanded to"));
    assert!(oversized_length_error.contains("declared"));
}

#[test]
fn variant_major_eight_bit_decode_preserves_selection_order_and_imputes_missing_values() {
    let all_present_block = probability_block(3, &[2, 2, 2], 0, 8, &[255, 0, 0, 255, 0, 0]);

    let identity = decode_blocks(std::slice::from_ref(&all_present_block), 3, &[0, 1, 2], true)
        .unwrap_or_else(|_| panic!("identity variant-major decode should succeed"));
    assert_eq!(identity.output_values, vec![0.0, 1.0, 2.0]);
    assert!((identity.dosage_sum[0] - 3.0).abs() < 1.0e-6);
    assert!((identity.dosage_square_sum[0] - 5.0).abs() < 1.0e-6);
    assert_eq!(identity.observation_count, vec![3]);
    assert_eq!(identity.zero_count, Some(vec![1]));
    assert_eq!(identity.homozygous_alternate_count, Some(vec![1]));

    let contiguous = decode_blocks(std::slice::from_ref(&all_present_block), 3, &[1, 2], true)
        .unwrap_or_else(|_| panic!("contiguous variant-major decode should succeed"));
    assert_eq!(contiguous.output_values, vec![1.0, 2.0]);
    assert_eq!(contiguous.observation_count, vec![2]);

    let noncontiguous = decode_blocks(std::slice::from_ref(&all_present_block), 3, &[2, 0], true)
        .unwrap_or_else(|_| panic!("non-contiguous variant-major decode should succeed"));
    assert_eq!(noncontiguous.output_values, vec![2.0, 0.0]);
    assert_eq!(noncontiguous.dosage_sum, vec![2.0]);

    let missing_block = probability_block(3, &[2, 0x82, 2], 0, 8, &[255, 0, 0, 0, 0, 0]);
    let missing = decode_blocks(&[missing_block], 3, &[2, 1, 0], false)
        .unwrap_or_else(|_| panic!("missing variant-major decode should succeed"));
    assert_eq!(missing.output_values, vec![2.0, 1.0, 0.0]);
    assert_eq!(missing.dosage_sum, vec![2.0]);
    assert_eq!(missing.dosage_square_sum, vec![4.0]);
    assert_eq!(missing.observation_count, vec![2]);

    let fractional_missing_block = probability_block(3, &[2, 0x82, 2], 0, 8, &[128, 0, 0, 0, 64, 64]);
    let fractional_missing = decode_blocks(&[fractional_missing_block], 3, &[2, 1, 0], false)
        .unwrap_or_else(|_| panic!("fractional missing variant-major decode should succeed"));
    let first_selected_dosage = 318.0_f32 / 255.0;
    let third_selected_dosage = 254.0_f32 / 255.0;
    let imputed_dosage = 286.0_f32 / 255.0;
    for (observed_value, expected_value) in
        fractional_missing.output_values.iter().zip([first_selected_dosage, imputed_dosage, third_selected_dosage])
    {
        assert!((observed_value - expected_value).abs() < 1.0e-6);
    }
    assert!((fractional_missing.dosage_sum[0] - (572.0 / 255.0)).abs() < 1.0e-6);
    assert!(
        (fractional_missing.dosage_square_sum[0] - ((254.0_f32.powi(2) + 318.0_f32.powi(2)) / 65_025.0)).abs() < 1.0e-6
    );
    assert_eq!(fractional_missing.observation_count, vec![2]);
}

#[test]
fn variant_major_eight_bit_subsets_validate_unselected_probability_pairs() {
    let invalid_unselected_block = probability_block(3, &[2, 2, 2], 0, 8, &[255, 1, 255, 0, 0, 255]);
    for sample_indices in [&[1, 2][..], &[2, 1][..]] {
        let failure = decode_blocks(std::slice::from_ref(&invalid_unselected_block), 3, sample_indices, false)
            .expect_err("a corrupt unselected probability pair should fail subset decode");
        assert!(failure.source.to_string().contains("sum above 255"));
    }
}

#[test]
fn variant_major_tile_collects_statistics_and_rejects_shape_mismatches() {
    let first_block = probability_block(2, &[2, 2], 0, 8, &[255, 0, 0, 255]);
    let second_block = probability_block(2, &[2, 2], 0, 8, &[0, 0, 255, 0]);
    let decoded = decode_blocks(&[first_block.clone(), second_block.clone()], 2, &[0, 1], true)
        .unwrap_or_else(|_| panic!("variant-major tile should decode"));
    assert_eq!(decoded.output_values, vec![0.0, 1.0, 2.0, 0.0]);
    assert_eq!(decoded.dosage_sum, vec![1.0, 2.0]);
    assert_eq!(decoded.dosage_square_sum, vec![1.0, 4.0]);
    assert_eq!(decoded.observation_count, vec![2, 2]);
    assert_eq!(decoded.zero_count, Some(vec![1, 1]));
    assert_eq!(decoded.homozygous_alternate_count, Some(vec![0, 1]));

    let mut source_bytes = first_block;
    let second_offset = source_bytes.len();
    source_bytes.extend_from_slice(&second_block);
    let variant_records =
        [test_variant_record_at(0, second_offset), test_variant_record_at(second_offset, second_block.len())];
    let sample_selection = build_sample_selection(2, &[0, 1]).expect("identity selection should build");
    let mut thread_scratch = ThreadScratch::default();
    let mut output_values = vec![MaybeUninit::<f32>::uninit(); 3];
    let mut dosage_sum = vec![0.0_f32; 2];
    let mut dosage_square_sum = vec![0.0_f32; 2];
    let mut observation_count = vec![0_i32; 2];
    let mut tile_stats = VariantMajorTileStatsMut {
        dosage_sum: &mut dosage_sum,
        dosage_square_sum: &mut dosage_square_sum,
        observation_count: &mut observation_count,
        sparse_candidate_counts: None,
    };
    let shape_failure = decode_variant_major_dosage_tile(
        VariantMajorTileDecodeRequest {
            source_window: BgenByteWindow::from_bytes(&source_bytes),
            compression_type: CompressionType::None,
            sample_count: 2,
            sample_selection: &sample_selection,
            variant_records: &variant_records,
            tile_variant_start_index: 0,
        },
        &mut output_values,
        &mut tile_stats,
        &mut thread_scratch,
    )
    .expect_err("short output should fail");
    assert!(shape_failure.source.to_string().contains("contains 3 values, expected 4"));

    let mut short_dosage_sum = vec![0.0_f32; 1];
    let short_stats = VariantMajorTileStatsMut {
        dosage_sum: &mut short_dosage_sum,
        dosage_square_sum: &mut dosage_square_sum,
        observation_count: &mut observation_count,
        sparse_candidate_counts: None,
    };
    assert!(
        validate_variant_major_tile_stats_lengths(&short_stats, 2)
            .expect_err("short statistics buffer should fail")
            .to_string()
            .contains("shape mismatch")
    );
}

#[test]
fn generic_variant_major_decode_covers_phased_values_and_corrupt_blocks() {
    let unphased_block = probability_block(2, &[2, 2], 0, 2, &[3, 0, 0, 3]);
    let unphased = decode_blocks(&[unphased_block], 2, &[0, 1], false)
        .unwrap_or_else(|_| panic!("generic unphased decode should succeed"));
    assert_eq!(unphased.output_values, vec![0.0, 1.0]);

    let phased_block = probability_block(2, &[2, 2], 1, 2, &[3, 0, 0, 3]);
    let phased = decode_blocks(&[phased_block], 2, &[0, 1], false)
        .unwrap_or_else(|_| panic!("generic phased decode should succeed"));
    assert_eq!(phased.output_values, vec![1.0, 1.0]);

    let phased_missing_block = probability_block(2, &[0x82, 2], 1, 2, &[0, 0, 0, 3]);
    let phased_missing = decode_blocks(&[phased_missing_block], 2, &[0, 1], false)
        .unwrap_or_else(|_| panic!("generic phased missing decode should succeed"));
    assert_eq!(phased_missing.output_values, vec![1.0, 1.0]);
    assert_eq!(phased_missing.observation_count, vec![1]);

    let mut invalid_blocks = vec![
        (probability_block(1, &[2], 0, 2, &[3, 0]), 2, "file header"),
        (probability_block(1, &[1], 0, 2, &[3, 0]), 1, "non-diploid"),
        (probability_block(1, &[0x42], 0, 2, &[0, 0]), 1, "reserved ploidy flag bits"),
        (probability_block(1, &[0x82], 0, 2, &[1, 0]), 1, "nonzero probabilities for missing"),
        (probability_block(1, &[2], 2, 2, &[3, 0]), 1, "phased flag"),
        (probability_block(1, &[2], 0, 8, &[255, 1]), 1, "sum above 255"),
    ];
    let mut non_biallelic_block = probability_block(1, &[2], 0, 2, &[3, 0]);
    non_biallelic_block[4..6].copy_from_slice(&3_u16.to_le_bytes());
    invalid_blocks.push((non_biallelic_block, 1, "biallelic"));
    let mut bad_ploidy_bounds_block = probability_block(1, &[2], 0, 2, &[3, 0]);
    bad_ploidy_bounds_block[6] = 1;
    invalid_blocks.push((bad_ploidy_bounds_block, 1, "ploidy bounds"));
    let mut bad_bit_count_block = probability_block(1, &[2], 0, 2, &[3, 0]);
    bad_bit_count_block[probability_bit_count_offset(1)] = 0;
    invalid_blocks.push((bad_bit_count_block, 1, "requires a value"));
    let mut truncated_block = probability_block(1, &[2], 0, 8, &[255, 0]);
    truncated_block.pop();
    invalid_blocks.push((truncated_block, 1, "requires exactly"));
    let mut nonzero_padding_block = probability_block(1, &[2], 0, 2, &[3, 0]);
    *nonzero_padding_block.last_mut().expect("probability byte should exist") |= 0x80;
    invalid_blocks.push((nonzero_padding_block, 1, "nonzero padding"));

    for (invalid_block, sample_count, expected_message) in invalid_blocks {
        let error_message = decode_error_message(invalid_block, sample_count);
        assert!(
            error_message.contains(expected_message),
            "expected `{expected_message}` in decode error, observed `{error_message}`"
        );
    }
}
