use std::io::Write;

use super::super::CompressionType;
use super::super::metadata::VariantRecord;
use super::super::sample_selection::build_sample_selection;
use super::*;

fn test_variant_record(probability_payload_length: usize) -> VariantRecord {
    test_variant_record_at(0, probability_payload_length, "variant")
}

fn test_variant_record_at(
    probability_payload_offset: usize,
    probability_payload_length: usize,
    resolved_variant_identifier: &str,
) -> VariantRecord {
    VariantRecord {
        probability_payload_offset,
        probability_payload_length,
        declared_uncompressed_block_length: probability_payload_length,
        chromosome: "22".to_string(),
        resolved_variant_identifier: resolved_variant_identifier.to_string(),
        position: 1,
        counted_allele: "A".to_string(),
        reference_allele: "G".to_string(),
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
    4 + 2 + 2 + sample_count + 1
}

fn zlib_compress(payload: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(payload).expect("payload should compress");
    encoder.finish().expect("compressed payload should finish")
}

#[test]
fn row_major_output_matrix_returns_requested_row_range_and_column() {
    let mut output_values = [0_i32; 8];
    let mut output_matrix = unsafe {
        RowMajorOutputMatrix::<i32>::from_pointer_address(output_values.as_mut_ptr() as usize, 4, "test row-major")
    }
    .expect("test output matrix should build");

    output_matrix.row_range_mut(1, 1, 2).expect("second row range should be available").copy_from_slice(&[5, 6]);
    unsafe {
        output_matrix.column_mut(0).expect("first column should be available").write_unchecked(1, 4);
    }

    assert_eq!(output_values, [0, 0, 0, 0, 4, 5, 6, 0]);
}

#[test]
fn row_major_output_matrix_rejects_invalid_boundary_state() {
    let null_result = unsafe { RowMajorOutputMatrix::<u8>::from_pointer_address(0, 1, "test row-major") };
    assert!(matches!(null_result, Err(error) if error.to_string().contains("output pointer is null")));

    let empty_row_result = unsafe {
        RowMajorOutputMatrix::<u8>::from_pointer_address(
            std::ptr::NonNull::<u8>::dangling().as_ptr() as usize,
            0,
            "test row-major",
        )
    };
    assert!(matches!(empty_row_result, Err(error) if error.to_string().contains("output row length must be positive")));

    let misaligned_result = unsafe { RowMajorOutputMatrix::<i32>::from_pointer_address(1, 1, "test row-major") };
    assert!(matches!(misaligned_result, Err(error) if error.to_string().contains("output pointer is not aligned")));

    let mut output_matrix = unsafe {
        RowMajorOutputMatrix::<u8>::from_pointer_address(
            std::ptr::NonNull::<u8>::dangling().as_ptr() as usize,
            usize::MAX,
            "test row-major",
        )
    }
    .expect("dangling pointer is acceptable when offset validation fails before dereference");
    let row_error = output_matrix.row_mut(2).expect_err("oversized row offset should fail");
    assert!(row_error.to_string().contains("Integer overflow while locating test row-major output row"));

    let mut output_values = [0_u8; 4];
    let mut output_matrix = unsafe {
        RowMajorOutputMatrix::<u8>::from_pointer_address(output_values.as_mut_ptr() as usize, 2, "test row-major")
    }
    .expect("test output matrix should build");
    let column_result = output_matrix.column_mut(2);
    assert!(
        matches!(column_result, Err(error) if error.to_string().contains("output column 2 exceeds the row length 2"))
    );

    let range_error = output_matrix.row_range_mut(0, 1, 2).expect_err("oversized row range should fail");
    assert!(range_error.to_string().contains("output row range exceeds the row length"));
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

    let packed_probabilities = pack_probabilities(&[1, 2, 3, 0], 2);
    let mut reader = PackedProbabilityReader::new(&packed_probabilities);
    for _ in 0..4 {
        reader.read_probability(2).expect("byte-aligned probability should decode");
    }
    assert!(reader.read_probability(2).expect_err("truncated stream").to_string().contains("ended"));
}

#[test]
fn row_major_decode_covers_eight_bit_identity_subset_and_missing_paths() {
    let variant_record = test_variant_record(0);
    let sample_selection = build_sample_selection(3, &[0, 1, 2]).expect("identity selection");
    let mut output = vec![0.0_f32; 3 * 2];
    let result = decode_unphased_eight_bit_dosages_into_row_major_matrix(
        &[2, 2, 2],
        &[255, 0, 0, 255, 0, 0],
        &sample_selection,
        &variant_record,
        output.as_mut_ptr() as usize,
        1,
        2,
        true,
        false,
        true,
        ThreadLocalProfileSnapshot::default(),
    )
    .expect("identity 8-bit row-major decode");
    assert_eq!(result.selected_observation_count, 0);
    assert!(result.selected_dosage_total > 0.0);
    assert!(output.iter().any(|value| *value > 0.0));

    let subset_selection = build_sample_selection(3, &[2, 0]).expect("subset selection");
    let mut subset_output = vec![0.0_f32; 2 * 2];
    let subset_result = decode_unphased_eight_bit_dosages_into_row_major_matrix(
        &[2, 0x82, 2],
        &[255, 0, 0, 255, 0, 0],
        &subset_selection,
        &variant_record,
        subset_output.as_mut_ptr() as usize,
        0,
        2,
        true,
        false,
        true,
        ThreadLocalProfileSnapshot::default(),
    )
    .expect("subset 8-bit row-major decode");
    assert!(subset_result.selected_dosage_total >= 0.0);
    assert!(subset_output.iter().all(|value| !value.is_nan()));

    let mut identity_missing_output = vec![0.0_f32; 3];
    let identity_missing_result = decode_unphased_eight_bit_dosages_into_row_major_matrix(
        &[2, 0x82, 2],
        &[255, 0, 0, 255, 0, 0],
        &sample_selection,
        &variant_record,
        identity_missing_output.as_mut_ptr() as usize,
        0,
        1,
        true,
        false,
        true,
        ThreadLocalProfileSnapshot::default(),
    )
    .expect("identity missing 8-bit row-major decode");
    assert!((identity_missing_result.selected_dosage_total - 2.0).abs() < f32::EPSILON);
    assert!(identity_missing_output[1].is_nan());

    let selected_all_present = build_sample_selection(3, &[2, 0]).expect("non-contiguous subset");
    let mut selected_all_present_output = vec![0.0_f32; 2];
    let selected_all_present_result = decode_unphased_eight_bit_dosages_into_row_major_matrix(
        &[2, 2, 2],
        &[255, 0, 0, 255, 0, 0],
        &selected_all_present,
        &variant_record,
        selected_all_present_output.as_mut_ptr() as usize,
        0,
        1,
        true,
        false,
        true,
        ThreadLocalProfileSnapshot::default(),
    )
    .expect("selected all-present 8-bit row-major decode");
    assert!((selected_all_present_result.selected_dosage_total - 2.0).abs() < f32::EPSILON);
    assert_eq!(selected_all_present_output, vec![2.0, 0.0]);
}

#[test]
fn row_major_all_present_selected_paths_match_sample_order() {
    let variant_record = test_variant_record(0);
    let probability_bytes = [255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];
    let ploidy_bytes = [2_u8; 6];

    for (sample_indices, expected_output, expected_total) in [
        (vec![0, 1, 2, 3, 4, 5], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0], 6.0),
        (vec![1, 2, 3], vec![1.0, 2.0, 0.0], 3.0),
        (vec![5, 0], vec![2.0, 0.0], 2.0),
        (vec![0, 2, 3, 4], vec![0.0, 2.0, 0.0, 1.0], 3.0),
    ] {
        let sample_selection =
            build_sample_selection(6, &sample_indices).expect("selected row-major samples should build");
        let mut output = vec![f32::NAN; sample_indices.len()];
        let result = decode_unphased_eight_bit_dosages_into_row_major_matrix(
            &ploidy_bytes,
            &probability_bytes,
            &sample_selection,
            &variant_record,
            output.as_mut_ptr() as usize,
            0,
            1,
            true,
            false,
            true,
            ThreadLocalProfileSnapshot::default(),
        )
        .expect("selected all-present row-major decode should succeed");

        assert_eq!(output, expected_output);
        assert!((result.selected_dosage_total - expected_total).abs() < f32::EPSILON);
    }
}

#[test]
fn variant_major_decode_covers_eight_bit_identity_subset_and_imputation_paths() {
    let variant_record = test_variant_record(0);
    let subset_selection = build_sample_selection(3, &[2, 1, 0]).expect("subset selection");
    let mut output = vec![0.0_f32; 3];
    let result = decode_unphased_eight_bit_dosages_into_variant_major_matrix(
        &[2, 0x82, 2],
        &[255, 0, 0, 255, 0, 0],
        &subset_selection,
        &variant_record,
        output.as_mut_ptr() as usize,
        0,
        3,
        true,
        false,
        ThreadLocalProfileSnapshot::default(),
    )
    .expect("8-bit variant-major decode");
    assert!(result.has_missing_values);
    assert_eq!(result.selected_observation_count, 2);
    assert!(output.iter().all(|value| !value.is_nan()));

    let identity_selection = build_sample_selection(3, &[0, 1, 2]).expect("identity selection");
    let mut identity_output = vec![0.0_f32; 3];
    let identity_result = decode_unphased_eight_bit_dosages_into_variant_major_matrix(
        &[2, 2, 2],
        &[255, 0, 0, 255, 0, 0],
        &identity_selection,
        &variant_record,
        identity_output.as_mut_ptr() as usize,
        0,
        3,
        true,
        true,
        ThreadLocalProfileSnapshot::default(),
    )
    .expect("trusted identity 8-bit variant-major decode");
    assert!(!identity_result.has_missing_values);
    assert_eq!(identity_result.selected_observation_count, 3);

    let contiguous_subset_selection = build_sample_selection(3, &[1, 2]).expect("contiguous subset selection");
    let mut contiguous_subset_output = vec![0.0_f32; 2];
    let contiguous_subset_result = decode_unphased_eight_bit_dosages_into_variant_major_matrix(
        &[2, 2, 2],
        &[255, 0, 0, 255, 0, 0],
        &contiguous_subset_selection,
        &variant_record,
        contiguous_subset_output.as_mut_ptr() as usize,
        0,
        2,
        true,
        false,
        ThreadLocalProfileSnapshot::default(),
    )
    .expect("contiguous subset 8-bit variant-major decode");
    assert!(!contiguous_subset_result.has_missing_values);
    assert_eq!(contiguous_subset_result.selected_observation_count, 2);
    assert_eq!(contiguous_subset_output, vec![1.0, 2.0]);

    let identity_missing_selection = build_sample_selection(3, &[0, 1, 2]).expect("identity selection");
    let mut identity_missing_output = vec![0.0_f32; 3];
    let identity_missing_result = decode_unphased_eight_bit_dosages_into_variant_major_matrix(
        &[2, 0x82, 2],
        &[255, 0, 0, 255, 0, 0],
        &identity_missing_selection,
        &variant_record,
        identity_missing_output.as_mut_ptr() as usize,
        0,
        3,
        true,
        false,
        ThreadLocalProfileSnapshot::default(),
    )
    .expect("identity missing 8-bit variant-major decode");
    assert!(identity_missing_result.has_missing_values);
    assert_eq!(identity_missing_result.selected_observation_count, 2);
    assert!(identity_missing_output.iter().all(|value| !value.is_nan()));

    let noncontiguous_selection = build_sample_selection(3, &[2, 0]).expect("non-contiguous subset");
    let mut noncontiguous_output = vec![0.0_f32; 2];
    let noncontiguous_result = decode_unphased_eight_bit_dosages_into_variant_major_matrix(
        &[2, 2, 2],
        &[255, 0, 0, 255, 0, 0],
        &noncontiguous_selection,
        &variant_record,
        noncontiguous_output.as_mut_ptr() as usize,
        0,
        2,
        true,
        false,
        ThreadLocalProfileSnapshot::default(),
    )
    .expect("non-contiguous all-present 8-bit variant-major decode");
    assert!(!noncontiguous_result.has_missing_values);
    assert_eq!(noncontiguous_result.selected_observation_count, 2);
    assert_eq!(noncontiguous_output, vec![2.0, 0.0]);
}

#[test]
#[allow(clippy::too_many_lines)]
fn tile_decoders_copy_dosages_and_collect_variant_major_stats() {
    let first_block = probability_block(2, &[2, 2], 0, 8, &[255, 0, 0, 255]);
    let second_block = probability_block(2, &[2, 2], 0, 8, &[0, 255, 255, 0]);
    let mut mmap = first_block.clone();
    mmap.extend_from_slice(&second_block);
    let variant_records = vec![
        test_variant_record_at(0, first_block.len(), "first"),
        test_variant_record_at(first_block.len(), second_block.len(), "second"),
    ];
    let sample_selection = build_sample_selection(2, &[0, 1]).expect("identity selection");
    let mut thread_scratch = ThreadScratch::default();
    let mut row_major_output = vec![0.0_f32; 4];
    let tile_result = decode_variant_dosage_tile_into_row_major_matrix(
        &mmap,
        CompressionType::None,
        2,
        &sample_selection,
        &variant_records,
        row_major_output.as_mut_ptr() as usize,
        2,
        0,
        true,
        false,
        true,
        &mut thread_scratch,
    )
    .expect("row-major tile should decode");
    assert_eq!(tile_result.selected_dosage_totals.len(), 2);
    assert_eq!(tile_result.profile_snapshot.decode_tile_count, 1);
    assert!(row_major_output.iter().any(|value| *value > 0.0));

    let mut disabled_profile_output = vec![0.0_f32; 4];
    let disabled_profile_result = decode_variant_dosage_tile_into_row_major_matrix(
        &mmap,
        CompressionType::None,
        2,
        &sample_selection,
        &variant_records,
        disabled_profile_output.as_mut_ptr() as usize,
        2,
        0,
        false,
        false,
        true,
        &mut thread_scratch,
    )
    .expect("row-major tile should decode without profiling");
    assert_eq!(disabled_profile_result.profile_snapshot, ThreadLocalProfileSnapshot::default());
    assert_eq!(disabled_profile_result.selected_dosage_totals, tile_result.selected_dosage_totals);
    assert_eq!(disabled_profile_output, row_major_output);

    let mut direct_output = vec![0.0_f32; 4];
    let direct_result = decode_variant_dosage_tile_direct_into_row_major_matrix(
        &mmap,
        CompressionType::None,
        2,
        &sample_selection,
        &variant_records,
        direct_output.as_mut_ptr() as usize,
        2,
        0,
        false,
        false,
        true,
        &mut thread_scratch,
    )
    .expect("direct row-major tile should decode");
    assert_eq!(direct_result.profile_snapshot, ThreadLocalProfileSnapshot::default());
    assert_eq!(direct_result.selected_dosage_totals, tile_result.selected_dosage_totals);
    assert_eq!(direct_output, row_major_output);

    let contiguous_sample_selection = build_sample_selection(2, &[1]).expect("contiguous subset selection");
    let mut contiguous_scratch_output = vec![0.0_f32; 2];
    decode_variant_dosage_tile_into_row_major_matrix(
        &mmap,
        CompressionType::None,
        2,
        &contiguous_sample_selection,
        &variant_records,
        contiguous_scratch_output.as_mut_ptr() as usize,
        2,
        0,
        false,
        false,
        false,
        &mut thread_scratch,
    )
    .expect("contiguous row-major tile should decode");
    let mut contiguous_direct_output = vec![0.0_f32; 2];
    decode_variant_dosage_tile_direct_into_row_major_matrix(
        &mmap,
        CompressionType::None,
        2,
        &contiguous_sample_selection,
        &variant_records,
        contiguous_direct_output.as_mut_ptr() as usize,
        2,
        0,
        false,
        false,
        false,
        &mut thread_scratch,
    )
    .expect("contiguous direct row-major tile should decode");
    assert_eq!(contiguous_direct_output, contiguous_scratch_output);

    let mut dosage_sum = vec![0.0_f32; 2];
    let mut dosage_square_sum = vec![0.0_f32; 2];
    let mut observation_count = vec![0_i32; 2];
    let mut zero_count = vec![0_i32; 2];
    let mut nonzero_count = vec![0_i32; 2];
    let mut homozygous_reference_count = vec![0_i32; 2];
    let mut heterozygous_count = vec![0_i32; 2];
    let mut homozygous_alternate_count = vec![0_i32; 2];
    let mut variant_major_stats = VariantMajorTileStatsMut {
        dosage_sum: &mut dosage_sum,
        dosage_square_sum: &mut dosage_square_sum,
        observation_count: &mut observation_count,
        zero_count: &mut zero_count,
        nonzero_count: &mut nonzero_count,
        homozygous_reference_count: &mut homozygous_reference_count,
        heterozygous_count: &mut heterozygous_count,
        homozygous_alternate_count: &mut homozygous_alternate_count,
    };
    let mut variant_major_output = vec![0.0_f32; 4];
    let variant_major_result = decode_variant_major_dosage_tile(
        &mmap,
        CompressionType::None,
        2,
        &sample_selection,
        &variant_records,
        variant_major_output.as_mut_ptr() as usize,
        2,
        0,
        true,
        false,
        &mut variant_major_stats,
        &mut thread_scratch,
    )
    .expect("variant-major tile should decode");
    assert!(!variant_major_result.has_missing_values);
    assert_eq!(variant_major_result.profile_snapshot.decode_tile_count, 1);
    assert_eq!(observation_count, vec![2, 2]);

    let mut short_dosage_sum = vec![0.0_f32; 1];
    let short_stats = VariantMajorTileStatsMut {
        dosage_sum: &mut short_dosage_sum,
        dosage_square_sum: &mut dosage_square_sum,
        observation_count: &mut observation_count,
        zero_count: &mut zero_count,
        nonzero_count: &mut nonzero_count,
        homozygous_reference_count: &mut homozygous_reference_count,
        heterozygous_count: &mut heterozygous_count,
        homozygous_alternate_count: &mut homozygous_alternate_count,
    };
    assert!(
        validate_variant_major_tile_stats_lengths(&short_stats, 2)
            .expect_err("short stats vector should fail")
            .to_string()
            .contains("shape mismatch")
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn generic_row_major_decode_covers_unphased_phased_subset_and_error_paths() {
    let sample_selection = build_sample_selection(2, &[0, 1]).expect("identity selection");
    let mut thread_scratch = ThreadScratch::default();
    let unphased_block = probability_block(2, &[2, 2], 0, 2, &[3, 0, 0, 3]);
    let variant_record = test_variant_record(unphased_block.len());
    let mut output = vec![0.0_f32; 2];
    let unphased_result = decode_variant_dosages_into_row_major_matrix(
        &unphased_block,
        CompressionType::None,
        2,
        &sample_selection,
        &variant_record,
        output.as_mut_ptr() as usize,
        0,
        1,
        true,
        false,
        true,
        &mut thread_scratch,
    )
    .expect("generic unphased row-major decode");
    assert!(unphased_result.selected_dosage_total >= 0.0);

    let phased_block = probability_block(2, &[2, 2], 1, 2, &[3, 0, 0, 3]);
    let variant_record = test_variant_record(phased_block.len());
    let subset_selection = build_sample_selection(2, &[1]).expect("subset selection");
    let mut subset_output = vec![0.0_f32; 1];
    decode_variant_dosages_into_row_major_matrix(
        &phased_block,
        CompressionType::None,
        2,
        &subset_selection,
        &variant_record,
        subset_output.as_mut_ptr() as usize,
        0,
        1,
        true,
        false,
        true,
        &mut thread_scratch,
    )
    .expect("generic phased subset row-major decode");

    let missing_unphased_block = probability_block(2, &[2, 0x82], 0, 2, &[3, 0, 0, 3]);
    let variant_record = test_variant_record(missing_unphased_block.len());
    let mut missing_output = vec![0.0_f32; 1];
    let missing_result = decode_variant_dosages_into_row_major_matrix(
        &missing_unphased_block,
        CompressionType::None,
        2,
        &build_sample_selection(2, &[1]).expect("missing subset selection"),
        &variant_record,
        missing_output.as_mut_ptr() as usize,
        0,
        1,
        true,
        false,
        true,
        &mut thread_scratch,
    )
    .expect("generic unphased missing subset row-major decode");
    assert!(missing_result.selected_dosage_total.abs() < f32::EPSILON);
    assert!(missing_output[0].is_nan());

    let missing_phased_block = probability_block(2, &[0x82, 2], 1, 2, &[3, 0, 0, 3]);
    let variant_record = test_variant_record(missing_phased_block.len());
    let mut identity_missing_output = vec![0.0_f32; 2];
    let identity_missing_result = decode_variant_dosages_into_row_major_matrix(
        &missing_phased_block,
        CompressionType::None,
        2,
        &sample_selection,
        &variant_record,
        identity_missing_output.as_mut_ptr() as usize,
        0,
        1,
        true,
        false,
        true,
        &mut thread_scratch,
    )
    .expect("generic phased missing identity row-major decode");
    assert!((identity_missing_result.selected_dosage_total - 1.0).abs() < f32::EPSILON);
    assert!(identity_missing_output[0].is_nan());

    let invalid_phased_block = probability_block(1, &[2], 2, 2, &[0, 0]);
    let variant_record = test_variant_record(invalid_phased_block.len());
    assert!(
        decode_variant_dosages_into_row_major_matrix(
            &invalid_phased_block,
            CompressionType::None,
            1,
            &build_sample_selection(1, &[0]).expect("identity selection"),
            &variant_record,
            output.as_mut_ptr() as usize,
            0,
            1,
            false,
            false,
            false,
            &mut thread_scratch,
        )
        .expect_err("invalid phased flag should fail")
        .to_string()
        .contains("phased flag")
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn generic_variant_major_decode_covers_unphased_phased_and_missing_paths() {
    let mut thread_scratch = ThreadScratch::default();
    let sample_selection = build_sample_selection(2, &[1, 0]).expect("subset selection");
    let unphased_block = probability_block(2, &[2, 0x82], 0, 2, &[3, 0, 0, 3]);
    let variant_record = test_variant_record(unphased_block.len());
    let mut output = vec![0.0_f32; 2];
    let result = decode_variant_dosages_into_variant_major_matrix(
        &unphased_block,
        CompressionType::None,
        2,
        &sample_selection,
        &variant_record,
        output.as_mut_ptr() as usize,
        0,
        2,
        true,
        false,
        &mut thread_scratch,
    )
    .expect("generic variant-major unphased decode");
    assert!(result.has_missing_values);
    assert!(output.iter().all(|value| !value.is_nan()));

    let phased_block = probability_block(2, &[2, 2], 1, 2, &[3, 0, 0, 3]);
    let variant_record = test_variant_record(phased_block.len());
    decode_variant_dosages_into_variant_major_matrix(
        &phased_block,
        CompressionType::None,
        2,
        &build_sample_selection(2, &[0, 1]).expect("identity selection"),
        &variant_record,
        output.as_mut_ptr() as usize,
        0,
        2,
        true,
        false,
        &mut thread_scratch,
    )
    .expect("generic variant-major phased decode");

    let phased_missing_block = probability_block(2, &[0x82, 2], 1, 2, &[3, 0, 0, 3]);
    let variant_record = test_variant_record(phased_missing_block.len());
    let mut missing_output = vec![0.0_f32; 2];
    let phased_missing_result = decode_variant_dosages_into_variant_major_matrix(
        &phased_missing_block,
        CompressionType::None,
        2,
        &build_sample_selection(2, &[0, 1]).expect("identity selection"),
        &variant_record,
        missing_output.as_mut_ptr() as usize,
        0,
        2,
        true,
        false,
        &mut thread_scratch,
    )
    .expect("generic variant-major phased missing decode");
    assert!(phased_missing_result.has_missing_values);
    assert!(missing_output.iter().all(|value| !value.is_nan()));

    let stored_sample_mismatch_block = probability_block(1, &[2], 0, 2, &[3, 0]);
    let variant_record = test_variant_record(stored_sample_mismatch_block.len());
    assert!(
        decode_variant_dosages_into_variant_major_matrix(
            &stored_sample_mismatch_block,
            CompressionType::None,
            2,
            &build_sample_selection(2, &[0, 1]).expect("identity selection"),
            &variant_record,
            output.as_mut_ptr() as usize,
            0,
            2,
            false,
            false,
            &mut thread_scratch,
        )
        .expect_err("sample count mismatch should fail")
        .to_string()
        .contains("file header")
    );

    let mut non_biallelic_block = probability_block(1, &[2], 0, 2, &[3, 0]);
    non_biallelic_block[4..6].copy_from_slice(&3_u16.to_le_bytes());
    let variant_record = test_variant_record(non_biallelic_block.len());
    assert!(
        decode_variant_dosages_into_variant_major_matrix(
            &non_biallelic_block,
            CompressionType::None,
            1,
            &build_sample_selection(1, &[0]).expect("identity selection"),
            &variant_record,
            output.as_mut_ptr() as usize,
            0,
            1,
            false,
            false,
            &mut thread_scratch,
        )
        .expect_err("non-biallelic variant should fail")
        .to_string()
        .contains("biallelic")
    );

    let mut bad_ploidy_bounds_block = probability_block(1, &[2], 0, 2, &[3, 0]);
    bad_ploidy_bounds_block[6] = 1;
    let variant_record = test_variant_record(bad_ploidy_bounds_block.len());
    assert!(
        decode_variant_dosages_into_variant_major_matrix(
            &bad_ploidy_bounds_block,
            CompressionType::None,
            1,
            &build_sample_selection(1, &[0]).expect("identity selection"),
            &variant_record,
            output.as_mut_ptr() as usize,
            0,
            1,
            false,
            false,
            &mut thread_scratch,
        )
        .expect_err("bad ploidy bounds should fail")
        .to_string()
        .contains("ploidy bounds")
    );

    let mut bad_bit_count_block = probability_block(1, &[2], 0, 2, &[3, 0]);
    bad_bit_count_block[probability_bit_count_offset(1)] = 0;
    let variant_record = test_variant_record(bad_bit_count_block.len());
    assert!(
        decode_variant_dosages_into_variant_major_matrix(
            &bad_bit_count_block,
            CompressionType::None,
            1,
            &build_sample_selection(1, &[0]).expect("identity selection"),
            &variant_record,
            output.as_mut_ptr() as usize,
            0,
            1,
            false,
            false,
            &mut thread_scratch,
        )
        .expect_err("bad bit count should fail")
        .to_string()
        .contains("requires a value")
    );

    let non_diploid_block = probability_block(1, &[1], 0, 2, &[3, 0]);
    let variant_record = test_variant_record(non_diploid_block.len());
    assert!(
        decode_variant_dosages_into_variant_major_matrix(
            &non_diploid_block,
            CompressionType::None,
            1,
            &build_sample_selection(1, &[0]).expect("identity selection"),
            &variant_record,
            output.as_mut_ptr() as usize,
            0,
            1,
            false,
            false,
            &mut thread_scratch,
        )
        .expect_err("non-diploid sample should fail")
        .to_string()
        .contains("non-diploid")
    );

    let invalid_phased_block = probability_block(1, &[2], 2, 2, &[3, 0]);
    let variant_record = test_variant_record(invalid_phased_block.len());
    assert!(
        decode_variant_dosages_into_variant_major_matrix(
            &invalid_phased_block,
            CompressionType::None,
            1,
            &build_sample_selection(1, &[0]).expect("identity selection"),
            &variant_record,
            output.as_mut_ptr() as usize,
            0,
            1,
            false,
            false,
            &mut thread_scratch,
        )
        .expect_err("invalid phased flag should fail")
        .to_string()
        .contains("phased flag")
    );
}

#[test]
fn byte_readers_probability_blocks_and_zlib_errors_report_clear_failures() {
    assert_eq!(read_u8_at(&[7], 0).expect("u8"), 7);
    assert_eq!(read_u16_at(&[1, 2], 0).expect("u16"), 513);
    assert_eq!(read_u32_at(&[1, 0, 0, 0], 0).expect("u32"), 1);
    assert!(read_exact_bytes(&[1, 2], 1, 5).expect_err("short read should fail").to_string().contains("end of file"));

    let mut thread_scratch = ThreadScratch::default();
    let mut profile = ThreadLocalProfileSnapshot::default();
    let variant_record = test_variant_record(3);
    assert!(
        read_probability_block(
            &[1, 2, 3],
            CompressionType::Zlib,
            &variant_record,
            &mut thread_scratch,
            &mut profile,
            true,
        )
        .expect_err("invalid zlib should fail")
        .to_string()
        .contains("I/O error")
    );

    let compressed_payload = zlib_compress(&[1, 2, 3, 4]);
    let mut successful_record = test_variant_record(compressed_payload.len());
    successful_record.declared_uncompressed_block_length = 4;
    let decompressed_block = read_probability_block(
        &compressed_payload,
        CompressionType::Zlib,
        &successful_record,
        &mut thread_scratch,
        &mut profile,
        true,
    )
    .expect("valid zlib block should decompress");
    assert_eq!(decompressed_block, &[1, 2, 3, 4]);

    let mut wrong_length_record = test_variant_record(compressed_payload.len());
    wrong_length_record.declared_uncompressed_block_length = 5;
    assert!(
        read_probability_block(
            &compressed_payload,
            CompressionType::Zlib,
            &wrong_length_record,
            &mut thread_scratch,
            &mut profile,
            false,
        )
        .expect_err("declared zlib length mismatch should fail")
        .to_string()
        .contains("declared")
    );
}
