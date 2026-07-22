use std::path::PathBuf;
use std::sync::Arc;

use g_genotype_contracts::{
    BgenSourceIdentity, ChunkOutputStatistics, NullableFloat32Column, RAW_DEFLATE_MEMBER_ALIGNMENT,
    VariantMetadataColumns, VariantMetadataInvariantError, VariantMetadataStore,
};

const FLOAT_TOLERANCE: f32 = 1.0e-7;

struct MetadataFixture {
    columns: VariantMetadataColumns,
    chromosome_x: Arc<str>,
}

fn build_metadata_fixture(range: std::ops::Range<usize>) -> MetadataFixture {
    let chromosome_22: Arc<str> = Arc::from("22");
    let chromosome_x: Arc<str> = Arc::from("X");
    let allele_a: Arc<str> = Arc::from("A");
    let allele_c: Arc<str> = Arc::from("C");
    let allele_g: Arc<str> = Arc::from("G");
    let allele_t: Arc<str> = Arc::from("T");
    let store = Arc::new(
        VariantMetadataStore::from_parts(
            vec![Arc::clone(&chromosome_22), Arc::clone(&chromosome_x), allele_a, allele_c, allele_g, allele_t]
                .into_boxed_slice(),
            vec![0, 1, 0].into_boxed_slice(),
            "rs1β".to_string().into_boxed_str(),
            vec![0, 3, 5, 5].into_boxed_slice(),
            vec![101, 202, 303].into_boxed_slice(),
            vec![2, 3, 4].into_boxed_slice(),
            vec![4, 5, 2].into_boxed_slice(),
        )
        .expect("test metadata store should satisfy its invariants"),
    );
    MetadataFixture {
        columns: VariantMetadataColumns::new(store, range).expect("test metadata range should be valid"),
        chromosome_x,
    }
}

fn metadata_store_result(
    chromosome_codes: Vec<u32>,
    variant_identifier_text: &str,
    variant_identifier_offsets: Vec<u32>,
    position: Vec<i64>,
    allele_one_codes: Vec<u32>,
    allele_two_codes: Vec<u32>,
) -> Result<VariantMetadataStore, VariantMetadataInvariantError> {
    VariantMetadataStore::from_parts(
        vec![Arc::from("22"), Arc::from("A")].into_boxed_slice(),
        chromosome_codes.into_boxed_slice(),
        variant_identifier_text.to_string().into_boxed_str(),
        variant_identifier_offsets.into_boxed_slice(),
        position.into_boxed_slice(),
        allele_one_codes.into_boxed_slice(),
        allele_two_codes.into_boxed_slice(),
    )
}

#[test]
fn raw_deflate_member_alignment_matches_the_shared_slab_contract() {
    assert_eq!(std::hint::black_box(RAW_DEFLATE_MEMBER_ALIGNMENT), 4);
}

#[test]
fn bgen_source_identity_retains_every_opened_file_field() {
    let identity = BgenSourceIdentity {
        configured_path: PathBuf::from("configured/input.bgen"),
        canonical_path: Some(PathBuf::from("/resolved/input.bgen")),
        device_identifier: 17,
        inode_identifier: 23,
        change_time_nanoseconds: -31,
        modification_time_nanoseconds: 47,
        file_size: 59,
    };

    let cloned_identity = identity.clone();

    assert_eq!(cloned_identity, identity);
    assert_eq!(cloned_identity.configured_path, PathBuf::from("configured/input.bgen"));
    assert_eq!(cloned_identity.canonical_path, Some(PathBuf::from("/resolved/input.bgen")));
    assert_eq!(cloned_identity.device_identifier, 17);
    assert_eq!(cloned_identity.inode_identifier, 23);
    assert_eq!(cloned_identity.change_time_nanoseconds, -31);
    assert_eq!(cloned_identity.modification_time_nanoseconds, 47);
    assert_eq!(cloned_identity.file_size, 59);
    assert!(format!("{cloned_identity:?}").contains("configured/input.bgen"));
}

#[test]
fn metadata_columns_expose_the_selected_range_without_copying_dictionary_values() {
    let MetadataFixture { columns, chromosome_x } = build_metadata_fixture(1..3);
    let cloned_columns = columns.clone();
    drop(columns);

    let mut chromosomes = cloned_columns.chromosomes();
    assert_eq!(chromosomes.len(), 2);
    assert_eq!(chromosomes.next(), Some("X"));
    assert_eq!(chromosomes.next(), Some("22"));
    assert_eq!(chromosomes.next(), None);
    assert_eq!(cloned_columns.variant_identifiers().collect::<Vec<_>>(), vec!["β", ""]);
    assert_eq!(cloned_columns.position(), &[202, 303]);
    assert_eq!(cloned_columns.allele_ones().collect::<Vec<_>>(), vec!["C", "G"]);
    assert_eq!(cloned_columns.allele_twos().collect::<Vec<_>>(), vec!["T", "A"]);
    assert_eq!(cloned_columns.len(), 2);
    assert!(!cloned_columns.is_empty());

    let shared_chromosome = cloned_columns.shared_chromosome(0).expect("the first selected chromosome should exist");
    assert!(Arc::ptr_eq(&shared_chromosome, &chromosome_x));
    assert_eq!(cloned_columns.shared_chromosome(2), None);
    assert_eq!(cloned_columns.shared_chromosome(usize::MAX), None);
}

#[test]
fn empty_metadata_range_has_empty_exact_size_views() {
    let fixture = build_metadata_fixture(2..2);

    assert!(fixture.columns.is_empty());
    assert_eq!(fixture.columns.len(), 0);
    assert_eq!(fixture.columns.chromosomes().len(), 0);
    assert_eq!(fixture.columns.variant_identifiers().len(), 0);
    assert_eq!(fixture.columns.allele_ones().len(), 0);
    assert_eq!(fixture.columns.allele_twos().len(), 0);
    assert_eq!(fixture.columns.position(), &[]);
    assert_eq!(fixture.columns.shared_chromosome(0), None);
}

#[test]
fn metadata_store_resolves_utf8_identifier_offsets() {
    let fixture = build_metadata_fixture(0..3);

    assert_eq!(fixture.columns.variant_identifiers().collect::<Vec<_>>(), vec!["rs1", "β", ""]);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn metadata_store_rejects_an_out_of_bounds_identifier_lookup() {
    let store = VariantMetadataStore::from_parts(
        vec![Arc::from("22"), Arc::from("A")].into_boxed_slice(),
        vec![0].into_boxed_slice(),
        "rs1".to_string().into_boxed_str(),
        vec![0, 3].into_boxed_slice(),
        vec![101].into_boxed_slice(),
        vec![1].into_boxed_slice(),
        vec![1].into_boxed_slice(),
    )
    .expect("test metadata store should satisfy its invariants");

    let _ = store.variant_identifier(1);
}

#[test]
fn metadata_store_accepts_valid_empty_parts() {
    let store = VariantMetadataStore::from_parts(
        Vec::<Arc<str>>::new().into_boxed_slice(),
        Vec::new().into_boxed_slice(),
        String::new().into_boxed_str(),
        vec![0].into_boxed_slice(),
        Vec::new().into_boxed_slice(),
        Vec::new().into_boxed_slice(),
        Vec::new().into_boxed_slice(),
    )
    .expect("empty metadata should satisfy every invariant");
    let columns = VariantMetadataColumns::new(Arc::new(store), 0..0).expect("the empty range should be valid");

    assert!(columns.is_empty());
    assert_eq!(columns.variant_identifiers().collect::<Vec<_>>(), Vec::<&str>::new());
}

#[test]
fn metadata_store_returns_exact_parallel_column_errors() {
    assert_eq!(
        metadata_store_result(vec![0, 0], "a", vec![0, 1], vec![101], vec![1], vec![1]).unwrap_err(),
        VariantMetadataInvariantError::ChromosomeCodeCountMismatch { variant_count: 1, chromosome_code_count: 2 }
    );
    assert_eq!(
        metadata_store_result(vec![0], "a", vec![0, 1], vec![101], Vec::new(), vec![1]).unwrap_err(),
        VariantMetadataInvariantError::AlleleOneCodeCountMismatch { variant_count: 1, allele_one_code_count: 0 }
    );
    assert_eq!(
        metadata_store_result(vec![0], "a", vec![0, 1], vec![101], vec![1], Vec::new()).unwrap_err(),
        VariantMetadataInvariantError::AlleleTwoCodeCountMismatch { variant_count: 1, allele_two_code_count: 0 }
    );
}

#[test]
fn metadata_store_returns_exact_identifier_offset_shape_errors() {
    assert_eq!(
        metadata_store_result(vec![0], "a", vec![0], vec![101], vec![1], vec![1]).unwrap_err(),
        VariantMetadataInvariantError::VariantIdentifierOffsetCountMismatch {
            variant_count: 1,
            expected_offset_count: 2,
            offset_count: 1,
        }
    );
    assert_eq!(
        metadata_store_result(vec![0], "abc", vec![1, 3], vec![101], vec![1], vec![1]).unwrap_err(),
        VariantMetadataInvariantError::VariantIdentifierOffsetStartMismatch { observed_offset: 1 }
    );
    assert_eq!(
        metadata_store_result(vec![0], "abc", vec![0, 2], vec![101], vec![1], vec![1]).unwrap_err(),
        VariantMetadataInvariantError::VariantIdentifierOffsetEndMismatch { text_length: 3, observed_offset: 2 }
    );
}

#[test]
fn metadata_store_returns_exact_identifier_offset_content_errors() {
    assert_eq!(
        metadata_store_result(vec![0, 0], "abc", vec![0, 4, 3], vec![101, 102], vec![1, 1], vec![1, 1]).unwrap_err(),
        VariantMetadataInvariantError::VariantIdentifierOffsetOutOfBounds {
            offset_index: 1,
            offset: 4,
            text_length: 3,
        }
    );
    assert_eq!(
        metadata_store_result(vec![0, 0], "aβ", vec![0, 2, 3], vec![101, 102], vec![1, 1], vec![1, 1]).unwrap_err(),
        VariantMetadataInvariantError::VariantIdentifierOffsetNotUtf8Boundary { offset_index: 1, offset: 2 }
    );
    assert_eq!(
        metadata_store_result(
            vec![0, 0, 0],
            "abc",
            vec![0, 2, 1, 3],
            vec![101, 102, 103],
            vec![1, 1, 1],
            vec![1, 1, 1],
        )
        .unwrap_err(),
        VariantMetadataInvariantError::VariantIdentifierOffsetOrder {
            preceding_offset_index: 1,
            following_offset_index: 2,
            preceding_offset: 2,
            following_offset: 1,
        }
    );
}

#[test]
fn metadata_invariant_display_handles_maximum_caller_index_without_overflow() {
    let error = VariantMetadataInvariantError::VariantIdentifierOffsetOrder {
        preceding_offset_index: usize::MAX,
        following_offset_index: 7,
        preceding_offset: 9,
        following_offset: 8,
    };

    assert_eq!(
        error.to_string(),
        "variant identifier offsets decrease between indices 18446744073709551615 and 7: 9 exceeds 8"
    );
}

#[test]
fn metadata_store_returns_exact_dictionary_code_errors() {
    assert_eq!(
        metadata_store_result(vec![2], "a", vec![0, 1], vec![101], vec![1], vec![1]).unwrap_err(),
        VariantMetadataInvariantError::ChromosomeCodeOutOfBounds { variant_index: 0, code: 2, dictionary_length: 2 }
    );
    assert_eq!(
        metadata_store_result(vec![0], "a", vec![0, 1], vec![101], vec![2], vec![1]).unwrap_err(),
        VariantMetadataInvariantError::AlleleOneCodeOutOfBounds { variant_index: 0, code: 2, dictionary_length: 2 }
    );
    assert_eq!(
        metadata_store_result(vec![0], "a", vec![0, 1], vec![101], vec![1], vec![2]).unwrap_err(),
        VariantMetadataInvariantError::AlleleTwoCodeOutOfBounds { variant_index: 0, code: 2, dictionary_length: 2 }
    );
}

#[test]
fn metadata_columns_return_exact_range_errors() {
    let store = Arc::new(
        metadata_store_result(vec![0], "a", vec![0, 1], vec![101], vec![1], vec![1])
            .expect("test metadata store should satisfy its invariants"),
    );
    let reversed_range_start = std::hint::black_box(1);
    let reversed_range_end = std::hint::black_box(0);

    assert_eq!(
        VariantMetadataColumns::new(Arc::clone(&store), reversed_range_start..reversed_range_end).unwrap_err(),
        VariantMetadataInvariantError::RangeStartAfterEnd { range_start: 1, range_end: 0 }
    );
    assert_eq!(
        VariantMetadataColumns::new(store, 0..2).unwrap_err(),
        VariantMetadataInvariantError::RangeOutOfBounds { range_start: 0, range_end: 2, variant_count: 1 }
    );
}

#[test]
fn nullable_float_column_packs_validity_bits_least_significant_bit_first() {
    let mut column = NullableFloat32Column { values: Vec::new(), validity_bytes: Vec::new() };
    let validity = [true, false, true, false, false, false, false, true, true, false, true];

    for (value_index, is_valid) in validity.into_iter().enumerate() {
        let value = f32::from(u16::try_from(value_index).expect("test value index should fit uint16")) + 0.25;
        column.push(value, is_valid);
    }

    assert_eq!(column.validity_bytes, vec![0b1000_0101, 0b0000_0101]);
    assert_eq!(column.values.len(), validity.len());
    for (value_index, value) in column.values.iter().enumerate() {
        let expected = f32::from(u16::try_from(value_index).expect("test value index should fit uint16")) + 0.25;
        assert!((value - expected).abs() < FLOAT_TOLERANCE);
    }
}

#[test]
fn chunk_output_statistics_preserve_output_columns() {
    let observed = ChunkOutputStatistics {
        allele_one_frequency: vec![0.25, 0.75],
        observation_count: vec![100, 99],
        info_score: NullableFloat32Column { values: vec![0.8, 0.0], validity_bytes: vec![0b0000_0001] },
    };
    let expected = ChunkOutputStatistics {
        allele_one_frequency: vec![0.25, 0.75],
        observation_count: vec![100, 99],
        info_score: NullableFloat32Column { values: vec![0.8, 0.0], validity_bytes: vec![0b0000_0001] },
    };

    assert_eq!(observed, expected);
    assert!(format!("{observed:?}").contains("allele_one_frequency"));
}
