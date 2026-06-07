use std::assert_matches;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use _core::genotype::bgen::{BgenReaderCore, CompressionType, set_bgen_decode_tile_variant_count};
use _core::genotype::common::{ChunkStats, GenotypeReaderCore, VariantMetadataColumns};
use _core::genotype::preprocess;
use _core::output::{OutputWriterSession, scan_committed_chunk_identifiers, validate_strict_manifest_chunks};
use _core::pipeline::Regenie2RunEngineCore;
use _core::regenie::{MultiPredictionSource, PredictionError, PredictionSource};
use _core::sample::{
    AlignmentInputs, MultiAlignmentInputs, SampleKeyMode, align_multi_sample_data,
    align_multi_sample_data_from_sample_file, align_sample_data, align_sample_data_from_sample_file,
};

static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

struct FixtureDirectory {
    path: PathBuf,
}

impl FixtureDirectory {
    fn new(label: &str) -> Self {
        let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("g-rust-coverage-{label}-{}-{fixture_id}", std::process::id()));
        fs::create_dir_all(&path).expect("fixture directory should be created");
        Self { path }
    }

    fn write_file(&self, file_name: &str, contents: &str) -> PathBuf {
        let path = self.path.join(file_name);
        fs::write(&path, contents).expect("fixture file should be written");
        path
    }
}

impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

fn assert_f32_vectors_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        assert!(
            (left_value - right_value).abs() <= tolerance,
            "left value {left_value} differed from right value {right_value} by more than {tolerance}"
        );
    }
}

fn haplotypes_bgen_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/bgen/haplotypes.bgen")
}

fn build_chunk_stats(row_count: usize) -> ChunkStats {
    ChunkStats {
        allele_one_frequency: vec![0.5; row_count],
        observation_count: vec![4; row_count],
        has_missing_values: false,
        dosage_sum: vec![4.0; row_count].into(),
        dosage_square_sum: vec![6.0; row_count],
        imputed_dosage_square_sum: vec![6.0; row_count],
        dosage_variance_numerator: vec![2.0; row_count],
        info_score: vec![Some(1.0); row_count],
        allele_count: vec![8.0; row_count].into(),
        minor_allele_count: vec![4.0; row_count],
        zero_count: vec![1; row_count],
        nonzero_count: vec![3; row_count],
        homozygous_reference_count: vec![1; row_count],
        heterozygous_count: vec![2; row_count],
        homozygous_alternate_count: vec![1; row_count],
        is_sparse_candidate: vec![false; row_count],
        is_rare_sparse_firth_candidate: vec![false; row_count],
    }
}

fn build_metadata(chunk_identifier: i64, row_count: usize) -> VariantMetadataColumns {
    VariantMetadataColumns {
        chromosome: vec!["22".to_string(); row_count],
        variant_identifier: (0..row_count).map(|row_index| format!("v{chunk_identifier}_{row_index}")).collect(),
        position: (0..row_count)
            .map(|row_index| chunk_identifier + i64::try_from(row_index).expect("row index fits i64") + 100)
            .collect(),
        allele_one: vec!["A".to_string(); row_count],
        allele_two: vec!["G".to_string(); row_count],
    }
}

fn write_session_chunk(session: &OutputWriterSession, chunk_identifier: i64, row_count: usize) {
    let metadata = build_metadata(chunk_identifier, row_count);
    let chunk_stats = build_chunk_stats(row_count);
    session
        .write_regenie2_native_chunk(
            chunk_identifier,
            chunk_identifier + i64::try_from(row_count).expect("row count fits i64"),
            &metadata,
            &chunk_stats,
            &vec![0.1; row_count],
            &vec![0.01; row_count],
            &vec![10.0; row_count],
            &vec![5.0; row_count],
            None,
        )
        .expect("native chunk should write");
}

fn write_minimal_bgen_header(path: &Path, header_length: u32, magic: [u8; 4], flags: u32) {
    let byte_count = usize::max(24, 4 + usize::try_from(header_length).expect("header length fits usize"));
    let mut bytes = vec![0_u8; byte_count];
    bytes[0..4].copy_from_slice(&20_u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&header_length.to_le_bytes());
    bytes[8..12].copy_from_slice(&0_u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&0_u32.to_le_bytes());
    bytes[16..20].copy_from_slice(&magic);
    if header_length >= 4 {
        let flag_offset = 4 + usize::try_from(header_length).expect("header length fits usize") - 4;
        bytes[flag_offset..flag_offset + 4].copy_from_slice(&flags.to_le_bytes());
    }
    fs::write(path, bytes).expect("minimal BGEN fixture should be written");
}

fn minimal_bgen_header_bytes(
    variant_count: u32,
    sample_count: u32,
    flags: u32,
    first_variant_offset: usize,
) -> Vec<u8> {
    let mut bytes = vec![0_u8; first_variant_offset];
    let offset = u32::try_from(first_variant_offset - 4).expect("BGEN offset fits u32");
    bytes[0..4].copy_from_slice(&offset.to_le_bytes());
    bytes[4..8].copy_from_slice(&20_u32.to_le_bytes());
    bytes[8..12].copy_from_slice(&variant_count.to_le_bytes());
    bytes[12..16].copy_from_slice(&sample_count.to_le_bytes());
    bytes[16..20].copy_from_slice(b"bgen");
    bytes[20..24].copy_from_slice(&flags.to_le_bytes());
    bytes
}

fn write_bgen_with_sample_block(path: &Path, sample_count: u32, sample_block: &[u8], first_variant_offset: usize) {
    let mut bytes = minimal_bgen_header_bytes(0, sample_count, (1_u32 << 31) | (2_u32 << 2), first_variant_offset);
    let sample_block_offset = 24;
    let required_length = sample_block_offset + sample_block.len();
    if bytes.len() < required_length {
        bytes.resize(required_length, 0);
    }
    bytes[sample_block_offset..sample_block_offset + sample_block.len()].copy_from_slice(sample_block);
    fs::write(path, bytes).expect("embedded sample BGEN fixture should be written");
}

fn write_bgen_with_single_variant(path: &Path, sample_count: u32, flags: u32, variant_payload: &[u8]) {
    let mut bytes = minimal_bgen_header_bytes(1, sample_count, flags, 24);
    bytes.extend_from_slice(variant_payload);
    fs::write(path, bytes).expect("single-variant BGEN fixture should be written");
}

fn write_bgen_with_variants(path: &Path, sample_count: u32, flags: u32, variant_payloads: &[Vec<u8>]) {
    let variant_count = u32::try_from(variant_payloads.len()).expect("variant count fits u32");
    let mut bytes = minimal_bgen_header_bytes(variant_count, sample_count, flags, 24);
    for variant_payload in variant_payloads {
        bytes.extend_from_slice(variant_payload);
    }
    fs::write(path, bytes).expect("multi-variant BGEN fixture should be written");
}

fn append_bgen_string(bytes: &mut Vec<u8>, value: &str) {
    let value_length = u16::try_from(value.len()).expect("BGEN string length fits u16");
    bytes.extend_from_slice(&value_length.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn bgen_variant_payload(
    variant_identifier: &str,
    rsid: &str,
    chromosome: &str,
    sample_count_in_probability_block: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    append_bgen_string(&mut bytes, variant_identifier);
    append_bgen_string(&mut bytes, rsid);
    append_bgen_string(&mut bytes, chromosome);
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(b"A");
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(b"G");
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&sample_count_in_probability_block.to_le_bytes());
    bytes
}

fn valid_bgen_variant_payload(sample_count_in_probability_block: u32) -> Vec<u8> {
    bgen_variant_payload("var", "rs", "22", sample_count_in_probability_block)
}

fn trusted_bgen_probability_block(sample_count: u32, probability_bytes: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&sample_count.to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.push(2);
    bytes.push(2);
    for _sample_index in 0..sample_count {
        bytes.push(2);
    }
    bytes.push(0);
    bytes.push(8);
    bytes.extend_from_slice(probability_bytes);
    bytes
}

fn trusted_bgen_variant_payload(
    variant_identifier: &str,
    rsid: &str,
    chromosome: &str,
    probability_block: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::new();
    append_bgen_string(&mut bytes, variant_identifier);
    append_bgen_string(&mut bytes, rsid);
    append_bgen_string(&mut bytes, chromosome);
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(b"A");
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(b"G");
    let probability_block_length = u32::try_from(probability_block.len()).expect("probability block fits u32");
    bytes.extend_from_slice(&probability_block_length.to_le_bytes());
    bytes.extend_from_slice(probability_block);
    bytes
}

#[test]
fn bgen_reader_exercises_metadata_dosage_preprocessing_and_profile_paths() {
    set_bgen_decode_tile_variant_count(1).expect("tile size should update");
    let reader = BgenReaderCore::open(&haplotypes_bgen_path(), false).expect("BGEN reader should open");
    reader.reset_profile();
    assert_eq!(reader.sample_count(), 4);
    assert_eq!(reader.variant_count(), 4);
    assert_eq!(reader.chromosome_boundary_indices(), vec![0, 4]);

    let metadata = reader.variant_metadata_slice(0, 2).expect("metadata slice should load");
    assert_eq!(metadata.chromosome.len(), 2);
    assert_eq!(metadata.variant_identifier.len(), 2);
    assert_eq!(metadata.position.len(), 2);
    assert_eq!(metadata.allele_one.len(), 2);
    assert_eq!(metadata.allele_two.len(), 2);

    let sample_indices = [3_i64, 1, 0];
    let dosage_values = reader.read_dosage_f32(&sample_indices, 0, 2).expect("dosages should decode");
    assert_eq!(dosage_values.len(), 6);
    assert!(dosage_values.iter().all(|value| (0.0..=2.0).contains(value)));
    let empty_dosage_values = reader.read_dosage_f32(&[], 0, 1).expect("empty sample selection should decode");
    assert!(empty_dosage_values.is_empty());
    let mut direct_output = vec![0.0_f32; 2];
    reader
        .read_dosage_f32_into_address(&[0], 0, 2, direct_output.as_mut_ptr() as usize, direct_output.len())
        .expect("direct pointer dosage read should decode");
    assert!(direct_output.iter().all(|value| (0.0..=2.0).contains(value)));

    reader.prepare_sample_selection(&sample_indices).expect("sample selection should prepare");
    let prepared_values = reader.read_dosage_f32_prepared(1, 3).expect("prepared dosages should decode");
    assert_eq!(prepared_values.len(), 6);
    let mut prepared_output = vec![0.0_f32; 3];
    reader
        .read_dosage_f32_into_address_prepared(0, 1, prepared_output.as_mut_ptr() as usize, prepared_output.len())
        .expect("prepared pointer dosage read should decode");
    let empty_row_stats = reader
        .read_preprocessed_dosage_f32_into_address_prepared(1, 1, prepared_output.as_mut_ptr() as usize, 0)
        .expect("empty row-major preprocessed range should return empty stats");
    assert!(empty_row_stats.allele_one_frequency.is_empty());

    let mut row_major_buffer = vec![0.0_f32; sample_indices.len() * 2];
    let row_major_stats = reader
        .read_preprocessed_dosage_f32_into_address_prepared(
            0,
            2,
            row_major_buffer.as_mut_ptr() as usize,
            row_major_buffer.len(),
        )
        .expect("row-major preprocessed dosages should decode");
    assert_eq!(row_major_stats.allele_one_frequency.len(), 2);

    let mut variant_major_buffer = vec![0.0_f32; sample_indices.len() * 2];
    let variant_major_stats = reader
        .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
            0,
            2,
            variant_major_buffer.as_mut_ptr() as usize,
            variant_major_buffer.len(),
        )
        .expect("variant-major preprocessed dosages should decode");
    assert_eq!(variant_major_stats.observation_count.len(), 2);

    let profile = reader.profile_snapshot();
    assert!(profile.sample_selection_prepare_count >= 1);
    assert!(profile.variant_decode_count >= 1);
    reader.reset_profile();
    assert_eq!(reader.profile_snapshot().variant_decode_count, 0);
    reader.clear_prepared_sample_selection().expect("sample selection should clear");
    let error = reader.read_dosage_f32_prepared(0, 1).expect_err("prepared read should require selected samples");
    assert!(error.to_string().contains("before binding aligned samples"));
    reader.prepare_sample_selection(&[]).expect("empty sample selection should prepare");
    let mut empty_variant_major_buffer = Vec::<f32>::new();
    let empty_sample_stats = reader
        .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
            0,
            2,
            empty_variant_major_buffer.as_mut_ptr() as usize,
            0,
        )
        .expect("variant-major preprocessed read should handle zero selected samples");
    assert_eq!(empty_sample_stats.allele_one_frequency.len(), 2);
    reader.clear_prepared_sample_selection().expect("sample selection should clear");
    assert!(reader.bgen_path().ends_with("haplotypes.bgen"));

    let trusted_error = BgenReaderCore::open(&haplotypes_bgen_path(), true)
        .expect("trusted BGEN reader should open")
        .validate_trusted_no_missing_diploid()
        .expect_err("phased fixture should be rejected by trusted validation");
    assert!(trusted_error.to_string().contains("phased"));

    let trusted_diploid_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/1kg_chr22_full.bgen");
    if trusted_diploid_path.exists() {
        let trusted_reader =
            BgenReaderCore::open(&trusted_diploid_path, true).expect("trusted diploid reader should open");
        trusted_reader.prepare_sample_selection(&[0, 1, 2, 3]).expect("trusted sample selection should prepare");
        let mut trusted_variant_major_buffer = vec![0.0_f32; 8];
        let trusted_stats = trusted_reader
            .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                0,
                2,
                trusted_variant_major_buffer.as_mut_ptr() as usize,
                trusted_variant_major_buffer.len(),
            )
            .expect("trusted variant-major preprocessed dosages should decode");
        assert_eq!(trusted_stats.allele_one_frequency.len(), 2);
    }
}

#[test]
fn bgen_reader_exercises_trusted_validation_and_variant_major_decode_paths() {
    let fixture = FixtureDirectory::new("bgen-trusted-reader");
    let probability_block = trusted_bgen_probability_block(3, &[0, 0, 255, 0, 0, 255]);
    let variant_payload = trusted_bgen_variant_payload("trusted-var", "rs-trusted", "22", &probability_block);
    let bgen_path = fixture.path.join("trusted.bgen");
    write_bgen_with_single_variant(&bgen_path, 3, 2 << 2, &variant_payload);

    let non_trusted_reader = BgenReaderCore::open(&bgen_path, false).expect("non-trusted reader should open");
    assert!(
        non_trusted_reader
            .mark_trusted_no_missing_diploid_validated()
            .expect_err("non-trusted reader cannot be marked trusted")
            .to_string()
            .contains("non-trusted")
    );

    let unvalidated_trusted_reader = BgenReaderCore::open(&bgen_path, true).expect("trusted reader should open");
    unvalidated_trusted_reader.prepare_sample_selection(&[0, 1, 2]).expect("identity selection should prepare");
    let mut unvalidated_packed_output = vec![0_u8; 6];
    assert!(
        unvalidated_trusted_reader
            .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
                0,
                1,
                unvalidated_packed_output.as_mut_ptr() as usize,
                unvalidated_packed_output.len(),
            )
            .expect_err("packed8 delivery should require trusted validation")
            .to_string()
            .contains("requires trusted no-missing diploid validation")
    );

    let trusted_reader = BgenReaderCore::open(&bgen_path, true).expect("trusted reader should open");
    trusted_reader.validate_trusted_no_missing_diploid().expect("trusted fixture should validate");
    trusted_reader.mark_trusted_no_missing_diploid_validated().expect("trusted mark should succeed");

    trusted_reader.prepare_sample_selection(&[0, 1, 2]).expect("identity selection should prepare");
    let mut variant_major_output = vec![f32::NAN; 3];
    let variant_major_stats = trusted_reader
        .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
            0,
            1,
            variant_major_output.as_mut_ptr() as usize,
            variant_major_output.len(),
        )
        .expect("trusted variant-major read should use the trusted decoder");
    assert_eq!(variant_major_stats.observation_count, vec![3]);
    assert_eq!(variant_major_stats.zero_count, vec![1]);
    assert_eq!(variant_major_output, vec![2.0, 0.0, 1.0]);

    let mut packed_output = vec![0_u8; 6];
    let packed_stats = trusted_reader
        .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
            0,
            1,
            packed_output.as_mut_ptr() as usize,
            packed_output.len(),
        )
        .expect("trusted packed8 probability pairs should decode");
    assert_eq!(packed_output, vec![0, 0, 255, 0, 0, 255]);
    assert_eq!(packed_stats.observation_count, variant_major_stats.observation_count);
    assert_eq!(packed_stats.zero_count, variant_major_stats.zero_count);
    assert_eq!(packed_stats.nonzero_count, variant_major_stats.nonzero_count);
    assert_f32_vectors_close(&packed_stats.dosage_sum, &variant_major_stats.dosage_sum, 1.0e-5);
    assert_f32_vectors_close(&packed_stats.dosage_square_sum, &variant_major_stats.dosage_square_sum, 1.0e-5);

    let empty_variant_stats = trusted_reader
        .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
            1,
            1,
            variant_major_output.as_mut_ptr() as usize,
            0,
        )
        .expect("empty trusted variant range should return empty stats");
    assert!(empty_variant_stats.allele_one_frequency.is_empty());

    trusted_reader.prepare_sample_selection(&[2, 0]).expect("non-contiguous selection should prepare");
    let prepared_values = trusted_reader.read_dosage_f32_prepared(0, 1).expect("prepared row read should decode");
    assert_eq!(prepared_values, vec![1.0, 2.0]);

    let mut row_major_output = vec![f32::NAN; 2];
    let row_major_stats = trusted_reader
        .read_preprocessed_dosage_f32_into_address_prepared(
            0,
            1,
            row_major_output.as_mut_ptr() as usize,
            row_major_output.len(),
        )
        .expect("trusted row-major preprocessed read should decode");
    assert_eq!(row_major_stats.observation_count, vec![2]);
    assert_eq!(row_major_output, vec![1.0, 2.0]);

    let mut noncontiguous_packed_output = vec![0_u8; 4];
    let noncontiguous_packed_stats = trusted_reader
        .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
            0,
            1,
            noncontiguous_packed_output.as_mut_ptr() as usize,
            noncontiguous_packed_output.len(),
        )
        .expect("trusted selected packed8 probability pairs should decode");
    assert_eq!(noncontiguous_packed_output, vec![0, 255, 0, 0]);
    assert_eq!(noncontiguous_packed_stats.observation_count, vec![2]);
    assert_f32_vectors_close(&noncontiguous_packed_stats.dosage_sum, &[3.0], 1.0e-5);
}

#[test]
fn trusted_bgen_identity_full_sample_chunk_decodes() {
    let trusted_diploid_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/1kg_chr22_full.bgen");
    if !trusted_diploid_path.exists() {
        return;
    }

    let selected_variant_count = 4;
    let reader = BgenReaderCore::open(&trusted_diploid_path, true).expect("trusted diploid reader should open");
    reader.mark_trusted_no_missing_diploid_validated().expect("trusted benchmark fixture should be marked validated");
    let all_sample_indices: Vec<i64> = (0..reader.sample_count())
        .map(|sample_index| i64::try_from(sample_index).expect("sample index should fit i64"))
        .collect();
    reader.prepare_sample_selection(&all_sample_indices).expect("identity sample selection should prepare");
    let mut output_values = vec![0.0_f32; reader.sample_count() * selected_variant_count];
    let chunk_stats = reader
        .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
            0,
            selected_variant_count,
            output_values.as_mut_ptr() as usize,
            output_values.len(),
        )
        .expect("trusted variant-major preprocessed dosages should decode");

    assert_eq!(chunk_stats.allele_one_frequency.len(), selected_variant_count);
    assert_eq!(chunk_stats.observation_count.len(), selected_variant_count);
    assert_eq!(chunk_stats.zero_count.len(), selected_variant_count);
    assert!(output_values.iter().all(|dosage_value| (0.0..=2.0).contains(dosage_value)));
}

#[test]
#[allow(clippy::too_many_lines)]
fn bgen_reader_covers_trait_object_and_header_error_contracts() {
    assert_eq!(CompressionType::try_from(0).expect("none compression"), CompressionType::None);
    assert_eq!(CompressionType::try_from(1).expect("zlib compression"), CompressionType::Zlib);
    assert!(
        CompressionType::try_from(3)
            .expect_err("unsupported compression should fail")
            .to_string()
            .contains("Unsupported")
    );

    let reader = BgenReaderCore::open(&haplotypes_bgen_path(), false).expect("BGEN reader should open");
    let genotype_reader: &dyn GenotypeReaderCore = &reader;
    assert_eq!(genotype_reader.sample_count(), 4);
    assert_eq!(genotype_reader.variant_count(), 4);
    assert_eq!(genotype_reader.sample_identifiers().len(), 4);
    assert_eq!(genotype_reader.chromosome_boundary_indices(), vec![0, 4]);
    genotype_reader.prepare_sample_selection(&[0, 1]).expect("trait object sample selection should prepare");
    let metadata = genotype_reader.variant_metadata_slice(0, 1).expect("trait object metadata should load");
    assert_eq!(metadata.chromosome.len(), 1);
    let mut output = vec![0.0_f32; 2];
    let stats = genotype_reader
        .read_preprocessed_dosage_f32_into_address_prepared(0, 1, output.as_mut_ptr() as usize, output.len())
        .expect("trait object preprocessed read should decode");
    assert_eq!(stats.observation_count.len(), 1);
    genotype_reader.clear_prepared_sample_selection().expect("trait object selection should clear");
    assert!(
        genotype_reader
            .variant_metadata_slice(3, 2)
            .expect_err("trait object invalid metadata range should fail")
            .to_string()
            .contains("bounds")
    );

    let fixture = FixtureDirectory::new("bgen-header-errors");
    assert!(BgenReaderCore::open(&fixture.path.join("missing.bgen"), false).is_err());
    let short_header_path = fixture.path.join("short-header.bgen");
    write_minimal_bgen_header(&short_header_path, 19, *b"bgen", 2 << 2);
    assert!(
        BgenReaderCore::open(&short_header_path, false)
            .expect_err("short header should fail")
            .to_string()
            .contains("at least 20")
    );
    let bad_magic_path = fixture.path.join("bad-magic.bgen");
    write_minimal_bgen_header(&bad_magic_path, 20, *b"nope", 2 << 2);
    assert!(
        BgenReaderCore::open(&bad_magic_path, false).expect_err("bad magic should fail").to_string().contains("magic")
    );
    let bad_layout_path = fixture.path.join("bad-layout.bgen");
    write_minimal_bgen_header(&bad_layout_path, 20, *b"bgen", 1 << 2);
    assert!(
        BgenReaderCore::open(&bad_layout_path, false)
            .expect_err("bad layout should fail")
            .to_string()
            .contains("Layout")
    );
    let unsupported_compression_path = fixture.path.join("bad-compression.bgen");
    write_minimal_bgen_header(&unsupported_compression_path, 20, *b"bgen", (2 << 2) | 3);
    assert!(
        BgenReaderCore::open(&unsupported_compression_path, false)
            .expect_err("bad compression should fail")
            .to_string()
            .contains("compression")
    );
    let empty_valid_path = fixture.path.join("empty-valid.bgen");
    write_minimal_bgen_header(&empty_valid_path, 20, *b"bgen", 2 << 2);
    let empty_reader = BgenReaderCore::open(&empty_valid_path, false).expect("empty BGEN should open");
    assert!(!empty_reader.contains_embedded_samples());
    assert!(empty_reader.sample_identifiers().is_empty());
    empty_reader.validate_trusted_no_missing_diploid().expect("empty trusted validation should pass");

    let sample_overlap_path = fixture.path.join("sample-overlap.bgen");
    let mut sample_overlap_block = Vec::new();
    sample_overlap_block.extend_from_slice(&8_u32.to_le_bytes());
    sample_overlap_block.extend_from_slice(&0_u32.to_le_bytes());
    write_bgen_with_sample_block(&sample_overlap_path, 0, &sample_overlap_block, 28);
    assert!(
        BgenReaderCore::open(&sample_overlap_path, false)
            .expect_err("overlapping sample block should fail")
            .to_string()
            .contains("overlaps")
    );
    let sample_count_mismatch_path = fixture.path.join("sample-count-mismatch.bgen");
    write_bgen_with_sample_block(&sample_count_mismatch_path, 2, &sample_overlap_block, 32);
    assert!(
        BgenReaderCore::open(&sample_count_mismatch_path, false)
            .expect_err("sample count mismatch should fail")
            .to_string()
            .contains("header reports")
    );
    let sample_length_mismatch_path = fixture.path.join("sample-length-mismatch.bgen");
    let mut sample_length_mismatch_block = Vec::new();
    sample_length_mismatch_block.extend_from_slice(&9_u32.to_le_bytes());
    sample_length_mismatch_block.extend_from_slice(&0_u32.to_le_bytes());
    sample_length_mismatch_block.push(0);
    write_bgen_with_sample_block(&sample_length_mismatch_path, 0, &sample_length_mismatch_block, 33);
    assert!(
        BgenReaderCore::open(&sample_length_mismatch_path, false)
            .expect_err("sample block length mismatch should fail")
            .to_string()
            .contains("sample block length")
    );

    let unsupported_allele_path = fixture.path.join("unsupported-allele.bgen");
    let mut unsupported_allele_payload = Vec::new();
    append_bgen_string(&mut unsupported_allele_payload, "var");
    append_bgen_string(&mut unsupported_allele_payload, "rs");
    append_bgen_string(&mut unsupported_allele_payload, "22");
    unsupported_allele_payload.extend_from_slice(&1_u32.to_le_bytes());
    unsupported_allele_payload.extend_from_slice(&3_u16.to_le_bytes());
    write_bgen_with_single_variant(&unsupported_allele_path, 0, 2 << 2, &unsupported_allele_payload);
    assert!(
        BgenReaderCore::open(&unsupported_allele_path, false)
            .expect_err("unsupported allele count should fail")
            .to_string()
            .contains("biallelic")
    );
    let compressed_prefix_path = fixture.path.join("compressed-prefix.bgen");
    let mut compressed_payload = valid_bgen_variant_payload(0);
    let payload_length_offset = compressed_payload.len() - 8;
    compressed_payload[payload_length_offset..payload_length_offset + 4].copy_from_slice(&2_u32.to_le_bytes());
    write_bgen_with_single_variant(&compressed_prefix_path, 0, (2 << 2) | 1, &compressed_payload);
    assert!(
        BgenReaderCore::open(&compressed_prefix_path, false)
            .expect_err("compressed prefix underflow should fail")
            .to_string()
            .contains("four-byte")
    );
    let beyond_end_path = fixture.path.join("beyond-end.bgen");
    let mut beyond_end_payload = valid_bgen_variant_payload(0);
    let beyond_length_offset = beyond_end_payload.len() - 8;
    beyond_end_payload[beyond_length_offset..beyond_length_offset + 4].copy_from_slice(&999_u32.to_le_bytes());
    beyond_end_payload.truncate(beyond_end_payload.len() - 4);
    write_bgen_with_single_variant(&beyond_end_path, 0, 2 << 2, &beyond_end_payload);
    assert!(
        BgenReaderCore::open(&beyond_end_path, false)
            .expect_err("variant beyond end should fail")
            .to_string()
            .contains("beyond")
    );
    let probability_sample_mismatch_path = fixture.path.join("probability-sample-mismatch.bgen");
    write_bgen_with_single_variant(&probability_sample_mismatch_path, 2, 2 << 2, &valid_bgen_variant_payload(1));
    assert!(
        BgenReaderCore::open(&probability_sample_mismatch_path, false)
            .expect_err("probability sample mismatch should fail")
            .to_string()
            .contains("file header reports")
    );
    let two_chromosome_path = fixture.path.join("two-chromosomes.bgen");
    write_bgen_with_variants(
        &two_chromosome_path,
        0,
        2 << 2,
        &[bgen_variant_payload("empty-rsid", "", "1", 0), bgen_variant_payload("with-rsid", "rs2", "2", 0)],
    );
    let two_chromosome_reader =
        BgenReaderCore::open(&two_chromosome_path, false).expect("two chromosome BGEN should open");
    assert_eq!(two_chromosome_reader.chromosome_boundary_indices(), vec![0, 1, 2]);
    let metadata = two_chromosome_reader.variant_metadata_slice(0, 2).expect("metadata should load");
    assert_eq!(metadata.variant_identifier, vec!["empty-rsid".to_string(), "rs2".to_string()]);
}

#[test]
fn bgen_reader_reports_bounds_selection_and_buffer_errors() {
    let reader = BgenReaderCore::open(&haplotypes_bgen_path(), false).expect("BGEN reader should open");
    assert!(reader.variant_metadata_slice(3, 2).expect_err("invalid range should fail").to_string().contains("bounds"));
    assert!(reader.read_dosage_f32(&[0], 0, 9).expect_err("invalid stop should fail").to_string().contains("bounds"));
    let sample_selection_error = reader.prepare_sample_selection(&[99]).expect_err("invalid sample should fail");
    assert!(sample_selection_error.to_string().to_ascii_lowercase().contains("sample"));
    assert!(
        reader
            .prepare_sample_selection(&[-1])
            .expect_err("negative sample should fail")
            .to_string()
            .contains("non-negative")
    );
    assert!(
        reader
            .prepare_sample_selection(&[1, 1])
            .expect_err("duplicate sample should fail")
            .to_string()
            .contains("more than once")
    );
    reader.prepare_sample_selection(&[0, 1]).expect("sample selection should prepare");
    let mut output = vec![0.0_f32; 1];
    assert!(
        reader
            .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                0,
                2,
                output.as_mut_ptr() as usize,
                output.len(),
            )
            .expect_err("shape mismatch should fail")
            .to_string()
            .contains("shape mismatch")
    );
    assert!(
        reader
            .read_dosage_f32_into_address(&[0, 1], 0, 2, output.as_mut_ptr() as usize, output.len())
            .expect_err("row-major buffer shape mismatch should fail")
            .to_string()
            .contains("shape mismatch")
    );
}

#[test]
fn pipeline_core_plans_chunks_with_resume_state() {
    let engine =
        Regenie2RunEngineCore::open_bgen(&haplotypes_bgen_path(), 2, None, false).expect("pipeline engine should open");
    let committed_chunks = [0_usize].into_iter().collect();
    let chunks = engine.plan_chunks(&committed_chunks).expect("chunks should plan");
    assert_eq!(chunks.iter().map(|chunk| chunk.variant_start_index).collect::<Vec<_>>(), vec![2]);
    assert_eq!(engine.reader().variant_count(), 4);
}

#[test]
fn output_session_finishes_records_manifest_and_supports_strict_resume_validation() {
    let fixture = FixtureDirectory::new("output");
    let run_directory = fixture.path.join("run");
    let chunks_directory = run_directory.join("chunks");
    fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
    fs::write(run_directory.join("run_manifest.json"), "{}\n").expect("manifest should be initialized");
    let session = OutputWriterSession::new(
        run_directory.to_string_lossy().into_owned(),
        chunks_directory.to_string_lossy().into_owned(),
        "regenie2_linear".to_string(),
        1,
        2,
        "arrow",
        false,
        2,
        "none".to_string(),
        "none".to_string(),
        true,
    )
    .expect("session should open");

    write_session_chunk(&session, 0, 2);
    write_session_chunk(&session, 2, 1);
    assert_eq!(session.finish().expect("session should finish"), None);

    assert_eq!(scan_committed_chunk_identifiers(&chunks_directory).expect("chunks should scan"), vec![0, 2]);
    let manifest_json = fs::read_to_string(run_directory.join("run_manifest.json")).expect("manifest should exist");
    assert_eq!(
        validate_strict_manifest_chunks(&chunks_directory, &manifest_json).expect("manifest should validate"),
        vec![0, 2],
    );
    let missing_chunks_directory = fixture.path.join("missing-chunks");
    assert!(
        scan_committed_chunk_identifiers(&missing_chunks_directory)
            .expect("missing chunk directory should scan as empty")
            .is_empty()
    );
    let manifest_value =
        serde_json::from_str::<serde_json::Value>(&manifest_json).expect("manifest should parse as JSON");
    let chunk_file_name = manifest_value
        .get("committed_chunks")
        .and_then(serde_json::Value::as_array)
        .and_then(|committed_chunks| committed_chunks.first())
        .and_then(|committed_chunk| committed_chunk.get("chunk_file_name"))
        .and_then(serde_json::Value::as_str)
        .expect("manifest should record a chunk file name");
    let missing_integer_manifest = format!(
        r#"{{"committed_chunks":[{{"variant_start_index":0,"variant_stop_index":2,"row_count":2,"chunk_file_name":"{chunk_file_name}"}}]}}"#
    );
    assert!(
        validate_strict_manifest_chunks(&chunks_directory, &missing_integer_manifest)
            .expect_err("manifest missing chunk identifier should fail")
            .to_string()
            .contains("chunk_identifier")
    );
    let missing_file_name_manifest =
        r#"{"committed_chunks":[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":2,"row_count":2}]}"#;
    assert!(
        validate_strict_manifest_chunks(&chunks_directory, missing_file_name_manifest)
            .expect_err("manifest missing chunk file name should fail")
            .to_string()
            .contains("chunk_file_name")
    );
    let missing_file_manifest = r#"{"committed_chunks":[{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":2,"row_count":2,"chunk_file_name":"missing.arrow"}]}"#;
    assert!(
        validate_strict_manifest_chunks(&chunks_directory, missing_file_manifest)
            .expect_err("manifest missing chunk file should fail")
            .to_string()
            .contains("missing chunk file")
    );
    let row_count_mismatch_manifest = format!(
        r#"{{"committed_chunks":[{{"chunk_identifier":0,"variant_start_index":0,"variant_stop_index":2,"row_count":99,"chunk_file_name":"{chunk_file_name}"}}]}}"#
    );
    assert!(
        validate_strict_manifest_chunks(&chunks_directory, &row_count_mismatch_manifest)
            .expect_err("manifest row count mismatch should fail")
            .to_string()
            .contains("row count")
    );
    let range_mismatch_manifest = format!(
        r#"{{"committed_chunks":[{{"chunk_identifier":0,"variant_start_index":1,"variant_stop_index":2,"row_count":2,"chunk_file_name":"{chunk_file_name}"}}]}}"#
    );
    assert!(
        validate_strict_manifest_chunks(&chunks_directory, &range_mismatch_manifest)
            .expect_err("manifest range mismatch should fail")
            .to_string()
            .contains("variant range")
    );
    let timing_json =
        fs::read_to_string(run_directory.join("output_stage_timings.json")).expect("timings should exist");
    assert!(timing_json.contains("rust_output_writer_total"));
}

#[test]
fn preprocess_public_helpers_cover_shape_missing_and_empty_paths() {
    let mut row_major_values = vec![0.0_f32, f32::NAN, 2.0, 1.0];
    let row_stats = preprocess::preprocess_row_major_dosage_matrix(&mut row_major_values, 2, 2)
        .expect("row-major preprocessing should handle missing values");
    assert!(row_stats.has_missing_values);
    assert_eq!(row_stats.observation_count, vec![2, 1]);
    assert!((row_major_values[1] - 1.0).abs() < f32::EPSILON);
    assert!(
        preprocess::preprocess_row_major_dosage_matrix(&mut [0.0_f32; 3], 2, 2)
            .expect_err("row-major shape mismatch should fail")
            .to_string()
            .contains("value count")
    );

    let variant_major_values = vec![0.0_f32, f32::NAN, 2.0, 1.0];
    let variant_stats = preprocess::summarize_variant_major_dosage_matrix(&variant_major_values, 2, 2)
        .expect("variant-major summarization should handle missing values");
    assert!(variant_stats.has_missing_values);
    assert_eq!(variant_stats.observation_count, vec![1, 2]);
    assert!(
        preprocess::summarize_variant_major_dosage_matrix(&[0.0_f32; 3], 2, 2)
            .expect_err("variant-major shape mismatch should fail")
            .to_string()
            .contains("value count")
    );
    let empty_stats = preprocess::build_empty_chunk_stats(3, true);
    assert_eq!(empty_stats.allele_one_frequency.len(), 3);
    assert!(empty_stats.has_missing_values);
}

#[test]
fn output_session_finalizes_and_marks_interrupted_runs() {
    let fixture = FixtureDirectory::new("finalize");
    let run_directory = fixture.path.join("run");
    let chunks_directory = run_directory.join("chunks");
    fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
    fs::write(run_directory.join("run_manifest.json"), "{}\n").expect("manifest should be initialized");
    let session = OutputWriterSession::new(
        run_directory.to_string_lossy().into_owned(),
        chunks_directory.to_string_lossy().into_owned(),
        "regenie2_binary".to_string(),
        1,
        1,
        "arrow",
        true,
        1,
        "zstd".to_string(),
        "none".to_string(),
        false,
    )
    .expect("session should open");

    write_session_chunk(&session, 0, 1);
    let final_path = session.finish().expect("session should finish").expect("final parquet should be returned");
    assert!(Path::new(&final_path).exists());

    let interrupted_run_directory = fixture.path.join("interrupted");
    let interrupted_chunks_directory = interrupted_run_directory.join("chunks");
    fs::create_dir_all(&interrupted_chunks_directory).expect("interrupted chunks directory should be created");
    fs::write(interrupted_run_directory.join("run_manifest.json"), "{}\n")
        .expect("interrupted manifest should be initialized");
    let interrupted_session = OutputWriterSession::new(
        interrupted_run_directory.to_string_lossy().into_owned(),
        interrupted_chunks_directory.to_string_lossy().into_owned(),
        "regenie2_linear".to_string(),
        1,
        1,
        "arrow",
        false,
        1,
        "none".to_string(),
        "none".to_string(),
        false,
    )
    .expect("interrupted session should open");
    write_session_chunk(&interrupted_session, 0, 1);
    interrupted_session.finish_interrupted("SIGTERM").expect("interrupted session should finish");
    let manifest_json = fs::read_to_string(interrupted_run_directory.join("run_manifest.json"))
        .expect("interrupted manifest should exist");
    assert!(manifest_json.contains("SIGTERM"));
}

#[test]
#[allow(clippy::too_many_lines)]
fn output_session_rejects_invalid_configuration_and_chunk_shapes() {
    let fixture = FixtureDirectory::new("output-errors");
    let run_directory = fixture.path.join("run");
    let chunks_directory = run_directory.join("chunks");
    fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
    let Err(writer_thread_error) = OutputWriterSession::new(
        run_directory.to_string_lossy().into_owned(),
        chunks_directory.to_string_lossy().into_owned(),
        "regenie2_linear".to_string(),
        0,
        1,
        "arrow",
        false,
        1,
        "none".to_string(),
        "none".to_string(),
        false,
    ) else {
        panic!("zero writer threads should fail");
    };
    assert!(writer_thread_error.to_string().contains("at least 1"));
    let Err(chunks_per_file_error) = OutputWriterSession::new(
        fixture.path.join("bad-chunks").to_string_lossy().into_owned(),
        fixture.path.join("bad-chunks/chunks").to_string_lossy().into_owned(),
        "regenie2_linear".to_string(),
        1,
        1,
        "arrow",
        false,
        0,
        "none".to_string(),
        "none".to_string(),
        false,
    ) else {
        panic!("zero chunks per file should fail");
    };
    assert!(chunks_per_file_error.to_string().contains("Chunks per Arrow file"));

    let session = OutputWriterSession::new(
        run_directory.to_string_lossy().into_owned(),
        chunks_directory.to_string_lossy().into_owned(),
        "unsupported".to_string(),
        1,
        1,
        "arrow",
        false,
        1,
        "none".to_string(),
        "none".to_string(),
        false,
    )
    .expect("session should open");
    let metadata = build_metadata(0, 1);
    let chunk_stats = build_chunk_stats(1);
    let error = session
        .write_regenie2_native_chunk(0, 1, &metadata, &chunk_stats, &[0.1], &[0.01], &[1.0], &[2.0], None)
        .expect_err("unsupported association mode should fail");
    assert!(error.to_string().contains("only supports"));
    session.abort().expect("session should abort");

    let mismatch_session = OutputWriterSession::new(
        fixture.path.join("mismatch").to_string_lossy().into_owned(),
        fixture.path.join("mismatch/chunks").to_string_lossy().into_owned(),
        "regenie2_linear".to_string(),
        1,
        1,
        "arrow",
        false,
        1,
        "none".to_string(),
        "none".to_string(),
        false,
    )
    .expect("mismatch session should open");
    let metadata = build_metadata(0, 2);
    let chunk_stats = build_chunk_stats(2);
    assert!(
        mismatch_session
            .write_regenie2_native_chunk(0, 1, &metadata, &chunk_stats, &[0.1], &[0.01], &[1.0], &[2.0], None)
            .expect_err("metadata bounds mismatch should fail")
            .to_string()
            .contains("bounds")
    );
    mismatch_session.abort().expect("mismatch session should abort");

    let length_session = OutputWriterSession::new(
        fixture.path.join("length").to_string_lossy().into_owned(),
        fixture.path.join("length/chunks").to_string_lossy().into_owned(),
        "regenie2_binary".to_string(),
        1,
        1,
        "arrow",
        false,
        1,
        "none".to_string(),
        "none".to_string(),
        false,
    )
    .expect("length session should open");
    let metadata = build_metadata(0, 2);
    let chunk_stats = build_chunk_stats(2);
    assert!(
        length_session
            .write_regenie2_native_chunk(0, 2, &metadata, &chunk_stats, &[0.1], &[0.01; 2], &[1.0; 2], &[2.0; 2], None)
            .expect_err("short beta vector should fail")
            .to_string()
            .contains("column lengths")
    );
    assert!(
        length_session
            .write_regenie2_native_chunk(
                0,
                2,
                &metadata,
                &chunk_stats,
                &[0.1; 2],
                &[0.01; 2],
                &[1.0; 2],
                &[2.0; 2],
                Some(&[1]),
            )
            .expect_err("short extra code vector should fail")
            .to_string()
            .contains("column lengths")
    );
    length_session.abort().expect("length session should abort");

    let closed_session = OutputWriterSession::new(
        fixture.path.join("closed").to_string_lossy().into_owned(),
        fixture.path.join("closed/chunks").to_string_lossy().into_owned(),
        "regenie2_linear".to_string(),
        1,
        1,
        "arrow",
        false,
        1,
        "none".to_string(),
        "none".to_string(),
        false,
    )
    .expect("closed session should open");
    closed_session.finish().expect("empty session should finish");
    assert!(closed_session.finish().expect("second finish should be harmless").is_none());
    assert!(
        closed_session
            .write_regenie2_native_chunk(
                0,
                1,
                &build_metadata(0, 1),
                &build_chunk_stats(1),
                &[0.1],
                &[0.01],
                &[1.0],
                &[2.0],
                None
            )
            .expect_err("write after close should fail")
            .to_string()
            .contains("already closed")
    );

    let worker_error_session = OutputWriterSession::new(
        fixture.path.join("worker-error").to_string_lossy().into_owned(),
        fixture.path.join("worker-error/chunks").to_string_lossy().into_owned(),
        "regenie2_linear".to_string(),
        1,
        1,
        "arrow",
        false,
        1,
        "unsupported-compression".to_string(),
        "none".to_string(),
        false,
    )
    .expect("worker error session should open");
    fs::create_dir_all(fixture.path.join("worker-error/chunks")).expect("worker error chunks directory should exist");
    write_session_chunk(&worker_error_session, 0, 1);
    assert!(
        worker_error_session
            .finish()
            .expect_err("unsupported compression should surface from worker")
            .to_string()
            .contains("compression")
    );
}

#[test]
fn sample_and_prediction_public_apis_cover_error_contracts_without_python() {
    let fixture = FixtureDirectory::new("sample-prediction");
    let phenotype_path = fixture
        .write_file("phenotypes.tsv", "FID\tIID\ttrait_a\ttrait_b\tcase\nF1\tI1\t1.0\t2.0\t1\nF2\tI2\t3.0\tNA\t2\n");
    let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF1\tI1\t40\nF2\tI2\t50\n");
    let aligned = align_sample_data(AlignmentInputs {
        sample_indices: vec![1, 0],
        family_identifiers: strings(&["F2", "F1"]),
        individual_identifiers: strings(&["I2", "I1"]),
        phenotype_path: phenotype_path.to_string_lossy().into_owned(),
        phenotype_name: "case".to_string(),
        covariate_path: Some(covariate_path.to_string_lossy().into_owned()),
        covariate_names: Some(strings(&["age"])),
        is_binary_trait: true,
        sample_key_mode: SampleKeyMode::FidIid,
    })
    .expect("sample data should align");
    assert_eq!(aligned.phenotype_vector, vec![0.0, 1.0]);

    let multi_aligned = align_multi_sample_data(MultiAlignmentInputs {
        sample_indices: vec![0, 1],
        family_identifiers: strings(&["F1", "F2"]),
        individual_identifiers: strings(&["I1", "I2"]),
        phenotype_path: phenotype_path.to_string_lossy().into_owned(),
        phenotype_names: strings(&["trait_a", "trait_b"]),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    })
    .expect("multi sample data should align");
    assert_eq!(multi_aligned.sample_indices, vec![0]);

    let loco_path = fixture.write_file("trait.loco", "FID_IID F2_I2 F1_I1\n22 2.0 1.0\n");
    let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));
    let prediction_source = PredictionSource::load(
        &prediction_list_path,
        "trait",
        &strings(&["F1", "F2"]),
        &strings(&["I1", "I2"]),
        SampleKeyMode::FidIid,
    )
    .expect("prediction source should load");
    assert_eq!(prediction_source.chromosome_predictions("chr22").expect("chr22 predictions"), &[1.0, 2.0]);
    let multi_prediction_source = MultiPredictionSource::load(
        &prediction_list_path,
        &strings(&["missing"]),
        &strings(&["F1"]),
        &strings(&["I1"]),
        SampleKeyMode::FidIid,
    )
    .expect_err("missing trait should fail");
    assert_matches!(multi_prediction_source, PredictionError::MissingPhenotype { .. });
}

#[test]
#[allow(clippy::too_many_lines)]
fn sample_alignment_public_apis_cover_table_and_input_errors() {
    let fixture = FixtureDirectory::new("sample-errors");
    let phenotype_path =
        fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\tother\nF1\tI1\t1.0\tNA\nF2\tI2\tNA\t2.0\n");
    let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF1\tI1\tbad\nF2\tI2\t50\n");
    let base_inputs = AlignmentInputs {
        sample_indices: vec![0, 1],
        family_identifiers: strings(&["F1", "F2"]),
        individual_identifiers: strings(&["I1", "I2"]),
        phenotype_path: phenotype_path.to_string_lossy().into_owned(),
        phenotype_name: "trait".to_string(),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };

    let mut length_inputs = base_inputs.clone();
    length_inputs.family_identifiers.pop();
    assert!(
        align_sample_data(length_inputs)
            .expect_err("single-trait length mismatch should fail")
            .contains("equal length")
    );

    let duplicate_inputs = AlignmentInputs {
        family_identifiers: strings(&["F1", "F1"]),
        individual_identifiers: strings(&["I1", "I1"]),
        ..base_inputs.clone()
    };
    assert!(
        align_sample_data(duplicate_inputs)
            .expect_err("duplicate fid/iid sample keys should fail")
            .contains("Duplicate sample key")
    );

    let no_match_inputs = AlignmentInputs {
        family_identifiers: strings(&["FX"]),
        individual_identifiers: strings(&["IX"]),
        sample_indices: vec![0],
        ..base_inputs.clone()
    };
    assert!(
        align_sample_data(no_match_inputs).expect_err("no aligned samples should fail").contains("No aligned samples")
    );

    let bad_covariate_inputs = AlignmentInputs {
        covariate_path: Some(covariate_path.to_string_lossy().into_owned()),
        covariate_names: Some(strings(&["age"])),
        ..base_inputs.clone()
    };
    assert!(
        align_sample_data(bad_covariate_inputs)
            .expect_err("invalid covariate value should fail")
            .contains("covariate value")
    );

    let inferred_empty_covariate_path = fixture.write_file("empty-covariates.tsv", "FID\tIID\nF1\tI1\n");
    let empty_covariate_inputs = AlignmentInputs {
        covariate_path: Some(inferred_empty_covariate_path.to_string_lossy().into_owned()),
        covariate_names: None,
        ..base_inputs.clone()
    };
    assert!(
        align_sample_data(empty_covariate_inputs)
            .expect_err("covariate table without data columns should fail")
            .contains("at least one")
    );

    let duplicate_phenotype_path =
        fixture.write_file("duplicate-phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\nF1\tI1\t2\n");
    let duplicate_phenotype_inputs = AlignmentInputs {
        phenotype_path: duplicate_phenotype_path.to_string_lossy().into_owned(),
        ..base_inputs.clone()
    };
    assert!(
        align_sample_data(duplicate_phenotype_inputs)
            .expect_err("duplicate phenotype keys should fail")
            .contains("Duplicate sample key")
    );

    let duplicate_iid_phenotype_path = fixture.write_file("duplicate-iid-phenotypes.tsv", "IID\ttrait\nI1\t1\nI1\t2\n");
    let duplicate_iid_inputs = AlignmentInputs {
        sample_indices: vec![0],
        family_identifiers: strings(&[""]),
        individual_identifiers: strings(&["I1"]),
        phenotype_path: duplicate_iid_phenotype_path.to_string_lossy().into_owned(),
        sample_key_mode: SampleKeyMode::Iid,
        ..base_inputs.clone()
    };
    assert!(
        align_sample_data(duplicate_iid_inputs)
            .expect_err("duplicate phenotype IID should fail")
            .contains("Duplicate IID")
    );

    let mut multi_length_inputs = MultiAlignmentInputs {
        sample_indices: vec![0, 1],
        family_identifiers: strings(&["F1", "F2"]),
        individual_identifiers: strings(&["I1", "I2"]),
        phenotype_path: phenotype_path.to_string_lossy().into_owned(),
        phenotype_names: strings(&["trait", "other"]),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };
    multi_length_inputs.individual_identifiers.pop();
    assert!(
        align_multi_sample_data(multi_length_inputs)
            .expect_err("multi-trait length mismatch should fail")
            .contains("equal length")
    );

    let no_common_multi_inputs = MultiAlignmentInputs {
        sample_indices: vec![0, 1],
        family_identifiers: strings(&["F1", "F2"]),
        individual_identifiers: strings(&["I1", "I2"]),
        phenotype_path: phenotype_path.to_string_lossy().into_owned(),
        phenotype_names: strings(&["trait", "other"]),
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_key_mode: SampleKeyMode::FidIid,
    };
    assert!(
        align_multi_sample_data(no_common_multi_inputs)
            .expect_err("multi-trait complete-case intersection should fail")
            .contains("complete-case")
    );

    let short_sample_path = fixture.write_file("short.sample", "ID_1 ID_2 missing\n");
    assert!(
        align_sample_data_from_sample_file(
            &short_sample_path,
            1,
            phenotype_path.to_string_lossy().into_owned(),
            "trait".to_string(),
            None,
            None,
            false,
            SampleKeyMode::FidIid,
        )
        .expect_err("sample file with one header should fail")
        .contains("at least two header lines")
    );
    let header_mismatch_sample_path = fixture.write_file("header-mismatch.sample", "ID_1 ID_2\n0\nF1 I1\n");
    assert!(
        align_multi_sample_data_from_sample_file(
            &header_mismatch_sample_path,
            1,
            phenotype_path.to_string_lossy().into_owned(),
            strings(&["trait"]),
            None,
            None,
            false,
            SampleKeyMode::FidIid,
        )
        .expect_err("sample header mismatch should fail")
        .contains("different column counts")
    );
    let bad_type_sample_path = fixture.write_file("bad-type.sample", "ID_1 ID_2\nD 0\nF1 I1\n");
    assert!(
        align_sample_data_from_sample_file(
            &bad_type_sample_path,
            1,
            phenotype_path.to_string_lossy().into_owned(),
            "trait".to_string(),
            None,
            None,
            false,
            SampleKeyMode::FidIid,
        )
        .expect_err("sample first identifier type should fail")
        .contains("first identifier")
    );
    let bad_id2_type_sample_path = fixture.write_file("bad-id2-type.sample", "ID_1 ID_2\n0 D\nF1 I1\n");
    assert!(
        align_sample_data_from_sample_file(
            &bad_id2_type_sample_path,
            1,
            phenotype_path.to_string_lossy().into_owned(),
            "trait".to_string(),
            None,
            None,
            false,
            SampleKeyMode::FidIid,
        )
        .expect_err("sample ID_2 type should fail")
        .contains("ID_2")
    );
    let row_length_sample_path = fixture.write_file("row-length.sample", "ID_1 ID_2 missing\n0 0 0\nF1 I1\n");
    assert!(
        align_sample_data_from_sample_file(
            &row_length_sample_path,
            1,
            phenotype_path.to_string_lossy().into_owned(),
            "trait".to_string(),
            None,
            None,
            false,
            SampleKeyMode::FidIid,
        )
        .expect_err("sample row length should fail")
        .contains("header declares")
    );
    let row_count_sample_path = fixture.write_file("row-count.sample", "ID_1 ID_2\n0 0\nF1 I1\n");
    assert!(
        align_sample_data_from_sample_file(
            &row_count_sample_path,
            2,
            phenotype_path.to_string_lossy().into_owned(),
            "trait".to_string(),
            None,
            None,
            false,
            SampleKeyMode::FidIid,
        )
        .expect_err("sample count mismatch should fail")
        .contains("BGEN sample count")
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn prediction_sources_cover_file_header_alignment_and_matrix_errors() {
    let fixture = FixtureDirectory::new("prediction-errors");
    let target_families = strings(&["F1", "F2"]);
    let target_individuals = strings(&["I1", "I2"]);
    assert_matches!(
        PredictionSource::load(
            &fixture.path.join("missing.list"),
            "trait",
            &target_families,
            &target_individuals,
            SampleKeyMode::FidIid,
        ),
        Err(PredictionError::PredictionListNotFound(_))
    );
    let empty_list_path = fixture.write_file("empty.list", "\n\n");
    assert_matches!(
        PredictionSource::load(&empty_list_path, "trait", &target_families, &target_individuals, SampleKeyMode::FidIid),
        Err(PredictionError::EmptyPredictionList(_))
    );
    let malformed_list_path = fixture.write_file("malformed.list", "trait a b\n");
    assert_matches!(
        PredictionSource::load(
            &malformed_list_path,
            "trait",
            &target_families,
            &target_individuals,
            SampleKeyMode::FidIid
        ),
        Err(PredictionError::InvalidPredictionListLine { .. })
    );
    let missing_loco_list_path = fixture.write_file("missing-loco.list", "trait missing.loco\n");
    assert_matches!(
        PredictionSource::load(
            &missing_loco_list_path,
            "trait",
            &target_families,
            &target_individuals,
            SampleKeyMode::FidIid
        ),
        Err(PredictionError::LocoFileNotFound(_))
    );

    for (file_name, contents, expected_fragment) in [
        ("empty-header.loco", "FID_IID\n22 0.1\n", "at least"),
        ("bad-marker.loco", "IID F1_I1\n22 0.1\n", "found"),
        ("bad-sample.loco", "FID_IID F1I1\n22 0.1\n", "sample identifier"),
        ("short-data.loco", "FID_IID F1_I1\n22\n", "data line"),
        ("count-mismatch.loco", "FID_IID F1_I1 F2_I2\n22 0.1\n", "expected 2"),
        ("invalid-value.loco", "FID_IID F1_I1\n22 nope\n", "prediction"),
        ("duplicate-chromosome.loco", "FID_IID F1_I1\n22 0.1\nchr22 0.2\n", "Duplicate"),
        ("missing-predictions.loco", "FID_IID F1_I1\n", "no chromosome"),
        ("duplicate-loco-key.loco", "FID_IID F1_I1 F1_I1\n22 0.1 0.2\n", "Duplicate"),
    ] {
        let loco_path = fixture.write_file(file_name, contents);
        let prediction_list_path =
            fixture.write_file(&format!("{file_name}.list"), &format!("trait {}\n", loco_path.display()));
        let error = PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::FidIid,
        )
        .expect_err("malformed LOCO fixture should fail");
        assert!(
            error.to_string().to_ascii_lowercase().contains(&expected_fragment.to_ascii_lowercase()),
            "unexpected error for {file_name}: {error}"
        );
    }

    let good_loco_path = fixture.write_file("good.loco", "FID_IID F1_I1 F2_I2\n22 0.1 0.2\n");
    let second_loco_path = fixture.write_file("second.loco", "FID_IID F1_I1 F2_I2\n1 1.1 1.2\n");
    let prediction_list_path = fixture.write_file(
        "good.list",
        &format!("trait {}\nother {}\n", good_loco_path.display(), second_loco_path.display()),
    );
    assert_matches!(
        PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F1"]),
            &strings(&["I1", "I2"]),
            SampleKeyMode::FidIid,
        ),
        Err(PredictionError::TargetSampleLengthMismatch)
    );
    assert_matches!(
        PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F1", "F1"]),
            &strings(&["I1", "I1"]),
            SampleKeyMode::FidIid,
        ),
        Err(PredictionError::DuplicateTargetSampleKey { .. })
    );
    assert_matches!(
        PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F1", "F2"]),
            &strings(&["I1", "I1"]),
            SampleKeyMode::Iid,
        ),
        Err(PredictionError::DuplicateTargetIid { .. })
    );
    let duplicate_iid_loco_path = fixture.write_file("duplicate-iid.loco", "FID_IID F1_I1 F2_I1\n22 0.1 0.2\n");
    let duplicate_iid_list_path =
        fixture.write_file("duplicate-iid.list", &format!("trait {}\n", duplicate_iid_loco_path.display()));
    assert_matches!(
        PredictionSource::load(
            &duplicate_iid_list_path,
            "trait",
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::Iid,
        ),
        Err(PredictionError::DuplicateLocoIid { .. })
    );
    let many_missing_error = PredictionSource::load(
        &prediction_list_path,
        "trait",
        &strings(&["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]),
        &strings(&["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]),
        SampleKeyMode::FidIid,
    )
    .expect_err("missing target samples should fail");
    assert!(many_missing_error.to_string().contains("6 total"));

    let multi_source = MultiPredictionSource::load(
        &prediction_list_path,
        &strings(&["trait", "other"]),
        &target_families,
        &target_individuals,
        SampleKeyMode::FidIid,
    )
    .expect("multi prediction source should load");
    assert_matches!(multi_source.chromosome_prediction_matrix("X"), Err(PredictionError::MissingChromosome { .. }));
}
