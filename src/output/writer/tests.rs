use std::assert_matches;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{Array, Float32Array, Int32Array};
use arrow::ipc::reader::FileReader as ArrowFileReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader as ParquetFileReader, SerializedFileReader};

use crate::genotype::common::{ChunkStats, VariantMetadataColumns};
use crate::output::{NativeChunkHandle, finalization, manifest};

use super::*;

fn build_test_chunk_handle(chunk_identifier: i64) -> NativeChunkHandle {
    NativeChunkHandle::new(
        Arc::new(VariantMetadataColumns {
            chromosome: vec!["22".to_string()],
            variant_identifier: vec![format!("variant{chunk_identifier}")],
            position: vec![100 + chunk_identifier],
            allele_one: vec!["A".to_string()],
            allele_two: vec!["G".to_string()],
        }),
        Arc::new(ChunkStats {
            allele_one_frequency: vec![0.5],
            observation_count: vec![100],
            has_missing_values: false,
            dosage_sum: vec![0.0].into(),
            dosage_square_sum: vec![0.0],
            imputed_dosage_square_sum: vec![0.0],
            dosage_variance_numerator: vec![0.0],
            info_score: vec![Some(0.9)],
            allele_count: vec![0.0].into(),
            minor_allele_count: vec![0.0],
            zero_count: vec![0],
            nonzero_count: vec![0],
            homozygous_reference_count: vec![0],
            heterozygous_count: vec![0],
            homozygous_alternate_count: vec![0],
            is_sparse_candidate: vec![false],
            is_rare_sparse_firth_candidate: vec![false],
        }),
        chunk_identifier,
    )
}

fn build_test_chunk_with_handle(chunk_handle: NativeChunkHandle, extra_code: Option<Vec<i32>>) -> RegenieStep2ChunkJob {
    RegenieStep2ChunkJob {
        chunk_handle,
        beta: Arc::new(Float32Array::from(vec![0.1])),
        se: Arc::new(Float32Array::from(vec![0.01])),
        chisq: Arc::new(Float32Array::from(vec![10.0])),
        log10p: Arc::new(Float32Array::from(vec![5.0])),
        extra_code: extra_code.map(|values| Arc::new(Int32Array::from(values)) as ArrayRef),
    }
}

fn build_test_chunk(chunk_identifier: i64, extra_code: Option<Vec<i32>>) -> RegenieStep2ChunkJob {
    build_test_chunk_with_handle(build_test_chunk_handle(chunk_identifier), extra_code)
}

fn build_test_batch(chunks: Vec<RegenieStep2ChunkJob>) -> RegenieStep2ChunkWriteBatch {
    let first_chunk_identifier = chunks.first().map_or(0, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    let last_chunk_identifier =
        chunks.last().map_or(first_chunk_identifier, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    RegenieStep2ChunkWriteBatch {
        chunk_file_name: build_chunk_file_name(first_chunk_identifier, last_chunk_identifier),
        chunks,
    }
}

fn build_test_record_batches(chunks: Vec<RegenieStep2ChunkJob>) -> Vec<RecordBatch> {
    let write_batch = build_test_batch(chunks);
    let chunk_commits = build_run_manifest_chunk_commits(&write_batch, OutputFileFormat::Arrow, "none")
        .expect("chunk commits should build");
    let chunk_schema = build_regenie_step2_chunk_file_schema(&chunk_commits, OutputStatisticDtype::Float32)
        .expect("chunk schema should build");
    build_regenie_step2_record_batches(write_batch, chunk_schema).expect("record batches should build").record_batches
}

fn build_test_record_batch(chunk: RegenieStep2ChunkJob) -> RecordBatch {
    build_test_record_batches(vec![chunk]).pop().expect("one record batch should be present")
}

fn create_test_directory() -> PathBuf {
    let unique_suffix =
        SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
    let directory_path = std::env::temp_dir().join(format!("g-output-rust-test-{unique_suffix}"));
    std::fs::create_dir_all(&directory_path).expect("test directory should be created");
    directory_path
}

#[test]
fn runtime_error_helper_preserves_message() {
    let error = OutputWriterError::runtime("worker failed");

    assert_matches!(error, OutputWriterError::Runtime(message) if message == "worker failed");
}

#[test]
fn linear_record_batch_uses_shared_schema_and_null_extra() {
    let record_batch = build_test_record_batch(build_test_chunk(0, None));

    assert_eq!(record_batch.schema().fields().len(), 16);
    assert!(record_batch.schema().field_with_name("chunk_identifier").is_err());
    assert!(record_batch.schema().field_with_name("INFO").expect("INFO field should exist").is_nullable());
    assert!(record_batch.schema().field_with_name("EXTRA").expect("EXTRA field should exist").is_nullable());
    assert!(
        record_batch
            .schema()
            .field_with_name("CORRECTION_METHOD")
            .expect("CORRECTION_METHOD field should exist")
            .is_nullable()
    );
    assert!(
        record_batch
            .schema()
            .field_with_name("CORRECTION_STATUS")
            .expect("CORRECTION_STATUS field should exist")
            .is_nullable()
    );
    assert_eq!(record_batch.num_rows(), 1);
    let info_array = record_batch
        .column_by_name("INFO")
        .expect("INFO column should exist")
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("INFO column should be a float32 array");
    assert!((info_array.value(0) - 0.9).abs() < f32::EPSILON);
    assert_eq!(record_batch.column_by_name("EXTRA").expect("EXTRA column should exist").null_count(), 1);
    let correction_method_array = record_batch
        .column_by_name("CORRECTION_METHOD")
        .expect("CORRECTION_METHOD column should exist")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("CORRECTION_METHOD column should be a string array");
    let correction_status_array = record_batch
        .column_by_name("CORRECTION_STATUS")
        .expect("CORRECTION_STATUS column should exist")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("CORRECTION_STATUS column should be a string array");
    assert_eq!(correction_method_array.value(0), "score");
    assert_eq!(correction_status_array.value(0), "success");
}

#[test]
fn record_batch_build_reuses_constant_arrays_by_row_count() {
    let record_batches = build_test_record_batches(vec![build_test_chunk(0, None), build_test_chunk(1, None)]);

    assert_eq!(record_batches.len(), 2);
    let first_test_array = record_batches[0].column_by_name("TEST").expect("TEST column should exist");
    let second_test_array = record_batches[1].column_by_name("TEST").expect("TEST column should exist");
    let first_extra_array = record_batches[0].column_by_name("EXTRA").expect("EXTRA column should exist");
    let second_extra_array = record_batches[1].column_by_name("EXTRA").expect("EXTRA column should exist");

    assert!(Arc::ptr_eq(first_test_array, second_test_array));
    assert!(Arc::ptr_eq(first_extra_array, second_extra_array));
}

#[test]
fn record_batch_build_reuses_cached_chunk_metadata_and_statistic_arrays() {
    let chunk_handle = build_test_chunk_handle(0);
    let first_record_batch = build_test_record_batch(build_test_chunk_with_handle(chunk_handle.clone(), None));
    let second_record_batch = build_test_record_batch(build_test_chunk_with_handle(chunk_handle, None));

    let first_chromosome_array = first_record_batch.column_by_name("CHROM").expect("CHROM column should exist");
    let second_chromosome_array = second_record_batch.column_by_name("CHROM").expect("CHROM column should exist");
    let first_position_array = first_record_batch.column_by_name("GENPOS").expect("GENPOS column should exist");
    let second_position_array = second_record_batch.column_by_name("GENPOS").expect("GENPOS column should exist");
    let first_identifier_array = first_record_batch.column_by_name("ID").expect("ID column should exist");
    let second_identifier_array = second_record_batch.column_by_name("ID").expect("ID column should exist");
    let first_frequency_array = first_record_batch.column_by_name("A1FREQ").expect("A1FREQ column should exist");
    let second_frequency_array = second_record_batch.column_by_name("A1FREQ").expect("A1FREQ column should exist");
    let first_info_array = first_record_batch.column_by_name("INFO").expect("INFO column should exist");
    let second_info_array = second_record_batch.column_by_name("INFO").expect("INFO column should exist");
    let first_observation_count_array = first_record_batch.column_by_name("N").expect("N column should exist");
    let second_observation_count_array = second_record_batch.column_by_name("N").expect("N column should exist");

    assert!(Arc::ptr_eq(first_chromosome_array, second_chromosome_array));
    assert!(Arc::ptr_eq(first_position_array, second_position_array));
    assert!(Arc::ptr_eq(first_identifier_array, second_identifier_array));
    assert!(Arc::ptr_eq(first_frequency_array, second_frequency_array));
    assert!(Arc::ptr_eq(first_info_array, second_info_array));
    assert!(Arc::ptr_eq(first_observation_count_array, second_observation_count_array));
}

#[test]
fn binary_record_batch_maps_extra_codes_with_same_schema() {
    let linear_record_batch = build_test_record_batch(build_test_chunk(0, None));
    let binary_record_batch = build_test_record_batch(build_test_chunk(1, Some(vec![1])));

    assert_eq!(linear_record_batch.schema().fields(), binary_record_batch.schema().fields());
    assert_eq!(binary_record_batch.column_by_name("EXTRA").expect("EXTRA column should exist").null_count(), 1);
    let correction_method_array = binary_record_batch
        .column_by_name("CORRECTION_METHOD")
        .expect("CORRECTION_METHOD column should exist")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("CORRECTION_METHOD column should be a string array");
    let correction_status_array = binary_record_batch
        .column_by_name("CORRECTION_STATUS")
        .expect("CORRECTION_STATUS column should exist")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("CORRECTION_STATUS column should be a string array");
    assert_eq!(correction_method_array.value(0), "firth_approximate");
    assert_eq!(correction_status_array.value(0), "success");
}

#[test]
fn parquet_record_batch_dictionary_encodes_correction_columns_with_public_file_schema() {
    let write_batch = build_test_batch(vec![build_test_chunk(0, Some(vec![1]))]);
    let chunk_commits = build_run_manifest_chunk_commits(&write_batch, OutputFileFormat::Parquet, "none")
        .expect("chunk commits should build");
    let public_chunk_schema = build_regenie_step2_chunk_file_schema(&chunk_commits, OutputStatisticDtype::Float32)
        .expect("public chunk schema should build");
    let parquet_record_batch_schema = build_regenie_step2_parquet_record_batch_schema(public_chunk_schema.as_ref());

    assert_eq!(
        public_chunk_schema.field_with_name("CORRECTION_METHOD").expect("public method field should exist").data_type(),
        &DataType::Utf8
    );
    assert_eq!(
        public_chunk_schema.field_with_name("CORRECTION_STATUS").expect("public status field should exist").data_type(),
        &DataType::Utf8
    );
    assert_eq!(
        parquet_record_batch_schema
            .field_with_name("CORRECTION_METHOD")
            .expect("Parquet method field should exist")
            .data_type(),
        &DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8))
    );
    assert_eq!(
        parquet_record_batch_schema
            .field_with_name("CORRECTION_STATUS")
            .expect("Parquet status field should exist")
            .data_type(),
        &DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8))
    );

    let mut array_cache = RegenieStep2RecordBatchArrayCache::default();
    let record_batch_build_result = build_regenie_step2_record_batch(
        build_test_chunk(0, Some(vec![1])),
        parquet_record_batch_schema,
        &mut array_cache,
        RegenieStep2CorrectionArrayEncoding::Dictionary,
    )
    .expect("Parquet record batch should build");
    let correction_method_array = record_batch_build_result
        .record_batch
        .column_by_name("CORRECTION_METHOD")
        .expect("CORRECTION_METHOD column should exist")
        .as_any()
        .downcast_ref::<DictionaryArray<UInt8Type>>()
        .expect("CORRECTION_METHOD column should be dictionary encoded");
    let correction_status_array = record_batch_build_result
        .record_batch
        .column_by_name("CORRECTION_STATUS")
        .expect("CORRECTION_STATUS column should exist")
        .as_any()
        .downcast_ref::<DictionaryArray<UInt8Type>>()
        .expect("CORRECTION_STATUS column should be dictionary encoded");

    assert_eq!(correction_method_array.keys().value(0), CORRECTION_METHOD_FIRTH_APPROXIMATE_KEY);
    assert_eq!(correction_status_array.keys().value(0), CORRECTION_STATUS_SUCCESS_KEY);
    let method_values = correction_method_array
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("method dictionary values should be strings");
    let status_values = correction_status_array
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("status dictionary values should be strings");
    assert_eq!(method_values.value(usize::from(correction_method_array.keys().value(0))), "firth_approximate");
    assert_eq!(status_values.value(usize::from(correction_status_array.keys().value(0))), "success");
}

#[test]
fn grouped_arrow_file_writes_one_batch_per_compute_chunk_with_metadata_commits() {
    let run_directory = create_test_directory();
    let chunks_directory = run_directory.join("chunks");
    std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
    let write_result = write_regenie_step2_chunk_job(
        &chunks_directory,
        build_test_batch(vec![build_test_chunk(0, Some(vec![1])), build_test_chunk(1, Some(vec![2]))]),
        OutputFileFormat::Arrow,
        OutputStatisticDtype::Float32,
        "none",
        "none",
    )
    .expect("chunk batch should write");

    assert_eq!(write_result.chunk_commits.len(), 2);
    assert!(write_result.timing.metadata_array_build_seconds > 0.0);
    assert!(write_result.timing.arrow_batch_write_seconds > 0.0);
    assert!(write_result.timing.arrow_file_bytes > 0);
    assert!(write_result.timing.arrow_array_memory_bytes > 0);
    let chunk_file_path = chunks_directory.join("chunk_000000000_000000001.arrow");
    let input_file = File::open(chunk_file_path).expect("chunk file should open");
    let file_reader = ArrowFileReader::try_new(input_file, None).expect("Arrow reader should open");
    let chunk_metadata = file_reader
        .schema()
        .metadata()
        .get(schema::CHUNK_COMMITS_METADATA_KEY)
        .expect("chunk metadata should exist")
        .clone();
    assert!(chunk_metadata.contains("\"chunk_identifier\":0"));
    assert!(file_reader.schema().field_with_name("chunk_identifier").is_err());
    let batches = file_reader.collect::<Result<Vec<_>, _>>().expect("batches should read");
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batches[1].num_rows(), 1);

    std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
}

#[test]
fn grouped_parquet_part_writes_expected_schema_and_footer_commits() {
    let run_directory = create_test_directory();
    let parts_directory = run_directory.join("parts");
    std::fs::create_dir_all(&parts_directory).expect("parts directory should be created");
    let write_batch = RegenieStep2ChunkWriteBatch {
        chunk_file_name: build_part_file_name(0, 1),
        chunks: vec![build_test_chunk(0, Some(vec![1])), build_test_chunk(1, Some(vec![2]))],
    };
    let write_result = write_regenie_step2_chunk_job(
        &parts_directory,
        write_batch,
        OutputFileFormat::Parquet,
        OutputStatisticDtype::Float32,
        "none",
        "none",
    )
    .expect("Parquet part should write");

    assert_eq!(write_result.chunk_commits.len(), 2);
    assert_eq!(write_result.chunk_commits[0].output_format, "parquet");
    assert_eq!(write_result.chunk_commits[0].compression, "none");
    let part_file_path = parts_directory.join("part_000000000_000000001.parquet");
    assert!(part_file_path.exists());
    assert!(!parts_directory.join("part_000000000_000000001.parquet.tmp").exists());

    let parquet_schema =
        ParquetRecordBatchReaderBuilder::try_new(File::open(&part_file_path).expect("Parquet part should open"))
            .expect("Parquet reader should build")
            .schema()
            .clone();
    assert_eq!(parquet_schema.fields().len(), 16);
    assert!(parquet_schema.field_with_name("chunk_identifier").is_err());
    assert_eq!(
        parquet_schema.field_with_name("CORRECTION_METHOD").expect("CORRECTION_METHOD field should exist").data_type(),
        &DataType::Utf8
    );
    assert_eq!(
        parquet_schema.field_with_name("CORRECTION_STATUS").expect("CORRECTION_STATUS field should exist").data_type(),
        &DataType::Utf8
    );

    let parquet_file = File::open(part_file_path).expect("Parquet part should open");
    let parquet_reader = SerializedFileReader::new(parquet_file).expect("Parquet reader should open");
    assert_eq!(parquet_reader.metadata().file_metadata().num_rows(), 2);
    let key_value_metadata =
        parquet_reader.metadata().file_metadata().key_value_metadata().expect("footer metadata should exist");
    let chunk_metadata = key_value_metadata
        .iter()
        .find(|entry| entry.key == schema::CHUNK_COMMITS_METADATA_KEY)
        .and_then(|entry| entry.value.as_deref())
        .expect("chunk commit footer metadata should exist");
    assert!(chunk_metadata.contains("\"output_format\":\"parquet\""));
    assert!(chunk_metadata.contains("\"compression\":\"none\""));

    std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
}

#[test]
fn grouped_regenie_text_part_writes_tsv_and_sidecar_metadata() {
    let run_directory = create_test_directory();
    let regenie_directory = run_directory.join("regenie");
    std::fs::create_dir_all(&regenie_directory).expect("regenie directory should be created");
    let write_batch = RegenieStep2ChunkWriteBatch {
        chunk_file_name: build_regenie_text_part_file_name(0, 1),
        chunks: vec![build_test_chunk(0, Some(vec![3])), build_test_chunk(1, Some(vec![1]))],
    };
    let write_result = write_regenie_step2_chunk_job(
        &regenie_directory,
        write_batch,
        OutputFileFormat::Regenie,
        OutputStatisticDtype::Float32,
        "none",
        "none",
    )
    .expect("REGENIE text part should write");

    assert_eq!(write_result.chunk_commits.len(), 2);
    assert_eq!(write_result.chunk_commits[0].output_format, "regenie");
    assert_eq!(write_result.chunk_commits[0].compression, "none");
    let part_file_path = regenie_directory.join("part_000000000_000000001.regenie");
    assert!(part_file_path.exists());
    assert!(!regenie_directory.join("part_000000000_000000001.regenie.tmp").exists());

    let part_lines = std::fs::read_to_string(&part_file_path)
        .expect("REGENIE text part should be readable")
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    assert_eq!(
        part_lines[0],
        "CHROM\tGENPOS\tID\tALLELE0\tALLELE1\tA1FREQ\tINFO\tN\tTEST\tBETA\tSE\tCHISQ\tLOG10P\tEXTRA\tCORRECTION_METHOD\tCORRECTION_STATUS"
    );
    assert_eq!(
        part_lines[1],
        "22\t100\tvariant0\tG\tA\t0.5\t0.9\t100\tADD\t0.1\t0.01\t10\t5\tTEST_FAIL\tfirth_approximate\tfailed"
    );
    assert_eq!(
        part_lines[2],
        "22\t101\tvariant1\tG\tA\t0.5\t0.9\t100\tADD\t0.1\t0.01\t10\t5\tNA\tfirth_approximate\tsuccess"
    );

    let sidecar_text = std::fs::read_to_string(build_regenie_text_metadata_sidecar_path(&part_file_path))
        .expect("REGENIE text sidecar should be readable");
    assert!(sidecar_text.contains("\"output_format\":\"regenie\""));
    assert!(sidecar_text.contains("\"compression\":\"none\""));

    std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
}

#[test]
fn finalization_writes_footer_metadata() {
    let run_directory = create_test_directory();
    let chunks_directory = run_directory.join("chunks");
    std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
    let write_result = write_regenie_step2_chunk_job(
        &chunks_directory,
        build_test_batch(vec![build_test_chunk(0, Some(vec![1])), build_test_chunk(1, Some(vec![0]))]),
        OutputFileFormat::Arrow,
        OutputStatisticDtype::Float32,
        "zstd",
        "none",
    )
    .expect("chunk batch should write");
    std::fs::write(run_directory.join("run_manifest.json"), r#"{"committed_chunks":[]}"#)
        .expect("manifest should be written");
    manifest::record_run_manifest_chunk_commits(&run_directory, write_result.chunk_commits)
        .expect("manifest commits should record");

    let final_parquet_path = run_directory.join("final.parquet");
    finalization::write_final_parquet_from_chunk_files(
        &chunks_directory,
        &final_parquet_path,
        "regenie2_binary",
        OutputFileFormat::Arrow,
    )
    .expect("final parquet should write");

    let parquet_file = File::open(final_parquet_path).expect("final parquet should open");
    let parquet_reader = SerializedFileReader::new(parquet_file).expect("parquet reader should open");
    let key_value_metadata =
        parquet_reader.metadata().file_metadata().key_value_metadata().expect("footer metadata should exist");
    let metadata_value =
        |key: &str| key_value_metadata.iter().find(|entry| entry.key == key).and_then(|entry| entry.value.as_deref());
    assert_eq!(metadata_value("g.output.schema_version"), Some(schema::OUTPUT_SCHEMA_VERSION));
    assert_eq!(metadata_value("g.output.association_mode"), Some("regenie2_binary"));
    assert_eq!(metadata_value("g.output.chunk_file_count"), Some("1"));
    assert_eq!(metadata_value("g.output.row_count"), Some("2"));
    assert_eq!(metadata_value("g.output.writer"), Some("rust"));

    std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
}
