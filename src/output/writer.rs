#![allow(clippy::needless_pass_by_value)]

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::Schema;
use arrow::ipc::CompressionType;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use serde_json::json;
use thiserror::Error;

use crate::output::NativeChunkHandle;
use crate::output::manifest;
use crate::output::schema;

#[derive(Debug, Error)]
pub enum OutputWriterError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Runtime(String),
}

impl OutputWriterError {
    pub(crate) fn runtime(error: impl ToString) -> Self {
        Self::Runtime(error.to_string())
    }
}

pub(crate) struct RegenieStep2ChunkJob {
    pub(crate) chunk_handle: NativeChunkHandle,
    pub(crate) beta: Vec<f32>,
    pub(crate) se: Vec<f32>,
    pub(crate) chisq: Vec<f32>,
    pub(crate) log10p: Vec<f32>,
    pub(crate) extra_code: Option<Vec<i32>>,
}

pub(crate) struct RegenieStep2ChunkWriteBatch {
    pub(crate) chunk_file_name: String,
    pub(crate) chunks: Vec<RegenieStep2ChunkJob>,
}

#[derive(Clone, Copy)]
pub(crate) struct RegenieStep2ChunkWriteTiming {
    pub(crate) chunk_file_count: u64,
    pub(crate) chunk_count: u64,
    pub(crate) row_count: u64,
    pub(crate) record_batch_build_seconds: f64,
    pub(crate) arrow_file_write_seconds: f64,
    pub(crate) total_seconds: f64,
}

pub(crate) struct RegenieStep2ChunkWriteResult {
    pub(crate) chunk_commits: Vec<manifest::RunManifestChunkCommit>,
    pub(crate) timing: RegenieStep2ChunkWriteTiming,
}

pub(crate) fn write_regenie_step2_chunk_job(
    chunks_directory: &Path,
    job: RegenieStep2ChunkWriteBatch,
    arrow_compression: &str,
) -> Result<RegenieStep2ChunkWriteResult, String> {
    let total_start_time = Instant::now();
    let chunk_file_path = chunks_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = chunk_file_path.with_extension("arrow.tmp");
    let chunk_count = u64::try_from(job.chunks.len()).map_err(|error| error.to_string())?;
    let row_count = job
        .chunks
        .iter()
        .map(|chunk_job| u64::try_from(chunk_job.chunk_handle.row_count()).map_err(|error| error.to_string()))
        .sum::<Result<u64, String>>()?;
    let chunk_commits = build_run_manifest_chunk_commits(&job)?;

    let record_batch_build_start_time = Instant::now();
    let chunk_schema = build_regenie_step2_chunk_file_schema(&chunk_commits)?;
    let record_batches = build_regenie_step2_record_batches(job, Arc::clone(&chunk_schema))?;
    let record_batch_build_seconds = record_batch_build_start_time.elapsed().as_secs_f64();

    let arrow_file_write_start_time = Instant::now();
    write_record_batches_to_arrow_file(&record_batches, chunk_schema, &temporary_chunk_file_path, arrow_compression)?;
    std::fs::rename(&temporary_chunk_file_path, &chunk_file_path).map_err(|error| error.to_string())?;
    let arrow_file_write_seconds = arrow_file_write_start_time.elapsed().as_secs_f64();

    Ok(RegenieStep2ChunkWriteResult {
        chunk_commits,
        timing: RegenieStep2ChunkWriteTiming {
            chunk_file_count: 1,
            chunk_count,
            row_count,
            record_batch_build_seconds,
            arrow_file_write_seconds,
            total_seconds: total_start_time.elapsed().as_secs_f64(),
        },
    })
}

fn build_run_manifest_chunk_commits(
    job: &RegenieStep2ChunkWriteBatch,
) -> Result<Vec<manifest::RunManifestChunkCommit>, String> {
    job.chunks
        .iter()
        .map(|chunk_job| {
            let variant_stop_index = chunk_job.chunk_handle.variant_stop_index().map_err(|error| error.to_string())?;
            Ok(manifest::RunManifestChunkCommit {
                chunk_identifier: chunk_job.chunk_handle.chunk_identifier,
                variant_start_index: chunk_job.chunk_handle.variant_start_index(),
                variant_stop_index,
                row_count: chunk_job.chunk_handle.row_count(),
                chunk_file_name: job.chunk_file_name.clone(),
            })
        })
        .collect()
}

fn build_regenie_step2_chunk_file_schema(
    chunk_commits: &[manifest::RunManifestChunkCommit],
) -> Result<Arc<Schema>, String> {
    let chunk_commit_values = chunk_commits
        .iter()
        .map(|chunk_commit| {
            json!({
                "chunk_identifier": chunk_commit.chunk_identifier,
                "variant_start_index": chunk_commit.variant_start_index,
                "variant_stop_index": chunk_commit.variant_stop_index,
                "row_count": chunk_commit.row_count,
                "chunk_file_name": chunk_commit.chunk_file_name,
            })
        })
        .collect::<Vec<_>>();
    let mut metadata = HashMap::new();
    metadata.insert(
        schema::CHUNK_COMMITS_METADATA_KEY.to_string(),
        serde_json::to_string(&chunk_commit_values).map_err(|error| error.to_string())?,
    );
    Ok(Arc::new(schema::get_regenie_step2_chunk_schema().as_ref().clone().with_metadata(metadata)))
}

pub(crate) fn build_chunk_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("chunk_{first_chunk_identifier:09}.arrow");
    }
    format!("chunk_{first_chunk_identifier:09}_{last_chunk_identifier:09}.arrow")
}

fn build_regenie_step2_record_batches(
    job: RegenieStep2ChunkWriteBatch,
    chunk_schema: Arc<Schema>,
) -> Result<Vec<RecordBatch>, String> {
    job.chunks
        .into_iter()
        .map(|chunk_job| build_regenie_step2_record_batch(chunk_job, Arc::clone(&chunk_schema)))
        .collect()
}

fn build_regenie_step2_record_batch(
    chunk_job: RegenieStep2ChunkJob,
    chunk_schema: Arc<Schema>,
) -> Result<RecordBatch, String> {
    let row_count = chunk_job.chunk_handle.row_count();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(schema::build_dictionary_string_array(&chunk_job.chunk_handle.metadata.chromosome)?),
        Arc::new(Int64Array::from(chunk_job.chunk_handle.metadata.position.clone())),
        Arc::new(StringArray::from(chunk_job.chunk_handle.metadata.variant_identifier.clone())),
        Arc::new(schema::build_dictionary_string_array(&chunk_job.chunk_handle.metadata.allele_two)?),
        Arc::new(schema::build_dictionary_string_array(&chunk_job.chunk_handle.metadata.allele_one)?),
        Arc::new(Float32Array::from(chunk_job.chunk_handle.stats.allele_one_frequency.clone())),
        Arc::new(Float32Array::from(chunk_job.chunk_handle.stats.info_score.clone())),
        Arc::new(Int32Array::from(chunk_job.chunk_handle.stats.observation_count.clone())),
        Arc::new(schema::build_constant_dictionary_string_array(row_count, "ADD")?),
        Arc::new(Float32Array::from(chunk_job.beta)),
        Arc::new(Float32Array::from(chunk_job.se)),
        Arc::new(Float32Array::from(chunk_job.chisq)),
        Arc::new(Float32Array::from(chunk_job.log10p)),
        schema::build_extra_string_array(chunk_job.extra_code, row_count)?,
    ];
    RecordBatch::try_new(chunk_schema, columns).map_err(|error| error.to_string())
}

fn write_record_batches_to_arrow_file(
    record_batches: &[RecordBatch],
    chunk_schema: Arc<Schema>,
    chunk_file_path: &Path,
    arrow_compression: &str,
) -> Result<(), String> {
    let output_file = File::create(chunk_file_path).map_err(|error| error.to_string())?;
    let write_options = build_regenie_step2_ipc_write_options(arrow_compression)?;
    let mut writer = FileWriter::try_new_with_options(output_file, &chunk_schema, write_options)
        .map_err(|error| error.to_string())?;
    for record_batch in record_batches {
        writer.write(record_batch).map_err(|error| error.to_string())?;
    }
    writer.finish().map_err(|error| error.to_string())
}

fn build_regenie_step2_ipc_write_options(arrow_compression: &str) -> Result<IpcWriteOptions, String> {
    match arrow_compression.to_ascii_lowercase().as_str() {
        "zstd" => IpcWriteOptions::default()
            .try_with_compression(Some(CompressionType::ZSTD))
            .map_err(|error| error.to_string()),
        "none" => Ok(IpcWriteOptions::default()),
        unsupported_compression => {
            Err(format!("Arrow compression must be 'zstd' or 'none', observed '{unsupported_compression}'."))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow::array::Array;
    use arrow::ipc::reader::FileReader as ArrowFileReader;
    use parquet::file::reader::{FileReader as ParquetFileReader, SerializedFileReader};

    use crate::genotype::common::{ChunkStats, VariantMetadataColumns};
    use crate::output::finalization;

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
                dosage_sum: vec![0.0],
                dosage_square_sum: vec![0.0],
                imputed_dosage_square_sum: vec![0.0],
                dosage_variance_numerator: vec![0.0],
                info_score: vec![Some(0.9)],
                allele_count: vec![0.0],
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

    fn build_test_chunk(chunk_identifier: i64, extra_code: Option<Vec<i32>>) -> RegenieStep2ChunkJob {
        RegenieStep2ChunkJob {
            chunk_handle: build_test_chunk_handle(chunk_identifier),
            beta: vec![0.1],
            se: vec![0.01],
            chisq: vec![10.0],
            log10p: vec![5.0],
            extra_code,
        }
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

    fn build_test_record_batch(chunk: RegenieStep2ChunkJob) -> RecordBatch {
        let write_batch = build_test_batch(vec![chunk]);
        let chunk_commits = build_run_manifest_chunk_commits(&write_batch).expect("chunk commits should build");
        let chunk_schema = build_regenie_step2_chunk_file_schema(&chunk_commits).expect("chunk schema should build");
        build_regenie_step2_record_batches(write_batch, chunk_schema)
            .expect("record batches should build")
            .pop()
            .expect("one record batch should be present")
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

        assert!(matches!(error, OutputWriterError::Runtime(message) if message == "worker failed"));
    }

    #[test]
    fn linear_record_batch_uses_shared_schema_and_null_extra() {
        let record_batch = build_test_record_batch(build_test_chunk(0, None));

        assert_eq!(record_batch.schema().fields().len(), 14);
        assert!(record_batch.schema().field_with_name("chunk_identifier").is_err());
        assert!(record_batch.schema().field_with_name("INFO").expect("INFO field should exist").is_nullable());
        assert!(record_batch.schema().field_with_name("EXTRA").expect("EXTRA field should exist").is_nullable());
        assert_eq!(record_batch.num_rows(), 1);
        let info_array = record_batch
            .column_by_name("INFO")
            .expect("INFO column should exist")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("INFO column should be a float32 array");
        assert!((info_array.value(0) - 0.9).abs() < f32::EPSILON);
        assert_eq!(record_batch.column_by_name("EXTRA").expect("EXTRA column should exist").null_count(), 1);
    }

    #[test]
    fn binary_record_batch_maps_extra_codes_with_same_schema() {
        let linear_record_batch = build_test_record_batch(build_test_chunk(0, None));
        let binary_record_batch = build_test_record_batch(build_test_chunk(1, Some(vec![1])));

        assert_eq!(linear_record_batch.schema().fields(), binary_record_batch.schema().fields());
        let extra_array = binary_record_batch
            .column_by_name("EXTRA")
            .expect("EXTRA column should exist")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("EXTRA column should be a string array");
        assert_eq!(extra_array.value(0), "FIRTH");
    }

    #[test]
    fn grouped_arrow_file_writes_one_batch_per_compute_chunk_with_metadata_commits() {
        let run_directory = create_test_directory();
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
        let write_result = write_regenie_step2_chunk_job(
            &chunks_directory,
            build_test_batch(vec![build_test_chunk(0, Some(vec![1])), build_test_chunk(1, Some(vec![2]))]),
            "none",
        )
        .expect("chunk batch should write");

        assert_eq!(write_result.chunk_commits.len(), 2);
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
    fn finalization_writes_footer_metadata() {
        let run_directory = create_test_directory();
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
        write_regenie_step2_chunk_job(
            &chunks_directory,
            build_test_batch(vec![build_test_chunk(0, Some(vec![1])), build_test_chunk(1, Some(vec![0]))]),
            "zstd",
        )
        .expect("chunk batch should write");

        let final_parquet_path = run_directory.join("final.parquet");
        finalization::write_final_parquet_from_chunk_files(&chunks_directory, &final_parquet_path, "regenie2_binary")
            .expect("final parquet should write");

        let parquet_file = File::open(final_parquet_path).expect("final parquet should open");
        let parquet_reader = SerializedFileReader::new(parquet_file).expect("parquet reader should open");
        let key_value_metadata =
            parquet_reader.metadata().file_metadata().key_value_metadata().expect("footer metadata should exist");
        let metadata_value = |key: &str| {
            key_value_metadata.iter().find(|entry| entry.key == key).and_then(|entry| entry.value.as_deref())
        };
        assert_eq!(metadata_value("g.output.schema_version"), Some("1"));
        assert_eq!(metadata_value("g.output.association_mode"), Some("regenie2_binary"));
        assert_eq!(metadata_value("g.output.chunk_file_count"), Some("1"));
        assert_eq!(metadata_value("g.output.row_count"), Some("2"));
        assert_eq!(metadata_value("g.output.writer"), Some("rust"));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }
}
