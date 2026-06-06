#![allow(clippy::needless_pass_by_value)]

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::Schema;
use arrow::ipc::CompressionType;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use serde_json::json;
use thiserror::Error;

use crate::output::NativeChunkHandle;
use crate::output::manifest;
use crate::output::schema;

const REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE: usize = 122_880;

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
    pub(crate) beta: ArrayRef,
    pub(crate) se: ArrayRef,
    pub(crate) chisq: ArrayRef,
    pub(crate) log10p: ArrayRef,
    pub(crate) extra_code: Option<ArrayRef>,
}

pub(crate) struct RegenieStep2ChunkWriteBatch {
    pub(crate) chunk_file_name: String,
    pub(crate) chunks: Vec<RegenieStep2ChunkJob>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum OutputFileFormat {
    Arrow,
    Parquet,
}

impl OutputFileFormat {
    pub(crate) fn parse(output_format: &str) -> Result<Self, String> {
        match output_format {
            "arrow" => Ok(Self::Arrow),
            "parquet" => Ok(Self::Parquet),
            unsupported_output_format => {
                Err(format!("Output format must be 'arrow' or 'parquet', observed '{unsupported_output_format}'."))
            }
        }
    }

    fn value(self) -> &'static str {
        match self {
            Self::Arrow => "arrow",
            Self::Parquet => "parquet",
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct RegenieStep2RecordBatchBuildTiming {
    pub(crate) schema_metadata_build_seconds: f64,
    pub(crate) metadata_array_build_seconds: f64,
    pub(crate) statistic_array_build_seconds: f64,
    pub(crate) test_array_build_seconds: f64,
    pub(crate) result_array_build_seconds: f64,
    pub(crate) extra_array_build_seconds: f64,
    pub(crate) record_batch_try_new_seconds: f64,
    pub(crate) arrow_array_memory_bytes: u64,
}

impl RegenieStep2RecordBatchBuildTiming {
    fn add(&mut self, timing: Self) {
        self.schema_metadata_build_seconds += timing.schema_metadata_build_seconds;
        self.metadata_array_build_seconds += timing.metadata_array_build_seconds;
        self.statistic_array_build_seconds += timing.statistic_array_build_seconds;
        self.test_array_build_seconds += timing.test_array_build_seconds;
        self.result_array_build_seconds += timing.result_array_build_seconds;
        self.extra_array_build_seconds += timing.extra_array_build_seconds;
        self.record_batch_try_new_seconds += timing.record_batch_try_new_seconds;
        self.arrow_array_memory_bytes = self.arrow_array_memory_bytes.saturating_add(timing.arrow_array_memory_bytes);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RegenieStep2ChunkWriteTiming {
    pub(crate) chunk_file_count: u64,
    pub(crate) chunk_count: u64,
    pub(crate) row_count: u64,
    pub(crate) record_batch_build_seconds: f64,
    pub(crate) schema_metadata_build_seconds: f64,
    pub(crate) metadata_array_build_seconds: f64,
    pub(crate) statistic_array_build_seconds: f64,
    pub(crate) test_array_build_seconds: f64,
    pub(crate) result_array_build_seconds: f64,
    pub(crate) extra_array_build_seconds: f64,
    pub(crate) record_batch_try_new_seconds: f64,
    pub(crate) arrow_file_write_seconds: f64,
    pub(crate) arrow_file_create_seconds: f64,
    pub(crate) arrow_writer_init_seconds: f64,
    pub(crate) arrow_batch_write_seconds: f64,
    pub(crate) arrow_writer_finish_seconds: f64,
    pub(crate) arrow_file_rename_seconds: f64,
    pub(crate) arrow_array_memory_bytes: u64,
    pub(crate) arrow_file_bytes: u64,
    pub(crate) total_seconds: f64,
}

pub(crate) struct RegenieStep2ChunkWriteResult {
    pub(crate) chunk_commits: Vec<manifest::RunManifestChunkCommit>,
    pub(crate) timing: RegenieStep2ChunkWriteTiming,
}

pub(crate) fn write_regenie_step2_chunk_job(
    chunks_directory: &Path,
    job: RegenieStep2ChunkWriteBatch,
    output_format: OutputFileFormat,
    arrow_compression: &str,
    parquet_compression: &str,
) -> Result<RegenieStep2ChunkWriteResult, String> {
    let total_start_time = Instant::now();
    let chunk_file_path = chunks_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = match output_format {
        OutputFileFormat::Arrow => chunk_file_path.with_extension("arrow.tmp"),
        OutputFileFormat::Parquet => chunk_file_path.with_extension("parquet.tmp"),
    };
    let chunk_count = u64::try_from(job.chunks.len()).map_err(|error| error.to_string())?;
    let row_count = job
        .chunks
        .iter()
        .map(|chunk_job| u64::try_from(chunk_job.chunk_handle.row_count()).map_err(|error| error.to_string()))
        .sum::<Result<u64, String>>()?;
    let compression = match output_format {
        OutputFileFormat::Arrow => arrow_compression,
        OutputFileFormat::Parquet => parquet_compression,
    };
    let chunk_commits = build_run_manifest_chunk_commits(&job, output_format, compression)?;

    let record_batch_build_start_time = Instant::now();
    let schema_metadata_build_start_time = Instant::now();
    let chunk_schema = build_regenie_step2_chunk_file_schema(&chunk_commits)?;
    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming {
        schema_metadata_build_seconds: schema_metadata_build_start_time.elapsed().as_secs_f64(),
        ..RegenieStep2RecordBatchBuildTiming::default()
    };
    let record_batch_build_result = build_regenie_step2_record_batches(job, Arc::clone(&chunk_schema))?;
    record_batch_build_timing.add(record_batch_build_result.timing);
    let record_batch_build_seconds = record_batch_build_start_time.elapsed().as_secs_f64();

    let arrow_file_write_start_time = Instant::now();
    let arrow_file_write_timing = match output_format {
        OutputFileFormat::Arrow => write_record_batches_to_arrow_file(
            &record_batch_build_result.record_batches,
            chunk_schema,
            &temporary_chunk_file_path,
            arrow_compression,
        )?,
        OutputFileFormat::Parquet => write_record_batches_to_parquet_file(
            &record_batch_build_result.record_batches,
            chunk_schema,
            &temporary_chunk_file_path,
            parquet_compression,
            &chunk_commits,
        )?,
    };
    let arrow_file_rename_start_time = Instant::now();
    std::fs::rename(&temporary_chunk_file_path, &chunk_file_path).map_err(|error| error.to_string())?;
    let arrow_file_rename_seconds = arrow_file_rename_start_time.elapsed().as_secs_f64();
    let arrow_file_bytes = std::fs::metadata(&chunk_file_path).map_err(|error| error.to_string())?.len();
    let arrow_file_write_seconds = arrow_file_write_start_time.elapsed().as_secs_f64();

    Ok(RegenieStep2ChunkWriteResult {
        chunk_commits,
        timing: RegenieStep2ChunkWriteTiming {
            chunk_file_count: 1,
            chunk_count,
            row_count,
            record_batch_build_seconds,
            schema_metadata_build_seconds: record_batch_build_timing.schema_metadata_build_seconds,
            metadata_array_build_seconds: record_batch_build_timing.metadata_array_build_seconds,
            statistic_array_build_seconds: record_batch_build_timing.statistic_array_build_seconds,
            test_array_build_seconds: record_batch_build_timing.test_array_build_seconds,
            result_array_build_seconds: record_batch_build_timing.result_array_build_seconds,
            extra_array_build_seconds: record_batch_build_timing.extra_array_build_seconds,
            record_batch_try_new_seconds: record_batch_build_timing.record_batch_try_new_seconds,
            arrow_file_write_seconds,
            arrow_file_create_seconds: arrow_file_write_timing.file_create,
            arrow_writer_init_seconds: arrow_file_write_timing.writer_init,
            arrow_batch_write_seconds: arrow_file_write_timing.batch_write,
            arrow_writer_finish_seconds: arrow_file_write_timing.writer_finish,
            arrow_file_rename_seconds,
            arrow_array_memory_bytes: record_batch_build_timing.arrow_array_memory_bytes,
            arrow_file_bytes,
            total_seconds: total_start_time.elapsed().as_secs_f64(),
        },
    })
}

fn build_run_manifest_chunk_commits(
    job: &RegenieStep2ChunkWriteBatch,
    output_format: OutputFileFormat,
    compression: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, String> {
    job.chunks
        .iter()
        .map(|chunk_job| {
            let variant_stop_index = chunk_job.chunk_handle.variant_stop_index().map_err(|error| error.to_string())?;
            Ok(manifest::RunManifestChunkCommit {
                chunk_identifier: chunk_job.chunk_handle.chunk_identifier,
                output_format: output_format.value().to_string(),
                compression: compression.to_string(),
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
    let mut metadata = HashMap::new();
    metadata.insert(schema::CHUNK_COMMITS_METADATA_KEY.to_string(), build_chunk_commit_metadata_text(chunk_commits)?);
    Ok(Arc::new(schema::get_regenie_step2_chunk_schema().as_ref().clone().with_metadata(metadata)))
}

fn build_chunk_commit_metadata_text(chunk_commits: &[manifest::RunManifestChunkCommit]) -> Result<String, String> {
    let chunk_commit_values = chunk_commits
        .iter()
        .map(|chunk_commit| {
            json!({
                "chunk_identifier": chunk_commit.chunk_identifier,
                "output_format": chunk_commit.output_format,
                "compression": chunk_commit.compression,
                "variant_start_index": chunk_commit.variant_start_index,
                "variant_stop_index": chunk_commit.variant_stop_index,
                "row_count": chunk_commit.row_count,
                "chunk_file_name": chunk_commit.chunk_file_name,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&chunk_commit_values).map_err(|error| error.to_string())
}

pub(crate) fn build_chunk_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("chunk_{first_chunk_identifier:09}.arrow");
    }
    format!("chunk_{first_chunk_identifier:09}_{last_chunk_identifier:09}.arrow")
}

pub(crate) fn build_part_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("part_{first_chunk_identifier:09}.parquet");
    }
    format!("part_{first_chunk_identifier:09}_{last_chunk_identifier:09}.parquet")
}

pub(crate) fn build_output_file_name(
    output_format: OutputFileFormat,
    first_chunk_identifier: i64,
    last_chunk_identifier: i64,
) -> String {
    match output_format {
        OutputFileFormat::Arrow => build_chunk_file_name(first_chunk_identifier, last_chunk_identifier),
        OutputFileFormat::Parquet => build_part_file_name(first_chunk_identifier, last_chunk_identifier),
    }
}

fn build_regenie_step2_record_batches(
    job: RegenieStep2ChunkWriteBatch,
    chunk_schema: Arc<Schema>,
) -> Result<RegenieStep2RecordBatchBuildResult, String> {
    let mut timing = RegenieStep2RecordBatchBuildTiming::default();
    let mut array_cache = RegenieStep2RecordBatchArrayCache::default();
    let record_batches = job
        .chunks
        .into_iter()
        .map(|chunk_job| {
            let record_batch_build_result =
                build_regenie_step2_record_batch(chunk_job, Arc::clone(&chunk_schema), &mut array_cache)?;
            timing.add(record_batch_build_result.timing);
            Ok(record_batch_build_result.record_batch)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(RegenieStep2RecordBatchBuildResult { record_batches, timing })
}

struct RegenieStep2RecordBatchBuildResult {
    record_batches: Vec<RecordBatch>,
    timing: RegenieStep2RecordBatchBuildTiming,
}

struct RegenieStep2SingleRecordBatchBuildResult {
    record_batch: RecordBatch,
    timing: RegenieStep2RecordBatchBuildTiming,
}

#[derive(Default)]
struct RegenieStep2RecordBatchArrayCache {
    null_extra_arrays_by_row_count: HashMap<usize, ArrayRef>,
    test_arrays_by_row_count: HashMap<usize, ArrayRef>,
}

impl RegenieStep2RecordBatchArrayCache {
    fn null_extra_array(&mut self, row_count: usize) -> ArrayRef {
        Arc::clone(
            self.null_extra_arrays_by_row_count
                .entry(row_count)
                .or_insert_with(|| schema::build_null_extra_string_array(row_count)),
        )
    }

    fn test_array(&mut self, row_count: usize) -> ArrayRef {
        Arc::clone(
            self.test_arrays_by_row_count
                .entry(row_count)
                .or_insert_with(|| Arc::new(StringArray::from(vec!["ADD"; row_count]))),
        )
    }
}

fn build_regenie_step2_record_batch(
    chunk_job: RegenieStep2ChunkJob,
    chunk_schema: Arc<Schema>,
    array_cache: &mut RegenieStep2RecordBatchArrayCache,
) -> Result<RegenieStep2SingleRecordBatchBuildResult, String> {
    let row_count = chunk_job.chunk_handle.row_count();
    let metadata_array_build_start_time = Instant::now();
    let chromosome_array: ArrayRef = Arc::new(StringArray::from(chunk_job.chunk_handle.metadata.chromosome.clone()));
    let position_array: ArrayRef = Arc::new(Int64Array::from(chunk_job.chunk_handle.metadata.position.clone()));
    let variant_identifier_array: ArrayRef =
        Arc::new(StringArray::from(chunk_job.chunk_handle.metadata.variant_identifier.clone()));
    let allele_two_array: ArrayRef = Arc::new(StringArray::from(chunk_job.chunk_handle.metadata.allele_two.clone()));
    let allele_one_array: ArrayRef = Arc::new(StringArray::from(chunk_job.chunk_handle.metadata.allele_one.clone()));
    let metadata_array_build_seconds = metadata_array_build_start_time.elapsed().as_secs_f64();

    let statistic_array_build_start_time = Instant::now();
    let allele_one_frequency_array: ArrayRef =
        Arc::new(Float32Array::from(chunk_job.chunk_handle.stats.allele_one_frequency.clone()));
    let info_score_array: ArrayRef = Arc::new(Float32Array::from(chunk_job.chunk_handle.stats.info_score.clone()));
    let observation_count_array: ArrayRef =
        Arc::new(Int32Array::from(chunk_job.chunk_handle.stats.observation_count.clone()));
    let statistic_array_build_seconds = statistic_array_build_start_time.elapsed().as_secs_f64();

    let test_array_build_start_time = Instant::now();
    let test_array = array_cache.test_array(row_count);
    let test_array_build_seconds = test_array_build_start_time.elapsed().as_secs_f64();

    let result_array_build_start_time = Instant::now();
    let beta_array = chunk_job.beta;
    let standard_error_array = chunk_job.se;
    let chi_squared_array = chunk_job.chisq;
    let log10_p_value_array = chunk_job.log10p;
    let result_array_build_seconds = result_array_build_start_time.elapsed().as_secs_f64();

    let extra_array_build_start_time = Instant::now();
    let extra_array = match chunk_job.extra_code {
        Some(extra_code) => schema::build_extra_string_array(Some(extra_code), row_count)?,
        None => array_cache.null_extra_array(row_count),
    };
    let extra_array_build_seconds = extra_array_build_start_time.elapsed().as_secs_f64();

    let columns: Vec<ArrayRef> = vec![
        chromosome_array,
        position_array,
        variant_identifier_array,
        allele_two_array,
        allele_one_array,
        allele_one_frequency_array,
        info_score_array,
        observation_count_array,
        test_array,
        beta_array,
        standard_error_array,
        chi_squared_array,
        log10_p_value_array,
        extra_array,
    ];
    let arrow_array_memory_bytes = columns.iter().fold(0_u64, |total, column| {
        total.saturating_add(u64::try_from(column.get_array_memory_size()).unwrap_or(u64::MAX))
    });
    let record_batch_try_new_start_time = Instant::now();
    let record_batch = RecordBatch::try_new(chunk_schema, columns).map_err(|error| error.to_string())?;
    let record_batch_try_new_seconds = record_batch_try_new_start_time.elapsed().as_secs_f64();
    Ok(RegenieStep2SingleRecordBatchBuildResult {
        record_batch,
        timing: RegenieStep2RecordBatchBuildTiming {
            schema_metadata_build_seconds: 0.0,
            metadata_array_build_seconds,
            statistic_array_build_seconds,
            test_array_build_seconds,
            result_array_build_seconds,
            extra_array_build_seconds,
            record_batch_try_new_seconds,
            arrow_array_memory_bytes,
        },
    })
}

struct RegenieStep2ArrowFileWriteTiming {
    file_create: f64,
    writer_init: f64,
    batch_write: f64,
    writer_finish: f64,
}

fn write_record_batches_to_arrow_file(
    record_batches: &[RecordBatch],
    chunk_schema: Arc<Schema>,
    chunk_file_path: &Path,
    arrow_compression: &str,
) -> Result<RegenieStep2ArrowFileWriteTiming, String> {
    let file_create_start_time = Instant::now();
    let output_file = File::create(chunk_file_path).map_err(|error| error.to_string())?;
    let file_create = file_create_start_time.elapsed().as_secs_f64();

    let writer_init_start_time = Instant::now();
    let write_options = build_regenie_step2_ipc_write_options(arrow_compression)?;
    let mut writer = FileWriter::try_new_with_options(output_file, &chunk_schema, write_options)
        .map_err(|error| error.to_string())?;
    let writer_init = writer_init_start_time.elapsed().as_secs_f64();

    let mut batch_write = 0.0;
    for record_batch in record_batches {
        let batch_write_start_time = Instant::now();
        writer.write(record_batch).map_err(|error| error.to_string())?;
        batch_write += batch_write_start_time.elapsed().as_secs_f64();
    }
    let writer_finish_start_time = Instant::now();
    writer.finish().map_err(|error| error.to_string())?;
    let writer_finish = writer_finish_start_time.elapsed().as_secs_f64();
    Ok(RegenieStep2ArrowFileWriteTiming { file_create, writer_init, batch_write, writer_finish })
}

fn write_record_batches_to_parquet_file(
    record_batches: &[RecordBatch],
    chunk_schema: Arc<Schema>,
    chunk_file_path: &Path,
    parquet_compression: &str,
    chunk_commits: &[manifest::RunManifestChunkCommit],
) -> Result<RegenieStep2ArrowFileWriteTiming, String> {
    let file_create_start_time = Instant::now();
    let output_file = File::create(chunk_file_path).map_err(|error| error.to_string())?;
    let file_create = file_create_start_time.elapsed().as_secs_f64();

    let writer_init_start_time = Instant::now();
    let writer_properties = build_regenie_step2_parquet_writer_properties(parquet_compression)?;
    let mut writer =
        ArrowWriter::try_new(output_file, chunk_schema, Some(writer_properties)).map_err(|error| error.to_string())?;
    writer.append_key_value_metadata(KeyValue {
        key: schema::CHUNK_COMMITS_METADATA_KEY.to_string(),
        value: Some(build_chunk_commit_metadata_text(chunk_commits)?),
    });
    let writer_init = writer_init_start_time.elapsed().as_secs_f64();

    let mut batch_write = 0.0;
    for record_batch in record_batches {
        let batch_write_start_time = Instant::now();
        writer.write(record_batch).map_err(|error| error.to_string())?;
        batch_write += batch_write_start_time.elapsed().as_secs_f64();
    }
    let writer_finish_start_time = Instant::now();
    writer.close().map_err(|error| error.to_string())?;
    let writer_finish = writer_finish_start_time.elapsed().as_secs_f64();
    Ok(RegenieStep2ArrowFileWriteTiming { file_create, writer_init, batch_write, writer_finish })
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

fn build_regenie_step2_parquet_writer_properties(parquet_compression: &str) -> Result<WriterProperties, String> {
    let compression = match parquet_compression.to_ascii_lowercase().as_str() {
        "zstd" => Compression::ZSTD(ZstdLevel::default()),
        "none" => Compression::UNCOMPRESSED,
        unsupported_compression => {
            return Err(format!("Parquet compression must be 'zstd' or 'none', observed '{unsupported_compression}'."));
        }
    };
    Ok(WriterProperties::builder()
        .set_compression(compression)
        .set_max_row_group_row_count(Some(REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE))
        .set_dictionary_enabled(false)
        .set_column_dictionary_enabled(ColumnPath::from("CHROM"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE0"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE1"), true)
        .set_column_dictionary_enabled(ColumnPath::from("N"), true)
        .set_column_dictionary_enabled(ColumnPath::from("TEST"), true)
        .set_column_dictionary_enabled(ColumnPath::from("EXTRA"), true)
        .build())
}

#[cfg(test)]
mod tests {
    use std::assert_matches;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow::array::Array;
    use arrow::ipc::reader::FileReader as ArrowFileReader;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
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
            beta: Arc::new(Float32Array::from(vec![0.1])),
            se: Arc::new(Float32Array::from(vec![0.01])),
            chisq: Arc::new(Float32Array::from(vec![10.0])),
            log10p: Arc::new(Float32Array::from(vec![5.0])),
            extra_code: extra_code.map(|values| Arc::new(Int32Array::from(values)) as ArrayRef),
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

    fn build_test_record_batches(chunks: Vec<RegenieStep2ChunkJob>) -> Vec<RecordBatch> {
        let write_batch = build_test_batch(chunks);
        let chunk_commits = build_run_manifest_chunk_commits(&write_batch, OutputFileFormat::Arrow, "none")
            .expect("chunk commits should build");
        let chunk_schema = build_regenie_step2_chunk_file_schema(&chunk_commits).expect("chunk schema should build");
        build_regenie_step2_record_batches(write_batch, chunk_schema)
            .expect("record batches should build")
            .record_batches
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
    fn binary_record_batch_maps_extra_codes_with_same_schema() {
        let linear_record_batch = build_test_record_batch(build_test_chunk(0, None));
        let binary_record_batch = build_test_record_batch(build_test_chunk(1, Some(vec![1])));

        assert_eq!(linear_record_batch.schema().fields(), binary_record_batch.schema().fields());
        assert_eq!(binary_record_batch.column_by_name("EXTRA").expect("EXTRA column should exist").null_count(), 1);
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
        let write_result =
            write_regenie_step2_chunk_job(&parts_directory, write_batch, OutputFileFormat::Parquet, "none", "none")
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
        assert_eq!(parquet_schema.fields().len(), 14);
        assert!(parquet_schema.field_with_name("chunk_identifier").is_err());

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
    fn finalization_writes_footer_metadata() {
        let run_directory = create_test_directory();
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
        write_regenie_step2_chunk_job(
            &chunks_directory,
            build_test_batch(vec![build_test_chunk(0, Some(vec![1])), build_test_chunk(1, Some(vec![0]))]),
            OutputFileFormat::Arrow,
            "zstd",
            "none",
        )
        .expect("chunk batch should write");

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
