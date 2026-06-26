#![allow(clippy::needless_pass_by_value)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, ArrayRef, DictionaryArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema, UInt8Type};
use arrow::ipc::CompressionType;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use serde_json::json;

use crate::manifest;
use crate::schema;
use crate::schema::OutputStatisticDtype;

mod types;

pub use types::OutputFileFormat;
pub use types::OutputWriterError;
pub(crate) use types::{
    RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, RegenieStep2ChunkWriteResult, RegenieStep2ChunkWriteTiming,
    RegenieStep2RecordBatchBuildTiming,
};

#[cfg(test)]
mod tests;

const REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE: usize = 122_880;
pub(crate) const REGENIE_STEP2_TEXT_HEADER: &str = "CHROM\tGENPOS\tID\tALLELE0\tALLELE1\tA1FREQ\tINFO\tN\tTEST\tBETA\tSE\tCHISQ\tLOG10P\tEXTRA\tCORRECTION_METHOD\tCORRECTION_STATUS\n";
const REGENIE_STEP2_TEXT_MISSING_VALUE: &str = "NA";
const CORRECTION_METHOD_SCORE_KEY: u8 = 0;
const CORRECTION_METHOD_FIRTH_APPROXIMATE_KEY: u8 = 1;
const CORRECTION_METHOD_SPA_KEY: u8 = 2;
const CORRECTION_STATUS_SUCCESS_KEY: u8 = 0;
const CORRECTION_STATUS_FAILED_KEY: u8 = 1;

#[derive(Clone, Copy, Eq, PartialEq)]
enum RegenieStep2CorrectionArrayEncoding {
    String,
    Dictionary,
}

pub(crate) fn write_regenie_step2_chunk_job(
    chunks_directory: &Path,
    job: RegenieStep2ChunkWriteBatch,
    output_format: OutputFileFormat,
    output_statistic_dtype: OutputStatisticDtype,
    arrow_compression: &str,
    parquet_compression: &str,
) -> Result<RegenieStep2ChunkWriteResult, String> {
    let total_start_time = Instant::now();
    let chunk_file_path = chunks_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = match output_format {
        OutputFileFormat::Arrow => chunk_file_path.with_extension("arrow.tmp"),
        OutputFileFormat::Parquet => chunk_file_path.with_extension("parquet.tmp"),
        OutputFileFormat::Regenie => chunk_file_path.with_extension("regenie.tmp"),
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
        OutputFileFormat::Regenie => "none",
    };
    let chunk_commits = build_run_manifest_chunk_commits(&job, output_format, compression)?;

    let schema_metadata_build_start_time = Instant::now();
    let chunk_schema = build_regenie_step2_chunk_file_schema(&chunk_commits, output_statistic_dtype)?;
    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming {
        schema_metadata_build_seconds: schema_metadata_build_start_time.elapsed().as_secs_f64(),
        ..RegenieStep2RecordBatchBuildTiming::default()
    };
    let stream_write_result = match output_format {
        OutputFileFormat::Arrow => write_regenie_step2_chunks_to_arrow_file(
            job.chunks,
            chunk_schema,
            &temporary_chunk_file_path,
            arrow_compression,
        )?,
        OutputFileFormat::Parquet => write_regenie_step2_chunks_to_parquet_file(
            job.chunks,
            chunk_schema,
            &temporary_chunk_file_path,
            parquet_compression,
            &chunk_commits,
        )?,
        OutputFileFormat::Regenie => {
            write_regenie_step2_chunks_to_regenie_text_file(job.chunks, chunk_schema, &temporary_chunk_file_path)?
        }
    };
    record_batch_build_timing.add(stream_write_result.record_batch_build_timing);
    let record_batch_build_seconds =
        record_batch_build_timing.schema_metadata_build_seconds + stream_write_result.record_batch_build_seconds;
    let arrow_file_write_timing = stream_write_result.arrow_file_write_timing;
    let arrow_file_rename_start_time = Instant::now();
    std::fs::rename(&temporary_chunk_file_path, &chunk_file_path).map_err(|error| error.to_string())?;
    let arrow_file_rename_seconds = arrow_file_rename_start_time.elapsed().as_secs_f64();
    if output_format == OutputFileFormat::Regenie {
        write_regenie_text_metadata_sidecar(&chunk_file_path, &chunk_commits)?;
    }
    let arrow_file_bytes = std::fs::metadata(&chunk_file_path).map_err(|error| error.to_string())?.len();
    let arrow_file_write_seconds = arrow_file_write_timing.total_seconds();

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
    output_statistic_dtype: OutputStatisticDtype,
) -> Result<Arc<Schema>, String> {
    let mut metadata = HashMap::new();
    metadata.insert(schema::CHUNK_COMMITS_METADATA_KEY.to_string(), build_chunk_commit_metadata_text(chunk_commits)?);
    Ok(Arc::new(
        schema::get_regenie_step2_chunk_schema(output_statistic_dtype).as_ref().clone().with_metadata(metadata),
    ))
}

fn build_regenie_step2_parquet_record_batch_schema(chunk_schema: &Schema) -> Arc<Schema> {
    let fields = chunk_schema
        .fields()
        .iter()
        .map(|field| {
            let data_type = match field.name().as_str() {
                "CORRECTION_METHOD" | "CORRECTION_STATUS" => {
                    DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8))
                }
                _ => field.data_type().clone(),
            };
            Field::new(field.name().clone(), data_type, field.is_nullable()).with_metadata(field.metadata().clone())
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(fields, chunk_schema.metadata().clone()))
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

pub(crate) fn build_regenie_text_part_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("part_{first_chunk_identifier:09}.regenie");
    }
    format!("part_{first_chunk_identifier:09}_{last_chunk_identifier:09}.regenie")
}

pub(crate) fn build_regenie_text_metadata_sidecar_path(chunk_file_path: &Path) -> PathBuf {
    chunk_file_path.with_extension("regenie.json")
}

pub(crate) fn build_output_file_name(
    output_format: OutputFileFormat,
    first_chunk_identifier: i64,
    last_chunk_identifier: i64,
) -> String {
    match output_format {
        OutputFileFormat::Arrow => build_chunk_file_name(first_chunk_identifier, last_chunk_identifier),
        OutputFileFormat::Parquet => build_part_file_name(first_chunk_identifier, last_chunk_identifier),
        OutputFileFormat::Regenie => build_regenie_text_part_file_name(first_chunk_identifier, last_chunk_identifier),
    }
}

#[cfg(test)]
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
            let record_batch_build_result = build_regenie_step2_record_batch(
                chunk_job,
                Arc::clone(&chunk_schema),
                &mut array_cache,
                RegenieStep2CorrectionArrayEncoding::String,
            )?;
            timing.add(record_batch_build_result.timing);
            Ok(record_batch_build_result.record_batch)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(RegenieStep2RecordBatchBuildResult { record_batches, timing })
}

#[cfg(test)]
struct RegenieStep2RecordBatchBuildResult {
    record_batches: Vec<RecordBatch>,
    #[allow(dead_code)]
    timing: RegenieStep2RecordBatchBuildTiming,
}

struct RegenieStep2SingleRecordBatchBuildResult {
    record_batch: RecordBatch,
    timing: RegenieStep2RecordBatchBuildTiming,
}

struct RegenieStep2ChunkStreamWriteResult {
    record_batch_build_timing: RegenieStep2RecordBatchBuildTiming,
    record_batch_build_seconds: f64,
    arrow_file_write_timing: RegenieStep2ArrowFileWriteTiming,
}

#[derive(Default)]
struct RegenieStep2RecordBatchArrayCache {
    null_extra_arrays_by_row_count: HashMap<usize, ArrayRef>,
    test_arrays_by_row_count: HashMap<usize, ArrayRef>,
    constant_correction_dictionary_arrays: HashMap<CorrectionDictionaryArrayCacheKey, ArrayRef>,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct CorrectionDictionaryArrayCacheKey {
    row_count: usize,
    dictionary_kind: CorrectionDictionaryKind,
    dictionary_key: u8,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum CorrectionDictionaryKind {
    Method,
    Status,
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

    fn constant_correction_dictionary_array(
        &mut self,
        cache_key: CorrectionDictionaryArrayCacheKey,
        dictionary_values: ArrayRef,
    ) -> Result<ArrayRef, String> {
        if let Some(cached_array) = self.constant_correction_dictionary_arrays.get(&cache_key) {
            return Ok(Arc::clone(cached_array));
        }
        let dictionary_array =
            build_uint8_dictionary_array(vec![cache_key.dictionary_key; cache_key.row_count], dictionary_values)?;
        self.constant_correction_dictionary_arrays.insert(cache_key, Arc::clone(&dictionary_array));
        Ok(dictionary_array)
    }
}

fn build_regenie_step2_record_batch(
    chunk_job: RegenieStep2ChunkJob,
    chunk_schema: Arc<Schema>,
    array_cache: &mut RegenieStep2RecordBatchArrayCache,
    correction_array_encoding: RegenieStep2CorrectionArrayEncoding,
) -> Result<RegenieStep2SingleRecordBatchBuildResult, String> {
    let row_count = chunk_job.chunk_handle.row_count();
    let metadata_array_build_start_time = Instant::now();
    let cached_writer_arrays = chunk_job.chunk_handle.writer_arrays();
    let chromosome_array = Arc::clone(&cached_writer_arrays.chromosome);
    let position_array = Arc::clone(&cached_writer_arrays.position);
    let variant_identifier_array = Arc::clone(&cached_writer_arrays.variant_identifier);
    let allele_two_array = Arc::clone(&cached_writer_arrays.allele_two);
    let allele_one_array = Arc::clone(&cached_writer_arrays.allele_one);
    let metadata_array_build_seconds = metadata_array_build_start_time.elapsed().as_secs_f64();

    let statistic_array_build_start_time = Instant::now();
    let allele_one_frequency_array = Arc::clone(&cached_writer_arrays.allele_one_frequency);
    let info_score_array = Arc::clone(&cached_writer_arrays.info_score);
    let observation_count_array = Arc::clone(&cached_writer_arrays.observation_count);
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
    let extra_code_array = chunk_job.extra_code;
    let extra_array = match extra_code_array.as_ref() {
        Some(extra_code) => schema::build_extra_string_array(Some(Arc::clone(extra_code)), row_count)?,
        None => array_cache.null_extra_array(row_count),
    };
    let (correction_method_array, correction_status_array) = match correction_array_encoding {
        RegenieStep2CorrectionArrayEncoding::String => (
            schema::build_correction_method_array(extra_code_array.as_ref().map(Arc::clone), row_count)?,
            schema::build_correction_status_array(extra_code_array, row_count)?,
        ),
        RegenieStep2CorrectionArrayEncoding::Dictionary => (
            build_correction_method_dictionary_array(
                extra_code_array.as_ref().map(Arc::clone),
                row_count,
                array_cache,
            )?,
            build_correction_status_dictionary_array(extra_code_array, row_count, array_cache)?,
        ),
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
        correction_method_array,
        correction_status_array,
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

fn build_correction_method_dictionary_array(
    extra_code: Option<ArrayRef>,
    row_count: usize,
    array_cache: &mut RegenieStep2RecordBatchArrayCache,
) -> Result<ArrayRef, String> {
    build_correction_dictionary_array(
        extra_code,
        row_count,
        array_cache,
        CorrectionDictionaryKind::Method,
        build_correction_method_dictionary_values(),
        "correction method",
        |extra_code_value| match extra_code_value {
            0 => Some(CORRECTION_METHOD_SCORE_KEY),
            1 | 3 => Some(CORRECTION_METHOD_FIRTH_APPROXIMATE_KEY),
            2 => Some(CORRECTION_METHOD_SPA_KEY),
            _ => None,
        },
    )
}

fn build_correction_status_dictionary_array(
    extra_code: Option<ArrayRef>,
    row_count: usize,
    array_cache: &mut RegenieStep2RecordBatchArrayCache,
) -> Result<ArrayRef, String> {
    build_correction_dictionary_array(
        extra_code,
        row_count,
        array_cache,
        CorrectionDictionaryKind::Status,
        build_correction_status_dictionary_values(),
        "correction status",
        |extra_code_value| match extra_code_value {
            0..=2 => Some(CORRECTION_STATUS_SUCCESS_KEY),
            3 => Some(CORRECTION_STATUS_FAILED_KEY),
            _ => None,
        },
    )
}

fn build_correction_dictionary_array(
    extra_code: Option<ArrayRef>,
    row_count: usize,
    array_cache: &mut RegenieStep2RecordBatchArrayCache,
    dictionary_kind: CorrectionDictionaryKind,
    dictionary_values: ArrayRef,
    label_kind: &str,
    key_for_code: impl Fn(i32) -> Option<u8>,
) -> Result<ArrayRef, String> {
    let Some(extra_code_array) = extra_code else {
        return array_cache.constant_correction_dictionary_array(
            CorrectionDictionaryArrayCacheKey { row_count, dictionary_kind, dictionary_key: 0 },
            dictionary_values,
        );
    };
    let extra_code_values = extra_code_array
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| format!("REGENIE step 2 {label_kind} code must be an int32 array."))?;
    if extra_code_values.len() != row_count {
        return Err(format!("REGENIE step 2 {label_kind} row count does not match metadata row count."));
    }

    let mut dictionary_keys = Vec::with_capacity(row_count);
    let mut first_dictionary_key: Option<u8> = None;
    let mut all_keys_same = true;
    for row_index in 0..extra_code_values.len() {
        let dictionary_key = if extra_code_values.is_null(row_index) {
            0
        } else {
            let extra_code_value = extra_code_values.value(row_index);
            key_for_code(extra_code_value)
                .ok_or_else(|| format!("Unsupported REGENIE step 2 extra code: {extra_code_value}"))?
        };
        if let Some(previous_dictionary_key) = first_dictionary_key {
            all_keys_same &= previous_dictionary_key == dictionary_key;
        } else {
            first_dictionary_key = Some(dictionary_key);
        }
        dictionary_keys.push(dictionary_key);
    }

    if all_keys_same {
        return array_cache.constant_correction_dictionary_array(
            CorrectionDictionaryArrayCacheKey {
                row_count,
                dictionary_kind,
                dictionary_key: first_dictionary_key.unwrap_or(0),
            },
            dictionary_values,
        );
    }
    build_uint8_dictionary_array(dictionary_keys, dictionary_values)
}

fn build_correction_method_dictionary_values() -> ArrayRef {
    Arc::new(StringArray::from_iter_values(["score", "firth_approximate", "spa"]))
}

fn build_correction_status_dictionary_values() -> ArrayRef {
    Arc::new(StringArray::from_iter_values(["success", "failed"]))
}

fn build_uint8_dictionary_array(dictionary_keys: Vec<u8>, dictionary_values: ArrayRef) -> Result<ArrayRef, String> {
    let key_array = UInt8Array::from(dictionary_keys);
    DictionaryArray::<UInt8Type>::try_new(key_array, dictionary_values)
        .map(|dictionary_array| Arc::new(dictionary_array) as ArrayRef)
        .map_err(|error| error.to_string())
}

struct RegenieStep2ArrowFileWriteTiming {
    file_create: f64,
    writer_init: f64,
    batch_write: f64,
    writer_finish: f64,
}

impl RegenieStep2ArrowFileWriteTiming {
    fn total_seconds(&self) -> f64 {
        self.file_create + self.writer_init + self.batch_write + self.writer_finish
    }
}

fn write_regenie_step2_chunks_to_arrow_file(
    chunks: Vec<RegenieStep2ChunkJob>,
    chunk_schema: Arc<Schema>,
    chunk_file_path: &Path,
    arrow_compression: &str,
) -> Result<RegenieStep2ChunkStreamWriteResult, String> {
    let file_create_start_time = Instant::now();
    let output_file = File::create(chunk_file_path).map_err(|error| error.to_string())?;
    let file_create = file_create_start_time.elapsed().as_secs_f64();

    let writer_init_start_time = Instant::now();
    let write_options = build_regenie_step2_ipc_write_options(arrow_compression)?;
    let mut writer = FileWriter::try_new_with_options(output_file, &chunk_schema, write_options)
        .map_err(|error| error.to_string())?;
    let writer_init = writer_init_start_time.elapsed().as_secs_f64();

    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming::default();
    let mut record_batch_build_seconds = 0.0;
    let mut array_cache = RegenieStep2RecordBatchArrayCache::default();
    let mut batch_write = 0.0;
    for chunk_job in chunks {
        let record_batch_build_start_time = Instant::now();
        let record_batch_build_result = build_regenie_step2_record_batch(
            chunk_job,
            Arc::clone(&chunk_schema),
            &mut array_cache,
            RegenieStep2CorrectionArrayEncoding::String,
        )?;
        record_batch_build_seconds += record_batch_build_start_time.elapsed().as_secs_f64();
        record_batch_build_timing.add(record_batch_build_result.timing);

        let batch_write_start_time = Instant::now();
        writer.write(&record_batch_build_result.record_batch).map_err(|error| error.to_string())?;
        batch_write += batch_write_start_time.elapsed().as_secs_f64();
    }
    let writer_finish_start_time = Instant::now();
    writer.finish().map_err(|error| error.to_string())?;
    let writer_finish = writer_finish_start_time.elapsed().as_secs_f64();
    Ok(RegenieStep2ChunkStreamWriteResult {
        record_batch_build_timing,
        record_batch_build_seconds,
        arrow_file_write_timing: RegenieStep2ArrowFileWriteTiming {
            file_create,
            writer_init,
            batch_write,
            writer_finish,
        },
    })
}

fn write_regenie_step2_chunks_to_parquet_file(
    chunks: Vec<RegenieStep2ChunkJob>,
    chunk_schema: Arc<Schema>,
    chunk_file_path: &Path,
    parquet_compression: &str,
    chunk_commits: &[manifest::RunManifestChunkCommit],
) -> Result<RegenieStep2ChunkStreamWriteResult, String> {
    let file_create_start_time = Instant::now();
    let output_file = File::create(chunk_file_path).map_err(|error| error.to_string())?;
    let file_create = file_create_start_time.elapsed().as_secs_f64();

    let writer_init_start_time = Instant::now();
    let writer_properties = build_regenie_step2_parquet_writer_properties(parquet_compression)?;
    let mut writer = ArrowWriter::try_new(output_file, Arc::clone(&chunk_schema), Some(writer_properties))
        .map_err(|error| error.to_string())?;
    writer.append_key_value_metadata(KeyValue {
        key: schema::CHUNK_COMMITS_METADATA_KEY.to_string(),
        value: Some(build_chunk_commit_metadata_text(chunk_commits)?),
    });
    let writer_init = writer_init_start_time.elapsed().as_secs_f64();

    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming::default();
    let mut record_batch_build_seconds = 0.0;
    let mut array_cache = RegenieStep2RecordBatchArrayCache::default();
    let parquet_record_batch_schema = build_regenie_step2_parquet_record_batch_schema(&chunk_schema);
    let mut batch_write = 0.0;
    for chunk_job in chunks {
        let record_batch_build_start_time = Instant::now();
        let record_batch_build_result = build_regenie_step2_record_batch(
            chunk_job,
            Arc::clone(&parquet_record_batch_schema),
            &mut array_cache,
            RegenieStep2CorrectionArrayEncoding::Dictionary,
        )?;
        record_batch_build_seconds += record_batch_build_start_time.elapsed().as_secs_f64();
        record_batch_build_timing.add(record_batch_build_result.timing);

        let batch_write_start_time = Instant::now();
        writer.write(&record_batch_build_result.record_batch).map_err(|error| error.to_string())?;
        batch_write += batch_write_start_time.elapsed().as_secs_f64();
    }
    let writer_finish_start_time = Instant::now();
    writer.close().map_err(|error| error.to_string())?;
    let writer_finish = writer_finish_start_time.elapsed().as_secs_f64();
    Ok(RegenieStep2ChunkStreamWriteResult {
        record_batch_build_timing,
        record_batch_build_seconds,
        arrow_file_write_timing: RegenieStep2ArrowFileWriteTiming {
            file_create,
            writer_init,
            batch_write,
            writer_finish,
        },
    })
}

fn write_regenie_step2_chunks_to_regenie_text_file(
    chunks: Vec<RegenieStep2ChunkJob>,
    chunk_schema: Arc<Schema>,
    chunk_file_path: &Path,
) -> Result<RegenieStep2ChunkStreamWriteResult, String> {
    let file_create_start_time = Instant::now();
    let output_file = File::create(chunk_file_path).map_err(|error| error.to_string())?;
    let file_create = file_create_start_time.elapsed().as_secs_f64();

    let writer_init_start_time = Instant::now();
    let mut output_writer = BufWriter::new(output_file);
    output_writer.write_all(REGENIE_STEP2_TEXT_HEADER.as_bytes()).map_err(|error| error.to_string())?;
    let writer_init = writer_init_start_time.elapsed().as_secs_f64();

    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming::default();
    let mut record_batch_build_seconds = 0.0;
    let mut array_cache = RegenieStep2RecordBatchArrayCache::default();
    let mut batch_write = 0.0;
    for chunk_job in chunks {
        let record_batch_build_start_time = Instant::now();
        let record_batch_build_result = build_regenie_step2_record_batch(
            chunk_job,
            Arc::clone(&chunk_schema),
            &mut array_cache,
            RegenieStep2CorrectionArrayEncoding::String,
        )?;
        record_batch_build_seconds += record_batch_build_start_time.elapsed().as_secs_f64();
        record_batch_build_timing.add(record_batch_build_result.timing);

        let batch_write_start_time = Instant::now();
        write_regenie_step2_text_record_batch(&mut output_writer, &record_batch_build_result.record_batch)?;
        batch_write += batch_write_start_time.elapsed().as_secs_f64();
    }
    let writer_finish_start_time = Instant::now();
    output_writer.flush().map_err(|error| error.to_string())?;
    let writer_finish = writer_finish_start_time.elapsed().as_secs_f64();
    Ok(RegenieStep2ChunkStreamWriteResult {
        record_batch_build_timing,
        record_batch_build_seconds,
        arrow_file_write_timing: RegenieStep2ArrowFileWriteTiming {
            file_create,
            writer_init,
            batch_write,
            writer_finish,
        },
    })
}

fn write_regenie_step2_text_record_batch(
    output_writer: &mut BufWriter<File>,
    record_batch: &RecordBatch,
) -> Result<(), String> {
    let chromosome_array = required_string_column(record_batch, "CHROM")?;
    let position_array = required_int64_column(record_batch, "GENPOS")?;
    let variant_identifier_array = required_string_column(record_batch, "ID")?;
    let allele_zero_array = required_string_column(record_batch, "ALLELE0")?;
    let allele_one_array = required_string_column(record_batch, "ALLELE1")?;
    let allele_one_frequency_array = required_float32_column(record_batch, "A1FREQ")?;
    let info_score_array = required_float32_column(record_batch, "INFO")?;
    let observation_count_array = required_int32_column(record_batch, "N")?;
    let test_array = required_string_column(record_batch, "TEST")?;
    let beta_array = required_statistic_column(record_batch, "BETA")?;
    let standard_error_array = required_statistic_column(record_batch, "SE")?;
    let chi_squared_array = required_statistic_column(record_batch, "CHISQ")?;
    let log10_p_value_array = required_statistic_column(record_batch, "LOG10P")?;
    let extra_array = required_string_column(record_batch, "EXTRA")?;
    let correction_method_array = required_string_column(record_batch, "CORRECTION_METHOD")?;
    let correction_status_array = required_string_column(record_batch, "CORRECTION_STATUS")?;
    for row_index in 0..record_batch.num_rows() {
        write_regenie_text_string_value(output_writer, chromosome_array, row_index, "CHROM")?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_int64_value(output_writer, position_array, row_index)?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_string_value(output_writer, variant_identifier_array, row_index, "ID")?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_string_value(output_writer, allele_zero_array, row_index, "ALLELE0")?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_string_value(output_writer, allele_one_array, row_index, "ALLELE1")?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_float32_value(output_writer, allele_one_frequency_array, row_index)?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_float32_value(output_writer, info_score_array, row_index)?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_int32_value(output_writer, observation_count_array, row_index)?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_string_value(output_writer, test_array, row_index, "TEST")?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_statistic_value(output_writer, beta_array, row_index)?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_statistic_value(output_writer, standard_error_array, row_index)?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_statistic_value(output_writer, chi_squared_array, row_index)?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_statistic_value(output_writer, log10_p_value_array, row_index)?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_string_value(output_writer, extra_array, row_index, "EXTRA")?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_string_value(output_writer, correction_method_array, row_index, "CORRECTION_METHOD")?;
        output_writer.write_all(b"\t").map_err(|error| error.to_string())?;
        write_regenie_text_string_value(output_writer, correction_status_array, row_index, "CORRECTION_STATUS")?;
        output_writer.write_all(b"\n").map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn required_string_column<'a>(record_batch: &'a RecordBatch, column_name: &str) -> Result<&'a StringArray, String> {
    record_batch
        .column_by_name(column_name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| format!("REGENIE text writer could not read string column {column_name}."))
}

fn required_float32_column<'a>(record_batch: &'a RecordBatch, column_name: &str) -> Result<&'a Float32Array, String> {
    record_batch
        .column_by_name(column_name)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| format!("REGENIE text writer could not read float32 column {column_name}."))
}

#[derive(Clone, Copy)]
enum StatisticColumnRef<'a> {
    Float32(&'a Float32Array),
    Float64(&'a Float64Array),
}

fn required_statistic_column<'a>(
    record_batch: &'a RecordBatch,
    column_name: &str,
) -> Result<StatisticColumnRef<'a>, String> {
    let Some(column) = record_batch.column_by_name(column_name) else {
        return Err(format!("REGENIE text writer could not read statistic column {column_name}."));
    };
    if let Some(float32_column) = column.as_any().downcast_ref::<Float32Array>() {
        return Ok(StatisticColumnRef::Float32(float32_column));
    }
    if let Some(float64_column) = column.as_any().downcast_ref::<Float64Array>() {
        return Ok(StatisticColumnRef::Float64(float64_column));
    }
    Err(format!("REGENIE text writer could not read float32/float64 statistic column {column_name}."))
}

fn required_int32_column<'a>(record_batch: &'a RecordBatch, column_name: &str) -> Result<&'a Int32Array, String> {
    record_batch
        .column_by_name(column_name)
        .and_then(|column| column.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| format!("REGENIE text writer could not read int32 column {column_name}."))
}

fn required_int64_column<'a>(record_batch: &'a RecordBatch, column_name: &str) -> Result<&'a Int64Array, String> {
    record_batch
        .column_by_name(column_name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| format!("REGENIE text writer could not read int64 column {column_name}."))
}

fn write_regenie_text_string_value(
    output_writer: &mut BufWriter<File>,
    array: &StringArray,
    row_index: usize,
    column_name: &str,
) -> Result<(), String> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(|error| error.to_string());
    }
    let value = array.value(row_index);
    if value.contains('\t') || value.contains('\n') || value.contains('\r') {
        return Err(format!("REGENIE text writer found an unsupported separator in {column_name}."));
    }
    output_writer.write_all(value.as_bytes()).map_err(|error| error.to_string())
}

fn write_regenie_text_float32_value(
    output_writer: &mut BufWriter<File>,
    array: &Float32Array,
    row_index: usize,
) -> Result<(), String> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(|error| error.to_string());
    }
    let value = array.value(row_index);
    if !value.is_finite() {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(|error| error.to_string());
    }
    write!(output_writer, "{value}").map_err(|error| error.to_string())
}

fn write_regenie_text_statistic_value(
    output_writer: &mut BufWriter<File>,
    array: StatisticColumnRef<'_>,
    row_index: usize,
) -> Result<(), String> {
    match array {
        StatisticColumnRef::Float32(float32_array) => {
            write_regenie_text_float32_value(output_writer, float32_array, row_index)
        }
        StatisticColumnRef::Float64(float64_array) => {
            write_regenie_text_float64_value(output_writer, float64_array, row_index)
        }
    }
}

fn write_regenie_text_float64_value(
    output_writer: &mut BufWriter<File>,
    array: &Float64Array,
    row_index: usize,
) -> Result<(), String> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(|error| error.to_string());
    }
    let value = array.value(row_index);
    if !value.is_finite() {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(|error| error.to_string());
    }
    write!(output_writer, "{value}").map_err(|error| error.to_string())
}

fn write_regenie_text_int32_value(
    output_writer: &mut BufWriter<File>,
    array: &Int32Array,
    row_index: usize,
) -> Result<(), String> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(|error| error.to_string());
    }
    write!(output_writer, "{}", array.value(row_index)).map_err(|error| error.to_string())
}

fn write_regenie_text_int64_value(
    output_writer: &mut BufWriter<File>,
    array: &Int64Array,
    row_index: usize,
) -> Result<(), String> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(|error| error.to_string());
    }
    write!(output_writer, "{}", array.value(row_index)).map_err(|error| error.to_string())
}

fn write_regenie_text_metadata_sidecar(
    chunk_file_path: &Path,
    chunk_commits: &[manifest::RunManifestChunkCommit],
) -> Result<(), String> {
    let sidecar_path = build_regenie_text_metadata_sidecar_path(chunk_file_path);
    let temporary_sidecar_path = sidecar_path.with_extension("json.tmp");
    let metadata_text = build_chunk_commit_metadata_text(chunk_commits)?;
    std::fs::write(&temporary_sidecar_path, format!("{metadata_text}\n")).map_err(|error| error.to_string())?;
    std::fs::rename(&temporary_sidecar_path, &sidecar_path).map_err(|error| error.to_string())
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
        .set_column_dictionary_enabled(ColumnPath::from("CORRECTION_METHOD"), true)
        .set_column_dictionary_enabled(ColumnPath::from("CORRECTION_STATUS"), true)
        .build())
}
