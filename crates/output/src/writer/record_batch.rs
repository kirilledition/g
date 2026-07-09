use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, DictionaryArray, Int32Array, RecordBatch, StringArray, UInt8Array};
use arrow::datatypes::{DataType, Field, Schema, UInt8Type};

use crate::error::OutputError;
use crate::schema;

use super::{OutputWriterResult, RegenieStep2ChunkJob, RegenieStep2RecordBatchBuildTiming};

pub(super) const CORRECTION_METHOD_FIRTH_APPROXIMATE_KEY: u8 = 1;
pub(super) const CORRECTION_STATUS_SUCCESS_KEY: u8 = 0;

const CORRECTION_METHOD_SCORE_KEY: u8 = 0;
const CORRECTION_METHOD_SPA_KEY: u8 = 2;
const CORRECTION_STATUS_FAILED_KEY: u8 = 1;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum RegenieStep2CorrectionArrayEncoding {
    String,
    Dictionary,
}

pub(super) struct RegenieStep2SingleRecordBatchBuildResult {
    pub(super) record_batch: RecordBatch,
    pub(super) timing: RegenieStep2RecordBatchBuildTiming,
}

#[derive(Default)]
pub(super) struct RegenieStep2RecordBatchArrayCache {
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
    ) -> OutputWriterResult<ArrayRef> {
        if let Some(cached_array) = self.constant_correction_dictionary_arrays.get(&cache_key) {
            return Ok(Arc::clone(cached_array));
        }
        let dictionary_array =
            build_uint8_dictionary_array(vec![cache_key.dictionary_key; cache_key.row_count], dictionary_values)?;
        self.constant_correction_dictionary_arrays.insert(cache_key, Arc::clone(&dictionary_array));
        Ok(dictionary_array)
    }
}

pub(super) fn build_regenie_step2_parquet_record_batch_schema(chunk_schema: &Schema) -> Arc<Schema> {
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

pub(super) fn build_regenie_step2_record_batch(
    chunk_job: RegenieStep2ChunkJob,
    chunk_schema: Arc<Schema>,
    array_cache: &mut RegenieStep2RecordBatchArrayCache,
    correction_array_encoding: RegenieStep2CorrectionArrayEncoding,
) -> OutputWriterResult<RegenieStep2SingleRecordBatchBuildResult> {
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
    let record_batch = RecordBatch::try_new(chunk_schema, columns).map_err(OutputError::runtime)?;
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
) -> OutputWriterResult<ArrayRef> {
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
) -> OutputWriterResult<ArrayRef> {
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
) -> OutputWriterResult<ArrayRef> {
    let Some(extra_code_array) = extra_code else {
        return array_cache.constant_correction_dictionary_array(
            CorrectionDictionaryArrayCacheKey { row_count, dictionary_kind, dictionary_key: 0 },
            dictionary_values,
        );
    };
    let extra_code_values = extra_code_array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
        OutputError::InvalidInput(format!("REGENIE step 2 {label_kind} code must be an int32 array."))
    })?;
    if extra_code_values.len() != row_count {
        return Err(OutputError::InvalidInput(format!(
            "REGENIE step 2 {label_kind} row count does not match metadata row count."
        )));
    }

    let mut dictionary_keys = Vec::with_capacity(row_count);
    let mut first_dictionary_key: Option<u8> = None;
    let mut all_keys_same = true;
    for row_index in 0..extra_code_values.len() {
        let dictionary_key = if extra_code_values.is_null(row_index) {
            0
        } else {
            let extra_code_value = extra_code_values.value(row_index);
            key_for_code(extra_code_value).ok_or_else(|| {
                OutputError::InvalidInput(format!("Unsupported REGENIE step 2 extra code: {extra_code_value}"))
            })?
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

fn build_uint8_dictionary_array(dictionary_keys: Vec<u8>, dictionary_values: ArrayRef) -> OutputWriterResult<ArrayRef> {
    let key_array = UInt8Array::from(dictionary_keys);
    DictionaryArray::<UInt8Type>::try_new(key_array, dictionary_values)
        .map(|dictionary_array| Arc::new(dictionary_array) as ArrayRef)
        .map_err(OutputError::runtime)
}
