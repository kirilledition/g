use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use arrow::array::{Array, ArrayRef, DictionaryArray, RecordBatch, StringArray, UInt8Array};
use arrow::datatypes::{Schema, UInt8Type};

use super::{OutputResult, RegenieStep2ChunkJob, RegenieStep2RecordBatchBuildTiming};
use crate::error::OutputError;
use crate::timing::start_optional_timing;

pub(super) const CORRECTION_METHOD_FIRTH_APPROXIMATE_KEY: u8 = 1;
pub(super) const CORRECTION_STATUS_SUCCESS_KEY: u8 = 0;

const CORRECTION_METHOD_SCORE_KEY: u8 = 0;
const CORRECTION_STATUS_FAILED_KEY: u8 = 1;

static CORRECTION_METHOD_DICTIONARY_VALUES: LazyLock<ArrayRef> =
    LazyLock::new(|| Arc::new(StringArray::from_iter_values(["score", "firth_approximate"])));
static CORRECTION_STATUS_DICTIONARY_VALUES: LazyLock<ArrayRef> =
    LazyLock::new(|| Arc::new(StringArray::from_iter_values(["success", "failed"])));

pub(super) struct RegenieStep2SingleRecordBatchBuildResult {
    pub(super) record_batch: RecordBatch,
    pub(super) timing: RegenieStep2RecordBatchBuildTiming,
}

#[derive(Default)]
pub(super) struct RegenieStep2RecordBatchArrayCache {
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
    fn constant_correction_dictionary_array(
        &mut self,
        cache_key: CorrectionDictionaryArrayCacheKey,
        dictionary_values: ArrayRef,
    ) -> OutputResult<ArrayRef> {
        if let Some(cached_array) = self.constant_correction_dictionary_arrays.get(&cache_key) {
            return Ok(Arc::clone(cached_array));
        }
        let dictionary_array =
            build_uint8_dictionary_array(vec![cache_key.dictionary_key; cache_key.row_count], dictionary_values)?;
        self.constant_correction_dictionary_arrays.insert(cache_key, Arc::clone(&dictionary_array));
        Ok(dictionary_array)
    }
}

pub(super) fn build_regenie_step2_record_batch(
    chunk_job: RegenieStep2ChunkJob,
    chunk_schema: Arc<Schema>,
    array_cache: &mut RegenieStep2RecordBatchArrayCache,
    collect_stage_timings: bool,
) -> OutputResult<RegenieStep2SingleRecordBatchBuildResult> {
    let row_count = chunk_job.chunk_handle.row_count();
    let metadata_array_build_start_time = start_optional_timing(collect_stage_timings);
    let cached_writer_arrays = &chunk_job.chunk_handle.writer_arrays;
    let metadata_arrays = cached_writer_arrays.metadata.arrays();
    let chromosome_array = Arc::clone(&metadata_arrays.chromosome);
    let position_array = Arc::clone(&metadata_arrays.position);
    let variant_identifier_array = Arc::clone(&metadata_arrays.variant_identifier);
    let allele_two_array = Arc::clone(&metadata_arrays.allele_two);
    let allele_one_array = Arc::clone(&metadata_arrays.allele_one);
    let metadata_array_build_seconds =
        metadata_array_build_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());

    let statistic_array_build_start_time = start_optional_timing(collect_stage_timings);
    let allele_one_frequency_array = Arc::clone(&cached_writer_arrays.allele_one_frequency);
    let info_score_array = Arc::clone(&cached_writer_arrays.info_score);
    let observation_count_array = Arc::clone(&cached_writer_arrays.observation_count);
    let statistic_array_build_seconds =
        statistic_array_build_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());

    let result_array_build_start_time = start_optional_timing(collect_stage_timings);
    let beta_array = chunk_job.beta;
    let standard_error_array = chunk_job.se;
    let chi_squared_array = chunk_job.chisq;
    let log10_p_value_array = chunk_job.log10p;
    let result_array_build_seconds =
        result_array_build_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());

    let correction_dictionary_arrays =
        build_correction_dictionary_arrays(chunk_job.correction_code, row_count, array_cache)?;

    let columns: Vec<ArrayRef> = vec![
        chromosome_array,
        position_array,
        variant_identifier_array,
        allele_two_array,
        allele_one_array,
        allele_one_frequency_array,
        info_score_array,
        observation_count_array,
        beta_array,
        standard_error_array,
        chi_squared_array,
        log10_p_value_array,
        correction_dictionary_arrays.method,
        correction_dictionary_arrays.status,
    ];
    let arrow_array_memory_bytes = if collect_stage_timings {
        columns.iter().fold(0_u64, |total, column| {
            total.saturating_add(u64::try_from(column.get_array_memory_size()).unwrap_or(u64::MAX))
        })
    } else {
        0
    };
    let record_batch_try_new_start_time = start_optional_timing(collect_stage_timings);
    let record_batch = RecordBatch::try_new(chunk_schema, columns).map_err(OutputError::runtime)?;
    let record_batch_try_new_seconds =
        record_batch_try_new_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
    Ok(RegenieStep2SingleRecordBatchBuildResult {
        record_batch,
        timing: RegenieStep2RecordBatchBuildTiming {
            metadata_array_build_seconds,
            statistic_array_build_seconds,
            result_array_build_seconds,
            record_batch_try_new_seconds,
            arrow_array_memory_bytes,
        },
    })
}

struct CorrectionDictionaryArrays {
    method: ArrayRef,
    status: ArrayRef,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct CorrectionDictionaryKeys {
    method: u8,
    status: u8,
}

#[derive(Default)]
struct CorrectionDictionaryKeyColumn {
    first_key: Option<u8>,
    mixed_keys: Option<Vec<u8>>,
}

impl CorrectionDictionaryKeyColumn {
    fn push(&mut self, dictionary_key: u8, row_index: usize, row_count: usize) {
        let Some(first_key) = self.first_key else {
            self.first_key = Some(dictionary_key);
            return;
        };
        if let Some(keys) = self.mixed_keys.as_mut() {
            keys.push(dictionary_key);
        } else if dictionary_key != first_key {
            let mut keys = Vec::with_capacity(row_count);
            keys.resize(row_index, first_key);
            keys.push(dictionary_key);
            self.mixed_keys = Some(keys);
        }
    }

    fn build_array(
        self,
        row_count: usize,
        dictionary_kind: CorrectionDictionaryKind,
        dictionary_values: ArrayRef,
        array_cache: &mut RegenieStep2RecordBatchArrayCache,
    ) -> OutputResult<ArrayRef> {
        if let Some(keys) = self.mixed_keys {
            return build_uint8_dictionary_array(keys, dictionary_values);
        }
        array_cache.constant_correction_dictionary_array(
            CorrectionDictionaryArrayCacheKey {
                row_count,
                dictionary_kind,
                dictionary_key: self.first_key.unwrap_or(0),
            },
            dictionary_values,
        )
    }
}

fn build_correction_dictionary_arrays(
    correction_code: Option<ArrayRef>,
    row_count: usize,
    array_cache: &mut RegenieStep2RecordBatchArrayCache,
) -> OutputResult<CorrectionDictionaryArrays> {
    let Some(correction_code_array) = correction_code else {
        return build_constant_correction_dictionary_arrays(
            CorrectionDictionaryKeys { method: CORRECTION_METHOD_SCORE_KEY, status: CORRECTION_STATUS_SUCCESS_KEY },
            row_count,
            array_cache,
        );
    };
    let correction_code_values = correction_code_array
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| OutputError::InvalidInput("REGENIE step 2 correction code must be a uint8 array.".to_owned()))?;
    if correction_code_values.len() != row_count {
        return Err(OutputError::InvalidInput(
            "REGENIE step 2 correction-code row count does not match metadata row count.".to_owned(),
        ));
    }

    let mut method_keys = CorrectionDictionaryKeyColumn::default();
    let mut status_keys = CorrectionDictionaryKeyColumn::default();
    for row_index in 0..correction_code_values.len() {
        let dictionary_keys = if correction_code_values.is_null(row_index) {
            CorrectionDictionaryKeys { method: CORRECTION_METHOD_SCORE_KEY, status: CORRECTION_STATUS_SUCCESS_KEY }
        } else {
            let correction_code_value = correction_code_values.value(row_index);
            correction_dictionary_keys(correction_code_value)?
        };

        method_keys.push(dictionary_keys.method, row_index, row_count);
        status_keys.push(dictionary_keys.status, row_index, row_count);
    }

    Ok(CorrectionDictionaryArrays {
        method: method_keys.build_array(
            row_count,
            CorrectionDictionaryKind::Method,
            Arc::clone(&CORRECTION_METHOD_DICTIONARY_VALUES),
            array_cache,
        )?,
        status: status_keys.build_array(
            row_count,
            CorrectionDictionaryKind::Status,
            Arc::clone(&CORRECTION_STATUS_DICTIONARY_VALUES),
            array_cache,
        )?,
    })
}

fn correction_dictionary_keys(correction_code: u8) -> OutputResult<CorrectionDictionaryKeys> {
    match correction_code {
        0 => {
            Ok(CorrectionDictionaryKeys { method: CORRECTION_METHOD_SCORE_KEY, status: CORRECTION_STATUS_SUCCESS_KEY })
        }
        1 => Ok(CorrectionDictionaryKeys { method: CORRECTION_METHOD_SCORE_KEY, status: CORRECTION_STATUS_FAILED_KEY }),
        2 => Ok(CorrectionDictionaryKeys {
            method: CORRECTION_METHOD_FIRTH_APPROXIMATE_KEY,
            status: CORRECTION_STATUS_SUCCESS_KEY,
        }),
        3 => Ok(CorrectionDictionaryKeys {
            method: CORRECTION_METHOD_FIRTH_APPROXIMATE_KEY,
            status: CORRECTION_STATUS_FAILED_KEY,
        }),
        _ => Err(OutputError::InvalidInput(format!("Unsupported REGENIE step 2 correction code: {correction_code}"))),
    }
}

fn build_constant_correction_dictionary_arrays(
    dictionary_keys: CorrectionDictionaryKeys,
    row_count: usize,
    array_cache: &mut RegenieStep2RecordBatchArrayCache,
) -> OutputResult<CorrectionDictionaryArrays> {
    Ok(CorrectionDictionaryArrays {
        method: array_cache.constant_correction_dictionary_array(
            CorrectionDictionaryArrayCacheKey {
                row_count,
                dictionary_kind: CorrectionDictionaryKind::Method,
                dictionary_key: dictionary_keys.method,
            },
            Arc::clone(&CORRECTION_METHOD_DICTIONARY_VALUES),
        )?,
        status: array_cache.constant_correction_dictionary_array(
            CorrectionDictionaryArrayCacheKey {
                row_count,
                dictionary_kind: CorrectionDictionaryKind::Status,
                dictionary_key: dictionary_keys.status,
            },
            Arc::clone(&CORRECTION_STATUS_DICTIONARY_VALUES),
        )?,
    })
}

fn build_uint8_dictionary_array(dictionary_keys: Vec<u8>, dictionary_values: ArrayRef) -> OutputResult<ArrayRef> {
    let key_array = UInt8Array::from(dictionary_keys);
    DictionaryArray::<UInt8Type>::try_new(key_array, dictionary_values)
        .map(|dictionary_array| Arc::new(dictionary_array) as ArrayRef)
        .map_err(OutputError::runtime)
}
