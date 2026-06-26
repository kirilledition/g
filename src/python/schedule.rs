//! PyO3 adapters for engine scheduling policy helpers.

use std::collections::BTreeSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use g_engine::schedule as native_schedule;

#[pyclass]
pub(crate) struct NativeCallbackQueueLimits {
    #[pyo3(get)]
    dosage_queue_depth: usize,
    #[pyo3(get)]
    result_queue_depth: usize,
    #[pyo3(get)]
    result_in_flight_limit: usize,
    #[pyo3(get)]
    dosage_buffer_limit: usize,
}

impl From<native_schedule::NativeCallbackQueueLimits> for NativeCallbackQueueLimits {
    fn from(queue_limits: native_schedule::NativeCallbackQueueLimits) -> Self {
        Self {
            dosage_queue_depth: queue_limits.dosage_queue_depth,
            result_queue_depth: queue_limits.result_queue_depth,
            result_in_flight_limit: queue_limits.result_in_flight_limit,
            dosage_buffer_limit: queue_limits.dosage_buffer_limit,
        }
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn intersect_committed_chunk_identifier_sets(
    committed_chunk_identifier_sets: Vec<Vec<usize>>,
) -> Vec<usize> {
    let native_committed_chunk_identifier_sets: Vec<BTreeSet<usize>> = committed_chunk_identifier_sets
        .into_iter()
        .map(|chunk_identifiers| chunk_identifiers.into_iter().collect())
        .collect();
    native_schedule::intersect_committed_chunk_identifier_sets(&native_committed_chunk_identifier_sets)
        .into_iter()
        .collect()
}

#[pyfunction]
pub(crate) fn resolve_delivery_callback_batch_size(
    callback_batch_size: Option<i64>,
    variant_major_packed8_probability_pairs: bool,
) -> PyResult<usize> {
    native_schedule::resolve_delivery_callback_batch_size(callback_batch_size, variant_major_packed8_probability_pairs)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn resolve_grouped_union_callback_batch_size(native_callback_batch_size: i64) -> PyResult<usize> {
    native_schedule::resolve_grouped_union_callback_batch_size(native_callback_batch_size)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn resolve_native_callback_queue_limits(
    staging_depth: i64,
    native_callback_batch_size: i64,
    result_in_flight_limit: Option<i64>,
    dosage_buffer_limit: Option<i64>,
) -> PyResult<NativeCallbackQueueLimits> {
    native_schedule::resolve_native_callback_queue_limits(
        staging_depth,
        native_callback_batch_size,
        result_in_flight_limit,
        dosage_buffer_limit,
    )
    .map(Into::into)
    .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn resolve_bgen_delivery_method_value(
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
) -> String {
    native_schedule::resolve_bgen_delivery_method(
        variant_major_packed8_probability_pairs,
        has_native_multi_aligned_sample_data,
        has_native_aligned_sample_data,
    )
    .as_value()
    .to_string()
}

#[pyfunction]
pub(crate) fn resolve_writer_finish_thread_count(
    writer_session_count: i64,
    requested_thread_count: i64,
) -> PyResult<usize> {
    native_schedule::resolve_writer_finish_thread_count(writer_session_count, requested_thread_count)
        .map_err(|error| schedule_error_to_py(&error))
}

fn schedule_error_to_py(error: &native_schedule::ScheduleError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
