//! PyO3 adapters for engine scheduling policy helpers.

use std::collections::BTreeSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use g_engine::schedule as native_schedule;

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
