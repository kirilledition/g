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

#[pyclass]
pub(crate) struct NativeDosageBufferReusePlan {
    #[pyo3(get)]
    requires_slice: bool,
    #[pyo3(get)]
    slice_dimensions: Vec<usize>,
}

#[pyclass]
pub(crate) struct NativeVariantMajorDosageBatchHandoffPlan {
    #[pyo3(get)]
    chunk_count: usize,
}

#[pyclass]
pub(crate) struct NativeDosageBufferPoolState {
    inner: native_schedule::DosageBufferPoolState,
}

#[pyclass]
pub(crate) struct NativeResultInFlightSlotState {
    inner: native_schedule::ResultInFlightSlotState,
}

#[pymethods]
impl NativeDosageBufferPoolState {
    #[new]
    fn new(buffer_limit: usize) -> Self {
        Self { inner: native_schedule::DosageBufferPoolState::new(buffer_limit) }
    }

    #[getter]
    fn buffer_limit(&self) -> usize {
        self.inner.buffer_limit()
    }

    #[getter]
    fn allocated_count(&self) -> usize {
        self.inner.allocated_count()
    }

    #[getter]
    fn buffer_identifiers(&self) -> Vec<usize> {
        self.inner.buffer_identifiers()
    }

    fn has_available_slot(&self) -> bool {
        self.inner.has_available_slot()
    }

    fn owns_buffer(&self, buffer_identifier: usize) -> bool {
        self.inner.owns_buffer(buffer_identifier)
    }

    fn register_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.inner.register_buffer(buffer_identifier)
    }

    fn discard_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.inner.discard_buffer(buffer_identifier)
    }
}

#[pymethods]
impl NativeResultInFlightSlotState {
    #[new]
    fn new(slot_limit: usize) -> Self {
        Self { inner: native_schedule::ResultInFlightSlotState::new(slot_limit) }
    }

    #[getter]
    fn slot_limit(&self) -> usize {
        self.inner.slot_limit()
    }

    #[getter]
    fn occupied_count(&self) -> usize {
        self.inner.occupied_count()
    }

    fn has_available_slot(&self) -> bool {
        self.inner.has_available_slot()
    }

    fn acquire_slot(&mut self) -> bool {
        self.inner.acquire_slot()
    }

    fn release_slot(&mut self) -> bool {
        self.inner.release_slot()
    }
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

impl From<native_schedule::DosageBufferReusePlan> for NativeDosageBufferReusePlan {
    fn from(reuse_plan: native_schedule::DosageBufferReusePlan) -> Self {
        Self { requires_slice: reuse_plan.requires_slice, slice_dimensions: reuse_plan.slice_dimensions }
    }
}

impl From<native_schedule::VariantMajorDosageBatchHandoffPlan> for NativeVariantMajorDosageBatchHandoffPlan {
    fn from(batch_handoff_plan: native_schedule::VariantMajorDosageBatchHandoffPlan) -> Self {
        Self { chunk_count: batch_handoff_plan.chunk_count }
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
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_dosage_buffer_reuse(
    buffered_shape: Vec<usize>,
    expected_shape: Vec<usize>,
) -> Option<NativeDosageBufferReusePlan> {
    native_schedule::plan_dosage_buffer_reuse(&buffered_shape, &expected_shape).map(Into::into)
}

#[pyfunction]
pub(crate) fn plan_variant_major_dosage_batch_handoff(
    metadata_count: usize,
    genotype_matrix_by_variant_count: usize,
    chunk_stats_count: usize,
) -> PyResult<NativeVariantMajorDosageBatchHandoffPlan> {
    native_schedule::plan_variant_major_dosage_batch_handoff(
        metadata_count,
        genotype_matrix_by_variant_count,
        chunk_stats_count,
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
