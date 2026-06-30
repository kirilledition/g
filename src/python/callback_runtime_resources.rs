//! PyO3 owner for callback runtime native resources.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PySlice, PyTuple};

use super::callback_progress::{
    NativeCallbackChunkIdentity, NativeCallbackProgressCompletion, NativeCallbackProgressState,
    NativeCallbackProgressTelemetryEvent, NativeCallbackProgressUpdate, build_callback_chunk_identity,
};
use super::callback_queue::{
    NativeCallbackObjectQueue, NativeCallbackObjectQueueGetResult, NativeCallbackWaitSignal, NativeCallbackWorkerThread,
};
use super::callback_summary::{
    NativeBinaryCorrectionDiagnosticsRecordPlan, NativeBinaryCorrectionSummary, NativeBinaryCorrectionSummaryEmitPlan,
};
use super::schedule::{
    NativeCallbackQueueBackpressureObservation, NativeCallbackQueueGetObservationPlan,
    NativeCallbackQueuePutObservationPlan, NativeCallbackQueueStageBackpressureObservation,
    NativeCallbackSchedulerState, NativeCallbackWorkerAbortPlan, NativeCallbackWorkerErrorRaisePlan,
    NativeCallbackWorkerErrorUpdatePlan, NativeCallbackWorkerFinishPlan, NativeCallbackWorkerStartAttemptPlan,
    NativeDosageBufferPoolObservationPlan, NativeDosageBufferReturnAttemptPlan, NativeDosageBufferReusePlan,
    NativeDosageWorkDrainCompletionPlan, NativeDosageWorkHandoffPlan, NativeDosageWorkItemDispatchPlan,
    NativeDosageWorkItemStageDurationPlan, NativeResultInFlightAcquireObservationPlan,
    NativeResultInFlightReleaseObservationPlan, NativeResultWriteDrainCompletionPlan, NativeResultWriteHandoffPlan,
    NativeResultWriteItemDispatchPlan, NativeResultWriteItemResourceReleasePlan,
    NativeVariantMajorDosageBatchHandoffPlan,
};

const RESULT_WRITE_ITEM_KIND_SINGLE_RESULT: &str = "single_result";
const RESULT_WRITE_ITEM_KIND_MULTI_RESULT: &str = "multi_result";
const RESULT_WRITE_ITEM_KIND_STOP_SIGNAL: &str = "stop_signal";
const DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE: &str = "sample_major_dosage";
const DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE: &str = "variant_major_dosage";
const DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH: &str = "variant_major_dosage_batch";
const DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR: &str = "variant_major_packed8_probability_pair";
const DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL: &str = "stop_signal";

fn pending_diagnostics_count_from_object(pending_diagnostics: &Bound<'_, PyAny>) -> PyResult<i64> {
    let pending_diagnostics_count = pending_diagnostics.len()?;
    i64::try_from(pending_diagnostics_count).map_err(|_| {
        PyRuntimeError::new_err("Pending binary correction diagnostics count exceeds native summary capacity.")
    })
}

fn metadata_chromosome_value(metadata: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(chromosome_label) = metadata.getattr("chromosome_label")
        && !chromosome_label.is_none()
    {
        return Ok(chromosome_label.str()?.to_string_lossy().into_owned());
    }
    let chromosome_values = metadata.getattr("chromosome")?;
    let chromosome_value = chromosome_values.get_item(0)?;
    Ok(chromosome_value.str()?.to_string_lossy().into_owned())
}

fn callback_chunk_identity_from_metadata(metadata: &Bound<'_, PyAny>) -> PyResult<NativeCallbackChunkIdentity> {
    let chromosome = metadata_chromosome_value(metadata)?;
    let variant_start_index = metadata.getattr("variant_start_index")?.extract::<i64>()?;
    let variant_stop_index = metadata.getattr("variant_stop_index")?.extract::<i64>()?;
    Ok(build_callback_chunk_identity(chromosome, variant_start_index, variant_stop_index))
}

#[pyclass]
pub(crate) struct NativeCallbackRuntimeResources {
    callback_scheduler_state: Py<NativeCallbackSchedulerState>,
    progress_state: Py<NativeCallbackProgressState>,
    result_in_flight_slot_signal: Py<NativeCallbackWaitSignal>,
    dosage_buffer_pool_signal: Py<NativeCallbackWaitSignal>,
    dosage_queue: Py<NativeCallbackObjectQueue>,
    result_queue: Py<NativeCallbackObjectQueue>,
    free_dosage_buffers: Py<NativeCallbackObjectQueue>,
    binary_correction_summary: Py<NativeBinaryCorrectionSummary>,
    worker_thread: Py<NativeCallbackWorkerThread>,
    result_worker_thread: Py<NativeCallbackWorkerThread>,
    expected_result_work_item_kind: String,
    has_telemetry_session: bool,
    has_stage_timing_recorder: bool,
    flush_binary_correction_diagnostics_on_result_stop: bool,
    worker_start_lock: Mutex<()>,
}

#[pyclass]
pub(crate) struct NativeDosageBufferAcquireResult {
    dosage_buffer: Option<Py<PyAny>>,
    should_allocate: bool,
    free_buffer_count: usize,
    waited: bool,
    observation_plan: Option<Py<NativeDosageBufferPoolObservationPlan>>,
}

#[pyclass]
pub(crate) struct NativeDosageBufferReuseSelectionResult {
    dosage_buffer: Option<Py<PyAny>>,
    operation_result: Py<NativeDosageBufferPoolOperationResult>,
}

#[pyclass]
pub(crate) struct NativeDosageBufferPoolOperationResult {
    free_buffer_count: Option<usize>,
    observation_plan: Option<Py<NativeDosageBufferPoolObservationPlan>>,
}

#[pyclass]
pub(crate) struct NativeResultInFlightAcquireResult {
    should_retry_acquisition: bool,
    observation_plan: Option<Py<NativeResultInFlightAcquireObservationPlan>>,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerFinishLifecycleResult {
    finish_plan: NativeCallbackWorkerFinishPlan,
    shutdown_worker_name: Option<String>,
    shutdown_timeout_seconds: Option<f64>,
    progress_completion_event: Option<NativeCallbackProgressTelemetryEvent>,
    flush_binary_correction_pending_diagnostics: bool,
    binary_correction_summary_payload: Option<Py<PyDict>>,
}

#[pyclass]
pub(crate) struct NativeCallbackQueuePutResult {
    should_retry_put: bool,
    observation_plan: Option<Py<NativeCallbackQueuePutObservationPlan>>,
}

#[pyclass]
pub(crate) struct NativeResultWorkItemResourceReleaseResult {
    released_host_buffer: bool,
    free_buffer_count: Option<usize>,
    dosage_buffer_pool_observation_plan: Option<Py<NativeDosageBufferPoolObservationPlan>>,
    released_result_in_flight_slot: bool,
    result_in_flight_observation_plan: Option<Py<NativeResultInFlightReleaseObservationPlan>>,
}

#[pyclass]
pub(crate) struct NativeCallbackQueueGetObservedResult {
    item: Option<Py<PyAny>>,
    observation_plan: Py<NativeCallbackQueueGetObservationPlan>,
}

#[pyclass]
pub(crate) struct NativeDosageWorkItemDrainResult {
    item: Option<Py<PyAny>>,
    has_dosage_work_item: bool,
    drain_completion_plan: Py<NativeDosageWorkDrainCompletionPlan>,
}

#[pyclass]
pub(crate) struct NativeDosageWorkItemGetResult {
    item: Option<Py<PyAny>>,
    has_dosage_work_item: bool,
    observation_plan: Option<Py<NativeCallbackQueueGetObservationPlan>>,
    drain_completion_plan: Py<NativeDosageWorkDrainCompletionPlan>,
    dispatch_plan: Option<Py<NativeDosageWorkItemDispatchPlan>>,
}

#[pyclass]
pub(crate) struct NativeResultWriteItemDrainResult {
    item: Option<Py<PyAny>>,
    has_result_work_item: bool,
    drain_completion_plan: Py<NativeResultWriteDrainCompletionPlan>,
}

#[pyclass]
pub(crate) struct NativeResultWriteItemGetResult {
    item: Option<Py<PyAny>>,
    has_result_work_item: bool,
    observation_plan: Option<Py<NativeCallbackQueueGetObservationPlan>>,
    drain_completion_plan: Py<NativeResultWriteDrainCompletionPlan>,
    dispatch_plan: Option<Py<NativeResultWriteItemDispatchPlan>>,
}

#[pyclass]
pub(crate) struct NativeDosageWorkItemStageDurationAttribution {
    metadata_items: Py<PyTuple>,
    stage_duration_plan: Py<NativeDosageWorkItemStageDurationPlan>,
}

#[pymethods]
impl NativeCallbackRuntimeResources {
    #[new]
    #[pyo3(signature = (
        *,
        worker_name,
        dosage_worker_target,
        result_worker_target,
        staging_depth,
        native_callback_batch_size,
        expected_result_work_item_kind,
        has_telemetry_session,
        flush_binary_correction_diagnostics_on_result_stop,
        has_stage_timing_recorder = false,
        result_in_flight_limit = None,
        dosage_buffer_limit = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        worker_name: String,
        dosage_worker_target: &Bound<'_, PyAny>,
        result_worker_target: &Bound<'_, PyAny>,
        staging_depth: i64,
        native_callback_batch_size: i64,
        expected_result_work_item_kind: String,
        has_telemetry_session: bool,
        flush_binary_correction_diagnostics_on_result_stop: bool,
        has_stage_timing_recorder: bool,
        result_in_flight_limit: Option<i64>,
        dosage_buffer_limit: Option<i64>,
    ) -> PyResult<Self> {
        let callback_scheduler_state = NativeCallbackSchedulerState::from_limits(
            staging_depth,
            native_callback_batch_size,
            result_in_flight_limit,
            dosage_buffer_limit,
        )?;
        let expected_result_dispatch_plan = callback_scheduler_state
            .plan_result_write_item_dispatch_value(&expected_result_work_item_kind, &expected_result_work_item_kind)?;
        if expected_result_dispatch_plan.has_dispatch_error_value() {
            let error_message = expected_result_dispatch_plan
                .error_message_value()
                .unwrap_or("Native result write dispatch plan omitted the error message.");
            return Err(PyRuntimeError::new_err(error_message.to_owned()));
        }
        let dosage_queue_depth = callback_scheduler_state.dosage_queue_depth_value();
        let result_queue_depth = callback_scheduler_state.result_queue_depth_value();
        let dosage_buffer_limit = callback_scheduler_state.dosage_buffer_limit_value();
        let result_worker_name = format!("{worker_name}-writer");

        Ok(Self {
            callback_scheduler_state: Py::new(py, callback_scheduler_state)?,
            progress_state: Py::new(py, NativeCallbackProgressState::new_state())?,
            result_in_flight_slot_signal: Py::new(py, NativeCallbackWaitSignal::new_signal())?,
            dosage_buffer_pool_signal: Py::new(py, NativeCallbackWaitSignal::new_signal())?,
            dosage_queue: Py::new(py, NativeCallbackObjectQueue::with_capacity(dosage_queue_depth)?)?,
            result_queue: Py::new(py, NativeCallbackObjectQueue::with_capacity(result_queue_depth)?)?,
            free_dosage_buffers: Py::new(py, NativeCallbackObjectQueue::with_capacity(dosage_buffer_limit)?)?,
            binary_correction_summary: Py::new(py, NativeBinaryCorrectionSummary::new_summary())?,
            worker_thread: Py::new(
                py,
                NativeCallbackWorkerThread::from_target(py, dosage_worker_target, worker_name, true)?,
            )?,
            result_worker_thread: Py::new(
                py,
                NativeCallbackWorkerThread::from_target(py, result_worker_target, result_worker_name, true)?,
            )?,
            expected_result_work_item_kind,
            has_telemetry_session,
            has_stage_timing_recorder,
            flush_binary_correction_diagnostics_on_result_stop,
            worker_start_lock: Mutex::new(()),
        })
    }

    #[getter]
    fn callback_scheduler_state(&self, py: Python<'_>) -> Py<NativeCallbackSchedulerState> {
        self.callback_scheduler_state.clone_ref(py)
    }

    #[getter]
    fn progress_state(&self, py: Python<'_>) -> Py<NativeCallbackProgressState> {
        self.progress_state.clone_ref(py)
    }

    #[getter]
    fn result_in_flight_slot_signal(&self, py: Python<'_>) -> Py<NativeCallbackWaitSignal> {
        self.result_in_flight_slot_signal.clone_ref(py)
    }

    #[getter]
    fn dosage_buffer_pool_signal(&self, py: Python<'_>) -> Py<NativeCallbackWaitSignal> {
        self.dosage_buffer_pool_signal.clone_ref(py)
    }

    #[getter]
    fn dosage_queue(&self, py: Python<'_>) -> Py<NativeCallbackObjectQueue> {
        self.dosage_queue.clone_ref(py)
    }

    #[getter]
    fn result_queue(&self, py: Python<'_>) -> Py<NativeCallbackObjectQueue> {
        self.result_queue.clone_ref(py)
    }

    #[getter]
    fn free_dosage_buffers(&self, py: Python<'_>) -> Py<NativeCallbackObjectQueue> {
        self.free_dosage_buffers.clone_ref(py)
    }

    #[getter]
    fn binary_correction_summary(&self, py: Python<'_>) -> Py<NativeBinaryCorrectionSummary> {
        self.binary_correction_summary.clone_ref(py)
    }

    #[getter]
    fn worker_thread(&self, py: Python<'_>) -> Py<NativeCallbackWorkerThread> {
        self.worker_thread.clone_ref(py)
    }

    #[getter]
    fn result_worker_thread(&self, py: Python<'_>) -> Py<NativeCallbackWorkerThread> {
        self.result_worker_thread.clone_ref(py)
    }

    #[getter]
    fn dosage_worker_name(&self, py: Python<'_>) -> String {
        self.worker_thread.bind(py).borrow().name_value().to_owned()
    }

    #[getter]
    fn result_worker_name(&self, py: Python<'_>) -> String {
        self.result_worker_thread.bind(py).borrow().name_value().to_owned()
    }

    #[getter]
    fn dosage_worker_is_alive(&self, py: Python<'_>) -> PyResult<bool> {
        self.worker_thread.bind(py).borrow().is_thread_alive(py)
    }

    #[getter]
    fn result_worker_is_alive(&self, py: Python<'_>) -> PyResult<bool> {
        self.result_worker_thread.bind(py).borrow().is_thread_alive(py)
    }

    #[getter]
    fn has_started(&self, py: Python<'_>) -> bool {
        self.callback_scheduler_state.bind(py).borrow().has_started_value()
    }

    #[getter]
    fn native_callback_batch_size(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().native_callback_batch_size_value()
    }

    #[getter]
    fn dosage_queue_depth(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().dosage_queue_depth_value()
    }

    #[getter]
    fn result_queue_depth(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().result_queue_depth_value()
    }

    #[getter]
    fn result_in_flight_limit(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().result_in_flight_limit_value()
    }

    #[getter]
    fn dosage_buffer_limit(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().dosage_buffer_limit_value()
    }

    #[getter]
    fn dosage_queue_occupied_count(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().dosage_queue_occupied_count_value()
    }

    #[getter]
    fn result_queue_occupied_count(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().result_queue_occupied_count_value()
    }

    #[getter]
    fn result_in_flight_occupied_count(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().result_in_flight_occupied_count_value()
    }

    #[getter]
    fn dosage_buffer_allocated_count(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().dosage_buffer_allocated_count_value()
    }

    #[getter]
    fn free_dosage_buffer_count(&self, py: Python<'_>) -> PyResult<usize> {
        self.free_dosage_buffers.bind(py).borrow().occupied_count_value()
    }

    #[getter]
    fn dosage_buffer_identifiers(&self, py: Python<'_>) -> Vec<usize> {
        self.callback_scheduler_state.bind(py).borrow().dosage_buffer_identifiers_value()
    }

    #[getter]
    fn processed_chunk_count(&self, py: Python<'_>) -> i64 {
        self.progress_state.bind(py).borrow().processed_chunk_count_value()
    }

    #[getter]
    fn current_progress_chromosome(&self, py: Python<'_>) -> Option<String> {
        self.progress_state.bind(py).borrow().current_progress_chromosome_value()
    }

    fn record_processed_chunk(
        &self,
        py: Python<'_>,
        chunk_identity: &NativeCallbackChunkIdentity,
    ) -> NativeCallbackProgressUpdate {
        self.progress_state.bind(py).borrow_mut().record_processed_chunk_value(chunk_identity)
    }

    fn record_processed_chunk_for_metadata(
        &self,
        py: Python<'_>,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<NativeCallbackProgressUpdate> {
        let chunk_identity = callback_chunk_identity_from_metadata(metadata)?;
        Ok(self.record_processed_chunk(py, &chunk_identity))
    }

    fn record_progress_for_metadata(
        &self,
        py: Python<'_>,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<Option<NativeCallbackProgressUpdate>> {
        if !self.has_telemetry_session {
            self.record_processed_chunk_without_progress(py);
            return Ok(None);
        }
        self.record_processed_chunk_for_metadata(py, metadata).map(Some)
    }

    fn record_processed_chunk_without_progress(&self, py: Python<'_>) {
        self.progress_state.bind(py).borrow_mut().record_processed_chunk_without_progress_value();
    }

    fn finish_progress(&self, py: Python<'_>) -> Option<NativeCallbackProgressCompletion> {
        self.progress_state.bind(py).borrow_mut().finish_progress_value()
    }

    fn binary_correction_chunk_count_with_pending(
        &self,
        py: Python<'_>,
        pending_diagnostics_count: i64,
    ) -> PyResult<i64> {
        self.binary_correction_summary.bind(py).borrow().chunk_count_with_pending_value(pending_diagnostics_count)
    }

    fn binary_correction_chunk_count_with_pending_diagnostics(
        &self,
        py: Python<'_>,
        pending_diagnostics: &Bound<'_, PyAny>,
    ) -> PyResult<i64> {
        let pending_diagnostics_count = pending_diagnostics_count_from_object(pending_diagnostics)?;
        self.binary_correction_chunk_count_with_pending(py, pending_diagnostics_count)
    }

    fn add_binary_null_model_failure_count(&self, py: Python<'_>, failure_count: i64) -> PyResult<()> {
        self.binary_correction_summary.bind(py).borrow().add_null_model_failure_count_value(failure_count)
    }

    fn plan_binary_correction_diagnostics_record(
        &self,
        py: Python<'_>,
        has_diagnostics: bool,
    ) -> PyResult<NativeBinaryCorrectionDiagnosticsRecordPlan> {
        self.binary_correction_summary
            .bind(py)
            .borrow()
            .plan_diagnostics_record_value(self.has_telemetry_session, has_diagnostics)
    }

    fn plan_binary_correction_diagnostics_record_for_object(
        &self,
        py: Python<'_>,
        binary_chunk_diagnostics: &Bound<'_, PyAny>,
    ) -> PyResult<NativeBinaryCorrectionDiagnosticsRecordPlan> {
        let has_diagnostics = !binary_chunk_diagnostics.is_none();
        self.plan_binary_correction_diagnostics_record(py, has_diagnostics)
    }

    fn plan_binary_correction_summary_emit(
        &self,
        py: Python<'_>,
        pending_diagnostics_count: i64,
    ) -> PyResult<NativeBinaryCorrectionSummaryEmitPlan> {
        self.binary_correction_summary
            .bind(py)
            .borrow()
            .plan_summary_emit_value(self.has_telemetry_session, pending_diagnostics_count)
    }

    fn plan_binary_correction_summary_emit_for_pending_diagnostics(
        &self,
        py: Python<'_>,
        pending_diagnostics: &Bound<'_, PyAny>,
    ) -> PyResult<NativeBinaryCorrectionSummaryEmitPlan> {
        let pending_diagnostics_count = pending_diagnostics_count_from_object(pending_diagnostics)?;
        self.plan_binary_correction_summary_emit(py, pending_diagnostics_count)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_binary_correction_diagnostics_totals(
        &self,
        py: Python<'_>,
        chunk_count: i64,
        score_only_count: i64,
        score_test_candidate_count: i64,
        firth_candidate_count: i64,
        firth_converged_count: i64,
        firth_failed_count: i64,
        firth_numerical_failure_count: i64,
        firth_max_iteration_failure_count: i64,
        firth_invalid_statistic_failure_count: i64,
        firth_step_halving_failure_count: i64,
        pseudo_firth_attempt_count: i64,
        pseudo_firth_success_count: i64,
        nr_zero_start_attempt_count: i64,
        nr_zero_start_success_count: i64,
        nr_warm_start_attempt_count: i64,
        nr_warm_start_success_count: i64,
        sparse_correction_count: i64,
        dense_correction_count: i64,
    ) -> PyResult<()> {
        self.binary_correction_summary.bind(py).borrow().add_diagnostics_totals_value(
            chunk_count,
            score_only_count,
            score_test_candidate_count,
            firth_candidate_count,
            firth_converged_count,
            firth_failed_count,
            firth_numerical_failure_count,
            firth_max_iteration_failure_count,
            firth_invalid_statistic_failure_count,
            firth_step_halving_failure_count,
            pseudo_firth_attempt_count,
            pseudo_firth_success_count,
            nr_zero_start_attempt_count,
            nr_zero_start_success_count,
            nr_warm_start_attempt_count,
            nr_warm_start_success_count,
            sparse_correction_count,
            dense_correction_count,
        )
    }

    fn binary_correction_summary_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.binary_correction_summary.bind(py).borrow().summary_payload_value(py)
    }

    fn start_workers(&self, py: Python<'_>) -> PyResult<NativeCallbackWorkerStartAttemptPlan> {
        let _start_guard = self.worker_start_lock.lock().map_err(|_| {
            PyRuntimeError::new_err("native callback worker start lock was poisoned during worker startup")
        })?;
        let start_attempt_plan = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.plan_worker_start_attempt_value()
        };
        if start_attempt_plan.has_start_error_value() {
            return Ok(start_attempt_plan);
        }
        if start_attempt_plan.should_start_result_worker() {
            self.result_worker_thread.bind(py).borrow().start_thread(py)?;
        }
        if start_attempt_plan.should_start_dosage_worker() {
            self.worker_thread.bind(py).borrow().start_thread(py)?;
        }
        Ok(start_attempt_plan)
    }

    fn stop_dosage_worker(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Option<f64>> {
        let is_worker_alive = self.worker_thread.bind(py).borrow().is_thread_alive(py)?;
        let stop_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_worker_stop_value(timeout_seconds, is_worker_alive)
        };
        if !stop_plan.should_stop_value() {
            return Ok(None);
        }
        let stop_deadline = Instant::now() + normalize_timeout_duration(stop_plan.timeout_seconds_value());
        while Instant::now() < stop_deadline {
            let remaining_seconds = remaining_timeout_seconds(stop_deadline);
            let is_worker_alive = self.worker_thread.bind(py).borrow().is_thread_alive(py)?;
            let stop_poll_plan = {
                let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
                scheduler_state.plan_dosage_worker_stop_poll_value(remaining_seconds, is_worker_alive)
            };
            if !stop_poll_plan.should_stop_value() {
                return Ok(None);
            }
            let stop_signal = py.None();
            if self.try_put_dosage_work_item(py, stop_signal.bind(py), stop_poll_plan.poll_timeout_seconds_value())? {
                return Ok(None);
            }
        }
        Ok(Some(stop_plan.timeout_seconds_value()))
    }

    fn join_dosage_worker(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Option<f64>> {
        let join_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_worker_join_value(timeout_seconds)
        };
        if !join_plan.should_join_value() {
            return Ok(None);
        }
        self.worker_thread.bind(py).borrow().join_thread(py, Some(join_plan.timeout_seconds_value()))?;
        if self.worker_thread.bind(py).borrow().is_thread_alive(py)? {
            return Ok(Some(join_plan.timeout_seconds_value()));
        }
        Ok(None)
    }

    fn stop_result_worker(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Option<f64>> {
        let is_worker_alive = self.result_worker_thread.bind(py).borrow().is_thread_alive(py)?;
        let stop_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_worker_stop_value(timeout_seconds, is_worker_alive)
        };
        if !stop_plan.should_stop_value() {
            return Ok(None);
        }
        let stop_deadline = Instant::now() + normalize_timeout_duration(stop_plan.timeout_seconds_value());
        while Instant::now() < stop_deadline {
            let remaining_seconds = remaining_timeout_seconds(stop_deadline);
            let is_worker_alive = self.result_worker_thread.bind(py).borrow().is_thread_alive(py)?;
            let stop_poll_plan = {
                let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
                scheduler_state.plan_result_worker_stop_poll_value(remaining_seconds, is_worker_alive)
            };
            if !stop_poll_plan.should_stop_value() {
                return Ok(None);
            }
            let stop_signal = py.None();
            if self.try_put_result_write_item(py, stop_signal.bind(py), stop_poll_plan.poll_timeout_seconds_value())? {
                return Ok(None);
            }
        }
        Ok(Some(stop_plan.timeout_seconds_value()))
    }

    fn join_result_worker(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Option<f64>> {
        let join_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_worker_join_value(timeout_seconds)
        };
        if !join_plan.should_join_value() {
            return Ok(None);
        }
        self.result_worker_thread.bind(py).borrow().join_thread(py, Some(join_plan.timeout_seconds_value()))?;
        if self.result_worker_thread.bind(py).borrow().is_thread_alive(py)? {
            return Ok(Some(join_plan.timeout_seconds_value()));
        }
        Ok(None)
    }

    fn finish_worker_lifecycle(
        &self,
        py: Python<'_>,
        pending_diagnostics_count: i64,
    ) -> PyResult<NativeCallbackWorkerFinishLifecycleResult> {
        let finish_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_worker_finish_value()
        };
        let mut finish_result = NativeCallbackWorkerFinishLifecycleResult::from_finish_plan(&finish_plan);
        if finish_plan.stop_dosage_worker_value() {
            let timeout_seconds = self.stop_dosage_worker(py, Some(finish_plan.dosage_stop_timeout_seconds_value()))?;
            if let Some(timeout_seconds) = timeout_seconds {
                finish_result.record_shutdown_timeout(
                    self.worker_thread.bind(py).borrow().name_value().to_owned(),
                    timeout_seconds,
                );
                return Ok(finish_result);
            }
        }
        if finish_plan.join_dosage_worker_value() {
            let timeout_seconds = self.join_dosage_worker(py, Some(finish_plan.dosage_join_timeout_seconds_value()))?;
            if let Some(timeout_seconds) = timeout_seconds {
                finish_result.record_shutdown_timeout(
                    self.worker_thread.bind(py).borrow().name_value().to_owned(),
                    timeout_seconds,
                );
                return Ok(finish_result);
            }
        }
        if finish_plan.stop_result_worker_value() {
            let timeout_seconds = self.stop_result_worker(py, Some(finish_plan.result_stop_timeout_seconds_value()))?;
            if let Some(timeout_seconds) = timeout_seconds {
                finish_result.record_shutdown_timeout(
                    self.result_worker_thread.bind(py).borrow().name_value().to_owned(),
                    timeout_seconds,
                );
                return Ok(finish_result);
            }
        }
        if finish_plan.join_result_worker_value() {
            let timeout_seconds = self.join_result_worker(py, Some(finish_plan.result_join_timeout_seconds_value()))?;
            if let Some(timeout_seconds) = timeout_seconds {
                finish_result.record_shutdown_timeout(
                    self.result_worker_thread.bind(py).borrow().name_value().to_owned(),
                    timeout_seconds,
                );
                return Ok(finish_result);
            }
        }
        if finish_plan.complete_progress_value() {
            let progress_completion = self.progress_state.bind(py).borrow_mut().finish_progress_value();
            finish_result.record_progress_completion(progress_completion);
        }
        if finish_plan.emit_binary_correction_summary_value() {
            let summary_emit_plan = self
                .binary_correction_summary
                .bind(py)
                .borrow()
                .plan_summary_emit_value(self.has_telemetry_session, pending_diagnostics_count)?;
            finish_result.record_binary_correction_pending_diagnostics_flush(
                summary_emit_plan.should_flush_pending_diagnostics_value(),
            );
            if summary_emit_plan.should_emit_summary_value()
                && !summary_emit_plan.should_flush_pending_diagnostics_value()
            {
                let summary_payload = self.binary_correction_summary.bind(py).borrow().summary_payload_value(py)?;
                finish_result.record_binary_correction_summary_payload(summary_payload.unbind());
            }
        }
        Ok(finish_result)
    }

    fn finish_worker_lifecycle_for_pending_diagnostics(
        &self,
        py: Python<'_>,
        pending_diagnostics: &Bound<'_, PyAny>,
    ) -> PyResult<NativeCallbackWorkerFinishLifecycleResult> {
        let pending_diagnostics_count = pending_diagnostics_count_from_object(pending_diagnostics)?;
        self.finish_worker_lifecycle(py, pending_diagnostics_count)
    }

    fn abort_worker_lifecycle(&self, py: Python<'_>) -> PyResult<NativeCallbackWorkerAbortPlan> {
        let abort_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_worker_abort_value()
        };
        if abort_plan.stop_dosage_worker_value() {
            let _ = self.stop_dosage_worker(py, Some(abort_plan.dosage_stop_timeout_seconds_value()))?;
        }
        if abort_plan.stop_result_worker_value() {
            let _ = self.stop_result_worker(py, Some(abort_plan.result_stop_timeout_seconds_value()))?;
        }
        Ok(abort_plan)
    }

    fn plan_worker_error_raise(&self, py: Python<'_>) -> NativeCallbackWorkerErrorRaisePlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_worker_error_raise_value()
    }

    fn update_dosage_worker_error(
        &self,
        py: Python<'_>,
        error_message: Option<&str>,
    ) -> NativeCallbackWorkerErrorUpdatePlan {
        let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
        scheduler_state.update_dosage_worker_error_value(error_message)
    }

    fn update_result_worker_error(
        &self,
        py: Python<'_>,
        error_message: Option<&str>,
    ) -> NativeCallbackWorkerErrorUpdatePlan {
        let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
        scheduler_state.update_result_worker_error_value(error_message)
    }

    fn acquire_result_in_flight_slot_with_backpressure_timeout(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeResultInFlightAcquireObservationPlan> {
        let observed_generation = self.result_in_flight_slot_signal.bind(py).borrow().generation_value()?;
        let (attempt_plan, observation_plan) = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            let attempt_plan = scheduler_state.plan_result_in_flight_slot_acquire_backpressure_attempt_value();
            let observation_plan = scheduler_state.plan_result_in_flight_slot_acquire_observation_value(&attempt_plan);
            (attempt_plan, observation_plan)
        };
        if !attempt_plan.should_acquire_value() && attempt_plan.should_wait_value() {
            self.result_in_flight_slot_signal.bind(py).borrow().wait_for_change_value(
                py,
                observed_generation,
                attempt_plan.wait_timeout_seconds_value(),
            )?;
        }
        Ok(observation_plan)
    }

    fn acquire_result_in_flight_slot_with_backpressure_timeout_without_observation(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeResultInFlightAcquireResult> {
        let observed_generation = self.result_in_flight_slot_signal.bind(py).borrow().generation_value()?;
        let attempt_plan = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.plan_result_in_flight_slot_acquire_backpressure_attempt_value()
        };
        if !attempt_plan.should_acquire_value() && attempt_plan.should_wait_value() {
            self.result_in_flight_slot_signal.bind(py).borrow().wait_for_change_value(
                py,
                observed_generation,
                attempt_plan.wait_timeout_seconds_value(),
            )?;
        }
        NativeResultInFlightAcquireResult::from_acquire(py, !attempt_plan.should_acquire_value(), None)
    }

    fn acquire_result_in_flight_slot_with_optional_observation(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeResultInFlightAcquireResult> {
        let observed_generation = self.result_in_flight_slot_signal.bind(py).borrow().generation_value()?;
        let (attempt_plan, observation_plan) = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            let attempt_plan = scheduler_state.plan_result_in_flight_slot_acquire_backpressure_attempt_value();
            let observation_plan = if self.has_stage_timing_recorder {
                Some(scheduler_state.plan_result_in_flight_slot_acquire_observation_value(&attempt_plan))
            } else {
                None
            };
            (attempt_plan, observation_plan)
        };
        if !attempt_plan.should_acquire_value() && attempt_plan.should_wait_value() {
            self.result_in_flight_slot_signal.bind(py).borrow().wait_for_change_value(
                py,
                observed_generation,
                attempt_plan.wait_timeout_seconds_value(),
            )?;
        }
        NativeResultInFlightAcquireResult::from_acquire(py, !attempt_plan.should_acquire_value(), observation_plan)
    }

    fn release_result_in_flight_slot(&self, py: Python<'_>) -> PyResult<NativeResultInFlightReleaseObservationPlan> {
        self.release_result_in_flight_slot_without_observation(py)?;
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        Ok(scheduler_state.plan_result_in_flight_slot_release_observation_value())
    }

    fn release_result_in_flight_slot_with_optional_observation(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<NativeResultInFlightReleaseObservationPlan>> {
        self.release_result_in_flight_slot_without_observation(py)?;
        if !self.has_stage_timing_recorder {
            return Ok(None);
        }
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        Ok(Some(scheduler_state.plan_result_in_flight_slot_release_observation_value()))
    }

    fn release_result_work_item_pre_write_resources(
        &self,
        py: Python<'_>,
        host_dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeResultWorkItemResourceReleaseResult> {
        let has_host_dosage_buffer = !host_dosage_buffer.is_none();
        let resource_release_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_write_item_pre_write_resource_release_value(has_host_dosage_buffer)
        };
        self.release_result_work_item_resources_with_plan(py, &resource_release_plan, host_dosage_buffer)
    }

    fn release_result_work_item_pre_write_resources_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeResultWorkItemResourceReleaseResult> {
        let host_dosage_buffer = result_work_item_host_dosage_buffer_owner(py, work_item)?;
        self.release_result_work_item_pre_write_resources(py, host_dosage_buffer.bind(py))
    }

    fn release_result_work_item_final_resources(
        &self,
        py: Python<'_>,
        host_dosage_buffer: &Bound<'_, PyAny>,
        has_released_host_dosage_buffer: bool,
        release_in_flight_slot: bool,
    ) -> PyResult<NativeResultWorkItemResourceReleaseResult> {
        let has_host_dosage_buffer = !host_dosage_buffer.is_none();
        let resource_release_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_write_item_final_resource_release_value(
                has_host_dosage_buffer,
                has_released_host_dosage_buffer,
                release_in_flight_slot,
            )
        };
        self.release_result_work_item_resources_with_plan(py, &resource_release_plan, host_dosage_buffer)
    }

    fn release_result_work_item_final_resources_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
        has_released_host_dosage_buffer: bool,
    ) -> PyResult<NativeResultWorkItemResourceReleaseResult> {
        let host_dosage_buffer = result_work_item_host_dosage_buffer_owner(py, work_item)?;
        let release_in_flight_slot = result_work_item_release_in_flight_slot(work_item)?;
        self.release_result_work_item_final_resources(
            py,
            host_dosage_buffer.bind(py),
            has_released_host_dosage_buffer,
            release_in_flight_slot,
        )
    }

    fn release_result_work_item_in_flight_slot_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeResultWorkItemResourceReleaseResult> {
        let mut release_result = NativeResultWorkItemResourceReleaseResult::empty();
        if !result_work_item_release_in_flight_slot(work_item)? {
            return Ok(release_result);
        }
        self.record_result_work_item_in_flight_slot_release(py, &mut release_result)?;
        Ok(release_result)
    }

    fn acquire_dosage_buffer_with_backpressure_timeout(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeDosageBufferAcquireResult> {
        let observed_generation = self.dosage_buffer_pool_signal.bind(py).borrow().generation_value()?;
        let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
        let acquire_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_acquire_backpressure_attempt_value(free_buffer_count)
        };
        if acquire_plan.should_take_free_buffer_value() {
            let get_result = self.free_dosage_buffers.bind(py).borrow().get_item(py, 0.0)?;
            if !get_result.has_item_value() {
                return Err(PyRuntimeError::new_err(
                    "Native dosage-buffer free queue was empty after scheduler selected reuse.",
                ));
            }
            let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
            return Ok(NativeDosageBufferAcquireResult {
                dosage_buffer: get_result.into_item_value(),
                should_allocate: false,
                free_buffer_count,
                waited: false,
                observation_plan: None,
            });
        }
        if acquire_plan.should_allocate_value() {
            let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
            return Ok(NativeDosageBufferAcquireResult {
                dosage_buffer: None,
                should_allocate: true,
                free_buffer_count,
                waited: false,
                observation_plan: None,
            });
        }
        if acquire_plan.should_wait_value() {
            self.dosage_buffer_pool_signal.bind(py).borrow().wait_for_change_value(
                py,
                observed_generation,
                acquire_plan.wait_timeout_seconds_value(),
            )?;
            let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
            let observation_plan = if self.has_stage_timing_recorder {
                let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
                Some(Py::new(py, scheduler_state.plan_dosage_buffer_pool_consumer_wait_observation_value())?)
            } else {
                None
            };
            return Ok(NativeDosageBufferAcquireResult {
                dosage_buffer: None,
                should_allocate: false,
                free_buffer_count,
                waited: true,
                observation_plan,
            });
        }
        let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
        Ok(NativeDosageBufferAcquireResult {
            dosage_buffer: None,
            should_allocate: false,
            free_buffer_count,
            waited: false,
            observation_plan: None,
        })
    }

    fn register_dosage_buffer(&self, py: Python<'_>, buffer_identifier: usize) -> PyResult<usize> {
        let register_plan = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.plan_dosage_buffer_register_attempt_value(buffer_identifier)
        };
        if register_plan.has_registration_error_value() {
            return Err(PyRuntimeError::new_err("Native dosage-buffer pool has no available slot for allocation."));
        }
        self.free_dosage_buffers.bind(py).borrow().occupied_count_value()
    }

    fn register_dosage_buffer_with_observation(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        let free_buffer_count = self.register_dosage_buffer(py, buffer_identifier)?;
        let observation_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_pool_allocate_observation_value()
        };
        NativeDosageBufferPoolOperationResult::from_operation(py, Some(free_buffer_count), Some(observation_plan))
    }

    fn register_dosage_buffer_with_optional_observation(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        let free_buffer_count = self.register_dosage_buffer(py, buffer_identifier)?;
        if !self.has_stage_timing_recorder {
            return NativeDosageBufferPoolOperationResult::from_operation(py, Some(free_buffer_count), None);
        }
        let observation_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_pool_allocate_observation_value()
        };
        NativeDosageBufferPoolOperationResult::from_operation(py, Some(free_buffer_count), Some(observation_plan))
    }

    fn register_dosage_buffer_object_with_optional_observation(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        self.register_dosage_buffer_with_optional_observation(py, py_object_identifier(dosage_buffer))
    }

    fn return_dosage_buffer(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<Option<usize>> {
        let return_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_return_attempt_value(buffer_identifier)
        };
        if !return_plan.should_return_value() {
            return Ok(None);
        }
        let queued = self.free_dosage_buffers.bind(py).borrow().put_item(py, dosage_buffer.clone().unbind(), 0.0)?;
        if !queued {
            return Err(PyRuntimeError::new_err("Native dosage-buffer free queue had no slot for returned buffer."));
        }
        let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
        self.dosage_buffer_pool_signal.bind(py).borrow().notify_waiters_value()?;
        Ok(Some(free_buffer_count))
    }

    fn return_dosage_buffer_with_observation(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        let free_buffer_count = self.return_dosage_buffer(py, buffer_identifier, dosage_buffer)?;
        if free_buffer_count.is_none() {
            return NativeDosageBufferPoolOperationResult::from_operation(py, None, None);
        }
        let observation_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_pool_return_observation_value()
        };
        NativeDosageBufferPoolOperationResult::from_operation(py, free_buffer_count, Some(observation_plan))
    }

    fn return_dosage_buffer_with_optional_observation(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        let free_buffer_count = self.return_dosage_buffer(py, buffer_identifier, dosage_buffer)?;
        if free_buffer_count.is_none() || !self.has_stage_timing_recorder {
            return NativeDosageBufferPoolOperationResult::from_operation(py, free_buffer_count, None);
        }
        let observation_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_pool_return_observation_value()
        };
        NativeDosageBufferPoolOperationResult::from_operation(py, free_buffer_count, Some(observation_plan))
    }

    fn return_dosage_buffer_object_with_optional_observation(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        self.return_dosage_buffer_with_optional_observation(py, py_object_identifier(dosage_buffer), dosage_buffer)
    }

    fn return_dosage_buffer_owner_with_optional_observation(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        let dosage_buffer_owner = dosage_buffer_owner(py, dosage_buffer)?;
        self.return_dosage_buffer_with_optional_observation(
            py,
            py_object_identifier(dosage_buffer_owner.bind(py)),
            dosage_buffer_owner.bind(py),
        )
    }

    fn release_numpy_dosage_buffer_with_optional_observation(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        if !is_numpy_ndarray(dosage_buffer)? {
            return NativeDosageBufferPoolOperationResult::from_operation(py, None, None);
        }
        self.return_dosage_buffer_owner_with_optional_observation(py, dosage_buffer)
    }

    fn discard_dosage_buffer(&self, py: Python<'_>, buffer_identifier: usize) -> PyResult<Option<usize>> {
        let discard_plan = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.plan_dosage_buffer_discard_attempt_value(buffer_identifier)
        };
        if !discard_plan.should_discard_value() {
            return Ok(None);
        }
        let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
        self.dosage_buffer_pool_signal.bind(py).borrow().notify_waiters_value()?;
        Ok(Some(free_buffer_count))
    }

    fn discard_dosage_buffer_with_observation(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        let free_buffer_count = self.discard_dosage_buffer(py, buffer_identifier)?;
        if free_buffer_count.is_none() {
            return NativeDosageBufferPoolOperationResult::from_operation(py, None, None);
        }
        let observation_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_pool_discard_observation_value()
        };
        NativeDosageBufferPoolOperationResult::from_operation(py, free_buffer_count, Some(observation_plan))
    }

    fn discard_dosage_buffer_with_optional_observation(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        let free_buffer_count = self.discard_dosage_buffer(py, buffer_identifier)?;
        if free_buffer_count.is_none() || !self.has_stage_timing_recorder {
            return NativeDosageBufferPoolOperationResult::from_operation(py, free_buffer_count, None);
        }
        let observation_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_pool_discard_observation_value()
        };
        NativeDosageBufferPoolOperationResult::from_operation(py, free_buffer_count, Some(observation_plan))
    }

    fn discard_dosage_buffer_object_with_optional_observation(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        self.discard_dosage_buffer_with_optional_observation(py, py_object_identifier(dosage_buffer))
    }

    fn discard_dosage_buffer_owner_with_optional_observation(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageBufferPoolOperationResult> {
        let dosage_buffer_owner = dosage_buffer_owner(py, dosage_buffer)?;
        self.discard_dosage_buffer_with_optional_observation(py, py_object_identifier(dosage_buffer_owner.bind(py)))
    }

    fn plan_dosage_buffer_return_attempt(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
    ) -> NativeDosageBufferReturnAttemptPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_return_attempt_value(buffer_identifier)
    }

    fn plan_dosage_buffer_object_return_attempt(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> NativeDosageBufferReturnAttemptPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_return_attempt_value(py_object_identifier(dosage_buffer))
    }

    fn get_releasable_dosage_buffer_owner(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if !is_numpy_ndarray(dosage_buffer)? {
            return Ok(None);
        }
        let dosage_buffer_owner = dosage_buffer_owner(py, dosage_buffer)?;
        let return_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_return_attempt_value(py_object_identifier(dosage_buffer_owner.bind(py)))
        };
        if return_plan.should_return_value() { Ok(Some(dosage_buffer_owner)) } else { Ok(None) }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_dosage_buffer_reuse(
        &self,
        py: Python<'_>,
        buffered_shape: Vec<usize>,
        expected_shape: Vec<usize>,
    ) -> Option<NativeDosageBufferReusePlan> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_reuse_value(&buffered_shape, &expected_shape)
    }

    fn get_reusable_dosage_buffer(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
        expected_shape: Vec<usize>,
        expected_dtype: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let dosage_buffer_dtype = dosage_buffer.getattr("dtype")?;
        if !dosage_buffer_dtype.eq(expected_dtype)? {
            return Ok(None);
        }
        let expected_shape = expected_shape.into_boxed_slice();
        let buffered_shape = dosage_buffer.getattr("shape")?.extract::<Vec<usize>>()?;
        let reuse_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_reuse_value(&buffered_shape, &expected_shape)
        };
        let Some(reuse_plan) = reuse_plan else {
            return Ok(None);
        };
        if !reuse_plan.requires_slice_value() {
            return Ok(Some(dosage_buffer.clone().unbind()));
        }
        let slice_tuple = dosage_buffer_reuse_slice_tuple(py, reuse_plan.slice_dimensions_value())?;
        Ok(Some(dosage_buffer.get_item(slice_tuple)?.unbind()))
    }

    fn select_reusable_dosage_buffer_or_discard(
        &self,
        py: Python<'_>,
        dosage_buffer: &Bound<'_, PyAny>,
        expected_shape: Vec<usize>,
        expected_dtype: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageBufferReuseSelectionResult> {
        let reusable_dosage_buffer =
            self.get_reusable_dosage_buffer(py, dosage_buffer, expected_shape, expected_dtype)?;
        if let Some(reusable_dosage_buffer) = reusable_dosage_buffer {
            let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
            let observation_plan = if self.has_stage_timing_recorder {
                let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
                Some(scheduler_state.plan_dosage_buffer_pool_reuse_observation_value())
            } else {
                None
            };
            let reuse_operation_result =
                NativeDosageBufferPoolOperationResult::from_operation(py, Some(free_buffer_count), observation_plan)?;
            return NativeDosageBufferReuseSelectionResult::from_reusable_dosage_buffer(
                py,
                reusable_dosage_buffer,
                reuse_operation_result,
            );
        }
        let discard_operation_result = self.discard_dosage_buffer_owner_with_optional_observation(py, dosage_buffer)?;
        NativeDosageBufferReuseSelectionResult::from_discard(py, discard_operation_result)
    }

    fn try_put_dosage_work_item(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
        timeout_seconds: f64,
    ) -> PyResult<bool> {
        let deadline = Instant::now() + normalize_timeout_duration(timeout_seconds);
        loop {
            let attempt_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                scheduler_state.plan_dosage_queue_put_attempt_value(remaining_timeout_seconds(deadline))
            };
            if attempt_plan.should_put_value() {
                return self.put_dosage_work_item_after_slot_acquisition(py, work_item);
            }
            if !attempt_plan.should_wait_value() {
                return Ok(false);
            }
            self.dosage_queue
                .bind(py)
                .borrow()
                .wait_for_available_slot_value(py, attempt_plan.wait_timeout_seconds_value())?;
        }
    }

    fn try_put_dosage_work_item_with_backpressure_timeout(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let mut deadline = None;
        loop {
            let attempt_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                if let Some(deadline) = deadline {
                    scheduler_state.plan_dosage_queue_put_attempt_value(remaining_timeout_seconds(deadline))
                } else {
                    let attempt_plan = scheduler_state.plan_dosage_queue_put_backpressure_attempt_value();
                    if attempt_plan.should_wait_value() {
                        deadline = Some(
                            Instant::now() + normalize_timeout_duration(attempt_plan.wait_timeout_seconds_value()),
                        );
                    }
                    attempt_plan
                }
            };
            if attempt_plan.should_put_value() {
                return self.put_dosage_work_item_after_slot_acquisition(py, work_item);
            }
            if !attempt_plan.should_wait_value() {
                return Ok(false);
            }
            self.dosage_queue
                .bind(py)
                .borrow()
                .wait_for_available_slot_value(py, attempt_plan.wait_timeout_seconds_value())?;
        }
    }

    fn put_dosage_work_item_with_backpressure_observation(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeCallbackQueuePutObservationPlan> {
        let queued = self.try_put_dosage_work_item_with_backpressure_timeout(py, work_item)?;
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        Ok(scheduler_state.plan_dosage_queue_put_observation_value(queued))
    }

    fn put_dosage_work_item_with_optional_backpressure_observation(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeCallbackQueuePutResult> {
        let queued = self.try_put_dosage_work_item_with_backpressure_timeout(py, work_item)?;
        let observation_plan = if self.has_stage_timing_recorder {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            Some(scheduler_state.plan_dosage_queue_put_observation_value(queued))
        } else {
            None
        };
        NativeCallbackQueuePutResult::from_put(py, !queued, observation_plan)
    }

    fn get_dosage_work_item(&self, py: Python<'_>) -> PyResult<NativeCallbackObjectQueueGetResult> {
        loop {
            let has_queued_item = self.dosage_queue.bind(py).borrow().has_queued_item_value()?;
            let get_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                scheduler_state.plan_dosage_queue_get_attempt_value(has_queued_item)
            };
            if get_plan.has_release_error_value() {
                return Err(PyRuntimeError::new_err("Native dosage-queue state has no occupied slot to release."));
            }
            if get_plan.should_get_value() {
                let get_result = self.dosage_queue.bind(py).borrow().get_item(py, 0.0)?;
                if !get_result.has_item_value() {
                    let reacquired_slot = {
                        let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                        scheduler_state.acquire_dosage_queue_slot_value()
                    };
                    if !reacquired_slot {
                        return Err(PyRuntimeError::new_err(
                            "Native dosage queue storage was empty after scheduler slot release.",
                        ));
                    }
                    return Err(PyRuntimeError::new_err(
                        "Native dosage queue storage had no queued item after scheduler selected get.",
                    ));
                }
                return Ok(get_result);
            }
            if get_plan.should_wait_value() {
                self.dosage_queue
                    .bind(py)
                    .borrow()
                    .wait_for_queued_item_value(py, get_plan.wait_timeout_seconds_value())?;
            }
        }
    }

    fn get_dosage_work_item_with_observation(&self, py: Python<'_>) -> PyResult<NativeCallbackQueueGetObservedResult> {
        let get_result = self.get_dosage_work_item(py)?;
        let observation_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_queue_get_observation_value()
        };
        NativeCallbackQueueGetObservedResult::from_get_result(py, get_result, observation_plan)
    }

    fn get_dosage_work_item_with_drain_completion(&self, py: Python<'_>) -> PyResult<NativeDosageWorkItemDrainResult> {
        let get_result = self.get_dosage_work_item(py)?;
        let has_dosage_work_item = get_result.has_non_none_item_value(py);
        let drain_completion_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_work_drain_completion_value(has_dosage_work_item)
        };
        NativeDosageWorkItemDrainResult::from_get_result(py, get_result, has_dosage_work_item, drain_completion_plan)
    }

    fn get_dosage_work_item_with_observation_and_drain_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeDosageWorkItemGetResult> {
        let get_result = self.get_dosage_work_item(py)?;
        let has_dosage_work_item = get_result.has_non_none_item_value(py);
        let (observation_plan, drain_completion_plan) = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            (
                scheduler_state.plan_dosage_queue_get_observation_value(),
                scheduler_state.plan_dosage_work_drain_completion_value(has_dosage_work_item),
            )
        };
        NativeDosageWorkItemGetResult::from_get_result(
            py,
            get_result,
            has_dosage_work_item,
            Some(observation_plan),
            drain_completion_plan,
            None,
        )
    }

    fn get_dosage_work_item_with_optional_observation_and_drain_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeDosageWorkItemGetResult> {
        let get_result = self.get_dosage_work_item(py)?;
        let has_dosage_work_item = get_result.has_non_none_item_value(py);
        let (observation_plan, drain_completion_plan) = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            let observation_plan = if self.has_stage_timing_recorder {
                Some(scheduler_state.plan_dosage_queue_get_observation_value())
            } else {
                None
            };
            (observation_plan, scheduler_state.plan_dosage_work_drain_completion_value(has_dosage_work_item))
        };
        NativeDosageWorkItemGetResult::from_get_result(
            py,
            get_result,
            has_dosage_work_item,
            observation_plan,
            drain_completion_plan,
            None,
        )
    }

    fn get_validated_dosage_work_item_with_drain_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeDosageWorkItemGetResult> {
        let get_result = self.get_dosage_work_item(py)?;
        let item = get_result.into_item_value();
        let has_dosage_work_item = item.as_ref().is_some_and(|queued_item| !queued_item.bind(py).is_none());
        let drain_completion_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_work_drain_completion_value(has_dosage_work_item)
        };
        let dispatch_plan = self.dosage_work_dispatch_plan_for_optional_item(py, item.as_ref())?;
        NativeDosageWorkItemGetResult::from_item(
            py,
            item,
            has_dosage_work_item,
            None,
            drain_completion_plan,
            dispatch_plan,
        )
    }

    fn get_validated_dosage_work_item_with_optional_observation_and_drain_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeDosageWorkItemGetResult> {
        let get_result = self.get_dosage_work_item(py)?;
        let item = get_result.into_item_value();
        let has_dosage_work_item = item.as_ref().is_some_and(|queued_item| !queued_item.bind(py).is_none());
        let (observation_plan, drain_completion_plan) = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            let observation_plan = if self.has_stage_timing_recorder {
                Some(scheduler_state.plan_dosage_queue_get_observation_value())
            } else {
                None
            };
            (observation_plan, scheduler_state.plan_dosage_work_drain_completion_value(has_dosage_work_item))
        };
        let dispatch_plan = self.dosage_work_dispatch_plan_for_optional_item(py, item.as_ref())?;
        NativeDosageWorkItemGetResult::from_item(
            py,
            item,
            has_dosage_work_item,
            observation_plan,
            drain_completion_plan,
            dispatch_plan,
        )
    }

    fn plan_dosage_work_drain_completion(
        &self,
        py: Python<'_>,
        has_dosage_work_item: bool,
    ) -> NativeDosageWorkDrainCompletionPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_work_drain_completion_value(has_dosage_work_item)
    }

    fn plan_dosage_work_drain_completion_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> NativeDosageWorkDrainCompletionPlan {
        self.plan_dosage_work_drain_completion(py, !work_item.is_none())
    }

    fn plan_validated_dosage_work_item_dispatch(
        &self,
        py: Python<'_>,
        dosage_work_item_kind: &str,
    ) -> PyResult<NativeDosageWorkItemDispatchPlan> {
        self.dosage_work_dispatch_plan_for_kind(py, dosage_work_item_kind)
    }

    fn plan_validated_dosage_work_item_dispatch_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageWorkItemDispatchPlan> {
        let dosage_work_item_kind = classify_dosage_work_item_kind(work_item)?;
        self.plan_validated_dosage_work_item_dispatch(py, dosage_work_item_kind)
    }

    fn plan_dosage_work_item_stage_duration(
        &self,
        py: Python<'_>,
        dosage_work_item_kind: &str,
        chunk_count: usize,
        elapsed_seconds: f64,
    ) -> PyResult<NativeDosageWorkItemStageDurationPlan> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_work_item_stage_duration_value(dosage_work_item_kind, chunk_count, elapsed_seconds)
    }

    fn plan_dosage_work_item_stage_duration_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
        elapsed_seconds: f64,
    ) -> PyResult<NativeDosageWorkItemStageDurationPlan> {
        let descriptor = classify_dosage_work_item(work_item)?;
        self.plan_dosage_work_item_stage_duration(
            py,
            descriptor.dosage_work_item_kind,
            descriptor.chunk_count,
            elapsed_seconds,
        )
    }

    fn plan_dosage_work_item_stage_duration_attribution_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
        elapsed_seconds: f64,
    ) -> PyResult<NativeDosageWorkItemStageDurationAttribution> {
        let stage_duration_plan =
            self.plan_dosage_work_item_stage_duration_for_object(py, work_item, elapsed_seconds)?;
        let metadata_items = dosage_work_item_metadata_items(py, work_item)?;
        NativeDosageWorkItemStageDurationAttribution::from_attribution(py, metadata_items, stage_duration_plan)
    }

    fn plan_current_queue_backpressure_observation(
        &self,
        py: Python<'_>,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueBackpressureObservation> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_current_queue_backpressure_observation_value(
            queue_name,
            operation_name,
            elapsed_seconds,
            blocked,
        )
    }

    fn plan_current_queue_stage_backpressure_observation(
        &self,
        py: Python<'_>,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_current_queue_stage_backpressure_observation_value(
            queue_name,
            operation_name,
            elapsed_seconds,
            blocked,
        )
    }

    fn plan_dosage_queue_put_observation(&self, py: Python<'_>, queued: bool) -> NativeCallbackQueuePutObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_queue_put_observation_value(queued)
    }

    fn plan_dosage_queue_get_observation(&self, py: Python<'_>) -> NativeCallbackQueueGetObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_queue_get_observation_value()
    }

    fn plan_result_queue_put_observation(&self, py: Python<'_>, queued: bool) -> NativeCallbackQueuePutObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_result_queue_put_observation_value(queued)
    }

    fn plan_result_queue_get_observation(&self, py: Python<'_>) -> NativeCallbackQueueGetObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_result_queue_get_observation_value()
    }

    fn plan_dosage_buffer_pool_reuse_observation(&self, py: Python<'_>) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_reuse_observation_value()
    }

    fn plan_dosage_buffer_pool_return_observation(&self, py: Python<'_>) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_return_observation_value()
    }

    fn plan_dosage_buffer_pool_allocate_observation(&self, py: Python<'_>) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_allocate_observation_value()
    }

    fn plan_dosage_buffer_pool_discard_observation(&self, py: Python<'_>) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_discard_observation_value()
    }

    fn plan_dosage_buffer_pool_consumer_wait_observation(
        &self,
        py: Python<'_>,
    ) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_consumer_wait_observation_value()
    }

    fn plan_dosage_buffer_pool_backpressure_observation(
        &self,
        py: Python<'_>,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueBackpressureObservation> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_backpressure_observation_value(
            operation_name,
            free_buffer_count,
            elapsed_seconds,
            blocked,
        )
    }

    fn plan_dosage_buffer_pool_stage_backpressure_observation(
        &self,
        py: Python<'_>,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_stage_backpressure_observation_value(
            operation_name,
            free_buffer_count,
            elapsed_seconds,
            blocked,
        )
    }

    fn plan_variant_major_dosage_batch_handoff(
        &self,
        py: Python<'_>,
        metadata_count: usize,
        genotype_matrix_by_variant_count: usize,
        chunk_stats_count: usize,
    ) -> PyResult<NativeVariantMajorDosageBatchHandoffPlan> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_variant_major_dosage_batch_handoff_value(
            metadata_count,
            genotype_matrix_by_variant_count,
            chunk_stats_count,
        )
    }

    fn plan_variant_major_dosage_batch_handoff_for_sequences(
        &self,
        py: Python<'_>,
        metadata_batch: &Bound<'_, PyAny>,
        genotype_matrix_by_variant_batch: &Bound<'_, PyAny>,
        chunk_stats_batch: &Bound<'_, PyAny>,
    ) -> PyResult<NativeVariantMajorDosageBatchHandoffPlan> {
        self.plan_variant_major_dosage_batch_handoff(
            py,
            metadata_batch.len()?,
            genotype_matrix_by_variant_batch.len()?,
            chunk_stats_batch.len()?,
        )
    }

    fn plan_dosage_work_handoff(&self, py: Python<'_>, chunk_count: usize) -> PyResult<NativeDosageWorkHandoffPlan> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_work_handoff_value(chunk_count)
    }

    fn plan_dosage_work_handoff_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeDosageWorkHandoffPlan> {
        let descriptor = classify_dosage_work_item(work_item)?;
        self.plan_dosage_work_handoff(py, descriptor.chunk_count)
    }

    fn try_put_result_write_item(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
        timeout_seconds: f64,
    ) -> PyResult<bool> {
        let handoff_plan = self.plan_result_write_handoff(py, work_item)?;
        let deadline = Instant::now() + normalize_timeout_duration(timeout_seconds);
        loop {
            let attempt_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                scheduler_state.plan_result_queue_put_attempt_value(remaining_timeout_seconds(deadline))
            };
            if attempt_plan.should_put_value() && handoff_plan.should_enqueue_value() {
                return self.put_result_write_item_after_slot_acquisition(py, work_item);
            }
            if !attempt_plan.should_wait_value() {
                return Ok(false);
            }
            self.result_queue
                .bind(py)
                .borrow()
                .wait_for_available_slot_value(py, attempt_plan.wait_timeout_seconds_value())?;
        }
    }

    fn try_put_result_write_item_with_backpressure_timeout(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let handoff_plan = self.plan_result_write_handoff(py, work_item)?;
        let mut deadline = None;
        loop {
            let attempt_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                if let Some(deadline) = deadline {
                    scheduler_state.plan_result_queue_put_attempt_value(remaining_timeout_seconds(deadline))
                } else {
                    let attempt_plan = scheduler_state.plan_result_queue_put_backpressure_attempt_value();
                    if attempt_plan.should_wait_value() {
                        deadline = Some(
                            Instant::now() + normalize_timeout_duration(attempt_plan.wait_timeout_seconds_value()),
                        );
                    }
                    attempt_plan
                }
            };
            if attempt_plan.should_put_value() && handoff_plan.should_enqueue_value() {
                return self.put_result_write_item_after_slot_acquisition(py, work_item);
            }
            if !attempt_plan.should_wait_value() {
                return Ok(false);
            }
            self.result_queue
                .bind(py)
                .borrow()
                .wait_for_available_slot_value(py, attempt_plan.wait_timeout_seconds_value())?;
        }
    }

    fn put_result_write_item_with_backpressure_observation(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeCallbackQueuePutObservationPlan> {
        let queued = self.try_put_result_write_item_with_backpressure_timeout(py, work_item)?;
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        Ok(scheduler_state.plan_result_queue_put_observation_value(queued))
    }

    fn put_result_write_item_with_optional_backpressure_observation(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeCallbackQueuePutResult> {
        let queued = self.try_put_result_write_item_with_backpressure_timeout(py, work_item)?;
        let observation_plan = if self.has_stage_timing_recorder {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            Some(scheduler_state.plan_result_queue_put_observation_value(queued))
        } else {
            None
        };
        NativeCallbackQueuePutResult::from_put(py, !queued, observation_plan)
    }

    fn get_result_write_item(&self, py: Python<'_>) -> PyResult<NativeCallbackObjectQueueGetResult> {
        loop {
            let has_queued_item = self.result_queue.bind(py).borrow().has_queued_item_value()?;
            let get_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                scheduler_state.plan_result_queue_get_attempt_value(has_queued_item)
            };
            if get_plan.has_release_error_value() {
                return Err(PyRuntimeError::new_err("Native result-queue state has no occupied slot to release."));
            }
            if get_plan.should_get_value() {
                let get_result = self.result_queue.bind(py).borrow().get_item(py, 0.0)?;
                if !get_result.has_item_value() {
                    let reacquired_slot = {
                        let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                        scheduler_state.acquire_result_queue_slot_value()
                    };
                    if !reacquired_slot {
                        return Err(PyRuntimeError::new_err(
                            "Native result queue storage was empty after scheduler slot release.",
                        ));
                    }
                    return Err(PyRuntimeError::new_err(
                        "Native result queue storage had no queued item after scheduler selected get.",
                    ));
                }
                return Ok(get_result);
            }
            if get_plan.should_wait_value() {
                self.result_queue
                    .bind(py)
                    .borrow()
                    .wait_for_queued_item_value(py, get_plan.wait_timeout_seconds_value())?;
            }
        }
    }

    fn get_result_write_item_with_observation(&self, py: Python<'_>) -> PyResult<NativeCallbackQueueGetObservedResult> {
        let get_result = self.get_result_write_item(py)?;
        let observation_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_queue_get_observation_value()
        };
        NativeCallbackQueueGetObservedResult::from_get_result(py, get_result, observation_plan)
    }

    fn get_result_write_item_with_drain_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeResultWriteItemDrainResult> {
        let get_result = self.get_result_write_item(py)?;
        let has_result_work_item = get_result.has_non_none_item_value(py);
        let drain_completion_plan = self.plan_result_write_drain_completion_value(py, has_result_work_item);
        NativeResultWriteItemDrainResult::from_get_result(py, get_result, has_result_work_item, drain_completion_plan)
    }

    fn get_result_write_item_with_observation_and_drain_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeResultWriteItemGetResult> {
        let get_result = self.get_result_write_item(py)?;
        let has_result_work_item = get_result.has_non_none_item_value(py);
        let (observation_plan, drain_completion_plan) = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            (
                scheduler_state.plan_result_queue_get_observation_value(),
                scheduler_state.plan_result_write_drain_completion_value(
                    has_result_work_item,
                    self.flush_binary_correction_diagnostics_on_result_stop,
                ),
            )
        };
        NativeResultWriteItemGetResult::from_get_result(
            py,
            get_result,
            has_result_work_item,
            Some(observation_plan),
            drain_completion_plan,
            None,
        )
    }

    fn get_result_write_item_with_optional_observation_and_drain_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeResultWriteItemGetResult> {
        let get_result = self.get_result_write_item(py)?;
        let has_result_work_item = get_result.has_non_none_item_value(py);
        let (observation_plan, drain_completion_plan) = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            let observation_plan = if self.has_stage_timing_recorder {
                Some(scheduler_state.plan_result_queue_get_observation_value())
            } else {
                None
            };
            (
                observation_plan,
                scheduler_state.plan_result_write_drain_completion_value(
                    has_result_work_item,
                    self.flush_binary_correction_diagnostics_on_result_stop,
                ),
            )
        };
        NativeResultWriteItemGetResult::from_get_result(
            py,
            get_result,
            has_result_work_item,
            observation_plan,
            drain_completion_plan,
            None,
        )
    }

    fn get_validated_result_write_item_with_drain_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeResultWriteItemGetResult> {
        let get_result = self.get_result_write_item(py)?;
        let item = get_result.into_item_value();
        let has_result_work_item = item.as_ref().is_some_and(|queued_item| !queued_item.bind(py).is_none());
        let drain_completion_plan = self.plan_result_write_drain_completion_value(py, has_result_work_item);
        let dispatch_plan = self.result_write_dispatch_plan_for_optional_item(py, item.as_ref())?;
        NativeResultWriteItemGetResult::from_item(
            py,
            item,
            has_result_work_item,
            None,
            drain_completion_plan,
            dispatch_plan,
        )
    }

    fn get_validated_result_write_item_with_optional_observation_and_drain_completion(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeResultWriteItemGetResult> {
        let get_result = self.get_result_write_item(py)?;
        let item = get_result.into_item_value();
        let has_result_work_item = item.as_ref().is_some_and(|queued_item| !queued_item.bind(py).is_none());
        let (observation_plan, drain_completion_plan) = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            let observation_plan = if self.has_stage_timing_recorder {
                Some(scheduler_state.plan_result_queue_get_observation_value())
            } else {
                None
            };
            (
                observation_plan,
                scheduler_state.plan_result_write_drain_completion_value(
                    has_result_work_item,
                    self.flush_binary_correction_diagnostics_on_result_stop,
                ),
            )
        };
        let dispatch_plan = self.result_write_dispatch_plan_for_optional_item(py, item.as_ref())?;
        NativeResultWriteItemGetResult::from_item(
            py,
            item,
            has_result_work_item,
            observation_plan,
            drain_completion_plan,
            dispatch_plan,
        )
    }

    fn plan_result_write_drain_completion(
        &self,
        py: Python<'_>,
        has_result_work_item: bool,
    ) -> NativeResultWriteDrainCompletionPlan {
        self.plan_result_write_drain_completion_value(py, has_result_work_item)
    }

    fn plan_result_write_drain_completion_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> NativeResultWriteDrainCompletionPlan {
        self.plan_result_write_drain_completion_value(py, !work_item.is_none())
    }

    fn plan_validated_result_write_item_dispatch(
        &self,
        py: Python<'_>,
        result_work_item_kind: &str,
    ) -> PyResult<NativeResultWriteItemDispatchPlan> {
        self.result_write_dispatch_plan_for_kind(py, result_work_item_kind)
    }

    fn plan_validated_result_write_item_dispatch_for_object(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeResultWriteItemDispatchPlan> {
        let result_work_item_kind = classify_result_write_item_kind(work_item)?;
        self.plan_validated_result_write_item_dispatch(py, result_work_item_kind)
    }
}

#[pymethods]
impl NativeResultWorkItemResourceReleaseResult {
    #[getter]
    fn released_host_buffer(&self) -> bool {
        self.released_host_buffer
    }

    #[getter]
    fn free_buffer_count(&self) -> Option<usize> {
        self.free_buffer_count
    }

    #[getter]
    fn dosage_buffer_pool_observation_plan(&self, py: Python<'_>) -> Option<Py<NativeDosageBufferPoolObservationPlan>> {
        self.dosage_buffer_pool_observation_plan.as_ref().map(|plan| plan.clone_ref(py))
    }

    #[getter]
    fn released_result_in_flight_slot(&self) -> bool {
        self.released_result_in_flight_slot
    }

    #[getter]
    fn result_in_flight_observation_plan(
        &self,
        py: Python<'_>,
    ) -> Option<Py<NativeResultInFlightReleaseObservationPlan>> {
        self.result_in_flight_observation_plan.as_ref().map(|plan| plan.clone_ref(py))
    }

    #[getter]
    fn result_in_flight_resource_name(&self, py: Python<'_>) -> Option<String> {
        self.result_in_flight_observation_plan.as_ref().map(|plan| {
            let plan_bound = plan.bind(py);
            plan_bound.borrow().resource_name_value().to_owned()
        })
    }

    #[getter]
    fn result_in_flight_operation_name(&self, py: Python<'_>) -> Option<String> {
        self.result_in_flight_observation_plan.as_ref().map(|plan| {
            let plan_bound = plan.bind(py);
            plan_bound.borrow().operation_name_value().to_owned()
        })
    }

    #[getter]
    fn result_in_flight_blocked(&self, py: Python<'_>) -> Option<bool> {
        self.result_in_flight_observation_plan.as_ref().map(|plan| {
            let plan_bound = plan.bind(py);
            plan_bound.borrow().blocked_value()
        })
    }
}

impl NativeResultWorkItemResourceReleaseResult {
    fn empty() -> Self {
        Self {
            released_host_buffer: false,
            free_buffer_count: None,
            dosage_buffer_pool_observation_plan: None,
            released_result_in_flight_slot: false,
            result_in_flight_observation_plan: None,
        }
    }

    fn record_result_in_flight_release(
        &mut self,
        py: Python<'_>,
        release_observation_plan: Option<NativeResultInFlightReleaseObservationPlan>,
    ) -> PyResult<()> {
        self.released_result_in_flight_slot = true;
        self.result_in_flight_observation_plan = release_observation_plan.map(|plan| Py::new(py, plan)).transpose()?;
        Ok(())
    }

    fn record_host_buffer_return(
        &mut self,
        py: Python<'_>,
        free_buffer_count: Option<usize>,
        observation_plan: Option<NativeDosageBufferPoolObservationPlan>,
    ) -> PyResult<()> {
        self.released_host_buffer = true;
        self.free_buffer_count = free_buffer_count;
        if free_buffer_count.is_some() {
            let Some(observation_plan) = observation_plan else {
                return Ok(());
            };
            self.dosage_buffer_pool_observation_plan = Some(Py::new(py, observation_plan)?);
        }
        Ok(())
    }
}

#[pymethods]
impl NativeDosageBufferPoolOperationResult {
    #[getter]
    fn has_free_buffer_count(&self) -> bool {
        self.free_buffer_count.is_some()
    }

    #[getter]
    fn free_buffer_count(&self) -> Option<usize> {
        self.free_buffer_count
    }

    #[getter]
    fn observation_plan(&self, py: Python<'_>) -> Option<Py<NativeDosageBufferPoolObservationPlan>> {
        self.observation_plan.as_ref().map(|plan| plan.clone_ref(py))
    }
}

impl NativeDosageBufferPoolOperationResult {
    fn from_operation(
        py: Python<'_>,
        free_buffer_count: Option<usize>,
        observation_plan: Option<NativeDosageBufferPoolObservationPlan>,
    ) -> PyResult<Self> {
        Ok(Self { free_buffer_count, observation_plan: observation_plan.map(|plan| Py::new(py, plan)).transpose()? })
    }
}

#[pymethods]
impl NativeDosageBufferReuseSelectionResult {
    #[getter]
    fn dosage_buffer(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.dosage_buffer.as_ref().map(|dosage_buffer| dosage_buffer.clone_ref(py))
    }

    #[getter]
    fn operation_result(&self, py: Python<'_>) -> Py<NativeDosageBufferPoolOperationResult> {
        self.operation_result.clone_ref(py)
    }

    #[getter]
    fn reuse_operation_result(&self, py: Python<'_>) -> Option<Py<NativeDosageBufferPoolOperationResult>> {
        self.dosage_buffer.as_ref().map(|_| self.operation_result.clone_ref(py))
    }

    #[getter]
    fn discard_operation_result(&self, py: Python<'_>) -> Option<Py<NativeDosageBufferPoolOperationResult>> {
        if self.dosage_buffer.is_some() { None } else { Some(self.operation_result.clone_ref(py)) }
    }
}

impl NativeDosageBufferReuseSelectionResult {
    fn from_reusable_dosage_buffer(
        py: Python<'_>,
        dosage_buffer: Py<PyAny>,
        reuse_operation_result: NativeDosageBufferPoolOperationResult,
    ) -> PyResult<Self> {
        Ok(Self { dosage_buffer: Some(dosage_buffer), operation_result: Py::new(py, reuse_operation_result)? })
    }

    fn from_discard(py: Python<'_>, discard_operation_result: NativeDosageBufferPoolOperationResult) -> PyResult<Self> {
        Ok(Self { dosage_buffer: None, operation_result: Py::new(py, discard_operation_result)? })
    }
}

impl NativeResultInFlightAcquireResult {
    fn from_acquire(
        py: Python<'_>,
        should_retry_acquisition: bool,
        observation_plan: Option<NativeResultInFlightAcquireObservationPlan>,
    ) -> PyResult<Self> {
        Ok(Self {
            should_retry_acquisition,
            observation_plan: observation_plan.map(|plan| Py::new(py, plan)).transpose()?,
        })
    }
}

#[pymethods]
impl NativeResultInFlightAcquireResult {
    #[getter]
    fn should_retry_acquisition(&self) -> bool {
        self.should_retry_acquisition
    }

    #[getter]
    fn observation_plan(&self, py: Python<'_>) -> Option<Py<NativeResultInFlightAcquireObservationPlan>> {
        self.observation_plan.as_ref().map(|plan| plan.clone_ref(py))
    }
}

impl NativeCallbackQueuePutResult {
    fn from_put(
        py: Python<'_>,
        should_retry_put: bool,
        observation_plan: Option<NativeCallbackQueuePutObservationPlan>,
    ) -> PyResult<Self> {
        Ok(Self { should_retry_put, observation_plan: observation_plan.map(|plan| Py::new(py, plan)).transpose()? })
    }
}

#[pymethods]
impl NativeCallbackQueuePutResult {
    #[getter]
    fn should_retry_put(&self) -> bool {
        self.should_retry_put
    }

    #[getter]
    fn observation_plan(&self, py: Python<'_>) -> Option<Py<NativeCallbackQueuePutObservationPlan>> {
        self.observation_plan.as_ref().map(|plan| plan.clone_ref(py))
    }
}

#[pymethods]
impl NativeCallbackQueueGetObservedResult {
    #[getter]
    fn has_item(&self) -> bool {
        self.item.is_some()
    }

    #[getter]
    fn item(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.item.as_ref().map(|item| item.clone_ref(py))
    }

    #[getter]
    fn observation_plan(&self, py: Python<'_>) -> Py<NativeCallbackQueueGetObservationPlan> {
        self.observation_plan.clone_ref(py)
    }
}

impl NativeCallbackQueueGetObservedResult {
    fn from_get_result(
        py: Python<'_>,
        get_result: NativeCallbackObjectQueueGetResult,
        observation_plan: NativeCallbackQueueGetObservationPlan,
    ) -> PyResult<Self> {
        Ok(Self { item: get_result.into_item_value(), observation_plan: Py::new(py, observation_plan)? })
    }
}

impl NativeDosageWorkItemStageDurationAttribution {
    fn from_attribution(
        py: Python<'_>,
        metadata_items: Py<PyTuple>,
        stage_duration_plan: NativeDosageWorkItemStageDurationPlan,
    ) -> PyResult<Self> {
        Ok(Self { metadata_items, stage_duration_plan: Py::new(py, stage_duration_plan)? })
    }
}

#[pymethods]
impl NativeDosageWorkItemStageDurationAttribution {
    #[getter]
    fn metadata_items(&self, py: Python<'_>) -> Py<PyTuple> {
        self.metadata_items.clone_ref(py)
    }

    #[getter]
    fn stage_duration_plan(&self, py: Python<'_>) -> Py<NativeDosageWorkItemStageDurationPlan> {
        self.stage_duration_plan.clone_ref(py)
    }
}

#[pymethods]
impl NativeDosageWorkItemDrainResult {
    #[getter]
    fn has_dosage_work_item(&self) -> bool {
        self.has_dosage_work_item
    }

    #[getter]
    fn item(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.item.as_ref().map(|item| item.clone_ref(py))
    }

    #[getter]
    fn drain_completion_plan(&self, py: Python<'_>) -> Py<NativeDosageWorkDrainCompletionPlan> {
        self.drain_completion_plan.clone_ref(py)
    }
}

impl NativeDosageWorkItemDrainResult {
    fn from_get_result(
        py: Python<'_>,
        get_result: NativeCallbackObjectQueueGetResult,
        has_dosage_work_item: bool,
        drain_completion_plan: NativeDosageWorkDrainCompletionPlan,
    ) -> PyResult<Self> {
        Ok(Self {
            item: get_result.into_item_value(),
            has_dosage_work_item,
            drain_completion_plan: Py::new(py, drain_completion_plan)?,
        })
    }
}

#[pymethods]
impl NativeDosageWorkItemGetResult {
    #[getter]
    fn has_dosage_work_item(&self) -> bool {
        self.has_dosage_work_item
    }

    #[getter]
    fn item(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.item.as_ref().map(|item| item.clone_ref(py))
    }

    #[getter]
    fn observation_plan(&self, py: Python<'_>) -> Option<Py<NativeCallbackQueueGetObservationPlan>> {
        self.observation_plan.as_ref().map(|plan| plan.clone_ref(py))
    }

    #[getter]
    fn drain_completion_plan(&self, py: Python<'_>) -> Py<NativeDosageWorkDrainCompletionPlan> {
        self.drain_completion_plan.clone_ref(py)
    }

    #[getter]
    fn dispatch_plan(&self, py: Python<'_>) -> Option<Py<NativeDosageWorkItemDispatchPlan>> {
        self.dispatch_plan.as_ref().map(|plan| plan.clone_ref(py))
    }
}

impl NativeDosageWorkItemGetResult {
    fn from_get_result(
        py: Python<'_>,
        get_result: NativeCallbackObjectQueueGetResult,
        has_dosage_work_item: bool,
        observation_plan: Option<NativeCallbackQueueGetObservationPlan>,
        drain_completion_plan: NativeDosageWorkDrainCompletionPlan,
        dispatch_plan: Option<NativeDosageWorkItemDispatchPlan>,
    ) -> PyResult<Self> {
        Self::from_item(
            py,
            get_result.into_item_value(),
            has_dosage_work_item,
            observation_plan,
            drain_completion_plan,
            dispatch_plan,
        )
    }

    fn from_item(
        py: Python<'_>,
        item: Option<Py<PyAny>>,
        has_dosage_work_item: bool,
        observation_plan: Option<NativeCallbackQueueGetObservationPlan>,
        drain_completion_plan: NativeDosageWorkDrainCompletionPlan,
        dispatch_plan: Option<NativeDosageWorkItemDispatchPlan>,
    ) -> PyResult<Self> {
        Ok(Self {
            item,
            has_dosage_work_item,
            observation_plan: observation_plan.map(|plan| Py::new(py, plan)).transpose()?,
            drain_completion_plan: Py::new(py, drain_completion_plan)?,
            dispatch_plan: dispatch_plan.map(|plan| Py::new(py, plan)).transpose()?,
        })
    }
}

#[pymethods]
impl NativeResultWriteItemDrainResult {
    #[getter]
    fn has_result_work_item(&self) -> bool {
        self.has_result_work_item
    }

    #[getter]
    fn item(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.item.as_ref().map(|item| item.clone_ref(py))
    }

    #[getter]
    fn drain_completion_plan(&self, py: Python<'_>) -> Py<NativeResultWriteDrainCompletionPlan> {
        self.drain_completion_plan.clone_ref(py)
    }
}

impl NativeResultWriteItemDrainResult {
    fn from_get_result(
        py: Python<'_>,
        get_result: NativeCallbackObjectQueueGetResult,
        has_result_work_item: bool,
        drain_completion_plan: NativeResultWriteDrainCompletionPlan,
    ) -> PyResult<Self> {
        Ok(Self {
            item: get_result.into_item_value(),
            has_result_work_item,
            drain_completion_plan: Py::new(py, drain_completion_plan)?,
        })
    }
}

#[pymethods]
impl NativeResultWriteItemGetResult {
    #[getter]
    fn has_result_work_item(&self) -> bool {
        self.has_result_work_item
    }

    #[getter]
    fn item(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.item.as_ref().map(|item| item.clone_ref(py))
    }

    #[getter]
    fn observation_plan(&self, py: Python<'_>) -> Option<Py<NativeCallbackQueueGetObservationPlan>> {
        self.observation_plan.as_ref().map(|plan| plan.clone_ref(py))
    }

    #[getter]
    fn drain_completion_plan(&self, py: Python<'_>) -> Py<NativeResultWriteDrainCompletionPlan> {
        self.drain_completion_plan.clone_ref(py)
    }

    #[getter]
    fn dispatch_plan(&self, py: Python<'_>) -> Option<Py<NativeResultWriteItemDispatchPlan>> {
        self.dispatch_plan.as_ref().map(|plan| plan.clone_ref(py))
    }
}

impl NativeResultWriteItemGetResult {
    fn from_get_result(
        py: Python<'_>,
        get_result: NativeCallbackObjectQueueGetResult,
        has_result_work_item: bool,
        observation_plan: Option<NativeCallbackQueueGetObservationPlan>,
        drain_completion_plan: NativeResultWriteDrainCompletionPlan,
        dispatch_plan: Option<NativeResultWriteItemDispatchPlan>,
    ) -> PyResult<Self> {
        Self::from_item(
            py,
            get_result.into_item_value(),
            has_result_work_item,
            observation_plan,
            drain_completion_plan,
            dispatch_plan,
        )
    }

    fn from_item(
        py: Python<'_>,
        item: Option<Py<PyAny>>,
        has_result_work_item: bool,
        observation_plan: Option<NativeCallbackQueueGetObservationPlan>,
        drain_completion_plan: NativeResultWriteDrainCompletionPlan,
        dispatch_plan: Option<NativeResultWriteItemDispatchPlan>,
    ) -> PyResult<Self> {
        Ok(Self {
            item,
            has_result_work_item,
            observation_plan: observation_plan.map(|plan| Py::new(py, plan)).transpose()?,
            drain_completion_plan: Py::new(py, drain_completion_plan)?,
            dispatch_plan: dispatch_plan.map(|plan| Py::new(py, plan)).transpose()?,
        })
    }
}

#[pymethods]
impl NativeCallbackWorkerFinishLifecycleResult {
    #[getter]
    fn has_shutdown_timeout(&self) -> bool {
        self.shutdown_timeout_seconds.is_some()
    }

    #[getter]
    fn shutdown_worker_name(&self) -> Option<&str> {
        self.shutdown_worker_name.as_deref()
    }

    #[getter]
    fn shutdown_timeout_seconds(&self) -> Option<f64> {
        self.shutdown_timeout_seconds
    }

    #[getter]
    fn raise_worker_error(&self) -> bool {
        self.finish_plan.raise_worker_error_value()
    }

    #[getter]
    fn complete_progress(&self) -> bool {
        self.finish_plan.complete_progress_value()
    }

    #[getter]
    fn progress_completion_event(&self) -> Option<NativeCallbackProgressTelemetryEvent> {
        self.progress_completion_event.clone()
    }

    #[getter]
    fn emit_binary_correction_summary(&self) -> bool {
        self.finish_plan.emit_binary_correction_summary_value()
    }

    #[getter]
    fn flush_binary_correction_pending_diagnostics(&self) -> bool {
        self.flush_binary_correction_pending_diagnostics
    }

    #[getter]
    fn binary_correction_summary_payload(&self, py: Python<'_>) -> Option<Py<PyDict>> {
        self.binary_correction_summary_payload.as_ref().map(|summary_payload| summary_payload.clone_ref(py))
    }
}

impl NativeCallbackWorkerFinishLifecycleResult {
    fn from_finish_plan(finish_plan: &NativeCallbackWorkerFinishPlan) -> Self {
        Self {
            finish_plan: finish_plan.clone(),
            shutdown_worker_name: None,
            shutdown_timeout_seconds: None,
            progress_completion_event: None,
            flush_binary_correction_pending_diagnostics: false,
            binary_correction_summary_payload: None,
        }
    }

    fn record_shutdown_timeout(&mut self, worker_name: String, timeout_seconds: f64) {
        self.shutdown_worker_name = Some(worker_name);
        self.shutdown_timeout_seconds = Some(timeout_seconds);
    }

    fn record_progress_completion(&mut self, progress_completion: Option<NativeCallbackProgressCompletion>) {
        self.progress_completion_event = progress_completion.map(|completion| completion.telemetry_event_value());
    }

    fn record_binary_correction_summary_payload(&mut self, summary_payload: Py<PyDict>) {
        self.binary_correction_summary_payload = Some(summary_payload);
    }

    fn record_binary_correction_pending_diagnostics_flush(&mut self, should_flush_pending_diagnostics: bool) {
        self.flush_binary_correction_pending_diagnostics = should_flush_pending_diagnostics;
    }
}

#[pymethods]
impl NativeDosageBufferAcquireResult {
    #[getter]
    fn dosage_buffer(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.dosage_buffer.as_ref().map(|dosage_buffer| dosage_buffer.clone_ref(py))
    }

    #[getter]
    fn should_allocate(&self) -> bool {
        self.should_allocate
    }

    #[getter]
    fn free_buffer_count(&self) -> usize {
        self.free_buffer_count
    }

    #[getter]
    fn waited(&self) -> bool {
        self.waited
    }

    #[getter]
    fn observation_plan(&self, py: Python<'_>) -> Option<Py<NativeDosageBufferPoolObservationPlan>> {
        self.observation_plan.as_ref().map(|plan| plan.clone_ref(py))
    }
}

impl NativeCallbackRuntimeResources {
    fn result_write_dispatch_plan_for_kind(
        &self,
        py: Python<'_>,
        result_work_item_kind: &str,
    ) -> PyResult<NativeResultWriteItemDispatchPlan> {
        let dispatch_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state
                .plan_result_write_item_dispatch_value(result_work_item_kind, &self.expected_result_work_item_kind)?
        };
        if !dispatch_plan.has_dispatch_error_value() {
            return Ok(dispatch_plan);
        }
        let error_message = dispatch_plan
            .error_message_value()
            .unwrap_or("Native result write dispatch plan omitted the error message.");
        Err(PyRuntimeError::new_err(error_message.to_owned()))
    }

    fn result_write_dispatch_plan_for_optional_item(
        &self,
        py: Python<'_>,
        work_item: Option<&Py<PyAny>>,
    ) -> PyResult<Option<NativeResultWriteItemDispatchPlan>> {
        let Some(work_item) = work_item else {
            return Ok(None);
        };
        let work_item_bound = work_item.bind(py);
        if work_item_bound.is_none() {
            return Ok(None);
        }
        let result_work_item_kind = classify_result_write_item_kind(work_item_bound)?;
        self.result_write_dispatch_plan_for_kind(py, result_work_item_kind).map(Some)
    }

    fn dosage_work_dispatch_plan_for_kind(
        &self,
        py: Python<'_>,
        dosage_work_item_kind: &str,
    ) -> PyResult<NativeDosageWorkItemDispatchPlan> {
        let dispatch_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_work_item_dispatch_value(dosage_work_item_kind)?
        };
        if !dispatch_plan.has_dispatch_error_value() {
            return Ok(dispatch_plan);
        }
        let error_message = dispatch_plan
            .error_message_value()
            .unwrap_or("Native dosage work dispatch plan omitted the error message.");
        Err(PyRuntimeError::new_err(error_message.to_owned()))
    }

    fn dosage_work_dispatch_plan_for_optional_item(
        &self,
        py: Python<'_>,
        work_item: Option<&Py<PyAny>>,
    ) -> PyResult<Option<NativeDosageWorkItemDispatchPlan>> {
        let Some(work_item) = work_item else {
            return Ok(None);
        };
        let work_item_bound = work_item.bind(py);
        if work_item_bound.is_none() {
            return Ok(None);
        }
        let dosage_work_item_kind = classify_dosage_work_item_kind(work_item_bound)?;
        self.dosage_work_dispatch_plan_for_kind(py, dosage_work_item_kind).map(Some)
    }

    fn release_result_in_flight_slot_without_observation(&self, py: Python<'_>) -> PyResult<()> {
        let release_plan = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.plan_result_in_flight_slot_release_attempt_value()
        };
        if release_plan.has_release_error_value() {
            return Err(PyRuntimeError::new_err("Native result in-flight slot state has no occupied slot to release."));
        }
        self.result_in_flight_slot_signal.bind(py).borrow().notify_waiters_value()?;
        Ok(())
    }

    fn plan_result_write_drain_completion_value(
        &self,
        py: Python<'_>,
        has_result_work_item: bool,
    ) -> NativeResultWriteDrainCompletionPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_result_write_drain_completion_value(
            has_result_work_item,
            self.flush_binary_correction_diagnostics_on_result_stop,
        )
    }

    fn put_dosage_work_item_after_slot_acquisition(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let queued = self.dosage_queue.bind(py).borrow().put_item(py, work_item.clone().unbind(), 0.0)?;
        if queued {
            return Ok(true);
        }
        let released_slot = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.release_dosage_queue_slot_value()
        };
        if !released_slot {
            return Err(PyRuntimeError::new_err(
                "Native dosage queue storage rejected a put after scheduler slot acquisition.",
            ));
        }
        Err(PyRuntimeError::new_err("Native dosage queue storage had no slot after scheduler selected put."))
    }

    fn put_result_write_item_after_slot_acquisition(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let queued = self.result_queue.bind(py).borrow().put_item(py, work_item.clone().unbind(), 0.0)?;
        if queued {
            return Ok(true);
        }
        let released_slot = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.release_result_queue_slot_value()
        };
        if !released_slot {
            return Err(PyRuntimeError::new_err(
                "Native result queue storage rejected a put after scheduler slot acquisition.",
            ));
        }
        Err(PyRuntimeError::new_err("Native result queue storage had no slot after scheduler selected put."))
    }

    fn plan_result_write_handoff(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeResultWriteHandoffPlan> {
        let has_result_work_item = !work_item.is_none();
        let handoff_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_write_handoff_value(has_result_work_item)
        };
        if handoff_plan.has_result_work_item_value() != has_result_work_item {
            return Err(PyRuntimeError::new_err(
                "Native result write handoff plan disagrees with the queued result item.",
            ));
        }
        Ok(handoff_plan)
    }

    fn release_result_work_item_resources_with_plan(
        &self,
        py: Python<'_>,
        resource_release_plan: &NativeResultWriteItemResourceReleasePlan,
        host_dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeResultWorkItemResourceReleaseResult> {
        let mut release_result = NativeResultWorkItemResourceReleaseResult::empty();
        if resource_release_plan.should_release_host_buffer_value() {
            if host_dosage_buffer.is_none() {
                return Err(PyRuntimeError::new_err(
                    "Native result work item resource release plan selected a missing host buffer.",
                ));
            }
            let buffer_identifier = py_object_identifier(host_dosage_buffer);
            let free_buffer_count = self.return_dosage_buffer(py, buffer_identifier, host_dosage_buffer)?;
            let return_observation_plan = if self.has_stage_timing_recorder {
                let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
                Some(scheduler_state.plan_dosage_buffer_pool_return_observation_value())
            } else {
                None
            };
            release_result.record_host_buffer_return(py, free_buffer_count, return_observation_plan)?;
        }
        if resource_release_plan.should_release_result_in_flight_slot_value() {
            self.record_result_work_item_in_flight_slot_release(py, &mut release_result)?;
        }
        Ok(release_result)
    }
}

impl NativeCallbackRuntimeResources {
    fn record_result_work_item_in_flight_slot_release(
        &self,
        py: Python<'_>,
        release_result: &mut NativeResultWorkItemResourceReleaseResult,
    ) -> PyResult<()> {
        if self.has_stage_timing_recorder {
            let release_observation_plan = self.release_result_in_flight_slot(py)?;
            release_result.record_result_in_flight_release(py, Some(release_observation_plan))?;
        } else {
            self.release_result_in_flight_slot_without_observation(py)?;
            release_result.record_result_in_flight_release(py, None)?;
        }
        Ok(())
    }
}

fn py_object_identifier(object: &Bound<'_, PyAny>) -> usize {
    object.as_ptr() as usize
}

fn result_work_item_host_dosage_buffer_owner(py: Python<'_>, work_item: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let host_dosage_buffer = work_item.getattr("host_dosage_buffer")?;
    if host_dosage_buffer.is_none() {
        return Ok(host_dosage_buffer.unbind());
    }
    dosage_buffer_owner(py, &host_dosage_buffer)
}

fn result_work_item_release_in_flight_slot(work_item: &Bound<'_, PyAny>) -> PyResult<bool> {
    work_item.getattr("release_in_flight_slot")?.extract::<bool>()
}

fn dosage_buffer_owner(py: Python<'_>, dosage_buffer: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut dosage_buffer_owner = dosage_buffer.clone().unbind();
    loop {
        let next_owner = {
            let dosage_buffer_owner_bound = dosage_buffer_owner.bind(py);
            let Ok(dosage_buffer_base) = dosage_buffer_owner_bound.getattr("base") else {
                return Ok(dosage_buffer_owner);
            };
            if dosage_buffer_base.is_none() || !is_numpy_ndarray(&dosage_buffer_base)? {
                None
            } else {
                Some(dosage_buffer_base.unbind())
            }
        };
        let Some(next_owner) = next_owner else {
            return Ok(dosage_buffer_owner);
        };
        dosage_buffer_owner = next_owner;
    }
}

fn is_numpy_ndarray(object: &Bound<'_, PyAny>) -> PyResult<bool> {
    let object_type = object.get_type();
    if object_type.name()?.to_string_lossy() != "ndarray" {
        return Ok(false);
    }
    let module_name = object_type.getattr("__module__")?.extract::<String>()?;
    Ok(module_name == "numpy")
}

fn dosage_buffer_reuse_slice_tuple<'py>(py: Python<'py>, slice_dimensions: &[usize]) -> PyResult<Bound<'py, PyTuple>> {
    let mut slices = Vec::with_capacity(slice_dimensions.len());
    for &slice_dimension in slice_dimensions {
        let slice_stop = isize::try_from(slice_dimension)
            .map_err(|_| PyRuntimeError::new_err("Native dosage-buffer reuse slice dimension exceeds isize."))?;
        slices.push(PySlice::new(py, 0, slice_stop, 1));
    }
    PyTuple::new(py, slices)
}

fn dosage_work_item_metadata_items(py: Python<'_>, work_item: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
    if python_type_name(work_item)?.as_str() != "PreprocessedVariantMajorDosageChunkBatchWorkItem" {
        return Ok(PyTuple::new(py, [work_item.getattr("metadata")?.unbind()])?.unbind());
    }
    let work_items = work_item.getattr("work_items")?;
    let mut metadata_items = Vec::with_capacity(work_items.len()?);
    for item_index in 0..work_items.len()? {
        let chunk_work_item = work_items.get_item(item_index)?;
        metadata_items.push(chunk_work_item.getattr("metadata")?.unbind());
    }
    Ok(PyTuple::new(py, metadata_items)?.unbind())
}

struct DosageWorkItemDescriptor {
    dosage_work_item_kind: &'static str,
    chunk_count: usize,
}

fn classify_dosage_work_item(work_item: &Bound<'_, PyAny>) -> PyResult<DosageWorkItemDescriptor> {
    if work_item.is_none() {
        return Ok(DosageWorkItemDescriptor {
            dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL,
            chunk_count: 0,
        });
    }
    let descriptor = match python_type_name(work_item)?.as_str() {
        "PreprocessedDosageChunkWorkItem" => DosageWorkItemDescriptor {
            dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE,
            chunk_count: 1,
        },
        "PreprocessedVariantMajorDosageChunkWorkItem" => DosageWorkItemDescriptor {
            dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE,
            chunk_count: 1,
        },
        "PreprocessedVariantMajorDosageChunkBatchWorkItem" => DosageWorkItemDescriptor {
            dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH,
            chunk_count: work_item.getattr("work_items")?.len()?,
        },
        "PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem" => DosageWorkItemDescriptor {
            dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR,
            chunk_count: 1,
        },
        type_name => {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported preprocessed dosage work item type: {type_name}"
            )));
        }
    };
    Ok(descriptor)
}

fn classify_dosage_work_item_kind(work_item: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    Ok(classify_dosage_work_item(work_item)?.dosage_work_item_kind)
}

fn classify_result_write_item_kind(work_item: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    if work_item.is_none() {
        return Ok(RESULT_WRITE_ITEM_KIND_STOP_SIGNAL);
    }
    match python_type_name(work_item)?.as_str() {
        "Regenie2ResultWriteWorkItem" => Ok(RESULT_WRITE_ITEM_KIND_SINGLE_RESULT),
        "Regenie2MultiResultWriteWorkItem" => Ok(RESULT_WRITE_ITEM_KIND_MULTI_RESULT),
        type_name => Err(PyRuntimeError::new_err(format!("Unsupported result write work item type: {type_name}"))),
    }
}

fn python_type_name(object: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(object.get_type().name()?.to_string_lossy().into_owned())
}

fn normalize_timeout_duration(timeout_seconds: f64) -> Duration {
    if timeout_seconds.is_finite() && timeout_seconds > 0.0 {
        Duration::try_from_secs_f64(timeout_seconds).unwrap_or(Duration::MAX)
    } else {
        Duration::ZERO
    }
}

fn remaining_timeout_seconds(deadline: Instant) -> f64 {
    deadline.saturating_duration_since(Instant::now()).as_secs_f64()
}
