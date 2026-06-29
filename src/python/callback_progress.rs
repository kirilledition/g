//! PyO3 adapters for callback progress state.

use pyo3::prelude::*;

use g_engine::callback_progress as native_callback_progress;

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeCallbackChunkIdentity {
    inner: native_callback_progress::CallbackChunkIdentity,
}

#[pyclass]
pub(crate) struct NativeCallbackProgressUpdate {
    inner: native_callback_progress::CallbackProgressUpdate,
}

#[pyclass]
pub(crate) struct NativeCallbackProgressTelemetryEvent {
    inner: native_callback_progress::CallbackProgressTelemetryEvent,
}

#[pyclass]
pub(crate) struct NativeCallbackProgressTelemetryRecord {
    inner: native_callback_progress::CallbackProgressTelemetryRecord,
}

#[pyclass]
pub(crate) struct NativeCallbackProgressTelemetryPlan {
    inner: native_callback_progress::CallbackProgressTelemetryPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackProgressCompletion {
    inner: native_callback_progress::CallbackProgressCompletion,
}

#[pyclass]
pub(crate) struct NativeCallbackProgressState {
    inner: native_callback_progress::CallbackProgressState,
}

#[pymethods]
impl NativeCallbackChunkIdentity {
    #[getter]
    fn chunk_identifier(&self) -> i64 {
        self.inner.chunk_identifier
    }

    #[getter]
    fn chromosome(&self) -> String {
        self.inner.chromosome.clone()
    }

    #[getter]
    fn variant_start_index(&self) -> i64 {
        self.inner.variant_start_index
    }

    #[getter]
    fn variant_stop_index(&self) -> i64 {
        self.inner.variant_stop_index
    }

    #[getter]
    fn variant_count(&self) -> i64 {
        self.inner.variant_count
    }
}

#[pymethods]
impl NativeCallbackProgressUpdate {
    #[getter]
    fn processed_chunk_count(&self) -> i64 {
        self.inner.processed_chunk_count
    }

    #[getter]
    fn completed_chromosome(&self) -> Option<String> {
        self.inner.completed_chromosome.clone()
    }

    #[getter]
    fn completed_processed_chunk_count(&self) -> Option<i64> {
        self.inner.completed_processed_chunk_count
    }

    #[getter]
    fn started_chromosome(&self) -> Option<String> {
        self.inner.started_chromosome.clone()
    }

    #[getter]
    fn chunk_identity(&self) -> NativeCallbackChunkIdentity {
        self.inner.chunk_identity.clone().into()
    }

    #[getter]
    fn telemetry_plan(&self) -> NativeCallbackProgressTelemetryPlan {
        self.inner.telemetry_plan().into()
    }
}

#[pymethods]
impl NativeCallbackProgressTelemetryEvent {
    #[getter]
    fn event_name(&self) -> String {
        self.inner.event_name.clone()
    }

    #[getter]
    fn level(&self) -> String {
        self.inner.level.clone()
    }

    #[getter]
    fn chromosome(&self) -> String {
        self.inner.chromosome.clone()
    }

    #[getter]
    fn processed_chunk_count(&self) -> i64 {
        self.inner.processed_chunk_count
    }
}

#[pymethods]
impl NativeCallbackProgressTelemetryRecord {
    #[getter]
    fn processed_chunk_count(&self) -> i64 {
        self.inner.processed_chunk_count
    }

    #[getter]
    fn chromosome(&self) -> String {
        self.inner.chromosome.clone()
    }

    #[getter]
    fn chunk_identifier(&self) -> i64 {
        self.inner.chunk_identifier
    }

    #[getter]
    fn variant_start_index(&self) -> i64 {
        self.inner.variant_start_index
    }

    #[getter]
    fn variant_stop_index(&self) -> i64 {
        self.inner.variant_stop_index
    }

    #[getter]
    fn variant_count(&self) -> i64 {
        self.inner.variant_count
    }
}

#[pymethods]
impl NativeCallbackProgressTelemetryPlan {
    #[getter]
    fn events(&self) -> Vec<NativeCallbackProgressTelemetryEvent> {
        self.inner.events.iter().cloned().map(Into::into).collect()
    }

    #[getter]
    fn progress(&self) -> NativeCallbackProgressTelemetryRecord {
        self.inner.progress.clone().into()
    }
}

#[pymethods]
impl NativeCallbackProgressCompletion {
    #[getter]
    fn chromosome(&self) -> String {
        self.inner.chromosome.clone()
    }

    #[getter]
    fn processed_chunk_count(&self) -> i64 {
        self.inner.processed_chunk_count
    }

    #[getter]
    fn telemetry_event(&self) -> NativeCallbackProgressTelemetryEvent {
        self.inner.telemetry_event().into()
    }
}

#[pymethods]
impl NativeCallbackProgressState {
    #[new]
    fn new() -> Self {
        Self::new_state()
    }

    #[getter]
    fn processed_chunk_count(&self) -> i64 {
        self.inner.processed_chunk_count()
    }

    #[getter]
    fn current_progress_chromosome(&self) -> Option<String> {
        self.inner.current_progress_chromosome().map(str::to_owned)
    }

    fn record_processed_chunk(&mut self, chunk_identity: &NativeCallbackChunkIdentity) -> NativeCallbackProgressUpdate {
        self.inner.record_processed_chunk(chunk_identity.inner.clone()).into()
    }

    fn record_processed_chunk_without_progress(&mut self) {
        self.inner.record_processed_chunk_without_progress();
    }

    fn finish_progress(&mut self) -> Option<NativeCallbackProgressCompletion> {
        self.inner.finish_progress().map(Into::into)
    }
}

impl NativeCallbackProgressState {
    pub(crate) fn new_state() -> Self {
        Self { inner: native_callback_progress::CallbackProgressState::new() }
    }

    pub(crate) fn processed_chunk_count_value(&self) -> i64 {
        self.inner.processed_chunk_count()
    }

    pub(crate) fn current_progress_chromosome_value(&self) -> Option<String> {
        self.inner.current_progress_chromosome().map(str::to_owned)
    }

    pub(crate) fn record_processed_chunk_value(
        &mut self,
        chunk_identity: &NativeCallbackChunkIdentity,
    ) -> NativeCallbackProgressUpdate {
        self.inner.record_processed_chunk(chunk_identity.inner.clone()).into()
    }

    pub(crate) fn record_processed_chunk_without_progress_value(&mut self) {
        self.inner.record_processed_chunk_without_progress();
    }

    pub(crate) fn finish_progress_value(&mut self) -> Option<NativeCallbackProgressCompletion> {
        self.inner.finish_progress().map(Into::into)
    }
}

impl From<native_callback_progress::CallbackChunkIdentity> for NativeCallbackChunkIdentity {
    fn from(chunk_identity: native_callback_progress::CallbackChunkIdentity) -> Self {
        Self { inner: chunk_identity }
    }
}

impl From<native_callback_progress::CallbackProgressUpdate> for NativeCallbackProgressUpdate {
    fn from(progress_update: native_callback_progress::CallbackProgressUpdate) -> Self {
        Self { inner: progress_update }
    }
}

impl From<native_callback_progress::CallbackProgressTelemetryEvent> for NativeCallbackProgressTelemetryEvent {
    fn from(telemetry_event: native_callback_progress::CallbackProgressTelemetryEvent) -> Self {
        Self { inner: telemetry_event }
    }
}

impl From<native_callback_progress::CallbackProgressTelemetryRecord> for NativeCallbackProgressTelemetryRecord {
    fn from(telemetry_record: native_callback_progress::CallbackProgressTelemetryRecord) -> Self {
        Self { inner: telemetry_record }
    }
}

impl From<native_callback_progress::CallbackProgressTelemetryPlan> for NativeCallbackProgressTelemetryPlan {
    fn from(telemetry_plan: native_callback_progress::CallbackProgressTelemetryPlan) -> Self {
        Self { inner: telemetry_plan }
    }
}

impl From<native_callback_progress::CallbackProgressCompletion> for NativeCallbackProgressCompletion {
    fn from(progress_completion: native_callback_progress::CallbackProgressCompletion) -> Self {
        Self { inner: progress_completion }
    }
}

#[pyfunction]
pub(crate) fn build_callback_chunk_identity(
    chromosome: String,
    variant_start_index: i64,
    variant_stop_index: i64,
) -> NativeCallbackChunkIdentity {
    native_callback_progress::CallbackChunkIdentity::new(chromosome, variant_start_index, variant_stop_index).into()
}
