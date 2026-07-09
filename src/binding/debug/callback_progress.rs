//! PyO3 adapters for callback progress state.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_engine as native_callback_progress;

#[derive(Clone)]
pub(crate) struct NativeCallbackChunkIdentity {
    inner: native_callback_progress::CallbackChunkIdentity,
}

#[pyclass]
pub(crate) struct NativeCallbackProgressUpdate {
    inner: native_callback_progress::CallbackProgressUpdate,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeCallbackProgressTelemetryEvent {
    inner: native_callback_progress::CallbackProgressTelemetryEvent,
}

pub(crate) struct NativeCallbackProgressTelemetryRecord {
    inner: native_callback_progress::CallbackProgressTelemetryRecord,
}

pub(crate) struct NativeCallbackProgressTelemetryPlan {
    inner: native_callback_progress::CallbackProgressTelemetryPlan,
}

pub(crate) struct NativeCallbackProgressCompletion {
    inner: native_callback_progress::CallbackProgressCompletion,
}

#[pyclass]
pub(crate) struct NativeCallbackProgressState {
    inner: native_callback_progress::CallbackProgressState,
}

impl NativeCallbackProgressUpdate {
    pub(crate) fn telemetry_plan_value(&self) -> NativeCallbackProgressTelemetryPlan {
        self.inner.telemetry_plan().into()
    }
}

impl NativeCallbackProgressTelemetryEvent {
    pub(crate) fn event_name_value(&self) -> &str {
        self.inner.event_name.as_str()
    }

    pub(crate) fn level_value(&self) -> &str {
        self.inner.level.as_str()
    }

    pub(crate) fn chromosome_value(&self) -> &str {
        self.inner.chromosome.as_str()
    }

    pub(crate) const fn processed_chunk_count_value(&self) -> i64 {
        self.inner.processed_chunk_count
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

impl NativeCallbackProgressTelemetryRecord {
    pub(crate) const fn processed_chunk_count_value(&self) -> i64 {
        self.inner.processed_chunk_count
    }

    pub(crate) fn chromosome_value(&self) -> &str {
        self.inner.chromosome.as_str()
    }

    pub(crate) const fn chunk_identifier_value(&self) -> i64 {
        self.inner.chunk_identifier
    }

    pub(crate) const fn variant_start_index_value(&self) -> i64 {
        self.inner.variant_start_index
    }

    pub(crate) const fn variant_stop_index_value(&self) -> i64 {
        self.inner.variant_stop_index
    }

    pub(crate) const fn variant_count_value(&self) -> i64 {
        self.inner.variant_count
    }
}

impl NativeCallbackProgressTelemetryPlan {
    pub(crate) fn event_values(&self) -> Vec<NativeCallbackProgressTelemetryEvent> {
        self.inner.events.iter().cloned().map(Into::into).collect()
    }

    pub(crate) fn progress_value(&self) -> NativeCallbackProgressTelemetryRecord {
        self.inner.progress.clone().into()
    }
}

impl NativeCallbackProgressCompletion {
    pub(crate) fn telemetry_event_value(&self) -> NativeCallbackProgressTelemetryEvent {
        self.inner.telemetry_event().into()
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

#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_callback_chunk_identity(
    chromosome: String,
    variant_start_index: i64,
    variant_stop_index: i64,
) -> NativeCallbackChunkIdentity {
    native_callback_progress::CallbackChunkIdentity::new(chromosome, variant_start_index, variant_stop_index).into()
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCallbackProgressTelemetryEvent>()?;
    module.add_class::<NativeCallbackProgressUpdate>()?;
    Ok(())
}
