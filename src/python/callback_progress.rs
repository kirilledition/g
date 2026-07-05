//! PyO3 adapters for callback progress state.

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

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

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
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

impl NativeCallbackProgressCompletion {
    pub(crate) fn telemetry_event_value(&self) -> NativeCallbackProgressTelemetryEvent {
        self.inner.telemetry_event().into()
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
        self.telemetry_event_value()
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
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_callback_chunk_identity(
    chromosome: String,
    variant_start_index: i64,
    variant_stop_index: i64,
) -> NativeCallbackChunkIdentity {
    native_callback_progress::CallbackChunkIdentity::new(chromosome, variant_start_index, variant_stop_index).into()
}

#[pyfunction]
pub(crate) fn emit_callback_progress_update_telemetry(
    telemetry_session: &Bound<'_, PyAny>,
    progress_update: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if progress_update.is_none() {
        return Ok(());
    }
    let Some(native_telemetry_session) = require_native_telemetry_session(
        telemetry_session,
        "Native callback progress plan selected a missing telemetry session.",
    )?
    else {
        return Ok(());
    };

    let telemetry_plan = progress_update.getattr("telemetry_plan")?;
    let progress_events = telemetry_plan.getattr("events")?;
    for progress_event in progress_events.try_iter()? {
        native_telemetry_session.call_method1("emit_callback_progress_event", (progress_event?,))?;
    }

    let py = telemetry_session.py();
    let progress_record = telemetry_plan.getattr("progress")?;
    let progress_fields = PyDict::new(py);
    progress_fields.set_item("chromosome", progress_record.getattr("chromosome")?)?;
    progress_fields.set_item("chunk_identifier", progress_record.getattr("chunk_identifier")?)?;
    progress_fields.set_item("variant_start_index", progress_record.getattr("variant_start_index")?)?;
    progress_fields.set_item("variant_stop_index", progress_record.getattr("variant_stop_index")?)?;
    progress_fields.set_item("variant_count", progress_record.getattr("variant_count")?)?;
    native_telemetry_session
        .call_method1("emit_progress", (progress_record.getattr("processed_chunk_count")?, progress_fields))?;
    Ok(())
}

#[pyfunction]
pub(crate) fn emit_callback_progress_event_telemetry(
    telemetry_session: &Bound<'_, PyAny>,
    progress_event: &Bound<'_, PyAny>,
    missing_session_message: &str,
) -> PyResult<()> {
    if progress_event.is_none() {
        return Ok(());
    }
    let Some(native_telemetry_session) = require_native_telemetry_session(telemetry_session, missing_session_message)?
    else {
        return Ok(());
    };
    native_telemetry_session.call_method1("emit_callback_progress_event", (progress_event,))?;
    Ok(())
}

#[pyfunction]
pub(crate) fn emit_callback_progress_completion_telemetry(
    telemetry_session: &Bound<'_, PyAny>,
    progress_completion: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if telemetry_session.is_none() || progress_completion.is_none() {
        return Ok(());
    }
    let Some(native_telemetry_session) = optional_native_telemetry_session(telemetry_session.py(), telemetry_session)?
    else {
        return Ok(());
    };
    let progress_event = progress_completion.getattr("telemetry_event")?;
    native_telemetry_session.call_method1("emit_callback_progress_event", (progress_event,))?;
    Ok(())
}

fn require_native_telemetry_session<'py>(
    telemetry_session: &Bound<'py, PyAny>,
    missing_session_message: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if telemetry_session.is_none() {
        return Err(PyRuntimeError::new_err(missing_session_message.to_owned()));
    }
    optional_native_telemetry_session(telemetry_session.py(), telemetry_session)
}

fn optional_native_telemetry_session<'py>(
    py: Python<'py>,
    telemetry_session: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if telemetry_session.is_none() {
        return Ok(None);
    }
    match telemetry_session.getattr("native_telemetry_session") {
        Ok(native_telemetry_session) if native_telemetry_session.is_none() => Ok(None),
        Ok(native_telemetry_session) => Ok(Some(native_telemetry_session)),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Err(PyTypeError::new_err(
            "callback progress telemetry requires a TelemetrySession with a native telemetry session handle.",
        )),
        Err(error) => Err(error),
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCallbackChunkIdentity>()?;
    module.add_class::<NativeCallbackProgressCompletion>()?;
    module.add_class::<NativeCallbackProgressState>()?;
    module.add_class::<NativeCallbackProgressTelemetryEvent>()?;
    module.add_class::<NativeCallbackProgressTelemetryPlan>()?;
    module.add_class::<NativeCallbackProgressTelemetryRecord>()?;
    module.add_class::<NativeCallbackProgressUpdate>()?;
    module.add_function(wrap_pyfunction!(build_callback_chunk_identity, module)?)?;
    module.add_function(wrap_pyfunction!(emit_callback_progress_update_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(emit_callback_progress_event_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(emit_callback_progress_completion_telemetry, module)?)?;
    Ok(())
}
