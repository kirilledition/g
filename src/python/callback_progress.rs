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
}

#[pymethods]
impl NativeCallbackProgressState {
    #[new]
    fn new() -> Self {
        Self { inner: native_callback_progress::CallbackProgressState::new() }
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
