//! PyO3 adapter for Python-backed association backends.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use g_engine::{
    AssociationBackend, AssociationBatchResult, BackendError, EngineCoordinator, EngineError, EngineRunInput,
    EngineRunReport, GenotypeBatchView, PredictionView, PreparedGroupInput,
};

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativePreparedGroupInput {
    inner: PreparedGroupInput,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativePredictionView {
    chromosome: String,
    row_count: usize,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeGenotypeBatchView {
    chromosome: String,
    variant_count: usize,
    variant_offset: usize,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeAssociationBatchResult {
    inner: AssociationBatchResult,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeAssociationEngineRunReport {
    inner: EngineRunReport,
}

#[pyclass]
pub(crate) struct NativePythonAssociationBackend {
    backend: Py<PyAny>,
}

#[pymethods]
impl NativePreparedGroupInput {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(group_identifier: String, phenotype_count: usize) -> Self {
        Self { inner: PreparedGroupInput::new(group_identifier, phenotype_count) }
    }

    #[getter]
    fn group_identifier(&self) -> &str {
        self.inner.group_identifier.as_str()
    }

    #[getter]
    fn phenotype_count(&self) -> usize {
        self.inner.phenotype_count
    }
}

#[pymethods]
impl NativePredictionView {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(chromosome: String, row_count: usize) -> Self {
        Self { chromosome, row_count }
    }

    #[getter]
    fn chromosome(&self) -> &str {
        self.chromosome.as_str()
    }

    #[getter]
    fn row_count(&self) -> usize {
        self.row_count
    }
}

#[pymethods]
impl NativeGenotypeBatchView {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(chromosome: String, variant_count: usize, variant_offset: usize) -> Self {
        Self { chromosome, variant_count, variant_offset }
    }

    #[getter]
    fn chromosome(&self) -> &str {
        self.chromosome.as_str()
    }

    #[getter]
    fn variant_count(&self) -> usize {
        self.variant_count
    }

    #[getter]
    fn variant_offset(&self) -> usize {
        self.variant_offset
    }
}

#[pymethods]
impl NativeAssociationBatchResult {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(chromosome: String, variant_count: usize, statistic_sum: f64) -> Self {
        Self { inner: AssociationBatchResult::new(chromosome, variant_count, statistic_sum) }
    }

    #[getter]
    fn chromosome(&self) -> &str {
        self.inner.chromosome.as_str()
    }

    #[getter]
    fn variant_count(&self) -> usize {
        self.inner.variant_count
    }

    #[getter]
    fn statistic_sum(&self) -> f64 {
        self.inner.statistic_sum
    }
}

impl NativeAssociationBatchResult {
    fn native_result(&self) -> AssociationBatchResult {
        self.inner.clone()
    }
}

#[pymethods]
impl NativeAssociationEngineRunReport {
    #[getter]
    fn phase_history(&self) -> Vec<String> {
        self.inner.phase_history.iter().map(ToString::to_string).collect()
    }

    #[getter]
    fn result(&self) -> NativeAssociationBatchResult {
        NativeAssociationBatchResult { inner: self.inner.result.clone() }
    }
}

#[pymethods]
impl NativePythonAssociationBackend {
    #[new]
    fn new(backend: Py<PyAny>) -> Self {
        Self { backend }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn prepare_group(&mut self, group_identifier: String, phenotype_count: usize) -> PyResult<Py<PyAny>> {
        let input = PreparedGroupInput::new(group_identifier, phenotype_count);
        AssociationBackend::prepare_group(self, &input).map_err(|error| backend_error_to_py_runtime_error(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn prepare_chromosome(
        &mut self,
        group_state: Py<PyAny>,
        chromosome: String,
        prediction_chromosome: String,
        prediction_row_count: usize,
    ) -> PyResult<Py<PyAny>> {
        let predictions = PredictionView::new(prediction_chromosome.as_str(), prediction_row_count);
        AssociationBackend::prepare_chromosome(self, &group_state, chromosome.as_str(), predictions)
            .map_err(|error| backend_error_to_py_runtime_error(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn compute_batch(
        &mut self,
        chromosome_state: Py<PyAny>,
        batch_chromosome: String,
        variant_count: usize,
        variant_offset: usize,
    ) -> PyResult<NativeAssociationBatchResult> {
        let batch = GenotypeBatchView::new(batch_chromosome.as_str(), variant_count, variant_offset);
        AssociationBackend::compute_batch(self, &chromosome_state, batch)
            .map(|result| NativeAssociationBatchResult { inner: result })
            .map_err(|error| backend_error_to_py_runtime_error(&error))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn run_single_batch(
        &self,
        py: Python<'_>,
        group_identifier: String,
        phenotype_count: usize,
        chromosome: String,
        prediction_chromosome: String,
        prediction_row_count: usize,
        batch_chromosome: String,
        variant_count: usize,
        variant_offset: usize,
    ) -> PyResult<NativeAssociationEngineRunReport> {
        let group = PreparedGroupInput::new(group_identifier, phenotype_count);
        let predictions = PredictionView::new(prediction_chromosome.as_str(), prediction_row_count);
        let batch = GenotypeBatchView::new(batch_chromosome.as_str(), variant_count, variant_offset);
        let input = EngineRunInput::new(group, chromosome.as_str(), predictions, batch);
        let backend = Self { backend: self.backend.clone_ref(py) };
        let mut coordinator = EngineCoordinator::new(backend);
        coordinator
            .run_single_batch(&input)
            .map(|report| NativeAssociationEngineRunReport { inner: report })
            .map_err(|error| engine_error_to_py_runtime_error(&error))
    }
}

impl AssociationBackend for NativePythonAssociationBackend {
    type GroupState = Py<PyAny>;
    type ChromosomeState = Py<PyAny>;

    fn prepare_group(&mut self, input: &PreparedGroupInput) -> Result<Self::GroupState, BackendError> {
        Python::attach(|py| -> PyResult<Py<PyAny>> {
            let native_input = Py::new(py, NativePreparedGroupInput::from(input))?;
            Ok(self.backend.bind(py).call_method1("prepare_group", (native_input.bind(py),))?.unbind())
        })
        .map_err(|error| backend_error_from_py_error(&error))
    }

    fn prepare_chromosome(
        &mut self,
        group: &Self::GroupState,
        chromosome: &str,
        predictions: PredictionView<'_>,
    ) -> Result<Self::ChromosomeState, BackendError> {
        Python::attach(|py| -> PyResult<Py<PyAny>> {
            let native_predictions = Py::new(py, NativePredictionView::from(predictions))?;
            Ok(self
                .backend
                .bind(py)
                .call_method1("prepare_chromosome", (group.bind(py), chromosome, native_predictions.bind(py)))?
                .unbind())
        })
        .map_err(|error| backend_error_from_py_error(&error))
    }

    fn compute_batch(
        &mut self,
        chromosome: &Self::ChromosomeState,
        batch: GenotypeBatchView<'_>,
    ) -> Result<AssociationBatchResult, BackendError> {
        Python::attach(|py| -> PyResult<AssociationBatchResult> {
            let native_batch = Py::new(py, NativeGenotypeBatchView::from(batch))?;
            let result_object =
                self.backend.bind(py).call_method1("compute_batch", (chromosome.bind(py), native_batch.bind(py)))?;
            let native_result: PyRef<'_, NativeAssociationBatchResult> = result_object.extract().map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "python association backend compute_batch must return NativeAssociationBatchResult: {error}"
                ))
            })?;
            Ok(native_result.native_result())
        })
        .map_err(|error| backend_error_from_py_error(&error))
    }
}

impl From<&PreparedGroupInput> for NativePreparedGroupInput {
    fn from(input: &PreparedGroupInput) -> Self {
        Self { inner: input.clone() }
    }
}

impl From<PredictionView<'_>> for NativePredictionView {
    fn from(view: PredictionView<'_>) -> Self {
        Self { chromosome: view.chromosome.to_owned(), row_count: view.row_count }
    }
}

impl From<GenotypeBatchView<'_>> for NativeGenotypeBatchView {
    fn from(view: GenotypeBatchView<'_>) -> Self {
        Self {
            chromosome: view.chromosome.to_owned(),
            variant_count: view.variant_count,
            variant_offset: view.variant_offset,
        }
    }
}

fn backend_error_from_py_error(error: &PyErr) -> BackendError {
    BackendError::new(error.to_string())
}

fn backend_error_to_py_runtime_error(error: &BackendError) -> PyErr {
    PyRuntimeError::new_err(error.message().to_owned())
}

fn engine_error_to_py_runtime_error(error: &EngineError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
