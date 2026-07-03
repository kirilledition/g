//! PyO3 adapter for Python-backed association backends.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use g_engine::{
    AssociationBackend, AssociationBatchResult, BackendError, EngineChromosomeRunInput, EngineChromosomeRunReport,
    EngineCoordinator, EngineEffectError, EngineError, EngineGroupChromosomeInput, EngineGroupRunInput,
    EngineGroupRunReport, EngineRunEffects, EngineRunInput, EngineRunReport, GenotypeBatchView, PredictionView,
    PreparedGroupInput, RunPhase,
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
pub(crate) struct NativeAssociationChromosomeRunInput {
    chromosome: String,
    prediction_chromosome: String,
    prediction_row_count: usize,
    batches: Vec<NativeGenotypeBatchView>,
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

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeAssociationChromosomeRunReport {
    inner: EngineChromosomeRunReport,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeAssociationGroupRunReport {
    inner: EngineGroupRunReport,
}

#[pyclass]
pub(crate) struct NativePythonEngineRunEffects {
    effects: Py<PyAny>,
}

#[pyclass]
pub(crate) struct NativePythonAssociationBackend {
    backend: Py<PyAny>,
}

const REQUIRED_ENGINE_RUN_EFFECT_METHODS: &[&str] = &[
    "emit_phase_event",
    "open_inputs",
    "align_inputs",
    "validate_preflight",
    "validate_output_compatibility",
    "construct_writers",
    "write_batch_result",
    "drain_writers",
    "finalize_outputs",
    "abort_outputs",
];

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
impl NativeAssociationChromosomeRunInput {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new<'py>(
        chromosome: String,
        prediction_chromosome: String,
        prediction_row_count: usize,
        batches: Vec<PyRef<'py, NativeGenotypeBatchView>>,
    ) -> Self {
        let batches = batches.iter().map(|batch| (*batch).clone()).collect();
        Self { chromosome, prediction_chromosome, prediction_row_count, batches }
    }

    #[getter]
    fn chromosome(&self) -> &str {
        self.chromosome.as_str()
    }

    #[getter]
    fn prediction_chromosome(&self) -> &str {
        self.prediction_chromosome.as_str()
    }

    #[getter]
    fn prediction_row_count(&self) -> usize {
        self.prediction_row_count
    }

    #[getter]
    fn batches(&self) -> Vec<NativeGenotypeBatchView> {
        self.batches.clone()
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
impl NativeAssociationChromosomeRunReport {
    #[getter]
    fn phase_history(&self) -> Vec<String> {
        self.inner.phase_history.iter().map(ToString::to_string).collect()
    }

    #[getter]
    fn results(&self) -> Vec<NativeAssociationBatchResult> {
        self.inner.results.iter().cloned().map(|inner| NativeAssociationBatchResult { inner }).collect()
    }
}

#[pymethods]
impl NativeAssociationGroupRunReport {
    #[getter]
    fn phase_history(&self) -> Vec<String> {
        self.inner.phase_history.iter().map(ToString::to_string).collect()
    }

    #[getter]
    fn results(&self) -> Vec<NativeAssociationBatchResult> {
        self.inner.results.iter().cloned().map(|inner| NativeAssociationBatchResult { inner }).collect()
    }
}

#[pymethods]
impl NativePythonEngineRunEffects {
    #[new]
    fn new(py: Python<'_>, effects: Py<PyAny>) -> PyResult<Self> {
        validate_python_engine_run_effect_methods(py, &effects)?;
        Ok(Self { effects })
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
        self.run_single_batch_impl(
            py,
            group_identifier,
            phenotype_count,
            chromosome,
            prediction_chromosome,
            prediction_row_count,
            batch_chromosome,
            variant_count,
            variant_offset,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn run_single_batch_with_effects<'py>(
        &self,
        py: Python<'py>,
        group_identifier: String,
        phenotype_count: usize,
        chromosome: String,
        prediction_chromosome: String,
        prediction_row_count: usize,
        batch_chromosome: String,
        variant_count: usize,
        variant_offset: usize,
        effects: PyRef<'py, NativePythonEngineRunEffects>,
    ) -> PyResult<NativeAssociationEngineRunReport> {
        let mut native_effects = NativePythonEngineRunEffects { effects: effects.effects.clone_ref(py) };
        self.run_single_batch_impl(
            py,
            group_identifier,
            phenotype_count,
            chromosome,
            prediction_chromosome,
            prediction_row_count,
            batch_chromosome,
            variant_count,
            variant_offset,
            Some(&mut native_effects),
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn run_chromosome_batches<'py>(
        &self,
        py: Python<'py>,
        group_identifier: String,
        phenotype_count: usize,
        chromosome: String,
        prediction_chromosome: String,
        prediction_row_count: usize,
        batches: Vec<PyRef<'py, NativeGenotypeBatchView>>,
    ) -> PyResult<NativeAssociationChromosomeRunReport> {
        self.run_chromosome_batches_impl(
            py,
            group_identifier,
            phenotype_count,
            chromosome,
            prediction_chromosome,
            prediction_row_count,
            batches,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn run_chromosome_batches_with_effects<'py>(
        &self,
        py: Python<'py>,
        group_identifier: String,
        phenotype_count: usize,
        chromosome: String,
        prediction_chromosome: String,
        prediction_row_count: usize,
        batches: Vec<PyRef<'py, NativeGenotypeBatchView>>,
        effects: PyRef<'py, NativePythonEngineRunEffects>,
    ) -> PyResult<NativeAssociationChromosomeRunReport> {
        let mut native_effects = NativePythonEngineRunEffects { effects: effects.effects.clone_ref(py) };
        self.run_chromosome_batches_impl(
            py,
            group_identifier,
            phenotype_count,
            chromosome,
            prediction_chromosome,
            prediction_row_count,
            batches,
            Some(&mut native_effects),
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    fn run_group_chromosomes<'py>(
        &self,
        py: Python<'py>,
        group_identifier: String,
        phenotype_count: usize,
        chromosome_inputs: Vec<PyRef<'py, NativeAssociationChromosomeRunInput>>,
    ) -> PyResult<NativeAssociationGroupRunReport> {
        self.run_group_chromosomes_impl(py, group_identifier, phenotype_count, chromosome_inputs, None)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn run_group_chromosomes_with_effects<'py>(
        &self,
        py: Python<'py>,
        group_identifier: String,
        phenotype_count: usize,
        chromosome_inputs: Vec<PyRef<'py, NativeAssociationChromosomeRunInput>>,
        effects: PyRef<'py, NativePythonEngineRunEffects>,
    ) -> PyResult<NativeAssociationGroupRunReport> {
        let mut native_effects = NativePythonEngineRunEffects { effects: effects.effects.clone_ref(py) };
        self.run_group_chromosomes_impl(
            py,
            group_identifier,
            phenotype_count,
            chromosome_inputs,
            Some(&mut native_effects),
        )
    }
}

impl NativePythonAssociationBackend {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn run_single_batch_impl(
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
        effects: Option<&mut NativePythonEngineRunEffects>,
    ) -> PyResult<NativeAssociationEngineRunReport> {
        let group = PreparedGroupInput::new(group_identifier, phenotype_count);
        let predictions = PredictionView::new(prediction_chromosome.as_str(), prediction_row_count);
        let batch = GenotypeBatchView::new(batch_chromosome.as_str(), variant_count, variant_offset);
        let input = EngineRunInput::new(group, chromosome.as_str(), predictions, batch);
        let backend = Self { backend: self.backend.clone_ref(py) };
        let mut coordinator = EngineCoordinator::new(backend);
        let report = match effects {
            Some(effects) => coordinator.run_single_batch_with_effects(&input, effects),
            None => coordinator.run_single_batch(&input),
        };
        report
            .map(|inner| NativeAssociationEngineRunReport { inner })
            .map_err(|error| engine_error_to_py_runtime_error(&error))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn run_chromosome_batches_impl<'py>(
        &self,
        py: Python<'py>,
        group_identifier: String,
        phenotype_count: usize,
        chromosome: String,
        prediction_chromosome: String,
        prediction_row_count: usize,
        batches: Vec<PyRef<'py, NativeGenotypeBatchView>>,
        effects: Option<&mut NativePythonEngineRunEffects>,
    ) -> PyResult<NativeAssociationChromosomeRunReport> {
        let group = PreparedGroupInput::new(group_identifier, phenotype_count);
        let predictions = PredictionView::new(prediction_chromosome.as_str(), prediction_row_count);
        let batch_chromosomes = batches.iter().map(|batch| batch.chromosome.clone()).collect::<Vec<_>>();
        let batch_views = batches
            .iter()
            .zip(&batch_chromosomes)
            .map(|(batch, batch_chromosome)| {
                GenotypeBatchView::new(batch_chromosome.as_str(), batch.variant_count, batch.variant_offset)
            })
            .collect::<Vec<_>>();
        let input = EngineChromosomeRunInput::new(group, chromosome.as_str(), predictions, batch_views);
        let backend = Self { backend: self.backend.clone_ref(py) };
        let mut coordinator = EngineCoordinator::new(backend);
        let report = match effects {
            Some(effects) => coordinator.run_chromosome_batches_with_effects(&input, effects),
            None => coordinator.run_chromosome_batches(&input),
        };
        report
            .map(|inner| NativeAssociationChromosomeRunReport { inner })
            .map_err(|error| engine_error_to_py_runtime_error(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn run_group_chromosomes_impl<'py>(
        &self,
        py: Python<'py>,
        group_identifier: String,
        phenotype_count: usize,
        chromosome_inputs: Vec<PyRef<'py, NativeAssociationChromosomeRunInput>>,
        effects: Option<&mut NativePythonEngineRunEffects>,
    ) -> PyResult<NativeAssociationGroupRunReport> {
        let group = PreparedGroupInput::new(group_identifier, phenotype_count);
        let chromosome_names = chromosome_inputs.iter().map(|input| input.chromosome.clone()).collect::<Vec<_>>();
        let prediction_chromosome_names =
            chromosome_inputs.iter().map(|input| input.prediction_chromosome.clone()).collect::<Vec<_>>();
        let batch_chromosome_names = chromosome_inputs
            .iter()
            .map(|input| input.batches.iter().map(|batch| batch.chromosome.clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let chromosomes = chromosome_inputs
            .iter()
            .enumerate()
            .map(|(chromosome_index, chromosome_input)| {
                let batches = chromosome_input
                    .batches
                    .iter()
                    .zip(&batch_chromosome_names[chromosome_index])
                    .map(|(batch, batch_chromosome)| {
                        GenotypeBatchView::new(batch_chromosome.as_str(), batch.variant_count, batch.variant_offset)
                    })
                    .collect::<Vec<_>>();
                let predictions = PredictionView::new(
                    prediction_chromosome_names[chromosome_index].as_str(),
                    chromosome_input.prediction_row_count,
                );
                EngineGroupChromosomeInput::new(chromosome_names[chromosome_index].as_str(), predictions, batches)
            })
            .collect::<Vec<_>>();
        let input = EngineGroupRunInput::new(group, chromosomes);
        let backend = Self { backend: self.backend.clone_ref(py) };
        let mut coordinator = EngineCoordinator::new(backend);
        let report = match effects {
            Some(effects) => coordinator.run_group_chromosomes_with_effects(&input, effects),
            None => coordinator.run_group_chromosomes(&input),
        };
        report
            .map(|inner| NativeAssociationGroupRunReport { inner })
            .map_err(|error| engine_error_to_py_runtime_error(&error))
    }
}

impl EngineRunEffects for NativePythonEngineRunEffects {
    fn emit_phase_event(&mut self, phase: RunPhase) -> Result<(), EngineEffectError> {
        self.call_effect_method1("emit_phase_event", phase.to_string())
    }

    fn open_inputs(&mut self) -> Result<(), EngineEffectError> {
        self.call_effect_method0("open_inputs")
    }

    fn align_inputs(&mut self) -> Result<(), EngineEffectError> {
        self.call_effect_method0("align_inputs")
    }

    fn validate_preflight(&mut self) -> Result<(), EngineEffectError> {
        self.call_effect_method0("validate_preflight")
    }

    fn validate_output_compatibility(&mut self) -> Result<(), EngineEffectError> {
        self.call_effect_method0("validate_output_compatibility")
    }

    fn construct_writers(&mut self) -> Result<(), EngineEffectError> {
        self.call_effect_method0("construct_writers")
    }

    fn write_batch_result(&mut self, result: &AssociationBatchResult) -> Result<(), EngineEffectError> {
        Python::attach(|py| -> PyResult<()> {
            let effects = self.effects.bind(py);
            let native_result = Py::new(py, NativeAssociationBatchResult { inner: result.clone() })?;
            effects.call_method1("write_batch_result", (native_result.bind(py),))?;
            Ok(())
        })
        .map_err(|error| engine_effect_error_from_py_error(&error))
    }

    fn drain_writers(&mut self) -> Result<(), EngineEffectError> {
        self.call_effect_method0("drain_writers")
    }

    fn finalize_outputs(&mut self) -> Result<(), EngineEffectError> {
        self.call_effect_method0("finalize_outputs")
    }

    fn abort_outputs(&mut self, phase: RunPhase) {
        let _ = self.call_effect_method1("abort_outputs", phase.to_string());
    }
}

impl NativePythonEngineRunEffects {
    fn call_effect_method0(&self, method_name: &str) -> Result<(), EngineEffectError> {
        Python::attach(|py| -> PyResult<()> {
            let effects = self.effects.bind(py);
            effects.call_method0(method_name)?;
            Ok(())
        })
        .map_err(|error| engine_effect_error_from_py_error(&error))
    }

    fn call_effect_method1(&self, method_name: &str, argument: String) -> Result<(), EngineEffectError> {
        Python::attach(|py| -> PyResult<()> {
            let effects = self.effects.bind(py);
            effects.call_method1(method_name, (argument,))?;
            Ok(())
        })
        .map_err(|error| engine_effect_error_from_py_error(&error))
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

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeAssociationBatchResult>()?;
    module.add_class::<NativeAssociationChromosomeRunInput>()?;
    module.add_class::<NativeAssociationChromosomeRunReport>()?;
    module.add_class::<NativeAssociationEngineRunReport>()?;
    module.add_class::<NativeAssociationGroupRunReport>()?;
    module.add_class::<NativeGenotypeBatchView>()?;
    module.add_class::<NativePredictionView>()?;
    module.add_class::<NativePreparedGroupInput>()?;
    module.add_class::<NativePythonAssociationBackend>()?;
    module.add_class::<NativePythonEngineRunEffects>()?;
    Ok(())
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

fn engine_effect_error_from_py_error(error: &PyErr) -> EngineEffectError {
    EngineEffectError::new(error.to_string())
}

fn validate_python_engine_run_effect_methods(py: Python<'_>, effects: &Py<PyAny>) -> PyResult<()> {
    let effects = effects.bind(py);
    for method_name in REQUIRED_ENGINE_RUN_EFFECT_METHODS {
        if effects.hasattr(*method_name)? {
            continue;
        }
        return Err(PyRuntimeError::new_err(format!("NativePythonEngineRunEffects requires `{method_name}` method")));
    }
    Ok(())
}
