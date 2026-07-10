//! Typed PyO3 adapter for the coarse JAX association backend.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]

use numpy::ndarray::{Array2, Array3, Ix2};
use numpy::{
    IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_engine as native_engine;
use g_plan as native_plan;

/// Validated scalar policy required to construct the Python numerical configs.
#[pyclass(name = "JaxBackendConfig", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct JaxBackendConfig {
    association_mode: native_plan::AssociationMode,
    correction: native_plan::CorrectionPlan,
    kernels: native_plan::KernelPlan,
}

/// Trait-major phenotypes and sample-major covariates for one compute group.
#[pyclass(name = "JaxGroupInput")]
pub(crate) struct JaxGroupInput {
    phenotype_matrix: Py<PyArray2<f32>>,
    covariate_matrix: Py<PyArray2<f32>>,
}

enum OwnedGenotypeMatrix {
    Dosage(Py<PyArray2<f32>>),
    Packed8(Py<PyArray3<u8>>),
}

/// One variant-major genotype batch and its native summary statistics.
#[pyclass(name = "JaxGenotypeBatch")]
pub(crate) struct JaxGenotypeBatch {
    genotypes: OwnedGenotypeMatrix,
    dosage_sum: Py<PyArray1<f32>>,
    observation_count: Py<PyArray1<i32>>,
    imputed_dosage_square_sum: Option<Py<PyArray1<f32>>>,
    rare_sparse_mask: Option<Py<PyArray1<bool>>>,
}

/// Trait selection and host statistic precision for device materialization.
#[pyclass(name = "JaxMaterializationRequest")]
pub(crate) struct JaxMaterializationRequest {
    active_trait_indices: Vec<i32>,
}

/// Opaque chromosome state paired with native null-logistic policy input.
#[pyclass(name = "JaxPreparedChromosome")]
pub(crate) struct JaxPreparedChromosome {
    state: Py<PyAny>,
    null_logistic_converged: Option<Vec<bool>>,
}

/// Private adapter implementing the Python-free engine contract.
pub(crate) struct PyJaxBackend {
    backend: Py<PyAny>,
}

/// Error crossing the engine-to-Python backend boundary.
#[derive(Debug)]
pub(crate) enum PyJaxBackendError {
    InvalidInput(String),
    Python(PyErr),
}

impl std::fmt::Display for PyJaxBackendError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid JAX backend input: {message}"),
            Self::Python(error) => write!(formatter, "Python JAX backend failed: {error}"),
        }
    }
}

impl std::error::Error for PyJaxBackendError {}

impl JaxBackendConfig {
    pub(crate) fn new(plan: &native_plan::RunPlan) -> PyResult<Self> {
        native_engine::validate_jax_integer_domain(plan).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            association_mode: plan.association_mode,
            correction: plan.correction,
            kernels: plan.compute.kernels.clone(),
        })
    }
}

#[pymethods]
impl JaxBackendConfig {
    #[getter]
    fn association_mode(&self) -> &'static str {
        self.association_mode.as_str()
    }

    #[getter]
    fn correction_method(&self) -> &'static str {
        self.correction.method.as_str()
    }

    #[getter]
    fn correction_p_threshold(&self) -> f32 {
        self.correction.p_threshold.get()
    }

    #[getter]
    fn firth_se(&self) -> bool {
        self.correction.firth_se
    }

    #[getter]
    fn linear_minimum_variance(&self) -> f32 {
        self.kernels.linear.minimum_variance.get()
    }

    #[getter]
    fn linear_relative_variance_tolerance(&self) -> f32 {
        self.kernels.linear.relative_variance_tolerance.get()
    }

    #[getter]
    fn binary_minimum_probability(&self) -> f32 {
        self.kernels.binary_null.minimum_probability.get()
    }

    #[getter]
    fn binary_minimum_variance(&self) -> f32 {
        self.kernels.binary_null.minimum_variance.get()
    }

    #[getter]
    fn binary_relative_variance_tolerance(&self) -> f32 {
        self.kernels.binary_null.relative_variance_tolerance.get()
    }

    #[getter]
    fn binary_null_maximum_iterations(&self) -> i32 {
        i32::try_from(self.kernels.binary_null.maximum_iterations)
            .expect("JaxBackendConfig validates binary null iterations")
    }

    #[getter]
    fn binary_null_coefficient_tolerance(&self) -> f32 {
        self.kernels.binary_null.coefficient_tolerance.get()
    }

    #[getter]
    fn firth_batch_size(&self) -> i32 {
        i32::try_from(self.kernels.firth.batch_size).expect("JaxBackendConfig validates Firth batch size")
    }

    #[getter]
    fn firth_candidate_capacity(&self) -> i32 {
        i32::try_from(self.kernels.firth.candidate_capacity)
            .expect("JaxBackendConfig validates Firth candidate capacity")
    }

    #[getter]
    fn firth_maximum_iterations(&self) -> i32 {
        i32::try_from(self.kernels.firth.maximum_iterations)
            .expect("JaxBackendConfig validates Firth maximum iterations")
    }

    #[getter]
    fn firth_gradient_tolerance(&self) -> f64 {
        self.kernels.firth.gradient_tolerance.get()
    }

    #[getter]
    fn firth_coefficient_tolerance(&self) -> f64 {
        self.kernels.firth.coefficient_tolerance.get()
    }

    #[getter]
    fn firth_likelihood_tolerance(&self) -> f64 {
        self.kernels.firth.likelihood_tolerance.get()
    }

    #[getter]
    fn firth_maximum_step_size(&self) -> f64 {
        self.kernels.firth.maximum_step_size.get()
    }

    #[getter]
    fn firth_pseudo_maximum_iterations(&self) -> i32 {
        i32::try_from(self.kernels.firth.pseudo_maximum_iterations)
            .expect("JaxBackendConfig validates Firth pseudo maximum iterations")
    }

    #[getter]
    fn firth_pseudo_inner_maximum_iterations(&self) -> i32 {
        i32::try_from(self.kernels.firth.pseudo_inner_maximum_iterations)
            .expect("JaxBackendConfig validates Firth pseudo inner maximum iterations")
    }

    #[getter]
    fn firth_newton_raphson_zero_start_iterations(&self) -> i32 {
        i32::try_from(self.kernels.firth.newton_raphson_zero_start_iterations)
            .expect("JaxBackendConfig validates Firth Newton-Raphson zero-start iterations")
    }

    #[getter]
    fn firth_line_search_maximum_attempts(&self) -> i32 {
        i32::try_from(self.kernels.firth.line_search_maximum_attempts)
            .expect("JaxBackendConfig validates Firth line-search attempts")
    }

    #[getter]
    fn firth_step_halving_maximum_attempts(&self) -> i32 {
        i32::try_from(self.kernels.firth.step_halving_maximum_attempts)
            .expect("JaxBackendConfig validates Firth step-halving attempts")
    }

    #[getter]
    fn firth_initial_response_scale(&self) -> f64 {
        self.kernels.firth.initial_response_scale.get()
    }

    #[getter]
    fn firth_sparse_carrier_dosage_threshold(&self) -> f64 {
        self.kernels.firth.sparse_carrier_dosage_threshold.get()
    }

    #[getter]
    fn firth_step_halving_scale(&self) -> f64 {
        self.kernels.firth.step_halving_scale.get()
    }

    #[getter]
    fn use_block_firth_math(&self) -> bool {
        self.kernels.firth.use_block_math
    }

    #[getter]
    fn null_firth_maximum_iterations(&self) -> i32 {
        i32::try_from(self.kernels.null_firth.maximum_iterations)
            .expect("JaxBackendConfig validates null Firth maximum iterations")
    }

    #[getter]
    fn null_firth_gradient_tolerance(&self) -> f64 {
        self.kernels.null_firth.gradient_tolerance.get()
    }

    #[getter]
    fn null_firth_maximum_step_size(&self) -> f64 {
        self.kernels.null_firth.maximum_step_size.get()
    }

    #[getter]
    fn null_firth_fallback_iteration_multiplier(&self) -> i32 {
        i32::try_from(self.kernels.null_firth.fallback_iteration_multiplier)
            .expect("JaxBackendConfig validates null Firth fallback iteration multiplier")
    }

    #[getter]
    fn null_firth_fallback_step_divisor(&self) -> f64 {
        self.kernels.null_firth.fallback_step_divisor.get()
    }

    #[getter]
    fn null_firth_line_search_maximum_attempts(&self) -> i32 {
        i32::try_from(self.kernels.null_firth.line_search_maximum_attempts)
            .expect("JaxBackendConfig validates null Firth line-search attempts")
    }

    #[getter]
    fn null_firth_step_halving_scale(&self) -> f64 {
        self.kernels.null_firth.step_halving_scale.get()
    }
}

impl JaxGroupInput {
    fn from_native(py: Python<'_>, input: native_engine::GroupPreparationInput<'_>) -> Result<Self, PyJaxBackendError> {
        validate_value_count(
            input.phenotypes.values.len(),
            input.phenotypes.trait_count,
            input.phenotypes.sample_count,
            "phenotype matrix",
        )?;
        if input.covariates.sample_count != input.phenotypes.sample_count {
            return Err(PyJaxBackendError::InvalidInput(format!(
                "Covariate sample count {} does not match phenotype sample count {}.",
                input.covariates.sample_count, input.phenotypes.sample_count
            )));
        }
        validate_value_count(
            input.covariates.values.len(),
            input.covariates.sample_count,
            input.covariates.covariate_count,
            "covariate matrix",
        )?;
        Ok(Self {
            phenotype_matrix: Array2::from_shape_vec(
                (input.phenotypes.trait_count, input.phenotypes.sample_count),
                input.phenotypes.values.to_vec(),
            )
            .expect("validated phenotype matrix shape")
            .into_pyarray(py)
            .unbind(),
            covariate_matrix: Array2::from_shape_vec(
                (input.covariates.sample_count, input.covariates.covariate_count),
                input.covariates.values.to_vec(),
            )
            .expect("validated covariate matrix shape")
            .into_pyarray(py)
            .unbind(),
        })
    }
}

#[pymethods]
impl JaxGroupInput {
    #[getter]
    fn phenotype_matrix(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.phenotype_matrix.clone_ref(py)
    }

    #[getter]
    fn covariate_matrix(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.covariate_matrix.clone_ref(py)
    }
}

impl JaxGenotypeBatch {
    fn from_native(py: Python<'_>, input: native_engine::GenotypeBatchInput<'_>) -> Result<Self, PyJaxBackendError> {
        let (genotypes, variant_count) = match input.genotypes {
            native_engine::GenotypeMatrixView::Dosage(matrix) => {
                validate_value_count(matrix.values.len(), matrix.variant_count, matrix.sample_count, "dosage matrix")?;
                (
                    OwnedGenotypeMatrix::Dosage(
                        Array2::from_shape_vec((matrix.variant_count, matrix.sample_count), matrix.values.to_vec())
                            .expect("validated dosage matrix shape")
                            .into_pyarray(py)
                            .unbind(),
                    ),
                    matrix.variant_count,
                )
            }
            native_engine::GenotypeMatrixView::Packed8(matrix) => {
                let matrix_value_count = checked_product(matrix.variant_count, matrix.sample_count, "packed8 matrix")?;
                let expected_value_count = matrix_value_count.checked_mul(2).ok_or_else(|| {
                    PyJaxBackendError::InvalidInput("Packed8 matrix dimensions overflow usize.".to_string())
                })?;
                if matrix.values.len() != expected_value_count {
                    return Err(PyJaxBackendError::InvalidInput(format!(
                        "Packed8 matrix contains {} values but shape ({}, {}, 2) requires {expected_value_count}.",
                        matrix.values.len(),
                        matrix.variant_count,
                        matrix.sample_count
                    )));
                }
                (
                    OwnedGenotypeMatrix::Packed8(
                        Array3::from_shape_vec((matrix.variant_count, matrix.sample_count, 2), matrix.values.to_vec())
                            .expect("validated packed8 matrix shape")
                            .into_pyarray(py)
                            .unbind(),
                    ),
                    matrix.variant_count,
                )
            }
        };
        validate_vector_length(input.statistics.dosage_sum.len(), variant_count, "dosage_sum")?;
        validate_vector_length(input.statistics.observation_count.len(), variant_count, "observation_count")?;
        if let Some(values) = input.statistics.imputed_dosage_square_sum {
            validate_vector_length(values.len(), variant_count, "imputed_dosage_square_sum")?;
        }
        if let Some(values) = input.statistics.sparse_candidate_mask {
            validate_vector_length(values.len(), variant_count, "sparse_candidate_mask")?;
        }
        Ok(Self {
            genotypes,
            dosage_sum: input.statistics.dosage_sum.to_vec().into_pyarray(py).unbind(),
            observation_count: input.statistics.observation_count.to_vec().into_pyarray(py).unbind(),
            imputed_dosage_square_sum: input
                .statistics
                .imputed_dosage_square_sum
                .map(|values| values.to_vec().into_pyarray(py).unbind()),
            rare_sparse_mask: input
                .statistics
                .sparse_candidate_mask
                .map(|values| values.to_vec().into_pyarray(py).unbind()),
        })
    }
}

#[pymethods]
impl JaxGenotypeBatch {
    #[getter]
    fn dosage_matrix(&self, py: Python<'_>) -> Option<Py<PyArray2<f32>>> {
        let OwnedGenotypeMatrix::Dosage(values) = &self.genotypes else {
            return None;
        };
        Some(values.clone_ref(py))
    }

    #[getter]
    fn packed8_probabilities(&self, py: Python<'_>) -> Option<Py<PyArray3<u8>>> {
        let OwnedGenotypeMatrix::Packed8(values) = &self.genotypes else {
            return None;
        };
        Some(values.clone_ref(py))
    }

    #[getter]
    fn dosage_sum(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.dosage_sum.clone_ref(py)
    }

    #[getter]
    fn observation_count(&self, py: Python<'_>) -> Py<PyArray1<i32>> {
        self.observation_count.clone_ref(py)
    }

    #[getter]
    fn imputed_dosage_square_sum(&self, py: Python<'_>) -> Option<Py<PyArray1<f32>>> {
        self.imputed_dosage_square_sum.as_ref().map(|values| values.clone_ref(py))
    }

    #[getter]
    fn rare_sparse_mask(&self, py: Python<'_>) -> Option<Py<PyArray1<bool>>> {
        self.rare_sparse_mask.as_ref().map(|values| values.clone_ref(py))
    }
}

impl JaxMaterializationRequest {
    fn from_native(input: native_engine::MaterializationInput<'_>) -> Result<Self, PyJaxBackendError> {
        let active_trait_indices = input
            .active_trait_indices
            .iter()
            .copied()
            .map(|index| {
                i32::try_from(index).map_err(|_| {
                    PyJaxBackendError::InvalidInput(format!("Active trait index {index} exceeds JAX int32 capacity."))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { active_trait_indices })
    }
}

#[pymethods]
impl JaxMaterializationRequest {
    #[getter]
    fn active_trait_indices(&self) -> Vec<i32> {
        self.active_trait_indices.clone()
    }
}

#[pymethods]
impl JaxPreparedChromosome {
    #[new]
    #[pyo3(signature = (*, state, null_logistic_converged))]
    fn new(state: Py<PyAny>, null_logistic_converged: Option<&Bound<'_, PyArray1<bool>>>) -> PyResult<Self> {
        let null_logistic_converged =
            null_logistic_converged.map(|values| values.readonly().as_slice().map(<[bool]>::to_vec)).transpose()?;
        Ok(Self { state, null_logistic_converged })
    }
}

impl PyJaxBackend {
    pub(crate) fn new(backend: Py<PyAny>) -> Self {
        Self { backend }
    }
}

impl native_engine::AssociationBackend for PyJaxBackend {
    type ChromosomeState = Py<PyAny>;
    type DeviceResult = Py<PyAny>;
    type Error = PyJaxBackendError;
    type GroupState = Py<PyAny>;

    fn prepare_group(&self, input: native_engine::GroupPreparationInput<'_>) -> Result<Self::GroupState, Self::Error> {
        Python::attach(|py| {
            let bridge_input = JaxGroupInput::from_native(py, input)?;
            let input_object = Py::new(py, bridge_input).map_err(PyJaxBackendError::Python)?;
            self.backend
                .bind(py)
                .call_method1("prepare_group", (input_object,))
                .map(Bound::unbind)
                .map_err(PyJaxBackendError::Python)
        })
    }

    fn prepare_chromosome(
        &self,
        group: &Self::GroupState,
        predictions: native_engine::TraitMajorPredictionMatrixView<'_>,
    ) -> Result<native_engine::PreparedChromosome<Self::ChromosomeState>, Self::Error> {
        Python::attach(|py| {
            validate_value_count(
                predictions.values.len(),
                predictions.trait_count,
                predictions.sample_count,
                "prediction matrix",
            )?;
            let prediction_matrix = Array2::from_shape_vec(
                (predictions.trait_count, predictions.sample_count),
                predictions.values.to_vec(),
            )
            .expect("validated prediction matrix shape")
            .into_pyarray(py);
            let result = self
                .backend
                .bind(py)
                .call_method1("prepare_chromosome", (group.clone_ref(py), prediction_matrix))
                .map_err(PyJaxBackendError::Python)?;
            let prepared = result
                .extract::<PyRef<'_, JaxPreparedChromosome>>()
                .map_err(|error| PyJaxBackendError::InvalidInput(error.to_string()))?;
            Ok(native_engine::PreparedChromosome {
                state: prepared.state.clone_ref(py),
                null_logistic_converged: prepared.null_logistic_converged.clone(),
            })
        })
    }

    fn compute_batch(
        &self,
        chromosome: &Self::ChromosomeState,
        input: native_engine::GenotypeBatchInput<'_>,
    ) -> Result<Self::DeviceResult, Self::Error> {
        Python::attach(|py| {
            let bridge_input = JaxGenotypeBatch::from_native(py, input)?;
            let input_object = Py::new(py, bridge_input).map_err(PyJaxBackendError::Python)?;
            self.backend
                .bind(py)
                .call_method1("compute_batch", (chromosome.clone_ref(py), input_object))
                .map(Bound::unbind)
                .map_err(PyJaxBackendError::Python)
        })
    }

    fn materialize_batch(
        &self,
        result: Self::DeviceResult,
        input: native_engine::MaterializationInput<'_>,
    ) -> Result<native_engine::HostAssociationBatch, Self::Error> {
        let bridge_input = JaxMaterializationRequest::from_native(input)?;
        Python::attach(|py| {
            let input_object = Py::new(py, bridge_input).map_err(PyJaxBackendError::Python)?;
            let materialized = self
                .backend
                .bind(py)
                .call_method1("materialize_batch", (result, input_object))
                .map_err(PyJaxBackendError::Python)?;
            parse_host_association_batch(py, &materialized).map_err(PyJaxBackendError::Python)
        })
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<JaxBackendConfig>()?;
    module.add_class::<JaxGroupInput>()?;
    module.add_class::<JaxGenotypeBatch>()?;
    module.add_class::<JaxMaterializationRequest>()?;
    module.add_class::<JaxPreparedChromosome>()?;
    Ok(())
}

fn validate_value_count(
    observed_count: usize,
    row_count: usize,
    column_count: usize,
    value_label: &str,
) -> Result<(), PyJaxBackendError> {
    let expected_count = checked_product(row_count, column_count, value_label)?;
    if observed_count != expected_count {
        return Err(PyJaxBackendError::InvalidInput(format!(
            "{value_label} contains {observed_count} values but shape ({row_count}, {column_count}) requires {expected_count}."
        )));
    }
    Ok(())
}

fn checked_product(left: usize, right: usize, value_label: &str) -> Result<usize, PyJaxBackendError> {
    left.checked_mul(right)
        .ok_or_else(|| PyJaxBackendError::InvalidInput(format!("{value_label} dimensions overflow usize.")))
}

fn validate_vector_length(
    observed_count: usize,
    expected_count: usize,
    value_label: &str,
) -> Result<(), PyJaxBackendError> {
    if observed_count != expected_count {
        return Err(PyJaxBackendError::InvalidInput(format!(
            "{value_label} contains {observed_count} values but the genotype batch contains {expected_count} variants."
        )));
    }
    Ok(())
}

fn parse_host_association_batch(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
) -> PyResult<native_engine::HostAssociationBatch> {
    let beta_object = payload.getattr("beta")?;
    let standard_error_object = payload.getattr("standard_error")?;
    let chi_squared_object = payload.getattr("chi_squared")?;
    let log10_p_value_object = payload.getattr("log10_p_value")?;
    let statistics = parse_host_statistics(
        py,
        beta_object.cast::<PyUntypedArray>()?,
        standard_error_object.cast::<PyUntypedArray>()?,
        chi_squared_object.cast::<PyUntypedArray>()?,
        log10_p_value_object.cast::<PyUntypedArray>()?,
    )?;
    let (trait_count, variant_count) = (statistics.trait_count, statistics.variant_count);
    let correction_code_object = payload.getattr("correction_code")?;
    let correction_codes = if correction_code_object.is_none() {
        None
    } else {
        Some(parse_correction_codes(py, correction_code_object.cast::<PyUntypedArray>()?, trait_count, variant_count)?)
    };
    Ok(native_engine::HostAssociationBatch { statistics, correction_codes })
}

fn parse_host_statistics(
    py: Python<'_>,
    beta: &Bound<'_, PyUntypedArray>,
    standard_error: &Bound<'_, PyUntypedArray>,
    chi_squared: &Bound<'_, PyUntypedArray>,
    log10_p_value: &Bound<'_, PyUntypedArray>,
) -> PyResult<native_engine::HostAssociationStatistics> {
    let observed_dtype = beta.dtype();
    for (label, values) in
        [("standard_error", standard_error), ("chi_squared", chi_squared), ("log10_p_value", log10_p_value)]
    {
        if !values.dtype().is_equiv_to(&observed_dtype) {
            return Err(PyValueError::new_err(format!("{label} dtype must match beta dtype.")));
        }
    }
    if !observed_dtype.is_equiv_to(&dtype::<f32>(py)) {
        return Err(PyValueError::new_err("Host association statistics must use float32 dtype."));
    }
    parse_statistic_matrix::<f32>(beta, standard_error, chi_squared, log10_p_value)
}

fn parse_statistic_matrix<Statistic: numpy::Element + Copy>(
    beta: &Bound<'_, PyUntypedArray>,
    standard_error: &Bound<'_, PyUntypedArray>,
    chi_squared: &Bound<'_, PyUntypedArray>,
    log10_p_value: &Bound<'_, PyUntypedArray>,
) -> PyResult<native_engine::HostAssociationStatisticMatrix<Statistic>> {
    let beta = beta.cast::<PyArray<Statistic, Ix2>>()?.readonly();
    let standard_error = standard_error.cast::<PyArray<Statistic, Ix2>>()?.readonly();
    let chi_squared = chi_squared.cast::<PyArray<Statistic, Ix2>>()?.readonly();
    let log10_p_value = log10_p_value.cast::<PyArray<Statistic, Ix2>>()?.readonly();
    let expected_shape = beta.shape();
    for (label, observed_shape) in [
        ("standard_error", standard_error.shape()),
        ("chi_squared", chi_squared.shape()),
        ("log10_p_value", log10_p_value.shape()),
    ] {
        if observed_shape != expected_shape {
            return Err(PyValueError::new_err(format!(
                "{label} shape {observed_shape:?} does not match beta shape {expected_shape:?}."
            )));
        }
    }
    Ok(native_engine::HostAssociationStatisticMatrix {
        trait_count: expected_shape[0],
        variant_count: expected_shape[1],
        beta: beta.as_slice()?.to_vec(),
        standard_error: standard_error.as_slice()?.to_vec(),
        chi_squared: chi_squared.as_slice()?.to_vec(),
        log10_p_value: log10_p_value.as_slice()?.to_vec(),
    })
}

fn parse_correction_codes(
    py: Python<'_>,
    values: &Bound<'_, PyUntypedArray>,
    trait_count: usize,
    variant_count: usize,
) -> PyResult<native_engine::HostCorrectionCodeMatrix> {
    if !values.dtype().is_equiv_to(&dtype::<i32>(py)) {
        return Err(PyValueError::new_err("correction_code must use int32 dtype."));
    }
    let values = values.cast::<PyArray<i32, Ix2>>()?.readonly();
    if values.shape() != [trait_count, variant_count] {
        return Err(PyValueError::new_err(format!(
            "correction_code shape {:?} does not match statistic shape ({trait_count}, {variant_count}).",
            values.shape()
        )));
    }
    Ok(native_engine::HostCorrectionCodeMatrix { trait_count, variant_count, values: values.as_slice()?.to_vec() })
}
