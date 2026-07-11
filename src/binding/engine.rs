//! Private PyO3 adapter for the coarse JAX association backend.

#![allow(clippy::needless_pass_by_value)]

use numpy::ndarray::{Array2, Array3, Ix2};
use numpy::{
    IntoPyArray, PyArray, PyArray1, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use g_engine as native_engine;

/// Private adapter implementing the Python-free engine contract.
pub(crate) struct PyJaxBackend {
    backend: Py<PyAny>,
    kind: BackendKind,
}

#[derive(Clone, Copy)]
enum BackendKind {
    Linear,
    BinaryScore,
    BinaryFirth,
}

struct PythonGenotypeBatch {
    method_name: &'static str,
    values: Py<PyAny>,
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

pub(crate) fn create_jax_backend(
    py: Python<'_>,
    plan: g_runner::JaxAssociationBackendPlan<'_>,
) -> PyResult<PyJaxBackend> {
    let backend_module = PyModule::import(py, "g.jax_backend")?;
    match plan {
        g_runner::JaxAssociationBackendPlan::Linear(kernel) => {
            let keyword_arguments = PyDict::new(py);
            keyword_arguments.set_item("minimum_variance", kernel.minimum_variance.get())?;
            keyword_arguments.set_item("relative_variance_tolerance", kernel.relative_variance_tolerance.get())?;
            let backend = backend_module.getattr("LinearJaxBackend")?.call((), Some(&keyword_arguments))?.unbind();
            Ok(PyJaxBackend { backend, kind: BackendKind::Linear })
        }
        g_runner::JaxAssociationBackendPlan::BinaryScore(kernels) => {
            let keyword_arguments = binary_score_backend_keyword_arguments(py, kernels)?;
            let backend = backend_module.getattr("BinaryScoreJaxBackend")?.call((), Some(&keyword_arguments))?.unbind();
            Ok(PyJaxBackend { backend, kind: BackendKind::BinaryScore })
        }
        g_runner::JaxAssociationBackendPlan::BinaryFirth { correction, kernels } => {
            let keyword_arguments = binary_firth_backend_keyword_arguments(py, kernels, *correction)?;
            let backend = backend_module.getattr("BinaryFirthJaxBackend")?.call((), Some(&keyword_arguments))?.unbind();
            Ok(PyJaxBackend { backend, kind: BackendKind::BinaryFirth })
        }
    }
}

fn binary_score_backend_keyword_arguments<'py>(
    py: Python<'py>,
    kernels: &g_plan::KernelPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let keyword_arguments = PyDict::new(py);
    keyword_arguments.set_item("minimum_probability", kernels.binary_null.minimum_probability.get())?;
    keyword_arguments.set_item("minimum_variance", kernels.binary_null.minimum_variance.get())?;
    keyword_arguments.set_item("relative_variance_tolerance", kernels.binary_null.relative_variance_tolerance.get())?;
    keyword_arguments.set_item("null_logistic_maximum_iterations", kernels.binary_null.maximum_iterations)?;
    keyword_arguments
        .set_item("null_logistic_coefficient_tolerance", kernels.binary_null.coefficient_tolerance.get())?;
    Ok(keyword_arguments)
}

fn binary_firth_backend_keyword_arguments<'py>(
    py: Python<'py>,
    kernels: &g_plan::KernelPlan,
    correction: g_plan::CorrectionPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let keyword_arguments = binary_score_backend_keyword_arguments(py, kernels)?;
    keyword_arguments.set_item("p_threshold", correction.p_threshold.get())?;
    keyword_arguments.set_item("firth_se", correction.firth_se)?;
    keyword_arguments.set_item("firth_batch_size", kernels.firth.batch_size)?;
    keyword_arguments.set_item("firth_candidate_capacity", kernels.firth.candidate_capacity)?;
    keyword_arguments.set_item("firth_maximum_iterations", kernels.firth.maximum_iterations)?;
    keyword_arguments.set_item("firth_gradient_tolerance", kernels.firth.gradient_tolerance.get())?;
    keyword_arguments.set_item("firth_coefficient_tolerance", kernels.firth.coefficient_tolerance.get())?;
    keyword_arguments.set_item("firth_likelihood_tolerance", kernels.firth.likelihood_tolerance.get())?;
    keyword_arguments.set_item("firth_maximum_step_size", kernels.firth.maximum_step_size.get())?;
    keyword_arguments.set_item("firth_pseudo_maximum_iterations", kernels.firth.pseudo_maximum_iterations)?;
    keyword_arguments
        .set_item("firth_pseudo_inner_maximum_iterations", kernels.firth.pseudo_inner_maximum_iterations)?;
    keyword_arguments
        .set_item("firth_newton_raphson_zero_start_iterations", kernels.firth.newton_raphson_zero_start_iterations)?;
    keyword_arguments.set_item("firth_line_search_maximum_attempts", kernels.firth.line_search_maximum_attempts)?;
    keyword_arguments.set_item("firth_step_halving_maximum_attempts", kernels.firth.step_halving_maximum_attempts)?;
    keyword_arguments.set_item("firth_initial_response_scale", kernels.firth.initial_response_scale.get())?;
    keyword_arguments
        .set_item("firth_sparse_carrier_dosage_threshold", kernels.firth.sparse_carrier_dosage_threshold.get())?;
    keyword_arguments.set_item("firth_step_halving_scale", kernels.firth.step_halving_scale.get())?;
    keyword_arguments.set_item("firth_use_block_math", kernels.firth.use_block_math)?;
    keyword_arguments.set_item("null_firth_maximum_iterations", kernels.null_firth.maximum_iterations)?;
    keyword_arguments.set_item("null_firth_gradient_tolerance", kernels.null_firth.gradient_tolerance.get())?;
    keyword_arguments.set_item("null_firth_maximum_step_size", kernels.null_firth.maximum_step_size.get())?;
    keyword_arguments
        .set_item("null_firth_fallback_iteration_multiplier", kernels.null_firth.fallback_iteration_multiplier)?;
    keyword_arguments.set_item("null_firth_fallback_step_divisor", kernels.null_firth.fallback_step_divisor.get())?;
    keyword_arguments
        .set_item("null_firth_line_search_maximum_attempts", kernels.null_firth.line_search_maximum_attempts)?;
    keyword_arguments.set_item("null_firth_step_halving_scale", kernels.null_firth.step_halving_scale.get())?;
    Ok(keyword_arguments)
}

impl native_engine::AssociationBackend for PyJaxBackend {
    type ChromosomeState = Py<PyAny>;
    type DeviceResult = Py<PyAny>;
    type Error = PyJaxBackendError;
    type GroupState = Py<PyAny>;

    fn prepare_group(&self, input: native_engine::GroupPreparationInput) -> Result<Self::GroupState, Self::Error> {
        Python::attach(|py| {
            let phenotype_matrix = Array2::from_shape_vec(
                (input.phenotypes.trait_count, input.phenotypes.sample_count),
                input.phenotypes.values,
            )
            .expect("engine-validated phenotype matrix shape")
            .into_pyarray(py);
            let covariate_matrix = Array2::from_shape_vec(
                (input.covariates.sample_count, input.covariates.covariate_count),
                input.covariates.values,
            )
            .expect("engine-validated covariate matrix shape")
            .into_pyarray(py);
            self.backend
                .bind(py)
                .call_method1("prepare_group", (phenotype_matrix, covariate_matrix))
                .map(Bound::unbind)
                .map_err(PyJaxBackendError::Python)
        })
    }

    fn release_group(&self, group: Self::GroupState) {
        Python::attach(|_| drop(group));
    }

    fn prepare_chromosome(
        &self,
        group: &Self::GroupState,
        predictions: native_engine::TraitMajorMatrix,
    ) -> Result<native_engine::PreparedChromosome<Self::ChromosomeState>, Self::Error> {
        Python::attach(|py| {
            let prediction_matrix =
                Array2::from_shape_vec((predictions.trait_count, predictions.sample_count), predictions.values)
                    .expect("engine-validated prediction matrix shape")
                    .into_pyarray(py);
            let state = self
                .backend
                .bind(py)
                .call_method1("prepare_chromosome", (group.bind(py), prediction_matrix))
                .map_err(PyJaxBackendError::Python)?;
            let null_logistic_convergence = match self.kind {
                BackendKind::Linear => None,
                BackendKind::BinaryScore => Some(state.getattr("null_logistic_converged")),
                BackendKind::BinaryFirth => Some(
                    state.getattr("score_state").and_then(|score_state| score_state.getattr("null_logistic_converged")),
                ),
            }
            .transpose()
            .map_err(PyJaxBackendError::Python)?;
            let null_logistic_converged = if let Some(convergence_values) = null_logistic_convergence {
                let host_values = convergence_values.call_method0("__array__").map_err(PyJaxBackendError::Python)?;
                let readonly_values = host_values
                    .cast::<PyArray1<bool>>()
                    .map_err(|error| PyJaxBackendError::InvalidInput(error.to_string()))?
                    .readonly();
                Some(
                    readonly_values
                        .as_slice()
                        .map_err(|error| PyJaxBackendError::InvalidInput(error.to_string()))?
                        .to_vec(),
                )
            } else {
                None
            };
            Ok(native_engine::PreparedChromosome { state: state.unbind(), null_logistic_converged })
        })
    }

    fn release_chromosome(&self, chromosome: Self::ChromosomeState) {
        Python::attach(|_| drop(chromosome));
    }

    fn compute_batch(
        &self,
        chromosome: &Self::ChromosomeState,
        input: native_engine::GenotypeBatchInput,
    ) -> Result<Self::DeviceResult, Self::Error> {
        Python::attach(|py| match self.kind {
            BackendKind::Linear => compute_linear_batch(py, self.backend.bind(py), chromosome, input),
            BackendKind::BinaryScore => compute_binary_score_batch(py, self.backend.bind(py), chromosome, input),
            BackendKind::BinaryFirth => compute_binary_firth_batch(py, self.backend.bind(py), chromosome, input),
        })
    }

    fn materialize_batch(
        &self,
        result: Self::DeviceResult,
        active_trait_indices: Option<&[usize]>,
    ) -> Result<native_engine::HostAssociationBatch, Self::Error> {
        Python::attach(|py| {
            let active_trait_indices = active_trait_indices.map(|indices| {
                indices
                    .iter()
                    .copied()
                    .map(|index| i32::try_from(index).expect("engine preflight validated JAX int32 trait indices"))
                    .collect::<Vec<_>>()
                    .into_pyarray(py)
            });
            let materialized = self
                .backend
                .bind(py)
                .call_method1("materialize_batch", (result, active_trait_indices))
                .map_err(PyJaxBackendError::Python)?;
            parse_host_association_batch(py, &materialized).map_err(PyJaxBackendError::Python)
        })
    }
}

fn compute_linear_batch(
    py: Python<'_>,
    backend: &Bound<'_, PyAny>,
    chromosome: &Py<PyAny>,
    input: native_engine::GenotypeBatchInput,
) -> Result<Py<PyAny>, PyJaxBackendError> {
    let native_engine::GenotypeBatchInput { variant_count, sample_count, genotypes, statistics, .. } = input;
    let native_engine::GenotypeBatchStatistics { genotype_mean, imputed_dosage_square_sum, sparse_candidate_mask: _ } =
        statistics;
    let imputed_dosage_square_sum = imputed_dosage_square_sum
        .ok_or_else(|| {
            PyJaxBackendError::InvalidInput("linear association requires imputed dosage square sums".to_string())
        })?
        .into_pyarray(py);
    let genotype_mean = genotype_mean.into_pyarray(py);
    let genotype_batch = into_python_genotype_batch(py, genotypes, variant_count, sample_count);
    let result = backend.call_method1(
        genotype_batch.method_name,
        (chromosome.bind(py), genotype_batch.values, genotype_mean, imputed_dosage_square_sum),
    );
    result.map(Bound::unbind).map_err(PyJaxBackendError::Python)
}

fn compute_binary_score_batch(
    py: Python<'_>,
    backend: &Bound<'_, PyAny>,
    chromosome: &Py<PyAny>,
    input: native_engine::GenotypeBatchInput,
) -> Result<Py<PyAny>, PyJaxBackendError> {
    let native_engine::GenotypeBatchInput { variant_count, sample_count, genotypes, statistics, .. } = input;
    let native_engine::GenotypeBatchStatistics {
        genotype_mean,
        imputed_dosage_square_sum: _,
        sparse_candidate_mask: _,
    } = statistics;
    let genotype_mean = genotype_mean.into_pyarray(py);
    let genotype_batch = into_python_genotype_batch(py, genotypes, variant_count, sample_count);
    let result =
        backend.call_method1(genotype_batch.method_name, (chromosome.bind(py), genotype_batch.values, genotype_mean));
    result.map(Bound::unbind).map_err(PyJaxBackendError::Python)
}

fn compute_binary_firth_batch(
    py: Python<'_>,
    backend: &Bound<'_, PyAny>,
    chromosome: &Py<PyAny>,
    input: native_engine::GenotypeBatchInput,
) -> Result<Py<PyAny>, PyJaxBackendError> {
    let native_engine::GenotypeBatchInput { variant_count, sample_count, genotypes, statistics, .. } = input;
    let native_engine::GenotypeBatchStatistics { genotype_mean, imputed_dosage_square_sum: _, sparse_candidate_mask } =
        statistics;
    let sparse_candidate_mask = sparse_candidate_mask
        .ok_or_else(|| {
            PyJaxBackendError::InvalidInput("binary Firth association requires a sparse candidate mask".to_string())
        })?
        .into_pyarray(py);
    let genotype_mean = genotype_mean.into_pyarray(py);
    let genotype_batch = into_python_genotype_batch(py, genotypes, variant_count, sample_count);
    let result = backend.call_method1(
        genotype_batch.method_name,
        (chromosome.bind(py), genotype_batch.values, genotype_mean, sparse_candidate_mask),
    );
    result.map(Bound::unbind).map_err(PyJaxBackendError::Python)
}

fn into_python_genotype_batch(
    py: Python<'_>,
    genotypes: native_engine::OwnedGenotypeBuffer,
    variant_count: usize,
    sample_count: usize,
) -> PythonGenotypeBatch {
    match genotypes {
        native_engine::OwnedGenotypeBuffer::Dosage(values) => PythonGenotypeBatch {
            method_name: "compute_dosage_batch",
            values: Array2::from_shape_vec((variant_count, sample_count), values)
                .expect("engine-validated dosage matrix shape")
                .into_pyarray(py)
                .into_any()
                .unbind(),
        },
        native_engine::OwnedGenotypeBuffer::Packed8(values) => PythonGenotypeBatch {
            method_name: "compute_packed8_batch",
            values: Array3::from_shape_vec((variant_count, sample_count, 2), values)
                .expect("engine-validated packed8 matrix shape")
                .into_pyarray(py)
                .into_any()
                .unbind(),
        },
    }
}

fn parse_host_association_batch(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
) -> PyResult<native_engine::HostAssociationBatch> {
    let beta_object = payload.getattr("beta")?;
    let standard_error_object = payload.getattr("standard_error")?;
    let chi_squared_object = payload.getattr("chi_squared")?;
    let log10_p_value_object = payload.getattr("log10_p_value")?;
    let beta = beta_object.cast::<PyUntypedArray>()?;
    let standard_error = standard_error_object.cast::<PyUntypedArray>()?;
    let chi_squared = chi_squared_object.cast::<PyUntypedArray>()?;
    let log10_p_value = log10_p_value_object.cast::<PyUntypedArray>()?;
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
    let beta = beta.cast::<PyArray<f32, Ix2>>()?.readonly();
    let standard_error = standard_error.cast::<PyArray<f32, Ix2>>()?.readonly();
    let chi_squared = chi_squared.cast::<PyArray<f32, Ix2>>()?.readonly();
    let log10_p_value = log10_p_value.cast::<PyArray<f32, Ix2>>()?.readonly();
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
    let (trait_count, variant_count) = (expected_shape[0], expected_shape[1]);
    let correction_code_object = payload.getattr("correction_code")?;
    let correction_codes = if correction_code_object.is_none() {
        None
    } else {
        Some(parse_correction_codes(py, correction_code_object.cast::<PyUntypedArray>()?, trait_count, variant_count)?)
    };
    Ok(native_engine::HostAssociationBatch {
        trait_count,
        variant_count,
        beta: beta.as_slice()?.to_vec(),
        standard_error: standard_error.as_slice()?.to_vec(),
        chi_squared: chi_squared.as_slice()?.to_vec(),
        log10_p_value: log10_p_value.as_slice()?.to_vec(),
        correction_codes,
    })
}

fn parse_correction_codes(
    py: Python<'_>,
    values: &Bound<'_, PyUntypedArray>,
    trait_count: usize,
    variant_count: usize,
) -> PyResult<Vec<u8>> {
    if !values.dtype().is_equiv_to(&dtype::<u8>(py)) {
        return Err(PyValueError::new_err("correction_code must use uint8 dtype."));
    }
    let values = values.cast::<PyArray<u8, Ix2>>()?.readonly();
    if values.shape() != [trait_count, variant_count] {
        return Err(PyValueError::new_err(format!(
            "correction_code shape {:?} does not match statistic shape ({trait_count}, {variant_count}).",
            values.shape()
        )));
    }
    Ok(values.as_slice()?.to_vec())
}
