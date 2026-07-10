//! Typed PyO3 adapter for the coarse JAX association backend.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]

use numpy::ndarray::{Array2, Array3, Ix2};
use numpy::{
    IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray1,
    PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_engine as native_engine;
use g_interface::{RegenieConfigData, RegenieTraitTypeValue};
use g_plan as native_plan;

/// Validated scalar policy required to construct the Python numerical configs.
#[pyclass(name = "JaxBackendConfig", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct JaxBackendConfig {
    data: RegenieConfigData,
    association_mode: native_plan::AssociationMode,
    correction: native_plan::CorrectionPlan,
}

/// Trait-major phenotypes and sample-major covariates for one compute group.
#[pyclass(name = "JaxGroupInput")]
pub(crate) struct JaxGroupInput {
    phenotype_values: Vec<f32>,
    trait_count: usize,
    sample_count: usize,
    covariate_values: Vec<f32>,
    covariate_count: usize,
}

/// Trait-major LOCO predictions for one chromosome.
#[pyclass(name = "JaxChromosomeInput")]
pub(crate) struct JaxChromosomeInput {
    prediction_values: Vec<f32>,
    trait_count: usize,
    sample_count: usize,
}

enum OwnedGenotypeMatrix {
    Dosage(Py<PyArray2<f32>>),
    Packed8(Py<PyArray3<u8>>),
}

/// One variant-major genotype batch and its native summary statistics.
#[pyclass(name = "JaxGenotypeBatch")]
pub(crate) struct JaxGenotypeBatch {
    variant_start_index: usize,
    genotypes: OwnedGenotypeMatrix,
    dosage_sum: Py<PyArray1<f32>>,
    observation_count: Py<PyArray1<i32>>,
    imputed_dosage_square_sum: Option<Py<PyArray1<f32>>>,
    rare_sparse_mask: Option<Py<PyArray1<bool>>>,
}

/// Trait selection and host statistic precision for device materialization.
#[pyclass(name = "JaxMaterializationRequest")]
pub(crate) struct JaxMaterializationRequest {
    active_trait_indices: Vec<usize>,
    output_statistic_dtype: native_plan::FloatingPointDtype,
}

/// Host diagnostics produced while preparing a binary null model.
#[pyclass(name = "JaxNullModelDiagnostics", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct JaxNullModelDiagnostics {
    data: native_engine::NullModelDiagnostics,
}

/// Opaque chromosome state paired with native policy diagnostics.
#[pyclass(name = "JaxPreparedChromosome")]
pub(crate) struct JaxPreparedChromosome {
    state: Py<PyAny>,
    diagnostics: Option<native_engine::NullModelDiagnostics>,
}

/// Aggregate binary diagnostics for one materialized batch.
#[pyclass(name = "JaxBinaryDiagnostics", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct JaxBinaryDiagnostics {
    data: native_engine::BinaryBatchDiagnostics,
}

/// Trait-major host association arrays returned by Python materialization.
#[pyclass(name = "JaxHostAssociationBatch", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct JaxHostAssociationBatch {
    data: native_engine::HostAssociationBatch,
}

/// Private adapter implementing the Python-free engine contract.
pub(crate) struct PyJaxBackend {
    backend: Py<PyAny>,
}

impl JaxBackendConfig {
    pub(crate) fn new(data: RegenieConfigData) -> Result<Self, native_plan::HostPolicyError> {
        let association_mode = match data.trait_config.trait_type {
            RegenieTraitTypeValue::Quantitative => native_plan::AssociationMode::Regenie2Linear,
            RegenieTraitTypeValue::Binary => native_plan::AssociationMode::Regenie2Binary,
        };
        let correction = if association_mode == native_plan::AssociationMode::Regenie2Linear {
            native_plan::CorrectionPlan {
                method: native_plan::BinaryFallbackMethod::ScoreOnly,
                p_threshold: 0.05,
                firth_se: false,
            }
        } else {
            native_plan::normalize_binary_correction(
                data.binary.firth,
                data.binary.approx,
                data.binary.spa,
                f64::from(data.binary.p_threshold),
                data.binary.firth_se,
            )?
        };
        Ok(Self { data, association_mode, correction })
    }

    pub(crate) fn binary_kernel_config_json(&self) -> PyResult<Option<String>> {
        if self.association_mode != native_plan::AssociationMode::Regenie2Binary {
            return Ok(None);
        }
        let compute = &self.data.g_compute;
        serde_json::to_string(&serde_json::json!({
            "approximate_firth": {
                "coefficient_tolerance": compute.firth_coefficient_tolerance,
                "gradient_tolerance": compute.firth_gradient_tolerance,
                "initial_response_scale": compute.firth_initial_response_scale,
                "likelihood_tolerance": compute.firth_likelihood_tolerance,
                "line_search_maximum_attempts": compute.firth_line_search_maximum_attempts.get(),
                "maximum_iterations": compute.firth_maximum_iterations.get(),
                "maximum_step_size": compute.firth_maximum_step_size,
                "newton_raphson_zero_start_iterations": compute.firth_newton_raphson_zero_start_iterations.get(),
                "pseudo_inner_maximum_iterations": compute.firth_pseudo_inner_maximum_iterations.get(),
                "pseudo_maximum_iterations": compute.firth_pseudo_maximum_iterations.get(),
                "sparse_carrier_dosage_threshold": compute.firth_sparse_carrier_dosage_threshold,
                "step_halving_maximum_attempts": compute.firth_step_halving_maximum_attempts.get(),
                "step_halving_scale": compute.firth_step_halving_scale,
                "use_block_math": compute.use_block_firth_math,
            },
            "firth_candidate": {
                "batch_size": compute.firth_batch_size.get(),
                "candidate_capacity": compute.firth_candidate_capacity.get(),
            },
            "null_firth": {
                "fallback_iteration_multiplier": compute.null_firth_fallback_iteration_multiplier.get(),
                "fallback_step_divisor": compute.null_firth_fallback_step_divisor,
                "gradient_tolerance": compute.null_firth_gradient_tolerance,
                "line_search_maximum_attempts": compute.null_firth_line_search_maximum_attempts.get(),
                "maximum_iterations": compute.null_firth_maximum_iterations.get(),
                "maximum_step_size": compute.null_firth_maximum_step_size,
                "step_halving_scale": compute.null_firth_step_halving_scale,
            },
            "null_logistic": {
                "coefficient_tolerance": compute.binary_null_coefficient_tolerance,
                "maximum_iterations": compute.binary_null_maximum_iterations.get(),
            },
            "numerical": {
                "minimum_probability": compute.binary_minimum_probability,
                "minimum_variance": compute.binary_minimum_variance,
                "relative_variance_tolerance": compute.binary_relative_variance_tolerance,
            },
        }))
        .map(Some)
        .map_err(|error| PyValueError::new_err(format!("Could not serialize binary kernel config: {error}")))
    }
}

#[pymethods]
impl JaxBackendConfig {
    #[getter]
    fn association_mode(&self) -> &'static str {
        self.association_mode.as_str()
    }

    #[getter]
    fn score_dtype(&self) -> &'static str {
        self.data.g_compute.score_dtype.as_str()
    }

    #[getter]
    fn firth_dtype(&self) -> &'static str {
        self.data.g_compute.firth_dtype.as_str()
    }

    #[getter]
    fn correction_method(&self) -> &'static str {
        self.correction.method.as_str()
    }

    #[getter]
    fn correction_p_threshold(&self) -> f64 {
        self.correction.p_threshold
    }

    #[getter]
    fn firth_se(&self) -> bool {
        self.correction.firth_se
    }

    #[getter]
    fn linear_minimum_variance(&self) -> f32 {
        self.data.g_compute.linear_minimum_variance
    }

    #[getter]
    fn linear_relative_variance_tolerance(&self) -> f32 {
        self.data.g_compute.linear_relative_variance_tolerance
    }

    #[getter]
    fn binary_minimum_probability(&self) -> f32 {
        self.data.g_compute.binary_minimum_probability
    }

    #[getter]
    fn binary_minimum_variance(&self) -> f32 {
        self.data.g_compute.binary_minimum_variance
    }

    #[getter]
    fn binary_relative_variance_tolerance(&self) -> f32 {
        self.data.g_compute.binary_relative_variance_tolerance
    }

    #[getter]
    fn binary_null_maximum_iterations(&self) -> u32 {
        self.data.g_compute.binary_null_maximum_iterations.get()
    }

    #[getter]
    fn binary_null_coefficient_tolerance(&self) -> f32 {
        self.data.g_compute.binary_null_coefficient_tolerance
    }

    #[getter]
    fn firth_batch_size(&self) -> u32 {
        self.data.g_compute.firth_batch_size.get()
    }

    #[getter]
    fn firth_candidate_capacity(&self) -> u32 {
        self.data.g_compute.firth_candidate_capacity.get()
    }

    #[getter]
    fn firth_maximum_iterations(&self) -> u32 {
        self.data.g_compute.firth_maximum_iterations.get()
    }

    #[getter]
    fn firth_gradient_tolerance(&self) -> f32 {
        self.data.g_compute.firth_gradient_tolerance
    }

    #[getter]
    fn firth_coefficient_tolerance(&self) -> f32 {
        self.data.g_compute.firth_coefficient_tolerance
    }

    #[getter]
    fn firth_likelihood_tolerance(&self) -> f32 {
        self.data.g_compute.firth_likelihood_tolerance
    }

    #[getter]
    fn firth_maximum_step_size(&self) -> f32 {
        self.data.g_compute.firth_maximum_step_size
    }

    #[getter]
    fn firth_pseudo_maximum_iterations(&self) -> u32 {
        self.data.g_compute.firth_pseudo_maximum_iterations.get()
    }

    #[getter]
    fn firth_pseudo_inner_maximum_iterations(&self) -> u32 {
        self.data.g_compute.firth_pseudo_inner_maximum_iterations.get()
    }

    #[getter]
    fn firth_newton_raphson_zero_start_iterations(&self) -> u32 {
        self.data.g_compute.firth_newton_raphson_zero_start_iterations.get()
    }

    #[getter]
    fn firth_line_search_maximum_attempts(&self) -> u32 {
        self.data.g_compute.firth_line_search_maximum_attempts.get()
    }

    #[getter]
    fn firth_step_halving_maximum_attempts(&self) -> u32 {
        self.data.g_compute.firth_step_halving_maximum_attempts.get()
    }

    #[getter]
    fn firth_initial_response_scale(&self) -> f32 {
        self.data.g_compute.firth_initial_response_scale
    }

    #[getter]
    fn firth_sparse_carrier_dosage_threshold(&self) -> f32 {
        self.data.g_compute.firth_sparse_carrier_dosage_threshold
    }

    #[getter]
    fn firth_step_halving_scale(&self) -> f32 {
        self.data.g_compute.firth_step_halving_scale
    }

    #[getter]
    fn use_block_firth_math(&self) -> bool {
        self.data.g_compute.use_block_firth_math
    }

    #[getter]
    fn null_firth_maximum_iterations(&self) -> u32 {
        self.data.g_compute.null_firth_maximum_iterations.get()
    }

    #[getter]
    fn null_firth_gradient_tolerance(&self) -> f32 {
        self.data.g_compute.null_firth_gradient_tolerance
    }

    #[getter]
    fn null_firth_maximum_step_size(&self) -> f32 {
        self.data.g_compute.null_firth_maximum_step_size
    }

    #[getter]
    fn null_firth_fallback_iteration_multiplier(&self) -> u32 {
        self.data.g_compute.null_firth_fallback_iteration_multiplier.get()
    }

    #[getter]
    fn null_firth_fallback_step_divisor(&self) -> f32 {
        self.data.g_compute.null_firth_fallback_step_divisor
    }

    #[getter]
    fn null_firth_line_search_maximum_attempts(&self) -> u32 {
        self.data.g_compute.null_firth_line_search_maximum_attempts.get()
    }

    #[getter]
    fn null_firth_step_halving_scale(&self) -> f32 {
        self.data.g_compute.null_firth_step_halving_scale
    }
}

impl JaxGroupInput {
    fn from_native(input: native_engine::GroupPreparationInput<'_>) -> Result<Self, native_engine::BackendError> {
        validate_value_count(
            input.phenotypes.values.len(),
            input.phenotypes.trait_count,
            input.phenotypes.sample_count,
            "phenotype matrix",
        )?;
        if input.covariates.sample_count != input.phenotypes.sample_count {
            return Err(native_engine::BackendError::new(format!(
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
            phenotype_values: input.phenotypes.values.to_vec(),
            trait_count: input.phenotypes.trait_count,
            sample_count: input.phenotypes.sample_count,
            covariate_values: input.covariates.values.to_vec(),
            covariate_count: input.covariates.covariate_count,
        })
    }
}

#[pymethods]
impl JaxGroupInput {
    #[getter]
    fn phenotype_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        Array2::from_shape_vec((self.trait_count, self.sample_count), self.phenotype_values.clone())
            .expect("validated phenotype matrix shape")
            .into_pyarray(py)
    }

    #[getter]
    fn covariate_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        Array2::from_shape_vec((self.sample_count, self.covariate_count), self.covariate_values.clone())
            .expect("validated covariate matrix shape")
            .into_pyarray(py)
    }
}

impl JaxChromosomeInput {
    fn from_native(input: native_engine::ChromosomePreparationInput<'_>) -> Result<Self, native_engine::BackendError> {
        validate_value_count(
            input.predictions.values.len(),
            input.predictions.trait_count,
            input.predictions.sample_count,
            "prediction matrix",
        )?;
        Ok(Self {
            prediction_values: input.predictions.values.to_vec(),
            trait_count: input.predictions.trait_count,
            sample_count: input.predictions.sample_count,
        })
    }
}

#[pymethods]
impl JaxChromosomeInput {
    #[getter]
    fn prediction_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        Array2::from_shape_vec((self.trait_count, self.sample_count), self.prediction_values.clone())
            .expect("validated prediction matrix shape")
            .into_pyarray(py)
    }
}

impl JaxGenotypeBatch {
    fn from_native(
        py: Python<'_>,
        input: native_engine::GenotypeBatchInput<'_>,
    ) -> Result<Self, native_engine::BackendError> {
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
                let expected_value_count = matrix_value_count
                    .checked_mul(2)
                    .ok_or_else(|| native_engine::BackendError::new("Packed8 matrix dimensions overflow usize."))?;
                if matrix.values.len() != expected_value_count {
                    return Err(native_engine::BackendError::new(format!(
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
            variant_start_index: input.variant_start_index,
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
    fn variant_start_index(&self) -> usize {
        self.variant_start_index
    }

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
    fn from_native(input: native_engine::MaterializationInput<'_>) -> Self {
        Self {
            active_trait_indices: input.active_trait_indices.to_vec(),
            output_statistic_dtype: input.output_statistic_dtype,
        }
    }
}

#[pymethods]
impl JaxMaterializationRequest {
    #[getter]
    fn active_trait_indices(&self) -> Vec<usize> {
        self.active_trait_indices.clone()
    }

    #[getter]
    fn output_statistic_dtype(&self) -> &'static str {
        self.output_statistic_dtype.as_str()
    }
}

#[pymethods]
impl JaxNullModelDiagnostics {
    #[new]
    #[pyo3(signature = (*, logistic_converged, logistic_iteration_count, firth_iteration_count, firth_convergence_reason_code))]
    fn new(
        logistic_converged: PyReadonlyArray1<'_, bool>,
        logistic_iteration_count: PyReadonlyArray1<'_, i32>,
        firth_iteration_count: Option<PyReadonlyArray1<'_, i32>>,
        firth_convergence_reason_code: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<Self> {
        let converged = logistic_converged.as_slice()?.to_vec();
        let iteration_count = logistic_iteration_count.as_slice()?.to_vec();
        validate_py_vector_length(iteration_count.len(), converged.len(), "logistic_iteration_count")?;
        let firth_iterations =
            firth_iteration_count.map(|values| values.as_slice().map(<[i32]>::to_vec)).transpose()?;
        let firth_reasons =
            firth_convergence_reason_code.map(|values| values.as_slice().map(<[i32]>::to_vec)).transpose()?;
        if let Some(values) = &firth_iterations {
            validate_py_vector_length(values.len(), converged.len(), "firth_iteration_count")?;
        }
        if let Some(values) = &firth_reasons {
            validate_py_vector_length(values.len(), converged.len(), "firth_convergence_reason_code")?;
        }
        Ok(Self {
            data: native_engine::NullModelDiagnostics {
                logistic_converged: converged,
                logistic_iteration_count: iteration_count,
                firth_iteration_count: firth_iterations,
                firth_convergence_reason_code: firth_reasons,
            },
        })
    }

    #[getter]
    fn logistic_converged<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.data.logistic_converged.clone().into_pyarray(py)
    }

    #[getter]
    fn logistic_iteration_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.data.logistic_iteration_count.clone().into_pyarray(py)
    }

    #[getter]
    fn firth_iteration_count<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i32>>> {
        self.data.firth_iteration_count.clone().map(|values| values.into_pyarray(py))
    }

    #[getter]
    fn firth_convergence_reason_code<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i32>>> {
        self.data.firth_convergence_reason_code.clone().map(|values| values.into_pyarray(py))
    }
}

#[pymethods]
impl JaxPreparedChromosome {
    #[new]
    #[pyo3(signature = (*, state, diagnostics))]
    fn new(state: Py<PyAny>, diagnostics: Option<PyRef<'_, JaxNullModelDiagnostics>>) -> Self {
        Self { state, diagnostics: diagnostics.map(|value| value.data.clone()) }
    }

    #[getter]
    fn state(&self, py: Python<'_>) -> Py<PyAny> {
        self.state.clone_ref(py)
    }

    #[getter]
    fn diagnostics(&self, py: Python<'_>) -> PyResult<Option<Py<JaxNullModelDiagnostics>>> {
        self.diagnostics.clone().map(|data| Py::new(py, JaxNullModelDiagnostics { data })).transpose()
    }
}

#[pymethods]
impl JaxBinaryDiagnostics {
    #[new]
    #[pyo3(signature = (*, score_only_count, score_test_candidate_count, firth_candidate_count, firth_iteration_min, firth_iteration_median, firth_iteration_max, firth_converged_count, firth_failed_count, firth_numerical_failure_count, firth_max_iteration_failure_count, firth_invalid_statistic_failure_count, firth_step_halving_failure_count, pseudo_firth_attempt_count, pseudo_firth_success_count, newton_raphson_zero_start_attempt_count, newton_raphson_zero_start_success_count, newton_raphson_warm_start_attempt_count, newton_raphson_warm_start_success_count, sparse_correction_count, dense_correction_count))]
    fn new(
        score_only_count: i64,
        score_test_candidate_count: i64,
        firth_candidate_count: i64,
        firth_iteration_min: i64,
        firth_iteration_median: f64,
        firth_iteration_max: i64,
        firth_converged_count: i64,
        firth_failed_count: i64,
        firth_numerical_failure_count: i64,
        firth_max_iteration_failure_count: i64,
        firth_invalid_statistic_failure_count: i64,
        firth_step_halving_failure_count: i64,
        pseudo_firth_attempt_count: i64,
        pseudo_firth_success_count: i64,
        newton_raphson_zero_start_attempt_count: i64,
        newton_raphson_zero_start_success_count: i64,
        newton_raphson_warm_start_attempt_count: i64,
        newton_raphson_warm_start_success_count: i64,
        sparse_correction_count: i64,
        dense_correction_count: i64,
    ) -> Self {
        Self {
            data: native_engine::BinaryBatchDiagnostics {
                score_only_count,
                score_test_candidate_count,
                firth_candidate_count,
                firth_iteration_min,
                firth_iteration_median,
                firth_iteration_max,
                firth_converged_count,
                firth_failed_count,
                firth_numerical_failure_count,
                firth_max_iteration_failure_count,
                firth_invalid_statistic_failure_count,
                firth_step_halving_failure_count,
                pseudo_firth_attempt_count,
                pseudo_firth_success_count,
                newton_raphson_zero_start_attempt_count,
                newton_raphson_zero_start_success_count,
                newton_raphson_warm_start_attempt_count,
                newton_raphson_warm_start_success_count,
                sparse_correction_count,
                dense_correction_count,
            },
        }
    }

    #[getter]
    fn score_only_count(&self) -> i64 {
        self.data.score_only_count
    }

    #[getter]
    fn score_test_candidate_count(&self) -> i64 {
        self.data.score_test_candidate_count
    }

    #[getter]
    fn firth_candidate_count(&self) -> i64 {
        self.data.firth_candidate_count
    }

    #[getter]
    fn firth_iteration_min(&self) -> i64 {
        self.data.firth_iteration_min
    }

    #[getter]
    fn firth_iteration_median(&self) -> f64 {
        self.data.firth_iteration_median
    }

    #[getter]
    fn firth_iteration_max(&self) -> i64 {
        self.data.firth_iteration_max
    }

    #[getter]
    fn firth_converged_count(&self) -> i64 {
        self.data.firth_converged_count
    }

    #[getter]
    fn firth_failed_count(&self) -> i64 {
        self.data.firth_failed_count
    }

    #[getter]
    fn firth_numerical_failure_count(&self) -> i64 {
        self.data.firth_numerical_failure_count
    }

    #[getter]
    fn firth_max_iteration_failure_count(&self) -> i64 {
        self.data.firth_max_iteration_failure_count
    }

    #[getter]
    fn firth_invalid_statistic_failure_count(&self) -> i64 {
        self.data.firth_invalid_statistic_failure_count
    }

    #[getter]
    fn firth_step_halving_failure_count(&self) -> i64 {
        self.data.firth_step_halving_failure_count
    }

    #[getter]
    fn pseudo_firth_attempt_count(&self) -> i64 {
        self.data.pseudo_firth_attempt_count
    }

    #[getter]
    fn pseudo_firth_success_count(&self) -> i64 {
        self.data.pseudo_firth_success_count
    }

    #[getter]
    fn newton_raphson_zero_start_attempt_count(&self) -> i64 {
        self.data.newton_raphson_zero_start_attempt_count
    }

    #[getter]
    fn newton_raphson_zero_start_success_count(&self) -> i64 {
        self.data.newton_raphson_zero_start_success_count
    }

    #[getter]
    fn newton_raphson_warm_start_attempt_count(&self) -> i64 {
        self.data.newton_raphson_warm_start_attempt_count
    }

    #[getter]
    fn newton_raphson_warm_start_success_count(&self) -> i64 {
        self.data.newton_raphson_warm_start_success_count
    }

    #[getter]
    fn sparse_correction_count(&self) -> i64 {
        self.data.sparse_correction_count
    }

    #[getter]
    fn dense_correction_count(&self) -> i64 {
        self.data.dense_correction_count
    }
}

#[pymethods]
impl JaxHostAssociationBatch {
    #[new]
    #[pyo3(signature = (*, beta, standard_error, chi_squared, log10_p_value, extra_code, binary_diagnostics))]
    fn new(
        py: Python<'_>,
        beta: &Bound<'_, PyUntypedArray>,
        standard_error: &Bound<'_, PyUntypedArray>,
        chi_squared: &Bound<'_, PyUntypedArray>,
        log10_p_value: &Bound<'_, PyUntypedArray>,
        extra_code: Option<&Bound<'_, PyUntypedArray>>,
        binary_diagnostics: Option<PyRef<'_, JaxBinaryDiagnostics>>,
    ) -> PyResult<Self> {
        let statistics = parse_host_statistics(py, beta, standard_error, chi_squared, log10_p_value)?;
        let (trait_count, variant_count) = match &statistics {
            native_engine::HostAssociationStatistics::Float32(matrix) => (matrix.trait_count, matrix.variant_count),
            native_engine::HostAssociationStatistics::Float64(matrix) => (matrix.trait_count, matrix.variant_count),
        };
        let extra_codes =
            extra_code.map(|values| parse_extra_codes(py, values, trait_count, variant_count)).transpose()?;
        Ok(Self {
            data: native_engine::HostAssociationBatch {
                statistics,
                extra_codes,
                binary_diagnostics: binary_diagnostics.map(|value| value.data.clone()),
            },
        })
    }

    #[getter]
    fn beta(&self, py: Python<'_>) -> Py<PyAny> {
        statistic_array(py, &self.data.statistics, |matrix| &matrix.beta, |matrix| &matrix.beta)
    }

    #[getter]
    fn standard_error(&self, py: Python<'_>) -> Py<PyAny> {
        statistic_array(py, &self.data.statistics, |matrix| &matrix.standard_error, |matrix| &matrix.standard_error)
    }

    #[getter]
    fn chi_squared(&self, py: Python<'_>) -> Py<PyAny> {
        statistic_array(py, &self.data.statistics, |matrix| &matrix.chi_squared, |matrix| &matrix.chi_squared)
    }

    #[getter]
    fn log10_p_value(&self, py: Python<'_>) -> Py<PyAny> {
        statistic_array(py, &self.data.statistics, |matrix| &matrix.log10_p_value, |matrix| &matrix.log10_p_value)
    }

    #[getter]
    fn extra_code<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<i32>>> {
        self.data.extra_codes.as_ref().map(|matrix| {
            Array2::from_shape_vec((matrix.trait_count, matrix.variant_count), matrix.values.clone())
                .expect("validated extra-code matrix shape")
                .into_pyarray(py)
        })
    }

    #[getter]
    fn binary_diagnostics(&self, py: Python<'_>) -> PyResult<Option<Py<JaxBinaryDiagnostics>>> {
        self.data.binary_diagnostics.clone().map(|data| Py::new(py, JaxBinaryDiagnostics { data })).transpose()
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
    type GroupState = Py<PyAny>;

    fn prepare_group(
        &self,
        input: native_engine::GroupPreparationInput<'_>,
    ) -> Result<Self::GroupState, native_engine::BackendError> {
        let bridge_input = JaxGroupInput::from_native(input)?;
        Python::attach(|py| {
            let input_object = Py::new(py, bridge_input).map_err(backend_python_error)?;
            self.backend
                .bind(py)
                .call_method1("prepare_group", (input_object,))
                .map(Bound::unbind)
                .map_err(backend_python_error)
        })
    }

    fn prepare_chromosome(
        &self,
        group: &Self::GroupState,
        input: native_engine::ChromosomePreparationInput<'_>,
    ) -> Result<native_engine::PreparedChromosome<Self::ChromosomeState>, native_engine::BackendError> {
        let bridge_input = JaxChromosomeInput::from_native(input)?;
        Python::attach(|py| {
            let input_object = Py::new(py, bridge_input).map_err(backend_python_error)?;
            let result = self
                .backend
                .bind(py)
                .call_method1("prepare_chromosome", (group.clone_ref(py), input_object))
                .map_err(backend_python_error)?;
            let prepared = result
                .extract::<PyRef<'_, JaxPreparedChromosome>>()
                .map_err(|error| native_engine::BackendError::new(error.to_string()))?;
            Ok(native_engine::PreparedChromosome {
                state: prepared.state.clone_ref(py),
                null_model_diagnostics: prepared.diagnostics.clone(),
            })
        })
    }

    fn compute_batch(
        &self,
        chromosome: &Self::ChromosomeState,
        input: native_engine::GenotypeBatchInput<'_>,
    ) -> Result<Self::DeviceResult, native_engine::BackendError> {
        Python::attach(|py| {
            let bridge_input = JaxGenotypeBatch::from_native(py, input)?;
            let input_object = Py::new(py, bridge_input).map_err(backend_python_error)?;
            self.backend
                .bind(py)
                .call_method1("compute_batch", (chromosome.clone_ref(py), input_object))
                .map(Bound::unbind)
                .map_err(backend_python_error)
        })
    }

    fn materialize_batch(
        &self,
        result: Self::DeviceResult,
        input: native_engine::MaterializationInput<'_>,
    ) -> Result<native_engine::HostAssociationBatch, native_engine::BackendError> {
        let bridge_input = JaxMaterializationRequest::from_native(input);
        Python::attach(|py| {
            let input_object = Py::new(py, bridge_input).map_err(backend_python_error)?;
            let materialized = self
                .backend
                .bind(py)
                .call_method1("materialize_batch", (result, input_object))
                .map_err(backend_python_error)?;
            let host_batch = materialized
                .extract::<PyRef<'_, JaxHostAssociationBatch>>()
                .map_err(|error| native_engine::BackendError::new(error.to_string()))?;
            Ok(host_batch.data.clone())
        })
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<JaxBackendConfig>()?;
    module.add_class::<JaxGroupInput>()?;
    module.add_class::<JaxChromosomeInput>()?;
    module.add_class::<JaxGenotypeBatch>()?;
    module.add_class::<JaxMaterializationRequest>()?;
    module.add_class::<JaxPreparedChromosome>()?;
    module.add_class::<JaxNullModelDiagnostics>()?;
    module.add_class::<JaxBinaryDiagnostics>()?;
    module.add_class::<JaxHostAssociationBatch>()?;
    Ok(())
}

fn validate_value_count(
    observed_count: usize,
    row_count: usize,
    column_count: usize,
    value_label: &str,
) -> Result<(), native_engine::BackendError> {
    let expected_count = checked_product(row_count, column_count, value_label)?;
    if observed_count != expected_count {
        return Err(native_engine::BackendError::new(format!(
            "{value_label} contains {observed_count} values but shape ({row_count}, {column_count}) requires {expected_count}."
        )));
    }
    Ok(())
}

fn checked_product(left: usize, right: usize, value_label: &str) -> Result<usize, native_engine::BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| native_engine::BackendError::new(format!("{value_label} dimensions overflow usize.")))
}

fn validate_vector_length(
    observed_count: usize,
    expected_count: usize,
    value_label: &str,
) -> Result<(), native_engine::BackendError> {
    if observed_count != expected_count {
        return Err(native_engine::BackendError::new(format!(
            "{value_label} contains {observed_count} values but the genotype batch contains {expected_count} variants."
        )));
    }
    Ok(())
}

fn validate_py_vector_length(observed_count: usize, expected_count: usize, value_label: &str) -> PyResult<()> {
    if observed_count != expected_count {
        return Err(PyValueError::new_err(format!(
            "{value_label} contains {observed_count} values; expected {expected_count}."
        )));
    }
    Ok(())
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
    if observed_dtype.is_equiv_to(&dtype::<f32>(py)) {
        return parse_statistic_matrix::<f32>(beta, standard_error, chi_squared, log10_p_value)
            .map(native_engine::HostAssociationStatistics::Float32);
    }
    if observed_dtype.is_equiv_to(&dtype::<f64>(py)) {
        return parse_statistic_matrix::<f64>(beta, standard_error, chi_squared, log10_p_value)
            .map(native_engine::HostAssociationStatistics::Float64);
    }
    Err(PyValueError::new_err("Host association statistics must use float32 or float64 dtype."))
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

fn parse_extra_codes(
    py: Python<'_>,
    values: &Bound<'_, PyUntypedArray>,
    trait_count: usize,
    variant_count: usize,
) -> PyResult<native_engine::HostExtraCodeMatrix> {
    if !values.dtype().is_equiv_to(&dtype::<i32>(py)) {
        return Err(PyValueError::new_err("extra_code must use int32 dtype."));
    }
    let values = values.cast::<PyArray<i32, Ix2>>()?.readonly();
    if values.shape() != [trait_count, variant_count] {
        return Err(PyValueError::new_err(format!(
            "extra_code shape {:?} does not match statistic shape ({trait_count}, {variant_count}).",
            values.shape()
        )));
    }
    Ok(native_engine::HostExtraCodeMatrix { trait_count, variant_count, values: values.as_slice()?.to_vec() })
}

fn statistic_array<
    Float32Selector: FnOnce(&native_engine::HostAssociationStatisticMatrix<f32>) -> &Vec<f32>,
    Float64Selector: FnOnce(&native_engine::HostAssociationStatisticMatrix<f64>) -> &Vec<f64>,
>(
    py: Python<'_>,
    statistics: &native_engine::HostAssociationStatistics,
    float32_selector: Float32Selector,
    float64_selector: Float64Selector,
) -> Py<PyAny> {
    match statistics {
        native_engine::HostAssociationStatistics::Float32(matrix) => {
            Array2::from_shape_vec((matrix.trait_count, matrix.variant_count), float32_selector(matrix).clone())
                .expect("validated float32 statistic matrix shape")
                .into_pyarray(py)
                .into_any()
                .unbind()
        }
        native_engine::HostAssociationStatistics::Float64(matrix) => {
            Array2::from_shape_vec((matrix.trait_count, matrix.variant_count), float64_selector(matrix).clone())
                .expect("validated float64 statistic matrix shape")
                .into_pyarray(py)
                .into_any()
                .unbind()
        }
    }
}

fn backend_python_error(error: PyErr) -> native_engine::BackendError {
    native_engine::BackendError::new(error.to_string())
}
