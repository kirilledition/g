//! Private PyO3 adapter for the coarse JAX association backend.

mod array_conversion;
mod ffi_registration;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use g_engine as native_engine;
use g_genotype as native_genotype;
use g_input as native_input;

/// Private adapter implementing the Python-free engine contract.
pub(crate) struct PyJaxBackend {
    backend: Py<PyAny>,
    genotype_delivery_capability: native_engine::GenotypeDeliveryCapability,
    kind: BackendKind,
}

#[derive(Clone, Copy)]
enum BackendKind {
    Linear,
    BinaryScore,
    BinaryFirth,
}

pub(crate) enum TransferredGenotypeInput {
    Decoded { input: Py<PyAny>, output_statistics: g_genotype_contracts::ChunkOutputStatistics },
    CompressedPacked8(Py<PyAny>),
}

pub(crate) enum DeviceAssociationResult {
    Decoded { result: Py<PyAny>, output_statistics: g_genotype_contracts::ChunkOutputStatistics },
    CompressedPacked8(Py<PyAny>),
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
    device: g_plan::Device,
    plan: g_runner::JaxAssociationBackendPlan<'_>,
) -> PyResult<PyJaxBackend> {
    ffi_registration::validate_jax_runtime_versions(py)?;
    let genotype_delivery_capability = match device {
        g_plan::Device::Cpu => native_engine::GenotypeDeliveryCapability::HostOnly,
        g_plan::Device::Gpu => native_engine::GenotypeDeliveryCapability::RawDeflatePacked8,
    };
    let backend_module = PyModule::import(py, "g.jax_backend")?;
    match plan {
        g_runner::JaxAssociationBackendPlan::Linear(kernel) => {
            let keyword_arguments = PyDict::new(py);
            keyword_arguments.set_item("minimum_variance", kernel.minimum_variance.get())?;
            keyword_arguments.set_item("relative_variance_tolerance", kernel.relative_variance_tolerance.get())?;
            let backend = backend_module.getattr("LinearJaxBackend")?.call((), Some(&keyword_arguments))?.unbind();
            Ok(PyJaxBackend { backend, genotype_delivery_capability, kind: BackendKind::Linear })
        }
        g_runner::JaxAssociationBackendPlan::BinaryScore(kernels) => {
            let keyword_arguments = binary_score_backend_keyword_arguments(py, kernels)?;
            let backend = backend_module.getattr("BinaryScoreJaxBackend")?.call((), Some(&keyword_arguments))?.unbind();
            Ok(PyJaxBackend { backend, genotype_delivery_capability, kind: BackendKind::BinaryScore })
        }
        g_runner::JaxAssociationBackendPlan::BinaryFirth { correction, kernels } => {
            let use_cuda_firth_components =
                device == g_plan::Device::Gpu && ffi_registration::register_firth_components_ffi_target(py)?;
            let keyword_arguments =
                binary_firth_backend_keyword_arguments(py, kernels, *correction, use_cuda_firth_components)?;
            let backend = backend_module.getattr("BinaryFirthJaxBackend")?.call((), Some(&keyword_arguments))?.unbind();
            Ok(PyJaxBackend { backend, genotype_delivery_capability, kind: BackendKind::BinaryFirth })
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
    use_cuda_firth_components: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let keyword_arguments = binary_score_backend_keyword_arguments(py, kernels)?;
    keyword_arguments.set_item("p_threshold", correction.p_threshold.get())?;
    keyword_arguments.set_item("firth_se", correction.firth_se)?;
    keyword_arguments.set_item("firth_batch_size", kernels.firth.batch_size)?;
    keyword_arguments.set_item("firth_candidate_capacity", kernels.firth.candidate_capacity)?;
    keyword_arguments.set_item("firth_maximum_iterations", kernels.firth.maximum_iterations)?;
    keyword_arguments.set_item("firth_gradient_tolerance", kernels.firth.gradient_tolerance.get())?;
    keyword_arguments.set_item("firth_maximum_step_size", kernels.firth.maximum_step_size.get())?;
    keyword_arguments.set_item("firth_pseudo_maximum_iterations", kernels.firth.pseudo_maximum_iterations)?;
    keyword_arguments
        .set_item("firth_pseudo_inner_maximum_iterations", kernels.firth.pseudo_inner_maximum_iterations)?;
    keyword_arguments.set_item("firth_line_search_maximum_attempts", kernels.firth.line_search_maximum_attempts)?;
    keyword_arguments
        .set_item("firth_sparse_carrier_dosage_threshold", kernels.firth.sparse_carrier_dosage_threshold.get())?;
    keyword_arguments.set_item("use_cuda_firth_components", use_cuda_firth_components)?;
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
    type TransferredInput = TransferredGenotypeInput;
    type SharedSourceBatch = Py<PyAny>;
    type DeviceResult = DeviceAssociationResult;
    type Error = PyJaxBackendError;
    type GroupState = Py<PyAny>;

    fn genotype_delivery_capability(&self) -> native_engine::GenotypeDeliveryCapability {
        self.genotype_delivery_capability
    }

    fn supports_shared_source_batches(&self) -> bool {
        matches!(self.kind, BackendKind::Linear)
            && self.genotype_delivery_capability == native_engine::GenotypeDeliveryCapability::RawDeflatePacked8
    }

    fn prepare_shared_source(
        &self,
        input: native_genotype::GenotypeBatch,
    ) -> Result<Option<Self::SharedSourceBatch>, Self::Error> {
        if !self.supports_shared_source_batches() {
            return Ok(None);
        }
        let native_genotype::GenotypeBatchPayload::CompressedPacked8(batch) = input.payload else {
            return Err(PyJaxBackendError::InvalidInput("shared source requires compressed packed8 input".to_string()));
        };
        Python::attach(|py| {
            let arrays = array_conversion::compressed_batch_arrays(py, batch, input.logical_variant_count)?;
            self.backend
                .bind(py)
                .call_method1(
                    "prepare_shared_source",
                    (arrays.slab, arrays.metadata, input.compute_variant_count, input.sample_count),
                )
                .map(Bound::unbind)
                .map(Some)
                .map_err(PyJaxBackendError::Python)
        })
    }

    fn select_shared_source(
        &self,
        group: &Self::GroupState,
        source: &Self::SharedSourceBatch,
    ) -> Result<Option<Self::TransferredInput>, Self::Error> {
        if !self.supports_shared_source_batches() {
            return Ok(None);
        }
        Python::attach(|py| {
            self.backend
                .bind(py)
                .call_method1("select_shared_source", (group.bind(py), source.bind(py)))
                .map(Bound::unbind)
                .map(TransferredGenotypeInput::CompressedPacked8)
                .map(Some)
                .map_err(PyJaxBackendError::Python)
        })
    }

    fn release_shared_source(&self, source: Self::SharedSourceBatch) {
        Python::attach(|_| drop(source));
    }

    fn prepare_group(&self, input: native_engine::GroupPreparationInput) -> Result<Self::GroupState, Self::Error> {
        Python::attach(|py| {
            let native_engine::GroupPreparationInput { phenotypes, covariates, genotype_transfer } = input;
            let phenotype_matrix =
                Array2::from_shape_vec((phenotypes.trait_count, phenotypes.sample_count), phenotypes.values)
                    .expect("engine-validated phenotype matrix shape")
                    .into_pyarray(py);
            let covariate_matrix =
                Array2::from_shape_vec((covariates.sample_count, covariates.covariate_count), covariates.values)
                    .expect("engine-validated covariate matrix shape")
                    .into_pyarray(py);
            prepare_python_group(py, self.backend.bind(py), &phenotype_matrix, &covariate_matrix, genotype_transfer)
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
        predictions: native_input::ChromosomePredictionMatrix,
    ) -> Result<native_engine::PreparedChromosome<Self::ChromosomeState>, Self::Error> {
        Python::attach(|py| {
            let prediction_matrix = Array2::from_shape_vec(
                (predictions.trait_count, predictions.sample_count),
                predictions.prediction_values,
            )
            .expect("input-validated prediction matrix shape")
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

    fn transfer_batch(
        &self,
        group: &Self::GroupState,
        input: native_genotype::GenotypeBatch,
    ) -> Result<Self::TransferredInput, Self::Error> {
        Python::attach(|py| transfer_genotype_batch(py, self.backend.bind(py), group.bind(py), input))
    }

    fn compute_batch(
        &self,
        chromosome: &Self::ChromosomeState,
        input: Self::TransferredInput,
    ) -> Result<Self::DeviceResult, Self::Error> {
        Python::attach(|py| match input {
            TransferredGenotypeInput::Decoded { input, output_statistics } => self
                .backend
                .bind(py)
                .call_method1("compute_batch", (chromosome.bind(py), input))
                .map(Bound::unbind)
                .map(|result| DeviceAssociationResult::Decoded { result, output_statistics })
                .map_err(PyJaxBackendError::Python),
            TransferredGenotypeInput::CompressedPacked8(input) => self
                .backend
                .bind(py)
                .call_method1("compute_batch", (chromosome.bind(py), input))
                .map(Bound::unbind)
                .map(DeviceAssociationResult::CompressedPacked8)
                .map_err(PyJaxBackendError::Python),
        })
    }

    fn materialize_batch(
        &self,
        result: Self::DeviceResult,
        active_trait_indices: Option<&[usize]>,
        logical_variant_count: usize,
    ) -> Result<native_engine::MaterializedAssociationBatch, Self::Error> {
        Python::attach(|py| {
            let (result, output_statistics) = match result {
                DeviceAssociationResult::Decoded { result, output_statistics } => (result, Some(output_statistics)),
                DeviceAssociationResult::CompressedPacked8(result) => (result, None),
            };
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
                .call_method1("materialize_batch", (result, active_trait_indices, logical_variant_count))
                .map_err(PyJaxBackendError::Python)?;
            array_conversion::parse_host_materialized_batch(py, &materialized, output_statistics, logical_variant_count)
                .map_err(PyJaxBackendError::Python)
        })
    }
}

fn prepare_python_group<'py>(
    py: Python<'py>,
    backend: &Bound<'py, PyAny>,
    phenotype_matrix: &Bound<'py, PyArray2<f32>>,
    covariate_matrix: &Bound<'py, PyArray2<f32>>,
    genotype_transfer: native_engine::GenotypeTransferPreparation,
) -> PyResult<Bound<'py, PyAny>> {
    match genotype_transfer {
        native_engine::GenotypeTransferPreparation::Host => backend.call_method1(
            "prepare_group",
            (phenotype_matrix, covariate_matrix, py.None(), py.None(), py.None(), py.None()),
        ),
        native_engine::GenotypeTransferPreparation::CompressedPacked8(transfer) => {
            ffi_registration::register_nvcomp_ffi_target(py)?;
            let source_sample_count = transfer.file_sample_count;
            let selected_sample_count = transfer.selected_sample_count;
            match transfer.sample_selection {
                native_genotype::CompressedPacked8SampleSelection::Contiguous { file_index_start } => backend
                    .call_method1(
                        "prepare_group",
                        (
                            phenotype_matrix,
                            covariate_matrix,
                            source_sample_count,
                            selected_sample_count,
                            file_index_start,
                            py.None(),
                        ),
                    ),
                native_genotype::CompressedPacked8SampleSelection::Indexed { file_indices } => {
                    let selected_sample_indices = array_conversion::into_python_sample_selection(py, file_indices)?;
                    backend.call_method1(
                        "prepare_group",
                        (
                            phenotype_matrix,
                            covariate_matrix,
                            source_sample_count,
                            selected_sample_count,
                            py.None(),
                            selected_sample_indices,
                        ),
                    )
                }
            }
        }
    }
}

fn transfer_genotype_batch(
    py: Python<'_>,
    backend: &Bound<'_, PyAny>,
    group: &Bound<'_, PyAny>,
    input: native_genotype::GenotypeBatch,
) -> Result<TransferredGenotypeInput, PyJaxBackendError> {
    let native_genotype::GenotypeBatch {
        variant_start_index: _,
        logical_variant_count,
        compute_variant_count,
        sample_count,
        payload,
    } = input;
    match payload {
        native_genotype::GenotypeBatchPayload::Decoded { genotypes, statistics } => {
            let output_statistics = statistics.output;
            let native_genotype::ChunkComputeStatistics {
                genotype_mean,
                imputed_dosage_square_sum,
                sparse_candidate_mask,
            } = statistics.compute;
            let genotype_values =
                array_conversion::into_python_genotype_batch(py, genotypes, compute_variant_count, sample_count)
                    .map_err(PyJaxBackendError::Python)?;
            backend
                .call_method1(
                    "transfer_batch",
                    (
                        genotype_values,
                        genotype_mean.into_pyarray(py),
                        imputed_dosage_square_sum.map(|values| values.into_pyarray(py)),
                        sparse_candidate_mask.map(|values| values.into_pyarray(py)),
                    ),
                )
                .map(Bound::unbind)
                .map(|input| TransferredGenotypeInput::Decoded { input, output_statistics })
                .map_err(PyJaxBackendError::Python)
        }
        native_genotype::GenotypeBatchPayload::CompressedPacked8(batch) => {
            let arrays = array_conversion::compressed_batch_arrays(py, batch, logical_variant_count)?;
            backend
                .call_method1("transfer_compressed_batch", (group, arrays.slab, arrays.metadata, compute_variant_count))
                .map(Bound::unbind)
                .map(TransferredGenotypeInput::CompressedPacked8)
                .map_err(PyJaxBackendError::Python)
        }
    }
}
