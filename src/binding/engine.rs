//! Private PyO3 adapter for the coarse JAX association backend.

use std::sync::{Arc, OnceLock};

use numpy::ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3, Ix1, Ix2};
use numpy::{
    Element, IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray,
    PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
#[cfg(target_os = "linux")]
use pyo3::types::PyCapsule;
use pyo3::types::{PyDict, PyModule};

use g_engine as native_engine;
use g_genotype as native_genotype;
use g_input as native_input;
use g_output as native_output;

/// Private adapter implementing the Python-free engine contract.
pub(crate) struct PyJaxBackend {
    backend: Py<PyAny>,
    genotype_delivery_capability: native_engine::GenotypeDeliveryCapability,
    kind: BackendKind,
}

static NVCOMP_FFI_REGISTRATION: OnceLock<Result<(), String>> = OnceLock::new();
static FIRTH_COMPONENTS_FFI_REGISTRATION: OnceLock<bool> = OnceLock::new();

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

#[pyclass(frozen)]
struct Packed8ArrayOwner {
    values: native_genotype::PooledPacked8Buffer,
}

#[pyclass(frozen)]
struct CompressedPacked8BatchOwner {
    batch: native_genotype::CompressedPacked8Batch,
}

#[pyclass(frozen)]
struct SampleSelectionArrayOwner {
    file_indices: Arc<[u32]>,
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
            let use_cuda_firth_components = device == g_plan::Device::Gpu && register_firth_components_ffi_target(py);
            let keyword_arguments =
                binary_firth_backend_keyword_arguments(py, kernels, *correction, use_cuda_firth_components)?;
            let backend = backend_module.getattr("BinaryFirthJaxBackend")?.call((), Some(&keyword_arguments))?.unbind();
            Ok(PyJaxBackend { backend, genotype_delivery_capability, kind: BackendKind::BinaryFirth })
        }
    }
}

fn register_nvcomp_ffi_target(py: Python<'_>) -> PyResult<()> {
    NVCOMP_FFI_REGISTRATION
        .get_or_init(|| register_nvcomp_ffi_target_once(py).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|message| PyRuntimeError::new_err(message.clone()))
        .copied()
}

#[cfg(target_os = "linux")]
fn register_nvcomp_ffi_target_once(py: Python<'_>) -> PyResult<()> {
    let nvcomp_module = PyModule::import(py, "nvidia.libnvcomp").map_err(|error| {
        PyRuntimeError::new_err(format!(
            "GPU packed8 delivery requires the official nvidia-libnvcomp-cu12 package: {error}"
        ))
    })?;
    let loaded_library = nvcomp_module.call_method0("load_library").map_err(|error| {
        PyRuntimeError::new_err(format!("The official nvidia.libnvcomp loader failed to load libnvcomp.so.5: {error}"))
    })?;
    if loaded_library.is_none() {
        return Err(PyRuntimeError::new_err("The official nvidia.libnvcomp loader could not find libnvcomp.so.5."));
    }

    let capability = g_genotype_cuda::initialize_nvcomp_runtime(0)
        .map_err(|error| PyRuntimeError::new_err(format!("nvCOMP runtime initialization failed: {error}")))?;
    let handler = g_genotype_cuda::packed8_deflate_ffi_handler(&capability);
    // SAFETY: `handler` is the process-lifetime address of the linked typed-XLA FFI
    // handler, and the capsule has no destructor or borrowed storage.
    let capsule = unsafe { PyCapsule::new_with_pointer(py, handler, c"xla._CUSTOM_CALL_TARGET")? };
    let keyword_arguments = PyDict::new(py);
    keyword_arguments.set_item("platform", "CUDA")?;
    keyword_arguments.set_item("api_version", 1)?;
    PyModule::import(py, "jax")?
        .getattr("ffi")?
        .call_method(
            "register_ffi_target",
            (g_genotype_cuda::PACKED8_DEFLATE_FFI_TARGET, capsule),
            Some(&keyword_arguments),
        )
        .map_err(|error| PyRuntimeError::new_err(format!("JAX nvCOMP FFI target registration failed: {error}")))?;
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn register_nvcomp_ffi_target_once(_py: Python<'_>) -> PyResult<()> {
    Err(PyRuntimeError::new_err("GPU packed8 delivery through nvCOMP is supported only on Linux."))
}

fn register_firth_components_ffi_target(py: Python<'_>) -> bool {
    *FIRTH_COMPONENTS_FFI_REGISTRATION.get_or_init(|| register_firth_components_ffi_target_once(py).unwrap_or(false))
}

#[cfg(target_os = "linux")]
fn register_firth_components_ffi_target_once(py: Python<'_>) -> PyResult<bool> {
    let Ok(capability) = g_compute_cuda::initialize_firth_components_runtime(0) else {
        return Ok(false);
    };
    let handler = g_compute_cuda::firth_components_ffi_handler(&capability);
    // SAFETY: The linked typed-XLA FFI handler has process lifetime, and the
    // capsule has no destructor or borrowed storage.
    let capsule = unsafe { PyCapsule::new_with_pointer(py, handler, c"xla._CUSTOM_CALL_TARGET")? };
    let keyword_arguments = PyDict::new(py);
    keyword_arguments.set_item("platform", "CUDA")?;
    keyword_arguments.set_item("api_version", 1)?;
    PyModule::import(py, "jax")?.getattr("ffi")?.call_method(
        "register_ffi_target",
        (g_compute_cuda::FIRTH_COMPONENTS_FFI_TARGET, capsule),
        Some(&keyword_arguments),
    )?;
    Ok(true)
}

#[cfg(not(target_os = "linux"))]
fn register_firth_components_ffi_target_once(_py: Python<'_>) -> PyResult<bool> {
    Ok(false)
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
    type DeviceResult = DeviceAssociationResult;
    type Error = PyJaxBackendError;
    type GroupState = Py<PyAny>;

    fn genotype_delivery_capability(&self) -> native_engine::GenotypeDeliveryCapability {
        self.genotype_delivery_capability
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
            parse_host_materialized_batch(py, &materialized, output_statistics, logical_variant_count)
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
            register_nvcomp_ffi_target(py)?;
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
                    let owner = Bound::new(py, SampleSelectionArrayOwner { file_indices })?;
                    let selection_view = ArrayView1::from(&owner.get().file_indices[..]);
                    let selected_sample_indices = unsafe {
                        // The frozen private owner retains immutable Arc storage as the
                        // ndarray base until Python finishes its one group-level upload.
                        PyArray1::borrow_from_array(&selection_view, owner.clone().into_any())
                    };
                    selected_sample_indices.readwrite().make_nonwriteable();
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
            let genotype_values = into_python_genotype_batch(py, genotypes, compute_variant_count, sample_count)
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
            let owner = Bound::new(py, CompressedPacked8BatchOwner { batch }).map_err(PyJaxBackendError::Python)?;
            let slab_view = ArrayView1::from(owner.get().batch.raw_deflate_slab());
            let compressed_slab = unsafe {
                // Both immutable arrays use the frozen owner as their base, so
                // pooled storage cannot be reclaimed before device_put consumes it.
                PyArray1::borrow_from_array(&slab_view, owner.clone().into_any())
            };
            compressed_slab.readwrite().make_nonwriteable();
            let metadata_view = ArrayView2::from_shape((logical_variant_count, 3), owner.get().batch.member_metadata())
                .map_err(|error| {
                    PyJaxBackendError::InvalidInput(format!("invalid compressed packed8 metadata shape: {error}"))
                })?;
            let compressed_metadata = unsafe {
                // The same frozen owner retains the metadata allocation for this
                // second non-writeable NumPy view.
                PyArray2::borrow_from_array(&metadata_view, owner.clone().into_any())
            };
            compressed_metadata.readwrite().make_nonwriteable();
            backend
                .call_method1(
                    "transfer_compressed_batch",
                    (group, compressed_slab, compressed_metadata, compute_variant_count),
                )
                .map(Bound::unbind)
                .map(TransferredGenotypeInput::CompressedPacked8)
                .map_err(PyJaxBackendError::Python)
        }
    }
}

fn into_python_genotype_batch(
    py: Python<'_>,
    genotypes: native_genotype::OwnedGenotypeBuffer,
    variant_count: usize,
    sample_count: usize,
) -> PyResult<Py<PyAny>> {
    match genotypes {
        native_genotype::OwnedGenotypeBuffer::Dosage(values) => {
            Ok(Array2::from_shape_vec((variant_count, sample_count), values)
                .expect("engine-validated dosage matrix shape")
                .into_pyarray(py)
                .into_any()
                .unbind())
        }
        native_genotype::OwnedGenotypeBuffer::Packed8(values) => {
            let owner = Bound::new(py, Packed8ArrayOwner { values })?;
            let array_view = ArrayView3::from_shape((variant_count, sample_count, 2), &owner.get().values[..])
                .map_err(|error| PyValueError::new_err(format!("Invalid packed8 genotype shape: {error}")))?;
            let values = unsafe {
                // The frozen private owner never mutates or reallocates its buffer. The ndarray
                // receives an owned reference to that owner as its base, so the pooled allocation
                // cannot be returned until the final ndarray reference is dropped.
                PyArray3::borrow_from_array(&array_view, owner.clone().into_any())
            };
            values.readwrite().make_nonwriteable();
            Ok(values.into_any().unbind())
        }
    }
}

fn parse_host_materialized_batch(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    output_statistics: Option<g_genotype_contracts::ChunkOutputStatistics>,
    logical_variant_count: usize,
) -> PyResult<native_engine::MaterializedAssociationBatch> {
    let association_payload = payload.getattr("association")?;
    let association = parse_host_association_batch(py, &association_payload, logical_variant_count)?;
    let raw_statistics_payload = payload.getattr("raw_packed8_statistics")?;
    let genotype_statistics = match (output_statistics, raw_statistics_payload.is_none()) {
        (Some(statistics), true) => native_engine::MaterializedGenotypeStatistics::Ready(statistics),
        (Some(_), false) => {
            return Err(PyValueError::new_err(
                "Host-decoded association output unexpectedly included packed8 raw statistics.",
            ));
        }
        (None, true) => {
            return Err(PyValueError::new_err(
                "Compressed packed8 association output omitted its raw genotype statistics.",
            ));
        }
        (None, false) => native_engine::MaterializedGenotypeStatistics::Packed8Raw(parse_packed8_raw_statistics(
            py,
            &raw_statistics_payload,
            logical_variant_count,
        )?),
    };
    Ok(native_engine::MaterializedAssociationBatch { association, genotype_statistics })
}

fn parse_packed8_raw_statistics(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    logical_variant_count: usize,
) -> PyResult<native_genotype::Packed8RawStatistics> {
    let dosage_sums =
        parse_host_vector::<u64>(py, &payload.getattr("dosage_sums")?, "dosage_sums", logical_variant_count)?;
    let dosage_square_sums = parse_host_vector::<u64>(
        py,
        &payload.getattr("dosage_square_sums")?,
        "dosage_square_sums",
        logical_variant_count,
    )?;
    let statuses = parse_host_vector::<u32>(py, &payload.getattr("statuses")?, "statuses", logical_variant_count)?;
    let selected_sample_count = payload.getattr("selected_sample_count")?.extract::<usize>()?;
    Ok(native_genotype::Packed8RawStatistics { dosage_sums, dosage_square_sums, statuses, selected_sample_count })
}

fn parse_host_vector<ElementType: Element + Copy>(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    label: &str,
    expected_value_count: usize,
) -> PyResult<Vec<ElementType>> {
    let values = payload.cast::<PyUntypedArray>()?;
    if !values.dtype().is_equiv_to(&dtype::<ElementType>(py)) {
        return Err(PyValueError::new_err(format!("{label} must use {} dtype.", dtype::<ElementType>(py))));
    }
    let values = values.cast::<PyArray<ElementType, Ix1>>()?.readonly();
    if values.shape() != [expected_value_count] {
        return Err(PyValueError::new_err(format!(
            "{label} shape {:?} does not match logical variant count {expected_value_count}.",
            values.shape()
        )));
    }
    Ok(copy_array_values(&values))
}

fn parse_host_association_batch(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    logical_variant_count: usize,
) -> PyResult<native_output::Regenie2StatisticBatch> {
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
    let (trait_count, materialized_variant_count) = (expected_shape[0], expected_shape[1]);
    if logical_variant_count != materialized_variant_count {
        return Err(PyValueError::new_err(format!(
            "materialized variant count {materialized_variant_count} does not match logical variant count {logical_variant_count}."
        )));
    }
    let correction_code_object = payload.getattr("correction_code")?;
    let correction_code = if correction_code_object.is_none() {
        None
    } else {
        Some(parse_correction_codes(
            py,
            correction_code_object.cast::<PyUntypedArray>()?,
            trait_count,
            logical_variant_count,
        )?)
    };
    Ok(native_output::Regenie2StatisticBatch {
        trait_count,
        variant_count: logical_variant_count,
        beta: copy_array_values(&beta),
        standard_error: copy_array_values(&standard_error),
        chi_squared: copy_array_values(&chi_squared),
        log10_p_value: copy_array_values(&log10_p_value),
        correction_code,
    })
}

fn copy_array_values<ElementType: Element + Copy, Dimension: numpy::ndarray::Dimension>(
    values: &PyReadonlyArray<'_, ElementType, Dimension>,
) -> Vec<ElementType> {
    match values.as_slice() {
        Ok(contiguous_values) => contiguous_values.to_vec(),
        Err(_) => values.as_array().iter().copied().collect(),
    }
}

fn parse_correction_codes(
    py: Python<'_>,
    values: &Bound<'_, PyUntypedArray>,
    trait_count: usize,
    logical_variant_count: usize,
) -> PyResult<Vec<u8>> {
    if !values.dtype().is_equiv_to(&dtype::<u8>(py)) {
        return Err(PyValueError::new_err("correction_code must use uint8 dtype."));
    }
    let values = values.cast::<PyArray<u8, Ix2>>()?.readonly();
    if values.shape() != [trait_count, logical_variant_count] {
        return Err(PyValueError::new_err(format!(
            "correction_code shape {:?} does not match statistic shape ({trait_count}, {logical_variant_count}).",
            values.shape()
        )));
    }
    Ok(copy_array_values(&values))
}
