//! PyO3 adapters for deterministic host-side planning policy.

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use g_plan as native_host_policy;

use super::config::NativePhenotypeComputeGroup;

#[pyclass]
pub(crate) struct NativeAssociationBackendPlan {
    backend_kind: String,
    association_mode: String,
    jax_device: String,
    genotype_format: String,
    uses_variant_major_packed8_delivery: bool,
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn plan_association_backend(
    association_mode: String,
    jax_device: String,
    gpu_genotype_format: String,
) -> PyResult<NativeAssociationBackendPlan> {
    let plan = native_host_policy::plan_association_backend(&association_mode, &jax_device, &gpu_genotype_format)
        .map_err(host_policy_error_to_py)?;
    Ok(NativeAssociationBackendPlan {
        backend_kind: plan.backend_kind.to_string(),
        association_mode: plan.association_mode,
        jax_device: plan.jax_device,
        genotype_format: plan.genotype_format,
        uses_variant_major_packed8_delivery: plan.uses_variant_major_packed8_delivery,
    })
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn resolve_association_mode_value(trait_type: String) -> String {
    native_host_policy::resolve_association_mode(&trait_type).to_string()
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn build_phenotype_compute_groups<'py>(
    py: Python<'py>,
    phenotype_names: Vec<String>,
    multi_phenotype_sample_mode: String,
) -> PyResult<Bound<'py, PyTuple>> {
    let groups = native_host_policy::build_phenotype_compute_groups(&phenotype_names, &multi_phenotype_sample_mode)
        .map_err(host_policy_error_to_py)?;
    let group_values = groups
        .into_iter()
        .map(|group| Py::new(py, NativePhenotypeComputeGroup::from_host_policy_payload(group)))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &group_values)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
fn build_phenotype_compute_group_id_value(
    group_mode: String,
    phenotype_indices: Vec<i64>,
    phenotype_names: Vec<String>,
    sample_mode: String,
    sample_set_fingerprint: Option<String>,
    covariate_design_fingerprint: Option<String>,
    prediction_alignment_fingerprint: Option<String>,
) -> String {
    native_host_policy::build_phenotype_compute_group_id(
        &group_mode,
        &phenotype_indices,
        &phenotype_names,
        &sample_mode,
        sample_set_fingerprint.as_deref(),
        covariate_design_fingerprint.as_deref(),
        prediction_alignment_fingerprint.as_deref(),
    )
}

#[pymethods]
impl NativeAssociationBackendPlan {
    #[getter]
    fn backend_kind(&self) -> &str {
        &self.backend_kind
    }

    #[getter]
    fn association_mode(&self) -> &str {
        &self.association_mode
    }

    #[getter]
    fn jax_device(&self) -> &str {
        &self.jax_device
    }

    #[getter]
    fn genotype_format(&self) -> &str {
        &self.genotype_format
    }

    #[getter]
    fn uses_variant_major_packed8_delivery(&self) -> bool {
        self.uses_variant_major_packed8_delivery
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeAssociationBackendPlan>()?;
    module.add_function(wrap_pyfunction!(plan_association_backend, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_association_mode_value, module)?)?;
    module.add_function(wrap_pyfunction!(build_phenotype_compute_groups, module)?)?;
    module.add_function(wrap_pyfunction!(build_phenotype_compute_group_id_value, module)?)?;
    Ok(())
}

fn host_policy_error_to_py(error: native_host_policy::HostPolicyError) -> PyErr {
    match error {
        native_host_policy::HostPolicyError::NotImplemented(message) => PyNotImplementedError::new_err(message),
        native_host_policy::HostPolicyError::Value(message) => PyValueError::new_err(message),
    }
}
