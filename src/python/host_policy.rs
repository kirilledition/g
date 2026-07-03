//! PyO3 adapters for deterministic host-side planning policy.

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

use g_plan as native_host_policy;

#[pyclass]
pub(crate) struct NativeHostPlanningPolicy;

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeHostAssociationBackendPlan {
    backend_kind: String,
    association_mode: String,
    jax_device: String,
    genotype_format: String,
    uses_variant_major_packed8_delivery: bool,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeHostPhenotypeComputeGroupPlan {
    group_mode: String,
    phenotype_indices: Vec<i64>,
    phenotype_names: Vec<String>,
    sample_mode: String,
    sample_set_fingerprint: Option<String>,
    covariate_design_fingerprint: Option<String>,
    prediction_alignment_fingerprint: Option<String>,
}

impl NativeHostAssociationBackendPlan {
    fn from_payload(payload: native_host_policy::AssociationBackendPlanPayload) -> Self {
        Self {
            backend_kind: payload.backend_kind.to_string(),
            association_mode: payload.association_mode,
            jax_device: payload.jax_device,
            genotype_format: payload.genotype_format,
            uses_variant_major_packed8_delivery: payload.uses_variant_major_packed8_delivery,
        }
    }
}

impl NativeHostPhenotypeComputeGroupPlan {
    fn from_payload(payload: native_host_policy::PhenotypeComputeGroupPayload) -> Self {
        Self {
            group_mode: payload.group_mode.to_string(),
            phenotype_indices: payload.phenotype_indices,
            phenotype_names: payload.phenotype_names,
            sample_mode: payload.sample_mode.to_string(),
            sample_set_fingerprint: payload.sample_set_fingerprint,
            covariate_design_fingerprint: payload.covariate_design_fingerprint,
            prediction_alignment_fingerprint: payload.prediction_alignment_fingerprint,
        }
    }
}

#[pymethods]
impl NativeHostPlanningPolicy {
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    fn plan_association_backend(
        &self,
        association_mode: String,
        jax_device: String,
        gpu_genotype_format: String,
    ) -> PyResult<NativeHostAssociationBackendPlan> {
        native_host_policy::plan_association_backend(&association_mode, &jax_device, &gpu_genotype_format)
            .map(NativeHostAssociationBackendPlan::from_payload)
            .map_err(host_policy_error_to_py)
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    fn plan_association_backend_payload<'py>(
        &self,
        py: Python<'py>,
        association_mode: String,
        jax_device: String,
        gpu_genotype_format: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        let plan = native_host_policy::plan_association_backend(&association_mode, &jax_device, &gpu_genotype_format)
            .map_err(host_policy_error_to_py)?;
        let payload = PyDict::new(py);
        payload.set_item("backend_kind", plan.backend_kind)?;
        payload.set_item("association_mode", plan.association_mode)?;
        payload.set_item("jax_device", plan.jax_device)?;
        payload.set_item("genotype_format", plan.genotype_format)?;
        payload.set_item("uses_variant_major_packed8_delivery", plan.uses_variant_major_packed8_delivery)?;
        Ok(payload)
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    fn resolve_association_mode_value(&self, trait_type: String) -> String {
        native_host_policy::resolve_association_mode(&trait_type).to_string()
    }

    #[allow(clippy::unused_self)]
    fn normalize_binary_correction_payload<'py>(
        &self,
        py: Python<'py>,
        firth: bool,
        approx: bool,
        spa: bool,
        p_threshold: f64,
        firth_se: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let plan = native_host_policy::normalize_binary_correction(firth, approx, spa, p_threshold, firth_se)
            .map_err(host_policy_error_to_py)?;
        let payload = PyDict::new(py);
        payload.set_item("method", plan.method)?;
        payload.set_item("p_threshold", plan.p_threshold)?;
        payload.set_item("firth_se", plan.firth_se)?;
        Ok(payload)
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    fn build_phenotype_compute_groups(
        &self,
        phenotype_names: Vec<String>,
        multi_phenotype_sample_mode: String,
    ) -> PyResult<Vec<NativeHostPhenotypeComputeGroupPlan>> {
        native_host_policy::build_phenotype_compute_groups(&phenotype_names, &multi_phenotype_sample_mode)
            .map(|groups| groups.into_iter().map(NativeHostPhenotypeComputeGroupPlan::from_payload).collect())
            .map_err(host_policy_error_to_py)
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    fn build_phenotype_compute_groups_payload<'py>(
        &self,
        py: Python<'py>,
        phenotype_names: Vec<String>,
        multi_phenotype_sample_mode: String,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let groups = native_host_policy::build_phenotype_compute_groups(&phenotype_names, &multi_phenotype_sample_mode)
            .map_err(host_policy_error_to_py)?;
        let group_payloads = groups
            .iter()
            .map(|group| phenotype_compute_group_payload_to_dict(py, group))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, &group_payloads)
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn build_phenotype_compute_group_id_value(
        &self,
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

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    fn build_phenotype_output_directory_name(&self, phenotype_index: i64, phenotype_name: String) -> String {
        native_host_policy::build_phenotype_output_directory_name(phenotype_index, &phenotype_name)
    }
}

#[pymethods]
impl NativeHostAssociationBackendPlan {
    #[getter]
    fn backend_kind(&self) -> &str {
        self.backend_kind.as_str()
    }

    #[getter]
    fn association_mode(&self) -> &str {
        self.association_mode.as_str()
    }

    #[getter]
    fn jax_device(&self) -> &str {
        self.jax_device.as_str()
    }

    #[getter]
    fn genotype_format(&self) -> &str {
        self.genotype_format.as_str()
    }

    #[getter]
    fn uses_variant_major_packed8_delivery(&self) -> bool {
        self.uses_variant_major_packed8_delivery
    }
}

#[pymethods]
impl NativeHostPhenotypeComputeGroupPlan {
    #[getter]
    fn group_mode(&self) -> &str {
        self.group_mode.as_str()
    }

    #[getter]
    fn phenotype_indices(&self) -> Vec<i64> {
        self.phenotype_indices.clone()
    }

    #[getter]
    fn phenotype_names(&self) -> Vec<String> {
        self.phenotype_names.clone()
    }

    #[getter]
    fn sample_mode(&self) -> &str {
        self.sample_mode.as_str()
    }

    #[getter]
    fn sample_set_fingerprint(&self) -> Option<String> {
        self.sample_set_fingerprint.clone()
    }

    #[getter]
    fn covariate_design_fingerprint(&self) -> Option<String> {
        self.covariate_design_fingerprint.clone()
    }

    #[getter]
    fn prediction_alignment_fingerprint(&self) -> Option<String> {
        self.prediction_alignment_fingerprint.clone()
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeHostPlanningPolicy>()?;
    module.add_class::<NativeHostAssociationBackendPlan>()?;
    module.add_class::<NativeHostPhenotypeComputeGroupPlan>()?;
    Ok(())
}

fn phenotype_compute_group_payload_to_dict<'py>(
    py: Python<'py>,
    group: &native_host_policy::PhenotypeComputeGroupPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("group_mode", group.group_mode)?;
    payload.set_item("phenotype_indices", PyTuple::new(py, &group.phenotype_indices)?)?;
    payload.set_item("phenotype_names", PyTuple::new(py, &group.phenotype_names)?)?;
    payload.set_item("sample_mode", group.sample_mode)?;
    set_optional_string(py, &payload, "sample_set_fingerprint", group.sample_set_fingerprint.as_deref())?;
    set_optional_string(py, &payload, "covariate_design_fingerprint", group.covariate_design_fingerprint.as_deref())?;
    set_optional_string(
        py,
        &payload,
        "prediction_alignment_fingerprint",
        group.prediction_alignment_fingerprint.as_deref(),
    )?;
    Ok(payload)
}

fn set_optional_string(py: Python<'_>, payload: &Bound<'_, PyDict>, key: &str, value: Option<&str>) -> PyResult<()> {
    match value {
        Some(text) => payload.set_item(key, text),
        None => payload.set_item(key, py.None()),
    }
}

fn host_policy_error_to_py(error: native_host_policy::HostPolicyError) -> PyErr {
    match error {
        native_host_policy::HostPolicyError::NotImplemented(message) => PyNotImplementedError::new_err(message),
        native_host_policy::HostPolicyError::Value(message) => PyValueError::new_err(message),
    }
}
