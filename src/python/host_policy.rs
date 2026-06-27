//! PyO3 adapters for deterministic host-side planning policy.

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use g_plan as native_host_policy;

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_association_backend_payload<'py>(
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

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn resolve_association_mode_value(trait_type: String) -> String {
    native_host_policy::resolve_association_mode(&trait_type).to_string()
}

#[pyfunction]
pub(crate) fn normalize_binary_correction_payload<'py>(
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

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_phenotype_compute_groups_payload<'py>(
    py: Python<'py>,
    phenotype_names: Vec<String>,
    multi_phenotype_sample_mode: String,
) -> PyResult<Bound<'py, PyTuple>> {
    let groups = native_host_policy::build_phenotype_compute_groups(&phenotype_names, &multi_phenotype_sample_mode)
        .map_err(host_policy_error_to_py)?;
    let group_payloads =
        groups.iter().map(|group| phenotype_compute_group_payload_to_dict(py, group)).collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &group_payloads)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_phenotype_compute_group_id_value(
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

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_phenotype_output_directory_name(phenotype_index: i64, phenotype_name: String) -> String {
    native_host_policy::build_phenotype_output_directory_name(phenotype_index, &phenotype_name)
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
