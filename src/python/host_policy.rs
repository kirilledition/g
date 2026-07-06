//! PyO3 adapters for deterministic host-side planning policy.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use g_plan as native_host_policy;

use super::config::NativePhenotypeComputeGroup;

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn build_phenotype_compute_groups<'py>(
    py: Python<'py>,
    phenotype_names: Vec<String>,
    multi_phenotype_sample_mode: String,
) -> PyResult<Bound<'py, PyTuple>> {
    let sample_mode = parse_multi_phenotype_sample_mode(&multi_phenotype_sample_mode)?;
    let groups = native_host_policy::build_phenotype_compute_groups(&phenotype_names, sample_mode)
        .map_err(host_policy_error_to_py)?;
    let group_values = groups
        .iter()
        .map(|group| Py::new(py, NativePhenotypeComputeGroup::from_native_group(group)))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &group_values)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(build_phenotype_compute_groups, module)?)?;
    Ok(())
}

fn parse_multi_phenotype_sample_mode(value: &str) -> PyResult<native_host_policy::MultiPhenotypeSampleMode> {
    match value {
        "per-phenotype" => Ok(native_host_policy::MultiPhenotypeSampleMode::PerPhenotype),
        "complete-case" => Ok(native_host_policy::MultiPhenotypeSampleMode::CompleteCase),
        unsupported_value => Err(PyValueError::new_err(format!(
            "multi_phenotype_sample_mode must be per-phenotype or complete-case, observed '{unsupported_value}'."
        ))),
    }
}

fn host_policy_error_to_py(error: native_host_policy::HostPolicyError) -> PyErr {
    match error {
        native_host_policy::HostPolicyError::NotImplemented(message) => PyValueError::new_err(message),
        native_host_policy::HostPolicyError::Value(message) => PyValueError::new_err(message),
    }
}
