//! PyO3 adapters for deterministic host-side planning policy.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use g_plan as native_host_policy;

use super::NativePhenotypeComputeGroup;
use crate::binding::errors;

#[pyclass]
pub(crate) struct NativeAssociationBackendPlan {
    inner: native_host_policy::AssociationBackendPlan,
}

#[pymethods]
impl NativeAssociationBackendPlan {
    #[new]
    fn new(association_mode: &str, jax_device: &str, gpu_genotype_format: &str) -> PyResult<Self> {
        let association_mode = parse_association_mode(association_mode)?;
        let jax_device = parse_device(jax_device)?;
        let gpu_genotype_format = parse_gpu_genotype_format(gpu_genotype_format)?;
        native_host_policy::plan_association_backend(association_mode, jax_device, gpu_genotype_format)
            .map(|inner| Self { inner })
            .map_err(|error| errors::convert_prepared_plan_error(&error))
    }

    #[getter]
    fn backend_kind(&self) -> &'static str {
        self.inner.kind.as_str()
    }

    #[getter]
    fn association_mode(&self) -> &'static str {
        self.inner.association_mode.as_str()
    }

    #[getter]
    fn jax_device(&self) -> &'static str {
        self.inner.device.as_str()
    }

    #[getter]
    fn genotype_format(&self) -> &'static str {
        self.inner.resolved_genotype_format.as_str()
    }

    #[getter]
    fn uses_variant_major_packed8_delivery(&self) -> bool {
        self.inner.resolved_genotype_format == native_host_policy::GpuGenotypeFormat::Packed8
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn build_phenotype_compute_groups<'py>(
    py: Python<'py>,
    phenotype_names: Vec<String>,
    multi_phenotype_sample_mode: String,
) -> PyResult<Bound<'py, PyTuple>> {
    let sample_mode = parse_multi_phenotype_sample_mode(&multi_phenotype_sample_mode)?;
    let groups = native_host_policy::build_phenotype_compute_groups(&phenotype_names, sample_mode)
        .map_err(errors::convert_host_policy_error)?;
    let group_values = groups
        .iter()
        .map(|group| Py::new(py, NativePhenotypeComputeGroup::from_native_group(group)))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &group_values)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeAssociationBackendPlan>()?;
    module.add_function(wrap_pyfunction!(build_phenotype_compute_groups, module)?)?;
    Ok(())
}

fn parse_association_mode(value: &str) -> PyResult<native_host_policy::AssociationMode> {
    native_host_policy::AssociationMode::from_str_value(value).ok_or_else(|| {
        enum_parse_error("association_mode", native_host_policy::AssociationMode::accepted_values(), value)
    })
}

fn parse_device(value: &str) -> PyResult<native_host_policy::Device> {
    native_host_policy::Device::from_str_value(value)
        .ok_or_else(|| enum_parse_error("jax_device", native_host_policy::Device::accepted_values(), value))
}

fn parse_gpu_genotype_format(value: &str) -> PyResult<native_host_policy::GpuGenotypeFormat> {
    native_host_policy::GpuGenotypeFormat::from_str_value(value).ok_or_else(|| {
        enum_parse_error("gpu_genotype_format", native_host_policy::GpuGenotypeFormat::accepted_values(), value)
    })
}

fn parse_multi_phenotype_sample_mode(value: &str) -> PyResult<native_host_policy::MultiPhenotypeSampleMode> {
    native_host_policy::MultiPhenotypeSampleMode::from_str_value(value).ok_or_else(|| {
        enum_parse_error(
            "multi_phenotype_sample_mode",
            native_host_policy::MultiPhenotypeSampleMode::accepted_values(),
            value,
        )
    })
}

fn enum_parse_error(field_name: &str, accepted_values: &[&str], observed_value: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{field_name} must be one of {}, observed '{observed_value}'.",
        accepted_values.join(", "),
    ))
}
