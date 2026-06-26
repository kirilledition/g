//! PyO3 adapters for engine-owned preflight helpers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use g_engine::preflight as native_preflight;

#[pyfunction]
pub(crate) fn resolve_preflight_variant_count(variant_count: i64, variant_limit: Option<i64>) -> PyResult<i64> {
    if variant_count <= 0 {
        return Err(PyValueError::new_err(native_preflight::PreflightError::EmptyBgenInput.to_string()));
    }
    if matches!(variant_limit, Some(limit) if limit <= 0) {
        return Err(PyValueError::new_err(native_preflight::PreflightError::EmptyBgenScan.to_string()));
    }
    let native_variant_count = usize_count("variant_count", variant_count)?;
    let native_variant_limit = variant_limit.map(|limit| usize_count("variant_limit", limit)).transpose()?;
    let scanned_variant_count =
        native_preflight::resolve_scanned_variant_count(native_variant_count, native_variant_limit)
            .map_err(|error| preflight_error_to_py(&error))?;
    i64::try_from(scanned_variant_count)
        .map_err(|error| PyValueError::new_err(format!("Scanned variant count exceeds int64 capacity: {error}")))
}

#[pyfunction]
pub(crate) fn build_preflight_report_payload<'py>(
    py: Python<'py>,
    sample_count: i64,
    covariate_count: i64,
    chromosome_count: i64,
    trusted_no_missing_diploid: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = native_preflight::build_preflight_report_payload(
        sample_count,
        covariate_count,
        chromosome_count,
        trusted_no_missing_diploid,
    )
    .map_err(|error| preflight_error_to_py(&error))?;
    let payload_dict = PyDict::new(py);
    payload_dict.set_item("sample_count", payload.sample_count)?;
    payload_dict.set_item("covariate_count", payload.covariate_count)?;
    payload_dict.set_item("chromosome_count", payload.chromosome_count)?;
    payload_dict.set_item("warning_messages", PyTuple::new(py, payload.warning_messages)?)?;
    Ok(payload_dict)
}

fn usize_count(label: &str, count: i64) -> PyResult<usize> {
    usize::try_from(count).map_err(|_| PyValueError::new_err(format!("{label} cannot be negative: {count}")))
}

fn preflight_error_to_py(error: &native_preflight::PreflightError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
