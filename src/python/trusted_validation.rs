//! PyO3 adapters for trusted BGEN validation cache metadata.

use std::path::Path;

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::trusted_validation as native_trusted_validation;

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_trusted_bgen_validation_fingerprint_value(
    bgen_path: String,
    sample_count: i64,
    variant_count: i64,
    trusted_no_missing_diploid: bool,
) -> PyResult<String> {
    native_trusted_validation::build_trusted_bgen_validation_fingerprint(
        &native_trusted_validation::TrustedBgenValidationFingerprintInput {
            bgen_path: bgen_path.into(),
            sample_count,
            variant_count,
            trusted_no_missing_diploid,
        },
    )
    .map_err(PyOSError::new_err)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_trusted_bgen_validation_cache_path_value(cache_directory: String, fingerprint: String) -> String {
    native_trusted_validation::build_trusted_bgen_validation_cache_path(Path::new(&cache_directory), &fingerprint)
        .display()
        .to_string()
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_trusted_bgen_validation_cache_payload<'py>(
    py: Python<'py>,
    fingerprint: String,
    bgen_path: String,
    sample_count: i64,
    variant_count: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let cache_payload = native_trusted_validation::build_trusted_bgen_validation_cache_payload(
        fingerprint,
        Path::new(&bgen_path),
        sample_count,
        variant_count,
    )
    .map_err(PyOSError::new_err)?;
    let payload = PyDict::new(py);
    payload.set_item("schema_version", cache_payload.schema_version)?;
    payload.set_item("fingerprint", cache_payload.fingerprint)?;
    payload.set_item("bgen_path", cache_payload.bgen_path)?;
    payload.set_item("sample_count", cache_payload.sample_count)?;
    payload.set_item("variant_count", cache_payload.variant_count)?;
    Ok(payload)
}
