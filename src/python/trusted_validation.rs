//! PyO3 adapters for trusted BGEN validation cache metadata.

use std::path::Path;

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use g_runtime::trusted_validation as native_trusted_validation;

#[pyclass]
pub(crate) struct NativeTrustedBgenValidationCacheLookupPlan {
    inner: native_trusted_validation::TrustedBgenValidationCacheLookupPlan,
}

#[pymethods]
impl NativeTrustedBgenValidationCacheLookupPlan {
    #[getter]
    fn should_mark_validated(&self) -> bool {
        self.inner.should_mark_validated
    }

    #[getter]
    fn should_validate(&self) -> bool {
        self.inner.should_validate
    }

    #[getter]
    fn should_write_cache(&self) -> bool {
        self.inner.should_write_cache
    }
}

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
pub(crate) fn default_trusted_bgen_validation_cache_directory_value() -> PyResult<String> {
    native_trusted_validation::default_trusted_bgen_validation_cache_directory()
        .map(|cache_directory| cache_directory.display().to_string())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_trusted_bgen_validation_cache_lookup(
    py: Python<'_>,
    validation_mode: String,
    cache_path: String,
) -> PyResult<NativeTrustedBgenValidationCacheLookupPlan> {
    let plan = py
        .detach(|| {
            native_trusted_validation::plan_trusted_bgen_validation_cache_lookup(
                &validation_mode,
                Path::new(&cache_path),
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(NativeTrustedBgenValidationCacheLookupPlan { inner: plan })
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

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn write_trusted_bgen_validation_cache_payload(
    py: Python<'_>,
    cache_path: String,
    fingerprint: String,
    bgen_path: String,
    sample_count: i64,
    variant_count: i64,
) -> PyResult<()> {
    py.detach(|| {
        native_trusted_validation::write_trusted_bgen_validation_cache_payload(
            Path::new(&cache_path),
            fingerprint,
            Path::new(&bgen_path),
            sample_count,
            variant_count,
        )
    })
    .map_err(PyOSError::new_err)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTrustedBgenValidationCacheLookupPlan>()?;
    module.add_function(wrap_pyfunction!(build_trusted_bgen_validation_cache_path_value, module)?)?;
    module.add_function(wrap_pyfunction!(build_trusted_bgen_validation_cache_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_trusted_bgen_validation_fingerprint_value, module)?)?;
    module.add_function(wrap_pyfunction!(default_trusted_bgen_validation_cache_directory_value, module)?)?;
    module.add_function(wrap_pyfunction!(plan_trusted_bgen_validation_cache_lookup, module)?)?;
    module.add_function(wrap_pyfunction!(write_trusted_bgen_validation_cache_payload, module)?)?;
    Ok(())
}
