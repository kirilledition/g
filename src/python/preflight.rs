//! PyO3 adapters for engine-owned preflight helpers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

use g_engine::preflight as native_preflight;

#[pyfunction]
pub(crate) fn resolve_preflight_variant_count(variant_count: i64, variant_limit: Option<i64>) -> PyResult<i64> {
    native_preflight::resolve_preflight_variant_count(variant_count, variant_limit)
        .map_err(|error| preflight_error_to_py(&error))
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

#[pyfunction]
pub(crate) fn validate_single_trait_preflight_shape_payload<'py>(
    py: Python<'py>,
    phenotype_sample_count: i64,
    covariate_dimension_count: i64,
    covariate_sample_count: i64,
    covariate_count: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = native_preflight::validate_single_trait_preflight_shape_payload(
        phenotype_sample_count,
        covariate_dimension_count,
        covariate_sample_count,
        covariate_count,
    )
    .map_err(|error| preflight_error_to_py(&error))?;
    let payload_dict = PyDict::new(py);
    payload_dict.set_item("sample_count", payload.sample_count)?;
    payload_dict.set_item("covariate_count", payload.covariate_count)?;
    Ok(payload_dict)
}

#[pyfunction]
pub(crate) fn validate_multi_trait_preflight_shape_payload<'py>(
    py: Python<'py>,
    phenotype_dimension_count: i64,
    phenotype_trait_count: i64,
    phenotype_sample_count: i64,
    covariate_dimension_count: i64,
    covariate_sample_count: i64,
    covariate_count: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = native_preflight::validate_multi_trait_preflight_shape_payload(
        phenotype_dimension_count,
        phenotype_trait_count,
        phenotype_sample_count,
        covariate_dimension_count,
        covariate_sample_count,
        covariate_count,
    )
    .map_err(|error| preflight_error_to_py(&error))?;
    let payload_dict = PyDict::new(py);
    payload_dict.set_item("trait_count", payload.trait_count)?;
    payload_dict.set_item("sample_count", payload.sample_count)?;
    payload_dict.set_item("covariate_count", payload.covariate_count)?;
    Ok(payload_dict)
}

#[pyfunction]
pub(crate) fn validate_binary_phenotype_case_control_counts(case_count: i64, control_count: i64) -> PyResult<()> {
    native_preflight::validate_binary_phenotype_case_control_counts(case_count, control_count)
        .map_err(|error| preflight_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn validate_finite_array(label: &str, all_values_finite: bool) -> PyResult<()> {
    native_preflight::validate_finite_array(label, all_values_finite).map_err(|error| preflight_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn validate_covariate_matrix_rank(covariate_rank: i64, covariate_count: i64) -> PyResult<()> {
    native_preflight::validate_covariate_matrix_rank(covariate_rank, covariate_count)
        .map_err(|error| preflight_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn validate_binary_phenotype_coding(is_binary_coded: bool) -> PyResult<()> {
    native_preflight::validate_binary_phenotype_coding(is_binary_coded).map_err(|error| preflight_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn validate_single_prediction_preflight_shape(
    chromosome: &str,
    prediction_shape: Vec<i64>,
    sample_count: i64,
) -> PyResult<()> {
    let prediction_shape = prediction_shape.into_boxed_slice();
    native_preflight::validate_single_prediction_preflight_shape(chromosome, &prediction_shape, sample_count)
        .map_err(|error| preflight_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn validate_multi_prediction_preflight_shape(
    chromosome: &str,
    prediction_shape: Vec<i64>,
    trait_count: i64,
    sample_count: i64,
) -> PyResult<()> {
    let prediction_shape = prediction_shape.into_boxed_slice();
    native_preflight::validate_multi_prediction_preflight_shape(
        chromosome,
        &prediction_shape,
        trait_count,
        sample_count,
    )
    .map_err(|error| preflight_error_to_py(&error))
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(build_preflight_report_payload, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_preflight_variant_count, module)?)?;
    module.add_function(wrap_pyfunction!(validate_binary_phenotype_case_control_counts, module)?)?;
    module.add_function(wrap_pyfunction!(validate_binary_phenotype_coding, module)?)?;
    module.add_function(wrap_pyfunction!(validate_covariate_matrix_rank, module)?)?;
    module.add_function(wrap_pyfunction!(validate_finite_array, module)?)?;
    module.add_function(wrap_pyfunction!(validate_multi_prediction_preflight_shape, module)?)?;
    module.add_function(wrap_pyfunction!(validate_multi_trait_preflight_shape_payload, module)?)?;
    module.add_function(wrap_pyfunction!(validate_single_prediction_preflight_shape, module)?)?;
    module.add_function(wrap_pyfunction!(validate_single_trait_preflight_shape_payload, module)?)?;
    Ok(())
}

fn preflight_error_to_py(error: &native_preflight::PreflightError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
