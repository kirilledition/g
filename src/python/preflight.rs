//! PyO3 adapters for engine-owned preflight helpers.

use numpy::ndarray::IxDyn;
use numpy::{Element, PyArray, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods, dtype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

use g_engine::preflight as native_preflight;

trait NativePreflightNumeric: Copy + Element {
    fn is_finite_value(self) -> bool;

    fn is_binary_zero(self) -> bool;

    fn is_binary_one(self) -> bool;
}

macro_rules! impl_float_preflight_numeric {
    ($($numeric_type:ty),* $(,)?) => {
        $(
            impl NativePreflightNumeric for $numeric_type {
                fn is_finite_value(self) -> bool {
                    self.is_finite()
                }

                fn is_binary_zero(self) -> bool {
                    matches!(self.classify(), std::num::FpCategory::Zero)
                }

                fn is_binary_one(self) -> bool {
                    self.to_bits() == <$numeric_type>::to_bits(1.0)
                }
            }
        )*
    };
}

macro_rules! impl_integer_preflight_numeric {
    ($($numeric_type:ty),* $(,)?) => {
        $(
            impl NativePreflightNumeric for $numeric_type {
                fn is_finite_value(self) -> bool {
                    true
                }

                fn is_binary_zero(self) -> bool {
                    self == 0
                }

                fn is_binary_one(self) -> bool {
                    self == 1
                }
            }
        )*
    };
}

impl_float_preflight_numeric!(f32, f64);
impl_integer_preflight_numeric!(i8, i16, i32, i64, u8, u16, u32, u64);

impl NativePreflightNumeric for bool {
    fn is_finite_value(self) -> bool {
        true
    }

    fn is_binary_zero(self) -> bool {
        !self
    }

    fn is_binary_one(self) -> bool {
        self
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct BinaryPhenotypeSummary {
    is_binary_coded: bool,
    case_count: i64,
    control_count: i64,
}

macro_rules! dispatch_preflight_numeric_array {
    ($py:expr, $array:expr, $function:path $(, $argument:expr)* $(,)?) => {{
        let element_type = $array.dtype();
        if element_type.is_equiv_to(&dtype::<f32>($py)) {
            let typed_array = $array.cast::<PyArray<f32, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<f64>($py)) {
            let typed_array = $array.cast::<PyArray<f64, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<i8>($py)) {
            let typed_array = $array.cast::<PyArray<i8, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<i16>($py)) {
            let typed_array = $array.cast::<PyArray<i16, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<i32>($py)) {
            let typed_array = $array.cast::<PyArray<i32, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<i64>($py)) {
            let typed_array = $array.cast::<PyArray<i64, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<u8>($py)) {
            let typed_array = $array.cast::<PyArray<u8, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<u16>($py)) {
            let typed_array = $array.cast::<PyArray<u16, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<u32>($py)) {
            let typed_array = $array.cast::<PyArray<u32, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<u64>($py)) {
            let typed_array = $array.cast::<PyArray<u64, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else if element_type.is_equiv_to(&dtype::<bool>($py)) {
            let typed_array = $array.cast::<PyArray<bool, IxDyn>>()?;
            $function($($argument,)* typed_array)
        } else {
            Err(PyValueError::new_err("Unsupported preflight numeric dtype."))
        }
    }};
}

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
pub(crate) fn validate_finite_array_values(
    py: Python<'_>,
    label: &str,
    values: &Bound<'_, PyUntypedArray>,
) -> PyResult<()> {
    dispatch_preflight_numeric_array!(py, values, validate_typed_finite_array, label)
}

#[pyfunction]
pub(crate) fn validate_covariate_matrix_rank(covariate_rank: i64, covariate_count: i64) -> PyResult<()> {
    native_preflight::validate_covariate_matrix_rank(covariate_rank, covariate_count)
        .map_err(|error| preflight_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn validate_binary_phenotype_array(
    py: Python<'_>,
    phenotype_values: &Bound<'_, PyUntypedArray>,
) -> PyResult<()> {
    dispatch_preflight_numeric_array!(py, phenotype_values, validate_typed_binary_phenotype)
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
    module.add_function(wrap_pyfunction!(validate_binary_phenotype_array, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_preflight_variant_count, module)?)?;
    module.add_function(wrap_pyfunction!(validate_covariate_matrix_rank, module)?)?;
    module.add_function(wrap_pyfunction!(validate_finite_array_values, module)?)?;
    module.add_function(wrap_pyfunction!(validate_multi_prediction_preflight_shape, module)?)?;
    module.add_function(wrap_pyfunction!(validate_multi_trait_preflight_shape_payload, module)?)?;
    module.add_function(wrap_pyfunction!(validate_single_prediction_preflight_shape, module)?)?;
    module.add_function(wrap_pyfunction!(validate_single_trait_preflight_shape_payload, module)?)?;
    Ok(())
}

fn validate_typed_finite_array<T>(label: &str, values: &Bound<'_, PyArray<T, IxDyn>>) -> PyResult<()>
where
    T: NativePreflightNumeric,
{
    let readonly_values = values.readonly();
    let all_values_finite = readonly_values.as_array().iter().copied().all(NativePreflightNumeric::is_finite_value);
    native_preflight::validate_finite_array(label, all_values_finite).map_err(|error| preflight_error_to_py(&error))
}

fn validate_typed_binary_phenotype<T>(phenotype_values: &Bound<'_, PyArray<T, IxDyn>>) -> PyResult<()>
where
    T: NativePreflightNumeric,
{
    let summary = summarize_binary_phenotype(phenotype_values);
    native_preflight::validate_binary_phenotype_coding(summary.is_binary_coded)
        .map_err(|error| preflight_error_to_py(&error))?;
    native_preflight::validate_binary_phenotype_case_control_counts(summary.case_count, summary.control_count)
        .map_err(|error| preflight_error_to_py(&error))
}

fn summarize_binary_phenotype<T>(phenotype_values: &Bound<'_, PyArray<T, IxDyn>>) -> BinaryPhenotypeSummary
where
    T: NativePreflightNumeric,
{
    let readonly_values = phenotype_values.readonly();
    let mut summary = BinaryPhenotypeSummary { is_binary_coded: true, ..BinaryPhenotypeSummary::default() };
    for value in readonly_values.as_array().iter().copied() {
        if value.is_binary_zero() {
            summary.control_count += 1;
        } else if value.is_binary_one() {
            summary.case_count += 1;
        } else {
            summary.is_binary_coded = false;
        }
    }
    summary
}

fn preflight_error_to_py(error: &native_preflight::PreflightError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
