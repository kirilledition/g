//! PyO3 adapters for engine-owned preflight helpers.

use nalgebra::DMatrix;
use numpy::ndarray::IxDyn;
use numpy::{Element, PyArray, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods, dtype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_engine as native_preflight;

use super::errors::convert_preflight_error;

trait NativePreflightNumeric: Copy + Element {
    fn is_finite_value(self) -> bool;

    fn is_binary_zero(self) -> bool;

    fn is_binary_one(self) -> bool;

    fn rank_tolerance_epsilon() -> f64;

    fn to_rank_f64(self) -> f64;
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

                fn rank_tolerance_epsilon() -> f64 {
                    f64::from(<$numeric_type>::EPSILON)
                }

                fn to_rank_f64(self) -> f64 {
                    f64::from(self)
                }
            }
        )*
    };
}

macro_rules! impl_lossless_integer_preflight_numeric {
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

                fn rank_tolerance_epsilon() -> f64 {
                    f64::EPSILON
                }

                fn to_rank_f64(self) -> f64 {
                    f64::from(self)
                }
            }
        )*
    };
}

macro_rules! impl_lossy_integer_preflight_numeric {
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

                fn rank_tolerance_epsilon() -> f64 {
                    f64::EPSILON
                }

                #[allow(clippy::cast_precision_loss)]
                fn to_rank_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_float_preflight_numeric!(f32, f64);
impl_lossless_integer_preflight_numeric!(i8, i16, i32, u8, u16, u32);
impl_lossy_integer_preflight_numeric!(i64, u64);

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

    fn rank_tolerance_epsilon() -> f64 {
        f64::EPSILON
    }

    fn to_rank_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
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

#[pyclass]
pub(crate) struct NativePreflightValidator;

#[pymethods]
impl NativePreflightValidator {
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    fn resolve_preflight_variant_count(&self, variant_count: i64, variant_limit: Option<i64>) -> PyResult<i64> {
        native_preflight::resolve_preflight_variant_count(variant_count, variant_limit)
            .map_err(|error| convert_preflight_error(&error))
    }

    #[allow(clippy::unused_self)]
    fn build_preflight_report(
        &self,
        sample_count: i64,
        covariate_count: i64,
        chromosome_count: i64,
        trusted_no_missing_diploid: bool,
    ) -> PyResult<(i64, i64, i64, Vec<String>)> {
        let report = native_preflight::build_preflight_report_payload(
            sample_count,
            covariate_count,
            chromosome_count,
            trusted_no_missing_diploid,
        )
        .map_err(|error| convert_preflight_error(&error))?;
        Ok((report.sample_count, report.covariate_count, report.chromosome_count, report.warning_messages))
    }

    #[allow(clippy::unused_self)]
    fn validate_single_trait_preflight_shape(
        &self,
        phenotype_sample_count: i64,
        covariate_dimension_count: i64,
        covariate_sample_count: i64,
        covariate_count: i64,
    ) -> PyResult<(i64, i64)> {
        let shape = native_preflight::validate_single_trait_preflight_shape_payload(
            phenotype_sample_count,
            covariate_dimension_count,
            covariate_sample_count,
            covariate_count,
        )
        .map_err(|error| convert_preflight_error(&error))?;
        Ok((shape.sample_count, shape.covariate_count))
    }

    #[allow(clippy::unused_self)]
    fn validate_multi_trait_preflight_shape(
        &self,
        phenotype_dimension_count: i64,
        phenotype_trait_count: i64,
        phenotype_sample_count: i64,
        covariate_dimension_count: i64,
        covariate_sample_count: i64,
        covariate_count: i64,
    ) -> PyResult<(i64, i64, i64)> {
        let shape = native_preflight::validate_multi_trait_preflight_shape_payload(
            phenotype_dimension_count,
            phenotype_trait_count,
            phenotype_sample_count,
            covariate_dimension_count,
            covariate_sample_count,
            covariate_count,
        )
        .map_err(|error| convert_preflight_error(&error))?;
        Ok((shape.trait_count, shape.sample_count, shape.covariate_count))
    }

    #[allow(clippy::unused_self)]
    fn validate_finite_array_values(
        &self,
        py: Python<'_>,
        label: &str,
        values: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        dispatch_preflight_numeric_array!(py, values, validate_typed_finite_array, label)
    }

    #[allow(clippy::unused_self)]
    fn validate_covariate_matrix_rank(&self, covariate_rank: i64, covariate_count: i64) -> PyResult<()> {
        native_preflight::validate_covariate_matrix_rank(covariate_rank, covariate_count)
            .map_err(|error| convert_preflight_error(&error))
    }

    #[allow(clippy::unused_self)]
    fn validate_covariate_matrix_rank_array(
        &self,
        py: Python<'_>,
        covariate_matrix: &Bound<'_, PyUntypedArray>,
        covariate_count: i64,
    ) -> PyResult<()> {
        dispatch_preflight_numeric_array!(py, covariate_matrix, validate_typed_covariate_matrix_rank, covariate_count)
    }

    #[allow(clippy::unused_self)]
    fn validate_binary_phenotype_array(
        &self,
        py: Python<'_>,
        phenotype_values: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        dispatch_preflight_numeric_array!(py, phenotype_values, validate_typed_binary_phenotype)
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    fn validate_single_prediction_preflight_shape(
        &self,
        chromosome: &str,
        prediction_shape: Vec<i64>,
        sample_count: i64,
    ) -> PyResult<()> {
        let prediction_shape = prediction_shape.into_boxed_slice();
        native_preflight::validate_single_prediction_preflight_shape(chromosome, &prediction_shape, sample_count)
            .map_err(|error| convert_preflight_error(&error))
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    fn validate_multi_prediction_preflight_shape(
        &self,
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
        .map_err(|error| convert_preflight_error(&error))
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePreflightValidator>()?;
    Ok(())
}

fn validate_typed_finite_array<T>(label: &str, values: &Bound<'_, PyArray<T, IxDyn>>) -> PyResult<()>
where
    T: NativePreflightNumeric,
{
    let readonly_values = values.readonly();
    let all_values_finite = readonly_values.as_array().iter().copied().all(NativePreflightNumeric::is_finite_value);
    native_preflight::validate_finite_array(label, all_values_finite).map_err(|error| convert_preflight_error(&error))
}

fn validate_typed_binary_phenotype<T>(phenotype_values: &Bound<'_, PyArray<T, IxDyn>>) -> PyResult<()>
where
    T: NativePreflightNumeric,
{
    let summary = summarize_binary_phenotype(phenotype_values);
    native_preflight::validate_binary_phenotype_coding(summary.is_binary_coded)
        .map_err(|error| convert_preflight_error(&error))?;
    native_preflight::validate_binary_phenotype_case_control_counts(summary.case_count, summary.control_count)
        .map_err(|error| convert_preflight_error(&error))
}

fn validate_typed_covariate_matrix_rank<T>(
    covariate_count: i64,
    covariate_matrix: &Bound<'_, PyArray<T, IxDyn>>,
) -> PyResult<()>
where
    T: NativePreflightNumeric,
{
    let covariate_rank = compute_covariate_matrix_rank(covariate_matrix)?;
    native_preflight::validate_covariate_matrix_rank(covariate_rank, covariate_count)
        .map_err(|error| convert_preflight_error(&error))
}

fn compute_covariate_matrix_rank<T>(covariate_matrix: &Bound<'_, PyArray<T, IxDyn>>) -> PyResult<i64>
where
    T: NativePreflightNumeric,
{
    let readonly_matrix = covariate_matrix.readonly();
    let matrix_values = readonly_matrix.as_array();
    let matrix_shape = matrix_values.shape();
    if matrix_shape.len() != 2 {
        return Err(convert_preflight_error(&native_preflight::PreflightError::CovariateMatrixDimension));
    }

    let row_count = matrix_shape[0];
    let column_count = matrix_shape[1];
    let rank_values = matrix_values.iter().copied().map(NativePreflightNumeric::to_rank_f64).collect::<Vec<_>>();
    let rank_matrix = DMatrix::from_row_slice(row_count, column_count, &rank_values);
    let singular_values = rank_matrix.svd(false, false).singular_values;
    let largest_singular_value = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance =
        largest_singular_value * dimension_count_as_f64(row_count.max(column_count)) * T::rank_tolerance_epsilon();
    let covariate_rank = singular_values.iter().filter(|singular_value| **singular_value > tolerance).count();
    i64::try_from(covariate_rank).map_err(|_| PyValueError::new_err("Covariate matrix rank exceeds supported count."))
}

#[allow(clippy::cast_precision_loss)]
fn dimension_count_as_f64(dimension_count: usize) -> f64 {
    dimension_count as f64
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
