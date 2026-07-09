use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn py_i64_to_usize(value: i64, field_name: &str) -> PyResult<usize> {
    if value < 0 {
        return Err(PyValueError::new_err(format!("{field_name} must be non-negative. Observed {value}.")));
    }
    usize::try_from(value).map_err(|_| PyValueError::new_err(format!("{field_name} does not fit into native usize.")))
}

pub(crate) fn optional_py_i64_to_usize(value: Option<i64>, field_name: &str) -> PyResult<Option<usize>> {
    value.map(|inner_value| py_i64_to_usize(inner_value, field_name)).transpose()
}

pub(crate) fn py_i64_slice_to_usize(values: &[i64], field_name: &str) -> PyResult<Vec<usize>> {
    values.iter().map(|value| py_i64_to_usize(*value, field_name)).collect()
}

pub(crate) fn usize_to_py_i64(value: usize, field_name: &str) -> PyResult<i64> {
    i64::try_from(value).map_err(|_| PyValueError::new_err(format!("{field_name} does not fit into native int64.")))
}

pub(crate) fn usize_slice_to_py_i64(values: &[usize], field_name: &str) -> PyResult<Vec<i64>> {
    values.iter().map(|value| usize_to_py_i64(*value, field_name)).collect()
}

pub(crate) fn optional_usize_to_py_i64(value: Option<usize>, field_name: &str) -> PyResult<Option<i64>> {
    value.map(|inner_value| usize_to_py_i64(inner_value, field_name)).transpose()
}
