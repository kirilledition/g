use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyModule, PyString, PyTuple};

use crate::config_frontend::{OptionTable, OptionValue};

pub(super) fn option_table_from_py_mapping(raw_options: &Bound<'_, PyAny>) -> PyResult<OptionTable> {
    let mapping = raw_options.cast::<PyMapping>()?;
    let items = mapping.call_method0("items")?;
    let mut option_table = BTreeMap::new();
    for item in items.try_iter()? {
        let item = item?;
        let tuple = item.cast::<PyTuple>()?;
        let key = tuple.get_item(0)?.extract::<String>()?;
        let value = option_value_from_py_any(&tuple.get_item(1)?)?;
        option_table.insert(key, value);
    }
    Ok(option_table)
}

fn option_value_from_py_any(value: &Bound<'_, PyAny>) -> PyResult<OptionValue> {
    if value.is_none() {
        return Ok(OptionValue::None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(OptionValue::Boolean(value.extract::<bool>()?));
    }
    if value.cast::<PyDict>().is_ok() {
        return option_table_from_py_mapping(value).map(OptionValue::Table);
    }
    if let Ok(list) = value.cast::<PyList>() {
        return string_list_from_py_iter(list.as_any()).map(OptionValue::List);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return string_list_from_py_iter(tuple.as_any()).map(OptionValue::List);
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(OptionValue::Integer(value.extract::<i64>()?));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(OptionValue::Float(value.extract::<f64>()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(OptionValue::String(value.extract::<String>()?));
    }
    if let Ok(enum_value) = value.getattr("value")
        && let Ok(enum_text) = enum_value.extract::<String>()
    {
        return Ok(OptionValue::String(enum_text));
    }
    Ok(OptionValue::String(py_string(value)?))
}

fn string_list_from_py_iter(value: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    value.try_iter()?.map(|item| item.and_then(|item_value| py_string(&item_value))).collect()
}

fn py_string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(value.str()?.to_string_lossy().into_owned())
}

pub(super) fn text_from_py_bytes_or_string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = value.extract::<String>() {
        return Ok(text);
    }
    if let Ok(bytes) = value.extract::<Vec<u8>>() {
        return String::from_utf8(bytes)
            .map_err(|error| PyValueError::new_err(format!("TOML config bytes must be UTF-8: {error}")));
    }
    py_string(value)
}

pub(super) fn option_table_to_py_dict<'py>(py: Python<'py>, option_table: &OptionTable) -> PyResult<Py<PyAny>> {
    let dictionary = PyDict::new(py);
    for (key, value) in option_table {
        dictionary.set_item(key, option_value_to_py_object(py, value)?)?;
    }
    Ok(dictionary.into_any().unbind())
}

fn option_value_to_py_object<'py>(py: Python<'py>, option_value: &OptionValue) -> PyResult<Py<PyAny>> {
    match option_value {
        OptionValue::None => Ok(py.None()),
        OptionValue::String(value) => Ok(PyString::new(py, value).into_any().unbind()),
        OptionValue::Integer(value) => Ok(value.into_pyobject(py)?.unbind().into_any()),
        OptionValue::Float(value) => Ok(value.into_pyobject(py)?.unbind().into_any()),
        OptionValue::Boolean(value) => Ok(PyBool::new(py, *value).to_owned().unbind().into_any()),
        OptionValue::List(values) => Ok(PyList::new(py, values)?.into_any().unbind()),
        OptionValue::Table(table) => option_table_to_py_dict(py, table),
    }
}

pub(super) fn path_to_string(path: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(path_text) = path.extract::<String>() {
        return Ok(path_text);
    }
    py_string(path)
}

pub(super) fn optional_path<'py>(py: Python<'py>, value: &Option<String>) -> PyResult<Py<PyAny>> {
    match value {
        Some(path_text) => path_value(py, path_text),
        None => Ok(py.None()),
    }
}

fn path_value<'py>(py: Python<'py>, value: &str) -> PyResult<Py<PyAny>> {
    let pathlib = PyModule::import(py, "pathlib")?;
    pathlib.getattr("Path")?.call1((value,)).map(Bound::unbind)
}

pub(super) fn enum_value<'py>(py: Python<'py>, enum_name: &str, value: &str) -> PyResult<Py<PyAny>> {
    let types_module = PyModule::import(py, "g.types")?;
    types_module.getattr(enum_name)?.call1((value,)).map(Bound::unbind)
}

pub(super) fn optional_enum_value<'py>(
    py: Python<'py>,
    enum_name: &str,
    value: &Option<String>,
) -> PyResult<Py<PyAny>> {
    match value {
        Some(value) => enum_value(py, enum_name, value),
        None => Ok(py.None()),
    }
}

pub(super) fn string_tuple<'py>(py: Python<'py>, values: &[String]) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, values)?.into_any().unbind())
}
