use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyMapping, PyModule, PyString, PyTuple};
use toml::{Table, Value};

pub(super) fn toml_table_from_py_mapping(raw_options: &Bound<'_, PyAny>) -> PyResult<Table> {
    let mapping = raw_options.cast::<PyMapping>()?;
    let items = mapping.call_method0("items")?;
    let mut option_table = Table::new();
    for item in items.try_iter()? {
        let item = item?;
        let tuple = item.cast::<PyTuple>()?;
        let key = tuple.get_item(0)?.extract::<String>()?;
        if let Some(value) = toml_value_from_py_any(&tuple.get_item(1)?)? {
            option_table.insert(key, value);
        }
    }
    Ok(option_table)
}

fn toml_value_from_py_any(value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Some(Value::Boolean(value.extract::<bool>()?)));
    }
    if value.cast::<PyMapping>().is_ok() {
        return toml_table_from_py_mapping(value).map(Value::Table).map(Some);
    }
    if let Ok(list) = value.cast::<PyList>() {
        return toml_array_from_py_iter(list.as_any()).map(Value::Array).map(Some);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return toml_array_from_py_iter(tuple.as_any()).map(Value::Array).map(Some);
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Some(Value::Integer(value.extract::<i64>()?)));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Some(Value::Float(value.extract::<f64>()?)));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Some(Value::String(value.extract::<String>()?)));
    }
    if let Ok(enum_value) = value.getattr("value")
        && let Ok(enum_text) = enum_value.extract::<String>()
    {
        return Ok(Some(Value::String(enum_text)));
    }
    Ok(Some(Value::String(py_string(value)?)))
}

fn toml_array_from_py_iter(value: &Bound<'_, PyAny>) -> PyResult<Vec<Value>> {
    let mut values = Vec::new();
    for item in value.try_iter()? {
        if let Some(toml_value) = toml_value_from_py_any(&item?)? {
            values.push(toml_value);
        }
    }
    Ok(values)
}

fn py_string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(value.str()?.to_string_lossy().into_owned())
}

pub(super) fn path_to_string(path: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(path_text) = path.extract::<String>() {
        return Ok(path_text);
    }
    py_string(path)
}

pub(super) fn optional_path(py: Python<'_>, value: Option<&String>) -> PyResult<Py<PyAny>> {
    match value {
        Some(path_text) => path_value(py, path_text),
        None => Ok(py.None()),
    }
}

fn path_value(py: Python<'_>, value: &str) -> PyResult<Py<PyAny>> {
    let pathlib = PyModule::import(py, "pathlib")?;
    pathlib.getattr("Path")?.call1((value,)).map(Bound::unbind)
}

pub(super) fn enum_value(py: Python<'_>, enum_name: &str, value: &str) -> PyResult<Py<PyAny>> {
    let types_module = PyModule::import(py, "g.types")?;
    types_module.getattr(enum_name)?.call1((value,)).map(Bound::unbind)
}

pub(super) fn optional_enum_value(py: Python<'_>, enum_name: &str, value: Option<&str>) -> PyResult<Py<PyAny>> {
    match value {
        Some(value) => enum_value(py, enum_name, value),
        None => Ok(py.None()),
    }
}

pub(super) fn string_tuple(py: Python<'_>, values: &[String]) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, values)?.into_any().unbind())
}
