//! JSON value conversion helpers for PyO3 boundary adapters.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyString, PyTuple};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

pub(crate) fn json_value_from_py_any(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if value.is_none() {
        return Ok(JsonValue::Null);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(JsonValue::Bool(value.extract::<bool>()?));
    }
    if let Ok(mapping) = value.cast::<PyMapping>() {
        return json_object_from_py_mapping(mapping).map(JsonValue::Object);
    }
    if let Ok(list) = value.cast::<PyList>() {
        return json_array_from_py_iter(list.as_any()).map(JsonValue::Array);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return json_array_from_py_iter(tuple.as_any()).map(JsonValue::Array);
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(signed_integer) = value.extract::<i64>() {
            return Ok(JsonValue::Number(JsonNumber::from(signed_integer)));
        }
        if let Ok(unsigned_integer) = value.extract::<u64>() {
            return Ok(JsonValue::Number(JsonNumber::from(unsigned_integer)));
        }
        return Ok(JsonValue::String(value.str()?.to_string_lossy().into_owned()));
    }
    if value.is_instance_of::<PyFloat>() {
        let float_value = value.extract::<f64>()?;
        let Some(number) = JsonNumber::from_f64(float_value) else {
            return Ok(JsonValue::String(value.str()?.to_string_lossy().into_owned()));
        };
        return Ok(JsonValue::Number(number));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(JsonValue::String(value.extract::<String>()?));
    }
    if let Some(path_text) = path_string(value)? {
        return Ok(JsonValue::String(path_text));
    }
    Ok(JsonValue::String(value.str()?.to_string_lossy().into_owned()))
}

pub(crate) fn json_text_to_py_object(py: Python<'_>, json_text: &str, payload_name: &str) -> PyResult<Py<PyAny>> {
    let json_value = serde_json::from_str::<JsonValue>(json_text).map_err(|error| {
        PyValueError::new_err(format!("Native {payload_name} payload must contain valid JSON: {error}"))
    })?;
    json_value_to_py_object(py, &json_value)
}

pub(crate) fn json_value_to_py_object(py: Python<'_>, value: &JsonValue) -> PyResult<Py<PyAny>> {
    match value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(boolean_value) => Ok(PyBool::new(py, *boolean_value).to_owned().into_any().unbind()),
        JsonValue::Number(number) => {
            if let Some(signed_integer) = number.as_i64() {
                return Ok(signed_integer.into_pyobject(py)?.into_any().unbind());
            }
            if let Some(unsigned_integer) = number.as_u64() {
                return Ok(unsigned_integer.into_pyobject(py)?.into_any().unbind());
            }
            let float_value = number.as_f64().ok_or_else(|| {
                PyValueError::new_err("Native JSON number could not be represented as a Python value.")
            })?;
            Ok(float_value.into_pyobject(py)?.into_any().unbind())
        }
        JsonValue::String(text) => Ok(text.into_pyobject(py)?.into_any().unbind()),
        JsonValue::Array(values) => {
            let python_values = values
                .iter()
                .map(|item_value| json_value_to_py_object(py, item_value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, python_values)?.into_any().unbind())
        }
        JsonValue::Object(object) => {
            let payload = PyDict::new(py);
            for (key, item_value) in object {
                payload.set_item(key, json_value_to_py_object(py, item_value)?)?;
            }
            Ok(payload.into_any().unbind())
        }
    }
}

fn json_object_from_py_mapping(mapping: &Bound<'_, PyMapping>) -> PyResult<JsonMap<String, JsonValue>> {
    let items = mapping.call_method0("items")?;
    let mut json_object = JsonMap::new();
    for item in items.try_iter()? {
        let item = item?;
        let tuple = item.cast::<PyTuple>()?;
        let key = tuple.get_item(0)?;
        let value = tuple.get_item(1)?;
        json_object.insert(json_key_from_py_any(&key)?, json_value_from_py_any(&value)?);
    }
    Ok(json_object)
}

fn json_array_from_py_iter(value: &Bound<'_, PyAny>) -> PyResult<Vec<JsonValue>> {
    let mut values = Vec::new();
    for item in value.try_iter()? {
        values.push(json_value_from_py_any(&item?)?);
    }
    Ok(values)
}

fn json_key_from_py_any(key: &Bound<'_, PyAny>) -> PyResult<String> {
    if key.is_none() {
        return Ok("null".to_string());
    }
    if key.is_instance_of::<PyBool>() {
        return Ok(if key.extract::<bool>()? { "true" } else { "false" }.to_string());
    }
    if key.is_instance_of::<PyString>() {
        return key.extract::<String>();
    }
    Ok(key.str()?.to_string_lossy().into_owned())
}

fn path_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let Ok(file_system_path_method) = value.getattr("__fspath__") else {
        return Ok(None);
    };
    let path_value = file_system_path_method.call0()?;
    if let Ok(path_text) = path_value.extract::<String>() {
        return Ok(Some(path_text));
    }
    let type_name = path_value.get_type().name()?.to_string_lossy().into_owned();
    Err(PyValueError::new_err(format!("__fspath__ returned unsupported type '{type_name}'; expected str.")))
}
