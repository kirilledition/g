use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyMapping, PyModule, PyString, PyTuple};
use toml::{Table, Value};

use g_interface as interface;
use g_interface::ConfigOptionValueKind;

const NATIVE_CONFIG_SECTION_NAMES: &[&str] =
    &["input", "trait", "binary", "compute", "output", "diagnostics", "metadata"];

pub(super) fn toml_table_from_py_mapping(raw_options: &Bound<'_, PyAny>) -> PyResult<Table> {
    let mapping = raw_options.cast::<PyMapping>()?;
    let items = mapping.call_method0("items")?;
    let mut option_table = Table::new();
    for item in items.try_iter()? {
        let item = item?;
        let tuple = item.cast::<PyTuple>()?;
        let key = tuple.get_item(0)?.extract::<String>()?;
        let value = toml_value_from_py_any(&tuple.get_item(1)?)?;
        option_table.insert(key, value);
    }
    Ok(option_table)
}

pub(super) fn normalized_toml_table_from_py_options(raw_options: &Bound<'_, PyAny>) -> PyResult<Table> {
    let mapping = raw_options.cast::<PyMapping>()?;
    let items = mapping.call_method0("items")?;
    let mut option_table = Table::new();
    for item in items.try_iter()? {
        let item = item?;
        let tuple = item.cast::<PyTuple>()?;
        let option_name = tuple.get_item(0)?.extract::<String>()?;
        let option_value = tuple.get_item(1)?;
        normalize_python_option(&mut option_table, &option_name, &option_value)?;
    }
    Ok(option_table)
}

fn normalize_python_option(
    option_table: &mut Table,
    option_name: &str,
    option_value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let Some(option_metadata) = metadata_for_flat_python_name(option_name) else {
        normalize_native_or_unknown_option(option_table, option_name, option_value)?;
        return Ok(());
    };
    if option_value.is_none() {
        return Err(PyValueError::new_err(format!(
            "Option {option_name} does not accept None; omit the key to leave it unset."
        )));
    }
    let normalized_value = normalize_python_option_value(option_name, option_metadata.value_kind, option_value)?;
    let section_value =
        option_table.entry(option_metadata.section.to_string()).or_insert_with(|| Value::Table(Table::new()));
    let Value::Table(section_table) = section_value else {
        option_table.insert(option_name.to_string(), normalized_value);
        return Ok(());
    };
    section_table.insert(option_metadata.toml_name.to_string(), normalized_value);
    Ok(())
}

fn normalize_native_or_unknown_option(
    option_table: &mut Table,
    option_name: &str,
    option_value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if NATIVE_CONFIG_SECTION_NAMES.contains(&option_name) {
        let normalized_value = toml_value_from_py_any(option_value)?;
        if let Value::Table(section_updates) = normalized_value {
            match option_table.get_mut(option_name) {
                Some(Value::Table(section_table)) => {
                    section_table.extend(section_updates);
                }
                Some(section_value) => {
                    *section_value = Value::Table(section_updates);
                }
                None => {
                    option_table.insert(option_name.to_string(), Value::Table(section_updates));
                }
            }
        } else {
            option_table.insert(option_name.to_string(), normalized_value);
        }
        return Ok(());
    }
    if option_value.cast::<PyMapping>().is_ok() {
        return Err(PyValueError::new_err(format!(
            "Unknown g regenie option: {}",
            flatten_unknown_option_name(option_name, option_value)?
        )));
    }
    Err(PyValueError::new_err(format!("Unknown g regenie option: {option_name}")))
}

fn metadata_for_flat_python_name(option_name: &str) -> Option<&'static interface::ConfigOptionMetadata> {
    interface::config_option_metadata().iter().find(|metadata| metadata.flat_python_names.contains(&option_name))
}

fn normalize_python_option_value(
    _option_name: &str,
    value_kind: ConfigOptionValueKind,
    option_value: &Bound<'_, PyAny>,
) -> PyResult<Value> {
    if value_kind != ConfigOptionValueKind::Boolean {
        return toml_value_from_py_any(option_value);
    }
    if option_value.is_instance_of::<PyBool>() {
        return Ok(Value::Boolean(option_value.extract::<bool>()?));
    }
    if option_value.is_instance_of::<PyString>() {
        let normalized_value = option_value.extract::<String>()?.trim().to_lowercase();
        if matches!(normalized_value.as_str(), "1" | "true" | "yes" | "on") {
            return Ok(Value::Boolean(true));
        }
        if matches!(normalized_value.as_str(), "0" | "false" | "no" | "off") {
            return Ok(Value::Boolean(false));
        }
    }
    Err(PyValueError::new_err("Boolean option value must be a bool or one of true/false/on/off/yes/no/1/0."))
}

fn flatten_unknown_option_name(option_name: &str, option_value: &Bound<'_, PyAny>) -> PyResult<String> {
    let mapping = option_value.cast::<PyMapping>()?;
    if mapping.len()? == 0 {
        return Ok(option_name.to_string());
    }
    let items = mapping.call_method0("items")?;
    let Some(item) = items.try_iter()?.next() else {
        return Ok(option_name.to_string());
    };
    let item = item?;
    let tuple = item.cast::<PyTuple>()?;
    let nested_key = tuple.get_item(0)?;
    let nested_key_text = nested_key.str()?.to_string_lossy().into_owned();
    let nested_value = tuple.get_item(1)?;
    if nested_value.cast::<PyMapping>().is_ok() {
        return Ok(format!("{option_name}.{}", flatten_unknown_option_name(&nested_key_text, &nested_value)?));
    }
    Ok(format!("{option_name}.{nested_key_text}"))
}

fn toml_value_from_py_any(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Err(PyValueError::new_err("Python option values do not accept None; omit the key to leave it unset."));
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Value::Boolean(value.extract::<bool>()?));
    }
    if value.cast::<PyMapping>().is_ok() {
        return toml_table_from_py_mapping(value).map(Value::Table);
    }
    if let Ok(list) = value.cast::<PyList>() {
        return toml_array_from_py_iter(list.as_any()).map(Value::Array);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return toml_array_from_py_iter(tuple.as_any()).map(Value::Array);
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Value::Integer(value.extract::<i64>()?));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Value::Float(value.extract::<f64>()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Value::String(value.extract::<String>()?));
    }
    if let Ok(enum_value) = value.getattr("value")
        && let Ok(enum_text) = enum_value.extract::<String>()
    {
        return Ok(Value::String(enum_text));
    }
    if let Some(path_text) = py_path_string(value)? {
        return Ok(Value::String(path_text));
    }
    let type_name = value.get_type().name()?.to_string_lossy().into_owned();
    Err(PyTypeError::new_err(format!(
        "Unsupported Python option value type '{type_name}'. Accepted values are None-free bool, int, float, str, pathlib/os.PathLike, enum values with string .value, mappings, lists, and tuples."
    )))
}

fn toml_array_from_py_iter(value: &Bound<'_, PyAny>) -> PyResult<Vec<Value>> {
    let mut values = Vec::new();
    for item in value.try_iter()? {
        values.push(toml_value_from_py_any(&item?)?);
    }
    Ok(values)
}

fn py_path_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let Ok(file_system_path_method) = value.getattr("__fspath__") else {
        return Ok(None);
    };
    let path_value = file_system_path_method.call0()?;
    if let Ok(path_text) = path_value.extract::<String>() {
        return Ok(Some(path_text));
    }
    let type_name = path_value.get_type().name()?.to_string_lossy().into_owned();
    Err(PyTypeError::new_err(format!("__fspath__ returned unsupported type '{type_name}'; expected str.")))
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
