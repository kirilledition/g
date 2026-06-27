//! PyO3 adapters for deterministic JAX runtime setup policy.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use g_runtime::jax_runtime as native_jax_runtime;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn resolve_jax_runtime_setup_payload<'py>(
    py: Python<'py>,
    requested_device: String,
    cache_directory: String,
    matmul_precision: Option<String>,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: i64,
    xla_autotune_cache: bool,
    transfer_guard: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let setup = native_jax_runtime::resolve_jax_runtime_setup(
        &requested_device,
        &cache_directory,
        matmul_precision.as_deref(),
        persistent_cache,
        persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds,
        xla_autotune_cache,
        transfer_guard,
    );
    jax_runtime_setup_payload_to_dict(py, &setup)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_jax_runtime_setup_diagnostic_payloads<'py>(
    py: Python<'py>,
    requested_device: String,
    platform_name: String,
    cache_directory: String,
    matmul_precision: String,
    persistent_cache_enabled: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: i64,
    xla_auxiliary_cache_mode: String,
    xla_auxiliary_cache_reason: String,
    transfer_guard_enabled: bool,
    gpu_validation_status: String,
    gpu_validation_message: Option<String>,
) -> PyResult<Bound<'py, PyTuple>> {
    let setup = native_jax_runtime::JaxRuntimeSetupPayload {
        requested_device,
        platform_name,
        cache_directory,
        matmul_precision,
        persistent_cache_enabled,
        persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds,
        xla_auxiliary_cache_mode,
        xla_auxiliary_cache_reason,
        transfer_guard_enabled,
        gpu_validation_status,
        gpu_validation_message,
    };
    let events = native_jax_runtime::build_jax_runtime_setup_diagnostic_events(&setup);
    let event_payloads = events
        .iter()
        .map(|event| jax_runtime_diagnostic_event_payload_to_dict(py, event))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &event_payloads)
}

fn jax_runtime_setup_payload_to_dict<'py>(
    py: Python<'py>,
    setup: &native_jax_runtime::JaxRuntimeSetupPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("requested_device", &setup.requested_device)?;
    payload.set_item("platform_name", &setup.platform_name)?;
    payload.set_item("cache_directory", &setup.cache_directory)?;
    payload.set_item("matmul_precision", &setup.matmul_precision)?;
    payload.set_item("persistent_cache_enabled", setup.persistent_cache_enabled)?;
    payload.set_item("persistent_cache_min_entry_size_bytes", setup.persistent_cache_min_entry_size_bytes)?;
    payload.set_item("persistent_cache_min_compile_time_seconds", setup.persistent_cache_min_compile_time_seconds)?;
    payload.set_item("xla_auxiliary_cache_mode", &setup.xla_auxiliary_cache_mode)?;
    payload.set_item("xla_auxiliary_cache_reason", &setup.xla_auxiliary_cache_reason)?;
    payload.set_item("transfer_guard_enabled", setup.transfer_guard_enabled)?;
    payload.set_item("gpu_validation_status", &setup.gpu_validation_status)?;
    set_optional_string(py, &payload, "gpu_validation_message", setup.gpu_validation_message.as_deref())?;
    Ok(payload)
}

fn jax_runtime_diagnostic_event_payload_to_dict<'py>(
    py: Python<'py>,
    event: &native_jax_runtime::JaxRuntimeDiagnosticEventPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("event_name", &event.event_name)?;
    payload.set_item("level", &event.level)?;
    payload.set_item("message", &event.message)?;
    let field_payloads = event
        .fields
        .iter()
        .map(|field| jax_runtime_diagnostic_field_payload_to_dict(py, field))
        .collect::<PyResult<Vec<_>>>()?;
    payload.set_item("fields", PyTuple::new(py, &field_payloads)?)?;
    Ok(payload)
}

fn jax_runtime_diagnostic_field_payload_to_dict<'py>(
    py: Python<'py>,
    field: &native_jax_runtime::JaxRuntimeDiagnosticFieldPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("name", &field.name)?;
    match &field.value {
        native_jax_runtime::JaxRuntimeDiagnosticValue::Boolean(value) => payload.set_item("value", *value)?,
        native_jax_runtime::JaxRuntimeDiagnosticValue::Integer(value) => payload.set_item("value", *value)?,
        native_jax_runtime::JaxRuntimeDiagnosticValue::Text(value) => payload.set_item("value", value)?,
    }
    Ok(payload)
}

fn set_optional_string(py: Python<'_>, payload: &Bound<'_, PyDict>, key: &str, value: Option<&str>) -> PyResult<()> {
    match value {
        Some(text) => payload.set_item(key, text),
        None => payload.set_item(key, py.None()),
    }
}
