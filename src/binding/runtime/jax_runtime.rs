//! JAX process configuration performed before backend construction.

use std::path::Path;

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use g_runtime as native_jax_runtime;

use super::{logging, telemetry_session};

pub(crate) fn configure_jax_runtime_before_backend_init(
    py: Python<'_>,
    setup_session: &mut native_jax_runtime::JaxRuntimeSetupSession,
    telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
) -> PyResult<()> {
    if !setup_session.should_configure() {
        return Ok(());
    }

    setup_session.create_cache_directory_if_configured().map_err(PyOSError::new_err)?;
    apply_jax_config_updates(py, &setup_session.config_updates())?;

    let validation_result = validate_gpu_with_default_probe_paths(py, setup_session);
    emit_native_jax_runtime_diagnostics(setup_session, telemetry_session)?;
    validation_result
}

fn apply_jax_config_updates(
    py: Python<'_>,
    updates: &[native_jax_runtime::JaxRuntimeConfigUpdatePayload],
) -> PyResult<()> {
    let update_function = py.import("jax")?.getattr("config")?.getattr("update")?;
    for update in updates {
        match &update.value {
            native_jax_runtime::JaxRuntimeConfigValue::Boolean(value) => {
                update_function.call1((update.setting_name.as_str(), value))?;
            }
            native_jax_runtime::JaxRuntimeConfigValue::Integer(value) => {
                update_function.call1((update.setting_name.as_str(), value))?;
            }
            native_jax_runtime::JaxRuntimeConfigValue::Text(value) => {
                update_function.call1((update.setting_name.as_str(), value.as_str()))?;
            }
        }
    }
    Ok(())
}

fn validate_gpu_with_default_probe_paths(
    py: Python<'_>,
    setup_session: &mut native_jax_runtime::JaxRuntimeSetupSession,
) -> PyResult<()> {
    if !setup_session.side_effect_plan().should_validate_gpu {
        return Ok(());
    }
    let probe_paths = native_jax_runtime::default_nvidia_driver_probe_paths();
    let nvidia_driver_visible = native_jax_runtime::nvidia_driver_files_are_visible(
        Path::new(&probe_paths.control_device_path),
        Path::new(&probe_paths.uvm_device_path),
        Path::new(&probe_paths.driver_directory_path),
    );
    if !nvidia_driver_visible {
        return complete_gpu_validation_or_raise(setup_session, false, false, &[]);
    }
    let devices = match observe_jax_devices(py) {
        Ok(devices) => devices,
        Err(_error) => return complete_gpu_validation_or_raise(setup_session, true, true, &[]),
    };
    complete_gpu_validation_or_raise(setup_session, true, false, &devices)
}

fn complete_gpu_validation_or_raise(
    setup_session: &mut native_jax_runtime::JaxRuntimeSetupSession,
    nvidia_driver_visible: bool,
    backend_initialization_failed: bool,
    devices: &[native_jax_runtime::JaxDeviceObservation],
) -> PyResult<()> {
    let validation_plan =
        native_jax_runtime::plan_jax_gpu_validation(nvidia_driver_visible, backend_initialization_failed, devices);
    let _ = setup_session.complete_validation(&validation_plan.status, Some(validation_plan.message.as_str()));
    if validation_plan.should_raise {
        return Err(PyRuntimeError::new_err(validation_plan.message));
    }
    Ok(())
}

fn emit_native_jax_runtime_diagnostics(
    setup_session: &native_jax_runtime::JaxRuntimeSetupSession,
    telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
) -> PyResult<()> {
    for event in setup_session.diagnostic_events() {
        let record_plan =
            native_jax_runtime::plan_jax_runtime_diagnostic_record(&event.level, telemetry_session.is_some());
        let fields_json =
            native_jax_runtime::serialize_jax_runtime_diagnostic_fields_json(&event.fields).map_err(|error| {
                PyValueError::new_err(format!("Failed to serialize JAX runtime diagnostic event fields: {error}"))
            })?;
        logging::emit_diagnostic_event(
            &record_plan.logging_level_name.to_lowercase(),
            &event.event_name,
            &event.message,
            Some(fields_json),
        )?;
        if let Some(telemetry_session) = telemetry_session {
            let fields = native_jax_runtime::JaxRuntimeDiagnosticFields::new(&event.fields);
            telemetry_session.emit_current_event(&event.event_name, &record_plan.telemetry_level, &fields)?;
        }
    }
    Ok(())
}

fn observe_jax_devices(py: Python<'_>) -> PyResult<Vec<native_jax_runtime::JaxDeviceObservation>> {
    let devices = py.import("jax")?.call_method0("devices")?;
    let mut device_observations = Vec::new();
    for device in devices.try_iter()? {
        let device = device?;
        device_observations.push(native_jax_runtime::JaxDeviceObservation {
            platform: python_attribute_to_string(&device, "platform")?,
            description: device.str()?.to_string_lossy().into_owned(),
        });
    }
    Ok(device_observations)
}

fn python_attribute_to_string(object: &Bound<'_, PyAny>, attribute_name: &str) -> PyResult<String> {
    match object.getattr(attribute_name) {
        Ok(value) => Ok(value.str()?.to_string_lossy().into_owned()),
        Err(_error) => Ok(String::new()),
    }
}
