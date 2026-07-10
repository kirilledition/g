//! Direct PyO3 calls required by the Rust-owned JAX runtime sequence.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use g_runner as native_jax_runtime;

pub(crate) fn apply_jax_config_updates(
    py: Python<'_>,
    updates: &[native_jax_runtime::JaxRuntimeConfigUpdate],
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

pub(crate) fn observe_jax_devices(py: Python<'_>) -> PyResult<Vec<native_jax_runtime::JaxDevice>> {
    let devices = py.import("jax")?.call_method0("devices")?;
    let mut device_observations = Vec::new();
    for device in devices.try_iter()? {
        let device = device?;
        device_observations.push(native_jax_runtime::JaxDevice {
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
