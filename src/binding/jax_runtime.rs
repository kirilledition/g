//! Direct PyO3 calls required by the Rust-owned JAX runtime sequence.

use pyo3::prelude::*;

use g_runner as native_jax_runtime;

pub(crate) fn apply_jax_config_updates(
    py: Python<'_>,
    updates: &[native_jax_runtime::JaxRuntimeConfigUpdate<'_>],
) -> PyResult<()> {
    let update_function = py.import("jax")?.getattr("config")?.getattr("update")?;
    for update in updates {
        match &update.value {
            native_jax_runtime::JaxRuntimeConfigValue::Boolean(value) => {
                update_function.call1((update.setting_name, value))?;
            }
            native_jax_runtime::JaxRuntimeConfigValue::Integer(value) => {
                update_function.call1((update.setting_name, value))?;
            }
            native_jax_runtime::JaxRuntimeConfigValue::Text(value) => {
                update_function.call1((update.setting_name, value.as_ref()))?;
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
            platform: device.getattr("platform")?.str()?.to_string_lossy().into_owned(),
            description: device.str()?.to_string_lossy().into_owned(),
        });
    }
    Ok(device_observations)
}
