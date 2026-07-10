//! Telemetry policy parsing for the native CLI.

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

use g_runtime as native_telemetry_policy;

pub(crate) fn parse_telemetry_mode(telemetry_mode: &str) -> PyResult<native_telemetry_policy::TelemetryMode> {
    native_telemetry_policy::TelemetryMode::from_str_value(telemetry_mode).ok_or_else(|| {
        PyValueError::new_err(format!(
            "telemetry_mode must be one of {}, observed '{telemetry_mode}'.",
            native_telemetry_policy::TelemetryMode::accepted_values().join(", "),
        ))
    })
}
