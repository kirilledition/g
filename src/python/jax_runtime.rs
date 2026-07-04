//! PyO3 adapters for deterministic JAX runtime setup policy.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyAttributeError, PyOSError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyModule};

use g_runtime::jax_runtime as native_jax_runtime;

use super::logging;

#[pyclass]
pub(crate) struct NativeJaxRuntimeDiagnosticRecordPlan {
    plan: native_jax_runtime::JaxRuntimeDiagnosticRecordPlan,
}

#[pyclass]
pub(crate) struct NativeJaxRuntimeDiagnosticPolicy;

#[pyclass]
pub(crate) struct NativeJaxRuntimeSetupReport {
    setup: native_jax_runtime::JaxRuntimeSetupPayload,
}

#[pyclass]
pub(crate) struct NativeNvidiaDriverProbePaths {
    paths: native_jax_runtime::NvidiaDriverProbePathsPayload,
}

#[pyclass]
pub(crate) struct NativeJaxRuntimeDiagnosticEvent {
    event: native_jax_runtime::JaxRuntimeDiagnosticEventPayload,
}

#[pyclass]
pub(crate) struct NativeJaxRuntimeDiagnosticField {
    field: native_jax_runtime::JaxRuntimeDiagnosticFieldPayload,
}

#[pymethods]
impl NativeJaxRuntimeDiagnosticRecordPlan {
    #[getter]
    fn logging_level_name(&self) -> &str {
        &self.plan.logging_level_name
    }

    #[getter]
    fn should_emit_telemetry(&self) -> bool {
        self.plan.should_emit_telemetry
    }

    #[getter]
    fn telemetry_level(&self) -> &str {
        &self.plan.telemetry_level
    }
}

#[pymethods]
impl NativeJaxRuntimeSetupReport {
    #[getter]
    fn requested_device(&self) -> &str {
        &self.setup.requested_device
    }

    #[getter]
    fn platform_name(&self) -> &str {
        &self.setup.platform_name
    }

    #[getter]
    fn cache_directory(&self) -> &str {
        &self.setup.cache_directory
    }

    #[getter]
    fn matmul_precision(&self) -> &str {
        &self.setup.matmul_precision
    }

    #[getter]
    fn persistent_cache_enabled(&self) -> bool {
        self.setup.persistent_cache_enabled
    }

    #[getter]
    fn persistent_cache_min_entry_size_bytes(&self) -> i64 {
        self.setup.persistent_cache_min_entry_size_bytes
    }

    #[getter]
    fn persistent_cache_min_compile_time_seconds(&self) -> i64 {
        self.setup.persistent_cache_min_compile_time_seconds
    }

    #[getter]
    fn xla_auxiliary_cache_mode(&self) -> &str {
        &self.setup.xla_auxiliary_cache_mode
    }

    #[getter]
    fn xla_auxiliary_cache_reason(&self) -> &str {
        &self.setup.xla_auxiliary_cache_reason
    }

    #[getter]
    fn transfer_guard_enabled(&self) -> bool {
        self.setup.transfer_guard_enabled
    }

    #[getter]
    fn gpu_validation_status(&self) -> &str {
        &self.setup.gpu_validation_status
    }

    #[getter]
    fn gpu_validation_message(&self) -> Option<String> {
        self.setup.gpu_validation_message.clone()
    }
}

#[pymethods]
impl NativeNvidiaDriverProbePaths {
    #[getter]
    fn control_device_path(&self) -> &str {
        &self.paths.control_device_path
    }

    #[getter]
    fn uvm_device_path(&self) -> &str {
        &self.paths.uvm_device_path
    }

    #[getter]
    fn driver_directory_path(&self) -> &str {
        &self.paths.driver_directory_path
    }
}

#[pymethods]
impl NativeJaxRuntimeDiagnosticEvent {
    #[getter]
    fn event_name(&self) -> &str {
        &self.event.event_name
    }

    #[getter]
    fn level(&self) -> &str {
        &self.event.level
    }

    #[getter]
    fn message(&self) -> &str {
        &self.event.message
    }

    #[getter]
    fn fields(&self) -> Vec<NativeJaxRuntimeDiagnosticField> {
        self.event.fields.iter().cloned().map(|field| NativeJaxRuntimeDiagnosticField { field }).collect()
    }
}

#[pymethods]
impl NativeJaxRuntimeDiagnosticField {
    #[getter]
    fn name(&self) -> &str {
        &self.field.name
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.field.value {
            native_jax_runtime::JaxRuntimeDiagnosticValue::Boolean(value) => {
                Ok(PyBool::new(py, *value).to_owned().into_any().unbind())
            }
            native_jax_runtime::JaxRuntimeDiagnosticValue::Integer(value) => {
                Ok(value.into_pyobject(py)?.into_any().unbind())
            }
            native_jax_runtime::JaxRuntimeDiagnosticValue::Text(value) => {
                Ok(value.into_pyobject(py)?.into_any().unbind())
            }
        }
    }
}

#[pymethods]
#[allow(clippy::unused_self)]
impl NativeJaxRuntimeDiagnosticPolicy {
    #[new]
    fn new() -> Self {
        Self
    }

    fn record_jax_runtime_diagnostic_event(
        &self,
        py: Python<'_>,
        event: &Bound<'_, PyAny>,
        telemetry_session: &Bound<'_, PyAny>,
    ) -> PyResult<NativeJaxRuntimeDiagnosticRecordPlan> {
        record_jax_runtime_diagnostic_event(py, event, telemetry_session)
    }
}

impl NativeJaxRuntimeDiagnosticRecordPlan {
    fn from_plan(plan: native_jax_runtime::JaxRuntimeDiagnosticRecordPlan) -> Self {
        Self { plan }
    }
}

#[pyclass]
pub(crate) struct NativeJaxRuntimeSetupSession {
    session: Mutex<native_jax_runtime::JaxRuntimeSetupSession>,
}

#[pymethods]
impl NativeJaxRuntimeSetupSession {
    #[getter]
    fn should_configure(&self) -> PyResult<bool> {
        Ok(self.lock_session()?.should_configure())
    }

    #[getter]
    fn should_validate_gpu(&self) -> PyResult<bool> {
        Ok(self.lock_session()?.side_effect_plan().should_validate_gpu)
    }

    fn setup_report(&self) -> PyResult<NativeJaxRuntimeSetupReport> {
        let session = self.lock_session()?;
        Ok(NativeJaxRuntimeSetupReport { setup: session.setup().clone() })
    }

    fn apply_config_updates(&self, py: Python<'_>) -> PyResult<usize> {
        let updates = self.lock_session()?.config_updates();
        let update_function = py.import("jax")?.getattr("config")?.getattr("update")?;
        for update in &updates {
            match &update.value {
                native_jax_runtime::JaxRuntimeConfigValue::Boolean(value) => {
                    update_function.call1((update.setting_name.as_str(), *value))?;
                }
                native_jax_runtime::JaxRuntimeConfigValue::Integer(value) => {
                    update_function.call1((update.setting_name.as_str(), *value))?;
                }
                native_jax_runtime::JaxRuntimeConfigValue::Text(value) => {
                    update_function.call1((update.setting_name.as_str(), value.as_str()))?;
                }
            }
        }
        Ok(updates.len())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn complete_validation_report(
        &self,
        gpu_validation_status: String,
        gpu_validation_message: Option<String>,
    ) -> PyResult<NativeJaxRuntimeSetupReport> {
        let mut session = self.lock_session()?;
        let completed_setup = session.complete_validation(&gpu_validation_status, gpu_validation_message.as_deref());
        Ok(NativeJaxRuntimeSetupReport { setup: completed_setup })
    }

    fn diagnostic_events(&self) -> PyResult<Vec<NativeJaxRuntimeDiagnosticEvent>> {
        let session = self.lock_session()?;
        Ok(session.diagnostic_events().iter().cloned().map(|event| NativeJaxRuntimeDiagnosticEvent { event }).collect())
    }

    fn create_cache_directory_if_configured(&self) -> PyResult<bool> {
        self.lock_session()?.create_cache_directory_if_configured().map_err(PyOSError::new_err)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn validate_gpu_if_configured<'py>(
        &self,
        py: Python<'py>,
        control_device_path: String,
        uvm_device_path: String,
        driver_directory_path: String,
    ) -> PyResult<NativeJaxRuntimeSetupReport> {
        if !self.lock_session()?.side_effect_plan().should_validate_gpu {
            let session = self.lock_session()?;
            return Ok(NativeJaxRuntimeSetupReport { setup: session.setup().clone() });
        }
        let nvidia_driver_visible = native_jax_runtime::nvidia_driver_files_are_visible(
            Path::new(&control_device_path),
            Path::new(&uvm_device_path),
            Path::new(&driver_directory_path),
        );
        if !nvidia_driver_visible {
            return self.complete_gpu_validation_or_raise(py, false, false, &[]);
        }
        let devices = match observe_jax_devices(py) {
            Ok(devices) => devices,
            Err(_error) => return self.complete_gpu_validation_or_raise(py, true, true, &[]),
        };
        self.complete_gpu_validation_or_raise(py, true, false, &devices)
    }

    fn validate_gpu_if_configured_with_default_probe_paths<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<NativeJaxRuntimeSetupReport> {
        let probe_paths = native_jax_runtime::default_nvidia_driver_probe_paths();
        self.validate_gpu_if_configured(
            py,
            probe_paths.control_device_path,
            probe_paths.uvm_device_path,
            probe_paths.driver_directory_path,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    fn nvidia_driver_files_are_visible(
        &self,
        control_device_path: String,
        uvm_device_path: String,
        driver_directory_path: String,
    ) -> PyResult<bool> {
        let _session = self.lock_session()?;
        Ok(native_jax_runtime::nvidia_driver_files_are_visible(
            Path::new(&control_device_path),
            Path::new(&uvm_device_path),
            Path::new(&driver_directory_path),
        ))
    }

    fn nvidia_driver_files_are_visible_with_default_probe_paths(&self) -> PyResult<bool> {
        let _session = self.lock_session()?;
        let probe_paths = native_jax_runtime::default_nvidia_driver_probe_paths();
        Ok(native_jax_runtime::nvidia_driver_files_are_visible(
            Path::new(&probe_paths.control_device_path),
            Path::new(&probe_paths.uvm_device_path),
            Path::new(&probe_paths.driver_directory_path),
        ))
    }

    fn default_nvidia_driver_probe_paths(&self) -> PyResult<NativeNvidiaDriverProbePaths> {
        let _session = self.lock_session()?;
        Ok(NativeNvidiaDriverProbePaths { paths: native_jax_runtime::default_nvidia_driver_probe_paths() })
    }
}

impl NativeJaxRuntimeSetupSession {
    pub(crate) fn from_session(session: native_jax_runtime::JaxRuntimeSetupSession) -> Self {
        Self { session: Mutex::new(session) }
    }

    pub(crate) fn native_session_snapshot(&self) -> PyResult<native_jax_runtime::JaxRuntimeSetupSession> {
        Ok(self.lock_session()?.clone())
    }

    fn lock_session(&self) -> PyResult<MutexGuard<'_, native_jax_runtime::JaxRuntimeSetupSession>> {
        self.session.lock().map_err(|_| PyValueError::new_err("JAX runtime setup session mutex was poisoned."))
    }

    fn complete_gpu_validation_or_raise<'py>(
        &self,
        _py: Python<'py>,
        nvidia_driver_visible: bool,
        backend_initialization_failed: bool,
        devices: &[native_jax_runtime::JaxDeviceObservation],
    ) -> PyResult<NativeJaxRuntimeSetupReport> {
        let validation_plan =
            native_jax_runtime::plan_jax_gpu_validation(nvidia_driver_visible, backend_initialization_failed, devices);
        let validation_message = validation_plan.message.clone();
        let mut session = self.lock_session()?;
        let completed_setup = session.complete_validation(&validation_plan.status, Some(validation_message.as_str()));
        if validation_plan.should_raise {
            return Err(PyRuntimeError::new_err(validation_message));
        }
        Ok(NativeJaxRuntimeSetupReport { setup: completed_setup })
    }
}

pub(crate) fn record_jax_runtime_diagnostic_event(
    py: Python<'_>,
    event: &Bound<'_, PyAny>,
    telemetry_session: &Bound<'_, PyAny>,
) -> PyResult<NativeJaxRuntimeDiagnosticRecordPlan> {
    let native_telemetry_session = optional_native_telemetry_session(py, telemetry_session)?;
    let plan = record_jax_runtime_diagnostic_log_plan(event, native_telemetry_session.is_some())?;
    if plan.should_emit_telemetry {
        let active_native_telemetry_session = native_telemetry_session.ok_or_else(|| {
            PyRuntimeError::new_err("Native JAX diagnostic telemetry plan selected a missing native session.")
        })?;
        let keyword_arguments = PyDict::new(py);
        keyword_arguments.set_item("telemetry_level", &plan.telemetry_level)?;
        active_native_telemetry_session.call_method(
            "emit_jax_runtime_diagnostic_event",
            (event,),
            Some(&keyword_arguments),
        )?;
    }
    Ok(NativeJaxRuntimeDiagnosticRecordPlan::from_plan(plan))
}

fn optional_native_telemetry_session<'py>(
    py: Python<'py>,
    telemetry_session: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if telemetry_session.is_none() {
        return Ok(None);
    }
    match telemetry_session.getattr("native_telemetry_session") {
        Ok(native_telemetry_session) if native_telemetry_session.is_none() => Ok(None),
        Ok(native_telemetry_session) => Ok(Some(native_telemetry_session)),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Err(PyTypeError::new_err(
            "JAX runtime diagnostic telemetry requires a TelemetrySession with a native telemetry session handle.",
        )),
        Err(error) => Err(error),
    }
}

fn record_jax_runtime_diagnostic_log_plan(
    event: &Bound<'_, PyAny>,
    has_telemetry_session: bool,
) -> PyResult<native_jax_runtime::JaxRuntimeDiagnosticRecordPlan> {
    let diagnostic_level = jax_runtime_diagnostic_event_level(event)?;
    let plan = native_jax_runtime::plan_jax_runtime_diagnostic_record(&diagnostic_level, has_telemetry_session);
    let (event_name, fields) = jax_runtime_diagnostic_event_fields_to_native(event)?;
    let message = event.getattr("message")?.extract::<String>()?;
    let fields_json = native_jax_runtime::serialize_jax_runtime_diagnostic_fields_json(&fields).map_err(|error| {
        PyValueError::new_err(format!("Failed to serialize JAX runtime diagnostic event fields: {error}"))
    })?;
    logging::emit_diagnostic_event(&plan.logging_level_name.to_lowercase(), &event_name, &message, Some(fields_json))?;
    Ok(plan)
}

fn jax_runtime_diagnostic_event_fields_to_native(
    event: &Bound<'_, PyAny>,
) -> PyResult<(String, Vec<native_jax_runtime::JaxRuntimeDiagnosticFieldPayload>)> {
    let event_name = event.getattr("event_name")?.extract::<String>()?;
    let mut fields = Vec::new();
    for field in event.getattr("fields")?.try_iter()? {
        let field = field?;
        let field_name = field.getattr("name")?.extract::<String>()?;
        let field_value = field.getattr("value")?;
        fields.push(native_jax_runtime::JaxRuntimeDiagnosticFieldPayload {
            name: field_name,
            value: jax_runtime_diagnostic_value_from_py(&field_value)?,
        });
    }
    Ok((event_name, fields))
}

pub(crate) fn jax_runtime_diagnostic_event_fields_to_py_dict<'py>(
    py: Python<'py>,
    event: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let (event_name, native_fields) = jax_runtime_diagnostic_event_fields_to_native(event)?;
    let fields = PyDict::new(py);
    for native_field in native_fields {
        match native_field.value {
            native_jax_runtime::JaxRuntimeDiagnosticValue::Boolean(value) => {
                fields.set_item(native_field.name, value)?;
            }
            native_jax_runtime::JaxRuntimeDiagnosticValue::Integer(value) => {
                fields.set_item(native_field.name, value)?;
            }
            native_jax_runtime::JaxRuntimeDiagnosticValue::Text(value) => {
                fields.set_item(native_field.name, value)?;
            }
        }
    }
    Ok((event_name, fields))
}

fn jax_runtime_diagnostic_value_from_py(
    value: &Bound<'_, PyAny>,
) -> PyResult<native_jax_runtime::JaxRuntimeDiagnosticValue> {
    if let Ok(boolean_value) = value.extract::<bool>() {
        return Ok(native_jax_runtime::JaxRuntimeDiagnosticValue::Boolean(boolean_value));
    }
    if let Ok(integer_value) = value.extract::<i64>() {
        return Ok(native_jax_runtime::JaxRuntimeDiagnosticValue::Integer(integer_value));
    }
    let text_value = value
        .extract::<String>()
        .or_else(|_| value.str().map(|string_value| string_value.to_string_lossy().into_owned()))?;
    Ok(native_jax_runtime::JaxRuntimeDiagnosticValue::Text(text_value))
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeJaxRuntimeDiagnosticEvent>()?;
    module.add_class::<NativeJaxRuntimeDiagnosticField>()?;
    module.add_class::<NativeJaxRuntimeDiagnosticPolicy>()?;
    module.add_class::<NativeJaxRuntimeDiagnosticRecordPlan>()?;
    module.add_class::<NativeJaxRuntimeSetupReport>()?;
    module.add_class::<NativeJaxRuntimeSetupSession>()?;
    module.add_class::<NativeNvidiaDriverProbePaths>()?;
    Ok(())
}

fn jax_runtime_diagnostic_event_level(event: &Bound<'_, PyAny>) -> PyResult<String> {
    let level = event.getattr("level")?;
    if let Ok(level_value) = level.getattr("value") {
        return level_value.extract::<String>();
    }
    level.extract::<String>()
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
