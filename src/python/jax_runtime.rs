//! PyO3 adapters for deterministic JAX runtime setup policy.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyAttributeError, PyOSError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};

use g_runtime::jax_runtime as native_jax_runtime;

use super::logging;

#[pyclass]
pub(crate) struct NativeJaxRuntimeDiagnosticRecordPlan {
    plan: native_jax_runtime::JaxRuntimeDiagnosticRecordPlan,
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
    #[new]
    fn new(setup_payload: &Bound<'_, PyAny>, should_configure: bool) -> PyResult<Self> {
        Ok(Self::from_session(native_jax_runtime::JaxRuntimeSetupSession::new(
            should_configure,
            parse_jax_runtime_setup_payload(setup_payload)?,
        )))
    }

    #[getter]
    fn should_configure(&self) -> PyResult<bool> {
        Ok(self.lock_session()?.should_configure())
    }

    fn setup_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let session = self.lock_session()?;
        jax_runtime_setup_payload_to_dict(py, session.setup())
    }

    fn side_effect_plan_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let session = self.lock_session()?;
        jax_runtime_setup_side_effect_plan_to_dict(py, &session.side_effect_plan())
    }

    fn config_update_payloads<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let session = self.lock_session()?;
        let update_payloads = session
            .config_updates()
            .iter()
            .map(|update| jax_runtime_config_update_payload_to_dict(py, update))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, &update_payloads)
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
    fn complete_validation_payload<'py>(
        &self,
        py: Python<'py>,
        gpu_validation_status: String,
        gpu_validation_message: Option<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut session = self.lock_session()?;
        let completed_setup = session.complete_validation(&gpu_validation_status, gpu_validation_message.as_deref());
        jax_runtime_setup_payload_to_dict(py, &completed_setup)
    }

    fn diagnostic_event_payloads<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let session = self.lock_session()?;
        let event_payloads = session
            .diagnostic_events()
            .iter()
            .map(|event| jax_runtime_diagnostic_event_payload_to_dict(py, event))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, &event_payloads)
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
    ) -> PyResult<Bound<'py, PyDict>> {
        if !self.lock_session()?.side_effect_plan().should_validate_gpu {
            let session = self.lock_session()?;
            return jax_runtime_setup_payload_to_dict(py, session.setup());
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
        py: Python<'py>,
        nvidia_driver_visible: bool,
        backend_initialization_failed: bool,
        devices: &[native_jax_runtime::JaxDeviceObservation],
    ) -> PyResult<Bound<'py, PyDict>> {
        let validation_plan =
            native_jax_runtime::plan_jax_gpu_validation(nvidia_driver_visible, backend_initialization_failed, devices);
        let validation_message = validation_plan.message.clone();
        let mut session = self.lock_session()?;
        let completed_setup = session.complete_validation(&validation_plan.status, Some(validation_message.as_str()));
        if validation_plan.should_raise {
            return Err(PyRuntimeError::new_err(validation_message));
        }
        jax_runtime_setup_payload_to_dict(py, &completed_setup)
    }
}

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
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_jax_runtime_setup_side_effects_payload<'py>(
    py: Python<'py>,
    requested_device: String,
    persistent_cache_enabled: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let plan = native_jax_runtime::plan_jax_runtime_setup_side_effects(&requested_device, persistent_cache_enabled);
    jax_runtime_setup_side_effect_plan_to_dict(py, &plan)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_jax_runtime_diagnostic_record_payload<'py>(
    py: Python<'py>,
    diagnostic_level: String,
    has_telemetry_session: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let plan = native_jax_runtime::plan_jax_runtime_diagnostic_record(&diagnostic_level, has_telemetry_session);
    jax_runtime_diagnostic_record_plan_to_dict(py, &plan)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_jax_runtime_diagnostic_record(
    diagnostic_level: String,
    has_telemetry_session: bool,
) -> NativeJaxRuntimeDiagnosticRecordPlan {
    NativeJaxRuntimeDiagnosticRecordPlan::from_plan(native_jax_runtime::plan_jax_runtime_diagnostic_record(
        &diagnostic_level,
        has_telemetry_session,
    ))
}

#[pyfunction]
pub(crate) fn record_jax_runtime_diagnostic_log_event(
    py: Python<'_>,
    event: &Bound<'_, PyAny>,
    has_telemetry_session: bool,
) -> PyResult<NativeJaxRuntimeDiagnosticRecordPlan> {
    record_jax_runtime_diagnostic_log_event_plan(py, event, has_telemetry_session)
        .map(NativeJaxRuntimeDiagnosticRecordPlan::from_plan)
}

#[pyfunction]
pub(crate) fn record_jax_runtime_diagnostic_event(
    py: Python<'_>,
    event: &Bound<'_, PyAny>,
    telemetry_session: &Bound<'_, PyAny>,
) -> PyResult<NativeJaxRuntimeDiagnosticRecordPlan> {
    let native_telemetry_session = optional_native_telemetry_session(py, telemetry_session)?;
    let plan = record_jax_runtime_diagnostic_log_event_plan(py, event, native_telemetry_session.is_some())?;
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

fn record_jax_runtime_diagnostic_log_event_plan(
    py: Python<'_>,
    event: &Bound<'_, PyAny>,
    has_telemetry_session: bool,
) -> PyResult<native_jax_runtime::JaxRuntimeDiagnosticRecordPlan> {
    let diagnostic_level = jax_runtime_diagnostic_event_level(event)?;
    let plan = native_jax_runtime::plan_jax_runtime_diagnostic_record(&diagnostic_level, has_telemetry_session);
    let (event_name, fields) = jax_runtime_diagnostic_event_fields_to_py_dict(py, event)?;
    let message = event.getattr("message")?.extract::<String>()?;
    logging::emit_diagnostic_event_fields(
        py,
        &plan.logging_level_name.to_lowercase(),
        &event_name,
        &message,
        fields.as_any(),
    )?;
    Ok(plan)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn nvidia_driver_files_are_visible_value(
    control_device_path: String,
    uvm_device_path: String,
    driver_directory_path: String,
) -> bool {
    native_jax_runtime::nvidia_driver_files_are_visible(
        Path::new(&control_device_path),
        Path::new(&uvm_device_path),
        Path::new(&driver_directory_path),
    )
}

#[pyfunction]
pub(crate) fn default_nvidia_driver_probe_paths_payload<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    nvidia_driver_probe_paths_payload_to_dict(py, &native_jax_runtime::default_nvidia_driver_probe_paths())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn complete_jax_runtime_setup_validation_payload<'py>(
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
) -> PyResult<Bound<'py, PyDict>> {
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
        gpu_validation_status: String::new(),
        gpu_validation_message: None,
    };
    let completed_setup = native_jax_runtime::complete_jax_runtime_setup_validation(
        &setup,
        &gpu_validation_status,
        gpu_validation_message.as_deref(),
    );
    jax_runtime_setup_payload_to_dict(py, &completed_setup)
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

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_jax_runtime_config_update_payloads<'py>(
    py: Python<'py>,
    platform_name: String,
    cache_directory: String,
    matmul_precision: String,
    persistent_cache_enabled: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: i64,
    xla_auxiliary_cache_mode: String,
    transfer_guard_enabled: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let setup = native_jax_runtime::JaxRuntimeSetupPayload {
        requested_device: String::new(),
        platform_name,
        cache_directory,
        matmul_precision,
        persistent_cache_enabled,
        persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds,
        xla_auxiliary_cache_mode,
        xla_auxiliary_cache_reason: String::new(),
        transfer_guard_enabled,
        gpu_validation_status: String::new(),
        gpu_validation_message: None,
    };
    let updates = native_jax_runtime::plan_jax_runtime_config_updates(&setup);
    let update_payloads = updates
        .iter()
        .map(|update| jax_runtime_config_update_payload_to_dict(py, update))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &update_payloads)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_jax_gpu_validation_payload<'py>(
    py: Python<'py>,
    nvidia_driver_visible: bool,
    backend_initialization_failed: bool,
    device_platforms: Vec<String>,
    device_descriptions: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    if device_platforms.len() != device_descriptions.len() {
        return Err(PyValueError::new_err("JAX GPU validation device platform and description counts must match."));
    }
    let devices = device_platforms
        .into_iter()
        .zip(device_descriptions)
        .map(|(platform, description)| native_jax_runtime::JaxDeviceObservation { platform, description })
        .collect::<Vec<_>>();
    let plan =
        native_jax_runtime::plan_jax_gpu_validation(nvidia_driver_visible, backend_initialization_failed, &devices);
    jax_gpu_validation_plan_to_dict(py, &plan)
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

fn parse_jax_runtime_setup_payload(payload: &Bound<'_, PyAny>) -> PyResult<native_jax_runtime::JaxRuntimeSetupPayload> {
    Ok(native_jax_runtime::JaxRuntimeSetupPayload {
        requested_device: payload.get_item("requested_device")?.extract::<String>()?,
        platform_name: payload.get_item("platform_name")?.extract::<String>()?,
        cache_directory: payload.get_item("cache_directory")?.extract::<String>()?,
        matmul_precision: payload.get_item("matmul_precision")?.extract::<String>()?,
        persistent_cache_enabled: payload.get_item("persistent_cache_enabled")?.extract::<bool>()?,
        persistent_cache_min_entry_size_bytes: payload
            .get_item("persistent_cache_min_entry_size_bytes")?
            .extract::<i64>()?,
        persistent_cache_min_compile_time_seconds: payload
            .get_item("persistent_cache_min_compile_time_seconds")?
            .extract::<i64>()?,
        xla_auxiliary_cache_mode: payload.get_item("xla_auxiliary_cache_mode")?.extract::<String>()?,
        xla_auxiliary_cache_reason: payload.get_item("xla_auxiliary_cache_reason")?.extract::<String>()?,
        transfer_guard_enabled: payload.get_item("transfer_guard_enabled")?.extract::<bool>()?,
        gpu_validation_status: payload.get_item("gpu_validation_status")?.extract::<String>()?,
        gpu_validation_message: extract_optional_string(&payload.get_item("gpu_validation_message")?)?,
    })
}

fn jax_runtime_diagnostic_record_plan_to_dict<'py>(
    py: Python<'py>,
    plan: &native_jax_runtime::JaxRuntimeDiagnosticRecordPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("logging_level_name", &plan.logging_level_name)?;
    payload.set_item("should_emit_telemetry", plan.should_emit_telemetry)?;
    payload.set_item("telemetry_level", &plan.telemetry_level)?;
    Ok(payload)
}

fn jax_runtime_setup_side_effect_plan_to_dict<'py>(
    py: Python<'py>,
    plan: &native_jax_runtime::JaxRuntimeSetupSideEffectPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("should_create_cache_directory", plan.should_create_cache_directory)?;
    payload.set_item("should_validate_gpu", plan.should_validate_gpu)?;
    Ok(payload)
}

fn nvidia_driver_probe_paths_payload_to_dict<'py>(
    py: Python<'py>,
    paths: &native_jax_runtime::NvidiaDriverProbePathsPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("control_device_path", &paths.control_device_path)?;
    payload.set_item("uvm_device_path", &paths.uvm_device_path)?;
    payload.set_item("driver_directory_path", &paths.driver_directory_path)?;
    Ok(payload)
}

fn jax_gpu_validation_plan_to_dict<'py>(
    py: Python<'py>,
    plan: &native_jax_runtime::JaxGpuValidationPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("status", &plan.status)?;
    payload.set_item("message", &plan.message)?;
    payload.set_item("should_raise", plan.should_raise)?;
    Ok(payload)
}

fn jax_runtime_config_update_payload_to_dict<'py>(
    py: Python<'py>,
    update: &native_jax_runtime::JaxRuntimeConfigUpdatePayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("setting_name", &update.setting_name)?;
    match &update.value {
        native_jax_runtime::JaxRuntimeConfigValue::Boolean(value) => payload.set_item("value", *value)?,
        native_jax_runtime::JaxRuntimeConfigValue::Integer(value) => payload.set_item("value", *value)?,
        native_jax_runtime::JaxRuntimeConfigValue::Text(value) => payload.set_item("value", value)?,
    }
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

pub(crate) fn jax_runtime_diagnostic_event_fields_to_py_dict<'py>(
    py: Python<'py>,
    event: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let event_name = event.getattr("event_name")?.extract::<String>()?;
    let fields = PyDict::new(py);
    for field in event.getattr("fields")?.try_iter()? {
        let field = field?;
        let field_name = field.getattr("name")?.extract::<String>()?;
        let field_value = field.getattr("value")?;
        fields.set_item(field_name, field_value)?;
    }
    Ok((event_name, fields))
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeJaxRuntimeDiagnosticRecordPlan>()?;
    module.add_class::<NativeJaxRuntimeSetupSession>()?;
    module.add_function(wrap_pyfunction!(resolve_jax_runtime_setup_payload, module)?)?;
    module.add_function(wrap_pyfunction!(complete_jax_runtime_setup_validation_payload, module)?)?;
    module.add_function(wrap_pyfunction!(nvidia_driver_files_are_visible_value, module)?)?;
    module.add_function(wrap_pyfunction!(default_nvidia_driver_probe_paths_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_jax_runtime_setup_diagnostic_payloads, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_runtime_config_update_payloads, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_runtime_diagnostic_record, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_runtime_diagnostic_record_payload, module)?)?;
    module.add_function(wrap_pyfunction!(record_jax_runtime_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_jax_runtime_diagnostic_log_event, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_runtime_setup_side_effects_payload, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_gpu_validation_payload, module)?)?;
    Ok(())
}

fn jax_runtime_diagnostic_event_level(event: &Bound<'_, PyAny>) -> PyResult<String> {
    let level = event.getattr("level")?;
    if let Ok(level_value) = level.getattr("value") {
        return level_value.extract::<String>();
    }
    level.extract::<String>()
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

fn extract_optional_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() { Ok(None) } else { Ok(Some(value.extract::<String>()?)) }
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
