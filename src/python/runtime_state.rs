//! PyO3 handle for native process runtime state.

use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use g_runtime::runtime_policy as native_runtime_policy;
use g_runtime::runtime_state as native_runtime_state;

#[pyclass]
pub(crate) struct NativeRuntimeCompatibilityToken {
    token: native_runtime_state::RuntimeCompatibilityToken,
}

#[pyclass]
pub(crate) struct NativeRayonThreadPoolConfigurationPlan {
    inner: native_runtime_state::RayonThreadPoolConfigurationPlan,
}

#[pyclass]
pub(crate) struct NativeJaxRuntimeSetupLifecyclePlan {
    inner: native_runtime_state::JaxRuntimeSetupLifecyclePlan,
}

#[pyclass]
pub(crate) struct NativeRuntimeState {
    state: Mutex<native_runtime_state::ProcessRuntimeState>,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_jax_runtime_policy_payload<'py>(
    py: Python<'py>,
    device: String,
    cache_directory: Option<String>,
    matmul_precision: Option<String>,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: i64,
    xla_autotune_cache: bool,
    transfer_guard: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = native_runtime_state::build_jax_runtime_policy_payload(
        &device,
        cache_directory.as_deref(),
        matmul_precision.as_deref(),
        persistent_cache,
        persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds,
        xla_autotune_cache,
        transfer_guard,
    );
    jax_runtime_policy_payload_to_dict(py, &payload)
}

#[pymethods]
impl NativeRayonThreadPoolConfigurationPlan {
    #[getter]
    fn should_configure(&self) -> bool {
        self.inner.should_configure
    }

    #[getter]
    fn thread_count(&self) -> Option<i64> {
        self.inner.thread_count
    }
}

#[pymethods]
impl NativeJaxRuntimeSetupLifecyclePlan {
    #[getter]
    fn should_configure(&self) -> bool {
        self.inner.should_configure
    }
}

#[pymethods]
impl NativeRuntimeState {
    #[new]
    fn new() -> Self {
        Self { state: Mutex::new(native_runtime_state::ProcessRuntimeState::default()) }
    }

    #[getter]
    fn rayon_thread_count(&self) -> PyResult<Option<i64>> {
        Ok(self.lock_state()?.rayon_thread_count)
    }

    fn logging_runtime_policy_payload<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let state = self.lock_state()?;
        state.logging_policy.as_ref().map(|policy| logging_runtime_policy_payload_to_dict(py, policy)).transpose()
    }

    fn jax_runtime_policy_payload<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let state = self.lock_state()?;
        state.jax_policy.as_ref().map(|policy| jax_runtime_policy_payload_to_dict(py, policy)).transpose()
    }

    fn require_compatible_runtime_policy(
        &self,
        logging_policy_payload: &Bound<'_, PyAny>,
        rayon_thread_count: Option<i64>,
        jax_policy_payload: &Bound<'_, PyAny>,
    ) -> PyResult<NativeRuntimeCompatibilityToken> {
        let logging_policy = parse_logging_runtime_policy_payload(logging_policy_payload)?;
        let jax_policy = parse_jax_runtime_policy_payload(jax_policy_payload)?;
        let token = self
            .lock_state()?
            .require_compatible_runtime_policy(&logging_policy, rayon_thread_count, &jax_policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeRuntimeCompatibilityToken { token })
    }

    fn require_compatible_logging_runtime_policy(&self, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let logging_policy = parse_logging_runtime_policy_payload(payload)?;
        self.lock_state()?
            .require_compatible_logging_policy(&logging_policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn record_logging_runtime_policy(&self, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let logging_policy = parse_logging_runtime_policy_payload(payload)?;
        self.lock_state()?.record_logging_policy(logging_policy);
        Ok(())
    }

    fn require_compatible_rayon_thread_count(&self, thread_count: Option<i64>) -> PyResult<()> {
        self.lock_state()?
            .require_compatible_rayon_thread_count(thread_count)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn record_rayon_thread_count(&self, thread_count: i64) -> PyResult<()> {
        self.lock_state()?.record_rayon_thread_count(thread_count);
        Ok(())
    }

    fn plan_rayon_thread_pool_configuration(
        &self,
        thread_count: i64,
    ) -> PyResult<NativeRayonThreadPoolConfigurationPlan> {
        let plan = self
            .lock_state()?
            .plan_rayon_thread_pool_configuration(thread_count)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeRayonThreadPoolConfigurationPlan { inner: plan })
    }

    fn effective_rayon_thread_count(&self, requested_thread_count: Option<i64>) -> PyResult<Option<i64>> {
        Ok(self.lock_state()?.effective_rayon_thread_count(requested_thread_count))
    }

    fn require_compatible_jax_runtime_policy(&self, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let jax_policy = parse_jax_runtime_policy_payload(payload)?;
        self.lock_state()?
            .require_compatible_jax_policy(&jax_policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn record_jax_runtime_policy(&self, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let jax_policy = parse_jax_runtime_policy_payload(payload)?;
        self.lock_state()?.record_jax_policy(jax_policy);
        Ok(())
    }

    fn plan_jax_runtime_setup_lifecycle(
        &self,
        payload: &Bound<'_, PyAny>,
    ) -> PyResult<NativeJaxRuntimeSetupLifecyclePlan> {
        let jax_policy = parse_jax_runtime_policy_payload(payload)?;
        let plan = self
            .lock_state()?
            .plan_jax_runtime_setup_lifecycle(&jax_policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeJaxRuntimeSetupLifecyclePlan { inner: plan })
    }
}

impl NativeRuntimeCompatibilityToken {
    pub(crate) fn native_token(&self) -> &native_runtime_state::RuntimeCompatibilityToken {
        &self.token
    }
}

impl NativeRuntimeState {
    fn lock_state(&self) -> PyResult<MutexGuard<'_, native_runtime_state::ProcessRuntimeState>> {
        self.state.lock().map_err(|_| PyRuntimeError::new_err("Runtime state mutex was poisoned."))
    }
}

fn parse_logging_runtime_policy_payload(
    payload: &Bound<'_, PyAny>,
) -> PyResult<native_runtime_policy::LoggingRuntimePolicyPayload> {
    Ok(native_runtime_policy::LoggingRuntimePolicyPayload {
        log_filter: payload.get_item("log_filter")?.extract::<String>()?,
        log_file: extract_optional_string(&payload.get_item("log_file")?)?,
        log_stderr: payload.get_item("log_stderr")?.extract::<bool>()?,
        log_queue_size: payload.get_item("log_queue_size")?.extract::<i64>()?,
        log_lossy: payload.get_item("log_lossy")?.extract::<bool>()?,
        include_source_location: payload.get_item("include_source_location")?.extract::<bool>()?,
        include_span_events: payload.get_item("include_span_events")?.extract::<bool>()?,
        trace_file: extract_optional_string(&payload.get_item("trace_file")?)?,
        trace_filter: payload.get_item("trace_filter")?.extract::<String>()?,
        trace_event_cap: extract_optional_i64(&payload.get_item("trace_event_cap")?)?,
    })
}

fn extract_optional_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() { Ok(None) } else { Ok(Some(value.extract::<String>()?)) }
}

fn extract_optional_i64(value: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if value.is_none() { Ok(None) } else { Ok(Some(value.extract::<i64>().map_err(PyValueError::new_err)?)) }
}

fn parse_jax_runtime_policy_payload(
    payload: &Bound<'_, PyAny>,
) -> PyResult<native_runtime_state::JaxRuntimePolicyPayload> {
    Ok(native_runtime_state::JaxRuntimePolicyPayload {
        device: payload.get_item("device")?.extract::<String>()?,
        cache_directory: extract_optional_string(&payload.get_item("cache_directory")?)?,
        matmul_precision: extract_optional_string(&payload.get_item("matmul_precision")?)?,
        persistent_cache: payload.get_item("persistent_cache")?.extract::<bool>()?,
        persistent_cache_min_entry_size_bytes: payload
            .get_item("persistent_cache_min_entry_size_bytes")?
            .extract::<i64>()?,
        persistent_cache_min_compile_time_seconds: payload
            .get_item("persistent_cache_min_compile_time_seconds")?
            .extract::<i64>()?,
        xla_autotune_cache: payload.get_item("xla_autotune_cache")?.extract::<bool>()?,
        transfer_guard: payload.get_item("transfer_guard")?.extract::<bool>()?,
    })
}

fn logging_runtime_policy_payload_to_dict<'py>(
    py: Python<'py>,
    payload: &native_runtime_policy::LoggingRuntimePolicyPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let python_payload = PyDict::new(py);
    python_payload.set_item("log_filter", &payload.log_filter)?;
    python_payload.set_item("log_file", &payload.log_file)?;
    python_payload.set_item("log_stderr", payload.log_stderr)?;
    python_payload.set_item("log_queue_size", payload.log_queue_size)?;
    python_payload.set_item("log_lossy", payload.log_lossy)?;
    python_payload.set_item("include_source_location", payload.include_source_location)?;
    python_payload.set_item("include_span_events", payload.include_span_events)?;
    python_payload.set_item("trace_file", &payload.trace_file)?;
    python_payload.set_item("trace_filter", &payload.trace_filter)?;
    python_payload.set_item("trace_event_cap", payload.trace_event_cap)?;
    Ok(python_payload)
}

fn jax_runtime_policy_payload_to_dict<'py>(
    py: Python<'py>,
    payload: &native_runtime_state::JaxRuntimePolicyPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let python_payload = PyDict::new(py);
    python_payload.set_item("device", &payload.device)?;
    python_payload.set_item("cache_directory", &payload.cache_directory)?;
    python_payload.set_item("matmul_precision", &payload.matmul_precision)?;
    python_payload.set_item("persistent_cache", payload.persistent_cache)?;
    python_payload.set_item("persistent_cache_min_entry_size_bytes", payload.persistent_cache_min_entry_size_bytes)?;
    python_payload
        .set_item("persistent_cache_min_compile_time_seconds", payload.persistent_cache_min_compile_time_seconds)?;
    python_payload.set_item("xla_autotune_cache", payload.xla_autotune_cache)?;
    python_payload.set_item("transfer_guard", payload.transfer_guard)?;
    Ok(python_payload)
}
