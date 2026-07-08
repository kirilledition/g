//! PyO3 handle for native process runtime state.

#![allow(clippy::needless_pass_by_value)]

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_genotype::debug::set_bgen_decode_tile_variant_count;
use g_runtime::debug as native_runtime_paths;
use g_runtime::debug as native_runtime_policy;
use g_runtime::debug as native_runtime_state;

use super::errors;
use super::jax_runtime::NativeJaxRuntimeSetupSession;
use super::logging;
use super::run_events;
use super::telemetry_policy;

#[pyclass]
pub(crate) struct NativeRuntimeCompatibilityToken {
    token: native_runtime_state::RuntimeCompatibilityToken,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeLoggingRuntimePolicy {
    policy: native_runtime_policy::LoggingRuntimePolicyPayload,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeJaxRuntimePolicy {
    policy: native_runtime_state::JaxRuntimePolicyPayload,
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
pub(crate) struct NativeRuntimePolicy {
    policy: native_runtime_state::RuntimePolicyPayload,
}

#[pyclass]
pub(crate) struct NativeRunRuntime {
    runtime: native_runtime_state::RunRuntime,
}

#[pyclass]
pub(crate) struct NativeRuntimeState {
    state: Arc<Mutex<native_runtime_state::ProcessRuntimeState>>,
}

static GLOBAL_PROCESS_RUNTIME_STATE: OnceLock<Arc<Mutex<native_runtime_state::ProcessRuntimeState>>> = OnceLock::new();

fn global_process_runtime_state() -> NativeRuntimeState {
    NativeRuntimeState {
        state: Arc::clone(
            GLOBAL_PROCESS_RUNTIME_STATE
                .get_or_init(|| Arc::new(Mutex::new(native_runtime_state::ProcessRuntimeState::default()))),
        ),
    }
}

#[pymethods]
impl NativeLoggingRuntimePolicy {
    #[getter]
    fn log_filter(&self) -> &str {
        &self.policy.log_filter
    }

    #[getter]
    fn log_file(&self) -> Option<String> {
        self.policy.log_file.clone()
    }

    #[getter]
    fn log_stderr(&self) -> bool {
        self.policy.log_stderr
    }

    #[getter]
    fn log_queue_size(&self) -> i64 {
        self.policy.log_queue_size
    }

    #[getter]
    fn log_lossy(&self) -> bool {
        self.policy.log_lossy
    }

    #[getter]
    fn include_source_location(&self) -> bool {
        self.policy.include_source_location
    }

    #[getter]
    fn include_span_events(&self) -> bool {
        self.policy.include_span_events
    }

    #[getter]
    fn trace_file(&self) -> Option<String> {
        self.policy.trace_file.clone()
    }

    #[getter]
    fn trace_filter(&self) -> &str {
        &self.policy.trace_filter
    }

    #[getter]
    fn trace_event_cap(&self) -> Option<i64> {
        self.policy.trace_event_cap
    }
}

#[pymethods]
impl NativeJaxRuntimePolicy {
    #[getter]
    fn device(&self) -> &str {
        &self.policy.device
    }

    #[getter]
    fn cache_directory(&self) -> Option<String> {
        self.policy.cache_directory.clone()
    }

    #[getter]
    fn matmul_precision(&self) -> Option<String> {
        self.policy.matmul_precision.clone()
    }

    #[getter]
    fn persistent_cache(&self) -> bool {
        self.policy.persistent_cache
    }

    #[getter]
    fn persistent_cache_min_entry_size_bytes(&self) -> i64 {
        self.policy.persistent_cache_min_entry_size_bytes
    }

    #[getter]
    fn persistent_cache_min_compile_time_seconds(&self) -> i64 {
        self.policy.persistent_cache_min_compile_time_seconds
    }

    #[getter]
    fn xla_autotune_cache(&self) -> bool {
        self.policy.xla_autotune_cache
    }

    #[getter]
    fn transfer_guard(&self) -> bool {
        self.policy.transfer_guard
    }
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
impl NativeRuntimePolicy {
    #[getter]
    fn rayon_thread_count(&self) -> Option<i64> {
        self.policy.rayon_thread_count
    }

    fn logging_runtime_policy(&self) -> NativeLoggingRuntimePolicy {
        NativeLoggingRuntimePolicy { policy: self.policy.logging_policy.clone() }
    }

    fn jax_runtime_policy(&self) -> NativeJaxRuntimePolicy {
        NativeJaxRuntimePolicy { policy: self.policy.jax_policy.clone() }
    }
}

#[pymethods]
impl NativeRunRuntime {
    #[getter]
    fn rayon_thread_count(&self) -> Option<i64> {
        self.runtime.runtime_policy.rayon_thread_count
    }

    fn logging_runtime_policy(&self) -> NativeLoggingRuntimePolicy {
        NativeLoggingRuntimePolicy { policy: self.runtime.runtime_policy.logging_policy.clone() }
    }

    fn jax_runtime_policy(&self) -> NativeJaxRuntimePolicy {
        NativeJaxRuntimePolicy { policy: self.runtime.runtime_policy.jax_policy.clone() }
    }

    fn runtime_compatibility_token(&self) -> NativeRuntimeCompatibilityToken {
        NativeRuntimeCompatibilityToken { token: self.runtime.compatibility_token.clone() }
    }
}

#[pymethods]
impl NativeRuntimeState {
    #[new]
    fn new() -> Self {
        Self { state: Arc::new(Mutex::new(native_runtime_state::ProcessRuntimeState::default())) }
    }

    #[staticmethod]
    fn global_process_runtime_state() -> Self {
        global_process_runtime_state()
    }

    #[getter]
    fn rayon_thread_count(&self) -> PyResult<Option<i64>> {
        Ok(self.lock_state()?.rayon_thread_count)
    }

    fn logging_runtime_policy(&self) -> PyResult<Option<NativeLoggingRuntimePolicy>> {
        let state = self.lock_state()?;
        Ok(state.logging_policy.clone().map(|policy| NativeLoggingRuntimePolicy { policy }))
    }

    fn jax_runtime_policy(&self) -> PyResult<Option<NativeJaxRuntimePolicy>> {
        let state = self.lock_state()?;
        Ok(state.jax_policy.clone().map(|policy| NativeJaxRuntimePolicy { policy }))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn default_local_cache_directory_value(&self, directory_name: String) -> PyResult<String> {
        let _state = self.lock_state()?;
        Ok(native_runtime_paths::default_local_cache_directory(&directory_name).to_string_lossy().into_owned())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn build_logging_runtime_policy(
        &self,
        log_filter: String,
        log_file: Option<String>,
        log_stderr: bool,
        log_queue_size: i64,
        log_lossy: bool,
        include_source_location: bool,
        include_span_events: bool,
        trace_file: Option<String>,
        trace_filter: String,
        trace_event_cap: Option<i64>,
        telemetry_mode: String,
        telemetry_stream_file: Option<String>,
    ) -> PyResult<NativeLoggingRuntimePolicy> {
        let _state = self.lock_state()?;
        let parsed_telemetry_mode = telemetry_policy::parse_telemetry_mode(&telemetry_mode)?;
        let policy = native_runtime_policy::build_logging_runtime_policy(
            log_filter,
            log_file,
            log_stderr,
            log_queue_size,
            log_lossy,
            include_source_location,
            include_span_events,
            trace_file,
            trace_filter,
            trace_event_cap,
            parsed_telemetry_mode,
            telemetry_stream_file,
        );
        Ok(NativeLoggingRuntimePolicy { policy })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn build_jax_runtime_policy(
        &self,
        device: String,
        cache_directory: Option<String>,
        matmul_precision: Option<String>,
        persistent_cache: bool,
        persistent_cache_min_entry_size_bytes: i64,
        persistent_cache_min_compile_time_seconds: i64,
        xla_autotune_cache: bool,
        transfer_guard: bool,
    ) -> PyResult<NativeJaxRuntimePolicy> {
        let _state = self.lock_state()?;
        let policy = native_runtime_state::build_jax_runtime_policy_payload(
            &device,
            cache_directory.as_deref(),
            matmul_precision.as_deref(),
            persistent_cache,
            persistent_cache_min_entry_size_bytes,
            persistent_cache_min_compile_time_seconds,
            xla_autotune_cache,
            transfer_guard,
        );
        Ok(NativeJaxRuntimePolicy { policy })
    }

    fn build_runtime_policy_handle(
        &self,
        logging_policy: PyRef<'_, NativeLoggingRuntimePolicy>,
        rayon_thread_count: Option<i64>,
        jax_policy: PyRef<'_, NativeJaxRuntimePolicy>,
    ) -> PyResult<NativeRuntimePolicy> {
        let _state = self.lock_state()?;
        Ok(NativeRuntimePolicy {
            policy: native_runtime_state::RuntimePolicyPayload {
                logging_policy: logging_policy.policy.clone(),
                rayon_thread_count,
                jax_policy: jax_policy.policy.clone(),
            },
        })
    }

    fn require_compatible_runtime_policy(
        &self,
        logging_policy: PyRef<'_, NativeLoggingRuntimePolicy>,
        rayon_thread_count: Option<i64>,
        jax_policy: PyRef<'_, NativeJaxRuntimePolicy>,
    ) -> PyResult<NativeRuntimeCompatibilityToken> {
        let token = self
            .lock_state()?
            .require_compatible_runtime_policy(&logging_policy.policy, rayon_thread_count, &jax_policy.policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeRuntimeCompatibilityToken { token })
    }

    fn build_run_runtime(&self, runtime_policy: &NativeRuntimePolicy) -> PyResult<NativeRunRuntime> {
        let runtime = self
            .lock_state()?
            .build_run_runtime(runtime_policy.policy.clone())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeRunRuntime { runtime })
    }

    fn require_compatible_runtime_policy_handle(
        &self,
        runtime_policy: &NativeRuntimePolicy,
    ) -> PyResult<NativeRuntimeCompatibilityToken> {
        let token = self
            .lock_state()?
            .require_compatible_runtime_policy_payload(&runtime_policy.policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeRuntimeCompatibilityToken { token })
    }

    fn record_logging_runtime_policy(&self, logging_policy: PyRef<'_, NativeLoggingRuntimePolicy>) -> PyResult<()> {
        self.lock_state()?.record_logging_policy(logging_policy.policy.clone());
        Ok(())
    }

    fn initialize_logging_runtime_policy(
        &self,
        py: Python<'_>,
        logging_policy: PyRef<'_, NativeLoggingRuntimePolicy>,
    ) -> PyResult<bool> {
        let log_queue_size = non_negative_i64_to_usize(logging_policy.policy.log_queue_size, "log_queue_size")?;
        let trace_event_cap =
            optional_non_negative_i64_to_usize(logging_policy.policy.trace_event_cap, "trace_event_cap")?;
        let mut state = self.lock_state()?;
        state
            .require_compatible_logging_policy(&logging_policy.policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let initialized_logging = logging::initialize_logging(
            py,
            Some(logging_policy.policy.log_filter.clone()),
            logging_policy.policy.log_file.clone(),
            logging_policy.policy.log_stderr,
            log_queue_size,
            logging_policy.policy.log_lossy,
            logging_policy.policy.include_source_location,
            logging_policy.policy.include_span_events,
            logging_policy.policy.trace_file.clone(),
            Some(logging_policy.policy.trace_filter.clone()),
            trace_event_cap,
        )?;
        state.record_logging_policy(logging_policy.policy.clone());
        Ok(initialized_logging)
    }

    #[expect(clippy::unused_self, reason = "PyO3 exposes logging shutdown as a bound runtime-state operation.")]
    fn shutdown_logging_runtime(&self) -> PyResult<()> {
        logging::shutdown_logging()
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

    fn configure_runtime_knobs(
        &self,
        bgen_decode_tile_variant_count: i64,
        rayon_thread_count: Option<i64>,
    ) -> PyResult<Option<NativeRayonThreadPoolConfigurationPlan>> {
        run_events::record_native_runtime_knobs_configured_diagnostic_event(
            bgen_decode_tile_variant_count,
            rayon_thread_count,
        )?;
        let tile_variant_count =
            non_negative_i64_to_usize(bgen_decode_tile_variant_count, "bgen_decode_tile_variant_count")?;
        set_bgen_decode_tile_variant_count(tile_variant_count)
            .map_err(|error| errors::convert_bgen_error("configure_runtime_knobs", error))?;
        let Some(thread_count) = rayon_thread_count else {
            return Ok(None);
        };
        let plan = self
            .lock_state()?
            .configure_rayon_thread_pool(thread_count)
            .map_err(|error| errors::convert_rayon_thread_pool_configuration_error(&error))?;
        Ok(Some(NativeRayonThreadPoolConfigurationPlan { inner: plan }))
    }

    fn complete_jax_runtime_setup(&self, jax_policy: PyRef<'_, NativeJaxRuntimePolicy>) -> PyResult<()> {
        self.lock_state()?
            .complete_jax_runtime_setup(jax_policy.policy.clone())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn complete_jax_runtime_setup_session(
        &self,
        jax_policy: PyRef<'_, NativeJaxRuntimePolicy>,
        setup_session: &NativeJaxRuntimeSetupSession,
    ) -> PyResult<()> {
        let native_setup_session = setup_session.native_session_snapshot()?;
        self.lock_state()?
            .complete_jax_runtime_setup_session(jax_policy.policy.clone(), &native_setup_session)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn plan_jax_runtime_setup_lifecycle(
        &self,
        jax_policy: PyRef<'_, NativeJaxRuntimePolicy>,
    ) -> PyResult<NativeJaxRuntimeSetupLifecyclePlan> {
        let plan = self
            .lock_state()?
            .plan_jax_runtime_setup_lifecycle(&jax_policy.policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeJaxRuntimeSetupLifecyclePlan { inner: plan })
    }

    #[allow(clippy::needless_pass_by_value)]
    fn build_jax_runtime_setup_session(
        &self,
        jax_policy: PyRef<'_, NativeJaxRuntimePolicy>,
        resolved_cache_directory: String,
    ) -> PyResult<NativeJaxRuntimeSetupSession> {
        let session = self
            .lock_state()?
            .build_jax_runtime_setup_session(&jax_policy.policy, &resolved_cache_directory)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeJaxRuntimeSetupSession::from_session(session))
    }

    fn build_jax_runtime_setup_session_resolving_cache_directory(
        &self,
        jax_policy: PyRef<'_, NativeJaxRuntimePolicy>,
    ) -> PyResult<NativeJaxRuntimeSetupSession> {
        let session = self
            .lock_state()?
            .build_jax_runtime_setup_session_resolving_cache_directory(&jax_policy.policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(NativeJaxRuntimeSetupSession::from_session(session))
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

pub(crate) fn initialize_process_logging_runtime_policy(
    py: Python<'_>,
    logging_policy: native_runtime_policy::LoggingRuntimePolicyPayload,
) -> PyResult<bool> {
    let log_queue_size = non_negative_i64_to_usize(logging_policy.log_queue_size, "log_queue_size")?;
    let trace_event_cap = optional_non_negative_i64_to_usize(logging_policy.trace_event_cap, "trace_event_cap")?;
    let runtime_state = global_process_runtime_state();
    let mut state = runtime_state.lock_state()?;
    state
        .require_compatible_logging_policy(&logging_policy)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let initialized_logging = logging::initialize_logging(
        py,
        Some(logging_policy.log_filter.clone()),
        logging_policy.log_file.clone(),
        logging_policy.log_stderr,
        log_queue_size,
        logging_policy.log_lossy,
        logging_policy.include_source_location,
        logging_policy.include_span_events,
        logging_policy.trace_file.clone(),
        Some(logging_policy.trace_filter.clone()),
        trace_event_cap,
    )?;
    state.record_logging_policy(logging_policy);
    Ok(initialized_logging)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeJaxRuntimePolicy>()?;
    module.add_class::<NativeJaxRuntimeSetupLifecyclePlan>()?;
    module.add_class::<NativeLoggingRuntimePolicy>()?;
    module.add_class::<NativeRayonThreadPoolConfigurationPlan>()?;
    module.add_class::<NativeRunRuntime>()?;
    module.add_class::<NativeRuntimeCompatibilityToken>()?;
    module.add_class::<NativeRuntimePolicy>()?;
    module.add_class::<NativeRuntimeState>()?;
    Ok(())
}

fn non_negative_i64_to_usize(value: i64, field_name: &str) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| PyValueError::new_err(format!("{field_name} must be non-negative.")))
}

fn optional_non_negative_i64_to_usize(value: Option<i64>, field_name: &str) -> PyResult<Option<usize>> {
    value.map(|inner_value| non_negative_i64_to_usize(inner_value, field_name)).transpose()
}
