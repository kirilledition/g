//! Native process runtime configuration used by the CLI.

use std::sync::{Mutex, MutexGuard, OnceLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use g_runtime as native_runtime;

use super::{errors, jax_runtime, logging, run_events};

static GLOBAL_PROCESS_RUNTIME_STATE: OnceLock<Mutex<native_runtime::ProcessRuntimeState>> = OnceLock::new();

pub(crate) fn configure_cli_process_runtime(
    py: Python<'_>,
    run_plan: &g_plan::RunPlan,
    logging_policy: &native_runtime::LoggingRuntimePolicyPayload,
    telemetry_session: Option<&native_runtime::TelemetryRunSession>,
) -> PyResult<()> {
    let bgen_decode_tile_variant_count = run_plan.compute.bgen_decode_tile_variant_count;
    let rayon_thread_count = run_plan.compute.cpu_thread_count.map(i64::from);
    let jax_policy = native_runtime::build_jax_runtime_policy_payload(run_plan)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let runtime_state = global_process_runtime_state();

    lock_runtime_state(runtime_state)?
        .require_compatible_runtime_policy(logging_policy, rayon_thread_count, &jax_policy)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    run_events::emit_run_diagnostic_event_payload(
        &native_runtime::build_native_runtime_knobs_configured_diagnostic_payload(
            i64::from(bgen_decode_tile_variant_count),
            rayon_thread_count,
        ),
    )?;
    let mut setup_session = {
        let mut state = lock_runtime_state(runtime_state)?;
        if let Some(thread_count) = rayon_thread_count {
            state
                .configure_rayon_thread_pool(thread_count)
                .map_err(|error| errors::convert_rayon_thread_pool_configuration_error(&error))?;
        }
        state
            .build_jax_runtime_setup_session_resolving_cache_directory(&jax_policy)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
    };
    let should_configure_jax = setup_session.should_configure();
    jax_runtime::configure_jax_runtime_before_backend_init(py, &mut setup_session, telemetry_session)?;
    let mut state = lock_runtime_state(runtime_state)?;
    state
        .require_compatible_runtime_policy(logging_policy, rayon_thread_count, &jax_policy)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    if should_configure_jax {
        state
            .complete_jax_runtime_setup_session(jax_policy, &setup_session)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    }
    Ok(())
}

pub(crate) fn initialize_process_logging_runtime_policy(
    py: Python<'_>,
    logging_policy: native_runtime::LoggingRuntimePolicyPayload,
) -> PyResult<bool> {
    let log_queue_size = non_negative_i64_to_usize(logging_policy.log_queue_size, "log_queue_size")?;
    let trace_event_cap =
        logging_policy.trace_event_cap.map(|value| non_negative_i64_to_usize(value, "trace_event_cap")).transpose()?;
    let runtime_state = global_process_runtime_state();
    let mut state = lock_runtime_state(runtime_state)?;
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

fn non_negative_i64_to_usize(value: i64, field_name: &str) -> PyResult<usize> {
    if value < 0 {
        return Err(PyValueError::new_err(format!("{field_name} must be non-negative. Observed {value}.")));
    }
    usize::try_from(value).map_err(|_| PyValueError::new_err(format!("{field_name} does not fit into native usize.")))
}

fn global_process_runtime_state() -> &'static Mutex<native_runtime::ProcessRuntimeState> {
    GLOBAL_PROCESS_RUNTIME_STATE.get_or_init(|| Mutex::new(native_runtime::ProcessRuntimeState::default()))
}

fn lock_runtime_state(
    runtime_state: &Mutex<native_runtime::ProcessRuntimeState>,
) -> PyResult<MutexGuard<'_, native_runtime::ProcessRuntimeState>> {
    runtime_state.lock().map_err(|_| PyRuntimeError::new_err("Runtime state mutex was poisoned."))
}
