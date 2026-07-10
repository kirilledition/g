//! Native process runtime configuration used by the CLI.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use g_genotype::set_bgen_decode_tile_variant_count;
use g_interface as interface;
use g_runtime as native_runtime;

use crate::binding::convert::int::{non_negative_i64_to_usize, optional_non_negative_i64_to_usize};

use super::{errors, jax_runtime, logging, run_events, telemetry_session};

static GLOBAL_PROCESS_RUNTIME_STATE: OnceLock<Mutex<native_runtime::ProcessRuntimeState>> = OnceLock::new();

pub(crate) fn configure_cli_process_runtime(
    py: Python<'_>,
    config: &interface::RegenieConfigData,
    logging_policy: &native_runtime::LoggingRuntimePolicyPayload,
    telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
) -> PyResult<()> {
    let bgen_decode_tile_variant_count = config.g_compute.bgen_decode_tile_variant_count.get();
    let rayon_thread_count = config.trait_config.threads.map(|thread_count| i64::from(thread_count.get()));
    let jax_policy = build_cli_jax_runtime_policy(config)?;
    let runtime_policy = native_runtime::RuntimePolicyPayload {
        logging_policy: logging_policy.clone(),
        rayon_thread_count,
        jax_policy: jax_policy.clone(),
    };
    let runtime_state = global_process_runtime_state();

    lock_runtime_state(runtime_state)?
        .require_compatible_runtime_policy_payload(&runtime_policy)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    run_events::record_native_runtime_knobs_configured_diagnostic_event(
        i64::from(bgen_decode_tile_variant_count),
        rayon_thread_count,
    )?;
    let tile_variant_count = usize::try_from(bgen_decode_tile_variant_count)
        .map_err(|_| PyValueError::new_err("bgen_decode_tile_variant_count does not fit into usize."))?;
    set_bgen_decode_tile_variant_count(tile_variant_count)
        .map_err(|error| errors::convert_bgen_error("configure_cli_process_runtime", error))?;

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
    if should_configure_jax {
        lock_runtime_state(runtime_state)?
            .complete_jax_runtime_setup_session(jax_policy, &setup_session)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    }

    lock_runtime_state(runtime_state)?
        .require_compatible_runtime_policy_payload(&runtime_policy)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(())
}

pub(crate) fn initialize_process_logging_runtime_policy(
    py: Python<'_>,
    logging_policy: native_runtime::LoggingRuntimePolicyPayload,
) -> PyResult<bool> {
    let log_queue_size = non_negative_i64_to_usize(logging_policy.log_queue_size, "log_queue_size")?;
    let trace_event_cap = optional_non_negative_i64_to_usize(logging_policy.trace_event_cap, "trace_event_cap")?;
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

fn global_process_runtime_state() -> &'static Mutex<native_runtime::ProcessRuntimeState> {
    GLOBAL_PROCESS_RUNTIME_STATE.get_or_init(|| Mutex::new(native_runtime::ProcessRuntimeState::default()))
}

fn build_cli_jax_runtime_policy(
    config: &interface::RegenieConfigData,
) -> PyResult<native_runtime::JaxRuntimePolicyPayload> {
    let compute_config = &config.g_compute;
    let cache_directory = compute_config.jax_cache_dir.as_deref().map(expand_home_directory).transpose()?;
    Ok(native_runtime::build_jax_runtime_policy_payload(
        compute_config.device.as_str(),
        cache_directory.as_deref(),
        compute_config.jax_matmul_precision.map(g_interface::JaxMatmulPrecisionValue::as_str),
        compute_config.jax_persistent_cache,
        compute_config.jax_persistent_cache_min_entry_size_bytes,
        i64::from(compute_config.jax_persistent_cache_min_compile_time_seconds),
        compute_config.jax_xla_autotune_cache,
        compute_config.jax_transfer_guard,
    ))
}

fn expand_home_directory(path_value: &str) -> PyResult<String> {
    if path_value == "~" {
        return home_directory().map(|home_directory| home_directory.to_string_lossy().into_owned());
    }
    let Some(relative_path) = path_value.strip_prefix("~/") else {
        return Ok(path_value.to_string());
    };
    Ok(home_directory()?.join(Path::new(relative_path)).to_string_lossy().into_owned())
}

fn home_directory() -> PyResult<PathBuf> {
    std::env::var_os("HOME")
        .filter(|home_directory| !home_directory.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| PyValueError::new_err("Cannot expand jax_cache_dir because HOME is not set."))
}

fn lock_runtime_state(
    runtime_state: &Mutex<native_runtime::ProcessRuntimeState>,
) -> PyResult<MutexGuard<'_, native_runtime::ProcessRuntimeState>> {
    runtime_state.lock().map_err(|_| PyRuntimeError::new_err("Runtime state mutex was poisoned."))
}
