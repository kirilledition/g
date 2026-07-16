//! CLI dispatch and native run lifecycle coordination.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use g_engine::{AssociationBackend, RunHooks};
use g_interface::{CliDispatch, CompiledCliRun};
use g_runtime::{NativeRunSession, NativeRunSessionPolicy, ProcessRuntimeState, TelemetryRunSession};
use serde::Serialize;

use crate::backend_plan::JaxAssociationBackendPlan;
use crate::cli_output::{CliRunResult, render_completed_lines, render_failed_lines, render_interrupted_lines};
use crate::jax_runtime::{
    JaxDevice, JaxRuntimeConfigUpdate, JaxRuntimeSetupSession, JaxRuntimeState, build_jax_runtime_policy,
    emit_jax_runtime_setup_diagnostics, nvidia_driver_files_are_visible, plan_jax_gpu_validation,
    plan_jax_runtime_config_updates,
};
use crate::native_session_policy::project_native_run_session_policy;

static GLOBAL_PROCESS_RUNTIME_STATE: OnceLock<Mutex<RunnerProcessRuntimeState>> = OnceLock::new();

const CLI_RUNTIME_FAILURE_EXIT_CODE: i32 = 1;
const RUN_FAILED_EVENT_NAME: &str = "run_failed";

#[derive(Serialize)]
struct RunFailedTelemetryFields<'fields> {
    failure_kind: &'static str,
    error_type: &'fields str,
    error_message: &'fields str,
}

#[derive(Serialize)]
struct NativeRuntimeKnobsDiagnosticFields {
    threads: Option<i64>,
}

#[derive(Serialize)]
struct TerminalLineDiagnosticFields<'line> {
    line: &'line str,
}

/// Process interruption classified at the Python boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeRunInterruption {
    /// The native SIGTERM watcher requested a resumable shutdown.
    Sigterm,
    /// Python delivered SIGINT before output was flushed.
    Sigint,
    /// Python delivered SIGINT after resumable output was flushed.
    FlushedSigint,
}

/// Python-host error details rendered by the Rust terminal lifecycle.
#[derive(Debug, Eq, PartialEq)]
pub struct NativeRunFailure {
    pub error_type: String,
    pub error_message: String,
}

#[derive(Default)]
struct RunnerProcessRuntimeState {
    native: ProcessRuntimeState,
    jax: JaxRuntimeState,
}

/// Operations the Python host must perform while Rust owns the run lifecycle.
pub trait NativeRunHost: Send {
    /// Concrete association backend retaining opaque Python state.
    type Backend: AssociationBackend + 'static;
    /// Python-boundary error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Install the Python logging bridge after Rust initialized logging sinks.
    ///
    /// # Errors
    ///
    /// Returns the host error when the bridge cannot be installed.
    fn install_python_logging(&mut self) -> Result<(), Self::Error>;

    /// Apply validated JAX configuration updates through the Python runtime.
    ///
    /// # Errors
    ///
    /// Returns the host error when JAX rejects an update.
    fn apply_jax_config_updates(&mut self, updates: &[JaxRuntimeConfigUpdate<'_>]) -> Result<(), Self::Error>;

    /// Return Python-owned JAX device observations for GPU validation.
    ///
    /// # Errors
    ///
    /// Returns the host error when JAX device discovery fails.
    fn observe_jax_devices(&mut self) -> Result<Vec<JaxDevice>, Self::Error>;

    /// Construct the opaque JAX association backend from canonical device and
    /// mode-specific policy.
    ///
    /// # Errors
    ///
    /// Returns the host error when backend construction fails.
    fn create_backend(
        &mut self,
        device: g_plan::Device,
        plan: JaxAssociationBackendPlan<'_>,
    ) -> Result<Arc<Self::Backend>, Self::Error>;

    /// Check Python signals.
    ///
    /// # Errors
    ///
    /// Returns the host interruption error when a signal is pending.
    fn check_interruption(&mut self) -> Result<(), Self::Error>;

    /// Construct the Python-host interruption used for a native SIGTERM request.
    fn sigterm_interruption_error(&mut self) -> Self::Error;

    /// Mark an interrupted run whose resumable output was successfully flushed.
    fn flushed_interruption_error(&mut self, error: Self::Error) -> Self::Error;

    /// Name the signal associated with an engine interruption, when known.
    fn interruption_signal_name(error: &Self::Error) -> Option<&str>;

    /// Classify a terminal Python-boundary error as a resumable interruption.
    fn interruption_kind(&mut self, error: &Self::Error) -> Option<NativeRunInterruption>;

    /// Convert a Python-free run failure into the host error type.
    fn run_error(&mut self, message: String) -> Self::Error;

    /// Convert a host error into the terminal telemetry payload.
    fn failed_event(&mut self, error: &Self::Error) -> NativeRunFailure;

    /// Read the current Python thread name for telemetry labels.
    ///
    /// # Errors
    ///
    /// Returns the host error when the thread name cannot be observed.
    fn current_thread_name(&mut self) -> Result<String, Self::Error>;

    /// Release the Python interpreter while executing CPU-bound Rust work.
    fn detach<T, Operation>(operation: Operation) -> T
    where
        T: Send,
        Operation: FnOnce() -> T + Send;
}

struct HostRunHooks<'host, Host> {
    host: &'host mut Host,
}

impl<Host> RunHooks for HostRunHooks<'_, Host>
where
    Host: NativeRunHost,
{
    type Error = Host::Error;

    fn check_interruption(&mut self) -> Result<(), Self::Error> {
        check_interruption(self.host)
    }

    fn interruption_signal_name(error: &Self::Error) -> Option<&str> {
        Host::interruption_signal_name(error)
    }
}

/// Dispatch a CLI invocation and run it through the Python host adapter.
///
/// # Errors
///
/// Returns only failures that prevent reading the Python thread name. Normal
/// run failures are rendered into the returned terminal result.
pub fn run_cli<Host>(arguments: &[String], host: &mut Host) -> Result<CliRunResult, Host::Error>
where
    Host: NativeRunHost,
    Host::Backend: AssociationBackend + 'static,
{
    match g_interface::dispatch_cli(arguments) {
        CliDispatch::Exit { exit_code, stdout, stderr } => {
            Ok(CliRunResult::from_frontend_output(exit_code, &stdout, &stderr))
        }
        CliDispatch::Runs(compiled_runs) => run_compiled_runs(compiled_runs, host),
    }
}

fn run_compiled_runs<Host>(mut compiled_runs: Vec<CompiledCliRun>, host: &mut Host) -> Result<CliRunResult, Host::Error>
where
    Host: NativeRunHost,
    Host::Backend: AssociationBackend + 'static,
{
    if compiled_runs.len() == 1 {
        let compiled_run = compiled_runs.pop().expect("one compiled run was checked");
        return run_compiled_cli(compiled_run.run_plan, compiled_run.effective_config_toml, host);
    }
    if let Err(error) = preflight_compiled_runs(&compiled_runs, host) {
        return Ok(failed_terminal_result(host, None, &error, None));
    }
    let mut output = CliRunResult::default();
    for (run_index, compiled_run) in compiled_runs.into_iter().enumerate() {
        let run_result = run_compiled_cli(compiled_run.run_plan, compiled_run.effective_config_toml, host)?;
        let run_failed = run_result.exit_code != 0;
        output.append(run_result);
        if run_failed {
            output.stderr_chunks.push(format!("Run {} stopped.\n", run_index + 1));
            break;
        }
    }
    Ok(output)
}

fn preflight_compiled_runs<Host>(compiled_runs: &[CompiledCliRun], host: &mut Host) -> Result<(), Host::Error>
where
    Host: NativeRunHost,
{
    if compiled_runs.is_empty() {
        return Err(host.run_error("Execution requires at least one compiled run.".to_string()));
    }
    let native_session_policies = compiled_runs
        .iter()
        .map(|compiled_run| project_native_run_session_policy(&compiled_run.run_plan))
        .collect::<Vec<_>>();
    let mut native_policies = compiled_runs.iter().zip(&native_session_policies);
    let Some((first_compiled_run, first_native_session_policy)) = native_policies.next() else {
        return Err(host.run_error("Execution requires at least one native runtime policy.".to_string()));
    };
    let first_rayon_thread_count = first_compiled_run.run_plan.compute.cpu_thread_count.map(i64::from);
    for (run_index, (compiled_run, native_session_policy)) in native_policies.enumerate() {
        first_native_session_policy
            .require_compatible_process_logging_policy(native_session_policy)
            .map_err(|error| host.run_error(format!("Run 1 and run {}: {error}", run_index + 2)))?;
        let rayon_thread_count = compiled_run.run_plan.compute.cpu_thread_count.map(i64::from);
        if first_rayon_thread_count != rayon_thread_count {
            let first_thread_description = first_rayon_thread_count
                .map_or_else(|| "automatic".to_string(), |thread_count| thread_count.to_string());
            let requested_thread_description =
                rayon_thread_count.map_or_else(|| "automatic".to_string(), |thread_count| thread_count.to_string());
            return Err(host.run_error(format!(
                "Run 1 requested Rayon threads={first_thread_description}, but run {} requested Rayon \
                 threads={requested_thread_description}. Rayon thread policy is process-global.",
                run_index + 2,
            )));
        }
    }

    let mut jax_policies = Vec::with_capacity(compiled_runs.len());
    for (run_index, compiled_run) in compiled_runs.iter().enumerate() {
        let jax_policy = build_jax_runtime_policy(&compiled_run.run_plan)
            .map_err(|error| host.run_error(format!("Run {}: {error}", run_index + 1)))?;
        jax_policies.push(jax_policy);
    }
    JaxRuntimeState::require_mutually_compatible(&jax_policies).map_err(|error| host.run_error(error.to_string()))?;

    let runtime_state = global_process_runtime_state();
    let state = lock_runtime_state(runtime_state).map_err(|message| host.run_error(message))?;
    for (run_index, ((compiled_run, native_session_policy), jax_policy)) in
        compiled_runs.iter().zip(&native_session_policies).zip(&jax_policies).enumerate()
    {
        let rayon_thread_count = compiled_run.run_plan.compute.cpu_thread_count.map(i64::from);
        state
            .native
            .require_compatible_runtime_policy(native_session_policy, rayon_thread_count)
            .map_err(|error| host.run_error(format!("Run {}: {error}", run_index + 1)))?;
        state
            .jax
            .require_compatible(jax_policy)
            .map_err(|error| host.run_error(format!("Run {}: {error}", run_index + 1)))?;
    }
    Ok(())
}

fn run_compiled_cli<Host>(
    run_plan: g_plan::RunPlan,
    effective_config_toml: String,
    host: &mut Host,
) -> Result<CliRunResult, Host::Error>
where
    Host: NativeRunHost,
    Host::Backend: AssociationBackend + 'static,
{
    let mut output = CliRunResult::default();
    let native_session_policy = project_native_run_session_policy(&run_plan);
    let mut native_session = match open_compatible_native_run_session(host, native_session_policy) {
        Ok(session) => session,
        Err(error) => {
            let terminal_result = failed_terminal_result(host, None, &error, None);
            output.append(terminal_result);
            return Ok(output);
        }
    };
    let thread_name = host.current_thread_name()?;
    let mut execution_result = (|| {
        host.install_python_logging()?;
        let runtime_start_time = Instant::now();
        configure_process_runtime(
            host,
            &run_plan,
            native_session.policy(),
            native_session.telemetry_session(),
            &thread_name,
        )?;
        native_session.record_stage_duration("jax_runtime_configuration", runtime_start_time);

        let backend_start_time = Instant::now();
        let backend =
            host.create_backend(run_plan.compute.device, JaxAssociationBackendPlan::from_run_plan(&run_plan))?;
        native_session.record_stage_duration("jax_backend_initialization", backend_start_time);

        let telemetry_session = native_session.telemetry_session().clone();
        let stage_timing_recorder = native_session.stage_timing_recorder();
        let thread_name_for_run = thread_name.as_str();
        let execution_result = Host::detach(|| {
            let mut hooks = HostRunHooks { host };
            g_engine::execute_coordinated_run(
                run_plan,
                effective_config_toml,
                backend,
                &mut hooks,
                &telemetry_session,
                thread_name_for_run,
                stage_timing_recorder,
            )
        });
        execution_result.map_err(|error| match error {
            g_engine::EngineRunError::Interrupted(error) => host.flushed_interruption_error(error),
            g_engine::EngineRunError::Failure { message } => host.run_error(message),
        })
    })();
    let timing_result = native_session.finish_timing().map_err(|error| host.run_error(error.to_string()));
    if execution_result.is_ok()
        && let Err(error) = timing_result
    {
        execution_result = Err(error);
    }
    if execution_result.is_ok()
        && let Err(error) = check_interruption(host)
    {
        execution_result = Err(error);
    }
    let mut terminal_result = match execution_result {
        Ok(artifacts) => completed_terminal_result(host, &artifacts)?,
        Err(error) => terminal_result_from_error(host, Some(native_session.telemetry_session()), &thread_name, &error),
    };
    let mut close_result =
        finish_telemetry_result(native_session.telemetry_session(), &thread_name, terminal_result.exit_code);
    let mut logging_result = match native_session.finish_logging() {
        Ok(()) => CliRunResult { exit_code: close_result.exit_code, ..CliRunResult::default() },
        Err(error) => runtime_close_failure_result(close_result.exit_code, "LoggingSinkError", &error.to_string()),
    };
    if let Err(error) = check_interruption(host) {
        terminal_result = terminal_result_from_error(host, None, &thread_name, &error);
        close_result = CliRunResult { exit_code: terminal_result.exit_code, ..CliRunResult::default() };
        logging_result = CliRunResult { exit_code: terminal_result.exit_code, ..CliRunResult::default() };
    }
    output.append(terminal_result);
    output.append(close_result);
    output.append(logging_result);
    Ok(output)
}

fn check_interruption<Host>(host: &mut Host) -> Result<(), Host::Error>
where
    Host: NativeRunHost,
{
    host.check_interruption()?;
    if g_runtime::sigterm_shutdown_requested() {
        return Err(host.sigterm_interruption_error());
    }
    Ok(())
}

fn open_compatible_native_run_session<Host>(
    host: &mut Host,
    policy: NativeRunSessionPolicy,
) -> Result<NativeRunSession, Host::Error>
where
    Host: NativeRunHost,
{
    let runtime_state = global_process_runtime_state();
    let mut state = lock_runtime_state(runtime_state).map_err(|message| host.run_error(message))?;
    NativeRunSession::new(&mut state.native, policy).map_err(|error| host.run_error(error.to_string()))
}

fn configure_process_runtime<Host>(
    host: &mut Host,
    run_plan: &g_plan::RunPlan,
    logging_policy: &NativeRunSessionPolicy,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
) -> Result<(), Host::Error>
where
    Host: NativeRunHost,
{
    let rayon_thread_count = run_plan.compute.cpu_thread_count.map(i64::from);
    let jax_policy = build_jax_runtime_policy(run_plan).map_err(|error| host.run_error(error.to_string()))?;
    let runtime_state = global_process_runtime_state();
    {
        let state = lock_runtime_state(runtime_state).map_err(|message| host.run_error(message))?;
        state
            .native
            .require_compatible_runtime_policy(logging_policy, rayon_thread_count)
            .map_err(|error| host.run_error(error.to_string()))?;
        state.jax.require_compatible(&jax_policy).map_err(|error| host.run_error(error.to_string()))?;
    }
    g_runtime::emit_diagnostic_event(
        "debug",
        "native_runtime_knobs_configured",
        "Configuring native runtime knobs.",
        &NativeRuntimeKnobsDiagnosticFields { threads: rayon_thread_count },
    )
    .map_err(|error| host.run_error(format!("Failed to serialize runtime diagnostic event: {error}")))?;
    let setup_preparation_required = {
        let mut state = lock_runtime_state(runtime_state).map_err(|message| host.run_error(message))?;
        if let Some(thread_count) = rayon_thread_count {
            state
                .native
                .configure_rayon_thread_pool(thread_count)
                .map_err(|error| host.run_error(error.to_string()))?;
        }
        state.jax.setup_preparation_required(&jax_policy).map_err(|error| host.run_error(error.to_string()))?
    };
    if setup_preparation_required {
        jax_policy.create_cache_directory_if_configured().map_err(|error| host.run_error(error.to_string()))?;
    }
    let mut setup_session = {
        let mut state = lock_runtime_state(runtime_state).map_err(|message| host.run_error(message))?;
        state.jax.reserve_setup(&jax_policy).map_err(|error| host.run_error(error.to_string()))?
    };
    let should_configure_jax = setup_session.should_configure;
    configure_jax_runtime(host, &mut setup_session, telemetry_session, thread_name)?;
    let gpu_validation_status = setup_session.gpu_validation_status;
    let mut state = lock_runtime_state(runtime_state).map_err(|message| host.run_error(message))?;
    state
        .native
        .require_compatible_runtime_policy(logging_policy, rayon_thread_count)
        .map_err(|error| host.run_error(error.to_string()))?;
    if should_configure_jax {
        state
            .jax
            .complete_setup(jax_policy, gpu_validation_status)
            .map_err(|error| host.run_error(error.to_string()))?;
    } else {
        state.jax.require_compatible(&jax_policy).map_err(|error| host.run_error(error.to_string()))?;
    }
    Ok(())
}

fn configure_jax_runtime<Host>(
    host: &mut Host,
    setup_session: &mut JaxRuntimeSetupSession<'_>,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
) -> Result<(), Host::Error>
where
    Host: NativeRunHost,
{
    if !setup_session.should_configure {
        return Ok(());
    }
    let config_updates = plan_jax_runtime_config_updates(setup_session);
    host.apply_jax_config_updates(&config_updates)?;
    if setup_session.gpu_validation_status == crate::jax_runtime::JaxGpuValidationStatus::Pending {
        let nvidia_driver_visible = nvidia_driver_files_are_visible();
        let (backend_initialization_failed, devices) = if nvidia_driver_visible {
            match host.observe_jax_devices() {
                Ok(devices) => (false, devices),
                Err(_error) => (true, Vec::new()),
            }
        } else {
            (false, Vec::new())
        };
        let validation_plan = plan_jax_gpu_validation(nvidia_driver_visible, backend_initialization_failed, &devices);
        if validation_plan.status == crate::jax_runtime::JaxGpuValidationStatus::Failed {
            return Err(host.run_error(validation_plan.message.into_owned()));
        }
        setup_session.complete_gpu_validation(validation_plan.status, validation_plan.message);
    }
    emit_jax_runtime_setup_diagnostics(setup_session, telemetry_session, thread_name)
        .map_err(|message| host.run_error(message))
}

fn global_process_runtime_state() -> &'static Mutex<RunnerProcessRuntimeState> {
    GLOBAL_PROCESS_RUNTIME_STATE.get_or_init(|| Mutex::new(RunnerProcessRuntimeState::default()))
}

fn lock_runtime_state(
    runtime_state: &Mutex<RunnerProcessRuntimeState>,
) -> Result<MutexGuard<'_, RunnerProcessRuntimeState>, String> {
    runtime_state.lock().map_err(|_| "Runtime state mutex was poisoned.".to_string())
}

fn completed_terminal_result<Host>(
    host: &mut Host,
    artifacts: &[g_engine::PhenotypeRunArtifact],
) -> Result<CliRunResult, Host::Error>
where
    Host: NativeRunHost,
{
    let stdout_lines = render_completed_lines(artifacts);
    record_terminal_lines(&stdout_lines, "info", "native_cli_completed_line", "Native CLI completion detail.")
        .map_err(|message| host.run_error(message))?;
    Ok(CliRunResult::from_lines(0, stdout_lines, Vec::new()))
}

fn terminal_result_from_error<Host>(
    host: &mut Host,
    telemetry_session: Option<&TelemetryRunSession>,
    thread_name: &str,
    error: &Host::Error,
) -> CliRunResult
where
    Host: NativeRunHost,
{
    if let Some(interruption) = host.interruption_kind(error) {
        return interrupted_terminal_result(interruption);
    }
    failed_terminal_result(host, telemetry_session, error, Some(thread_name))
}

fn interrupted_terminal_result(interruption: NativeRunInterruption) -> CliRunResult {
    let (signal_name, exit_code, flushed_for_resume) = match interruption {
        NativeRunInterruption::Sigterm => ("SIGTERM", 143, true),
        NativeRunInterruption::Sigint => ("SIGINT", 130, false),
        NativeRunInterruption::FlushedSigint => ("SIGINT", 130, true),
    };
    let stderr_lines = render_interrupted_lines(signal_name, flushed_for_resume);
    let _ =
        record_terminal_lines(&stderr_lines, "warn", "native_cli_interrupted_line", "Native CLI interruption detail.");
    CliRunResult::from_lines(exit_code, Vec::new(), stderr_lines)
}

fn failed_terminal_result<Host>(
    host: &mut Host,
    telemetry_session: Option<&TelemetryRunSession>,
    error: &Host::Error,
    thread_name: Option<&str>,
) -> CliRunResult
where
    Host: NativeRunHost,
{
    let failure = host.failed_event(error);
    if let Some(telemetry_session) = telemetry_session
        && let Some(thread_name) = thread_name
    {
        let _ = telemetry_session.emit_current_event(
            thread_name,
            RUN_FAILED_EVENT_NAME,
            "error",
            &RunFailedTelemetryFields {
                failure_kind: "exception",
                error_type: &failure.error_type,
                error_message: &failure.error_message,
            },
        );
    }
    let stderr_lines = render_failed_lines(&failure.error_type, &failure.error_message);
    let _ = record_terminal_lines(&stderr_lines, "error", "native_cli_failed_line", "Native CLI failure detail.");
    CliRunResult::from_lines(CLI_RUNTIME_FAILURE_EXIT_CODE, Vec::new(), stderr_lines)
}

fn finish_telemetry_result(
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    current_exit_code: i32,
) -> CliRunResult {
    match telemetry_session.finish(thread_name) {
        Ok(()) => CliRunResult { exit_code: current_exit_code, ..CliRunResult::default() },
        Err(error) => runtime_close_failure_result(current_exit_code, "TelemetryRunError", &error.to_string()),
    }
}

fn runtime_close_failure_result(current_exit_code: i32, error_type: &str, error_message: &str) -> CliRunResult {
    if current_exit_code == 0 {
        let stderr_lines = render_failed_lines(error_type, error_message);
        let _ = record_terminal_lines(&stderr_lines, "error", "native_cli_failed_line", "Native CLI failure detail.");
        CliRunResult::from_lines(CLI_RUNTIME_FAILURE_EXIT_CODE, Vec::new(), stderr_lines)
    } else {
        CliRunResult { exit_code: current_exit_code, ..CliRunResult::default() }
    }
}

fn record_terminal_lines(lines: &[String], level: &str, event_name: &str, message: &str) -> Result<(), String> {
    for line in lines {
        g_runtime::emit_diagnostic_event(level, event_name, message, &TerminalLineDiagnosticFields { line })
            .map_err(|error| format!("Failed to serialize terminal diagnostic event: {error}"))?;
    }
    Ok(())
}
