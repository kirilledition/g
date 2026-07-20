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

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex, OnceLock};

    use g_engine::RunHooks;
    use g_interface::CompiledCliRun;

    use super::{
        HostRunHooks, RunnerProcessRuntimeState, check_interruption, completed_terminal_result, configure_jax_runtime,
        failed_terminal_result, interrupted_terminal_result, lock_runtime_state, preflight_compiled_runs, run_cli,
        run_compiled_cli, run_compiled_runs, runtime_close_failure_result, terminal_result_from_error,
    };
    use crate::jax_runtime::{JaxRuntimeSetupSession, build_jax_runtime_policy};
    use crate::test_support::{BackendPlanKind, TemporaryRunFixture, TestErrorKind, TestHostError, TestNativeRunHost};
    use crate::{CliRunResult, NativeRunInterruption};

    static LIFECYCLE_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn compiled_run(run_plan: g_plan::RunPlan) -> CompiledCliRun {
        CompiledCliRun { run_plan, effective_config_toml: "[test]\nfixture = true\n".to_string() }
    }

    #[test]
    fn frontend_exit_bypasses_native_host() {
        let mut host = TestNativeRunHost::default();
        let output = run_cli(&["--help".to_string()], &mut host).expect("help dispatch should succeed");
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout_chunks.concat().contains("Usage: g"));
        assert!(output.stderr_chunks.is_empty());
        assert!(host.calls.is_empty());

        let missing_command_output = run_cli(&[], &mut host).expect("missing command dispatch should succeed");
        assert_eq!(missing_command_output.exit_code, 2);
        assert!(missing_command_output.stdout_chunks.concat().contains("Usage: g"));
        assert!(host.calls.is_empty());
    }

    #[test]
    fn preflight_rejects_empty_run_collection() {
        let mut host = TestNativeRunHost::default();
        let error = preflight_compiled_runs(&[], &mut host).expect_err("empty batch should fail");
        assert_eq!(error.message, "Execution requires at least one compiled run.");
    }

    #[test]
    fn preflight_rejects_process_global_thread_and_logging_conflicts() {
        let first_plan = crate::test_support::run_plan(Path::new("first-run"), g_plan::AssociationMode::Regenie2Linear);
        let mut second_plan =
            crate::test_support::run_plan(Path::new("second-run"), g_plan::AssociationMode::Regenie2Linear);
        second_plan.compute.cpu_thread_count = Some(4);
        let mut host = TestNativeRunHost::default();
        let thread_error = preflight_compiled_runs(&[compiled_run(first_plan), compiled_run(second_plan)], &mut host)
            .expect_err("thread conflict should fail");
        assert!(thread_error.message.contains("Run 1 requested Rayon threads=automatic"));
        assert!(thread_error.message.contains("run 2 requested Rayon threads=4"));

        let first_plan = crate::test_support::run_plan(Path::new("first-run"), g_plan::AssociationMode::Regenie2Linear);
        let mut second_plan =
            crate::test_support::run_plan(Path::new("second-run"), g_plan::AssociationMode::Regenie2Linear);
        second_plan.telemetry = g_plan::TelemetryMode::Profile;
        let logging_error = preflight_compiled_runs(&[compiled_run(first_plan), compiled_run(second_plan)], &mut host)
            .expect_err("logging conflict should fail");
        assert!(logging_error.message.contains("Process-global logging policies differ"));
    }

    #[test]
    fn preflight_rejects_process_global_jax_conflicts() {
        let mut first_plan =
            crate::test_support::run_plan(Path::new("first-run"), g_plan::AssociationMode::Regenie2Linear);
        first_plan.compute.jax_cache_directory = Some("first-cache".to_string());
        let mut second_plan =
            crate::test_support::run_plan(Path::new("second-run"), g_plan::AssociationMode::Regenie2Linear);
        second_plan.compute.jax_cache_directory = Some("second-cache".to_string());
        let mut host = TestNativeRunHost::default();
        let cache_error = preflight_compiled_runs(&[compiled_run(first_plan), compiled_run(second_plan)], &mut host)
            .expect_err("cache conflict should fail");
        assert!(cache_error.message.contains("incompatible process-global JAX settings"));
        assert!(cache_error.message.contains("Run 1 and run 2"));

        let first_plan = crate::test_support::run_plan(Path::new("first-run"), g_plan::AssociationMode::Regenie2Linear);
        let mut second_plan =
            crate::test_support::run_plan(Path::new("second-run"), g_plan::AssociationMode::Regenie2Linear);
        second_plan.compute.device = g_plan::Device::Gpu;
        let device_error = preflight_compiled_runs(&[compiled_run(first_plan), compiled_run(second_plan)], &mut host)
            .expect_err("device conflict should fail");
        assert!(device_error.message.contains("device=cpu"));
        assert!(device_error.message.contains("device=gpu"));
    }

    #[test]
    fn compiled_run_collection_reports_preflight_failure_as_terminal_output() {
        let mut host = TestNativeRunHost::default();
        let output = run_compiled_runs(Vec::new(), &mut host).expect("preflight failure should become CLI output");
        assert_eq!(output.exit_code, 1);
        assert_eq!(output.stderr_chunks, ["Error: Execution requires at least one compiled run.\n"]);
    }

    #[test]
    fn host_hooks_forward_interruption_and_signal_classification() {
        let interruption = TestHostError::sigint("stop");
        let mut host = TestNativeRunHost {
            interruption_results: [Err(interruption.clone())].into(),
            ..TestNativeRunHost::default()
        };
        let mut hooks = HostRunHooks { host: &mut host };
        assert_eq!(hooks.check_interruption(), Err(interruption.clone()));
        assert_eq!(
            <HostRunHooks<'_, TestNativeRunHost> as RunHooks>::interruption_signal_name(&interruption),
            Some("SIGINT")
        );
        assert_eq!(host.calls, ["check_interruption"]);
    }

    #[test]
    fn direct_interruption_check_preserves_host_error() {
        let interruption = TestHostError::sigint("pending signal");
        let mut host = TestNativeRunHost {
            interruption_results: [Err(interruption.clone())].into(),
            ..TestNativeRunHost::default()
        };
        assert_eq!(check_interruption(&mut host), Err(interruption));
    }

    #[test]
    fn terminal_results_distinguish_success_failure_and_interruptions() {
        let mut host = TestNativeRunHost::default();
        let completion = completed_terminal_result(
            &mut host,
            &[g_engine::PhenotypeRunArtifact {
                output_run_directory: "run".to_string(),
                parquet_dataset_directory: "run/parquet".to_string(),
            }],
        )
        .expect("completion output should render");
        assert_eq!(completion.exit_code, 0);
        assert_eq!(
            completion.stdout_chunks,
            ["Success. Run saved to run\n", "Parquet dataset saved to run/parquet\n",]
        );

        let failure = TestHostError::failure("backend failed");
        assert_eq!(
            failed_terminal_result(&mut host, None, &failure, None),
            CliRunResult {
                exit_code: 1,
                stdout_chunks: Vec::new(),
                stderr_chunks: vec!["Error: backend failed\n".to_string()],
            }
        );
        assert_eq!(terminal_result_from_error(&mut host, None, "thread", &failure).exit_code, 1);

        for (interruption, expected_exit_code, expected_resume_text) in [
            (NativeRunInterruption::Sigint, 130, false),
            (NativeRunInterruption::FlushedSigint, 130, true),
            (NativeRunInterruption::Sigterm, 143, true),
        ] {
            let output = interrupted_terminal_result(interruption);
            assert_eq!(output.exit_code, expected_exit_code);
            assert_eq!(
                output.stderr_chunks.concat().contains("saved committed output for resume"),
                expected_resume_text
            );
        }

        let sigint_error = TestHostError::sigint("interrupted");
        assert_eq!(terminal_result_from_error(&mut host, None, "thread", &sigint_error).exit_code, 130);
    }

    #[test]
    fn runtime_close_failure_never_overwrites_an_existing_failure() {
        let close_failure = runtime_close_failure_result(0, "TelemetryRunError", "flush failed");
        assert_eq!(close_failure.exit_code, 1);
        assert_eq!(close_failure.stderr_chunks, ["Error: flush failed\n"]);
        assert_eq!(
            runtime_close_failure_result(130, "TelemetryRunError", "flush failed"),
            CliRunResult { exit_code: 130, ..CliRunResult::default() }
        );
    }

    #[test]
    fn jax_configuration_skips_reconfiguration_and_applies_cpu_policy_once() {
        let mut run_plan =
            crate::test_support::run_plan(Path::new("jax-configuration"), g_plan::AssociationMode::Regenie2Linear);
        run_plan.compute.jax_cache_directory = Some("runner-cache".to_string());
        let policy = build_jax_runtime_policy(&run_plan).expect("test JAX policy should build");
        let mut host = TestNativeRunHost::default();
        let mut skipped_session = JaxRuntimeSetupSession::new(false, &policy);
        configure_jax_runtime(
            &mut host,
            &mut skipped_session,
            &g_runtime::TelemetryRunSession::default(),
            "test-thread",
        )
        .expect("configured runtime should be reused");
        assert!(host.calls.is_empty());

        let mut setup_session = JaxRuntimeSetupSession::new(true, &policy);
        configure_jax_runtime(&mut host, &mut setup_session, &g_runtime::TelemetryRunSession::default(), "test-thread")
            .expect("CPU runtime should configure");
        assert_eq!(host.calls, ["apply_jax_config_updates"]);
        assert_eq!(host.config_update_names.len(), 7);
    }

    #[test]
    fn invalid_bgen_lifecycle_configures_fake_backend_and_translates_engine_failure() {
        let lifecycle_lock = LIFECYCLE_TEST_LOCK.get_or_init(|| Mutex::new(())).lock().expect("test lock should open");
        let fixture = TemporaryRunFixture::new();
        let mut run_plan = fixture.run_plan(g_plan::AssociationMode::Regenie2Linear);
        run_plan.compute.jax_cache_directory = Some(fixture.root_path().join("jax-cache").display().to_string());
        let mut host = TestNativeRunHost::default();
        let output = run_compiled_cli(run_plan, "[test]\nfixture = true\n".to_string(), &mut host)
            .expect("engine failure should become terminal output");
        drop(lifecycle_lock);

        assert_eq!(output.exit_code, 1);
        let error_text = output.stderr_chunks.concat();
        assert!(error_text.contains("Unexpected end of file while reading BGEN bytes"));
        assert_eq!(host.backend_plan_kinds, [BackendPlanKind::Linear]);
        assert_eq!(host.config_update_names.len(), 7);
        assert!(host.calls.starts_with(&[
            "current_thread_name",
            "install_python_logging",
            "apply_jax_config_updates",
            "create_backend",
        ]));
        assert_eq!(host.calls.last(), Some(&"check_interruption"));
        assert!(!host.calls.contains(&"observe_jax_devices"));
    }

    #[test]
    fn thread_name_failure_escapes_before_runtime_or_backend_setup() {
        let lifecycle_lock = LIFECYCLE_TEST_LOCK.get_or_init(|| Mutex::new(())).lock().expect("test lock should open");
        let fixture = TemporaryRunFixture::new();
        let expected_error = TestHostError::failure("thread unavailable");
        let mut host =
            TestNativeRunHost { current_thread_error: Some(expected_error.clone()), ..TestNativeRunHost::default() };
        let result =
            run_compiled_cli(fixture.run_plan(g_plan::AssociationMode::Regenie2Linear), String::new(), &mut host);
        drop(lifecycle_lock);
        assert_eq!(result, Err(expected_error));
        assert_eq!(host.calls, ["current_thread_name"]);
    }

    #[test]
    fn poisoned_runtime_mutex_is_reported_without_panicking() {
        let runtime_state = Arc::new(Mutex::new(RunnerProcessRuntimeState::default()));
        let poisoned_state = Arc::clone(&runtime_state);
        let _ = std::thread::spawn(move || {
            let _guard = poisoned_state.lock().expect("test mutex should initially lock");
            panic!("poison runner test mutex");
        })
        .join();
        let error = match lock_runtime_state(&runtime_state) {
            Ok(_guard) => panic!("poisoned mutex should fail"),
            Err(error) => error,
        };
        assert_eq!(error, "Runtime state mutex was poisoned.");
    }

    #[test]
    fn host_error_classification_covers_all_error_kinds() {
        let mut host = TestNativeRunHost::default();
        let sigterm_error = TestHostError { kind: TestErrorKind::Sigterm, message: "term".to_string() };
        let flushed_error = TestHostError { kind: TestErrorKind::FlushedSigint, message: "flushed".to_string() };
        assert_eq!(terminal_result_from_error(&mut host, None, "thread", &sigterm_error).exit_code, 143);
        assert_eq!(terminal_result_from_error(&mut host, None, "thread", &flushed_error).exit_code, 130);
    }
}
