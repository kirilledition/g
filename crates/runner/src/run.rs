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

enum PrimaryOutcome<Error> {
    Completed { artifacts: Vec<g_engine::PhenotypeRunArtifact> },
    Interrupted { interruption: NativeRunInterruption },
    Failed { error: Error },
}

type EngineExecutionResult<Error> = Result<Vec<g_engine::PhenotypeRunArtifact>, g_engine::EngineRunError<Error>>;
type HostExecutionResult<Error> = Result<EngineExecutionResult<Error>, Error>;

#[derive(Debug, Eq, PartialEq)]
struct AncillaryFailure {
    error_type: &'static str,
    error_message: String,
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

    /// Classify an engine interruption after resumable output was successfully flushed.
    fn flushed_interruption_kind(&mut self, error: Self::Error) -> NativeRunInterruption;

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
    let execution_result: Result<_, Host::Error> = (|| {
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
        Ok(Host::detach(|| {
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
        }))
    })();
    let primary_outcome = resolve_primary_outcome(host, execution_result);
    let mut ancillary_failure = None;
    if let Err(error) = native_session.finish_timing() {
        observe_ancillary_failure(
            &mut ancillary_failure,
            AncillaryFailure { error_type: "TimingFileError", error_message: error.to_string() },
        );
    }
    let late_interruption_observed = observe_late_interruption(host, &primary_outcome);
    let mut terminal_result = terminal_result_from_primary_outcome(
        host,
        Some(native_session.telemetry_session()),
        &thread_name,
        &primary_outcome,
        ancillary_failure.as_ref(),
    );
    if let Err(error) = native_session.telemetry_session().finish(&thread_name) {
        let failure = AncillaryFailure { error_type: "TelemetryRunError", error_message: error.to_string() };
        if observe_ancillary_failure(&mut ancillary_failure, failure) && primary_outcome.is_completed() {
            let failure = ancillary_failure.as_ref().expect("the first ancillary failure was just recorded");
            terminal_result.append(runtime_close_failure_result(0, failure.error_type, &failure.error_message));
        }
    }
    let logging_result = finish_logging_after_late_interruption_observation(
        late_interruption_observed,
        || {
            let _ = observe_late_interruption(host, &primary_outcome);
        },
        || native_session.finish_logging(),
    );
    if let Err(error) = logging_result {
        let failure = AncillaryFailure { error_type: "LoggingSinkError", error_message: error.to_string() };
        if observe_ancillary_failure(&mut ancillary_failure, failure) && primary_outcome.is_completed() {
            let failure = ancillary_failure.as_ref().expect("the first ancillary failure was just recorded");
            terminal_result.append(runtime_close_failure_result(0, failure.error_type, &failure.error_message));
        }
    }
    output.append(terminal_result);
    Ok(output)
}

fn finish_logging_after_late_interruption_observation<ObserveInterruption, FinishLogging>(
    late_interruption_observed: bool,
    observe_interruption: ObserveInterruption,
    finish_logging: FinishLogging,
) -> Result<(), g_runtime::LoggingSinkError>
where
    ObserveInterruption: FnOnce(),
    FinishLogging: FnOnce() -> Result<(), g_runtime::LoggingSinkError>,
{
    if !late_interruption_observed {
        observe_interruption();
    }
    finish_logging()
}

fn resolve_primary_outcome<Host>(
    host: &mut Host,
    execution_result: HostExecutionResult<Host::Error>,
) -> PrimaryOutcome<Host::Error>
where
    Host: NativeRunHost,
{
    match execution_result {
        Ok(Ok(artifacts)) => PrimaryOutcome::Completed { artifacts },
        Ok(Err(g_engine::EngineRunError::Interrupted(error))) => {
            PrimaryOutcome::Interrupted { interruption: host.flushed_interruption_kind(error) }
        }
        Ok(Err(g_engine::EngineRunError::Failure { message })) => {
            PrimaryOutcome::Failed { error: host.run_error(message) }
        }
        Err(error) => match host.interruption_kind(&error) {
            Some(interruption) => PrimaryOutcome::Interrupted { interruption },
            None => PrimaryOutcome::Failed { error },
        },
    }
}

impl<Error> PrimaryOutcome<Error> {
    const fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    const fn name(&self) -> &'static str {
        match self {
            Self::Completed { .. } => "completed",
            Self::Interrupted { .. } => "interrupted",
            Self::Failed { .. } => "failed",
        }
    }
}

fn observe_ancillary_failure(first_failure: &mut Option<AncillaryFailure>, failure: AncillaryFailure) -> bool {
    tracing::warn!(
        target: "g.runner",
        error_type = failure.error_type,
        error_message = failure.error_message,
        "Observed an ancillary native run failure after the primary execution outcome."
    );
    if first_failure.is_some() {
        return false;
    }
    *first_failure = Some(failure);
    true
}

fn observe_late_interruption<Host>(host: &mut Host, primary_outcome: &PrimaryOutcome<Host::Error>) -> bool
where
    Host: NativeRunHost,
{
    if let Err(error) = check_interruption(host) {
        let failure = host.failed_event(&error);
        tracing::warn!(
            target: "g.runner",
            primary_outcome = primary_outcome.name(),
            error_type = failure.error_type,
            error_message = failure.error_message,
            "Ignored an interruption observed after the primary execution outcome was fixed."
        );
        return true;
    }
    false
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
    configure_process_runtime_with_jax_diagnostics(host, run_plan, logging_policy, |setup_session| {
        emit_jax_runtime_setup_diagnostics(setup_session, telemetry_session, thread_name)
    })
}

fn configure_process_runtime_with_jax_diagnostics<Host, EmitDiagnostics>(
    host: &mut Host,
    run_plan: &g_plan::RunPlan,
    logging_policy: &NativeRunSessionPolicy,
    emit_diagnostics: EmitDiagnostics,
) -> Result<(), Host::Error>
where
    Host: NativeRunHost,
    EmitDiagnostics: FnOnce(&JaxRuntimeSetupSession<'_>) -> Result<(), String>,
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
    if let Err(error) = g_runtime::emit_diagnostic_event(
        "debug",
        "native_runtime_knobs_configured",
        "Configuring native runtime knobs.",
        &NativeRuntimeKnobsDiagnosticFields { threads: rayon_thread_count },
    ) {
        tracing::warn!(
            target: "g.runner",
            error = %error,
            "Failed to emit native runtime-knob diagnostic event."
        );
    }
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
    configure_jax_runtime(host, &mut setup_session)?;
    let gpu_validation_status = setup_session.gpu_validation_status;
    {
        let mut state = lock_runtime_state(runtime_state).map_err(|message| host.run_error(message))?;
        state
            .native
            .require_compatible_runtime_policy(logging_policy, rayon_thread_count)
            .map_err(|error| host.run_error(error.to_string()))?;
        if should_configure_jax {
            state
                .jax
                .complete_setup(jax_policy.clone(), gpu_validation_status)
                .map_err(|error| host.run_error(error.to_string()))?;
        } else {
            state.jax.require_compatible(&jax_policy).map_err(|error| host.run_error(error.to_string()))?;
        }
    }
    if should_configure_jax && let Err(error) = emit_diagnostics(&setup_session) {
        tracing::warn!(
            target: "g.runner",
            error,
            "Failed to emit completed JAX runtime setup diagnostics."
        );
    }
    Ok(())
}

fn configure_jax_runtime<Host>(
    host: &mut Host,
    setup_session: &mut JaxRuntimeSetupSession<'_>,
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
    Ok(())
}

fn global_process_runtime_state() -> &'static Mutex<RunnerProcessRuntimeState> {
    GLOBAL_PROCESS_RUNTIME_STATE.get_or_init(|| Mutex::new(RunnerProcessRuntimeState::default()))
}

fn lock_runtime_state(
    runtime_state: &Mutex<RunnerProcessRuntimeState>,
) -> Result<MutexGuard<'_, RunnerProcessRuntimeState>, String> {
    runtime_state.lock().map_err(|_| "Runtime state mutex was poisoned.".to_string())
}

fn completed_terminal_result(artifacts: &[g_engine::PhenotypeRunArtifact]) -> CliRunResult {
    let stdout_lines = render_completed_lines(artifacts);
    record_terminal_lines_best_effort(
        &stdout_lines,
        "info",
        "native_cli_completed_line",
        "Native CLI completion detail.",
    );
    CliRunResult::from_lines(0, stdout_lines, Vec::new())
}

fn terminal_result_from_primary_outcome<Host>(
    host: &mut Host,
    telemetry_session: Option<&TelemetryRunSession>,
    thread_name: &str,
    primary_outcome: &PrimaryOutcome<Host::Error>,
    ancillary_failure: Option<&AncillaryFailure>,
) -> CliRunResult
where
    Host: NativeRunHost,
{
    match primary_outcome {
        PrimaryOutcome::Completed { artifacts } => {
            let mut result = completed_terminal_result(artifacts);
            if let Some(failure) = ancillary_failure {
                result.append(runtime_close_failure_result(0, failure.error_type, &failure.error_message));
            }
            result
        }
        PrimaryOutcome::Interrupted { interruption } => interrupted_terminal_result(*interruption),
        PrimaryOutcome::Failed { error } => failed_terminal_result(host, telemetry_session, error, Some(thread_name)),
    }
}

fn interrupted_terminal_result(interruption: NativeRunInterruption) -> CliRunResult {
    let (signal_name, exit_code, flushed_for_resume) = match interruption {
        NativeRunInterruption::Sigterm => ("SIGTERM", 143, true),
        NativeRunInterruption::Sigint => ("SIGINT", 130, false),
        NativeRunInterruption::FlushedSigint => ("SIGINT", 130, true),
    };
    let stderr_lines = render_interrupted_lines(signal_name, flushed_for_resume);
    record_terminal_lines_best_effort(
        &stderr_lines,
        "warn",
        "native_cli_interrupted_line",
        "Native CLI interruption detail.",
    );
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
    record_terminal_lines_best_effort(&stderr_lines, "error", "native_cli_failed_line", "Native CLI failure detail.");
    CliRunResult::from_lines(CLI_RUNTIME_FAILURE_EXIT_CODE, Vec::new(), stderr_lines)
}

fn runtime_close_failure_result(current_exit_code: i32, error_type: &str, error_message: &str) -> CliRunResult {
    if current_exit_code == 0 {
        let stderr_lines = render_failed_lines(error_type, error_message);
        record_terminal_lines_best_effort(
            &stderr_lines,
            "error",
            "native_cli_failed_line",
            "Native CLI failure detail.",
        );
        CliRunResult::from_lines(CLI_RUNTIME_FAILURE_EXIT_CODE, Vec::new(), stderr_lines)
    } else {
        CliRunResult { exit_code: current_exit_code, ..CliRunResult::default() }
    }
}

fn record_terminal_lines_best_effort(lines: &[String], level: &str, event_name: &str, message: &str) {
    for line in lines {
        if let Err(error) =
            g_runtime::emit_diagnostic_event(level, event_name, message, &TerminalLineDiagnosticFields { line })
        {
            tracing::warn!(
                target: "g.runner",
                error = %error,
                terminal_event = event_name,
                "Failed to emit native terminal diagnostic event."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use g_engine::RunHooks;
    use g_interface::CompiledCliRun;

    use super::{
        AncillaryFailure, HostRunHooks, PrimaryOutcome, RunnerProcessRuntimeState, check_interruption,
        completed_terminal_result, configure_jax_runtime, configure_process_runtime_with_jax_diagnostics,
        failed_terminal_result, finish_logging_after_late_interruption_observation, global_process_runtime_state,
        interrupted_terminal_result, lock_runtime_state, observe_ancillary_failure, observe_late_interruption,
        preflight_compiled_runs, resolve_primary_outcome, run_cli, run_compiled_cli, run_compiled_runs,
        runtime_close_failure_result, terminal_result_from_primary_outcome,
    };
    use crate::jax_runtime::{JaxRuntimeSetupSession, build_jax_runtime_policy};
    use crate::test_support::{
        BackendPlanKind, TemporaryRunFixture, TestErrorKind, TestHostError, TestNativeRunHost,
        execute_isolated_test_body,
    };
    use crate::{CliRunResult, NativeRunInterruption};

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
        let completion = completed_terminal_result(&[g_engine::PhenotypeRunArtifact {
            output_run_directory: "run".to_string(),
            parquet_dataset_directory: "run/parquet".to_string(),
        }]);
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
    fn primary_outcome_resolution_preserves_failures_artifacts_and_first_ancillary_error() {
        let artifacts = vec![g_engine::PhenotypeRunArtifact {
            output_run_directory: "completed-run".to_string(),
            parquet_dataset_directory: "completed-run/parquet".to_string(),
        }];
        let mut completed_host = TestNativeRunHost {
            interruption_results: [Err(TestHostError::sigint("late signal"))].into(),
            ..TestNativeRunHost::default()
        };
        let completed_outcome = resolve_primary_outcome(&mut completed_host, Ok(Ok(artifacts)));
        let mut ancillary_failure = None;
        assert!(observe_ancillary_failure(
            &mut ancillary_failure,
            AncillaryFailure { error_type: "TimingFileError", error_message: "timing write failed".to_string() },
        ));
        assert!(!observe_ancillary_failure(
            &mut ancillary_failure,
            AncillaryFailure { error_type: "TelemetryRunError", error_message: "telemetry close failed".to_string() },
        ));
        assert_eq!(
            ancillary_failure,
            Some(AncillaryFailure { error_type: "TimingFileError", error_message: "timing write failed".to_string() })
        );

        assert!(observe_late_interruption(&mut completed_host, &completed_outcome));
        let completed_result = terminal_result_from_primary_outcome(
            &mut completed_host,
            None,
            "test-thread",
            &completed_outcome,
            ancillary_failure.as_ref(),
        );
        assert_eq!(completed_result.exit_code, 1);
        assert_eq!(
            completed_result.stdout_chunks,
            ["Success. Run saved to completed-run\n", "Parquet dataset saved to completed-run/parquet\n",]
        );
        assert_eq!(completed_result.stderr_chunks, ["Error: timing write failed\n"]);
        assert!(!completed_result.stderr_chunks.concat().contains("late signal"));

        for primary_message in ["backend failed first", "output failed first"] {
            let mut failed_host = TestNativeRunHost {
                interruption_results: [Err(TestHostError::sigint("later signal"))].into(),
                ..TestNativeRunHost::default()
            };
            let failed_outcome =
                resolve_primary_outcome(&mut failed_host, Err(TestHostError::failure(primary_message)));
            assert!(observe_late_interruption(&mut failed_host, &failed_outcome));
            let failed_result = terminal_result_from_primary_outcome(
                &mut failed_host,
                None,
                "test-thread",
                &failed_outcome,
                ancillary_failure.as_ref(),
            );
            assert_eq!(failed_result.exit_code, 1);
            assert_eq!(failed_result.stdout_chunks, Vec::<String>::new());
            assert_eq!(failed_result.stderr_chunks, [format!("Error: {primary_message}\n")]);
            assert!(!failed_result.stderr_chunks.concat().contains("timing write failed"));
            assert!(!failed_result.stderr_chunks.concat().contains("later signal"));
        }

        for (interruption_error, expected_interruption, expected_exit_code) in [
            (TestHostError::sigint("SIGINT"), NativeRunInterruption::Sigint, 130),
            (
                TestHostError { kind: TestErrorKind::FlushedSigint, message: "flushed SIGINT".to_string() },
                NativeRunInterruption::FlushedSigint,
                130,
            ),
            (
                TestHostError { kind: TestErrorKind::Sigterm, message: "SIGTERM".to_string() },
                NativeRunInterruption::Sigterm,
                143,
            ),
        ] {
            let mut interrupted_host = TestNativeRunHost::default();
            let interrupted_outcome = resolve_primary_outcome(&mut interrupted_host, Err(interruption_error));
            assert!(matches!(
                interrupted_outcome,
                PrimaryOutcome::Interrupted { interruption }
                    if interruption == expected_interruption
            ));
            let interrupted_result = terminal_result_from_primary_outcome(
                &mut interrupted_host,
                None,
                "test-thread",
                &interrupted_outcome,
                ancillary_failure.as_ref(),
            );
            assert_eq!(interrupted_result.exit_code, expected_exit_code);
            assert!(!interrupted_result.stderr_chunks.concat().contains("timing write failed"));
        }
    }

    #[test]
    fn engine_proven_interruption_is_structural() {
        let mut host = TestNativeRunHost::default();
        let outcome = resolve_primary_outcome(
            &mut host,
            Ok(Err(g_engine::EngineRunError::Interrupted(TestHostError::failure("unclassified hook interruption")))),
        );
        assert!(matches!(outcome, PrimaryOutcome::Interrupted { interruption: NativeRunInterruption::FlushedSigint }));
    }

    #[test]
    fn fallback_late_interruption_observation_precedes_logging_finish() {
        let events = std::cell::RefCell::new(Vec::new());
        finish_logging_after_late_interruption_observation(
            false,
            || events.borrow_mut().push("observe_interruption"),
            || {
                events.borrow_mut().push("finish_logging");
                Ok(())
            },
        )
        .expect("test logging finish should succeed");
        assert_eq!(events.into_inner(), ["observe_interruption", "finish_logging"]);
    }

    #[test]
    fn completed_primary_outcome_ignores_late_signal_without_an_ancillary_failure() {
        let mut host = TestNativeRunHost {
            interruption_results: [Err(TestHostError::sigint("too late"))].into(),
            ..TestNativeRunHost::default()
        };
        let completed_outcome = resolve_primary_outcome(
            &mut host,
            Ok(Ok(vec![g_engine::PhenotypeRunArtifact {
                output_run_directory: "durable-run".to_string(),
                parquet_dataset_directory: "durable-run/parquet".to_string(),
            }])),
        );
        assert!(observe_late_interruption(&mut host, &completed_outcome));
        let result = terminal_result_from_primary_outcome(&mut host, None, "test-thread", &completed_outcome, None);
        assert_eq!(result.exit_code, 0);
        assert!(result.stderr_chunks.is_empty());
        assert!(result.stdout_chunks.concat().contains("durable-run/parquet"));
    }

    #[test]
    fn jax_configuration_skips_reconfiguration_and_applies_cpu_policy_once() {
        let mut run_plan =
            crate::test_support::run_plan(Path::new("jax-configuration"), g_plan::AssociationMode::Regenie2Linear);
        run_plan.compute.jax_cache_directory = Some("runner-cache".to_string());
        let policy = build_jax_runtime_policy(&run_plan).expect("test JAX policy should build");
        let mut host = TestNativeRunHost::default();
        let mut skipped_session = JaxRuntimeSetupSession::new(false, &policy);
        configure_jax_runtime(&mut host, &mut skipped_session).expect("configured runtime should be reused");
        assert!(host.calls.is_empty());

        let mut setup_session = JaxRuntimeSetupSession::new(true, &policy);
        configure_jax_runtime(&mut host, &mut setup_session).expect("CPU runtime should configure");
        assert_eq!(host.calls, ["apply_jax_config_updates"]);
        assert_eq!(host.config_update_names.len(), 7);
    }

    #[test]
    fn jax_setup_is_committed_before_best_effort_diagnostic_failure() {
        if !execute_isolated_test_body(
            "run::tests::jax_setup_is_committed_before_best_effort_diagnostic_failure",
            "G_RUNNER_JAX_DIAGNOSTIC_FAILURE_TEST_CHILD",
        ) {
            return;
        }
        let fixture = TemporaryRunFixture::new();
        let mut run_plan = fixture.run_plan(g_plan::AssociationMode::Regenie2Linear);
        run_plan.compute.jax_cache_directory = Some(fixture.root_path().join("jax-cache").display().to_string());
        let expected_policy = build_jax_runtime_policy(&run_plan).expect("test JAX policy should build");
        let logging_policy = crate::native_session_policy::project_native_run_session_policy(&run_plan);
        let mut host = TestNativeRunHost::default();

        configure_process_runtime_with_jax_diagnostics(&mut host, &run_plan, &logging_policy, |_setup_session| {
            let state = lock_runtime_state(global_process_runtime_state())
                .expect("runner runtime state should be available during diagnostic emission");
            state
                .jax
                .require_compatible(&expected_policy)
                .expect("JAX setup must be committed before diagnostics are emitted");
            Err("intentional completed-setup diagnostic failure".to_string())
        })
        .expect("diagnostic failure should not fail completed JAX setup");
        assert_eq!(host.calls, ["apply_jax_config_updates"]);

        configure_process_runtime_with_jax_diagnostics(
            &mut host,
            &run_plan,
            &logging_policy,
            |_setup_session| -> Result<(), String> {
                panic!("compatible configured JAX runtime should not re-emit setup diagnostics")
            },
        )
        .expect("compatible JAX setup should remain reusable");
        assert_eq!(host.calls, ["apply_jax_config_updates"]);
    }

    #[test]
    fn invalid_bgen_lifecycle_configures_fake_backend_and_translates_engine_failure() {
        if !execute_isolated_test_body(
            "run::tests::invalid_bgen_lifecycle_configures_fake_backend_and_translates_engine_failure",
            "G_RUNNER_INVALID_BGEN_LIFECYCLE_TEST_CHILD",
        ) {
            return;
        }
        let fixture = TemporaryRunFixture::new();
        let mut run_plan = fixture.run_plan(g_plan::AssociationMode::Regenie2Linear);
        run_plan.compute.jax_cache_directory = Some(fixture.root_path().join("jax-cache").display().to_string());
        let mut host = TestNativeRunHost::default();
        let output = run_compiled_cli(run_plan, "[test]\nfixture = true\n".to_string(), &mut host)
            .expect("engine failure should become terminal output");

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
        if !execute_isolated_test_body(
            "run::tests::thread_name_failure_escapes_before_runtime_or_backend_setup",
            "G_RUNNER_THREAD_NAME_FAILURE_TEST_CHILD",
        ) {
            return;
        }
        let fixture = TemporaryRunFixture::new();
        let expected_error = TestHostError::failure("thread unavailable");
        let mut host =
            TestNativeRunHost { current_thread_error: Some(expected_error.clone()), ..TestNativeRunHost::default() };
        let result =
            run_compiled_cli(fixture.run_plan(g_plan::AssociationMode::Regenie2Linear), String::new(), &mut host);
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
    fn primary_outcome_classification_covers_remaining_interruption_kinds() {
        let mut host = TestNativeRunHost::default();
        let sigterm_error = TestHostError { kind: TestErrorKind::Sigterm, message: "term".to_string() };
        let flushed_error = TestHostError { kind: TestErrorKind::FlushedSigint, message: "flushed".to_string() };
        assert!(matches!(
            resolve_primary_outcome(&mut host, Err(sigterm_error)),
            PrimaryOutcome::Interrupted { interruption: NativeRunInterruption::Sigterm }
        ));
        assert!(matches!(
            resolve_primary_outcome(&mut host, Err(flushed_error)),
            PrimaryOutcome::Interrupted { interruption: NativeRunInterruption::FlushedSigint }
        ));
    }
}
