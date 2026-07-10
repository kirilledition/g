//! CLI dispatch and native run lifecycle coordination.

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use g_engine::{AssociationBackend, JaxBackendSettings, RunHooks};
use g_interface::CliDispatch;
use g_runtime::{
    CLI_RUNTIME_FAILURE_EXIT_CODE, CliOutputBuffer, CliTerminalResult, JaxRuntimeDiagnosticFields,
    JaxRuntimeSetupSession, LoggingRuntimePolicyPayload, NativeRunSession, ProcessRuntimeState, TelemetryRunSession,
};

static GLOBAL_PROCESS_RUNTIME_STATE: OnceLock<Mutex<ProcessRuntimeState>> = OnceLock::new();

/// Native CLI output that the Python bootstrap forwards verbatim.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliRunResult {
    pub exit_code: i32,
    pub stdout_chunks: Vec<String>,
    pub stderr_chunks: Vec<String>,
}

/// Python-host logging inputs derived from the immutable run policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LoggingSetup {
    log_filter: String,
    log_file: Option<String>,
    log_stderr: bool,
    log_queue_size: usize,
    log_lossy: bool,
    include_source_location: bool,
    include_span_events: bool,
    trace_file: Option<String>,
    trace_filter: String,
    trace_event_cap: Option<usize>,
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

/// JAX runtime value applied through the Python host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JaxRuntimeConfigValue {
    Boolean(bool),
    Integer(i64),
    Text(String),
}

/// One JAX runtime setting applied through the Python host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeConfigUpdate {
    pub setting_name: String,
    pub value: JaxRuntimeConfigValue,
}

/// JAX device information observed by the Python host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxDevice {
    pub platform: String,
    pub description: String,
}

/// Python-host error details rendered by the Rust terminal lifecycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRunFailure {
    pub error_type: String,
    pub error_message: String,
}

/// Operations the Python host must perform while Rust owns the run lifecycle.
pub trait NativeRunHost: Send {
    /// Concrete association backend retaining opaque Python state.
    type Backend: AssociationBackend + 'static;
    /// Python-boundary error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Install the Python logging bridge after Rust initialized logging sinks.
    fn install_python_logging(&mut self) -> Result<(), Self::Error>;

    /// Apply validated JAX configuration updates through the Python runtime.
    fn apply_jax_config_updates(&mut self, updates: &[JaxRuntimeConfigUpdate]) -> Result<(), Self::Error>;

    /// Return Python-owned JAX device observations for GPU validation.
    fn observe_jax_devices(&mut self) -> Result<Vec<JaxDevice>, Self::Error>;

    /// Construct the opaque JAX association backend from validated scalar settings.
    fn create_backend(&mut self, settings: JaxBackendSettings) -> Result<Arc<Self::Backend>, Self::Error>;

    /// Check Python signals.
    fn check_interruption(&mut self) -> Result<(), Self::Error>;

    /// Construct the Python-host interruption used for a native SIGTERM request.
    fn sigterm_interruption_error(&mut self) -> Self::Error;

    /// Mark an interrupted run whose resumable output was successfully flushed.
    fn flushed_interruption_error(&mut self, error: Self::Error) -> Self::Error;

    /// Name the signal associated with an engine interruption, when known.
    fn interruption_signal_name(error: &Self::Error) -> Option<&str>;

    /// Classify a terminal Python-boundary error as a resumable interruption.
    fn interruption_kind(&mut self, error: &Self::Error) -> Option<NativeRunInterruption>;

    /// Convert a Python-free engine failure into the host error type.
    fn engine_error(&mut self, message: String) -> Self::Error;

    /// Construct a host error for a native lifecycle failure.
    fn native_runtime_error(&mut self, message: String) -> Self::Error;

    /// Convert a host error into the terminal telemetry payload.
    fn failed_event(&mut self, error: &Self::Error) -> NativeRunFailure;

    /// Read the current Python thread name for telemetry labels.
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
    <Host::Backend as AssociationBackend>::ChromosomeState: 'static,
    <Host::Backend as AssociationBackend>::DeviceResult: 'static,
{
    match g_interface::dispatch_cli(arguments) {
        CliDispatch::Exit { exit_code, stdout, stderr } => Ok(cli_result_from_output(exit_code, &stdout, &stderr)),
        CliDispatch::Run(compiled_run) => {
            run_compiled_cli(compiled_run.run_plan, compiled_run.effective_config_toml, host)
        }
    }
}

fn run_compiled_cli<Host>(
    run_plan: g_plan::RunPlan,
    effective_config_toml: String,
    host: &mut Host,
) -> Result<CliRunResult, Host::Error>
where
    Host: NativeRunHost,
    Host::Backend: AssociationBackend + 'static,
    <Host::Backend as AssociationBackend>::ChromosomeState: 'static,
    <Host::Backend as AssociationBackend>::DeviceResult: 'static,
{
    let mut output = CliOutputBuffer::default();
    let mut native_session = match NativeRunSession::new(&run_plan) {
        Ok(session) => session,
        Err(error) => {
            let error = host.native_runtime_error(error.to_string());
            let terminal_result = failed_terminal_result(host, None, &error, None);
            let exit_code = output.append_terminal_result(terminal_result);
            return Ok(CliRunResult {
                exit_code,
                stdout_chunks: output.stdout_chunks,
                stderr_chunks: output.stderr_chunks,
            });
        }
    };
    let thread_name = host.current_thread_name()?;
    let mut execution_result = (|| {
        initialize_process_logging_runtime_policy(host, &native_session.logging_policy)?;
        let runtime_start_time = Instant::now();
        configure_process_runtime(
            host,
            &run_plan,
            &native_session.logging_policy,
            &native_session.telemetry_session,
            &thread_name,
        )?;
        native_session.record_stage_duration("jax_runtime_configuration", runtime_start_time);

        let backend_start_time = Instant::now();
        let backend_settings = JaxBackendSettings::from_run_plan(&run_plan)
            .map_err(|error| host.native_runtime_error(error.to_string()))?;
        let backend = host.create_backend(backend_settings)?;
        native_session.record_stage_duration("jax_backend_initialization", backend_start_time);

        let telemetry_session = &native_session.telemetry_session;
        let stage_timing_recorder = native_session.stage_timing_recorder.as_mut();
        let thread_name_for_run = thread_name.as_str();
        let execution_result = Host::detach(|| {
            let mut hooks = HostRunHooks { host };
            g_engine::execute_coordinated_run(
                run_plan,
                effective_config_toml,
                backend,
                &mut hooks,
                telemetry_session,
                thread_name_for_run,
                stage_timing_recorder,
            )
        });
        execution_result.map_err(|error| match error {
            g_engine::EngineRunError::Interrupted(error) => host.flushed_interruption_error(error),
            g_engine::EngineRunError::Failure { message } => host.engine_error(message),
        })
    })();
    if execution_result.is_ok()
        && let Err(error) = check_interruption(host)
    {
        execution_result = Err(error);
    }
    let timing_result = native_session.finish_timing().map_err(|error| host.native_runtime_error(error.to_string()));
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
        Err(error) => terminal_result_from_error(host, Some(&native_session.telemetry_session), &thread_name, &error),
    };
    let mut close_result =
        finish_telemetry_result(&native_session.telemetry_session, &thread_name, terminal_result.exit_code);
    if let Err(error) = check_interruption(host) {
        terminal_result = terminal_result_from_error(host, None, &thread_name, &error);
        close_result = CliTerminalResult {
            exit_code: terminal_result.exit_code,
            stdout_lines: Vec::new(),
            stderr_lines: Vec::new(),
        };
    }
    let _ = output.append_terminal_result(terminal_result);
    let exit_code = output.append_terminal_result(close_result);
    Ok(CliRunResult { exit_code, stdout_chunks: output.stdout_chunks, stderr_chunks: output.stderr_chunks })
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

fn cli_result_from_output(exit_code: i32, stdout: &str, stderr: &str) -> CliRunResult {
    let output = CliOutputBuffer::from_frontend_output(stdout, stderr);
    CliRunResult { exit_code, stdout_chunks: output.stdout_chunks, stderr_chunks: output.stderr_chunks }
}

fn initialize_process_logging_runtime_policy<Host>(
    host: &mut Host,
    logging_policy: &LoggingRuntimePolicyPayload,
) -> Result<(), Host::Error>
where
    Host: NativeRunHost,
{
    let setup = logging_setup(logging_policy).map_err(|message| host.native_runtime_error(message))?;
    let runtime_state = global_process_runtime_state();
    let mut state = lock_runtime_state(runtime_state).map_err(|message| host.native_runtime_error(message))?;
    state
        .require_compatible_logging_policy(logging_policy)
        .map_err(|error| host.native_runtime_error(error.to_string()))?;
    let logging_config = g_runtime::LoggingSinkConfig {
        log_filter: Some(setup.log_filter.as_str()),
        log_file: setup.log_file.as_deref().map(Path::new),
        log_stderr: setup.log_stderr,
        log_queue_size: setup.log_queue_size,
        log_lossy: setup.log_lossy,
        include_source_location: setup.include_source_location,
        include_span_events: setup.include_span_events,
        trace_file: setup.trace_file.as_deref().map(Path::new),
        trace_filter: Some(setup.trace_filter.as_str()),
        trace_event_cap: setup.trace_event_cap,
    };
    match g_runtime::initialize_logging_sinks(logging_config, || host.install_python_logging()) {
        Ok(_initialized) => {}
        Err(g_runtime::LoggingSinkInitializationError::HostLogging(error)) => return Err(error),
        Err(g_runtime::LoggingSinkInitializationError::Sink(error)) => {
            return Err(host.native_runtime_error(error.to_string()));
        }
    }
    state.record_logging_policy(logging_policy.clone());
    Ok(())
}

fn configure_process_runtime<Host>(
    host: &mut Host,
    run_plan: &g_plan::RunPlan,
    logging_policy: &LoggingRuntimePolicyPayload,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
) -> Result<(), Host::Error>
where
    Host: NativeRunHost,
{
    let bgen_decode_tile_variant_count = run_plan.compute.bgen_decode_tile_variant_count;
    let rayon_thread_count = run_plan.compute.cpu_thread_count.map(i64::from);
    let jax_policy = g_runtime::build_jax_runtime_policy_payload(run_plan)
        .map_err(|error| host.native_runtime_error(error.to_string()))?;
    let runtime_state = global_process_runtime_state();
    lock_runtime_state(runtime_state)
        .map_err(|message| host.native_runtime_error(message))?
        .require_compatible_runtime_policy(logging_policy, rayon_thread_count, &jax_policy)
        .map_err(|error| host.native_runtime_error(error.to_string()))?;
    g_runtime::emit_run_diagnostic_event(&g_runtime::build_native_runtime_knobs_configured_diagnostic_payload(
        i64::from(bgen_decode_tile_variant_count),
        rayon_thread_count,
    ))
    .map_err(|error| host.native_runtime_error(format!("Failed to serialize runtime diagnostic event: {error}")))?;
    let mut setup_session = {
        let mut state = lock_runtime_state(runtime_state).map_err(|message| host.native_runtime_error(message))?;
        if let Some(thread_count) = rayon_thread_count {
            state
                .configure_rayon_thread_pool(thread_count)
                .map_err(|error| host.native_runtime_error(error.to_string()))?;
        }
        state
            .build_jax_runtime_setup_session_resolving_cache_directory(&jax_policy)
            .map_err(|error| host.native_runtime_error(error.to_string()))?
    };
    let should_configure_jax = setup_session.should_configure();
    configure_jax_runtime(host, &mut setup_session, telemetry_session, thread_name)?;
    let mut state = lock_runtime_state(runtime_state).map_err(|message| host.native_runtime_error(message))?;
    state
        .require_compatible_runtime_policy(logging_policy, rayon_thread_count, &jax_policy)
        .map_err(|error| host.native_runtime_error(error.to_string()))?;
    if should_configure_jax {
        state
            .complete_jax_runtime_setup_session(jax_policy, &setup_session)
            .map_err(|error| host.native_runtime_error(error.to_string()))?;
    }
    Ok(())
}

fn configure_jax_runtime<Host>(
    host: &mut Host,
    setup_session: &mut JaxRuntimeSetupSession,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
) -> Result<(), Host::Error>
where
    Host: NativeRunHost,
{
    if !setup_session.should_configure() {
        return Ok(());
    }
    setup_session
        .create_cache_directory_if_configured()
        .map_err(|error| host.native_runtime_error(error.to_string()))?;
    let config_updates = setup_session
        .config_updates()
        .into_iter()
        .map(|update| JaxRuntimeConfigUpdate {
            setting_name: update.setting_name,
            value: match update.value {
                g_runtime::JaxRuntimeConfigValue::Boolean(value) => JaxRuntimeConfigValue::Boolean(value),
                g_runtime::JaxRuntimeConfigValue::Integer(value) => JaxRuntimeConfigValue::Integer(value),
                g_runtime::JaxRuntimeConfigValue::Text(value) => JaxRuntimeConfigValue::Text(value),
            },
        })
        .collect::<Vec<_>>();
    host.apply_jax_config_updates(&config_updates)?;
    if setup_session.setup().gpu_validation_status == "pending" {
        let probe_paths = g_runtime::default_nvidia_driver_probe_paths();
        let nvidia_driver_visible = g_runtime::nvidia_driver_files_are_visible(
            Path::new(&probe_paths.control_device_path),
            Path::new(&probe_paths.uvm_device_path),
            Path::new(&probe_paths.driver_directory_path),
        );
        let (backend_initialization_failed, devices) = if nvidia_driver_visible {
            match host.observe_jax_devices() {
                Ok(devices) => (false, devices),
                Err(_error) => (true, Vec::new()),
            }
        } else {
            (false, Vec::new())
        };
        let runtime_devices = devices
            .iter()
            .map(|device| g_runtime::JaxDeviceObservation {
                platform: device.platform.clone(),
                description: device.description.clone(),
            })
            .collect::<Vec<_>>();
        let validation_plan =
            g_runtime::plan_jax_gpu_validation(nvidia_driver_visible, backend_initialization_failed, &runtime_devices);
        setup_session.complete_validation(&validation_plan.status, Some(validation_plan.message.as_str()));
        if validation_plan.should_raise {
            return Err(host.native_runtime_error(validation_plan.message));
        }
    }
    for event in setup_session.diagnostic_events() {
        let record_plan = g_runtime::plan_jax_runtime_diagnostic_record(&event.level);
        let fields = JaxRuntimeDiagnosticFields::new(&event.fields);
        g_runtime::emit_diagnostic_event(
            &record_plan.logging_level_name.to_lowercase(),
            &event.event_name,
            &event.message,
            &fields,
        )
        .map_err(|error| {
            host.native_runtime_error(format!("Failed to serialize JAX runtime diagnostic event fields: {error}"))
        })?;
        telemetry_session
            .emit_current_event(thread_name, &event.event_name, &record_plan.telemetry_level, &fields)
            .map_err(|error| host.native_runtime_error(error.to_string()))?;
    }
    Ok(())
}

fn logging_setup(logging_policy: &LoggingRuntimePolicyPayload) -> Result<LoggingSetup, String> {
    Ok(LoggingSetup {
        log_filter: logging_policy.log_filter.clone(),
        log_file: logging_policy.log_file.clone(),
        log_stderr: logging_policy.log_stderr,
        log_queue_size: non_negative_i64_to_usize(logging_policy.log_queue_size, "log_queue_size")?,
        log_lossy: logging_policy.log_lossy,
        include_source_location: logging_policy.include_source_location,
        include_span_events: logging_policy.include_span_events,
        trace_file: logging_policy.trace_file.clone(),
        trace_filter: logging_policy.trace_filter.clone(),
        trace_event_cap: logging_policy
            .trace_event_cap
            .map(|value| non_negative_i64_to_usize(value, "trace_event_cap"))
            .transpose()?,
    })
}

fn non_negative_i64_to_usize(value: i64, field_name: &str) -> Result<usize, String> {
    if value < 0 {
        return Err(format!("{field_name} must be non-negative. Observed {value}."));
    }
    usize::try_from(value).map_err(|_| format!("{field_name} does not fit into native usize."))
}

fn global_process_runtime_state() -> &'static Mutex<ProcessRuntimeState> {
    GLOBAL_PROCESS_RUNTIME_STATE.get_or_init(|| Mutex::new(ProcessRuntimeState::default()))
}

fn lock_runtime_state(
    runtime_state: &Mutex<ProcessRuntimeState>,
) -> Result<MutexGuard<'_, ProcessRuntimeState>, String> {
    runtime_state.lock().map_err(|_| "Runtime state mutex was poisoned.".to_string())
}

fn completed_terminal_result<Host>(
    host: &mut Host,
    artifacts: &[g_runtime::PhenotypeRunArtifacts],
) -> Result<CliTerminalResult, Host::Error>
where
    Host: NativeRunHost,
{
    let terminal_result = CliTerminalResult {
        exit_code: 0,
        stdout_lines: g_runtime::render_run_completed_lines(artifacts),
        stderr_lines: Vec::new(),
    };
    record_terminal_lines(&terminal_result.stdout_lines, g_runtime::build_native_cli_completed_line_diagnostic_payload)
        .map_err(|message| host.native_runtime_error(message))?;
    Ok(terminal_result)
}

fn terminal_result_from_error<Host>(
    host: &mut Host,
    telemetry_session: Option<&TelemetryRunSession>,
    thread_name: &str,
    error: &Host::Error,
) -> CliTerminalResult
where
    Host: NativeRunHost,
{
    if let Some(interruption) = host.interruption_kind(error) {
        return interrupted_terminal_result(interruption);
    }
    failed_terminal_result(host, telemetry_session, error, Some(thread_name))
}

fn interrupted_terminal_result(interruption: NativeRunInterruption) -> CliTerminalResult {
    let (signal_name, exit_code, flushed_for_resume) = match interruption {
        NativeRunInterruption::Sigterm => ("SIGTERM", 143, true),
        NativeRunInterruption::Sigint => ("SIGINT", 130, false),
        NativeRunInterruption::FlushedSigint => ("SIGINT", 130, true),
    };
    let interrupted_event = g_runtime::build_run_interrupted_event_payload(signal_name, exit_code, flushed_for_resume);
    let terminal_result = CliTerminalResult {
        exit_code: interrupted_event.exit_code,
        stdout_lines: Vec::new(),
        stderr_lines: g_runtime::render_run_interrupted_lines(&interrupted_event),
    };
    let _ = record_terminal_lines(
        &terminal_result.stderr_lines,
        g_runtime::build_native_cli_interrupted_line_diagnostic_payload,
    );
    terminal_result
}

fn failed_terminal_result<Host>(
    host: &mut Host,
    telemetry_session: Option<&TelemetryRunSession>,
    error: &Host::Error,
    thread_name: Option<&str>,
) -> CliTerminalResult
where
    Host: NativeRunHost,
{
    let failure = host.failed_event(error);
    let failed_event = g_runtime::build_run_failed_event_payload(&failure.error_type, &failure.error_message);
    if let Some(telemetry_session) = telemetry_session
        && let Some(thread_name) = thread_name
    {
        let _ = telemetry_session.emit_run_failed_event(thread_name, &failed_event);
    }
    let terminal_result = CliTerminalResult {
        exit_code: CLI_RUNTIME_FAILURE_EXIT_CODE,
        stdout_lines: Vec::new(),
        stderr_lines: g_runtime::render_run_failed_lines(&failed_event),
    };
    let _ = record_terminal_lines(
        &terminal_result.stderr_lines,
        g_runtime::build_native_cli_failed_line_diagnostic_payload,
    );
    terminal_result
}

fn finish_telemetry_result(
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    current_exit_code: i32,
) -> CliTerminalResult {
    match telemetry_session.finish(thread_name) {
        Ok(()) => {
            CliTerminalResult { exit_code: current_exit_code, stdout_lines: Vec::new(), stderr_lines: Vec::new() }
        }
        Err(error) => telemetry_close_failure_result(current_exit_code, &error),
    }
}

fn telemetry_close_failure_result(current_exit_code: i32, error: &g_runtime::TelemetryRunError) -> CliTerminalResult {
    let failed_event = g_runtime::build_run_failed_event_payload("TelemetryRunError", &error.to_string());
    let terminal_result = if current_exit_code == 0 {
        CliTerminalResult {
            exit_code: CLI_RUNTIME_FAILURE_EXIT_CODE,
            stdout_lines: Vec::new(),
            stderr_lines: g_runtime::render_run_failed_lines(&failed_event),
        }
    } else {
        CliTerminalResult { exit_code: current_exit_code, stdout_lines: Vec::new(), stderr_lines: Vec::new() }
    };
    let _ = record_terminal_lines(
        &terminal_result.stderr_lines,
        g_runtime::build_native_cli_failed_line_diagnostic_payload,
    );
    terminal_result
}

fn record_terminal_lines(
    lines: &[String],
    diagnostic: fn(&str) -> g_runtime::RunDiagnosticEventPayload,
) -> Result<(), String> {
    for line in lines {
        let payload = diagnostic(line);
        g_runtime::emit_run_diagnostic_event(&payload)
            .map_err(|error| format!("Failed to serialize terminal diagnostic event: {error}"))?;
    }
    Ok(())
}
