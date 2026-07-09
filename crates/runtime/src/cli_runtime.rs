//! Native CLI runtime lifecycle state.

use crate::run_events::{
    RunArtifactsPayload, RunFailedEventPayload, RunInterruptedEventPayload, build_run_completed_event_from_artifacts,
    render_run_completed_lines, render_run_failed_lines, render_run_interrupted_lines,
};

pub const CLI_RUNTIME_FAILURE_EXIT_CODE: i32 = 1;
pub const NATIVE_CLI_OUTPUT_LOG_LIMIT: i64 = 4096;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliRunLifecycleState {
    runner_started: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliRunFailureTelemetryPlan {
    pub should_log_run_failed_to_telemetry: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliRunFailedTelemetryEmissionPlan {
    pub should_emit: bool,
    pub should_suppress_errors: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliTelemetryCloseFailurePlan {
    pub should_report_failure: bool,
    pub exit_code: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliExitCodeRangeError {
    pub exit_code: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliTerminalResult {
    pub exit_code: i32,
    pub stdout_lines: Vec<String>,
    pub stderr_lines: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliOutputChunks {
    pub stdout_chunks: Vec<String>,
    pub stderr_chunks: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliOutputBuffer {
    stdout_chunks: Vec<String>,
    stderr_chunks: Vec<String>,
}

impl CliRunLifecycleState {
    #[must_use]
    pub const fn runner_started(&self) -> bool {
        self.runner_started
    }

    pub const fn mark_runner_started(&mut self) {
        self.runner_started = true;
    }

    #[must_use]
    pub const fn plan_run_failed_telemetry(&self) -> CliRunFailureTelemetryPlan {
        CliRunFailureTelemetryPlan { should_log_run_failed_to_telemetry: !self.runner_started }
    }
}

impl CliTerminalResult {
    #[must_use]
    pub const fn new(exit_code: i32, stdout_lines: Vec<String>, stderr_lines: Vec<String>) -> Self {
        Self { exit_code, stdout_lines, stderr_lines }
    }

    #[must_use]
    pub const fn success(stdout_lines: Vec<String>) -> Self {
        Self::new(0, stdout_lines, Vec::new())
    }

    #[must_use]
    pub const fn interrupted(exit_code: i32, stderr_lines: Vec<String>) -> Self {
        Self::new(exit_code, Vec::new(), stderr_lines)
    }

    #[must_use]
    pub const fn failed(stderr_lines: Vec<String>) -> Self {
        Self::new(CLI_RUNTIME_FAILURE_EXIT_CODE, Vec::new(), stderr_lines)
    }

    #[must_use]
    pub const fn empty(exit_code: i32) -> Self {
        Self::new(exit_code, Vec::new(), Vec::new())
    }
}

impl CliOutputBuffer {
    #[must_use]
    pub fn from_frontend_output(stdout_text: &str, stderr_text: &str) -> Self {
        let mut buffer = Self::default();
        buffer.push_stdout_text(stdout_text);
        buffer.push_stderr_text(stderr_text);
        buffer
    }

    pub fn push_stdout_text(&mut self, text: &str) {
        push_text_chunk(&mut self.stdout_chunks, text);
    }

    pub fn push_stderr_text(&mut self, text: &str) {
        push_text_chunk(&mut self.stderr_chunks, text);
    }

    #[must_use]
    pub fn append_terminal_result(&mut self, terminal_result: CliTerminalResult) -> i32 {
        self.stdout_chunks.extend(lines_to_chunks(terminal_result.stdout_lines));
        self.stderr_chunks.extend(lines_to_chunks(terminal_result.stderr_lines));
        terminal_result.exit_code
    }

    #[must_use]
    pub fn into_chunks(self) -> CliOutputChunks {
        CliOutputChunks { stdout_chunks: self.stdout_chunks, stderr_chunks: self.stderr_chunks }
    }
}

#[must_use]
pub const fn plan_cli_telemetry_close_failure(
    current_exit_code: i32,
    runtime_failure_exit_code: i32,
) -> CliTelemetryCloseFailurePlan {
    let should_report_failure = current_exit_code == 0;
    CliTelemetryCloseFailurePlan {
        should_report_failure,
        exit_code: if should_report_failure { runtime_failure_exit_code } else { current_exit_code },
    }
}

#[must_use]
pub const fn plan_cli_run_failed_telemetry_emission(
    should_log_run_failed_to_telemetry: bool,
    has_telemetry_session: bool,
) -> CliRunFailedTelemetryEmissionPlan {
    CliRunFailedTelemetryEmissionPlan {
        should_emit: should_log_run_failed_to_telemetry && has_telemetry_session,
        should_suppress_errors: true,
    }
}

#[must_use]
pub fn build_completed_cli_terminal_result(artifacts: &RunArtifactsPayload) -> CliTerminalResult {
    let completed_event = build_run_completed_event_from_artifacts(artifacts);
    CliTerminalResult::success(render_run_completed_lines(&completed_event))
}

pub fn build_interrupted_cli_terminal_result(
    interrupted_event: &RunInterruptedEventPayload,
) -> Result<CliTerminalResult, CliExitCodeRangeError> {
    let exit_code = i32::try_from(interrupted_event.exit_code)
        .map_err(|_| CliExitCodeRangeError { exit_code: interrupted_event.exit_code })?;
    Ok(CliTerminalResult::interrupted(exit_code, render_run_interrupted_lines(interrupted_event)))
}

#[must_use]
pub fn build_failed_cli_terminal_result(failed_event: &RunFailedEventPayload) -> CliTerminalResult {
    CliTerminalResult::failed(render_run_failed_lines(failed_event))
}

#[must_use]
pub fn build_telemetry_close_failure_cli_terminal_result(
    current_exit_code: i32,
    failed_event: &RunFailedEventPayload,
) -> CliTerminalResult {
    let close_failure_plan = plan_cli_telemetry_close_failure(current_exit_code, CLI_RUNTIME_FAILURE_EXIT_CODE);
    if !close_failure_plan.should_report_failure {
        return CliTerminalResult::empty(close_failure_plan.exit_code);
    }
    CliTerminalResult::new(close_failure_plan.exit_code, Vec::new(), render_run_failed_lines(failed_event))
}

fn push_text_chunk(chunks: &mut Vec<String>, text: &str) {
    if !text.is_empty() {
        chunks.push(text.to_string());
    }
}

fn lines_to_chunks(lines: Vec<String>) -> impl Iterator<Item = String> {
    lines.into_iter().map(|line| format!("{line}\n"))
}
