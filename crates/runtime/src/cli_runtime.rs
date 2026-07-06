//! Native CLI runtime lifecycle state.

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
pub struct CliTerminalResult {
    pub exit_code: i32,
    pub stdout_lines: Vec<String>,
    pub stderr_lines: Vec<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_run_failed_telemetry_before_runner_starts() {
        let state = CliRunLifecycleState::default();

        assert!(!state.runner_started());
        assert!(state.plan_run_failed_telemetry().should_log_run_failed_to_telemetry);
    }

    #[test]
    fn suppresses_duplicate_run_failed_telemetry_after_runner_starts() {
        let mut state = CliRunLifecycleState::default();

        state.mark_runner_started();

        assert!(state.runner_started());
        assert!(!state.plan_run_failed_telemetry().should_log_run_failed_to_telemetry);
    }

    #[test]
    fn reports_telemetry_close_failure_after_successful_run() {
        assert_eq!(
            plan_cli_telemetry_close_failure(0, 1),
            CliTelemetryCloseFailurePlan { should_report_failure: true, exit_code: 1 },
        );
    }

    #[test]
    fn preserves_nonzero_exit_code_after_telemetry_close_failure() {
        assert_eq!(
            plan_cli_telemetry_close_failure(130, 1),
            CliTelemetryCloseFailurePlan { should_report_failure: false, exit_code: 130 },
        );
    }

    #[test]
    fn emits_run_failed_telemetry_when_requested_and_session_exists() {
        assert_eq!(
            plan_cli_run_failed_telemetry_emission(true, true),
            CliRunFailedTelemetryEmissionPlan { should_emit: true, should_suppress_errors: true },
        );
    }

    #[test]
    fn skips_run_failed_telemetry_when_not_requested_or_session_missing() {
        assert_eq!(
            plan_cli_run_failed_telemetry_emission(false, true),
            CliRunFailedTelemetryEmissionPlan { should_emit: false, should_suppress_errors: true },
        );
        assert_eq!(
            plan_cli_run_failed_telemetry_emission(true, false),
            CliRunFailedTelemetryEmissionPlan { should_emit: false, should_suppress_errors: true },
        );
    }
}
