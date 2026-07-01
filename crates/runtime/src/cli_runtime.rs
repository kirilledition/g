//! Native CLI runtime lifecycle state.

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliRunLifecycleState {
    runner_started: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliRunFailureTelemetryPlan {
    pub should_log_run_failed_to_telemetry: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliTelemetryCloseFailurePlan {
    pub should_report_failure: bool,
    pub exit_code: i32,
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
}
