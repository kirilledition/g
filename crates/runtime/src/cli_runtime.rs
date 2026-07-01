//! Native CLI runtime lifecycle state.

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliRunLifecycleState {
    runner_started: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliRunFailureTelemetryPlan {
    pub should_log_run_failed_to_telemetry: bool,
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
}
