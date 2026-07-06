//! Callback worker lifecycle, shutdown, and error propagation policy.

#![allow(clippy::module_name_repetitions)]

pub(crate) const CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS: f64 = 0.1;
pub(crate) const CALLBACK_WORKER_STOP_POLL_TIMEOUT_CAP_SECONDS: f64 = 0.1;
pub(crate) const CALLBACK_WORKER_START_RESULT_WORKER_ACTION: &str = "start_result_worker";
pub(crate) const CALLBACK_WORKER_START_DOSAGE_WORKER_ACTION: &str = "start_dosage_worker";
pub(crate) const CALLBACK_WORKER_FINISH_STOP_DOSAGE_WORKER_ACTION: &str = "stop_dosage_worker";
pub(crate) const CALLBACK_WORKER_FINISH_JOIN_DOSAGE_WORKER_ACTION: &str = "join_dosage_worker";
pub(crate) const CALLBACK_WORKER_FINISH_STOP_RESULT_WORKER_ACTION: &str = "stop_result_worker";
pub(crate) const CALLBACK_WORKER_FINISH_JOIN_RESULT_WORKER_ACTION: &str = "join_result_worker";
pub(crate) const CALLBACK_WORKER_FINISH_RAISE_WORKER_ERROR_ACTION: &str = "raise_worker_error";
pub(crate) const CALLBACK_WORKER_FINISH_COMPLETE_PROGRESS_ACTION: &str = "complete_progress";
pub(crate) const CALLBACK_WORKER_FINISH_EMIT_BINARY_CORRECTION_SUMMARY_ACTION: &str = "emit_binary_correction_summary";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackWorkerStartPlan {
    pub start_actions: Vec<String>,
}

impl CallbackWorkerStartPlan {
    #[must_use]
    pub fn should_start(&self) -> bool {
        !self.start_actions.is_empty()
    }

    #[must_use]
    pub fn start_result_worker(&self) -> bool {
        self.contains_start_action(CALLBACK_WORKER_START_RESULT_WORKER_ACTION)
    }

    #[must_use]
    pub fn start_dosage_worker(&self) -> bool {
        self.contains_start_action(CALLBACK_WORKER_START_DOSAGE_WORKER_ACTION)
    }

    fn contains_start_action(&self, start_action: &str) -> bool {
        self.start_actions.iter().any(|candidate_action| candidate_action == start_action)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackWorkerStartAttemptPlan {
    pub start_actions: Vec<String>,
    pub has_marked_started: bool,
    pub has_start_error: bool,
    pub error_message: Option<String>,
}

impl CallbackWorkerStartAttemptPlan {
    #[must_use]
    pub fn should_start(&self) -> bool {
        !self.start_actions.is_empty()
    }

    #[must_use]
    pub fn start_result_worker(&self) -> bool {
        self.contains_start_action(CALLBACK_WORKER_START_RESULT_WORKER_ACTION)
    }

    #[must_use]
    pub fn start_dosage_worker(&self) -> bool {
        self.contains_start_action(CALLBACK_WORKER_START_DOSAGE_WORKER_ACTION)
    }

    fn contains_start_action(&self, start_action: &str) -> bool {
        self.start_actions.iter().any(|candidate_action| candidate_action == start_action)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CallbackWorkerLifecycleState {
    started: bool,
}

impl CallbackWorkerLifecycleState {
    #[must_use]
    pub const fn new() -> Self {
        Self { started: false }
    }

    #[must_use]
    pub const fn has_started(&self) -> bool {
        self.started
    }

    pub fn mark_started(&mut self) -> bool {
        if self.started {
            return false;
        }
        self.started = true;
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackWorkerShutdownTimeouts {
    pub dosage_worker_join_timeout_seconds: f64,
    pub result_worker_join_timeout_seconds: f64,
    pub graceful_dosage_worker_join_timeout_seconds: f64,
    pub graceful_result_worker_join_timeout_seconds: f64,
    pub worker_abort_stop_timeout_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackWorkerJoinPlan {
    pub should_join: bool,
    pub timeout_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackWorkerStopPlan {
    pub should_stop: bool,
    pub timeout_seconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallbackWorkerFinishPlan {
    pub finish_actions: Vec<String>,
    pub dosage_stop_timeout_seconds: f64,
    pub dosage_join_timeout_seconds: f64,
    pub result_stop_timeout_seconds: f64,
    pub result_join_timeout_seconds: f64,
}

impl CallbackWorkerFinishPlan {
    #[must_use]
    pub fn stop_dosage_worker(&self) -> bool {
        self.contains_finish_action(CALLBACK_WORKER_FINISH_STOP_DOSAGE_WORKER_ACTION)
    }

    #[must_use]
    pub fn join_dosage_worker(&self) -> bool {
        self.contains_finish_action(CALLBACK_WORKER_FINISH_JOIN_DOSAGE_WORKER_ACTION)
    }

    #[must_use]
    pub fn stop_result_worker(&self) -> bool {
        self.contains_finish_action(CALLBACK_WORKER_FINISH_STOP_RESULT_WORKER_ACTION)
    }

    #[must_use]
    pub fn join_result_worker(&self) -> bool {
        self.contains_finish_action(CALLBACK_WORKER_FINISH_JOIN_RESULT_WORKER_ACTION)
    }

    #[must_use]
    pub fn raise_worker_error(&self) -> bool {
        self.contains_finish_action(CALLBACK_WORKER_FINISH_RAISE_WORKER_ERROR_ACTION)
    }

    #[must_use]
    pub fn complete_progress(&self) -> bool {
        self.contains_finish_action(CALLBACK_WORKER_FINISH_COMPLETE_PROGRESS_ACTION)
    }

    #[must_use]
    pub fn emit_binary_correction_summary(&self) -> bool {
        self.contains_finish_action(CALLBACK_WORKER_FINISH_EMIT_BINARY_CORRECTION_SUMMARY_ACTION)
    }

    fn contains_finish_action(&self, finish_action: &str) -> bool {
        self.finish_actions.iter().any(|candidate_action| candidate_action == finish_action)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallbackWorkerAbortPlan {
    pub abort_actions: Vec<String>,
    pub dosage_stop_timeout_seconds: f64,
    pub result_stop_timeout_seconds: f64,
}

impl CallbackWorkerAbortPlan {
    #[must_use]
    pub fn stop_dosage_worker(&self) -> bool {
        self.contains_abort_action(CALLBACK_WORKER_FINISH_STOP_DOSAGE_WORKER_ACTION)
    }

    #[must_use]
    pub fn stop_result_worker(&self) -> bool {
        self.contains_abort_action(CALLBACK_WORKER_FINISH_STOP_RESULT_WORKER_ACTION)
    }

    fn contains_abort_action(&self, abort_action: &str) -> bool {
        self.abort_actions.iter().any(|candidate_action| candidate_action == abort_action)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackWorkerStopPollPlan {
    pub should_stop: bool,
    pub poll_timeout_seconds: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackWorkerErrorRaisePlan {
    pub should_raise: bool,
    pub raise_dosage_worker_error: bool,
    pub raise_result_worker_error: bool,
    pub error_message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackWorkerErrorUpdatePlan {
    pub had_error: bool,
    pub has_error: bool,
    pub error_message: Option<String>,
}

#[must_use]
pub const fn callback_worker_shutdown_timeouts() -> CallbackWorkerShutdownTimeouts {
    CallbackWorkerShutdownTimeouts {
        dosage_worker_join_timeout_seconds: 60.0,
        result_worker_join_timeout_seconds: 60.0,
        graceful_dosage_worker_join_timeout_seconds: 300.0,
        graceful_result_worker_join_timeout_seconds: 300.0,
        worker_abort_stop_timeout_seconds: 1.0,
    }
}

#[must_use]
pub const fn callback_worker_backpressure_poll_timeout_seconds() -> f64 {
    CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS
}

#[must_use]
pub fn resolve_callback_worker_stop_poll_timeout_seconds(remaining_timeout_seconds: f64) -> f64 {
    if remaining_timeout_seconds.is_nan() || remaining_timeout_seconds <= 0.0 {
        return 0.0;
    }
    if remaining_timeout_seconds > CALLBACK_WORKER_STOP_POLL_TIMEOUT_CAP_SECONDS {
        return CALLBACK_WORKER_STOP_POLL_TIMEOUT_CAP_SECONDS;
    }
    remaining_timeout_seconds
}

#[must_use]
pub const fn should_attempt_callback_worker_stop(
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> bool {
    has_started && !has_worker_error && is_worker_alive
}

fn plan_callback_worker_join(
    timeout_seconds: Option<f64>,
    has_started: bool,
    default_timeout_seconds: f64,
) -> CallbackWorkerJoinPlan {
    CallbackWorkerJoinPlan {
        should_join: has_started,
        timeout_seconds: timeout_seconds.unwrap_or(default_timeout_seconds),
    }
}

fn plan_callback_worker_stop(
    timeout_seconds: Option<f64>,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
    default_timeout_seconds: f64,
) -> CallbackWorkerStopPlan {
    CallbackWorkerStopPlan {
        should_stop: should_attempt_callback_worker_stop(has_started, has_worker_error, is_worker_alive),
        timeout_seconds: timeout_seconds.unwrap_or(default_timeout_seconds),
    }
}

#[must_use]
pub fn plan_dosage_callback_worker_join(timeout_seconds: Option<f64>, has_started: bool) -> CallbackWorkerJoinPlan {
    plan_callback_worker_join(
        timeout_seconds,
        has_started,
        callback_worker_shutdown_timeouts().dosage_worker_join_timeout_seconds,
    )
}

#[must_use]
pub fn plan_result_callback_worker_join(timeout_seconds: Option<f64>, has_started: bool) -> CallbackWorkerJoinPlan {
    plan_callback_worker_join(
        timeout_seconds,
        has_started,
        callback_worker_shutdown_timeouts().result_worker_join_timeout_seconds,
    )
}

#[must_use]
pub fn plan_dosage_callback_worker_stop(
    timeout_seconds: Option<f64>,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> CallbackWorkerStopPlan {
    plan_callback_worker_stop(
        timeout_seconds,
        has_started,
        has_worker_error,
        is_worker_alive,
        callback_worker_shutdown_timeouts().dosage_worker_join_timeout_seconds,
    )
}

#[must_use]
pub fn plan_result_callback_worker_stop(
    timeout_seconds: Option<f64>,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> CallbackWorkerStopPlan {
    plan_callback_worker_stop(
        timeout_seconds,
        has_started,
        has_worker_error,
        is_worker_alive,
        callback_worker_shutdown_timeouts().result_worker_join_timeout_seconds,
    )
}

#[must_use]
pub fn plan_callback_worker_finish() -> CallbackWorkerFinishPlan {
    let shutdown_timeouts = callback_worker_shutdown_timeouts();
    CallbackWorkerFinishPlan {
        finish_actions: vec![
            CALLBACK_WORKER_FINISH_STOP_DOSAGE_WORKER_ACTION.to_string(),
            CALLBACK_WORKER_FINISH_JOIN_DOSAGE_WORKER_ACTION.to_string(),
            CALLBACK_WORKER_FINISH_STOP_RESULT_WORKER_ACTION.to_string(),
            CALLBACK_WORKER_FINISH_JOIN_RESULT_WORKER_ACTION.to_string(),
            CALLBACK_WORKER_FINISH_RAISE_WORKER_ERROR_ACTION.to_string(),
            CALLBACK_WORKER_FINISH_COMPLETE_PROGRESS_ACTION.to_string(),
            CALLBACK_WORKER_FINISH_EMIT_BINARY_CORRECTION_SUMMARY_ACTION.to_string(),
        ],
        dosage_stop_timeout_seconds: shutdown_timeouts.dosage_worker_join_timeout_seconds,
        dosage_join_timeout_seconds: shutdown_timeouts.graceful_dosage_worker_join_timeout_seconds,
        result_stop_timeout_seconds: shutdown_timeouts.result_worker_join_timeout_seconds,
        result_join_timeout_seconds: shutdown_timeouts.graceful_result_worker_join_timeout_seconds,
    }
}

#[must_use]
pub fn plan_callback_worker_abort() -> CallbackWorkerAbortPlan {
    let shutdown_timeouts = callback_worker_shutdown_timeouts();
    CallbackWorkerAbortPlan {
        abort_actions: vec![
            CALLBACK_WORKER_FINISH_STOP_DOSAGE_WORKER_ACTION.to_string(),
            CALLBACK_WORKER_FINISH_STOP_RESULT_WORKER_ACTION.to_string(),
        ],
        dosage_stop_timeout_seconds: shutdown_timeouts.worker_abort_stop_timeout_seconds,
        result_stop_timeout_seconds: shutdown_timeouts.worker_abort_stop_timeout_seconds,
    }
}

#[must_use]
pub fn plan_callback_worker_stop_poll(
    remaining_timeout_seconds: f64,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> CallbackWorkerStopPollPlan {
    CallbackWorkerStopPollPlan {
        should_stop: should_attempt_callback_worker_stop(has_started, has_worker_error, is_worker_alive),
        poll_timeout_seconds: resolve_callback_worker_stop_poll_timeout_seconds(remaining_timeout_seconds),
    }
}

#[must_use]
pub fn format_dosage_callback_worker_error_message(error_message: &str) -> String {
    format!("native pipeline callback worker failed: {error_message}")
}

#[must_use]
pub fn format_result_callback_worker_error_message(error_message: &str) -> String {
    format!("native pipeline result writer worker failed: {error_message}")
}

pub(crate) fn update_callback_worker_error(
    worker_error_message: &mut Option<String>,
    error_message: Option<&str>,
    format_worker_error_message: fn(&str) -> String,
) -> CallbackWorkerErrorUpdatePlan {
    let had_error = worker_error_message.is_some();
    *worker_error_message = error_message.map(format_worker_error_message);
    CallbackWorkerErrorUpdatePlan {
        had_error,
        has_error: worker_error_message.is_some(),
        error_message: worker_error_message.clone(),
    }
}

pub(crate) fn plan_callback_worker_error_raise(
    dosage_worker_error_message: Option<&str>,
    result_worker_error_message: Option<&str>,
) -> CallbackWorkerErrorRaisePlan {
    if let Some(error_message) = dosage_worker_error_message {
        return CallbackWorkerErrorRaisePlan {
            should_raise: true,
            raise_dosage_worker_error: true,
            raise_result_worker_error: false,
            error_message: Some(error_message.to_string()),
        };
    }
    if let Some(error_message) = result_worker_error_message {
        return CallbackWorkerErrorRaisePlan {
            should_raise: true,
            raise_dosage_worker_error: false,
            raise_result_worker_error: true,
            error_message: Some(error_message.to_string()),
        };
    }
    CallbackWorkerErrorRaisePlan {
        should_raise: false,
        raise_dosage_worker_error: false,
        raise_result_worker_error: false,
        error_message: None,
    }
}

#[must_use]
pub fn plan_callback_worker_start(has_started: bool) -> CallbackWorkerStartPlan {
    if has_started {
        return CallbackWorkerStartPlan { start_actions: Vec::new() };
    }
    CallbackWorkerStartPlan {
        start_actions: vec![
            CALLBACK_WORKER_START_RESULT_WORKER_ACTION.to_string(),
            CALLBACK_WORKER_START_DOSAGE_WORKER_ACTION.to_string(),
        ],
    }
}

pub(crate) fn plan_callback_worker_start_attempt(
    lifecycle_state: &mut CallbackWorkerLifecycleState,
) -> CallbackWorkerStartAttemptPlan {
    let start_plan = plan_callback_worker_start(lifecycle_state.has_started());
    if !start_plan.should_start() {
        return CallbackWorkerStartAttemptPlan {
            start_actions: Vec::new(),
            has_marked_started: false,
            has_start_error: false,
            error_message: None,
        };
    }
    let has_marked_started = lifecycle_state.mark_started();
    CallbackWorkerStartAttemptPlan {
        start_actions: start_plan.start_actions,
        has_marked_started,
        has_start_error: !has_marked_started,
        error_message: if has_marked_started {
            None
        } else {
            Some("Native callback worker lifecycle was already marked started.".to_string())
        },
    }
}
