use super::timeouts::{callback_worker_shutdown_timeouts, resolve_callback_worker_stop_poll_timeout_seconds};
use super::types::{CallbackWorkerJoinPlan, CallbackWorkerStopPlan, CallbackWorkerStopPollPlan};

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
