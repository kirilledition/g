use super::types::CallbackWorkerShutdownTimeouts;

pub(crate) const CALLBACK_WORKER_BACKPRESSURE_POLL_TIMEOUT_SECONDS: f64 = 0.1;
pub(crate) const CALLBACK_WORKER_STOP_POLL_TIMEOUT_CAP_SECONDS: f64 = 0.1;

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
