//! Callback worker lifecycle, shutdown, and error propagation policy.

#![allow(clippy::module_name_repetitions)]

mod errors;
mod finish;
mod start;
mod stop;
mod timeouts;
mod types;

pub use errors::{format_dosage_callback_worker_error_message, format_result_callback_worker_error_message};
pub(crate) use errors::{plan_callback_worker_error_raise, update_callback_worker_error};
pub use finish::{plan_callback_worker_abort, plan_callback_worker_finish};
pub use start::plan_callback_worker_start;
pub(crate) use start::plan_callback_worker_start_attempt;
pub use stop::{
    plan_callback_worker_stop_poll, plan_dosage_callback_worker_join, plan_dosage_callback_worker_stop,
    plan_result_callback_worker_join, plan_result_callback_worker_stop, should_attempt_callback_worker_stop,
};
pub use timeouts::{
    callback_worker_backpressure_poll_timeout_seconds, callback_worker_shutdown_timeouts,
    resolve_callback_worker_stop_poll_timeout_seconds,
};
pub use types::{
    CallbackWorkerAbortPlan, CallbackWorkerErrorRaisePlan, CallbackWorkerErrorUpdatePlan, CallbackWorkerFinishAction,
    CallbackWorkerFinishPlan, CallbackWorkerJoinPlan, CallbackWorkerLifecycleState, CallbackWorkerShutdownTimeouts,
    CallbackWorkerStartAction, CallbackWorkerStartAttemptPlan, CallbackWorkerStartPlan, CallbackWorkerStopPlan,
    CallbackWorkerStopPollPlan,
};
