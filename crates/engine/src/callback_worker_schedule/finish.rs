use super::timeouts::callback_worker_shutdown_timeouts;
use super::types::{CallbackWorkerAbortPlan, CallbackWorkerFinishAction, CallbackWorkerFinishPlan};

#[must_use]
pub fn plan_callback_worker_finish() -> CallbackWorkerFinishPlan {
    let shutdown_timeouts = callback_worker_shutdown_timeouts();
    CallbackWorkerFinishPlan {
        finish_actions: vec![
            CallbackWorkerFinishAction::StopDosageWorker,
            CallbackWorkerFinishAction::JoinDosageWorker,
            CallbackWorkerFinishAction::StopResultWorker,
            CallbackWorkerFinishAction::JoinResultWorker,
            CallbackWorkerFinishAction::RaiseWorkerError,
            CallbackWorkerFinishAction::CompleteProgress,
            CallbackWorkerFinishAction::EmitBinaryCorrectionSummary,
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
        abort_actions: vec![CallbackWorkerFinishAction::StopDosageWorker, CallbackWorkerFinishAction::StopResultWorker],
        dosage_stop_timeout_seconds: shutdown_timeouts.worker_abort_stop_timeout_seconds,
        result_stop_timeout_seconds: shutdown_timeouts.worker_abort_stop_timeout_seconds,
    }
}
