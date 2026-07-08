use super::types::{
    CallbackWorkerLifecycleState, CallbackWorkerStartAction, CallbackWorkerStartAttemptPlan, CallbackWorkerStartPlan,
};

#[must_use]
pub fn plan_callback_worker_start(has_started: bool) -> CallbackWorkerStartPlan {
    if has_started {
        return CallbackWorkerStartPlan { start_actions: Vec::new() };
    }
    CallbackWorkerStartPlan {
        start_actions: vec![CallbackWorkerStartAction::StartResultWorker, CallbackWorkerStartAction::StartDosageWorker],
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
