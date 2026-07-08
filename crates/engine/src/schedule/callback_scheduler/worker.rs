use crate::schedule::{
    CallbackWorkerAbortPlan, CallbackWorkerErrorRaisePlan, CallbackWorkerErrorUpdatePlan, CallbackWorkerFinishPlan,
    CallbackWorkerJoinPlan, CallbackWorkerStartAttemptPlan, CallbackWorkerStartPlan, CallbackWorkerStopPlan,
    CallbackWorkerStopPollPlan, callback_worker_backpressure_poll_timeout_seconds,
    format_dosage_callback_worker_error_message, format_result_callback_worker_error_message,
    plan_callback_worker_abort, plan_callback_worker_error_raise, plan_callback_worker_finish,
    plan_callback_worker_start, plan_callback_worker_start_attempt, plan_callback_worker_stop_poll,
    plan_dosage_callback_worker_join, plan_dosage_callback_worker_stop, plan_result_callback_worker_join,
    plan_result_callback_worker_stop, update_callback_worker_error,
};

use super::CallbackSchedulerState;

impl CallbackSchedulerState {
    #[must_use]
    pub const fn has_started(&self) -> bool {
        self.worker_lifecycle_state.has_started()
    }

    pub fn mark_started(&mut self) -> bool {
        self.worker_lifecycle_state.mark_started()
    }

    #[must_use]
    pub fn plan_worker_start(&self) -> CallbackWorkerStartPlan {
        plan_callback_worker_start(self.has_started())
    }

    #[must_use]
    pub fn plan_worker_start_attempt(&mut self) -> CallbackWorkerStartAttemptPlan {
        plan_callback_worker_start_attempt(&mut self.worker_lifecycle_state)
    }

    #[must_use]
    pub fn dosage_worker_error_message(&self) -> Option<&str> {
        self.dosage_worker_error_message.as_deref()
    }

    #[must_use]
    pub fn result_worker_error_message(&self) -> Option<&str> {
        self.result_worker_error_message.as_deref()
    }

    #[must_use]
    pub fn has_dosage_worker_error(&self) -> bool {
        self.dosage_worker_error_message.is_some()
    }

    #[must_use]
    pub fn has_result_worker_error(&self) -> bool {
        self.result_worker_error_message.is_some()
    }

    #[must_use]
    pub fn plan_worker_error_raise(&self) -> CallbackWorkerErrorRaisePlan {
        plan_callback_worker_error_raise(self.dosage_worker_error_message(), self.result_worker_error_message())
    }

    pub fn record_dosage_worker_error(&mut self, error_message: &str) {
        self.dosage_worker_error_message = Some(format_dosage_callback_worker_error_message(error_message));
    }

    pub fn record_result_worker_error(&mut self, error_message: &str) {
        self.result_worker_error_message = Some(format_result_callback_worker_error_message(error_message));
    }

    pub fn update_dosage_worker_error(&mut self, error_message: Option<&str>) -> CallbackWorkerErrorUpdatePlan {
        update_callback_worker_error(
            &mut self.dosage_worker_error_message,
            error_message,
            format_dosage_callback_worker_error_message,
        )
    }

    pub fn update_result_worker_error(&mut self, error_message: Option<&str>) -> CallbackWorkerErrorUpdatePlan {
        update_callback_worker_error(
            &mut self.result_worker_error_message,
            error_message,
            format_result_callback_worker_error_message,
        )
    }

    pub fn clear_dosage_worker_error(&mut self) -> bool {
        let had_error = self.has_dosage_worker_error();
        self.dosage_worker_error_message = None;
        had_error
    }

    pub fn clear_result_worker_error(&mut self) -> bool {
        let had_error = self.has_result_worker_error();
        self.result_worker_error_message = None;
        had_error
    }

    #[must_use]
    pub const fn backpressure_poll_timeout_seconds(&self) -> f64 {
        callback_worker_backpressure_poll_timeout_seconds()
    }

    #[must_use]
    pub fn plan_worker_finish(&self) -> CallbackWorkerFinishPlan {
        plan_callback_worker_finish()
    }

    #[must_use]
    pub fn plan_worker_abort(&self) -> CallbackWorkerAbortPlan {
        plan_callback_worker_abort()
    }

    #[must_use]
    pub fn plan_dosage_worker_join(&self, timeout_seconds: Option<f64>) -> CallbackWorkerJoinPlan {
        plan_dosage_callback_worker_join(timeout_seconds, self.has_started())
    }

    #[must_use]
    pub fn plan_result_worker_join(&self, timeout_seconds: Option<f64>) -> CallbackWorkerJoinPlan {
        plan_result_callback_worker_join(timeout_seconds, self.has_started())
    }

    #[must_use]
    pub fn plan_dosage_worker_stop(
        &self,
        timeout_seconds: Option<f64>,
        is_worker_alive: bool,
    ) -> CallbackWorkerStopPlan {
        plan_dosage_callback_worker_stop(
            timeout_seconds,
            self.has_started(),
            self.has_dosage_worker_error(),
            is_worker_alive,
        )
    }

    #[must_use]
    pub fn plan_result_worker_stop(
        &self,
        timeout_seconds: Option<f64>,
        is_worker_alive: bool,
    ) -> CallbackWorkerStopPlan {
        plan_result_callback_worker_stop(
            timeout_seconds,
            self.has_started(),
            self.has_result_worker_error(),
            is_worker_alive,
        )
    }

    #[must_use]
    pub fn plan_dosage_worker_stop_poll(
        &self,
        remaining_timeout_seconds: f64,
        is_worker_alive: bool,
    ) -> CallbackWorkerStopPollPlan {
        plan_callback_worker_stop_poll(
            remaining_timeout_seconds,
            self.has_started(),
            self.has_dosage_worker_error(),
            is_worker_alive,
        )
    }

    #[must_use]
    pub fn plan_result_worker_stop_poll(
        &self,
        remaining_timeout_seconds: f64,
        is_worker_alive: bool,
    ) -> CallbackWorkerStopPollPlan {
        plan_callback_worker_stop_poll(
            remaining_timeout_seconds,
            self.has_started(),
            self.has_result_worker_error(),
            is_worker_alive,
        )
    }
}
