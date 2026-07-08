use crate::schedule::{
    CallbackQueueGetAttemptPlan, CallbackQueueGetObservationPlan, CallbackQueuePutAttemptPlan,
    CallbackQueuePutObservationPlan, DOSAGE_QUEUE_NAME, RESULT_QUEUE_NAME,
    callback_worker_backpressure_poll_timeout_seconds, plan_callback_queue_get_attempt,
    plan_callback_queue_get_observation, plan_callback_queue_put_attempt, plan_callback_queue_put_observation,
};

use super::CallbackSchedulerState;

impl CallbackSchedulerState {
    #[must_use]
    pub const fn dosage_queue_depth(&self) -> usize {
        self.queue_limits.dosage_queue_depth
    }

    #[must_use]
    pub const fn dosage_queue_capacity(&self) -> usize {
        self.dosage_queue_state.queue_capacity()
    }

    #[must_use]
    pub const fn dosage_queue_occupied_count(&self) -> usize {
        self.dosage_queue_state.occupied_count()
    }

    #[must_use]
    pub const fn has_available_dosage_queue_slot(&self) -> bool {
        self.dosage_queue_state.has_available_slot()
    }

    pub fn acquire_dosage_queue_slot(&mut self) -> bool {
        self.dosage_queue_state.acquire_slot()
    }

    pub fn release_dosage_queue_slot(&mut self) -> bool {
        self.dosage_queue_state.release_slot()
    }

    #[must_use]
    pub fn plan_dosage_queue_put_attempt(&mut self, wait_timeout_seconds: f64) -> CallbackQueuePutAttemptPlan {
        plan_callback_queue_put_attempt(&mut self.dosage_queue_state, wait_timeout_seconds)
    }

    #[must_use]
    pub fn plan_dosage_queue_put_backpressure_attempt(&mut self) -> CallbackQueuePutAttemptPlan {
        self.plan_dosage_queue_put_attempt(callback_worker_backpressure_poll_timeout_seconds())
    }

    #[must_use]
    pub fn plan_dosage_queue_put_observation(&self, queued: bool) -> CallbackQueuePutObservationPlan {
        debug_assert!(self.dosage_queue_state.queue_capacity() > 0);
        plan_callback_queue_put_observation(DOSAGE_QUEUE_NAME, queued)
    }

    #[must_use]
    pub fn plan_dosage_queue_get_attempt(&mut self, has_queued_item: bool) -> CallbackQueueGetAttemptPlan {
        plan_callback_queue_get_attempt(&mut self.dosage_queue_state, has_queued_item)
    }

    #[must_use]
    pub fn plan_dosage_queue_get_observation(&self) -> CallbackQueueGetObservationPlan {
        debug_assert!(self.dosage_queue_state.queue_capacity() > 0);
        plan_callback_queue_get_observation(DOSAGE_QUEUE_NAME)
    }

    #[must_use]
    pub const fn result_queue_depth(&self) -> usize {
        self.queue_limits.result_queue_depth
    }

    #[must_use]
    pub const fn result_queue_capacity(&self) -> usize {
        self.result_queue_state.queue_capacity()
    }

    #[must_use]
    pub const fn result_queue_occupied_count(&self) -> usize {
        self.result_queue_state.occupied_count()
    }

    #[must_use]
    pub const fn has_available_result_queue_slot(&self) -> bool {
        self.result_queue_state.has_available_slot()
    }

    pub fn acquire_result_queue_slot(&mut self) -> bool {
        self.result_queue_state.acquire_slot()
    }

    pub fn release_result_queue_slot(&mut self) -> bool {
        self.result_queue_state.release_slot()
    }

    #[must_use]
    pub fn plan_result_queue_put_attempt(&mut self, wait_timeout_seconds: f64) -> CallbackQueuePutAttemptPlan {
        plan_callback_queue_put_attempt(&mut self.result_queue_state, wait_timeout_seconds)
    }

    #[must_use]
    pub fn plan_result_queue_put_backpressure_attempt(&mut self) -> CallbackQueuePutAttemptPlan {
        self.plan_result_queue_put_attempt(callback_worker_backpressure_poll_timeout_seconds())
    }

    #[must_use]
    pub fn plan_result_queue_put_observation(&self, queued: bool) -> CallbackQueuePutObservationPlan {
        debug_assert!(self.result_queue_state.queue_capacity() > 0);
        plan_callback_queue_put_observation(RESULT_QUEUE_NAME, queued)
    }

    #[must_use]
    pub fn plan_result_queue_get_attempt(&mut self, has_queued_item: bool) -> CallbackQueueGetAttemptPlan {
        plan_callback_queue_get_attempt(&mut self.result_queue_state, has_queued_item)
    }

    #[must_use]
    pub fn plan_result_queue_get_observation(&self) -> CallbackQueueGetObservationPlan {
        debug_assert!(self.result_queue_state.queue_capacity() > 0);
        plan_callback_queue_get_observation(RESULT_QUEUE_NAME)
    }
}
