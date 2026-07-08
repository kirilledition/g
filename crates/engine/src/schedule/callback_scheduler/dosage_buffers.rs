use crate::schedule::{
    DOSAGE_BUFFER_POOL_NAME, DosageBufferAcquireAttemptPlan, DosageBufferDiscardAttemptPlan,
    DosageBufferPoolObservationPlan, DosageBufferRegisterAttemptPlan, DosageBufferReturnAttemptPlan,
    QUEUE_ALLOCATE_OPERATION, QUEUE_CONSUMER_WAIT_OPERATION, QUEUE_DISCARD_OPERATION, QUEUE_RETURN_OPERATION,
    QUEUE_REUSE_OPERATION, callback_worker_backpressure_poll_timeout_seconds, plan_dosage_buffer_acquire_attempt,
    plan_dosage_buffer_discard_attempt, plan_dosage_buffer_pool_observation, plan_dosage_buffer_register_attempt,
    plan_dosage_buffer_return_attempt,
};

use super::CallbackSchedulerState;

impl CallbackSchedulerState {
    #[must_use]
    pub const fn dosage_buffer_pool_limit(&self) -> usize {
        self.dosage_buffer_pool_state.buffer_limit()
    }

    #[must_use]
    pub fn dosage_buffer_allocated_count(&self) -> usize {
        self.dosage_buffer_pool_state.allocated_count()
    }

    #[must_use]
    pub fn dosage_buffer_identifiers(&self) -> Vec<usize> {
        self.dosage_buffer_pool_state.buffer_identifiers()
    }

    #[must_use]
    pub fn has_available_dosage_buffer_slot(&self) -> bool {
        self.dosage_buffer_pool_state.has_available_slot()
    }

    #[must_use]
    pub fn owns_dosage_buffer(&self, buffer_identifier: usize) -> bool {
        self.dosage_buffer_pool_state.owns_buffer(buffer_identifier)
    }

    pub fn register_dosage_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.dosage_buffer_pool_state.register_buffer(buffer_identifier)
    }

    pub fn discard_dosage_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.dosage_buffer_pool_state.discard_buffer(buffer_identifier)
    }

    #[must_use]
    pub fn plan_dosage_buffer_acquire_attempt(
        &self,
        free_buffer_count: usize,
        wait_timeout_seconds: f64,
    ) -> DosageBufferAcquireAttemptPlan {
        plan_dosage_buffer_acquire_attempt(&self.dosage_buffer_pool_state, free_buffer_count, wait_timeout_seconds)
    }

    #[must_use]
    pub fn plan_dosage_buffer_acquire_backpressure_attempt(
        &self,
        free_buffer_count: usize,
    ) -> DosageBufferAcquireAttemptPlan {
        self.plan_dosage_buffer_acquire_attempt(free_buffer_count, callback_worker_backpressure_poll_timeout_seconds())
    }

    #[must_use]
    pub fn plan_dosage_buffer_register_attempt(&mut self, buffer_identifier: usize) -> DosageBufferRegisterAttemptPlan {
        plan_dosage_buffer_register_attempt(&mut self.dosage_buffer_pool_state, buffer_identifier)
    }

    #[must_use]
    pub fn plan_dosage_buffer_return_attempt(&self, buffer_identifier: usize) -> DosageBufferReturnAttemptPlan {
        plan_dosage_buffer_return_attempt(&self.dosage_buffer_pool_state, buffer_identifier)
    }

    #[must_use]
    pub fn plan_dosage_buffer_discard_attempt(&mut self, buffer_identifier: usize) -> DosageBufferDiscardAttemptPlan {
        plan_dosage_buffer_discard_attempt(&mut self.dosage_buffer_pool_state, buffer_identifier)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_reuse_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_REUSE_OPERATION, false)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_return_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_RETURN_OPERATION, false)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_allocate_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_ALLOCATE_OPERATION, false)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_discard_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_DISCARD_OPERATION, false)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_consumer_wait_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_CONSUMER_WAIT_OPERATION, true)
    }

    pub(super) fn dosage_buffer_pool_backpressure_observation(
        &self,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<crate::schedule::CallbackQueueBackpressureObservation, crate::schedule::ScheduleError> {
        crate::schedule::plan_callback_queue_backpressure_observation(
            DOSAGE_BUFFER_POOL_NAME,
            operation_name,
            free_buffer_count,
            self.dosage_buffer_pool_state.buffer_limit(),
            elapsed_seconds,
            blocked,
        )
    }

    pub(super) fn dosage_buffer_pool_stage_backpressure_observation(
        &self,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<crate::schedule::CallbackQueueStageBackpressureObservation, crate::schedule::ScheduleError> {
        crate::schedule::plan_callback_queue_stage_backpressure_observation(
            DOSAGE_BUFFER_POOL_NAME,
            operation_name,
            free_buffer_count,
            self.dosage_buffer_pool_state.buffer_limit(),
            elapsed_seconds,
            blocked,
        )
    }
}
