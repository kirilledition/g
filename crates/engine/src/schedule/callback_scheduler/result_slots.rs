use crate::schedule::{
    ResultInFlightAcquireAttemptPlan, ResultInFlightAcquireObservationPlan, ResultInFlightReleaseAttemptPlan,
    ResultInFlightReleaseObservationPlan, callback_worker_backpressure_poll_timeout_seconds,
    plan_result_in_flight_slot_acquire_attempt, plan_result_in_flight_slot_acquire_observation,
    plan_result_in_flight_slot_release_attempt, plan_result_in_flight_slot_release_observation,
};

use super::CallbackSchedulerState;

impl CallbackSchedulerState {
    #[must_use]
    pub const fn result_in_flight_slot_limit(&self) -> usize {
        self.result_in_flight_slot_state.slot_limit()
    }

    #[must_use]
    pub const fn result_in_flight_occupied_count(&self) -> usize {
        self.result_in_flight_slot_state.occupied_count()
    }

    #[must_use]
    pub const fn has_available_result_in_flight_slot(&self) -> bool {
        self.result_in_flight_slot_state.has_available_slot()
    }

    pub fn acquire_result_in_flight_slot(&mut self) -> bool {
        self.result_in_flight_slot_state.acquire_slot()
    }

    pub fn release_result_in_flight_slot(&mut self) -> bool {
        self.result_in_flight_slot_state.release_slot()
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_acquire_attempt(
        &mut self,
        wait_timeout_seconds: f64,
    ) -> ResultInFlightAcquireAttemptPlan {
        plan_result_in_flight_slot_acquire_attempt(&mut self.result_in_flight_slot_state, wait_timeout_seconds)
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_acquire_backpressure_attempt(&mut self) -> ResultInFlightAcquireAttemptPlan {
        self.plan_result_in_flight_slot_acquire_attempt(callback_worker_backpressure_poll_timeout_seconds())
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_acquire_observation(
        &self,
        acquire_attempt_plan: &ResultInFlightAcquireAttemptPlan,
    ) -> ResultInFlightAcquireObservationPlan {
        debug_assert_eq!(acquire_attempt_plan.slot_limit, self.result_in_flight_slot_state.slot_limit());
        plan_result_in_flight_slot_acquire_observation(acquire_attempt_plan)
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_release_attempt(&mut self) -> ResultInFlightReleaseAttemptPlan {
        plan_result_in_flight_slot_release_attempt(&mut self.result_in_flight_slot_state)
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_release_observation(&self) -> ResultInFlightReleaseObservationPlan {
        debug_assert!(self.result_in_flight_slot_state.slot_limit() > 0);
        plan_result_in_flight_slot_release_observation()
    }
}
