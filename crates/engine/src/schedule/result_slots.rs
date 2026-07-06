use super::{
    QUEUE_PRODUCER_BLOCKING_OPERATION, RESULT_IN_FLIGHT_SLOTS_NAME, RESULT_SLOT_ACQUIRE_OPERATION,
    RESULT_SLOT_RELEASE_OPERATION, normalize_callback_queue_wait_timeout_seconds,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResultInFlightAcquireAttemptPlan {
    pub should_acquire: bool,
    pub should_wait: bool,
    pub wait_timeout_seconds: f64,
    pub occupied_count: usize,
    pub slot_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultInFlightAcquireObservationPlan {
    pub resource_name: String,
    pub operation_name: String,
    pub blocked: bool,
    pub should_retry_acquisition: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResultInFlightReleaseAttemptPlan {
    pub should_release: bool,
    pub has_release_error: bool,
    pub occupied_count: usize,
    pub slot_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultInFlightReleaseObservationPlan {
    pub resource_name: String,
    pub operation_name: String,
    pub blocked: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultInFlightSlotState {
    slot_limit: usize,
    occupied_count: usize,
}

impl ResultInFlightSlotState {
    #[must_use]
    pub const fn new(slot_limit: usize) -> Self {
        Self { slot_limit, occupied_count: 0 }
    }

    #[must_use]
    pub const fn slot_limit(&self) -> usize {
        self.slot_limit
    }

    #[must_use]
    pub const fn occupied_count(&self) -> usize {
        self.occupied_count
    }

    #[must_use]
    pub const fn has_available_slot(&self) -> bool {
        self.occupied_count < self.slot_limit
    }

    pub fn acquire_slot(&mut self) -> bool {
        if !self.has_available_slot() {
            return false;
        }
        self.occupied_count += 1;
        true
    }

    pub fn release_slot(&mut self) -> bool {
        if self.occupied_count == 0 {
            return false;
        }
        self.occupied_count -= 1;
        true
    }
}

pub(super) fn plan_result_in_flight_slot_acquire_attempt(
    slot_state: &mut ResultInFlightSlotState,
    wait_timeout_seconds: f64,
) -> ResultInFlightAcquireAttemptPlan {
    if slot_state.acquire_slot() {
        return ResultInFlightAcquireAttemptPlan {
            should_acquire: true,
            should_wait: false,
            wait_timeout_seconds: 0.0,
            occupied_count: slot_state.occupied_count(),
            slot_limit: slot_state.slot_limit(),
        };
    }
    let normalized_wait_timeout_seconds = normalize_callback_queue_wait_timeout_seconds(wait_timeout_seconds);
    ResultInFlightAcquireAttemptPlan {
        should_acquire: false,
        should_wait: normalized_wait_timeout_seconds > 0.0,
        wait_timeout_seconds: normalized_wait_timeout_seconds,
        occupied_count: slot_state.occupied_count(),
        slot_limit: slot_state.slot_limit(),
    }
}

#[must_use]
pub fn plan_result_in_flight_slot_acquire_observation(
    acquire_attempt_plan: &ResultInFlightAcquireAttemptPlan,
) -> ResultInFlightAcquireObservationPlan {
    if acquire_attempt_plan.should_acquire {
        return ResultInFlightAcquireObservationPlan {
            resource_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
            operation_name: RESULT_SLOT_ACQUIRE_OPERATION.to_string(),
            blocked: false,
            should_retry_acquisition: false,
        };
    }
    ResultInFlightAcquireObservationPlan {
        resource_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
        operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
        blocked: true,
        should_retry_acquisition: true,
    }
}

pub(super) fn plan_result_in_flight_slot_release_attempt(
    slot_state: &mut ResultInFlightSlotState,
) -> ResultInFlightReleaseAttemptPlan {
    let released_slot = slot_state.release_slot();
    ResultInFlightReleaseAttemptPlan {
        should_release: released_slot,
        has_release_error: !released_slot,
        occupied_count: slot_state.occupied_count(),
        slot_limit: slot_state.slot_limit(),
    }
}

#[must_use]
pub fn plan_result_in_flight_slot_release_observation() -> ResultInFlightReleaseObservationPlan {
    ResultInFlightReleaseObservationPlan {
        resource_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
        operation_name: RESULT_SLOT_RELEASE_OPERATION.to_string(),
        blocked: false,
    }
}
