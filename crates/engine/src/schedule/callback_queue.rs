use super::{
    QUEUE_CONSUMER_WAIT_OPERATION, QUEUE_PRODUCER_BLOCKING_OPERATION, QUEUE_PUT_OPERATION,
    callback_worker_backpressure_poll_timeout_seconds,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackQueuePutAttemptPlan {
    pub should_put: bool,
    pub should_wait: bool,
    pub wait_timeout_seconds: f64,
    pub queue_depth: usize,
    pub queue_capacity: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackQueuePutObservationPlan {
    pub queue_name: String,
    pub operation_name: String,
    pub blocked: bool,
    pub should_retry_put: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackQueueGetAttemptPlan {
    pub should_get: bool,
    pub should_wait: bool,
    pub has_release_error: bool,
    pub wait_timeout_seconds: f64,
    pub queue_depth: usize,
    pub queue_capacity: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackQueueGetObservationPlan {
    pub queue_name: String,
    pub operation_name: String,
    pub blocked: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackQueueOccupancyState {
    queue_capacity: usize,
    occupied_count: usize,
}

impl CallbackQueueOccupancyState {
    #[must_use]
    pub const fn new(queue_capacity: usize) -> Self {
        Self { queue_capacity, occupied_count: 0 }
    }

    #[must_use]
    pub const fn queue_capacity(&self) -> usize {
        self.queue_capacity
    }

    #[must_use]
    pub const fn occupied_count(&self) -> usize {
        self.occupied_count
    }

    #[must_use]
    pub const fn has_available_slot(&self) -> bool {
        self.occupied_count < self.queue_capacity
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

pub(super) fn plan_callback_queue_put_attempt(
    queue_state: &mut CallbackQueueOccupancyState,
    wait_timeout_seconds: f64,
) -> CallbackQueuePutAttemptPlan {
    if queue_state.acquire_slot() {
        return CallbackQueuePutAttemptPlan {
            should_put: true,
            should_wait: false,
            wait_timeout_seconds: 0.0,
            queue_depth: queue_state.occupied_count(),
            queue_capacity: queue_state.queue_capacity(),
        };
    }
    let normalized_wait_timeout_seconds = super::normalize_callback_queue_wait_timeout_seconds(wait_timeout_seconds);
    CallbackQueuePutAttemptPlan {
        should_put: false,
        should_wait: normalized_wait_timeout_seconds > 0.0,
        wait_timeout_seconds: normalized_wait_timeout_seconds,
        queue_depth: queue_state.occupied_count(),
        queue_capacity: queue_state.queue_capacity(),
    }
}

#[must_use]
pub fn plan_callback_queue_put_observation(queue_name: &str, queued: bool) -> CallbackQueuePutObservationPlan {
    if queued {
        return CallbackQueuePutObservationPlan {
            queue_name: queue_name.to_string(),
            operation_name: QUEUE_PUT_OPERATION.to_string(),
            blocked: false,
            should_retry_put: false,
        };
    }
    CallbackQueuePutObservationPlan {
        queue_name: queue_name.to_string(),
        operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
        blocked: true,
        should_retry_put: true,
    }
}

pub(super) fn plan_callback_queue_get_attempt(
    queue_state: &mut CallbackQueueOccupancyState,
    has_queued_item: bool,
) -> CallbackQueueGetAttemptPlan {
    if has_queued_item {
        let released_slot = queue_state.release_slot();
        return CallbackQueueGetAttemptPlan {
            should_get: released_slot,
            should_wait: false,
            has_release_error: !released_slot,
            wait_timeout_seconds: 0.0,
            queue_depth: queue_state.occupied_count(),
            queue_capacity: queue_state.queue_capacity(),
        };
    }
    CallbackQueueGetAttemptPlan {
        should_get: false,
        should_wait: true,
        has_release_error: false,
        wait_timeout_seconds: callback_worker_backpressure_poll_timeout_seconds(),
        queue_depth: queue_state.occupied_count(),
        queue_capacity: queue_state.queue_capacity(),
    }
}

#[must_use]
pub fn plan_callback_queue_get_observation(queue_name: &str) -> CallbackQueueGetObservationPlan {
    CallbackQueueGetObservationPlan {
        queue_name: queue_name.to_string(),
        operation_name: QUEUE_CONSUMER_WAIT_OPERATION.to_string(),
        blocked: true,
    }
}
