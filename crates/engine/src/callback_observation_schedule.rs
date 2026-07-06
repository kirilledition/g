//! Callback queue and bounded-resource observation policy.

use crate::schedule::ScheduleError;

pub(crate) const DOSAGE_QUEUE_NAME: &str = "dosage_queue";
pub(crate) const RESULT_QUEUE_NAME: &str = "result_queue";
pub(crate) const DOSAGE_BUFFER_POOL_NAME: &str = "dosage_buffer_pool";
pub(crate) const RESULT_IN_FLIGHT_SLOTS_NAME: &str = "result_in_flight_slots";
pub(crate) const QUEUE_PUT_OPERATION: &str = "put";
pub(crate) const QUEUE_PRODUCER_BLOCKING_OPERATION: &str = "producer_blocking";
pub(crate) const QUEUE_CONSUMER_WAIT_OPERATION: &str = "consumer_wait";
pub(crate) const QUEUE_REUSE_OPERATION: &str = "reuse";
pub(crate) const QUEUE_RETURN_OPERATION: &str = "return";
pub(crate) const QUEUE_RETURN_FULL_OPERATION: &str = "return_full";
pub(crate) const QUEUE_ALLOCATE_OPERATION: &str = "allocate";
pub(crate) const QUEUE_DISCARD_OPERATION: &str = "discard";
pub(crate) const RESULT_SLOT_ACQUIRE_OPERATION: &str = "acquire";
pub(crate) const RESULT_SLOT_RELEASE_OPERATION: &str = "release";

#[derive(Clone, Debug, PartialEq)]
pub struct CallbackQueueStageObservationPlan {
    pub queue_name: String,
    pub operation_name: String,
    pub stage_name: String,
    pub blocked_seconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallbackQueueOperationObservationPlan {
    pub queue_name: String,
    pub operation_name: String,
    pub blocked_seconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallbackQueueBackpressureObservation {
    pub queue_name: String,
    pub operation_name: String,
    pub queue_depth: usize,
    pub queue_capacity: usize,
    pub elapsed_seconds: f64,
    pub blocked_seconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallbackQueueStageBackpressureObservation {
    pub queue_name: String,
    pub operation_name: String,
    pub stage_name: String,
    pub queue_depth: usize,
    pub queue_capacity: usize,
    pub elapsed_seconds: f64,
    pub blocked_seconds: f64,
}

fn resolve_callback_queue_stage_name(queue_name: &str, operation_name: &str) -> Option<&'static str> {
    match (queue_name, operation_name) {
        (DOSAGE_QUEUE_NAME, QUEUE_PUT_OPERATION) => Some("callback_queue_put"),
        (DOSAGE_QUEUE_NAME, QUEUE_PRODUCER_BLOCKING_OPERATION) => Some("callback_queue_producer_blocking"),
        (DOSAGE_QUEUE_NAME, QUEUE_CONSUMER_WAIT_OPERATION) => Some("callback_queue_consumer_wait"),
        (RESULT_QUEUE_NAME, QUEUE_PUT_OPERATION) => Some("result_queue_put"),
        (RESULT_QUEUE_NAME, QUEUE_PRODUCER_BLOCKING_OPERATION) => Some("result_queue_producer_blocking"),
        (RESULT_QUEUE_NAME, QUEUE_CONSUMER_WAIT_OPERATION) => Some("result_queue_consumer_wait"),
        (DOSAGE_BUFFER_POOL_NAME, QUEUE_CONSUMER_WAIT_OPERATION) => Some("dosage_buffer_pool_consumer_wait"),
        (RESULT_IN_FLIGHT_SLOTS_NAME, RESULT_SLOT_ACQUIRE_OPERATION) => Some("result_in_flight_slot_acquire"),
        (RESULT_IN_FLIGHT_SLOTS_NAME, QUEUE_PRODUCER_BLOCKING_OPERATION) => Some("result_in_flight_producer_blocking"),
        _ => None,
    }
}

fn callback_queue_operation_is_supported(queue_name: &str, operation_name: &str) -> bool {
    matches!(
        (queue_name, operation_name),
        (
            DOSAGE_QUEUE_NAME | RESULT_QUEUE_NAME,
            QUEUE_PUT_OPERATION | QUEUE_PRODUCER_BLOCKING_OPERATION | QUEUE_CONSUMER_WAIT_OPERATION,
        ) | (
            DOSAGE_BUFFER_POOL_NAME,
            QUEUE_CONSUMER_WAIT_OPERATION
                | QUEUE_REUSE_OPERATION
                | QUEUE_RETURN_OPERATION
                | QUEUE_RETURN_FULL_OPERATION
                | QUEUE_ALLOCATE_OPERATION
                | QUEUE_DISCARD_OPERATION,
        ) | (
            RESULT_IN_FLIGHT_SLOTS_NAME,
            RESULT_SLOT_ACQUIRE_OPERATION | QUEUE_PRODUCER_BLOCKING_OPERATION | RESULT_SLOT_RELEASE_OPERATION,
        )
    )
}

/// Plan one aggregate callback queue or bounded-resource observation.
///
/// # Errors
///
/// Returns an error when the queue/resource and operation pair is not part of
/// the callback scheduler observation contract.
pub fn plan_callback_queue_operation_observation(
    queue_name: &str,
    operation_name: &str,
    elapsed_seconds: f64,
    blocked: bool,
) -> Result<CallbackQueueOperationObservationPlan, ScheduleError> {
    if !callback_queue_operation_is_supported(queue_name, operation_name) {
        return Err(ScheduleError::UnsupportedCallbackQueueOperation {
            queue_name: queue_name.to_string(),
            operation_name: operation_name.to_string(),
        });
    }
    Ok(CallbackQueueOperationObservationPlan {
        queue_name: queue_name.to_string(),
        operation_name: operation_name.to_string(),
        blocked_seconds: if blocked { elapsed_seconds } else { 0.0 },
    })
}

/// Plan one complete aggregate callback queue or bounded-resource observation payload.
///
/// # Errors
///
/// Returns an error when the queue/resource and operation pair is not part of
/// the callback scheduler observation contract.
pub fn plan_callback_queue_backpressure_observation(
    queue_name: &str,
    operation_name: &str,
    queue_depth: usize,
    queue_capacity: usize,
    elapsed_seconds: f64,
    blocked: bool,
) -> Result<CallbackQueueBackpressureObservation, ScheduleError> {
    let operation_plan =
        plan_callback_queue_operation_observation(queue_name, operation_name, elapsed_seconds, blocked)?;
    Ok(CallbackQueueBackpressureObservation {
        queue_name: operation_plan.queue_name,
        operation_name: operation_plan.operation_name,
        queue_depth,
        queue_capacity,
        elapsed_seconds,
        blocked_seconds: operation_plan.blocked_seconds,
    })
}

/// Plan one timed callback queue or bounded-resource observation.
///
/// # Errors
///
/// Returns an error when the queue/resource and operation pair does not have a
/// canonical callback timing stage.
pub fn plan_callback_queue_stage_observation(
    queue_name: &str,
    operation_name: &str,
    elapsed_seconds: f64,
    blocked: bool,
) -> Result<CallbackQueueStageObservationPlan, ScheduleError> {
    let Some(stage_name) = resolve_callback_queue_stage_name(queue_name, operation_name) else {
        return Err(ScheduleError::UnsupportedCallbackQueueStageOperation {
            queue_name: queue_name.to_string(),
            operation_name: operation_name.to_string(),
        });
    };
    let operation_plan =
        plan_callback_queue_operation_observation(queue_name, operation_name, elapsed_seconds, blocked)?;
    Ok(CallbackQueueStageObservationPlan {
        queue_name: operation_plan.queue_name,
        operation_name: operation_plan.operation_name,
        stage_name: stage_name.to_string(),
        blocked_seconds: operation_plan.blocked_seconds,
    })
}

/// Plan one complete timed callback queue or bounded-resource observation payload.
///
/// # Errors
///
/// Returns an error when the queue/resource and operation pair does not have a
/// canonical callback timing stage.
pub fn plan_callback_queue_stage_backpressure_observation(
    queue_name: &str,
    operation_name: &str,
    queue_depth: usize,
    queue_capacity: usize,
    elapsed_seconds: f64,
    blocked: bool,
) -> Result<CallbackQueueStageBackpressureObservation, ScheduleError> {
    let stage_plan = plan_callback_queue_stage_observation(queue_name, operation_name, elapsed_seconds, blocked)?;
    Ok(CallbackQueueStageBackpressureObservation {
        queue_name: stage_plan.queue_name,
        operation_name: stage_plan.operation_name,
        stage_name: stage_plan.stage_name,
        queue_depth,
        queue_capacity,
        elapsed_seconds,
        blocked_seconds: stage_plan.blocked_seconds,
    })
}
