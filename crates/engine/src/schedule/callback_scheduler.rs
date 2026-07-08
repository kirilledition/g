use super::{
    CallbackQueueOccupancyState, CallbackWorkerLifecycleState, DosageBufferPoolState, ResultInFlightSlotState,
    ScheduleError,
};

mod dosage_buffers;
mod dosage_work;
mod limits;
mod observations;
mod queues;
mod result_slots;
mod result_write;
mod worker;

pub use limits::{NativeCallbackQueueLimits, resolve_native_callback_queue_limits};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackSchedulerState {
    pub(super) queue_limits: NativeCallbackQueueLimits,
    pub(super) native_callback_batch_size: usize,
    pub(super) dosage_queue_state: CallbackQueueOccupancyState,
    pub(super) result_queue_state: CallbackQueueOccupancyState,
    pub(super) result_in_flight_slot_state: ResultInFlightSlotState,
    pub(super) dosage_buffer_pool_state: DosageBufferPoolState,
    pub(super) worker_lifecycle_state: CallbackWorkerLifecycleState,
    pub(super) dosage_worker_error_message: Option<String>,
    pub(super) result_worker_error_message: Option<String>,
}

impl CallbackSchedulerState {
    /// Build the native callback scheduler state for one callback runner.
    ///
    /// # Errors
    ///
    /// Returns an error when queue limits or bounded resource limits are invalid.
    pub fn new(
        staging_depth: i64,
        native_callback_batch_size: i64,
        result_in_flight_limit: Option<i64>,
        dosage_buffer_limit: Option<i64>,
    ) -> Result<Self, ScheduleError> {
        let queue_limits = resolve_native_callback_queue_limits(
            staging_depth,
            native_callback_batch_size,
            result_in_flight_limit,
            dosage_buffer_limit,
        )?;
        let native_callback_batch_size = usize::try_from(native_callback_batch_size).map_err(|_| {
            ScheduleError::CallbackBatchSizeOverflow { callback_batch_size: native_callback_batch_size }
        })?;
        Ok(Self {
            queue_limits,
            native_callback_batch_size,
            dosage_queue_state: CallbackQueueOccupancyState::new(queue_limits.dosage_queue_depth),
            result_queue_state: CallbackQueueOccupancyState::new(queue_limits.result_queue_depth),
            result_in_flight_slot_state: ResultInFlightSlotState::new(queue_limits.result_in_flight_limit),
            dosage_buffer_pool_state: DosageBufferPoolState::new(queue_limits.dosage_buffer_limit),
            worker_lifecycle_state: CallbackWorkerLifecycleState::new(),
            dosage_worker_error_message: None,
            result_worker_error_message: None,
        })
    }

    #[must_use]
    pub const fn queue_limits(&self) -> NativeCallbackQueueLimits {
        self.queue_limits
    }

    #[must_use]
    pub const fn native_callback_batch_size(&self) -> usize {
        self.native_callback_batch_size
    }

    #[must_use]
    pub const fn result_in_flight_limit(&self) -> usize {
        self.queue_limits.result_in_flight_limit
    }

    #[must_use]
    pub const fn dosage_buffer_limit(&self) -> usize {
        self.queue_limits.dosage_buffer_limit
    }
}
