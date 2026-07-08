use crate::schedule::{
    CallbackQueueBackpressureObservation, CallbackQueueStageBackpressureObservation, DOSAGE_QUEUE_NAME,
    RESULT_IN_FLIGHT_SLOTS_NAME, RESULT_QUEUE_NAME, ScheduleError, plan_callback_queue_backpressure_observation,
    plan_callback_queue_stage_backpressure_observation,
};

use super::CallbackSchedulerState;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CallbackBoundedResourceOccupancy {
    queue_depth: usize,
    queue_capacity: usize,
}

impl CallbackSchedulerState {
    /// Plan a callback queue or result-slot observation using native occupancy.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue/resource and operation pair is not part
    /// of the native-owned callback scheduler observation contract.
    pub fn plan_current_queue_backpressure_observation(
        &self,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueBackpressureObservation, ScheduleError> {
        let occupancy = self.current_queue_occupancy(queue_name, operation_name)?;
        plan_callback_queue_backpressure_observation(
            queue_name,
            operation_name,
            occupancy.queue_depth,
            occupancy.queue_capacity,
            elapsed_seconds,
            blocked,
        )
    }

    /// Plan a dosage-buffer pool observation using Python-owned free depth.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation is not part of the dosage-buffer
    /// pool observation contract.
    pub fn plan_dosage_buffer_pool_backpressure_observation(
        &self,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueBackpressureObservation, ScheduleError> {
        self.dosage_buffer_pool_backpressure_observation(operation_name, free_buffer_count, elapsed_seconds, blocked)
    }

    /// Plan a timed callback queue or result-slot observation using native occupancy.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue/resource and operation pair does not
    /// have a canonical callback timing stage in the native scheduler contract.
    pub fn plan_current_queue_stage_backpressure_observation(
        &self,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueStageBackpressureObservation, ScheduleError> {
        let occupancy = self.current_queue_occupancy(queue_name, operation_name)?;
        plan_callback_queue_stage_backpressure_observation(
            queue_name,
            operation_name,
            occupancy.queue_depth,
            occupancy.queue_capacity,
            elapsed_seconds,
            blocked,
        )
    }

    /// Plan a timed dosage-buffer pool observation using Python-owned free depth.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation does not have a canonical
    /// dosage-buffer pool timing stage.
    pub fn plan_dosage_buffer_pool_stage_backpressure_observation(
        &self,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueStageBackpressureObservation, ScheduleError> {
        self.dosage_buffer_pool_stage_backpressure_observation(
            operation_name,
            free_buffer_count,
            elapsed_seconds,
            blocked,
        )
    }

    fn current_queue_occupancy(
        &self,
        queue_name: &str,
        operation_name: &str,
    ) -> Result<CallbackBoundedResourceOccupancy, ScheduleError> {
        let occupancy = match queue_name {
            DOSAGE_QUEUE_NAME => CallbackBoundedResourceOccupancy {
                queue_depth: self.dosage_queue_state.occupied_count(),
                queue_capacity: self.dosage_queue_state.queue_capacity(),
            },
            RESULT_QUEUE_NAME => CallbackBoundedResourceOccupancy {
                queue_depth: self.result_queue_state.occupied_count(),
                queue_capacity: self.result_queue_state.queue_capacity(),
            },
            RESULT_IN_FLIGHT_SLOTS_NAME => CallbackBoundedResourceOccupancy {
                queue_depth: self.result_in_flight_slot_state.occupied_count(),
                queue_capacity: self.result_in_flight_slot_state.slot_limit(),
            },
            _ => {
                return Err(ScheduleError::UnsupportedCallbackQueueOperation {
                    queue_name: queue_name.to_string(),
                    operation_name: operation_name.to_string(),
                });
            }
        };
        Ok(occupancy)
    }
}
