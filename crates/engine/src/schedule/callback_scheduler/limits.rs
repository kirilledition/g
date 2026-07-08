use crate::schedule::ScheduleError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCallbackQueueLimits {
    pub dosage_queue_depth: usize,
    pub result_queue_depth: usize,
    pub result_in_flight_limit: usize,
    pub dosage_buffer_limit: usize,
}

/// Resolve native callback queue depths and bounded resource limits.
///
/// # Errors
///
/// Returns an error when a configured limit is non-positive, cannot fit in
/// `usize`, the default `staging_depth + 1` limit would overflow, or the
/// callback batch size cannot fit in the effective dosage buffer limit.
pub fn resolve_native_callback_queue_limits(
    staging_depth: i64,
    native_callback_batch_size: i64,
    result_in_flight_limit: Option<i64>,
    dosage_buffer_limit: Option<i64>,
) -> Result<NativeCallbackQueueLimits, ScheduleError> {
    if staging_depth <= 0 {
        return Err(ScheduleError::NonPositiveStagingDepth);
    }
    if native_callback_batch_size <= 0 {
        return Err(ScheduleError::NonPositiveCallbackBatchSize);
    }
    if matches!(result_in_flight_limit, Some(limit) if limit <= 0) {
        return Err(ScheduleError::NonPositiveResultInFlightLimit);
    }
    if matches!(dosage_buffer_limit, Some(limit) if limit <= 0) {
        return Err(ScheduleError::NonPositiveDosageBufferLimit);
    }

    let staging_depth =
        usize::try_from(staging_depth).map_err(|_| ScheduleError::StagingDepthOverflow { staging_depth })?;
    let native_callback_batch_size = usize::try_from(native_callback_batch_size)
        .map_err(|_| ScheduleError::CallbackBatchSizeOverflow { callback_batch_size: native_callback_batch_size })?;
    let default_limit =
        staging_depth.checked_add(1).ok_or(ScheduleError::QueueLimitDefaultOverflow { staging_depth })?;
    let result_in_flight_limit = result_in_flight_limit
        .map(|limit| {
            usize::try_from(limit)
                .map_err(|_| ScheduleError::ResultInFlightLimitOverflow { result_in_flight_limit: limit })
        })
        .transpose()?
        .unwrap_or(default_limit);
    let dosage_buffer_limit = dosage_buffer_limit
        .map(|limit| {
            usize::try_from(limit).map_err(|_| ScheduleError::DosageBufferLimitOverflow { dosage_buffer_limit: limit })
        })
        .transpose()?
        .unwrap_or(default_limit);
    if dosage_buffer_limit < native_callback_batch_size {
        return Err(ScheduleError::CallbackBatchSizeExceedsDosageBufferLimit { dosage_buffer_limit });
    }

    Ok(NativeCallbackQueueLimits {
        dosage_queue_depth: staging_depth,
        result_queue_depth: staging_depth,
        result_in_flight_limit,
        dosage_buffer_limit,
    })
}
