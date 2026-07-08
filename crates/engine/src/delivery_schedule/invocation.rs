use crate::schedule::ScheduleError;

use super::types::{BgenDeliveryInvocationPlan, BgenDeliveryMethod};

const DEFAULT_DELIVERY_CALLBACK_BATCH_SIZE: i64 = 1;

/// Resolve the callback batch size for one native BGEN delivery mode.
///
/// # Errors
///
/// Returns an error when the requested batch size is non-positive, cannot fit
/// in `usize`, or requests packed8 callback batching, which is not supported.
pub fn resolve_delivery_callback_batch_size(
    callback_batch_size: Option<i64>,
    variant_major_packed8_probability_pairs: bool,
) -> Result<usize, ScheduleError> {
    let requested_callback_batch_size = callback_batch_size.unwrap_or(DEFAULT_DELIVERY_CALLBACK_BATCH_SIZE);
    if requested_callback_batch_size <= 0 {
        return Err(ScheduleError::NonPositiveCallbackBatchSize);
    }
    let resolved_callback_batch_size = usize::try_from(requested_callback_batch_size)
        .map_err(|_| ScheduleError::CallbackBatchSizeOverflow { callback_batch_size: requested_callback_batch_size })?;
    if variant_major_packed8_probability_pairs && resolved_callback_batch_size > 1 {
        return Err(ScheduleError::Packed8CallbackBatchSize);
    }
    Ok(resolved_callback_batch_size)
}

/// Plan one native BGEN delivery invocation.
///
/// # Errors
///
/// Returns an error when the requested callback batch size is invalid for the
/// selected delivery mode.
pub fn plan_bgen_delivery_invocation(
    callback_batch_size: Option<i64>,
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
) -> Result<BgenDeliveryInvocationPlan, ScheduleError> {
    let callback_batch_size =
        resolve_delivery_callback_batch_size(callback_batch_size, variant_major_packed8_probability_pairs)?;
    let delivery_method = resolve_bgen_delivery_method(
        variant_major_packed8_probability_pairs,
        has_native_multi_aligned_sample_data,
        has_native_aligned_sample_data,
    );
    Ok(BgenDeliveryInvocationPlan { delivery_method, callback_batch_size })
}

/// Resolve the callback batch size for grouped union BGEN delivery.
///
/// # Errors
///
/// Returns an error when the requested batch size is non-positive, cannot fit
/// in `usize`, or requests grouped union callback batching, which is not
/// supported.
pub fn resolve_grouped_union_callback_batch_size(callback_batch_size: i64) -> Result<usize, ScheduleError> {
    let resolved_callback_batch_size = resolve_delivery_callback_batch_size(Some(callback_batch_size), false)?;
    if resolved_callback_batch_size > 1 {
        return Err(ScheduleError::GroupedUnionCallbackBatchSize);
    }
    Ok(resolved_callback_batch_size)
}

#[must_use]
pub const fn resolve_bgen_delivery_method(
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
) -> BgenDeliveryMethod {
    match (
        variant_major_packed8_probability_pairs,
        has_native_multi_aligned_sample_data,
        has_native_aligned_sample_data,
    ) {
        (true, true, _) => BgenDeliveryMethod::Packed8NativeMultiAlignedSamples,
        (true, false, true) => BgenDeliveryMethod::Packed8NativeAlignedSamples,
        (true, false, false) => BgenDeliveryMethod::Packed8SampleIndices,
        (false, true, _) => BgenDeliveryMethod::DosageNativeMultiAlignedSamples,
        (false, false, true) => BgenDeliveryMethod::DosageNativeAlignedSamples,
        (false, false, false) => BgenDeliveryMethod::DosageSampleIndices,
    }
}
