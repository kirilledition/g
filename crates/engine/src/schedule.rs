//! Pure scheduling and resume policy helpers for engine-owned delivery.

use std::collections::BTreeSet;

const DEFAULT_DELIVERY_CALLBACK_BATCH_SIZE: i64 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCallbackQueueLimits {
    pub dosage_queue_depth: usize,
    pub result_queue_depth: usize,
    pub result_in_flight_limit: usize,
    pub dosage_buffer_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageBufferReusePlan {
    pub requires_slice: bool,
    pub slice_dimensions: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenDeliveryMethod {
    DosageNativeMultiAlignedSamples,
    DosageNativeAlignedSamples,
    DosageSampleIndices,
    Packed8NativeMultiAlignedSamples,
    Packed8NativeAlignedSamples,
    Packed8SampleIndices,
}

impl BgenDeliveryMethod {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::DosageNativeMultiAlignedSamples => "dosage_native_multi_aligned_samples",
            Self::DosageNativeAlignedSamples => "dosage_native_aligned_samples",
            Self::DosageSampleIndices => "dosage_sample_indices",
            Self::Packed8NativeMultiAlignedSamples => "packed8_native_multi_aligned_samples",
            Self::Packed8NativeAlignedSamples => "packed8_native_aligned_samples",
            Self::Packed8SampleIndices => "packed8_sample_indices",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ScheduleError {
    #[error("staging_depth must be positive.")]
    NonPositiveStagingDepth,
    #[error("native_callback_batch_size must be positive.")]
    NonPositiveCallbackBatchSize,
    #[error("native_callback_batch_size > 1 is not supported for packed8 BGEN delivery.")]
    Packed8CallbackBatchSize,
    #[error("native_callback_batch_size > 1 is not supported for grouped union BGEN delivery.")]
    GroupedUnionCallbackBatchSize,
    #[error("native_callback_batch_size must not exceed the effective dosage_buffer_limit ({dosage_buffer_limit}).")]
    CallbackBatchSizeExceedsDosageBufferLimit { dosage_buffer_limit: usize },
    #[error("result_in_flight_limit must be positive when provided.")]
    NonPositiveResultInFlightLimit,
    #[error("dosage_buffer_limit must be positive when provided.")]
    NonPositiveDosageBufferLimit,
    #[error("staging_depth exceeds platform capacity: {staging_depth}")]
    StagingDepthOverflow { staging_depth: i64 },
    #[error("native_callback_batch_size exceeds platform capacity: {callback_batch_size}")]
    CallbackBatchSizeOverflow { callback_batch_size: i64 },
    #[error("result_in_flight_limit exceeds platform capacity: {result_in_flight_limit}")]
    ResultInFlightLimitOverflow { result_in_flight_limit: i64 },
    #[error("dosage_buffer_limit exceeds platform capacity: {dosage_buffer_limit}")]
    DosageBufferLimitOverflow { dosage_buffer_limit: i64 },
    #[error("queue limit default for staging_depth exceeds platform capacity: {staging_depth}")]
    QueueLimitDefaultOverflow { staging_depth: usize },
    #[error("Writer finish thread count must be positive.")]
    NonPositiveWriterFinishThreadCount,
    #[error("Writer session count exceeds platform capacity: {session_count}")]
    WriterSessionCountOverflow { session_count: i64 },
    #[error("Writer finish thread count exceeds platform capacity: {thread_count}")]
    WriterFinishThreadCountOverflow { thread_count: i64 },
}

#[must_use]
pub fn intersect_committed_chunk_identifier_sets(
    committed_chunk_identifier_sets: &[BTreeSet<usize>],
) -> BTreeSet<usize> {
    let Some((first_set, remaining_sets)) = committed_chunk_identifier_sets.split_first() else {
        return BTreeSet::new();
    };
    let mut shared_chunk_identifiers = first_set.clone();
    for committed_chunk_identifier_set in remaining_sets {
        shared_chunk_identifiers.retain(|chunk_identifier| committed_chunk_identifier_set.contains(chunk_identifier));
    }
    shared_chunk_identifiers
}

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
pub fn plan_dosage_buffer_reuse(buffered_shape: &[usize], expected_shape: &[usize]) -> Option<DosageBufferReusePlan> {
    if buffered_shape.len() != expected_shape.len() {
        return None;
    }
    if buffered_shape
        .iter()
        .zip(expected_shape)
        .any(|(buffered_dimension, expected_dimension)| buffered_dimension < expected_dimension)
    {
        return None;
    }
    Some(DosageBufferReusePlan {
        requires_slice: buffered_shape != expected_shape,
        slice_dimensions: expected_shape.to_vec(),
    })
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

/// Resolve the thread count used to finish output writer sessions.
///
/// # Errors
///
/// Returns an error when at least one writer must finish and the requested
/// thread count is non-positive or cannot fit in `usize`.
pub fn resolve_writer_finish_thread_count(
    writer_session_count: i64,
    requested_thread_count: i64,
) -> Result<usize, ScheduleError> {
    if writer_session_count <= 0 {
        return Ok(0);
    }
    if requested_thread_count <= 0 {
        return Err(ScheduleError::NonPositiveWriterFinishThreadCount);
    }
    let writer_session_count = usize::try_from(writer_session_count)
        .map_err(|_| ScheduleError::WriterSessionCountOverflow { session_count: writer_session_count })?;
    let requested_thread_count = usize::try_from(requested_thread_count)
        .map_err(|_| ScheduleError::WriterFinishThreadCountOverflow { thread_count: requested_thread_count })?;
    Ok(writer_session_count.min(requested_thread_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_empty_set_for_empty_inputs() {
        let shared_chunk_identifiers = intersect_committed_chunk_identifier_sets(&[]);

        assert!(shared_chunk_identifiers.is_empty());
    }

    #[test]
    fn intersects_committed_chunk_identifiers_across_outputs() {
        let committed_chunk_identifier_sets =
            [BTreeSet::from([0_usize, 32, 64]), BTreeSet::from([32, 64, 96]), BTreeSet::from([32, 128])];

        let shared_chunk_identifiers = intersect_committed_chunk_identifier_sets(&committed_chunk_identifier_sets);

        assert_eq!(shared_chunk_identifiers, BTreeSet::from([32]));
    }

    #[test]
    fn preserves_single_output_committed_chunk_identifiers() {
        let committed_chunk_identifier_sets = [BTreeSet::from([64_usize, 0])];

        let shared_chunk_identifiers = intersect_committed_chunk_identifier_sets(&committed_chunk_identifier_sets);

        assert_eq!(shared_chunk_identifiers, BTreeSet::from([0, 64]));
    }

    #[test]
    fn resolves_delivery_callback_batch_size_default_and_explicit_values() {
        assert_eq!(resolve_delivery_callback_batch_size(None, false).unwrap(), 1);
        assert_eq!(resolve_delivery_callback_batch_size(Some(3), false).unwrap(), 3);
        assert_eq!(resolve_delivery_callback_batch_size(Some(1), true).unwrap(), 1);
    }

    #[test]
    fn rejects_invalid_delivery_callback_batch_sizes() {
        assert_eq!(
            resolve_delivery_callback_batch_size(Some(0), false).unwrap_err(),
            ScheduleError::NonPositiveCallbackBatchSize,
        );
        assert_eq!(
            resolve_delivery_callback_batch_size(Some(2), true).unwrap_err(),
            ScheduleError::Packed8CallbackBatchSize,
        );
    }

    #[test]
    fn resolves_grouped_union_callback_batch_size() {
        assert_eq!(resolve_grouped_union_callback_batch_size(1).unwrap(), 1);
    }

    #[test]
    fn rejects_invalid_grouped_union_callback_batch_sizes() {
        assert_eq!(
            resolve_grouped_union_callback_batch_size(0).unwrap_err(),
            ScheduleError::NonPositiveCallbackBatchSize,
        );
        assert_eq!(
            resolve_grouped_union_callback_batch_size(2).unwrap_err(),
            ScheduleError::GroupedUnionCallbackBatchSize,
        );
    }

    #[test]
    fn plans_dosage_buffer_reuse_for_exact_and_larger_shapes() {
        assert_eq!(
            plan_dosage_buffer_reuse(&[2, 3], &[2, 3]).unwrap(),
            DosageBufferReusePlan { requires_slice: false, slice_dimensions: vec![2, 3] },
        );
        assert_eq!(
            plan_dosage_buffer_reuse(&[4, 5], &[2, 3]).unwrap(),
            DosageBufferReusePlan { requires_slice: true, slice_dimensions: vec![2, 3] },
        );
    }

    #[test]
    fn rejects_incompatible_dosage_buffer_reuse_shapes() {
        assert_eq!(plan_dosage_buffer_reuse(&[2, 3], &[2, 3, 1]), None);
        assert_eq!(plan_dosage_buffer_reuse(&[2, 3], &[3, 2]), None);
    }

    #[test]
    fn resolves_native_callback_queue_limits() {
        assert_eq!(
            resolve_native_callback_queue_limits(3, 1, None, None).unwrap(),
            NativeCallbackQueueLimits {
                dosage_queue_depth: 3,
                result_queue_depth: 3,
                result_in_flight_limit: 4,
                dosage_buffer_limit: 4,
            },
        );
        assert_eq!(
            resolve_native_callback_queue_limits(3, 2, Some(7), Some(8)).unwrap(),
            NativeCallbackQueueLimits {
                dosage_queue_depth: 3,
                result_queue_depth: 3,
                result_in_flight_limit: 7,
                dosage_buffer_limit: 8,
            },
        );
    }

    #[test]
    fn rejects_invalid_native_callback_queue_limits() {
        assert_eq!(
            resolve_native_callback_queue_limits(0, 1, None, None).unwrap_err(),
            ScheduleError::NonPositiveStagingDepth,
        );
        assert_eq!(
            resolve_native_callback_queue_limits(1, 0, None, None).unwrap_err(),
            ScheduleError::NonPositiveCallbackBatchSize,
        );
        assert_eq!(
            resolve_native_callback_queue_limits(1, 1, Some(0), None).unwrap_err(),
            ScheduleError::NonPositiveResultInFlightLimit,
        );
        assert_eq!(
            resolve_native_callback_queue_limits(1, 1, None, Some(0)).unwrap_err(),
            ScheduleError::NonPositiveDosageBufferLimit,
        );
        assert_eq!(
            resolve_native_callback_queue_limits(1, 3, None, Some(2)).unwrap_err(),
            ScheduleError::CallbackBatchSizeExceedsDosageBufferLimit { dosage_buffer_limit: 2 },
        );
    }

    #[test]
    fn resolves_writer_finish_thread_count() {
        assert_eq!(resolve_writer_finish_thread_count(0, 0).unwrap(), 0);
        assert_eq!(resolve_writer_finish_thread_count(-1, 0).unwrap(), 0);
        assert_eq!(resolve_writer_finish_thread_count(3, 1).unwrap(), 1);
        assert_eq!(resolve_writer_finish_thread_count(3, 2).unwrap(), 2);
        assert_eq!(resolve_writer_finish_thread_count(3, 5).unwrap(), 3);
    }

    #[test]
    fn rejects_invalid_writer_finish_thread_count_when_writers_exist() {
        assert_eq!(
            resolve_writer_finish_thread_count(1, 0).unwrap_err(),
            ScheduleError::NonPositiveWriterFinishThreadCount,
        );
    }

    #[test]
    fn resolves_bgen_delivery_method_with_native_alignment_precedence() {
        assert_eq!(
            resolve_bgen_delivery_method(false, true, true),
            BgenDeliveryMethod::DosageNativeMultiAlignedSamples,
        );
        assert_eq!(resolve_bgen_delivery_method(false, false, true), BgenDeliveryMethod::DosageNativeAlignedSamples,);
        assert_eq!(resolve_bgen_delivery_method(false, false, false), BgenDeliveryMethod::DosageSampleIndices);
        assert_eq!(
            resolve_bgen_delivery_method(true, true, true),
            BgenDeliveryMethod::Packed8NativeMultiAlignedSamples,
        );
        assert_eq!(resolve_bgen_delivery_method(true, false, true), BgenDeliveryMethod::Packed8NativeAlignedSamples,);
        assert_eq!(resolve_bgen_delivery_method(true, false, false), BgenDeliveryMethod::Packed8SampleIndices,);
    }
}
