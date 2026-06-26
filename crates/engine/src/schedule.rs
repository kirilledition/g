//! Pure scheduling and resume policy helpers for engine-owned delivery.

use std::collections::BTreeSet;

const DEFAULT_DELIVERY_CALLBACK_BATCH_SIZE: i64 = 1;

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
    #[error("native_callback_batch_size must be positive.")]
    NonPositiveCallbackBatchSize,
    #[error("native_callback_batch_size > 1 is not supported for packed8 BGEN delivery.")]
    Packed8CallbackBatchSize,
    #[error("native_callback_batch_size exceeds platform capacity: {callback_batch_size}")]
    CallbackBatchSizeOverflow { callback_batch_size: i64 },
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
