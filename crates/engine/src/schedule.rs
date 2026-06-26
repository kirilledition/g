//! Pure scheduling and resume policy helpers for engine-owned delivery.

use std::collections::BTreeSet;

const DEFAULT_DELIVERY_CALLBACK_BATCH_SIZE: i64 = 1;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ScheduleError {
    #[error("native_callback_batch_size must be positive.")]
    NonPositiveCallbackBatchSize,
    #[error("native_callback_batch_size > 1 is not supported for packed8 BGEN delivery.")]
    Packed8CallbackBatchSize,
    #[error("native_callback_batch_size exceeds platform capacity: {callback_batch_size}")]
    CallbackBatchSizeOverflow { callback_batch_size: i64 },
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
}
