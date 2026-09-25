//! Checked admission and pending-range planning for two-group source reuse.

use std::collections::{BTreeMap, BTreeSet};

const SHARED_SOURCE_BYTE_LIMIT: usize = 128 * 1024 * 1024;
const RETAINED_STATE_BYTE_LIMIT: usize = 64 * 1024 * 1024;

/// Dimensions of one group's retained linear association state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TiledGroupGeometry {
    pub(crate) samples: usize,
    pub(crate) covariates: usize,
    pub(crate) traits: usize,
}

/// Admit exactly two positive groups within the retained array payload budgets.
///
/// The source charge includes full-source probability pairs and variant statuses.
/// State charges include both groups, their chromosome arrays, and indexed sample
/// selections. These limits exclude preparation temporaries and allocator reserves.
#[must_use]
pub(crate) fn within_retained_memory_limits(
    compute_variant_count: usize,
    file_sample_count: usize,
    groups: &[TiledGroupGeometry],
) -> bool {
    if compute_variant_count == 0 || file_sample_count == 0 || groups.len() != 2 {
        return false;
    }
    let Some(source_bytes) = file_sample_count
        .checked_mul(2)
        .and_then(|sample_bytes| sample_bytes.checked_add(4))
        .and_then(|variant_bytes| variant_bytes.checked_mul(compute_variant_count))
    else {
        return false;
    };
    if source_bytes > SHARED_SOURCE_BYTE_LIMIT {
        return false;
    }
    let mut retained_state_bytes = 0_usize;
    for group in groups {
        if group.samples == 0 || group.samples > file_sample_count || group.covariates == 0 || group.traits == 0 {
            return false;
        }
        let Some(group_bytes) = retained_group_bytes(group) else {
            return false;
        };
        let Some(combined_bytes) = retained_state_bytes.checked_add(group_bytes) else {
            return false;
        };
        if combined_bytes > RETAINED_STATE_BYTE_LIMIT {
            return false;
        }
        retained_state_bytes = combined_bytes;
    }
    true
}

fn retained_group_bytes(group: &TiledGroupGeometry) -> Option<usize> {
    let sample_array_bytes = group.covariates.checked_add(group.traits)?.checked_mul(group.samples)?.checked_mul(12)?;
    let projection_bytes = group.traits.checked_mul(group.covariates)?.checked_mul(4)?;
    let trait_summary_bytes = group.traits.checked_mul(4)?;
    let selection_bytes = group.samples.checked_mul(4)?;
    sample_array_bytes
        .checked_add(projection_bytes)?
        .checked_add(trait_summary_bytes)?
        .checked_add(8)?
        .checked_add(selection_bytes)
}

/// Return the ascending union of pending chunks, deduplicating exact ranges.
///
/// Input order need not match, and either input may be empty.
///
/// # Errors
///
/// Rejects empty or reversed ranges, conflicting stops for the same start, and
/// overlaps between distinct ranges. Adjacent ranges remain separate chunks.
pub(crate) fn merge_pending_chunks(
    left: &[g_genotype::ChunkSpec],
    right: &[g_genotype::ChunkSpec],
) -> Result<Vec<g_genotype::ChunkSpec>, String> {
    let mut ranges = BTreeMap::new();
    for chunk in left.iter().chain(right) {
        if chunk.variant_start_index >= chunk.variant_stop_index {
            return Err(format!(
                "Invalid pending chunk range {}..{}: start must precede stop.",
                chunk.variant_start_index, chunk.variant_stop_index
            ));
        }
        if let Some(existing_stop) = ranges.insert(chunk.variant_start_index, chunk.variant_stop_index)
            && existing_stop != chunk.variant_stop_index
        {
            return Err(format!(
                "Conflicting pending chunk ranges at {}: stops {existing_stop} and {}.",
                chunk.variant_start_index, chunk.variant_stop_index
            ));
        }
    }
    let mut merged = Vec::with_capacity(ranges.len());
    let mut previous_stop = None;
    for (variant_start_index, variant_stop_index) in ranges {
        if let Some(stop) = previous_stop
            && variant_start_index < stop
        {
            return Err(format!(
                "Overlapping pending chunk range {variant_start_index}..{variant_stop_index} before previous stop {stop}."
            ));
        }
        merged.push(g_genotype::ChunkSpec { variant_start_index, variant_stop_index });
        previous_stop = Some(variant_stop_index);
    }
    Ok(merged)
}

/// Return whether both plans contain an identical nonempty pending range.
///
/// This tests common work only; callers validate the complete union separately.
#[must_use]
pub(crate) fn has_shared_pending_chunk(left: &[g_genotype::ChunkSpec], right: &[g_genotype::ChunkSpec]) -> bool {
    let left_ranges = left
        .iter()
        .filter(|chunk| chunk.variant_start_index < chunk.variant_stop_index)
        .map(|chunk| (chunk.variant_start_index, chunk.variant_stop_index))
        .collect::<BTreeSet<_>>();
    right.iter().any(|chunk| left_ranges.contains(&(chunk.variant_start_index, chunk.variant_stop_index)))
}

#[cfg(test)]
mod tests {
    use super::{TiledGroupGeometry, has_shared_pending_chunk, merge_pending_chunks, within_retained_memory_limits};

    fn geometry(samples: usize) -> TiledGroupGeometry {
        TiledGroupGeometry { samples, covariates: 1, traits: 1 }
    }

    fn chunk(variant_start_index: usize, variant_stop_index: usize) -> g_genotype::ChunkSpec {
        g_genotype::ChunkSpec { variant_start_index, variant_stop_index }
    }

    #[test]
    fn source_budget_accepts_exact_ceiling_and_rejects_next_variant() {
        let groups = [geometry(1); 2];
        assert!(within_retained_memory_limits(16_777_216, 2, &groups));
        assert!(!within_retained_memory_limits(16_777_217, 2, &groups));
    }

    #[test]
    fn state_budget_accepts_exact_ceiling_and_rejects_next_sample() {
        // Each one-trait, one-covariate group retains 28 * samples + 16 bytes.
        let groups = [geometry(1_198_372); 2];
        assert!(within_retained_memory_limits(1, 1_198_372, &groups));
        let larger_groups = [geometry(1_198_372), geometry(1_198_373)];
        assert!(!within_retained_memory_limits(1, 1_198_373, &larger_groups));
    }

    #[test]
    fn admission_requires_exactly_two_positive_consistent_groups() {
        let valid = geometry(1);
        assert!(!within_retained_memory_limits(1, 1, &[]));
        assert!(!within_retained_memory_limits(1, 1, &[valid]));
        assert!(!within_retained_memory_limits(1, 1, &[valid; 3]));
        assert!(!within_retained_memory_limits(0, 1, &[valid; 2]));
        assert!(!within_retained_memory_limits(1, 0, &[valid; 2]));
        for invalid in [
            geometry(0),
            geometry(2),
            TiledGroupGeometry { covariates: 0, ..valid },
            TiledGroupGeometry { traits: 0, ..valid },
        ] {
            assert!(!within_retained_memory_limits(1, 1, &[valid, invalid]));
            assert!(!within_retained_memory_limits(1, 1, &[invalid, valid]));
        }
    }

    #[test]
    fn admission_rejects_overflow_without_wrapping() {
        let valid = geometry(1);
        assert!(!within_retained_memory_limits(usize::MAX, 1, &[valid; 2]));
        assert!(!within_retained_memory_limits(1, usize::MAX, &[valid; 2]));
        for invalid in [
            TiledGroupGeometry { covariates: usize::MAX, ..valid },
            TiledGroupGeometry { traits: usize::MAX, ..valid },
            TiledGroupGeometry { covariates: usize::MAX / 8, traits: usize::MAX / 8, ..valid },
        ] {
            assert!(!within_retained_memory_limits(1, 1, &[valid, invalid]));
        }
    }

    #[test]
    fn merges_asymmetric_plans_in_order_and_deduplicates_exact_ranges() {
        let left = [chunk(8, 12), chunk(0, 4), chunk(8, 12)];
        let right = [chunk(4, 8), chunk(8, 12), chunk(20, 24)];
        let expected = vec![chunk(0, 4), chunk(4, 8), chunk(8, 12), chunk(20, 24)];
        assert_eq!(merge_pending_chunks(&left, &right).unwrap(), expected);
        assert_eq!(merge_pending_chunks(&right, &left).unwrap(), expected);
        assert!(has_shared_pending_chunk(&left, &right));
    }

    #[test]
    fn empty_and_disjoint_pending_plans_are_valid_without_shared_work() {
        let left = [chunk(0, 4)];
        let right = [chunk(8, 12)];
        assert!(merge_pending_chunks(&[], &[]).unwrap().is_empty());
        assert_eq!(merge_pending_chunks(&left, &[]).unwrap(), left);
        assert_eq!(merge_pending_chunks(&[], &right).unwrap(), right);
        assert_eq!(merge_pending_chunks(&left, &right).unwrap(), vec![chunk(0, 4), chunk(8, 12)]);
        assert!(!has_shared_pending_chunk(&[], &left));
        assert!(!has_shared_pending_chunk(&left, &[]));
        assert!(!has_shared_pending_chunk(&left, &right));
    }

    #[test]
    fn rejects_invalid_conflicting_and_overlapping_ranges() {
        assert!(merge_pending_chunks(&[chunk(4, 4)], &[]).is_err());
        assert!(merge_pending_chunks(&[], &[chunk(5, 4)]).is_err());
        assert!(merge_pending_chunks(&[chunk(0, 4)], &[chunk(0, 5)]).is_err());
        assert!(merge_pending_chunks(&[chunk(0, 4)], &[chunk(2, 6)]).is_err());
        assert!(merge_pending_chunks(&[chunk(0, 8)], &[chunk(2, 4)]).is_err());
        assert!(merge_pending_chunks(&[chunk(0, 4), chunk(2, 6)], &[]).is_err());
        assert!(!has_shared_pending_chunk(&[chunk(4, 4)], &[chunk(4, 4)]));
        assert!(!has_shared_pending_chunk(&[chunk(0, 4)], &[chunk(0, 5)]));
        assert!(!has_shared_pending_chunk(&[chunk(0, 4)], &[chunk(2, 6)]));
    }

    #[test]
    fn merges_maximum_index_without_arithmetic_overflow() {
        let last = [chunk(usize::MAX - 1, usize::MAX)];
        assert_eq!(merge_pending_chunks(&last, &last).unwrap(), last);
        assert!(has_shared_pending_chunk(&last, &last));
    }
}
