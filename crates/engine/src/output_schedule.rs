//! Output writer scheduling and committed-chunk policy.

use std::collections::BTreeSet;
use std::sync::Arc;

#[must_use]
pub fn intersect_committed_chunk_identifier_sets<T>(committed_chunk_identifier_sets: &[Arc<BTreeSet<T>>]) -> BTreeSet<T>
where
    T: Copy + Ord,
{
    let Some((first_set, remaining_sets)) = committed_chunk_identifier_sets.split_first() else {
        return BTreeSet::new();
    };
    let mut shared_chunk_identifiers = first_set.as_ref().clone();
    for committed_chunk_identifier_set in remaining_sets {
        shared_chunk_identifiers.retain(|chunk_identifier| committed_chunk_identifier_set.contains(chunk_identifier));
    }
    shared_chunk_identifiers
}

/// Plan which multi-trait writer lanes still need one chunk.
///
/// # Errors
///
/// Returns an error when the committed chunk identifier set count does not
/// match the writer session count.
pub fn active_trait_indices_for_chunk(
    writer_session_count: usize,
    chunk_identifier: usize,
    committed_chunk_identifier_sets: &[Arc<BTreeSet<usize>>],
) -> Result<Vec<usize>, String> {
    if committed_chunk_identifier_sets.len() != writer_session_count {
        return Err(format!(
            "Committed chunk identifier set count ({}) must match writer session count ({writer_session_count}).",
            committed_chunk_identifier_sets.len()
        ));
    }
    let active_trait_indices =
        committed_chunk_identifier_sets
            .iter()
            .enumerate()
            .filter_map(|(trait_index, committed_chunk_identifier_set)| {
                if committed_chunk_identifier_set.contains(&chunk_identifier) { None } else { Some(trait_index) }
            })
            .collect();
    Ok(active_trait_indices)
}
