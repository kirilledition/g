//! Output writer scheduling and committed-chunk policy.

use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) enum ActiveTraitSelection {
    All,
    Indices(Vec<usize>),
}

#[must_use]
pub(crate) fn intersect_committed_chunk_identifier_sets<T>(
    committed_chunk_identifier_sets: &[Arc<BTreeSet<T>>],
) -> BTreeSet<T>
where
    T: Copy + Ord,
{
    let Some((smallest_set_index, smallest_set)) =
        committed_chunk_identifier_sets.iter().enumerate().min_by_key(|(_, set)| set.len())
    else {
        return BTreeSet::new();
    };
    let mut shared_chunk_identifiers = smallest_set.as_ref().clone();
    for (set_index, committed_chunk_identifier_set) in committed_chunk_identifier_sets.iter().enumerate() {
        if set_index == smallest_set_index {
            continue;
        }
        shared_chunk_identifiers.retain(|chunk_identifier| committed_chunk_identifier_set.contains(chunk_identifier));
        if shared_chunk_identifiers.is_empty() {
            break;
        }
    }
    shared_chunk_identifiers
}

/// Plan which multi-trait writer lanes still need one chunk.
///
/// # Errors
///
/// Returns an error when the committed chunk identifier set count does not
/// match the writer session count.
pub(crate) fn active_trait_selection_for_chunk(
    writer_session_count: usize,
    chunk_identifier: usize,
    committed_chunk_identifier_sets: &[Arc<BTreeSet<usize>>],
) -> Result<ActiveTraitSelection, String> {
    if committed_chunk_identifier_sets.len() != writer_session_count {
        return Err(format!(
            "Committed chunk identifier set count ({}) must match writer session count ({writer_session_count}).",
            committed_chunk_identifier_sets.len()
        ));
    }
    let Some(first_committed_trait_index) = committed_chunk_identifier_sets
        .iter()
        .position(|committed_chunk_identifier_set| committed_chunk_identifier_set.contains(&chunk_identifier))
    else {
        return Ok(ActiveTraitSelection::All);
    };
    let mut active_trait_indices = Vec::with_capacity(writer_session_count.saturating_sub(1));
    active_trait_indices.extend(0..first_committed_trait_index);
    active_trait_indices.extend(
        committed_chunk_identifier_sets.iter().enumerate().skip(first_committed_trait_index + 1).filter_map(
            |(trait_index, committed_chunk_identifier_set)| {
                (!committed_chunk_identifier_set.contains(&chunk_identifier)).then_some(trait_index)
            },
        ),
    );
    Ok(ActiveTraitSelection::Indices(active_trait_indices))
}
