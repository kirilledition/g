//! Output writer scheduling and committed-chunk policy.

use std::collections::BTreeSet;

use crate::schedule::ScheduleError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiTraitChunkWritePlan {
    pub active_trait_indices: Vec<usize>,
    pub total_trait_count: usize,
}

impl MultiTraitChunkWritePlan {
    #[must_use]
    pub fn active_trait_count(&self) -> usize {
        self.active_trait_indices.len()
    }

    #[must_use]
    pub fn all_traits_committed(&self) -> bool {
        self.active_trait_indices.is_empty()
    }
}

#[must_use]
pub fn intersect_committed_chunk_identifier_sets<T>(committed_chunk_identifier_sets: &[BTreeSet<T>]) -> BTreeSet<T>
where
    T: Copy + Ord,
{
    let Some((first_set, remaining_sets)) = committed_chunk_identifier_sets.split_first() else {
        return BTreeSet::new();
    };
    let mut shared_chunk_identifiers = first_set.clone();
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
pub fn plan_multi_trait_chunk_write(
    writer_session_count: usize,
    chunk_identifier: usize,
    committed_chunk_identifier_sets: &[BTreeSet<usize>],
) -> Result<MultiTraitChunkWritePlan, ScheduleError> {
    if committed_chunk_identifier_sets.len() != writer_session_count {
        return Err(ScheduleError::MultiTraitCommittedChunkSetCountMismatch {
            writer_session_count,
            committed_set_count: committed_chunk_identifier_sets.len(),
        });
    }
    let active_trait_indices =
        committed_chunk_identifier_sets
            .iter()
            .enumerate()
            .filter_map(|(trait_index, committed_chunk_identifier_set)| {
                if committed_chunk_identifier_set.contains(&chunk_identifier) { None } else { Some(trait_index) }
            })
            .collect();
    Ok(MultiTraitChunkWritePlan { active_trait_indices, total_trait_count: writer_session_count })
}
