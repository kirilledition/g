use std::collections::BTreeSet;

use crate::schedule;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineOutputInitialization {
    committed_chunk_identifier_sets: Vec<Vec<i64>>,
}

impl PipelineOutputInitialization {
    #[must_use]
    pub fn new(committed_chunk_identifier_sets: Vec<Vec<i64>>) -> Self {
        Self { committed_chunk_identifier_sets }
    }

    #[must_use]
    pub fn committed_chunk_identifier_sets(&self) -> &[Vec<i64>] {
        &self.committed_chunk_identifier_sets
    }

    #[must_use]
    pub fn committed_chunk_identifiers(&self, output_index: usize) -> Option<&[i64]> {
        self.committed_chunk_identifier_sets.get(output_index).map(Vec::as_slice)
    }

    #[must_use]
    pub fn committed_chunk_counts(&self) -> Vec<usize> {
        self.committed_chunk_identifier_sets.iter().map(Vec::len).collect()
    }

    #[must_use]
    pub fn output_count(&self) -> usize {
        self.committed_chunk_identifier_sets.len()
    }

    #[must_use]
    pub fn shared_committed_chunk_identifiers(&self) -> Vec<i64> {
        Self::shared_committed_chunk_identifiers_across(std::iter::once(self))
    }

    #[must_use]
    pub fn shared_committed_chunk_identifiers_across<'a, I>(initializations: I) -> Vec<i64>
    where
        I: IntoIterator<Item = &'a PipelineOutputInitialization>,
    {
        let committed_chunk_identifier_sets = initializations
            .into_iter()
            .flat_map(|initialization| {
                initialization
                    .committed_chunk_identifier_sets
                    .iter()
                    .map(|chunk_identifiers| chunk_identifiers.iter().copied().collect::<BTreeSet<_>>())
            })
            .collect::<Vec<_>>();
        schedule::intersect_committed_chunk_identifier_sets(&committed_chunk_identifier_sets).into_iter().collect()
    }
}
