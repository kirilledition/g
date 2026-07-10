use std::collections::{HashMap, HashSet};

use crate::error::InputResult;

#[must_use]
pub fn build_union_sample_indices<Groups, SampleIndices>(sample_indices_by_group: Groups) -> Vec<usize>
where
    Groups: IntoIterator<Item = SampleIndices>,
    SampleIndices: AsRef<[usize]>,
{
    let mut seen_sample_indices: HashSet<usize> = HashSet::new();
    let mut union_sample_indices: Vec<usize> = Vec::new();
    for sample_indices in sample_indices_by_group {
        for sample_index in sample_indices.as_ref() {
            if seen_sample_indices.insert(*sample_index) {
                union_sample_indices.push(*sample_index);
            }
        }
    }
    union_sample_indices
}

pub fn build_group_sample_position_array(
    union_sample_indices: &[usize],
    group_sample_indices: &[usize],
) -> InputResult<Vec<isize>> {
    let mut union_position_by_sample_index: HashMap<usize, isize> = HashMap::with_capacity(union_sample_indices.len());
    for (sample_position, sample_index) in union_sample_indices.iter().enumerate() {
        let sample_position = isize::try_from(sample_position)
            .map_err(|_| "Union sample position exceeds platform index capacity.".to_string())?;
        union_position_by_sample_index.insert(*sample_index, sample_position);
    }
    group_sample_indices
        .iter()
        .map(|sample_index| {
            union_position_by_sample_index
                .get(sample_index)
                .copied()
                .ok_or_else(|| format!("Group sample index {sample_index} is absent from the union sample index set."))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}
