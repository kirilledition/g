use std::collections::HashMap;

use crate::error::InputResult;

#[must_use]
pub fn build_union_sample_indices<Groups, SampleIndices>(sample_indices_by_group: Groups) -> Vec<usize>
where
    Groups: IntoIterator<Item = SampleIndices>,
    SampleIndices: AsRef<[usize]>,
{
    let mut union_sample_indices = Vec::new();
    for sample_indices in sample_indices_by_group {
        union_sample_indices.extend_from_slice(sample_indices.as_ref());
    }
    union_sample_indices.sort_unstable();
    union_sample_indices.dedup();
    union_sample_indices
}

pub fn build_group_sample_position_arrays<Groups, SampleIndices>(
    union_sample_indices: &[usize],
    sample_indices_by_group: Groups,
) -> InputResult<Vec<Vec<usize>>>
where
    Groups: IntoIterator<Item = SampleIndices>,
    SampleIndices: AsRef<[usize]>,
{
    let mut union_position_by_sample_index: HashMap<usize, usize> = HashMap::with_capacity(union_sample_indices.len());
    for (sample_position, sample_index) in union_sample_indices.iter().enumerate() {
        union_position_by_sample_index.insert(*sample_index, sample_position);
    }
    sample_indices_by_group
        .into_iter()
        .map(|group_sample_indices| {
            group_sample_indices
                .as_ref()
                .iter()
                .map(|sample_index| {
                    union_position_by_sample_index.get(sample_index).copied().ok_or_else(|| {
                        format!("Group sample index {sample_index} is absent from the union sample index set.")
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<Vec<_>>, _>>()
        .map_err(Into::into)
}
