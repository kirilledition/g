use super::BgenError;

#[derive(Debug)]
pub(super) struct SampleSelection {
    pub(super) selected_sample_count: usize,
    pub(super) file_to_selected_index: Vec<usize>,
    pub(super) selected_file_indices: Vec<usize>,
    pub(super) is_identity: bool,
    pub(super) contiguous_file_index_start: Option<usize>,
}

pub(super) fn build_sample_selection(
    sample_count: usize,
    sample_indices: &[usize],
) -> Result<SampleSelection, BgenError> {
    let mut file_to_selected_index = vec![usize::MAX; sample_count];
    let mut selected_file_indices = Vec::with_capacity(sample_indices.len());
    let mut is_identity = sample_indices.len() == sample_count;
    for (selected_index, sample_index) in sample_indices.iter().copied().enumerate() {
        if sample_index >= sample_count {
            return Err(BgenError::Range(format!(
                "Sample index {sample_index} is out of bounds for a BGEN file with {sample_count} samples.",
            )));
        }
        if file_to_selected_index[sample_index] != usize::MAX {
            return Err(BgenError::Range(format!(
                "Sample index {sample_index} was requested more than once in the same read.",
            )));
        }
        file_to_selected_index[sample_index] = selected_index;
        selected_file_indices.push(sample_index);
        if sample_index != selected_index {
            is_identity = false;
        }
    }
    let contiguous_file_index_start = selected_file_indices.first().copied().filter(|_| {
        selected_file_indices
            .array_windows::<2>()
            .all(|[previous_file_index, next_file_index]| *next_file_index == *previous_file_index + 1)
    });
    Ok(SampleSelection {
        selected_sample_count: sample_indices.len(),
        file_to_selected_index,
        selected_file_indices,
        is_identity,
        contiguous_file_index_start,
    })
}
