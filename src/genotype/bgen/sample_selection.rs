use super::BgenError;

#[derive(Debug)]
pub(super) struct SampleSelection {
    pub(super) selected_sample_count: usize,
    pub(super) file_to_selected_index: Vec<usize>,
    pub(super) is_identity: bool,
}

pub(super) fn build_sample_selection(
    sample_count: usize,
    sample_indices: &[i64],
) -> Result<SampleSelection, BgenError> {
    let mut file_to_selected_index = vec![usize::MAX; sample_count];
    let mut is_identity = sample_indices.len() == sample_count;
    for (selected_index, raw_sample_index) in sample_indices.iter().enumerate() {
        let sample_index = usize::try_from(*raw_sample_index).map_err(|_| {
            BgenError::Range(format!("Sample indices must be non-negative. Observed sample index {raw_sample_index}."))
        })?;
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
        if sample_index != selected_index {
            is_identity = false;
        }
    }
    Ok(SampleSelection { selected_sample_count: sample_indices.len(), file_to_selected_index, is_identity })
}
