use super::BgenError;

#[derive(Debug)]
pub(super) enum SampleSelection {
    Identity { sample_count: usize },
    Contiguous { file_index_start: usize, sample_count: usize },
    Indexed { file_to_selected_index: Box<[usize]>, selected_file_indices: Box<[usize]> },
}

impl SampleSelection {
    pub(super) fn selected_sample_count(&self) -> usize {
        match self {
            Self::Identity { sample_count } | Self::Contiguous { sample_count, .. } => *sample_count,
            Self::Indexed { selected_file_indices, .. } => selected_file_indices.len(),
        }
    }

    pub(super) fn is_identity(&self) -> bool {
        matches!(self, Self::Identity { .. })
    }

    pub(super) fn contiguous_file_index_start(&self) -> Option<usize> {
        match self {
            Self::Contiguous { file_index_start, sample_count } if *sample_count > 0 => Some(*file_index_start),
            _ => None,
        }
    }

    pub(super) fn indexed_file_indices(&self) -> Option<&[usize]> {
        match self {
            Self::Indexed { selected_file_indices, .. } => Some(selected_file_indices),
            _ => None,
        }
    }

    pub(super) fn selected_index(&self, file_sample_index: usize) -> Option<usize> {
        match self {
            Self::Identity { sample_count } => (file_sample_index < *sample_count).then_some(file_sample_index),
            Self::Contiguous { file_index_start, sample_count } => file_sample_index
                .checked_sub(*file_index_start)
                .filter(|selected_index| *selected_index < *sample_count),
            Self::Indexed { file_to_selected_index, .. } => file_to_selected_index
                .get(file_sample_index)
                .copied()
                .filter(|selected_index| *selected_index != usize::MAX),
        }
    }
}

pub(super) fn build_sample_selection(
    sample_count: usize,
    sample_indices: &[usize],
) -> Result<SampleSelection, BgenError> {
    let mut is_identity = sample_indices.len() == sample_count;
    let mut is_contiguous = true;
    let mut previous_sample_index = None;
    for (selected_index, sample_index) in sample_indices.iter().copied().enumerate() {
        if sample_index >= sample_count {
            return Err(BgenError::Range(format!(
                "Sample index {sample_index} is out of bounds for a BGEN file with {sample_count} samples.",
            )));
        }
        if sample_index != selected_index {
            is_identity = false;
        }
        if previous_sample_index.is_some_and(|previous_index| sample_index != previous_index + 1) {
            is_contiguous = false;
        }
        previous_sample_index = Some(sample_index);
    }
    if is_identity {
        return Ok(SampleSelection::Identity { sample_count });
    }
    if is_contiguous {
        return Ok(SampleSelection::Contiguous {
            file_index_start: sample_indices.first().copied().unwrap_or(0),
            sample_count: sample_indices.len(),
        });
    }

    let mut file_to_selected_index = vec![usize::MAX; sample_count];
    for (selected_index, sample_index) in sample_indices.iter().copied().enumerate() {
        if file_to_selected_index[sample_index] != usize::MAX {
            return Err(BgenError::Range(format!(
                "Sample index {sample_index} was requested more than once in the same read.",
            )));
        }
        file_to_selected_index[sample_index] = selected_index;
    }
    Ok(SampleSelection::Indexed {
        file_to_selected_index: file_to_selected_index.into_boxed_slice(),
        selected_file_indices: sample_indices.into(),
    })
}
