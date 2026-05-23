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
    sample_indices: &[i64],
) -> Result<SampleSelection, BgenError> {
    let mut file_to_selected_index = vec![usize::MAX; sample_count];
    let mut selected_file_indices = Vec::with_capacity(sample_indices.len());
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
        selected_file_indices.push(sample_index);
        if sample_index != selected_index {
            is_identity = false;
        }
    }
    let contiguous_file_index_start = selected_file_indices
        .first()
        .copied()
        .filter(|_| selected_file_indices.windows(2).all(|sample_window| sample_window[1] == sample_window[0] + 1));
    Ok(SampleSelection {
        selected_sample_count: sample_indices.len(),
        file_to_selected_index,
        selected_file_indices,
        is_identity,
        contiguous_file_index_start,
    })
}

#[cfg(test)]
mod tests {
    use super::build_sample_selection;

    #[test]
    fn sample_selection_records_contiguous_file_index_start() {
        let identity_selection = build_sample_selection(4, &[0, 1, 2, 3]).expect("identity selection should build");
        assert!(identity_selection.is_identity);
        assert_eq!(identity_selection.contiguous_file_index_start, Some(0));

        let contiguous_subset = build_sample_selection(5, &[1, 2, 3]).expect("contiguous subset should build");
        assert!(!contiguous_subset.is_identity);
        assert_eq!(contiguous_subset.contiguous_file_index_start, Some(1));

        let shuffled_subset = build_sample_selection(5, &[1, 3, 2]).expect("shuffled subset should build");
        assert_eq!(shuffled_subset.contiguous_file_index_start, None);

        let empty_subset = build_sample_selection(5, &[]).expect("empty subset should build");
        assert_eq!(empty_subset.contiguous_file_index_start, None);
    }
}
