use std::collections::{HashMap, HashSet};

pub(super) struct SampleRowIndicesByKey<'identifier> {
    sample_row_indices: HashMap<(&'identifier str, &'identifier str), usize>,
}

pub(super) struct ObservedTableSampleKeys {
    observed_sample_rows: Vec<bool>,
    unmatched_sample_keys: HashSet<(String, String)>,
}

pub(super) fn build_sample_row_indices_by_key<'identifier>(
    family_identifiers: &'identifier [String],
    individual_identifiers: &'identifier [String],
) -> Result<SampleRowIndicesByKey<'identifier>, String> {
    if family_identifiers.len() != individual_identifiers.len() {
        return Err(format!(
            "Sample identifier arrays must have equal length: family_identifiers={}, individual_identifiers={}.",
            family_identifiers.len(),
            individual_identifiers.len(),
        ));
    }

    let mut sample_row_indices = HashMap::with_capacity(individual_identifiers.len());
    for (sample_row_index, (family_identifier, individual_identifier)) in
        family_identifiers.iter().zip(individual_identifiers).enumerate()
    {
        if family_identifier.is_empty() {
            return Err(format!(
                "Empty FID found at row {sample_row_index} in BGEN/sample identifiers; FID and IID must both be non-empty."
            ));
        }
        if individual_identifier.is_empty() {
            return Err(format!(
                "Empty IID found at row {sample_row_index} in BGEN/sample identifiers; FID and IID must both be non-empty."
            ));
        }
        let sample_key = (family_identifier.as_str(), individual_identifier.as_str());
        if sample_row_indices.insert(sample_key, sample_row_index).is_some() {
            return Err(format!(
                "Duplicate sample key '{family_identifier}_{individual_identifier}' found in BGEN/sample identifiers; (FID, IID) pairs must be unique."
            ));
        }
    }
    Ok(SampleRowIndicesByKey { sample_row_indices })
}

impl SampleRowIndicesByKey<'_> {
    pub(super) fn sample_row_index(&self, family_identifier: &str, individual_identifier: &str) -> Option<usize> {
        self.sample_row_indices.get(&(family_identifier, individual_identifier)).copied()
    }
}

impl ObservedTableSampleKeys {
    pub(super) fn new(sample_count: usize) -> Self {
        Self { observed_sample_rows: vec![false; sample_count], unmatched_sample_keys: HashSet::new() }
    }

    pub(super) fn insert(
        &mut self,
        family_identifier: &str,
        individual_identifier: &str,
        sample_row_index: Option<usize>,
    ) -> bool {
        sample_row_index.map_or_else(
            || self.unmatched_sample_keys.insert((family_identifier.to_string(), individual_identifier.to_string())),
            |sample_row_index| insert_observed_sample_row(&mut self.observed_sample_rows, sample_row_index),
        )
    }
}

fn insert_observed_sample_row(observed_sample_rows: &mut [bool], sample_row_index: usize) -> bool {
    let observed_sample_row = observed_sample_rows
        .get_mut(sample_row_index)
        .expect("sample row indices originate from the same identifier arrays as the observation tracker");
    if *observed_sample_row {
        return false;
    }
    *observed_sample_row = true;
    true
}

pub(super) fn duplicate_table_sample_key_error(
    source_name: &str,
    family_identifier: &str,
    individual_identifier: &str,
) -> String {
    format!(
        "Duplicate sample key '{family_identifier}_{individual_identifier}' found in {source_name}; (FID, IID) pairs must be unique."
    )
}
