use std::collections::{HashMap, HashSet};

use g_plan::SampleKeyMode;

pub(super) enum SampleRowIndicesByKey<'identifier> {
    Iid(HashMap<&'identifier str, usize>),
    FidIid(HashMap<(&'identifier str, &'identifier str), usize>),
}

pub(super) enum ObservedTableSampleKeys {
    Iid { observed_sample_rows: Vec<bool>, unmatched_individual_identifiers: HashSet<String> },
    FidIid { observed_sample_rows: Vec<bool>, unmatched_sample_keys: HashSet<(String, String)> },
}

pub(super) fn build_sample_row_indices_by_key<'identifier>(
    sample_key_mode: SampleKeyMode,
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

    match sample_key_mode {
        SampleKeyMode::Iid => {
            let mut sample_row_indices_by_key = HashMap::with_capacity(individual_identifiers.len());
            for (sample_row_index, individual_identifier) in individual_identifiers.iter().enumerate() {
                if individual_identifier.is_empty() {
                    continue;
                }
                if sample_row_indices_by_key.insert(individual_identifier.as_str(), sample_row_index).is_some() {
                    return Err(format!(
                        "Duplicate IID '{individual_identifier}' found in BGEN/sample identifiers; sample_key_mode='iid' requires unique non-null IID values. Use sample_key_mode='fid_iid' for datasets with non-globally-unique IID."
                    ));
                }
            }
            Ok(SampleRowIndicesByKey::Iid(sample_row_indices_by_key))
        }
        SampleKeyMode::FidIid => {
            let mut sample_row_indices_by_key = HashMap::with_capacity(individual_identifiers.len());
            for (sample_row_index, (family_identifier, individual_identifier)) in
                family_identifiers.iter().zip(individual_identifiers).enumerate()
            {
                if individual_identifier.is_empty() {
                    continue;
                }
                let sample_key = (family_identifier.as_str(), individual_identifier.as_str());
                if sample_row_indices_by_key.insert(sample_key, sample_row_index).is_some() {
                    return Err(format!(
                        "Duplicate sample key '{family_identifier}_{individual_identifier}' found in BGEN/sample identifiers; sample_key_mode='fid_iid' requires unique (FID, IID) values."
                    ));
                }
            }
            Ok(SampleRowIndicesByKey::FidIid(sample_row_indices_by_key))
        }
    }
}

impl SampleRowIndicesByKey<'_> {
    pub(super) fn sample_row_index(&self, family_identifier: &str, individual_identifier: &str) -> Option<usize> {
        match self {
            Self::Iid(sample_row_indices_by_key) => sample_row_indices_by_key.get(individual_identifier).copied(),
            Self::FidIid(sample_row_indices_by_key) => {
                sample_row_indices_by_key.get(&(family_identifier, individual_identifier)).copied()
            }
        }
    }
}

impl ObservedTableSampleKeys {
    pub(super) fn new(sample_key_mode: SampleKeyMode, sample_count: usize) -> Self {
        match sample_key_mode {
            SampleKeyMode::Iid => Self::Iid {
                observed_sample_rows: vec![false; sample_count],
                unmatched_individual_identifiers: HashSet::new(),
            },
            SampleKeyMode::FidIid => {
                Self::FidIid { observed_sample_rows: vec![false; sample_count], unmatched_sample_keys: HashSet::new() }
            }
        }
    }

    pub(super) fn insert(
        &mut self,
        family_identifier: &str,
        individual_identifier: &str,
        sample_row_index: Option<usize>,
    ) -> bool {
        match self {
            Self::Iid { observed_sample_rows, unmatched_individual_identifiers } => sample_row_index.map_or_else(
                || unmatched_individual_identifiers.insert(individual_identifier.to_string()),
                |sample_row_index| insert_observed_sample_row(observed_sample_rows, sample_row_index),
            ),
            Self::FidIid { observed_sample_rows, unmatched_sample_keys } => sample_row_index.map_or_else(
                || unmatched_sample_keys.insert((family_identifier.to_string(), individual_identifier.to_string())),
                |sample_row_index| insert_observed_sample_row(observed_sample_rows, sample_row_index),
            ),
        }
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
    sample_key_mode: SampleKeyMode,
    family_identifier: &str,
    individual_identifier: &str,
) -> String {
    if sample_key_mode == SampleKeyMode::FidIid {
        return format!(
            "Duplicate sample key '{family_identifier}_{individual_identifier}' found in {source_name}; sample_key_mode='fid_iid' requires unique (FID, IID) values."
        );
    }
    format!(
        "Duplicate IID '{individual_identifier}' found in {source_name}; sample_key_mode='iid' requires unique non-null IID values."
    )
}
