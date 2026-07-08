use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::sample::SampleKeyMode;

use super::PredictionError;
use super::alignment::{
    build_sample_alignment_indices, validate_loco_sample_keys, validate_unique_loco_individual_identifiers,
};
use super::loco::{LocoPredictions, parse_loco_file};

#[derive(Debug, Default)]
pub(super) struct LocoPredictionCache {
    predictions_by_path: HashMap<PathBuf, LocoPredictions>,
}

#[derive(Debug, Default)]
pub(super) struct LocoAlignmentCache {
    alignment_indices_by_path: HashMap<PathBuf, Vec<usize>>,
}

impl LocoPredictionCache {
    pub(super) fn predictions(&mut self, loco_file_path: &Path) -> Result<&LocoPredictions, PredictionError> {
        let cache_key = cache_key_for_loco_path(loco_file_path);
        match self.predictions_by_path.entry(cache_key) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let loco_predictions = parse_loco_file(loco_file_path)?;
                Ok(entry.insert(loco_predictions))
            }
        }
    }

    #[cfg(test)]
    pub(super) fn cached_file_count(&self) -> usize {
        self.predictions_by_path.len()
    }
}

impl LocoAlignmentCache {
    pub(super) fn alignment_indices(
        &mut self,
        loco_file_path: &Path,
        loco_predictions: &LocoPredictions,
        target_family_identifiers: &[String],
        target_individual_identifiers: &[String],
        sample_key_mode: SampleKeyMode,
    ) -> Result<&[usize], PredictionError> {
        let cache_key = cache_key_for_loco_path(loco_file_path);
        match self.alignment_indices_by_path.entry(cache_key) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut().as_slice()),
            std::collections::hash_map::Entry::Vacant(entry) => {
                validate_loco_sample_keys(&loco_predictions.sample_index)?;
                if sample_key_mode == SampleKeyMode::Iid {
                    validate_unique_loco_individual_identifiers(&loco_predictions.sample_index)?;
                }
                let alignment_indices = build_sample_alignment_indices(
                    &loco_predictions.sample_index,
                    target_family_identifiers,
                    target_individual_identifiers,
                    sample_key_mode,
                )?;
                Ok(entry.insert(alignment_indices).as_slice())
            }
        }
    }
}

fn cache_key_for_loco_path(loco_file_path: &Path) -> PathBuf {
    loco_file_path.canonicalize().unwrap_or_else(|_| loco_file_path.to_path_buf())
}
