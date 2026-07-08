#![allow(clippy::missing_errors_doc)]

use std::path::{Path, PathBuf};

mod alignment;
mod cache;
mod chromosome;
mod error;
mod list;
mod loco;
mod source;

pub use error::PredictionError;
pub use source::{ChromosomePredictionMatrix, MultiPredictionSource, PredictionSource};

#[cfg(test)]
use alignment::align_prediction_values;
#[cfg(test)]
use cache::LocoPredictionCache;
#[cfg(test)]
use chromosome::normalize_chromosome;
#[cfg(test)]
use list::PredictionListEntry;
use list::{find_prediction_list_entry, parse_prediction_list_file};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PredictionLocoPath {
    pub phenotype_name: String,
    pub loco_file_path: PathBuf,
}

pub fn resolve_prediction_loco_paths(
    prediction_list_path: &Path,
    phenotype_names: &[String],
) -> Result<Vec<PredictionLocoPath>, PredictionError> {
    let entries = parse_prediction_list_file(prediction_list_path)?;
    phenotype_names
        .iter()
        .map(|phenotype_name| {
            let entry = find_prediction_list_entry(&entries, phenotype_name)?;
            Ok(PredictionLocoPath {
                phenotype_name: phenotype_name.clone(),
                loco_file_path: entry.loco_file_path.clone(),
            })
        })
        .collect()
}
