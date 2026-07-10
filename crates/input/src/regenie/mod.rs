#![allow(clippy::missing_errors_doc)]

use std::path::{Path, PathBuf};

use crate::error::InputResult;

mod alignment;
mod cache;
mod error;
mod list;
mod loco;
mod source;

pub use error::PredictionError;
pub use source::ChromosomePredictionMatrix;
pub(crate) use source::{PredictionSource, PredictionSourceLoader};

use list::{find_prediction_list_entry, parse_prediction_list_file};

#[must_use]
fn normalize_chromosome(chromosome: &str) -> String {
    let normalized = chromosome.to_ascii_lowercase();
    let without_prefix = normalized.strip_prefix("chr").unwrap_or(&normalized);
    if without_prefix.chars().all(|character| character.is_ascii_digit()) {
        without_prefix.parse::<u64>().map_or_else(|_| without_prefix.to_string(), |value| value.to_string())
    } else {
        without_prefix.to_string()
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PredictionLocoPath {
    pub phenotype_name: String,
    pub loco_file_path: PathBuf,
}

pub fn resolve_prediction_loco_paths(
    prediction_list_path: &Path,
    phenotype_names: &[String],
) -> InputResult<Vec<PredictionLocoPath>> {
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
        .collect::<Result<Vec<_>, PredictionError>>()
        .map_err(Into::into)
}
