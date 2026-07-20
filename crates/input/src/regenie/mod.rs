use std::path::{Path, PathBuf};
use std::sync::Arc;

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

use list::parse_prediction_list_file;

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

#[derive(Debug, Eq, PartialEq)]
pub struct PredictionLocoPath {
    pub phenotype_name: Arc<str>,
    pub loco_file_path: PathBuf,
}

/// Resolve one LOCO prediction file for every requested phenotype.
///
/// # Errors
///
/// Returns an error when the prediction list cannot be read or does not define
/// a LOCO file for every requested phenotype.
pub fn resolve_prediction_loco_paths(
    prediction_list_path: &Path,
    phenotype_names: &[String],
) -> InputResult<Vec<PredictionLocoPath>> {
    let prediction_list = parse_prediction_list_file(prediction_list_path)?;
    phenotype_names
        .iter()
        .map(|phenotype_name| {
            Ok(PredictionLocoPath {
                phenotype_name: Arc::from(phenotype_name.as_str()),
                loco_file_path: prediction_list.loco_file_path(phenotype_name)?.to_path_buf(),
            })
        })
        .collect::<Result<Vec<_>, PredictionError>>()
        .map_err(Into::into)
}
