use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::PredictionError;

#[derive(Debug, Clone)]
pub(super) struct PredictionListEntry {
    pub(super) phenotype_name: String,
    pub(super) loco_file_path: PathBuf,
}

pub(super) fn find_prediction_list_entry<'entry>(
    entries: &'entry [PredictionListEntry],
    phenotype_name: &str,
) -> Result<&'entry PredictionListEntry, PredictionError> {
    entries.iter().find(|entry| entry.phenotype_name == phenotype_name).ok_or_else(|| {
        PredictionError::MissingPhenotype {
            phenotype_name: phenotype_name.to_string(),
            available_phenotypes: entries.iter().map(|entry| entry.phenotype_name.clone()).collect(),
        }
    })
}

pub(super) fn parse_prediction_list_file(
    prediction_list_path: &Path,
) -> Result<Vec<PredictionListEntry>, PredictionError> {
    if !prediction_list_path.exists() {
        return Err(PredictionError::PredictionListNotFound(prediction_list_path.to_path_buf()));
    }

    let prediction_list_directory = prediction_list_path.parent().unwrap_or_else(|| Path::new(""));
    let file = File::open(prediction_list_path)?;
    let mut entries = Vec::new();
    for (line_index, line_result) in BufReader::new(file).lines().enumerate() {
        let line_number = line_index + 1;
        let line = line_result?;
        let mut fields = line.split_whitespace();
        let Some(phenotype_name) = fields.next() else {
            continue;
        };
        let Some(loco_file_path) = fields.next() else {
            return Err(PredictionError::InvalidPredictionListLine { line_number, field_count: 1 });
        };
        if fields.next().is_some() {
            let field_count = 3 + fields.count();
            return Err(PredictionError::InvalidPredictionListLine { line_number, field_count });
        }
        let raw_loco_file_path = PathBuf::from(loco_file_path);
        let resolved_loco_file_path = if raw_loco_file_path.is_absolute() {
            raw_loco_file_path
        } else {
            prediction_list_directory.join(raw_loco_file_path)
        };
        entries.push(PredictionListEntry {
            phenotype_name: phenotype_name.to_string(),
            loco_file_path: resolved_loco_file_path,
        });
    }

    if entries.is_empty() {
        return Err(PredictionError::EmptyPredictionList(prediction_list_path.to_path_buf()));
    }
    Ok(entries)
}
