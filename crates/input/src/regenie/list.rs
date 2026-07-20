use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::PredictionError;

#[derive(Debug)]
pub(super) struct PredictionList {
    entries_by_phenotype: HashMap<String, PredictionListEntry>,
}

#[derive(Debug)]
struct PredictionListEntry {
    loco_file_path: PathBuf,
    line_number: usize,
}

impl PredictionList {
    pub(super) fn loco_file_path(&self, phenotype_name: &str) -> Result<&Path, PredictionError> {
        self.entries_by_phenotype.get(phenotype_name).map(|entry| entry.loco_file_path.as_path()).ok_or_else(|| {
            let mut available_phenotypes = self.entries_by_phenotype.keys().cloned().collect::<Vec<_>>();
            available_phenotypes.sort_unstable();
            PredictionError::MissingPhenotype { phenotype_name: phenotype_name.to_string(), available_phenotypes }
        })
    }
}

pub(super) fn parse_prediction_list_file(prediction_list_path: &Path) -> Result<PredictionList, PredictionError> {
    if !prediction_list_path.exists() {
        return Err(PredictionError::PredictionListNotFound(prediction_list_path.to_path_buf()));
    }

    let prediction_list_directory = prediction_list_path.parent().unwrap_or_else(|| Path::new(""));
    let file = File::open(prediction_list_path)?;
    let mut entries_by_phenotype: HashMap<String, PredictionListEntry> = HashMap::new();
    for (line_index, line_result) in BufReader::new(file).lines().enumerate() {
        let line_number = line_index + 1;
        let line = line_result?;
        let mut fields = line.split_ascii_whitespace();
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
        if let Some(first_entry) = entries_by_phenotype.get(phenotype_name) {
            return Err(PredictionError::DuplicatePredictionListPhenotype {
                phenotype_name: phenotype_name.to_string(),
                first_line_number: first_entry.line_number,
                duplicate_line_number: line_number,
            });
        }
        entries_by_phenotype.insert(
            phenotype_name.to_string(),
            PredictionListEntry { loco_file_path: resolved_loco_file_path, line_number },
        );
    }

    if entries_by_phenotype.is_empty() {
        return Err(PredictionError::EmptyPredictionList(prediction_list_path.to_path_buf()));
    }
    Ok(PredictionList { entries_by_phenotype })
}
