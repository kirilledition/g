use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::PredictionError;

#[derive(Debug)]
pub(super) struct PredictionList {
    paths_by_phenotype: HashMap<String, PathBuf>,
}

impl PredictionList {
    pub(super) fn loco_file_path(&self, phenotype_name: &str) -> Result<&Path, PredictionError> {
        self.paths_by_phenotype.get(phenotype_name).map(PathBuf::as_path).ok_or_else(|| {
            let mut available_phenotypes = self.paths_by_phenotype.keys().cloned().collect::<Vec<_>>();
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
    let mut paths_by_phenotype = HashMap::new();
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
        paths_by_phenotype.entry(phenotype_name.to_string()).or_insert(resolved_loco_file_path);
    }

    if paths_by_phenotype.is_empty() {
        return Err(PredictionError::EmptyPredictionList(prediction_list_path.to_path_buf()));
    }
    Ok(PredictionList { paths_by_phenotype })
}
