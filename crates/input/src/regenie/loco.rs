use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use super::PredictionError;
use super::chromosome::normalize_chromosome;

#[derive(Debug)]
pub(super) struct LocoPredictions {
    pub(super) sample_index: LocoSampleIndex,
    pub(super) chromosome_predictions: HashMap<String, Arc<[f32]>>,
}

#[derive(Debug)]
pub(super) struct LocoSampleIndex {
    pub(super) family_identifiers: Vec<String>,
    pub(super) individual_identifiers: Vec<String>,
}

pub(super) fn parse_loco_file(loco_file_path: &Path) -> Result<LocoPredictions, PredictionError> {
    if !loco_file_path.exists() {
        return Err(PredictionError::LocoFileNotFound(loco_file_path.to_path_buf()));
    }

    let file = File::open(loco_file_path)?;
    let mut sample_index = None;
    let mut chromosome_predictions = HashMap::new();
    for (line_index, line_result) in BufReader::new(file).lines().enumerate() {
        let line_number = line_index + 1;
        let line = line_result?;
        let stripped_line = line.trim();
        if stripped_line.is_empty() {
            continue;
        }
        if line_number == 1 {
            sample_index = Some(parse_loco_sample_identifiers(stripped_line)?);
            continue;
        }
        let mut fields = stripped_line.split_whitespace();
        let Some(chromosome_field) = fields.next() else {
            return Err(PredictionError::InvalidLocoDataLine { line_number, field_count: 0 });
        };
        let prediction_field_count = fields.clone().count();
        if prediction_field_count == 0 {
            return Err(PredictionError::InvalidLocoDataLine { line_number, field_count: 1 });
        }
        let sample_index_reference =
            sample_index.as_ref().ok_or_else(|| PredictionError::MissingLocoHeader(loco_file_path.to_path_buf()))?;
        if prediction_field_count != sample_index_reference.family_identifiers.len() {
            return Err(PredictionError::LocoPredictionCountMismatch {
                line_number,
                expected_count: sample_index_reference.family_identifiers.len(),
                observed_count: prediction_field_count,
            });
        }
        let chromosome = normalize_chromosome(chromosome_field);
        if chromosome_predictions.contains_key(&chromosome) {
            return Err(PredictionError::DuplicateChromosome { chromosome });
        }
        let prediction_values = fields
            .map(|value| {
                value.parse::<f32>().map_err(|source| PredictionError::InvalidPredictionValue {
                    line_number,
                    value: value.to_string(),
                    source,
                })
            })
            .collect::<Result<Vec<f32>, PredictionError>>()?;
        chromosome_predictions.insert(chromosome, prediction_values.into());
    }

    let sample_index = sample_index.ok_or_else(|| PredictionError::MissingLocoHeader(loco_file_path.to_path_buf()))?;
    if chromosome_predictions.is_empty() {
        return Err(PredictionError::MissingChromosomePredictions(loco_file_path.to_path_buf()));
    }
    Ok(LocoPredictions { sample_index, chromosome_predictions })
}

fn parse_loco_sample_identifiers(header_line: &str) -> Result<LocoSampleIndex, PredictionError> {
    let mut fields = header_line.split_whitespace();
    let Some(observed_marker) = fields.next() else {
        return Err(PredictionError::EmptyLocoHeader);
    };
    let sample_identifier_count = fields.clone().count();
    if sample_identifier_count == 0 {
        return Err(PredictionError::EmptyLocoHeader);
    }
    if observed_marker != "FID_IID" {
        return Err(PredictionError::InvalidLocoHeaderMarker { observed_marker: observed_marker.to_string() });
    }

    let mut family_identifiers = Vec::with_capacity(sample_identifier_count);
    let mut individual_identifiers = Vec::with_capacity(sample_identifier_count);
    for (sample_index, sample_identifier) in fields.enumerate() {
        let Some((family_identifier, individual_identifier)) = sample_identifier.split_once('_') else {
            return Err(PredictionError::InvalidLocoSampleIdentifier {
                sample_index,
                sample_identifier: sample_identifier.to_string(),
            });
        };
        family_identifiers.push(family_identifier.to_string());
        individual_identifiers.push(individual_identifier.to_string());
    }

    Ok(LocoSampleIndex { family_identifiers, individual_identifiers })
}
