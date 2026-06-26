use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PredictionError {
    #[error("Prediction list file not found: {0}")]
    PredictionListNotFound(PathBuf),
    #[error("LOCO file not found: {0}")]
    LocoFileNotFound(PathBuf),
    #[error("Prediction list file is empty: {0}")]
    EmptyPredictionList(PathBuf),
    #[error("Prediction list line {line_number}: expected 2 space-delimited fields, found {field_count}.")]
    InvalidPredictionListLine { line_number: usize, field_count: usize },
    #[error(
        "Phenotype '{phenotype_name}' not found in prediction list. Available phenotypes: {available_phenotypes:?}"
    )]
    MissingPhenotype { phenotype_name: String, available_phenotypes: Vec<String> },
    #[error("LOCO header must contain at least the FID_IID marker and one sample identifier.")]
    EmptyLocoHeader,
    #[error("LOCO header must start with 'FID_IID', found '{observed_marker}'.")]
    InvalidLocoHeaderMarker { observed_marker: String },
    #[error(
        "Sample identifier at position {sample_index} ('{sample_identifier}') does not contain underscore separator for FID_IID format."
    )]
    InvalidLocoSampleIdentifier { sample_index: usize, sample_identifier: String },
    #[error("LOCO data line {line_number}: expected chromosome and predictions, found {field_count} fields.")]
    InvalidLocoDataLine { line_number: usize, field_count: usize },
    #[error("LOCO data line {line_number}: expected {expected_count} predictions, found {observed_count}.")]
    LocoPredictionCountMismatch { line_number: usize, expected_count: usize, observed_count: usize },
    #[error("LOCO file contains duplicate chromosome: {chromosome}")]
    DuplicateChromosome { chromosome: String },
    #[error("LOCO file is empty or missing header: {0}")]
    MissingLocoHeader(PathBuf),
    #[error("LOCO file contains no chromosome predictions: {0}")]
    MissingChromosomePredictions(PathBuf),
    #[error("Target family and individual identifier arrays must have the same length.")]
    TargetSampleLengthMismatch,
    #[error("Duplicate target sample key: {sample_key}")]
    DuplicateTargetSampleKey { sample_key: String },
    #[error("Duplicate LOCO sample key: {sample_key}")]
    DuplicateLocoSampleKey { sample_key: String },
    #[error(
        "Duplicate target IID '{individual_identifier}' found; sample_key_mode='iid' requires unique non-null IID values."
    )]
    DuplicateTargetIid { individual_identifier: String },
    #[error("Empty target IID found; sample_key_mode='iid' requires non-null IID values.")]
    EmptyTargetIid,
    #[error(
        "Duplicate LOCO IID '{individual_identifier}' found; sample_key_mode='iid' requires unique non-null IID values."
    )]
    DuplicateLocoIid { individual_identifier: String },
    #[error("Empty LOCO IID found; sample_key_mode='iid' requires non-null IID values.")]
    EmptyLocoIid,
    #[error("Target samples not found in LOCO file: {0}")]
    MissingTargetSamples(String),
    #[error(
        "Chromosome '{chromosome}' (normalized: '{normalized_chromosome}') not found in LOCO file. Available chromosomes: {available_chromosomes:?}"
    )]
    MissingChromosome { chromosome: String, normalized_chromosome: String, available_chromosomes: Vec<String> },
    #[error("Failed to parse LOCO prediction value '{value}' on line {line_number}: {source}")]
    InvalidPredictionValue { line_number: usize, value: String, source: std::num::ParseFloatError },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
