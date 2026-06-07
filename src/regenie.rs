#![allow(clippy::missing_errors_doc)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use crate::sample::{MultiAlignedSampleData, SampleKeyMode};

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

#[derive(Debug, Clone)]
struct PredictionListEntry {
    phenotype_name: String,
    loco_file_path: PathBuf,
}

#[derive(Debug)]
struct LocoPredictions {
    sample_index: LocoSampleIndex,
    chromosome_predictions: HashMap<String, Arc<[f32]>>,
}

#[derive(Debug)]
struct LocoSampleIndex {
    family_identifiers: Vec<String>,
    individual_identifiers: Vec<String>,
}

#[derive(Debug, Default)]
struct LocoPredictionCache {
    predictions_by_path: HashMap<PathBuf, LocoPredictions>,
}

#[derive(Debug, Default)]
struct LocoAlignmentCache {
    alignment_indices_by_path: HashMap<PathBuf, Vec<usize>>,
}

#[derive(Debug)]
pub struct PredictionSource {
    chromosome_predictions: HashMap<String, Arc<[f32]>>,
}

#[derive(Debug)]
pub struct MultiPredictionSource {
    phenotype_names: Vec<String>,
    chromosome_predictions_by_trait: Vec<HashMap<String, Arc<[f32]>>>,
}

impl LocoPredictionCache {
    fn predictions(&mut self, loco_file_path: &Path) -> Result<&LocoPredictions, PredictionError> {
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
    fn cached_file_count(&self) -> usize {
        self.predictions_by_path.len()
    }
}

impl LocoAlignmentCache {
    fn alignment_indices(
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

impl PredictionSource {
    pub fn load(
        prediction_list_path: &Path,
        phenotype_name: &str,
        target_family_identifiers: &[String],
        target_individual_identifiers: &[String],
        sample_key_mode: SampleKeyMode,
    ) -> Result<Self, PredictionError> {
        validate_target_sample_keys(target_family_identifiers, target_individual_identifiers)?;
        if sample_key_mode == SampleKeyMode::Iid {
            validate_unique_target_individual_identifiers(target_individual_identifiers)?;
        }
        let entries = parse_prediction_list_file(prediction_list_path)?;
        let entry = find_prediction_list_entry(&entries, phenotype_name)?;
        let mut loco_prediction_cache = LocoPredictionCache::default();
        let mut loco_alignment_cache = LocoAlignmentCache::default();
        let aligned_predictions = load_aligned_chromosome_predictions(
            entry,
            &mut loco_prediction_cache,
            &mut loco_alignment_cache,
            target_family_identifiers,
            target_individual_identifiers,
            sample_key_mode,
        )?;
        Ok(Self { chromosome_predictions: aligned_predictions })
    }

    pub fn chromosome_predictions(&self, chromosome: &str) -> Result<&[f32], PredictionError> {
        let normalized_chromosome = normalize_chromosome(chromosome);
        self.chromosome_predictions.get(&normalized_chromosome).map(|values| values.as_ref()).ok_or_else(|| {
            let mut available_chromosomes: Vec<String> = self.chromosome_predictions.keys().cloned().collect();
            available_chromosomes.sort();
            PredictionError::MissingChromosome {
                chromosome: chromosome.to_string(),
                normalized_chromosome,
                available_chromosomes,
            }
        })
    }
}

impl MultiPredictionSource {
    pub fn load(
        prediction_list_path: &Path,
        phenotype_names: &[String],
        target_family_identifiers: &[String],
        target_individual_identifiers: &[String],
        sample_key_mode: SampleKeyMode,
    ) -> Result<Self, PredictionError> {
        let entries = parse_prediction_list_file(prediction_list_path)?;
        let mut loco_prediction_cache = LocoPredictionCache::default();
        Self::load_from_entries_with_cache(
            &entries,
            phenotype_names,
            target_family_identifiers,
            target_individual_identifiers,
            sample_key_mode,
            &mut loco_prediction_cache,
        )
    }

    pub fn load_grouped(
        prediction_list_path: &Path,
        aligned_sample_data_groups: &[&MultiAlignedSampleData],
        sample_key_mode: SampleKeyMode,
    ) -> Result<Vec<Self>, PredictionError> {
        let entries = parse_prediction_list_file(prediction_list_path)?;
        let mut loco_prediction_cache = LocoPredictionCache::default();
        aligned_sample_data_groups
            .iter()
            .map(|aligned_sample_data| {
                Self::load_from_entries_with_cache(
                    &entries,
                    &aligned_sample_data.phenotype_names,
                    &aligned_sample_data.family_identifiers,
                    &aligned_sample_data.individual_identifiers,
                    sample_key_mode,
                    &mut loco_prediction_cache,
                )
            })
            .collect()
    }

    fn load_from_entries_with_cache(
        entries: &[PredictionListEntry],
        phenotype_names: &[String],
        target_family_identifiers: &[String],
        target_individual_identifiers: &[String],
        sample_key_mode: SampleKeyMode,
        loco_prediction_cache: &mut LocoPredictionCache,
    ) -> Result<Self, PredictionError> {
        validate_target_sample_keys(target_family_identifiers, target_individual_identifiers)?;
        if sample_key_mode == SampleKeyMode::Iid {
            validate_unique_target_individual_identifiers(target_individual_identifiers)?;
        }
        let mut chromosome_predictions_by_trait = Vec::with_capacity(phenotype_names.len());
        let mut loco_alignment_cache = LocoAlignmentCache::default();
        for phenotype_name in phenotype_names {
            let entry = find_prediction_list_entry(entries, phenotype_name)?;
            let aligned_predictions = load_aligned_chromosome_predictions(
                entry,
                loco_prediction_cache,
                &mut loco_alignment_cache,
                target_family_identifiers,
                target_individual_identifiers,
                sample_key_mode,
            )?;
            chromosome_predictions_by_trait.push(aligned_predictions);
        }
        Ok(Self { phenotype_names: phenotype_names.to_vec(), chromosome_predictions_by_trait })
    }

    pub fn chromosome_prediction_matrix(&self, chromosome: &str) -> Result<(usize, usize, Vec<f32>), PredictionError> {
        let normalized_chromosome = normalize_chromosome(chromosome);
        let trait_count = self.phenotype_names.len();
        let mut prediction_matrix_values = Vec::new();
        let mut sample_count = None;
        for chromosome_predictions in &self.chromosome_predictions_by_trait {
            let Some(prediction_values) = chromosome_predictions.get(&normalized_chromosome) else {
                let mut available_chromosomes: Vec<String> = chromosome_predictions.keys().cloned().collect();
                available_chromosomes.sort();
                return Err(PredictionError::MissingChromosome {
                    chromosome: chromosome.to_string(),
                    normalized_chromosome,
                    available_chromosomes,
                });
            };
            if let Some(expected_sample_count) = sample_count
                && prediction_values.len() != expected_sample_count
            {
                return Err(PredictionError::LocoPredictionCountMismatch {
                    line_number: 0,
                    expected_count: expected_sample_count,
                    observed_count: prediction_values.len(),
                });
            } else if sample_count.is_none() {
                sample_count = Some(prediction_values.len());
                prediction_matrix_values.reserve_exact(trait_count * prediction_values.len());
            }
            prediction_matrix_values.extend_from_slice(prediction_values.as_ref());
        }
        Ok((trait_count, sample_count.unwrap_or(0), prediction_matrix_values))
    }
}

#[must_use]
pub fn normalize_chromosome(chromosome: &str) -> String {
    let normalized = chromosome.to_ascii_lowercase();
    let without_prefix = normalized.strip_prefix("chr").unwrap_or(&normalized);
    if without_prefix.chars().all(|character| character.is_ascii_digit()) {
        without_prefix.parse::<u64>().map_or_else(|_| without_prefix.to_string(), |value| value.to_string())
    } else {
        without_prefix.to_string()
    }
}

fn find_prediction_list_entry<'entry>(
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

fn load_aligned_chromosome_predictions(
    entry: &PredictionListEntry,
    loco_prediction_cache: &mut LocoPredictionCache,
    loco_alignment_cache: &mut LocoAlignmentCache,
    target_family_identifiers: &[String],
    target_individual_identifiers: &[String],
    sample_key_mode: SampleKeyMode,
) -> Result<HashMap<String, Arc<[f32]>>, PredictionError> {
    let loco_predictions = loco_prediction_cache.predictions(&entry.loco_file_path)?;
    let alignment_indices = loco_alignment_cache.alignment_indices(
        &entry.loco_file_path,
        loco_predictions,
        target_family_identifiers,
        target_individual_identifiers,
        sample_key_mode,
    )?;
    Ok(align_chromosome_predictions(loco_predictions, alignment_indices))
}

fn align_chromosome_predictions(
    loco_predictions: &LocoPredictions,
    alignment_indices: &[usize],
) -> HashMap<String, Arc<[f32]>> {
    loco_predictions
        .chromosome_predictions
        .iter()
        .map(|(chromosome, prediction_values)| {
            (chromosome.clone(), align_prediction_values(prediction_values, alignment_indices))
        })
        .collect()
}

fn align_prediction_values(prediction_values: &Arc<[f32]>, alignment_indices: &[usize]) -> Arc<[f32]> {
    if is_identity_alignment(alignment_indices, prediction_values.len()) {
        return Arc::clone(prediction_values);
    }
    alignment_indices.iter().map(|sample_index| prediction_values[*sample_index]).collect::<Vec<f32>>().into()
}

fn is_identity_alignment(alignment_indices: &[usize], source_sample_count: usize) -> bool {
    alignment_indices.len() == source_sample_count
        && alignment_indices
            .iter()
            .enumerate()
            .all(|(expected_sample_index, sample_index)| expected_sample_index == *sample_index)
}

fn parse_prediction_list_file(prediction_list_path: &Path) -> Result<Vec<PredictionListEntry>, PredictionError> {
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

fn cache_key_for_loco_path(loco_file_path: &Path) -> PathBuf {
    loco_file_path.canonicalize().unwrap_or_else(|_| loco_file_path.to_path_buf())
}

fn parse_loco_file(loco_file_path: &Path) -> Result<LocoPredictions, PredictionError> {
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

fn validate_target_sample_keys(
    target_family_identifiers: &[String],
    target_individual_identifiers: &[String],
) -> Result<(), PredictionError> {
    if target_family_identifiers.len() != target_individual_identifiers.len() {
        return Err(PredictionError::TargetSampleLengthMismatch);
    }
    let mut observed_sample_keys = HashMap::with_capacity(target_family_identifiers.len());
    for (family_identifier, individual_identifier) in
        target_family_identifiers.iter().zip(target_individual_identifiers.iter())
    {
        let sample_key = (family_identifier.as_str(), individual_identifier.as_str());
        let occurrence_count = observed_sample_keys.entry(sample_key).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(PredictionError::DuplicateTargetSampleKey {
                sample_key: format!("{family_identifier}_{individual_identifier}"),
            });
        }
    }
    Ok(())
}

fn validate_loco_sample_keys(loco_sample_index: &LocoSampleIndex) -> Result<(), PredictionError> {
    let mut observed_sample_keys = HashMap::with_capacity(loco_sample_index.family_identifiers.len());
    for (family_identifier, individual_identifier) in
        loco_sample_index.family_identifiers.iter().zip(loco_sample_index.individual_identifiers.iter())
    {
        let sample_key = (family_identifier.as_str(), individual_identifier.as_str());
        let occurrence_count = observed_sample_keys.entry(sample_key).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(PredictionError::DuplicateLocoSampleKey {
                sample_key: format!("{family_identifier}_{individual_identifier}"),
            });
        }
    }
    Ok(())
}

fn validate_unique_target_individual_identifiers(
    target_individual_identifiers: &[String],
) -> Result<(), PredictionError> {
    let mut observed_individual_identifiers = HashMap::with_capacity(target_individual_identifiers.len());
    for individual_identifier in target_individual_identifiers {
        if individual_identifier.is_empty() {
            return Err(PredictionError::EmptyTargetIid);
        }
        let occurrence_count = observed_individual_identifiers.entry(individual_identifier.as_str()).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(PredictionError::DuplicateTargetIid { individual_identifier: individual_identifier.clone() });
        }
    }
    Ok(())
}

fn validate_unique_loco_individual_identifiers(loco_sample_index: &LocoSampleIndex) -> Result<(), PredictionError> {
    let mut observed_individual_identifiers = HashMap::with_capacity(loco_sample_index.individual_identifiers.len());
    for individual_identifier in &loco_sample_index.individual_identifiers {
        if individual_identifier.is_empty() {
            return Err(PredictionError::EmptyLocoIid);
        }
        let occurrence_count = observed_individual_identifiers.entry(individual_identifier.as_str()).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(PredictionError::DuplicateLocoIid { individual_identifier: individual_identifier.clone() });
        }
    }
    Ok(())
}

fn build_sample_alignment_indices(
    loco_sample_index: &LocoSampleIndex,
    target_family_identifiers: &[String],
    target_individual_identifiers: &[String],
    sample_key_mode: SampleKeyMode,
) -> Result<Vec<usize>, PredictionError> {
    if target_family_identifiers.len() != target_individual_identifiers.len() {
        return Err(PredictionError::TargetSampleLengthMismatch);
    }

    if sample_key_mode == SampleKeyMode::Iid {
        return build_individual_identifier_alignment_indices(loco_sample_index, target_individual_identifiers);
    }
    build_family_individual_identifier_alignment_indices(
        loco_sample_index,
        target_family_identifiers,
        target_individual_identifiers,
    )
}

fn build_individual_identifier_alignment_indices(
    loco_sample_index: &LocoSampleIndex,
    target_individual_identifiers: &[String],
) -> Result<Vec<usize>, PredictionError> {
    let mut loco_lookup = HashMap::with_capacity(loco_sample_index.individual_identifiers.len());
    for (sample_index, individual_identifier) in loco_sample_index.individual_identifiers.iter().enumerate() {
        loco_lookup.insert(individual_identifier.as_str(), sample_index);
    }

    let mut alignment_indices = Vec::with_capacity(target_individual_identifiers.len());
    let mut missing_samples = Vec::new();
    for individual_identifier in target_individual_identifiers {
        if let Some(sample_index) = loco_lookup.get(individual_identifier.as_str()) {
            alignment_indices.push(*sample_index);
        } else {
            missing_samples.push(individual_identifier.clone());
        }
    }

    if !missing_samples.is_empty() {
        return Err(PredictionError::MissingTargetSamples(format_missing_samples(&missing_samples)));
    }
    Ok(alignment_indices)
}

fn build_family_individual_identifier_alignment_indices(
    loco_sample_index: &LocoSampleIndex,
    target_family_identifiers: &[String],
    target_individual_identifiers: &[String],
) -> Result<Vec<usize>, PredictionError> {
    let mut loco_lookup = HashMap::with_capacity(loco_sample_index.family_identifiers.len());
    for (sample_index, (family_identifier, individual_identifier)) in
        loco_sample_index.family_identifiers.iter().zip(loco_sample_index.individual_identifiers.iter()).enumerate()
    {
        loco_lookup.insert((family_identifier.as_str(), individual_identifier.as_str()), sample_index);
    }

    let mut alignment_indices = Vec::with_capacity(target_family_identifiers.len());
    let mut missing_samples = Vec::new();
    for (family_identifier, individual_identifier) in
        target_family_identifiers.iter().zip(target_individual_identifiers.iter())
    {
        let key = (family_identifier.as_str(), individual_identifier.as_str());
        if let Some(sample_index) = loco_lookup.get(&key) {
            alignment_indices.push(*sample_index);
        } else {
            missing_samples.push(format!("{family_identifier}_{individual_identifier}"));
        }
    }

    if !missing_samples.is_empty() {
        return Err(PredictionError::MissingTargetSamples(format_missing_samples(&missing_samples)));
    }
    Ok(alignment_indices)
}

fn format_missing_samples(missing_samples: &[String]) -> String {
    let mut sample_list = missing_samples.iter().take(5).cloned().collect::<Vec<String>>().join(", ");
    if missing_samples.len() > 5 {
        let _ = write!(sample_list, ", ... ({} total)", missing_samples.len());
    }
    sample_list
}

#[cfg(test)]
mod tests {
    use std::assert_matches;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::sample::SampleKeyMode;

    use super::{
        LocoPredictionCache, MultiPredictionSource, PredictionError, PredictionListEntry, PredictionSource,
        normalize_chromosome,
    };

    static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

    struct FixtureDirectory {
        path: PathBuf,
    }

    impl FixtureDirectory {
        fn new() -> Self {
            let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("g-regenie-tests-{}-{fixture_id}", std::process::id()));
            fs::create_dir_all(&path).expect("regenie test fixture directory should be created");
            Self { path }
        }

        fn write_file(&self, file_name: &str, contents: &str) -> PathBuf {
            let path = self.path.join(file_name);
            fs::write(&path, contents).expect("regenie test fixture should be written");
            path
        }
    }

    impl Drop for FixtureDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn normalizes_chromosome_labels() {
        assert_eq!(normalize_chromosome("chr22"), "22");
        assert_eq!(normalize_chromosome("CHR01"), "1");
        assert_eq!(normalize_chromosome("chrX"), "x");
    }

    #[test]
    fn prediction_source_aligns_loco_samples_and_normalizes_chromosomes() {
        let fixture = FixtureDirectory::new();
        let loco_path = fixture.write_file("trait.loco", "FID_IID F2_I2 F1_I1\nchr01 0.2 0.1\nX 0.4 0.3\n");
        let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

        let source = PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F1", "F2"]),
            &strings(&["I1", "I2"]),
            SampleKeyMode::FidIid,
        )
        .expect("prediction source should load");

        assert_eq!(source.chromosome_predictions("1").expect("chr1 predictions"), &[0.1, 0.2]);
        assert_eq!(source.chromosome_predictions("chrX").expect("chrX predictions"), &[0.3, 0.4]);
    }

    #[test]
    fn prediction_source_resolves_relative_loco_paths_from_prediction_list_directory() {
        let fixture = FixtureDirectory::new();
        fixture.write_file("trait.loco", "FID_IID F1_I1\n22 0.7\n");
        let prediction_list_path = fixture.write_file("pred.list", "trait trait.loco\n");

        let source = PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::FidIid,
        )
        .expect("relative LOCO path should resolve from prediction-list directory");

        assert_eq!(source.chromosome_predictions("22").expect("chr22 predictions"), &[0.7]);
    }

    #[test]
    fn identity_loco_alignment_reuses_prediction_buffer() {
        let prediction_values: Arc<[f32]> = vec![1.0, 2.0, 3.0].into();

        let identity_aligned_values = super::align_prediction_values(&prediction_values, &[0, 1, 2]);
        let reordered_values = super::align_prediction_values(&prediction_values, &[2, 0]);

        assert!(Arc::ptr_eq(&prediction_values, &identity_aligned_values));
        assert!(!Arc::ptr_eq(&prediction_values, &reordered_values));
        assert_eq!(reordered_values.as_ref(), &[3.0, 1.0]);
    }

    #[test]
    fn multi_prediction_source_builds_trait_major_prediction_matrix() {
        let fixture = FixtureDirectory::new();
        let first_loco_path = fixture.write_file("first.loco", "FID_IID F1_I1 F2_I2\n22 1.0 2.0\n");
        let second_loco_path = fixture.write_file("second.loco", "FID_IID F1_I1 F2_I2\n22 3.0 4.0\n");
        let prediction_list_path = fixture.write_file(
            "pred.list",
            &format!("first {}\nsecond {}\n", first_loco_path.display(), second_loco_path.display()),
        );

        let source = MultiPredictionSource::load(
            &prediction_list_path,
            &strings(&["first", "second"]),
            &strings(&["F1", "F2"]),
            &strings(&["I1", "I2"]),
            SampleKeyMode::FidIid,
        )
        .expect("multi prediction source should load");

        let (trait_count, sample_count, prediction_values) =
            source.chromosome_prediction_matrix("chr22").expect("chr22 prediction matrix should be available");
        assert_eq!(trait_count, 2);
        assert_eq!(sample_count, 2);
        assert_eq!(prediction_values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn multi_prediction_source_reuses_cached_loco_file_for_repeated_paths() {
        let fixture = FixtureDirectory::new();
        let loco_path = fixture.write_file("shared.loco", "FID_IID F1_I1 F2_I2\n22 1.0 2.0\n");
        let entries = vec![
            PredictionListEntry { phenotype_name: "first".to_string(), loco_file_path: loco_path.clone() },
            PredictionListEntry { phenotype_name: "second".to_string(), loco_file_path: loco_path },
        ];
        let mut loco_prediction_cache = LocoPredictionCache::default();

        let first_source = MultiPredictionSource::load_from_entries_with_cache(
            &entries,
            &strings(&["first", "second"]),
            &strings(&["F1", "F2"]),
            &strings(&["I1", "I2"]),
            SampleKeyMode::FidIid,
            &mut loco_prediction_cache,
        )
        .expect("multi prediction source should load repeated LOCO path");

        let (_, _, prediction_values) =
            first_source.chromosome_prediction_matrix("22").expect("shared LOCO predictions should align");
        assert_eq!(prediction_values, vec![1.0, 2.0, 1.0, 2.0]);
        assert_eq!(loco_prediction_cache.cached_file_count(), 1);

        let second_source = MultiPredictionSource::load_from_entries_with_cache(
            &entries,
            &strings(&["second"]),
            &strings(&["F2"]),
            &strings(&["I2"]),
            SampleKeyMode::FidIid,
            &mut loco_prediction_cache,
        )
        .expect("second grouped-style load should reuse cached LOCO path");

        let (_, _, grouped_prediction_values) =
            second_source.chromosome_prediction_matrix("chr22").expect("subset predictions should align");
        assert_eq!(grouped_prediction_values, vec![2.0]);
        assert_eq!(loco_prediction_cache.cached_file_count(), 1);
    }

    #[test]
    fn multi_prediction_source_reports_iid_and_matrix_consistency_errors() {
        let fixture = FixtureDirectory::new();
        let loco_path = fixture.write_file("duplicate-iid.loco", "FID_IID F1_I1 F2_I1\n22 1.0 2.0\n");
        let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

        let duplicate_target_error = MultiPredictionSource::load(
            &prediction_list_path,
            &strings(&["trait"]),
            &strings(&["F1", "F2"]),
            &strings(&["I1", "I1"]),
            SampleKeyMode::Iid,
        )
        .expect_err("duplicate target IIDs should fail in IID mode");
        assert_matches!(duplicate_target_error, PredictionError::DuplicateTargetIid { .. });

        let duplicate_loco_error = MultiPredictionSource::load(
            &prediction_list_path,
            &strings(&["trait"]),
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::Iid,
        )
        .expect_err("duplicate LOCO IIDs should fail in IID mode");
        assert_matches!(duplicate_loco_error, PredictionError::DuplicateLocoIid { .. });

        let empty_target_error = MultiPredictionSource::load(
            &prediction_list_path,
            &strings(&["trait"]),
            &strings(&["F1"]),
            &strings(&[""]),
            SampleKeyMode::Iid,
        )
        .expect_err("empty target IID should fail in IID mode");
        assert_matches!(empty_target_error, PredictionError::EmptyTargetIid);

        let empty_loco_path = fixture.write_file("empty-iid.loco", "FID_IID F1_\n22 1.0\n");
        let empty_loco_list_path =
            fixture.write_file("empty-iid.list", &format!("trait {}\n", empty_loco_path.display()));
        let empty_loco_error = MultiPredictionSource::load(
            &empty_loco_list_path,
            &strings(&["trait"]),
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::Iid,
        )
        .expect_err("empty LOCO IID should fail in IID mode");
        assert_matches!(empty_loco_error, PredictionError::EmptyLocoIid);

        let source = MultiPredictionSource {
            phenotype_names: strings(&["first", "second"]),
            chromosome_predictions_by_trait: vec![
                HashMap::from([("22".to_string(), Arc::<[f32]>::from(vec![1.0, 2.0]))]),
                HashMap::from([("22".to_string(), Arc::<[f32]>::from(vec![3.0]))]),
            ],
        };
        let matrix_error =
            source.chromosome_prediction_matrix("chr22").expect_err("inconsistent trait sample counts should fail");
        assert_matches!(
            matrix_error,
            PredictionError::LocoPredictionCountMismatch { expected_count: 2, observed_count: 1, .. }
        );
    }

    #[test]
    fn prediction_source_reports_missing_phenotype_and_chromosome() {
        let fixture = FixtureDirectory::new();
        let loco_path = fixture.write_file("trait.loco", "FID_IID F1_I1\n22 1.0\n");
        let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

        let missing_phenotype_error = PredictionSource::load(
            &prediction_list_path,
            "missing",
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::FidIid,
        )
        .expect_err("missing phenotype should be rejected");
        assert_matches!(missing_phenotype_error, PredictionError::MissingPhenotype { .. });

        let source = PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::FidIid,
        )
        .expect("prediction source should load");
        let missing_chromosome_error =
            source.chromosome_predictions("chr1").expect_err("missing chromosome should be rejected");
        assert_matches!(missing_chromosome_error, PredictionError::MissingChromosome { .. });
    }

    #[test]
    fn prediction_source_rejects_malformed_prediction_list_and_loco_files() {
        let fixture = FixtureDirectory::new();
        let malformed_list_path = fixture.write_file("bad.list", "trait only extra\n");
        let malformed_list_error = PredictionSource::load(
            &malformed_list_path,
            "trait",
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::FidIid,
        )
        .expect_err("malformed prediction list should be rejected");
        assert_matches!(
            malformed_list_error,
            PredictionError::InvalidPredictionListLine { line_number: 1, field_count: 3 }
        );

        let duplicate_chromosome_loco_path = fixture.write_file("duplicate.loco", "FID_IID F1_I1\n22 1.0\nchr22 2.0\n");
        let duplicate_list_path =
            fixture.write_file("duplicate.list", &format!("trait {}\n", duplicate_chromosome_loco_path.display()));
        let duplicate_error = PredictionSource::load(
            &duplicate_list_path,
            "trait",
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::FidIid,
        )
        .expect_err("duplicate chromosome should be rejected");
        assert_matches!(duplicate_error, PredictionError::DuplicateChromosome { .. });

        let invalid_value_loco_path = fixture.write_file("invalid.loco", "FID_IID F1_I1\n22 nope\n");
        let invalid_value_list_path =
            fixture.write_file("invalid.list", &format!("trait {}\n", invalid_value_loco_path.display()));
        let invalid_value_error = PredictionSource::load(
            &invalid_value_list_path,
            "trait",
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::FidIid,
        )
        .expect_err("invalid prediction value should be rejected");
        assert_matches!(invalid_value_error, PredictionError::InvalidPredictionValue { .. });
    }

    #[test]
    fn prediction_source_validates_target_and_loco_sample_keys() {
        let fixture = FixtureDirectory::new();
        let loco_path = fixture.write_file("trait.loco", "FID_IID F1_I1 F1_I1\n22 1.0 2.0\n");
        let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

        let duplicate_target_error = PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F1", "F1"]),
            &strings(&["I1", "I1"]),
            SampleKeyMode::FidIid,
        )
        .expect_err("duplicate target sample key should be rejected");
        assert_matches!(duplicate_target_error, PredictionError::DuplicateTargetSampleKey { .. });

        let duplicate_loco_error = PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F1"]),
            &strings(&["I1"]),
            SampleKeyMode::FidIid,
        )
        .expect_err("duplicate LOCO sample key should be rejected");
        assert_matches!(duplicate_loco_error, PredictionError::DuplicateLocoSampleKey { .. });
    }

    #[test]
    fn prediction_source_reports_missing_target_samples() {
        let fixture = FixtureDirectory::new();
        let loco_path = fixture.write_file("trait.loco", "FID_IID F1_I1\n22 1.0\n");
        let prediction_list_path = fixture.write_file("pred.list", &format!("trait {}\n", loco_path.display()));

        let error = PredictionSource::load(
            &prediction_list_path,
            "trait",
            &strings(&["F2"]),
            &strings(&["I2"]),
            SampleKeyMode::FidIid,
        )
        .expect_err("missing target sample should be rejected");

        assert_matches!(error, PredictionError::MissingTargetSamples(_));
    }
}
