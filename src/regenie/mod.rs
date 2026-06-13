#![allow(clippy::missing_errors_doc)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::sample::{MultiAlignedSampleData, SampleKeyMode};

mod error;

pub use error::PredictionError;

#[cfg(test)]
mod tests;

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
    chromosome_prediction_matrix_cache: Mutex<HashMap<String, CachedChromosomePredictionMatrix>>,
}

#[derive(Debug, Clone)]
struct CachedChromosomePredictionMatrix {
    trait_count: usize,
    sample_count: usize,
    prediction_values: Arc<[f32]>,
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
        self.chromosome_predictions.get(&normalized_chromosome).map(std::convert::AsRef::as_ref).ok_or_else(|| {
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
        Ok(Self {
            phenotype_names: phenotype_names.to_vec(),
            chromosome_predictions_by_trait,
            chromosome_prediction_matrix_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn chromosome_prediction_matrix(&self, chromosome: &str) -> Result<(usize, usize, Vec<f32>), PredictionError> {
        let normalized_chromosome = normalize_chromosome(chromosome);
        if let Some(cached_matrix) = self.lock_chromosome_prediction_matrix_cache().get(&normalized_chromosome).cloned()
        {
            return Ok((
                cached_matrix.trait_count,
                cached_matrix.sample_count,
                cached_matrix.prediction_values.to_vec(),
            ));
        }
        let cached_matrix = self.build_chromosome_prediction_matrix(&normalized_chromosome, chromosome)?;
        self.lock_chromosome_prediction_matrix_cache().insert(normalized_chromosome, cached_matrix.clone());
        Ok((cached_matrix.trait_count, cached_matrix.sample_count, cached_matrix.prediction_values.to_vec()))
    }

    fn build_chromosome_prediction_matrix(
        &self,
        normalized_chromosome: &str,
        requested_chromosome: &str,
    ) -> Result<CachedChromosomePredictionMatrix, PredictionError> {
        let trait_count = self.phenotype_names.len();
        let mut prediction_matrix_values = Vec::new();
        let mut sample_count = None;
        for chromosome_predictions in &self.chromosome_predictions_by_trait {
            let Some(prediction_values) = chromosome_predictions.get(normalized_chromosome) else {
                let mut available_chromosomes: Vec<String> = chromosome_predictions.keys().cloned().collect();
                available_chromosomes.sort();
                return Err(PredictionError::MissingChromosome {
                    chromosome: requested_chromosome.to_string(),
                    normalized_chromosome: normalized_chromosome.to_string(),
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
        Ok(CachedChromosomePredictionMatrix {
            trait_count,
            sample_count: sample_count.unwrap_or(0),
            prediction_values: prediction_matrix_values.into(),
        })
    }

    fn lock_chromosome_prediction_matrix_cache(
        &self,
    ) -> std::sync::MutexGuard<'_, HashMap<String, CachedChromosomePredictionMatrix>> {
        self.chromosome_prediction_matrix_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    fn cached_chromosome_prediction_matrix_count(&self) -> usize {
        self.lock_chromosome_prediction_matrix_cache().len()
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
