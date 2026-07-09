//! Prediction source loading and chromosome matrix assembly.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::sample::{MultiAlignedSampleData, SampleKeyMode};

use super::alignment::{
    align_prediction_values, validate_target_sample_keys, validate_unique_target_individual_identifiers,
};
use super::cache::{LocoAlignmentCache, LocoPredictionCache};
use super::chromosome::normalize_chromosome;
use super::error::PredictionError;
use super::list::{PredictionListEntry, find_prediction_list_entry, parse_prediction_list_file};

#[derive(Debug)]
pub struct PredictionSource {
    chromosome_predictions: HashMap<String, Arc<[f32]>>,
}

#[derive(Debug)]
pub struct MultiPredictionSource {
    pub(super) phenotype_names: Vec<String>,
    pub(super) chromosome_predictions_by_trait: Vec<HashMap<String, Arc<[f32]>>>,
    pub(super) chromosome_prediction_matrix_cache: Mutex<HashMap<String, CachedChromosomePredictionMatrix>>,
}

#[derive(Debug, Clone)]
pub struct ChromosomePredictionMatrix {
    pub trait_count: usize,
    pub sample_count: usize,
    pub prediction_values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct CachedChromosomePredictionMatrix {
    trait_count: usize,
    sample_count: usize,
    prediction_values: Arc<[f32]>,
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

    pub(super) fn load_from_entries_with_cache(
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

    pub fn chromosome_prediction_matrix(
        &self,
        chromosome: &str,
    ) -> Result<ChromosomePredictionMatrix, PredictionError> {
        let normalized_chromosome = normalize_chromosome(chromosome);
        {
            let chromosome_prediction_matrix_cache = self.lock_chromosome_prediction_matrix_cache();
            if let Some(cached_matrix) = chromosome_prediction_matrix_cache.get(&normalized_chromosome) {
                return Ok(chromosome_prediction_matrix_from_cached(cached_matrix));
            }
        }
        let cached_matrix = self.build_chromosome_prediction_matrix(&normalized_chromosome, chromosome)?;
        self.lock_chromosome_prediction_matrix_cache().insert(normalized_chromosome, cached_matrix.clone());
        Ok(chromosome_prediction_matrix_from_cached(&cached_matrix))
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
}

fn chromosome_prediction_matrix_from_cached(
    cached_matrix: &CachedChromosomePredictionMatrix,
) -> ChromosomePredictionMatrix {
    ChromosomePredictionMatrix {
        trait_count: cached_matrix.trait_count,
        sample_count: cached_matrix.sample_count,
        prediction_values: cached_matrix.prediction_values.to_vec(),
    }
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
    Ok(loco_predictions
        .chromosome_predictions
        .iter()
        .map(|(chromosome, prediction_values)| {
            (chromosome.clone(), align_prediction_values(prediction_values, alignment_indices))
        })
        .collect())
}
