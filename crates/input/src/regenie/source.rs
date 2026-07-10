//! Prediction source loading and chromosome matrix assembly.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use g_plan::SampleKeyMode;

use super::alignment::{
    align_prediction_values, validate_target_sample_keys, validate_unique_target_individual_identifiers,
};
use super::cache::{LocoAlignmentCache, LocoPredictionCache};
use super::error::PredictionError;
use super::list::{PredictionListEntry, find_prediction_list_entry, parse_prediction_list_file};
use super::normalize_chromosome;

#[derive(Debug)]
pub(crate) struct PredictionSource {
    trait_count: usize,
    sample_count: usize,
    pub(super) chromosome_predictions_by_trait: Vec<HashMap<String, Arc<[f32]>>>,
    pub(super) chromosome_prediction_matrix_cache: Mutex<HashMap<String, Arc<ChromosomePredictionMatrix>>>,
}

pub(crate) struct PredictionSourceLoader {
    entries: Vec<PredictionListEntry>,
    loco_prediction_cache: LocoPredictionCache,
}

#[derive(Debug, Clone)]
pub struct ChromosomePredictionMatrix {
    pub trait_count: usize,
    pub sample_count: usize,
    pub prediction_values: Arc<[f32]>,
}

impl PredictionSourceLoader {
    pub(crate) fn new(prediction_list_path: &Path) -> Result<Self, PredictionError> {
        Ok(Self {
            entries: parse_prediction_list_file(prediction_list_path)?,
            loco_prediction_cache: LocoPredictionCache::default(),
        })
    }

    pub(crate) fn load(
        &mut self,
        phenotype_names: &[String],
        target_family_identifiers: &[&str],
        target_individual_identifiers: &[&str],
        sample_key_mode: g_plan::SampleKeyMode,
    ) -> Result<PredictionSource, PredictionError> {
        PredictionSource::load(
            &self.entries,
            phenotype_names,
            target_family_identifiers,
            target_individual_identifiers,
            sample_key_mode,
            &mut self.loco_prediction_cache,
        )
    }
}

impl PredictionSource {
    fn load(
        entries: &[PredictionListEntry],
        phenotype_names: &[String],
        target_family_identifiers: &[&str],
        target_individual_identifiers: &[&str],
        sample_key_mode: g_plan::SampleKeyMode,
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
            trait_count: phenotype_names.len(),
            sample_count: target_individual_identifiers.len(),
            chromosome_predictions_by_trait,
            chromosome_prediction_matrix_cache: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn chromosome_prediction_matrix(
        &self,
        chromosome: &str,
    ) -> Result<Arc<ChromosomePredictionMatrix>, PredictionError> {
        let normalized_chromosome = normalize_chromosome(chromosome);
        {
            let chromosome_prediction_matrix_cache = self.lock_chromosome_prediction_matrix_cache();
            if let Some(cached_matrix) = chromosome_prediction_matrix_cache.get(&normalized_chromosome) {
                return Ok(Arc::clone(cached_matrix));
            }
        }
        let matrix = Arc::new(self.build_chromosome_prediction_matrix(&normalized_chromosome, chromosome)?);
        self.lock_chromosome_prediction_matrix_cache().insert(normalized_chromosome, Arc::clone(&matrix));
        Ok(matrix)
    }

    fn build_chromosome_prediction_matrix(
        &self,
        normalized_chromosome: &str,
        requested_chromosome: &str,
    ) -> Result<ChromosomePredictionMatrix, PredictionError> {
        let trait_count = self.trait_count;
        let mut prediction_matrix_values = Vec::new();
        prediction_matrix_values.reserve_exact(trait_count * self.sample_count);
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
            if prediction_values.len() != self.sample_count {
                return Err(PredictionError::LocoPredictionCountMismatch {
                    line_number: 0,
                    expected_count: self.sample_count,
                    observed_count: prediction_values.len(),
                });
            }
            prediction_matrix_values.extend_from_slice(prediction_values.as_ref());
        }
        Ok(ChromosomePredictionMatrix {
            trait_count,
            sample_count: self.sample_count,
            prediction_values: prediction_matrix_values.into(),
        })
    }

    fn lock_chromosome_prediction_matrix_cache(
        &self,
    ) -> std::sync::MutexGuard<'_, HashMap<String, Arc<ChromosomePredictionMatrix>>> {
        self.chromosome_prediction_matrix_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn load_aligned_chromosome_predictions(
    entry: &PredictionListEntry,
    loco_prediction_cache: &mut LocoPredictionCache,
    loco_alignment_cache: &mut LocoAlignmentCache,
    target_family_identifiers: &[&str],
    target_individual_identifiers: &[&str],
    sample_key_mode: g_plan::SampleKeyMode,
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
