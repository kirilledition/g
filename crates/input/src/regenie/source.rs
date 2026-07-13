//! Deferred prediction source indexing and chromosome matrix assembly.

use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use super::alignment::LocoSampleAlignment;
use super::cache::{LocoAlignmentCache, LocoFileIndexCache};
use super::error::PredictionError;
use super::loco::{LocoFileIndex, read_loco_chromosome_predictions_into};
use super::{PredictionLocoPath, normalize_chromosome};

#[derive(Debug)]
pub(crate) struct PredictionSource {
    trait_sources: Vec<PredictionTraitSource>,
    trait_count: usize,
    sample_count: usize,
    matrix_value_count: usize,
    planned_chromosomes: HashMap<String, PlannedChromosome>,
}

#[derive(Debug)]
struct PredictionTraitSource {
    file_index: Arc<LocoFileIndex>,
    sample_alignment: Arc<LocoSampleAlignment>,
}

#[derive(Debug)]
struct PlannedChromosome {
    remaining_uses: usize,
    retained_matrix: Option<ChromosomePredictionMatrix>,
}

pub(crate) struct PredictionSourceLoader<'paths> {
    prediction_loco_paths: &'paths [PredictionLocoPath],
    file_index_cache: LocoFileIndexCache,
}

#[derive(Clone, Debug)]
pub struct ChromosomePredictionMatrix {
    pub trait_count: usize,
    pub sample_count: usize,
    pub prediction_values: Vec<f32>,
}

impl<'paths> PredictionSourceLoader<'paths> {
    pub(crate) fn new(prediction_loco_paths: &'paths [PredictionLocoPath]) -> Self {
        Self { prediction_loco_paths, file_index_cache: LocoFileIndexCache::default() }
    }

    pub(crate) fn load(
        &mut self,
        phenotype_indices: &[usize],
        target_family_identifiers: &[String],
        target_individual_identifiers: &[String],
        target_sample_indices: &[usize],
    ) -> Result<PredictionSource, PredictionError> {
        let trait_count = phenotype_indices.len();
        let sample_count = target_sample_indices.len();
        let matrix_value_count = trait_count
            .checked_mul(sample_count)
            .ok_or(PredictionError::PredictionMatrixShapeOverflow { trait_count, sample_count })?;
        let mut alignment_cache = LocoAlignmentCache::default();
        let mut trait_sources = Vec::with_capacity(trait_count);
        for phenotype_index in phenotype_indices {
            let resolved_path = &self.prediction_loco_paths[*phenotype_index];
            let indexed_file = self.file_index_cache.index(&resolved_path.loco_file_path)?;
            let sample_alignment = alignment_cache.alignment(
                &indexed_file.file_index,
                &indexed_file.sample_index,
                target_family_identifiers,
                target_individual_identifiers,
                target_sample_indices,
            )?;
            trait_sources.push(PredictionTraitSource { file_index: indexed_file.file_index, sample_alignment });
        }
        Ok(PredictionSource {
            trait_sources,
            trait_count,
            sample_count,
            matrix_value_count,
            planned_chromosomes: HashMap::new(),
        })
    }
}

impl PredictionSource {
    pub(crate) fn alignment_source_digest(&self) -> [u8; 32] {
        let mut fingerprint_hash = Sha256::new();
        fingerprint_hash.update(b"prediction-alignment-source-v1");
        update_usize_fingerprint(&mut fingerprint_hash, self.trait_count);
        update_usize_fingerprint(&mut fingerprint_hash, self.sample_count);
        for trait_source in &self.trait_sources {
            fingerprint_hash.update(trait_source.file_index.source_digest);
            match trait_source.sample_alignment.as_ref() {
                LocoSampleAlignment::Identity => fingerprint_hash.update(b"identity"),
                LocoSampleAlignment::Indices(alignment_indices) => {
                    fingerprint_hash.update(b"indices");
                    update_usize_fingerprint(&mut fingerprint_hash, alignment_indices.len());
                    for sample_index in alignment_indices {
                        update_usize_fingerprint(&mut fingerprint_hash, *sample_index);
                    }
                }
            }
        }
        fingerprint_hash.finalize().into()
    }

    pub(crate) fn plan_uses(&mut self, chromosome_blocks: &[Arc<str>]) -> Result<(), PredictionError> {
        let mut planned_chromosomes = HashMap::new();
        for chromosome in chromosome_blocks {
            let normalized_chromosome = normalize_chromosome(chromosome);
            let plan = planned_chromosomes
                .entry(normalized_chromosome)
                .or_insert(PlannedChromosome { remaining_uses: 0, retained_matrix: None });
            plan.remaining_uses += 1;
        }
        for chromosome in planned_chromosomes.keys() {
            for trait_source in &self.trait_sources {
                if !trait_source.file_index.chromosome_rows.contains_key(chromosome) {
                    return Err(PredictionError::MissingChromosome {
                        chromosome: chromosome.clone(),
                        normalized_chromosome: chromosome.clone(),
                        available_chromosomes: sorted_chromosomes(&trait_source.file_index.chromosome_rows),
                    });
                }
            }
        }
        self.planned_chromosomes = planned_chromosomes;
        Ok(())
    }

    pub(crate) fn take_chromosome_prediction_matrix(
        &mut self,
        chromosome: &str,
    ) -> Result<ChromosomePredictionMatrix, PredictionError> {
        let normalized_chromosome = normalize_chromosome(chromosome);
        let Some(mut planned_chromosome) = self.planned_chromosomes.remove(&normalized_chromosome) else {
            return Err(PredictionError::MissingChromosome {
                chromosome: chromosome.to_string(),
                normalized_chromosome,
                available_chromosomes: sorted_chromosomes(&self.planned_chromosomes),
            });
        };
        if planned_chromosome.remaining_uses == 1 {
            return match planned_chromosome.retained_matrix {
                Some(matrix) => Ok(matrix),
                None => self.materialize_chromosome(&normalized_chromosome),
            };
        }

        if planned_chromosome.retained_matrix.is_none() {
            planned_chromosome.retained_matrix = Some(self.materialize_chromosome(&normalized_chromosome)?);
        }
        planned_chromosome.remaining_uses -= 1;
        let matrix = planned_chromosome
            .retained_matrix
            .as_ref()
            .expect("repeated chromosome materialization is retained immediately above")
            .clone();
        self.planned_chromosomes.insert(normalized_chromosome, planned_chromosome);
        Ok(matrix)
    }

    fn materialize_chromosome(&self, chromosome: &str) -> Result<ChromosomePredictionMatrix, PredictionError> {
        let mut prediction_values = Vec::with_capacity(self.matrix_value_count);
        let mut unaligned_prediction_values = Vec::new();
        for trait_source in &self.trait_sources {
            match trait_source.sample_alignment.as_ref() {
                LocoSampleAlignment::Identity => {
                    read_loco_chromosome_predictions_into(
                        &trait_source.file_index,
                        chromosome,
                        &mut prediction_values,
                    )?;
                }
                LocoSampleAlignment::Indices(alignment_indices) => {
                    unaligned_prediction_values.clear();
                    unaligned_prediction_values.reserve(trait_source.file_index.sample_count);
                    read_loco_chromosome_predictions_into(
                        &trait_source.file_index,
                        chromosome,
                        &mut unaligned_prediction_values,
                    )?;
                    prediction_values.extend(
                        alignment_indices.iter().map(|sample_index| unaligned_prediction_values[*sample_index]),
                    );
                }
            }
        }
        debug_assert_eq!(prediction_values.len(), self.matrix_value_count);
        Ok(ChromosomePredictionMatrix {
            trait_count: self.trait_count,
            sample_count: self.sample_count,
            prediction_values,
        })
    }
}

fn update_usize_fingerprint(fingerprint_hash: &mut Sha256, value: usize) {
    fingerprint_hash
        .update(u64::try_from(value).expect("supported Rust targets represent usize within u64").to_le_bytes());
}

fn sorted_chromosomes<Values>(predictions: &HashMap<String, Values>) -> Vec<String> {
    let mut chromosomes = predictions.keys().cloned().collect::<Vec<_>>();
    chromosomes.sort_unstable();
    chromosomes
}
