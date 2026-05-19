//! Shared contracts for native genotype readers.

#![allow(clippy::missing_errors_doc)]

use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkSpec {
    pub variant_start_index: usize,
    pub variant_stop_index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkStats {
    pub allele_one_frequency: Vec<f32>,
    pub observation_count: Vec<i32>,
    pub has_missing_values: bool,
    pub dosage_sum: Vec<f32>,
    pub dosage_square_sum: Vec<f32>,
    pub imputed_dosage_square_sum: Vec<f32>,
    pub dosage_variance_numerator: Vec<f32>,
    pub info_score: Vec<Option<f32>>,
    pub allele_count: Vec<f32>,
    pub minor_allele_count: Vec<f32>,
    pub zero_count: Vec<i32>,
    pub nonzero_count: Vec<i32>,
    pub homozygous_reference_count: Vec<i32>,
    pub heterozygous_count: Vec<i32>,
    pub homozygous_alternate_count: Vec<i32>,
    pub is_sparse_candidate: Vec<bool>,
    pub is_rare_sparse_firth_candidate: Vec<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariantMetadataColumns {
    pub chromosome: Vec<String>,
    pub variant_identifier: Vec<String>,
    pub position: Vec<i64>,
    pub allele_one: Vec<String>,
    pub allele_two: Vec<String>,
}

#[derive(Error, Debug)]
pub enum GenotypeError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Reader(String),
}

pub trait GenotypeReaderCore {
    fn sample_count(&self) -> usize;
    fn variant_count(&self) -> usize;
    fn sample_identifiers(&self) -> Vec<String>;
    fn chromosome_boundary_indices(&self) -> Vec<usize>;
    fn prepare_sample_selection(&self, sample_indices: &[i64]) -> Result<(), GenotypeError>;
    fn clear_prepared_sample_selection(&self) -> Result<(), GenotypeError>;
    fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<VariantMetadataColumns, GenotypeError>;
    fn read_preprocessed_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<ChunkStats, GenotypeError>;
}
