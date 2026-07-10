//! Shared contracts for native genotype readers.

#![allow(clippy::missing_errors_doc)]

use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkSpec {
    pub variant_start_index: usize,
    pub variant_stop_index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkStats {
    pub allele_one_frequency: Vec<f32>,
    pub observation_count: Vec<i32>,
    pub dosage_sum: Arc<[f32]>,
    pub imputed_dosage_square_sum: Vec<f32>,
    pub info_score: Vec<Option<f32>>,
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
