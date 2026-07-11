//! Genotype reader and preprocessing contracts.

#![allow(clippy::missing_errors_doc)]

use g_genotype_contracts::ChunkOutputStatistics;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DosageSummary {
    pub(crate) dosage_sum: f32,
    pub(crate) dosage_square_sum: f32,
    pub(crate) observation_count: i32,
    pub(crate) zero_count: i32,
    pub(crate) homozygous_alternate_count: i32,
}

#[derive(Debug, Eq, PartialEq)]
pub struct ChunkSpec {
    pub variant_start_index: usize,
    pub variant_stop_index: usize,
}

#[derive(Debug, PartialEq)]
pub struct ChunkStats {
    pub output: ChunkOutputStatistics,
    pub compute: ChunkComputeStatistics,
}

#[derive(Debug, PartialEq)]
pub struct ChunkComputeStatistics {
    pub genotype_mean: Vec<f32>,
    pub imputed_dosage_square_sum: Option<Vec<f32>>,
    pub sparse_candidate_mask: Option<Vec<bool>>,
}

/// Per-run policy for statistics retained after genotype decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkStatisticsPolicy {
    pub retain_imputed_dosage_square_sum: bool,
    pub collect_sparse_candidate_mask: bool,
}
