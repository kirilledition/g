//! Engine-neutral contracts for native association backends.

#![allow(clippy::missing_errors_doc)]

use crate::genotype::common::{ChunkSpec, ChunkStats, VariantMetadataColumns};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DosageMatrixLayout {
    SampleMajor,
    VariantMajor,
}

#[derive(Clone, Copy, Debug)]
pub struct AssociationChunk<'a> {
    pub chunk_spec: &'a ChunkSpec,
    pub metadata: &'a VariantMetadataColumns,
    pub stats: &'a ChunkStats,
    pub dosage_values: &'a [f32],
    pub selected_sample_count: usize,
    pub layout: DosageMatrixLayout,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssociationResults {
    pub beta: Vec<f32>,
    pub standard_error: Vec<f32>,
    pub chi_squared: Vec<f32>,
    pub log10_p_value: Vec<f32>,
    pub extra_code: Option<Vec<i32>>,
}

pub trait AssociationBackend {
    type Error;

    fn process_chunk(&mut self, chunk: AssociationChunk<'_>) -> Result<AssociationResults, Self::Error>;
}
