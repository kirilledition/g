//! Native pipeline orchestration building blocks.

#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeSet;
use std::path::Path;

use crate::genotype::bgen::{BgenError, BgenReaderCore};
use crate::genotype::common::{ChunkSpec, GenotypeError};
use crate::genotype::planner;

pub mod backend;

pub struct Regenie2RunEngineCore {
    reader: BgenReaderCore,
    chunk_size: usize,
    variant_limit: Option<usize>,
}

impl Regenie2RunEngineCore {
    pub fn open_bgen(
        bgen_path: &Path,
        chunk_size: usize,
        variant_limit: Option<usize>,
        trusted_no_missing_diploid: bool,
    ) -> Result<Self, BgenError> {
        let reader = BgenReaderCore::open(bgen_path, trusted_no_missing_diploid)?;
        Ok(Self { reader, chunk_size, variant_limit })
    }

    pub fn reader(&self) -> &BgenReaderCore {
        &self.reader
    }

    pub fn plan_chunks(&self, committed_chunk_identifiers: &BTreeSet<usize>) -> Result<Vec<ChunkSpec>, GenotypeError> {
        planner::plan_chromosome_homogeneous_chunks(
            self.reader.variant_count(),
            self.chunk_size,
            self.variant_limit,
            &self.reader.chromosome_boundary_indices(),
            committed_chunk_identifiers,
        )
    }
}
