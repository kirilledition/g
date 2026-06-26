//! Native pipeline orchestration building blocks.

use std::collections::BTreeSet;
use std::path::Path;

use g_genotype::bgen::{BgenError, BgenReaderCore};
use g_genotype::common::{ChunkSpec, GenotypeError};
use g_genotype::planner;

use crate::preflight::PreflightError;

pub struct Regenie2RunEngineCore {
    reader: BgenReaderCore,
    chunk_size: usize,
    variant_limit: Option<usize>,
}

impl Regenie2RunEngineCore {
    /// Open a BGEN-backed run engine.
    ///
    /// # Errors
    ///
    /// Returns an error when the BGEN reader cannot open or validate the
    /// requested file.
    pub fn open_bgen(
        bgen_path: &Path,
        chunk_size: usize,
        variant_limit: Option<usize>,
        trusted_no_missing_diploid: bool,
    ) -> Result<Self, BgenError> {
        let reader = BgenReaderCore::open(bgen_path, trusted_no_missing_diploid)?;
        Ok(Self { reader, chunk_size, variant_limit })
    }

    #[must_use]
    pub const fn reader(&self) -> &BgenReaderCore {
        &self.reader
    }

    /// Plan chromosome-homogeneous genotype chunks for this run.
    ///
    /// # Errors
    ///
    /// Returns an error when chunk sizing, variant limits, or committed chunk
    /// identifiers are inconsistent with the opened reader.
    pub fn plan_chunks(&self, committed_chunk_identifiers: &BTreeSet<usize>) -> Result<Vec<ChunkSpec>, GenotypeError> {
        planner::plan_chromosome_homogeneous_chunks(
            self.reader.variant_count(),
            self.chunk_size,
            self.variant_limit,
            &self.reader.chromosome_boundary_indices(),
            committed_chunk_identifiers,
        )
    }

    /// Resolve unique chromosome labels represented in the requested scan.
    ///
    /// # Errors
    ///
    /// Returns an error when chromosome boundary metadata cannot be read.
    pub fn required_chromosomes(&self, variant_limit: Option<usize>) -> Result<Vec<String>, PreflightError> {
        let variant_count = self.reader.variant_count();
        let scanned_variant_count = variant_limit.map_or(variant_count, |limit| limit.min(variant_count));
        if scanned_variant_count == 0 {
            return Ok(Vec::new());
        }
        let mut chromosome_labels = Vec::new();
        for chromosome_boundaries in self.reader.chromosome_boundary_indices().windows(2) {
            let chromosome_start_index = chromosome_boundaries[0];
            let chromosome_stop_index = chromosome_boundaries[1].min(scanned_variant_count);
            if chromosome_start_index >= chromosome_stop_index {
                continue;
            }
            let metadata = self
                .reader
                .variant_metadata_slice(chromosome_start_index, chromosome_start_index + 1)
                .map_err(|error| PreflightError::ChromosomeMetadata { message: error.to_string() })?;
            let Some(chromosome_label) = metadata.chromosome.into_iter().next() else {
                return Err(PreflightError::ChromosomeMetadata {
                    message: "Chromosome boundary metadata contained no chromosome label.".to_string(),
                });
            };
            chromosome_labels.push(chromosome_label);
        }
        Ok(chromosome_labels)
    }
}
