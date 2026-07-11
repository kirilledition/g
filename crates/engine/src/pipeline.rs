//! Native pipeline orchestration building blocks.

use std::collections::BTreeSet;
use std::path::Path;

use g_genotype::{BgenError, BgenReaderCore, ChunkSpec, GenotypeError};

pub(crate) struct BgenRunEngine {
    pub(crate) reader: BgenReaderCore,
    chunk_size: usize,
    variant_limit: Option<usize>,
}

impl BgenRunEngine {
    /// Open a BGEN-backed run engine.
    ///
    /// # Errors
    ///
    /// Returns an error when the BGEN reader cannot open or validate the
    /// requested file.
    pub(crate) fn open(
        bgen_path: &Path,
        chunk_size: usize,
        variant_limit: Option<usize>,
        trusted_no_missing_diploid: bool,
    ) -> Result<Self, BgenError> {
        let reader = BgenReaderCore::open(bgen_path, trusted_no_missing_diploid)?;
        Ok(Self { reader, chunk_size, variant_limit })
    }

    /// Plan chromosome-homogeneous genotype chunks for this run.
    ///
    /// # Errors
    ///
    /// Returns an error when chunk sizing, variant limits, or committed chunk
    /// identifiers are inconsistent with the opened reader.
    pub(crate) fn plan_chunks(
        &self,
        committed_chunk_identifiers: &BTreeSet<usize>,
    ) -> Result<Vec<ChunkSpec>, GenotypeError> {
        self.reader.plan_chromosome_homogeneous_chunks(self.chunk_size, self.variant_limit, committed_chunk_identifiers)
    }
}
