use std::path::Path;

use crate::bgen::{BgenError, BgenReaderCore, ReaderProfileSnapshot};
use crate::common::{ChunkStats, GenotypeReaderCore, VariantMetadataColumns};
use crate::error::{GenotypeError, GenotypeResult};

pub struct BgenGenotypeSource {
    reader: BgenReaderCore,
}

impl BgenGenotypeSource {
    /// Opens a BGEN-backed genotype source.
    ///
    /// # Errors
    ///
    /// Returns an error when the BGEN file cannot be opened or its header is unsupported.
    pub fn open(bgen_path: &Path, trusted_no_missing_diploid: bool) -> Result<Self, BgenError> {
        BgenReaderCore::open(bgen_path, trusted_no_missing_diploid).map(|reader| Self { reader })
    }

    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.reader.sample_count()
    }

    #[must_use]
    pub fn variant_count(&self) -> usize {
        self.reader.variant_count()
    }

    #[must_use]
    pub fn contains_embedded_samples(&self) -> bool {
        self.reader.contains_embedded_samples()
    }

    #[must_use]
    pub fn sample_identifiers(&self) -> Vec<String> {
        self.reader.sample_identifiers()
    }

    #[must_use]
    pub fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.reader.chromosome_boundary_indices()
    }

    /// Prepares a reusable sample selection for subsequent chunk reads.
    ///
    /// # Errors
    ///
    /// Returns an error when any sample index is invalid or the selection state cannot be updated.
    pub fn prepare_sample_selection(&self, sample_indices: &[i64]) -> Result<(), BgenError> {
        self.reader.prepare_sample_selection(sample_indices)
    }

    /// Clears the prepared sample selection.
    ///
    /// # Errors
    ///
    /// Returns an error when the selection state cannot be updated.
    pub fn clear_prepared_sample_selection(&self) -> Result<(), BgenError> {
        self.reader.clear_prepared_sample_selection()
    }

    pub fn reset_profile(&self) {
        self.reader.reset_profile();
    }

    #[must_use]
    pub fn profile_snapshot(&self) -> ReaderProfileSnapshot {
        self.reader.profile_snapshot()
    }

    /// Validates that the trusted no-missing diploid shortcut is valid for this source.
    ///
    /// # Errors
    ///
    /// Returns an error when the BGEN records violate trusted-layout assumptions.
    pub fn validate_trusted_no_missing_diploid(&self) -> Result<(), BgenError> {
        self.reader.validate_trusted_no_missing_diploid()
    }

    /// Marks trusted no-missing diploid validation as completed.
    ///
    /// # Errors
    ///
    /// Returns an error when the source was not opened in trusted mode.
    pub fn mark_trusted_no_missing_diploid_validated(&self) -> Result<(), BgenError> {
        self.reader.mark_trusted_no_missing_diploid_validated()
    }

    /// Reads metadata columns for a variant range.
    ///
    /// # Errors
    ///
    /// Returns an error when the variant range is outside the source bounds.
    pub fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<VariantMetadataColumns, BgenError> {
        self.reader.variant_metadata_slice(variant_start, variant_stop)
    }

    /// Reads row-major preprocessed dosage values into a caller-owned buffer.
    ///
    /// # Errors
    ///
    /// Returns an error when the prepared sample selection is missing, the range is invalid, or decoding fails.
    pub fn read_preprocessed_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<ChunkStats, BgenError> {
        self.reader.read_preprocessed_dosage_f32_into_address_prepared(
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )
    }

    /// Reads variant-major preprocessed dosage values into a caller-owned buffer.
    ///
    /// # Errors
    ///
    /// Returns an error when the prepared sample selection is missing, the range is invalid, or decoding fails.
    pub fn read_preprocessed_variant_major_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<ChunkStats, BgenError> {
        self.reader.read_preprocessed_variant_major_dosage_f32_into_address_prepared(
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )
    }

    /// Reads variant-major packed 8-bit probability pairs into a caller-owned buffer.
    ///
    /// # Errors
    ///
    /// Returns an error when trusted packed output preconditions fail, the range is invalid, or decoding fails.
    pub fn read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<ChunkStats, BgenError> {
        self.reader.read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )
    }

    #[must_use]
    pub fn bgen_path(&self) -> &Path {
        self.reader.bgen_path()
    }
}

impl GenotypeReaderCore for BgenGenotypeSource {
    fn sample_count(&self) -> usize {
        BgenGenotypeSource::sample_count(self)
    }

    fn variant_count(&self) -> usize {
        BgenGenotypeSource::variant_count(self)
    }

    fn sample_identifiers(&self) -> Vec<String> {
        BgenGenotypeSource::sample_identifiers(self)
    }

    fn chromosome_boundary_indices(&self) -> Vec<usize> {
        BgenGenotypeSource::chromosome_boundary_indices(self)
    }

    fn prepare_sample_selection(&self, sample_indices: &[i64]) -> GenotypeResult<()> {
        BgenGenotypeSource::prepare_sample_selection(self, sample_indices).map_err(convert_bgen_error)
    }

    fn clear_prepared_sample_selection(&self) -> GenotypeResult<()> {
        BgenGenotypeSource::clear_prepared_sample_selection(self).map_err(convert_bgen_error)
    }

    fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> GenotypeResult<VariantMetadataColumns> {
        BgenGenotypeSource::variant_metadata_slice(self, variant_start, variant_stop).map_err(convert_bgen_error)
    }

    fn read_preprocessed_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> GenotypeResult<ChunkStats> {
        BgenGenotypeSource::read_preprocessed_dosage_f32_into_address_prepared(
            self,
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )
        .map_err(convert_bgen_error)
    }
}

fn convert_bgen_error(error: BgenError) -> GenotypeError {
    let message = match error {
        BgenError::InvalidFormat(message) | BgenError::UnsupportedFormat(message) | BgenError::Range(message) => {
            message
        }
        BgenError::Io(error) => format!("I/O error while reading BGEN file: {error}"),
    };
    GenotypeError::Reader(message)
}
