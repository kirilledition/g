use std::sync::{Arc, OnceLock};

use arrow::array::{ArrayRef, Float32Array, Int32Array, Int64Array, StringArray};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};

use crate::error::OutputError;

use g_genotype_contracts::{ChunkOutputStatistics, NullableFloat32Column, VariantMetadataColumns};

pub(crate) struct NativeChunkWriterArrays {
    pub(crate) metadata: NativeVariantMetadataHandle,
    pub(crate) allele_one_frequency: ArrayRef,
    pub(crate) info_score: ArrayRef,
    pub(crate) observation_count: ArrayRef,
}

pub struct NativeVariantMetadataHandle {
    source: NativeVariantMetadataSource,
}

pub(crate) struct NativeVariantMetadataArrays {
    pub(crate) chromosome: ArrayRef,
    pub(crate) position: ArrayRef,
    pub(crate) variant_identifier: ArrayRef,
    pub(crate) allele_two: ArrayRef,
    pub(crate) allele_one: ArrayRef,
}

struct NativeVariantMetadataSource {
    metadata: VariantMetadataColumns,
    arrays: OnceLock<NativeVariantMetadataArrays>,
}

impl std::fmt::Debug for NativeVariantMetadataHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("NativeVariantMetadataHandle").field("row_count", &self.row_count()).finish()
    }
}

impl NativeVariantMetadataHandle {
    /// Retain one metadata slice for asynchronous output.
    ///
    /// # Errors
    ///
    /// Returns an error when a string column exceeds Arrow's `Utf8` offset
    /// width. Rejecting it here keeps the infallible lazy array construction
    /// from panicking on the output worker.
    pub fn try_new(metadata: &VariantMetadataColumns) -> Result<Self, OutputError> {
        validate_utf8_column_width("CHROM", metadata.chromosomes())?;
        validate_utf8_column_width("ID", metadata.variant_identifiers())?;
        validate_utf8_column_width("ALLELE0", metadata.allele_ones())?;
        validate_utf8_column_width("ALLELE1", metadata.allele_twos())?;
        Ok(Self { source: NativeVariantMetadataSource { metadata: metadata.clone(), arrays: OnceLock::new() } })
    }

    #[must_use]
    pub fn row_count(&self) -> usize {
        self.source.metadata.len()
    }

    pub(crate) fn arrays(&self) -> &NativeVariantMetadataArrays {
        self.source.arrays.get_or_init(|| {
            let metadata = &self.source.metadata;
            NativeVariantMetadataArrays {
                chromosome: Arc::new(StringArray::from_iter_values(metadata.chromosomes())),
                position: Arc::new(Int64Array::from_iter_values(metadata.position().iter().copied())),
                variant_identifier: Arc::new(StringArray::from_iter_values(metadata.variant_identifiers())),
                allele_two: Arc::new(StringArray::from_iter_values(metadata.allele_twos())),
                allele_one: Arc::new(StringArray::from_iter_values(metadata.allele_ones())),
            }
        })
    }
}

fn validate_utf8_column_width<'value>(
    column_name: &str,
    mut values: impl Iterator<Item = &'value str>,
) -> Result<(), OutputError> {
    let maximum_byte_count = usize::try_from(i32::MAX).expect("i32::MAX fits in usize");
    let byte_count = values.try_fold(0_usize, |total, value| total.checked_add(value.len())).ok_or_else(|| {
        OutputError::InvalidInput(format!("{column_name} UTF-8 byte count overflows the host index width."))
    })?;
    if byte_count > maximum_byte_count {
        return Err(OutputError::InvalidInput(format!(
            "{column_name} contains {byte_count} UTF-8 bytes, exceeding Arrow's {maximum_byte_count}-byte Utf8 offset limit."
        )));
    }
    Ok(())
}

impl NativeChunkWriterArrays {
    fn try_from_chunk_sources(
        metadata: NativeVariantMetadataHandle,
        statistics: ChunkOutputStatistics,
    ) -> Result<Self, OutputError> {
        let info_score = build_nullable_float32_array(statistics.info_score)?;
        Ok(Self {
            metadata,
            allele_one_frequency: Arc::new(Float32Array::from(statistics.allele_one_frequency)),
            info_score,
            observation_count: Arc::new(Int32Array::from(statistics.observation_count)),
        })
    }
}

fn build_nullable_float32_array(column: NullableFloat32Column) -> Result<ArrayRef, OutputError> {
    let value_count = column.values.len();
    let expected_validity_byte_count = value_count.div_ceil(8);
    if column.validity_bytes.len() != expected_validity_byte_count {
        return Err(OutputError::InvalidInput(format!(
            "INFO validity bitmap contains {} bytes, expected {expected_validity_byte_count} for {value_count} values.",
            column.validity_bytes.len(),
        )));
    }
    let validity = NullBuffer::new(BooleanBuffer::new(Buffer::from(column.validity_bytes), 0, value_count));
    Ok(Arc::new(Float32Array::new(ScalarBuffer::from(column.values), Some(validity))))
}

#[derive(Clone)]
pub struct NativeChunkHandle {
    pub(crate) chunk_identifier: i64,
    pub(crate) writer_arrays: Arc<NativeChunkWriterArrays>,
}

impl NativeChunkHandle {
    /// Build one output chunk while validating its row counts and nullable columns.
    ///
    /// # Errors
    ///
    /// Returns an error when a statistic column has the wrong row count or a
    /// packed validity bitmap does not match its value column.
    pub fn try_new(
        metadata: NativeVariantMetadataHandle,
        statistics: ChunkOutputStatistics,
        chunk_identifier: i64,
    ) -> Result<Self, OutputError> {
        let row_count = metadata.row_count();
        validate_row_count("allele one frequency", statistics.allele_one_frequency.len(), row_count)?;
        validate_row_count("INFO score", statistics.info_score.values.len(), row_count)?;
        validate_row_count("observation count", statistics.observation_count.len(), row_count)?;
        Ok(Self {
            chunk_identifier,
            writer_arrays: Arc::new(NativeChunkWriterArrays::try_from_chunk_sources(metadata, statistics)?),
        })
    }

    #[must_use]
    pub fn row_count(&self) -> usize {
        self.writer_arrays.metadata.row_count()
    }

    pub(crate) fn variant_stop_index(&self) -> Result<i64, OutputError> {
        let row_count = i64::try_from(self.row_count()).map_err(|_| {
            OutputError::InvalidInput("Rust output writer row count does not fit into int64.".to_string())
        })?;
        self.chunk_identifier.checked_add(row_count).ok_or_else(|| {
            OutputError::InvalidInput("Rust output writer variant stop index does not fit into int64.".to_string())
        })
    }
}

fn validate_row_count(column_name: &str, observed: usize, expected: usize) -> Result<(), OutputError> {
    if observed == expected {
        return Ok(());
    }
    Err(OutputError::InvalidInput(format!("Output {column_name} contains {observed} rows, expected {expected}.")))
}
