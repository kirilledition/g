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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Float32Array, Int64Array, StringArray};
    use g_genotype_contracts::{
        ChunkOutputStatistics, NullableFloat32Column, VariantMetadataColumns, VariantMetadataStore,
    };

    use super::{NativeChunkHandle, NativeVariantMetadataHandle};

    fn metadata_columns(row_count: usize) -> VariantMetadataColumns {
        let dictionary: Box<[Arc<str>]> = ["22", "A", "C", "G"].map(Arc::<str>::from).into();
        let identifiers = (0..row_count).map(|index| format!("variant-{index}-β")).collect::<Vec<_>>();
        let mut identifier_text = String::new();
        let mut identifier_offsets = vec![0_u32];
        for identifier in &identifiers {
            identifier_text.push_str(identifier);
            identifier_offsets.push(u32::try_from(identifier_text.len()).expect("test identifiers fit uint32"));
        }
        let store = Arc::new(
            VariantMetadataStore::from_parts(
                dictionary,
                vec![0_u32; row_count].into_boxed_slice(),
                identifier_text.into_boxed_str(),
                identifier_offsets.into_boxed_slice(),
                (0..row_count)
                    .map(|index| 100_i64 + i64::try_from(index).expect("test position fits int64"))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                vec![1_u32; row_count].into_boxed_slice(),
                vec![2_u32; row_count].into_boxed_slice(),
            )
            .expect("test metadata store should satisfy its invariants"),
        );
        VariantMetadataColumns::new(store, 0..row_count).expect("test metadata range should be valid")
    }

    fn statistics(
        allele_frequency_count: usize,
        info_count: usize,
        observation_count: usize,
        validity_bytes: Vec<u8>,
    ) -> ChunkOutputStatistics {
        ChunkOutputStatistics {
            allele_one_frequency: vec![0.25; allele_frequency_count],
            observation_count: vec![12; observation_count],
            info_score: NullableFloat32Column { values: vec![0.9; info_count], validity_bytes },
        }
    }

    #[test]
    fn chunk_arrays_preserve_utf8_metadata_and_nullable_info() {
        let metadata_handle = NativeVariantMetadataHandle::try_new(&metadata_columns(3)).expect("metadata is valid");
        let chunk_handle = NativeChunkHandle::try_new(
            metadata_handle,
            ChunkOutputStatistics {
                allele_one_frequency: vec![0.1, 0.2, 0.3],
                observation_count: vec![10, 11, 12],
                info_score: NullableFloat32Column {
                    values: vec![0.8, f32::NAN, 0.95],
                    validity_bytes: vec![0b0000_0101],
                },
            },
            7,
        )
        .expect("chunk is valid");

        assert_eq!(chunk_handle.row_count(), 3);
        assert_eq!(chunk_handle.variant_stop_index().expect("stop index fits"), 10);
        let writer_arrays = chunk_handle.writer_arrays.as_ref();
        let metadata_arrays = writer_arrays.metadata.arrays();
        let chromosomes = metadata_arrays.chromosome.as_any().downcast_ref::<StringArray>().expect("CHROM is Utf8");
        let positions = metadata_arrays.position.as_any().downcast_ref::<Int64Array>().expect("GENPOS is Int64");
        let identifiers =
            metadata_arrays.variant_identifier.as_any().downcast_ref::<StringArray>().expect("ID is Utf8");
        assert_eq!(chromosomes.iter().collect::<Vec<_>>(), vec![Some("22"), Some("22"), Some("22")]);
        assert_eq!(positions.values(), &[100, 101, 102]);
        assert_eq!(identifiers.value(2), "variant-2-β");

        let info_scores = writer_arrays.info_score.as_any().downcast_ref::<Float32Array>().expect("INFO is Float32");
        assert!(!info_scores.is_null(0));
        assert!(info_scores.is_null(1));
        assert!(!info_scores.is_null(2));
        assert!((info_scores.value(0) - 0.8).abs() < f32::EPSILON);
        assert!((info_scores.value(2) - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn chunk_rejects_every_statistics_row_count_mismatch() {
        let cases = [(2, 3, 3, "allele one frequency"), (3, 2, 3, "INFO score"), (3, 3, 2, "observation count")];
        for (allele_frequency_count, info_count, observation_count, expected_column) in cases {
            let metadata_handle =
                NativeVariantMetadataHandle::try_new(&metadata_columns(3)).expect("metadata is valid");
            let error = NativeChunkHandle::try_new(
                metadata_handle,
                statistics(allele_frequency_count, info_count, observation_count, vec![0b0000_0111]),
                0,
            )
            .err()
            .expect("mismatched column must fail");
            assert!(error.to_string().contains(expected_column));
        }
    }

    #[test]
    fn chunk_rejects_malformed_info_validity_bitmap_including_tail() {
        for (row_count, validity_bytes) in [(1, Vec::new()), (8, vec![u8::MAX, 0]), (9, vec![u8::MAX])] {
            let metadata_handle =
                NativeVariantMetadataHandle::try_new(&metadata_columns(row_count)).expect("metadata is valid");
            let error = NativeChunkHandle::try_new(
                metadata_handle,
                statistics(row_count, row_count, row_count, validity_bytes),
                0,
            )
            .err()
            .expect("malformed INFO bitmap must fail");
            assert!(error.to_string().contains("INFO validity bitmap"));
        }
    }

    #[test]
    fn chunk_rejects_variant_stop_overflow() {
        let metadata_handle = NativeVariantMetadataHandle::try_new(&metadata_columns(1)).expect("metadata is valid");
        let chunk_handle = NativeChunkHandle::try_new(metadata_handle, statistics(1, 1, 1, vec![1]), i64::MAX)
            .expect("chunk construction does not require the stop index");
        let error = chunk_handle.variant_stop_index().expect_err("stop index overflow must fail");
        assert!(error.to_string().contains("variant stop index"));
    }
}
