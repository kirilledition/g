use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Int32Array, Int64Array, StringArray};

use crate::error::OutputError;

use g_genotype::{ChunkStats, VariantMetadataColumns};

pub(crate) struct NativeChunkWriterArrays {
    pub(crate) chromosome: ArrayRef,
    pub(crate) position: ArrayRef,
    pub(crate) variant_identifier: ArrayRef,
    pub(crate) allele_two: ArrayRef,
    pub(crate) allele_one: ArrayRef,
    pub(crate) allele_one_frequency: ArrayRef,
    pub(crate) info_score: ArrayRef,
    pub(crate) observation_count: ArrayRef,
}

impl NativeChunkWriterArrays {
    pub(crate) fn from_chunk_sources(metadata: VariantMetadataColumns, statistics: ChunkStats) -> Self {
        Self {
            chromosome: Arc::new(StringArray::from(metadata.chromosome)),
            position: Arc::new(Int64Array::from(metadata.position)),
            variant_identifier: Arc::new(StringArray::from(metadata.variant_identifier)),
            allele_two: Arc::new(StringArray::from(metadata.allele_two)),
            allele_one: Arc::new(StringArray::from(metadata.allele_one)),
            allele_one_frequency: Arc::new(Float32Array::from(statistics.allele_one_frequency)),
            info_score: Arc::new(Float32Array::from(statistics.info_score)),
            observation_count: Arc::new(Int32Array::from(statistics.observation_count)),
        }
    }

    pub(crate) fn column_lengths(&self) -> [usize; 8] {
        [
            self.chromosome.len(),
            self.position.len(),
            self.variant_identifier.len(),
            self.allele_two.len(),
            self.allele_one.len(),
            self.allele_one_frequency.len(),
            self.info_score.len(),
            self.observation_count.len(),
        ]
    }
}

#[derive(Clone)]
pub struct NativeChunkHandle {
    pub(crate) chunk_identifier: i64,
    writer_arrays: Arc<NativeChunkWriterArrays>,
}

impl NativeChunkHandle {
    #[must_use]
    pub fn new(metadata: VariantMetadataColumns, statistics: ChunkStats, chunk_identifier: i64) -> Self {
        Self {
            chunk_identifier,
            writer_arrays: Arc::new(NativeChunkWriterArrays::from_chunk_sources(metadata, statistics)),
        }
    }

    #[must_use]
    pub fn row_count(&self) -> usize {
        self.writer_arrays.position.len()
    }

    pub(crate) fn variant_start_index(&self) -> i64 {
        self.chunk_identifier
    }

    pub(crate) fn variant_stop_index(&self) -> Result<i64, OutputError> {
        let row_count = i64::try_from(self.row_count()).map_err(|_| {
            OutputError::InvalidInput("Rust output writer row count does not fit into int64.".to_string())
        })?;
        self.chunk_identifier.checked_add(row_count).ok_or_else(|| {
            OutputError::InvalidInput("Rust output writer variant stop index does not fit into int64.".to_string())
        })
    }

    pub(crate) fn writer_arrays(&self) -> &NativeChunkWriterArrays {
        &self.writer_arrays
    }
}
