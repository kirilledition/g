use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Int32Array, Int64Array, StringArray};

use super::{NativeChunkStats, VariantMetadataColumns};

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
    pub(crate) fn from_chunk_sources(metadata: &VariantMetadataColumns, stats: &NativeChunkStats) -> Self {
        Self {
            chromosome: Arc::new(StringArray::from(metadata.chromosome.clone())),
            position: Arc::new(Int64Array::from(metadata.position.clone())),
            variant_identifier: Arc::new(StringArray::from(metadata.variant_identifier.clone())),
            allele_two: Arc::new(StringArray::from(metadata.allele_two.clone())),
            allele_one: Arc::new(StringArray::from(metadata.allele_one.clone())),
            allele_one_frequency: Arc::new(Float32Array::from(stats.allele_one_frequency.clone())),
            info_score: Arc::new(Float32Array::from(stats.info_score.clone())),
            observation_count: Arc::new(Int32Array::from(stats.observation_count.clone())),
        }
    }
}
