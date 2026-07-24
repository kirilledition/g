//! Public genotype crate facade.

pub use crate::bgen::{
    BgenContentSelector, BgenError, BgenOpenRequest, BgenReadSession, BgenReaderCore, CompressedPacked8Batch,
    CompressedPacked8BatchLayout, CompressedPacked8SampleSelection, CompressedPacked8Transfer,
};
pub use crate::common::{
    ChunkComputeStatistics, ChunkSpec, ChunkStatisticsPolicy, ChunkStats, GenotypeBatch, GenotypeBatchPayload,
    OwnedGenotypeBuffer, Packed8Compatibility, Packed8RawStatistics, PooledPacked8Buffer,
};
pub use crate::error::{GenotypeError, GenotypeResult};
