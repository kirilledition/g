//! Public genotype crate facade.

pub use crate::bgen::{
    BgenError, BgenReadSession, BgenReaderCore, CompressedPacked8Batch, CompressedPacked8BatchLayout,
    CompressedPacked8SampleSelection, CompressedPacked8Transfer,
};
pub use crate::common::{
    ChunkComputeStatistics, ChunkSpec, ChunkStatisticsPolicy, ChunkStats, DecodedGenotypeBatch, OwnedGenotypeBuffer,
    Packed8Compatibility, PooledPacked8Buffer,
};
pub use crate::error::{GenotypeError, GenotypeResult};
