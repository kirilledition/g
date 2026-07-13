//! Public genotype crate facade.

pub use crate::bgen::{BgenError, BgenReadSession, BgenReaderCore};
pub use crate::common::{
    ChunkComputeStatistics, ChunkSpec, ChunkStatisticsPolicy, ChunkStats, DecodedGenotypeBatch, OwnedGenotypeBuffer,
    Packed8Compatibility,
};
pub use crate::error::{GenotypeError, GenotypeResult};
