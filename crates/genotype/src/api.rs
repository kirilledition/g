//! Public genotype crate facade.

pub use crate::bgen::{BgenError, BgenReaderCore, set_bgen_decode_tile_variant_count};
pub use crate::buffer::{OutputBufferAddress, OutputValueCount};
pub use crate::common::{ChunkComputeStatistics, ChunkSpec, ChunkStatisticsPolicy, ChunkStats};
pub use crate::error::{GenotypeError, GenotypeResult};
pub use crate::preprocess::summarize_variant_major_dosage_matrix;
