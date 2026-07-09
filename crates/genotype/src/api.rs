//! Public genotype crate facade.

pub use crate::bgen::{
    BgenError, BgenReaderCore, ReaderProfileSnapshot, set_bgen_decode_tile_variant_count,
    set_bgen_row_major_direct_write_enabled,
};
pub use crate::buffer::raw_pointer::{OutputBufferAddress, OutputValueCount};
pub use crate::common::{ChunkSpec, ChunkStats, VariantMetadataColumns};
pub use crate::error::{GenotypeError, GenotypeResult};
pub use crate::preprocess::summarize_variant_major_dosage_matrix;
