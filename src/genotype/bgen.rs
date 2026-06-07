mod decode;
mod error;
mod format;
mod index;
mod metadata;
mod profile;
mod reader;
mod sample_selection;
mod simd;
mod trusted;

pub use decode::set_decode_tile_variant_count as set_bgen_decode_tile_variant_count;
pub use decode::set_row_major_direct_write_enabled as set_bgen_row_major_direct_write_enabled;
pub use error::BgenError;
pub use format::CompressionType;
pub use profile::ReaderProfileSnapshot;
pub use reader::BgenReaderCore;
