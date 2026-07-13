mod decode;
mod error;
mod format;
mod index;
mod metadata;
mod packed8;
mod packed8_cache;
mod reader;
mod sample_selection;
mod simd;

pub use error::BgenError;
pub(in crate::bgen) use format::CompressionType;
pub use reader::{BgenReadSession, BgenReaderCore};
