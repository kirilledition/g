mod decode;
mod error;
mod format;
mod index;
mod metadata;
mod packed8;
mod packed8_cache;
mod raw_deflate;
mod reader;
mod request;
mod sample_selection;
mod simd;
mod source;

#[cfg(feature = "benchmark-internals")]
pub(crate) use source::MAXIMUM_OWNED_SNAPSHOT_BYTE_COUNT;

pub use error::BgenError;
pub(in crate::bgen) use format::CompressionType;
pub use raw_deflate::{
    CompressedPacked8Batch, CompressedPacked8BatchLayout, CompressedPacked8SampleSelection, CompressedPacked8Transfer,
};
pub use reader::{BgenReadSession, BgenReaderCore};
pub use request::{BgenContentSelector, BgenOpenRequest};
