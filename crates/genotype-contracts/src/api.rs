//! Public genotype-contract facade.

/// Byte alignment of raw-DEFLATE members within a shared compressed slab.
///
/// Official nvCOMP 5.2 and 5.3 runtimes report a four-byte input requirement.
/// Producers must align both the slab base and every member offset to this
/// boundary so CPU and CUDA consumers share one canonical layout.
pub const RAW_DEFLATE_MEMBER_ALIGNMENT: usize = 4;

pub use crate::metadata::{VariantMetadataColumns, VariantMetadataStore};
pub use crate::source::BgenSourceIdentity;
pub use crate::statistics::{ChunkOutputStatistics, NullableFloat32Column};
