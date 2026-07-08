//! Public genotype crate facade.

pub use crate::bgen::{BgenError, ReaderProfileSnapshot};
pub use crate::common::{ChunkSpec, ChunkStats, GenotypeReaderCore, VariantMetadataColumns};
pub use crate::error::{GenotypeError, GenotypeResult};
pub use crate::source::BgenGenotypeSource;
