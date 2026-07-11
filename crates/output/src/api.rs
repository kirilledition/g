//! Public output crate facade.

pub use crate::chunk::{NativeChunkHandle, NativeVariantMetadataHandle};
pub use crate::error::OutputError;
pub use crate::manager::{CompletedOutputRun, OutputDeliveryState, OutputManager};
pub use crate::manifest::{CurrentRunManifestHeaderInput, ManifestFileFingerprintCache, PredictionLocoFileFingerprint};
pub use crate::session::OutputWriterSession;
pub use crate::write_plan::{Regenie2StatisticBatch, write_regenie2_multi_trait_chunk_f32};
