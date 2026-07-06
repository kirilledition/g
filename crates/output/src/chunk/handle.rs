use std::sync::{Arc, OnceLock};

use crate::error::OutputError;

use super::{NativeChunkStats, NativeChunkWriterArrays, VariantMetadataColumns};

#[derive(Clone)]
pub struct NativeChunkHandle {
    pub(crate) metadata: Arc<VariantMetadataColumns>,
    pub(crate) stats: Arc<NativeChunkStats>,
    pub(crate) chunk_identifier: i64,
    writer_arrays: Arc<OnceLock<NativeChunkWriterArrays>>,
}

impl NativeChunkHandle {
    #[must_use]
    pub fn new(metadata: Arc<VariantMetadataColumns>, stats: Arc<NativeChunkStats>, chunk_identifier: i64) -> Self {
        Self { metadata, stats, chunk_identifier, writer_arrays: Arc::new(OnceLock::new()) }
    }

    #[must_use]
    pub fn row_count(&self) -> usize {
        self.metadata.position.len()
    }

    pub(crate) fn variant_start_index(&self) -> i64 {
        self.chunk_identifier
    }

    pub(crate) fn variant_stop_index(&self) -> Result<i64, OutputError> {
        let row_count = i64::try_from(self.row_count()).map_err(|_| {
            OutputError::InvalidInput("Rust output writer row count does not fit into int64.".to_string())
        })?;
        self.chunk_identifier.checked_add(row_count).ok_or_else(|| {
            OutputError::InvalidInput("Rust output writer variant stop index does not fit into int64.".to_string())
        })
    }

    pub(crate) fn writer_arrays(&self) -> &NativeChunkWriterArrays {
        self.writer_arrays.get_or_init(|| NativeChunkWriterArrays::from_chunk_sources(&self.metadata, &self.stats))
    }
}
