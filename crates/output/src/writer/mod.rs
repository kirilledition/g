mod chunk_manifest;
mod job;
mod record_batch;
mod regenie_text;
mod streams;
mod types;

use crate::error::OutputResult;

pub(crate) use chunk_manifest::{build_output_file_name, build_regenie_text_metadata_sidecar_path};
pub(crate) use job::write_regenie_step2_chunk_job;
pub(crate) use regenie_text::REGENIE_STEP2_TEXT_HEADER;
pub use types::OutputFileFormat;
use types::{RegenieStep2ArrowFileWriteTiming, RegenieStep2ChunkStreamWriteResult};
pub(crate) use types::{
    RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, RegenieStep2ChunkWriteTiming, RegenieStep2RecordBatchBuildTiming,
};
