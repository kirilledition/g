mod chunk_manifest;
mod job;
mod record_batch;
mod streams;
mod types;

use parquet::basic::Encoding;
use parquet::file::properties::WriterVersion;

use crate::error::OutputResult;

pub(crate) use chunk_manifest::build_part_file_name;
pub(crate) use job::write_regenie_step2_chunk_job;
pub(crate) use types::{
    OutputPartPublication, RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, RegenieStep2ChunkWriteTiming,
    RegenieStep2RecordBatchBuildTiming,
};
use types::{RegenieStep2ChunkStreamWriteResult, RegenieStep2ParquetFileWriteTiming};

pub(crate) const REGENIE_STEP2_PARQUET_FLOAT_ENCODING: Encoding = Encoding::BYTE_STREAM_SPLIT;
pub(crate) const REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE: usize = 262_144;
pub(crate) const REGENIE_STEP2_PARQUET_WRITE_BATCH_SIZE: usize = 16_384;
pub(crate) const REGENIE_STEP2_PARQUET_WRITER_VERSION: WriterVersion = WriterVersion::PARQUET_2_0;
