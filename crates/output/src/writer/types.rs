use arrow::array::ArrayRef;

use crate::NativeChunkHandle;
use crate::manifest;

pub(crate) struct RegenieStep2ChunkJob {
    pub(crate) chunk_handle: NativeChunkHandle,
    pub(crate) beta: ArrayRef,
    pub(crate) se: ArrayRef,
    pub(crate) chisq: ArrayRef,
    pub(crate) log10p: ArrayRef,
    pub(crate) correction_code: Option<ArrayRef>,
}

pub(crate) struct RegenieStep2ChunkWriteBatch {
    pub(crate) chunk_file_name: String,
    pub(crate) chunks: Vec<RegenieStep2ChunkJob>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct RegenieStep2RecordBatchBuildTiming {
    pub(crate) metadata_array_build_seconds: f64,
    pub(crate) statistic_array_build_seconds: f64,
    pub(crate) result_array_build_seconds: f64,
    pub(crate) record_batch_try_new_seconds: f64,
    pub(crate) arrow_array_memory_bytes: u64,
}

impl RegenieStep2RecordBatchBuildTiming {
    pub(super) fn add(&mut self, timing: Self) {
        self.metadata_array_build_seconds += timing.metadata_array_build_seconds;
        self.statistic_array_build_seconds += timing.statistic_array_build_seconds;
        self.result_array_build_seconds += timing.result_array_build_seconds;
        self.record_batch_try_new_seconds += timing.record_batch_try_new_seconds;
        self.arrow_array_memory_bytes = self.arrow_array_memory_bytes.saturating_add(timing.arrow_array_memory_bytes);
    }
}

pub(super) struct RegenieStep2ParquetFileWriteTiming {
    pub(super) file_create: f64,
    pub(super) writer_init: f64,
    pub(super) batch_write: f64,
    pub(super) writer_finish: f64,
}

impl RegenieStep2ParquetFileWriteTiming {
    pub(super) fn total_seconds(&self) -> f64 {
        self.file_create + self.writer_init + self.batch_write + self.writer_finish
    }
}

pub(super) struct RegenieStep2ChunkStreamWriteResult {
    pub(super) record_batch_build_timing: RegenieStep2RecordBatchBuildTiming,
    pub(super) record_batch_build_seconds: f64,
    pub(super) parquet_file_write_timing: RegenieStep2ParquetFileWriteTiming,
}

#[derive(Clone, Copy)]
pub(crate) struct RegenieStep2ChunkWriteTiming {
    pub(crate) chunk_file_count: u64,
    pub(crate) chunk_count: u64,
    pub(crate) row_count: u64,
    pub(crate) record_batch_build_seconds: f64,
    pub(crate) metadata_array_build_seconds: f64,
    pub(crate) statistic_array_build_seconds: f64,
    pub(crate) result_array_build_seconds: f64,
    pub(crate) record_batch_try_new_seconds: f64,
    pub(crate) parquet_file_write_seconds: f64,
    pub(crate) parquet_file_create_seconds: f64,
    pub(crate) parquet_writer_init_seconds: f64,
    pub(crate) parquet_batch_write_seconds: f64,
    pub(crate) parquet_writer_finish_seconds: f64,
    pub(crate) parquet_file_rename_seconds: f64,
    pub(crate) arrow_array_memory_bytes: u64,
    pub(crate) parquet_file_bytes: u64,
    pub(crate) total_seconds: f64,
}

pub(crate) struct RegenieStep2ChunkWriteResult {
    pub(crate) chunk_commits: Vec<manifest::RunManifestChunkCommit>,
    pub(crate) timing: RegenieStep2ChunkWriteTiming,
}
