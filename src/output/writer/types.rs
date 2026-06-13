use arrow::array::ArrayRef;
use thiserror::Error;

use crate::output::NativeChunkHandle;
use crate::output::manifest;

#[derive(Debug, Error)]
pub enum OutputWriterError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Runtime(String),
}

impl OutputWriterError {
    pub(crate) fn runtime(error: impl ToString) -> Self {
        Self::Runtime(error.to_string())
    }
}

pub(crate) struct RegenieStep2ChunkJob {
    pub(crate) chunk_handle: NativeChunkHandle,
    pub(crate) beta: ArrayRef,
    pub(crate) se: ArrayRef,
    pub(crate) chisq: ArrayRef,
    pub(crate) log10p: ArrayRef,
    pub(crate) extra_code: Option<ArrayRef>,
}

pub(crate) struct RegenieStep2ChunkWriteBatch {
    pub(crate) chunk_file_name: String,
    pub(crate) chunks: Vec<RegenieStep2ChunkJob>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum OutputFileFormat {
    Arrow,
    Parquet,
    Regenie,
}

impl OutputFileFormat {
    pub(crate) fn parse(output_format: &str) -> Result<Self, String> {
        match output_format {
            "arrow" => Ok(Self::Arrow),
            "parquet" => Ok(Self::Parquet),
            "regenie" => Ok(Self::Regenie),
            unsupported_output_format => Err(format!(
                "Output format must be 'arrow', 'parquet', or 'regenie', observed '{unsupported_output_format}'."
            )),
        }
    }

    pub(super) fn value(self) -> &'static str {
        match self {
            Self::Arrow => "arrow",
            Self::Parquet => "parquet",
            Self::Regenie => "regenie",
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct RegenieStep2RecordBatchBuildTiming {
    pub(crate) schema_metadata_build_seconds: f64,
    pub(crate) metadata_array_build_seconds: f64,
    pub(crate) statistic_array_build_seconds: f64,
    pub(crate) test_array_build_seconds: f64,
    pub(crate) result_array_build_seconds: f64,
    pub(crate) extra_array_build_seconds: f64,
    pub(crate) record_batch_try_new_seconds: f64,
    pub(crate) arrow_array_memory_bytes: u64,
}

impl RegenieStep2RecordBatchBuildTiming {
    pub(super) fn add(&mut self, timing: Self) {
        self.schema_metadata_build_seconds += timing.schema_metadata_build_seconds;
        self.metadata_array_build_seconds += timing.metadata_array_build_seconds;
        self.statistic_array_build_seconds += timing.statistic_array_build_seconds;
        self.test_array_build_seconds += timing.test_array_build_seconds;
        self.result_array_build_seconds += timing.result_array_build_seconds;
        self.extra_array_build_seconds += timing.extra_array_build_seconds;
        self.record_batch_try_new_seconds += timing.record_batch_try_new_seconds;
        self.arrow_array_memory_bytes = self.arrow_array_memory_bytes.saturating_add(timing.arrow_array_memory_bytes);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RegenieStep2ChunkWriteTiming {
    pub(crate) chunk_file_count: u64,
    pub(crate) chunk_count: u64,
    pub(crate) row_count: u64,
    pub(crate) record_batch_build_seconds: f64,
    pub(crate) schema_metadata_build_seconds: f64,
    pub(crate) metadata_array_build_seconds: f64,
    pub(crate) statistic_array_build_seconds: f64,
    pub(crate) test_array_build_seconds: f64,
    pub(crate) result_array_build_seconds: f64,
    pub(crate) extra_array_build_seconds: f64,
    pub(crate) record_batch_try_new_seconds: f64,
    pub(crate) arrow_file_write_seconds: f64,
    pub(crate) arrow_file_create_seconds: f64,
    pub(crate) arrow_writer_init_seconds: f64,
    pub(crate) arrow_batch_write_seconds: f64,
    pub(crate) arrow_writer_finish_seconds: f64,
    pub(crate) arrow_file_rename_seconds: f64,
    pub(crate) arrow_array_memory_bytes: u64,
    pub(crate) arrow_file_bytes: u64,
    pub(crate) total_seconds: f64,
}

pub(crate) struct RegenieStep2ChunkWriteResult {
    pub(crate) chunk_commits: Vec<manifest::RunManifestChunkCommit>,
    pub(crate) timing: RegenieStep2ChunkWriteTiming,
}
