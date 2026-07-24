use std::path::PathBuf;

use arrow::array::ArrayRef;

use crate::NativeChunkHandle;
use crate::persistence::model::{OutputPartBinding, OutputTransactionIdentifier};
use crate::persistence::receipt::OutputPartReceipt;

pub(crate) struct OutputPartPublication {
    pub(crate) parts_directory: PathBuf,
    pub(crate) commits_directory: PathBuf,
    pub(crate) temporary_identifier: OutputTransactionIdentifier,
    pub(crate) binding: OutputPartBinding,
}

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

pub(super) struct RegenieStep2ChunkStreamWriteResult<OutputFile> {
    pub(super) output_file: OutputFile,
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
    pub(crate) parquet_file_sync_seconds: f64,
    pub(crate) parquet_file_hash_seconds: f64,
    pub(crate) parquet_file_publish_seconds: f64,
    pub(crate) parquet_directory_sync_seconds: f64,
    pub(crate) receipt_publish_seconds: f64,
    pub(crate) arrow_array_memory_bytes: u64,
    pub(crate) parquet_file_bytes: u64,
    pub(crate) total_seconds: f64,
}

pub(crate) struct RegenieStep2ChunkWriteResult {
    pub(crate) part_receipt: OutputPartReceipt,
    pub(crate) timing: RegenieStep2ChunkWriteTiming,
}

#[cfg(test)]
mod tests {
    use super::{RegenieStep2ParquetFileWriteTiming, RegenieStep2RecordBatchBuildTiming};

    #[test]
    fn record_batch_timing_adds_seconds_and_saturates_memory() {
        let mut total = RegenieStep2RecordBatchBuildTiming {
            metadata_array_build_seconds: 0.1,
            statistic_array_build_seconds: 0.2,
            result_array_build_seconds: 0.3,
            record_batch_try_new_seconds: 0.4,
            arrow_array_memory_bytes: u64::MAX - 1,
        };
        total.add(RegenieStep2RecordBatchBuildTiming {
            metadata_array_build_seconds: 0.5,
            statistic_array_build_seconds: 0.6,
            result_array_build_seconds: 0.7,
            record_batch_try_new_seconds: 0.8,
            arrow_array_memory_bytes: 2,
        });

        for (observed, expected) in [
            (total.metadata_array_build_seconds, 0.6),
            (total.statistic_array_build_seconds, 0.8),
            (total.result_array_build_seconds, 1.0),
            (total.record_batch_try_new_seconds, 1.2),
        ] {
            assert!((observed - expected).abs() < 1.0e-12);
        }
        assert_eq!(total.arrow_array_memory_bytes, u64::MAX);
    }

    #[test]
    fn parquet_file_timing_sums_all_stages() {
        let timing = RegenieStep2ParquetFileWriteTiming {
            file_create: 0.1,
            writer_init: 0.2,
            batch_write: 0.3,
            writer_finish: 0.4,
        };

        assert!((timing.total_seconds() - 1.0).abs() < f64::EPSILON);
    }
}
