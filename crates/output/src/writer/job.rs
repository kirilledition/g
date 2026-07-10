#![allow(clippy::needless_pass_by_value)]

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::error::{OutputError, OutputResult};
use crate::schema;

use super::types::{
    RegenieStep2ChunkWriteBatch, RegenieStep2ChunkWriteResult, RegenieStep2ChunkWriteTiming,
    RegenieStep2RecordBatchBuildTiming,
};
use super::{chunk_manifest, streams};

pub(crate) fn write_regenie_step2_chunk_job(
    parts_directory: &Path,
    job: RegenieStep2ChunkWriteBatch,
) -> OutputResult<RegenieStep2ChunkWriteResult> {
    let total_start_time = Instant::now();
    let chunk_file_path = parts_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = chunk_file_path.with_extension("parquet.tmp");
    let chunk_count = u64::try_from(job.chunks.len()).map_err(OutputError::runtime)?;
    let row_count = job.chunks.iter().try_fold(0_u64, |total_row_count, chunk_job| {
        let chunk_row_count = u64::try_from(chunk_job.chunk_handle.row_count()).map_err(OutputError::runtime)?;
        total_row_count.checked_add(chunk_row_count).ok_or_else(|| {
            OutputError::Runtime("Rust output writer aggregate row count overflowed uint64.".to_string())
        })
    })?;
    let chunk_commits = chunk_manifest::build_run_manifest_chunk_commits(&job)?;

    let chunk_schema = Arc::clone(schema::get_regenie_step2_chunk_schema());
    let parquet_record_batch_schema = Arc::clone(schema::get_regenie_step2_parquet_record_batch_schema());
    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming::default();
    let stream_write_result = streams::write_regenie_step2_chunks_to_parquet_file(
        job.chunks,
        &chunk_schema,
        &parquet_record_batch_schema,
        &temporary_chunk_file_path,
        &chunk_commits,
    )?;
    record_batch_build_timing.add(stream_write_result.record_batch_build_timing);
    let record_batch_build_seconds = stream_write_result.record_batch_build_seconds;
    let parquet_file_write_timing = stream_write_result.parquet_file_write_timing;
    let parquet_file_rename_start_time = Instant::now();
    std::fs::rename(&temporary_chunk_file_path, &chunk_file_path).map_err(OutputError::runtime)?;
    let parquet_file_rename_seconds = parquet_file_rename_start_time.elapsed().as_secs_f64();
    let parquet_file_bytes = std::fs::metadata(&chunk_file_path).map_err(OutputError::runtime)?.len();
    let parquet_file_write_seconds = parquet_file_write_timing.total_seconds();

    Ok(RegenieStep2ChunkWriteResult {
        chunk_commits,
        timing: RegenieStep2ChunkWriteTiming {
            chunk_file_count: 1,
            chunk_count,
            row_count,
            record_batch_build_seconds,
            metadata_array_build_seconds: record_batch_build_timing.metadata_array_build_seconds,
            statistic_array_build_seconds: record_batch_build_timing.statistic_array_build_seconds,
            result_array_build_seconds: record_batch_build_timing.result_array_build_seconds,
            record_batch_try_new_seconds: record_batch_build_timing.record_batch_try_new_seconds,
            parquet_file_write_seconds,
            parquet_file_create_seconds: parquet_file_write_timing.file_create,
            parquet_writer_init_seconds: parquet_file_write_timing.writer_init,
            parquet_batch_write_seconds: parquet_file_write_timing.batch_write,
            parquet_writer_finish_seconds: parquet_file_write_timing.writer_finish,
            parquet_file_rename_seconds,
            arrow_array_memory_bytes: record_batch_build_timing.arrow_array_memory_bytes,
            parquet_file_bytes,
            total_seconds: total_start_time.elapsed().as_secs_f64(),
        },
    })
}
