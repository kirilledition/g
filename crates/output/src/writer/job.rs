use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{OutputError, OutputResult};
use crate::persistence::io::{OutputIo, StdOutputIo, path_operation_error};
use crate::persistence::model::OutputTransactionIdentifier;
use crate::schema;
use crate::timing::start_optional_timing;

use super::types::{
    RegenieStep2ChunkStreamWriteResult, RegenieStep2ChunkWriteBatch, RegenieStep2ChunkWriteResult,
    RegenieStep2ChunkWriteTiming, RegenieStep2RecordBatchBuildTiming,
};
use super::{chunk_manifest, streams};

pub(crate) fn write_regenie_step2_chunk_job(
    parts_directory: &Path,
    transaction_identifier: &OutputTransactionIdentifier,
    job: RegenieStep2ChunkWriteBatch,
    collect_stage_timings: bool,
) -> OutputResult<RegenieStep2ChunkWriteResult> {
    write_regenie_step2_chunk_job_with_io(
        &StdOutputIo,
        parts_directory,
        transaction_identifier,
        job,
        collect_stage_timings,
    )
}

fn write_regenie_step2_chunk_job_with_io<OutputIoType: OutputIo>(
    output_io: &OutputIoType,
    parts_directory: &Path,
    transaction_identifier: &OutputTransactionIdentifier,
    job: RegenieStep2ChunkWriteBatch,
    collect_stage_timings: bool,
) -> OutputResult<RegenieStep2ChunkWriteResult> {
    let total_start_time = start_optional_timing(collect_stage_timings);
    let chunk_file_path = parts_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path =
        transaction_temporary_part_path(parts_directory, &job.chunk_file_name, transaction_identifier);
    let (chunk_count, row_count) = if collect_stage_timings {
        let chunk_count = u64::try_from(job.chunks.len()).map_err(OutputError::runtime)?;
        let row_count = job.chunks.iter().try_fold(0_u64, |total_row_count, chunk_job| {
            let chunk_row_count = u64::try_from(chunk_job.chunk_handle.row_count()).map_err(OutputError::runtime)?;
            total_row_count.checked_add(chunk_row_count).ok_or_else(|| {
                OutputError::Runtime("Rust output writer aggregate row count overflowed uint64.".to_string())
            })
        })?;
        (chunk_count, row_count)
    } else {
        (0, 0)
    };
    let chunk_commits = chunk_manifest::build_run_manifest_chunk_commits(&job)?;

    let file_create_start_time = start_optional_timing(collect_stage_timings);
    let output_file = output_io.create_new_file(&temporary_chunk_file_path).map_err(|error| {
        path_operation_error("create new temporary Parquet part", &temporary_chunk_file_path, &error)
    })?;
    let file_create_seconds = file_create_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
    let mut unpublished_part = UnpublishedPart::new(output_io, temporary_chunk_file_path.clone());

    let chunk_schema = Arc::clone(&schema::REGENIE_STEP2_CHUNK_SCHEMA);
    let parquet_record_batch_schema = Arc::clone(&schema::REGENIE_STEP2_PARQUET_RECORD_BATCH_FLOAT32_SCHEMA);
    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming::default();
    let RegenieStep2ChunkStreamWriteResult {
        output_file,
        record_batch_build_timing: stream_record_batch_build_timing,
        record_batch_build_seconds,
        parquet_file_write_timing,
    } = streams::write_regenie_step2_chunks_to_parquet_file(
        output_file,
        job.chunks,
        &chunk_schema,
        &parquet_record_batch_schema,
        &temporary_chunk_file_path,
        &chunk_commits,
        file_create_seconds,
        collect_stage_timings,
    )?;
    record_batch_build_timing.add(stream_record_batch_build_timing);
    output_io.sync_file(&output_file, &temporary_chunk_file_path).map_err(|error| {
        path_operation_error("synchronize temporary Parquet part", &temporary_chunk_file_path, &error)
    })?;
    drop(output_file);

    let parquet_file_rename_start_time = start_optional_timing(collect_stage_timings);
    // The unique temp prevents temp clobbering, not final-name races. Cross-process
    // final ownership belongs to the run-lease boundary; an existence precheck
    // here would be time-of-check/time-of-use unsafe.
    output_io.rename_file(&temporary_chunk_file_path, &chunk_file_path).map_err(|error| {
        OutputError::Runtime(format!(
            "Failed to rename temporary Parquet part '{}' to '{}': {error}",
            temporary_chunk_file_path.display(),
            chunk_file_path.display()
        ))
    })?;
    unpublished_part.mark_published();
    let parquet_file_rename_seconds =
        parquet_file_rename_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());

    output_io
        .sync_directory(parts_directory)
        .map_err(|error| path_operation_error("synchronize Parquet parts directory", parts_directory, &error))?;
    let parquet_file_bytes = if collect_stage_timings {
        output_io
            .file_size(&chunk_file_path)
            .map_err(|error| path_operation_error("read published Parquet part metadata", &chunk_file_path, &error))?
    } else {
        0
    };
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
            total_seconds: total_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64()),
        },
    })
}

fn transaction_temporary_part_path(
    parts_directory: &Path,
    chunk_file_name: &str,
    transaction_identifier: &OutputTransactionIdentifier,
) -> PathBuf {
    parts_directory.join(format!(".{chunk_file_name}.{}.tmp", transaction_identifier.as_str()))
}

struct UnpublishedPart<'output_io, OutputIoType: OutputIo> {
    output_io: &'output_io OutputIoType,
    temporary_path: PathBuf,
    cleanup_required: bool,
}

impl<'output_io, OutputIoType: OutputIo> UnpublishedPart<'output_io, OutputIoType> {
    fn new(output_io: &'output_io OutputIoType, temporary_path: PathBuf) -> Self {
        Self { output_io, temporary_path, cleanup_required: true }
    }

    fn mark_published(&mut self) {
        self.cleanup_required = false;
    }
}

impl<OutputIoType: OutputIo> Drop for UnpublishedPart<'_, OutputIoType> {
    fn drop(&mut self) {
        if self.cleanup_required {
            let _ = self.output_io.remove_file(&self.temporary_path);
        }
    }
}

#[cfg(test)]
mod tests;
