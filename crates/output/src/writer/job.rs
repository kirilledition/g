use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::error::{OutputError, OutputResult};
use crate::persistence::io::{FileIntegrity, NoReplacePublication, OutputIo, StdOutputIo, path_operation_error};
use crate::persistence::receipt::{OutputPartFooter, OutputPartReceipt, publish_part_receipt, read_part_footer};
use crate::schema;
use crate::timing::start_optional_timing;

use super::types::{
    OutputPartPublication, RegenieStep2ChunkStreamWriteResult, RegenieStep2ChunkWriteBatch,
    RegenieStep2ChunkWriteResult, RegenieStep2ChunkWriteTiming, RegenieStep2RecordBatchBuildTiming,
};
use super::{chunk_manifest, streams};

pub(crate) fn write_regenie_step2_chunk_job(
    publication: &OutputPartPublication,
    job: RegenieStep2ChunkWriteBatch,
    collect_stage_timings: bool,
) -> OutputResult<RegenieStep2ChunkWriteResult> {
    write_regenie_step2_chunk_job_with_io(&StdOutputIo, publication, job, collect_stage_timings)
}

struct OutputJobGeometry {
    chunk_count: u64,
    row_count: u64,
}

struct OutputJobTimingInput {
    geometry: OutputJobGeometry,
    record_batch_build_timing: RegenieStep2RecordBatchBuildTiming,
    record_batch_build_seconds: f64,
    parquet_file_write_timing: super::types::RegenieStep2ParquetFileWriteTiming,
    parquet_file_sync_seconds: f64,
    parquet_file_hash_seconds: f64,
    parquet_file_publish_seconds: f64,
    parquet_directory_sync_seconds: f64,
    receipt_publish_seconds: f64,
    parquet_file_bytes: u64,
    total_start_time: Option<Instant>,
}

fn write_regenie_step2_chunk_job_with_io<OutputIoType: OutputIo>(
    output_io: &OutputIoType,
    publication: &OutputPartPublication,
    job: RegenieStep2ChunkWriteBatch,
    collect_stage_timings: bool,
) -> OutputResult<RegenieStep2ChunkWriteResult> {
    let total_start_time = start_optional_timing(collect_stage_timings);
    let chunk_file_path = publication.parts_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = transaction_temporary_part_path(
        &publication.parts_directory,
        &job.chunk_file_name,
        &publication.temporary_identifier,
    );
    let geometry = output_job_geometry(&job, collect_stage_timings)?;
    let chunk_commits = chunk_manifest::build_run_manifest_chunk_commits(&job)?;
    let part_footer = OutputPartFooter::new(&publication.binding, job.chunk_file_name.clone(), chunk_commits.clone())?;

    let file_create_start_time = start_optional_timing(collect_stage_timings);
    let output_file = output_io.create_new_file(&temporary_chunk_file_path).map_err(|error| {
        path_operation_error("create new temporary Parquet part", &temporary_chunk_file_path, &error)
    })?;
    let file_create_seconds = file_create_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
    let mut unpublished_part = UnpublishedPart::new(output_io, temporary_chunk_file_path.clone());

    let RegenieStep2ChunkStreamWriteResult {
        output_file,
        record_batch_build_timing,
        record_batch_build_seconds,
        parquet_file_write_timing,
    } = streams::write_regenie_step2_chunks_to_parquet_file(
        output_file,
        streams::RegenieStep2ParquetStreamRequest {
            chunks: job.chunks,
            chunk_schema: &schema::REGENIE_STEP2_CHUNK_SCHEMA,
            parquet_record_batch_schema: &schema::REGENIE_STEP2_PARQUET_RECORD_BATCH_FLOAT32_SCHEMA,
            chunk_file_path: &temporary_chunk_file_path,
            part_footer: &part_footer,
            file_create_seconds,
            collect_stage_timings,
        },
    )?;
    let parquet_file_sync_start_time = start_optional_timing(collect_stage_timings);
    output_io.sync_file(&output_file, &temporary_chunk_file_path).map_err(|error| {
        path_operation_error("synchronize temporary Parquet part", &temporary_chunk_file_path, &error)
    })?;
    let parquet_file_sync_seconds =
        parquet_file_sync_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
    drop(output_file);

    let parquet_file_hash_start_time = start_optional_timing(collect_stage_timings);
    let part_integrity = output_io
        .file_integrity(&temporary_chunk_file_path)
        .map_err(|error| path_operation_error("hash temporary Parquet part", &temporary_chunk_file_path, &error))?;
    let parquet_file_hash_seconds =
        parquet_file_hash_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
    let parquet_file_publish_start_time = start_optional_timing(collect_stage_timings);
    publish_or_reconcile_part(
        output_io,
        &mut unpublished_part,
        &temporary_chunk_file_path,
        &chunk_file_path,
        &part_footer,
        &part_integrity,
    )?;
    let parquet_file_publish_seconds =
        parquet_file_publish_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());

    let parquet_directory_sync_start_time = start_optional_timing(collect_stage_timings);
    output_io.sync_directory(&publication.parts_directory).map_err(|error| {
        path_operation_error("synchronize Parquet parts directory", &publication.parts_directory, &error)
    })?;
    let parquet_directory_sync_seconds =
        parquet_directory_sync_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
    let receipt_publish_start_time = start_optional_timing(collect_stage_timings);
    let part_receipt = OutputPartReceipt::new(part_footer, part_integrity.clone())?;
    publish_part_receipt(&publication.commits_directory, &part_receipt)?;
    let receipt_publish_seconds =
        receipt_publish_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
    let parquet_file_bytes = if collect_stage_timings { part_integrity.size_bytes } else { 0 };
    Ok(RegenieStep2ChunkWriteResult {
        part_receipt,
        timing: build_output_job_timing(&OutputJobTimingInput {
            geometry,
            record_batch_build_timing,
            record_batch_build_seconds,
            parquet_file_write_timing,
            parquet_file_sync_seconds,
            parquet_file_hash_seconds,
            parquet_file_publish_seconds,
            parquet_directory_sync_seconds,
            receipt_publish_seconds,
            parquet_file_bytes,
            total_start_time,
        }),
    })
}

fn output_job_geometry(
    job: &RegenieStep2ChunkWriteBatch,
    collect_stage_timings: bool,
) -> OutputResult<OutputJobGeometry> {
    if !collect_stage_timings {
        return Ok(OutputJobGeometry { chunk_count: 0, row_count: 0 });
    }
    let chunk_count = u64::try_from(job.chunks.len()).map_err(OutputError::runtime)?;
    let row_count = job.chunks.iter().try_fold(0_u64, |total_row_count, chunk_job| {
        let chunk_row_count = u64::try_from(chunk_job.chunk_handle.row_count()).map_err(OutputError::runtime)?;
        total_row_count.checked_add(chunk_row_count).ok_or_else(|| {
            OutputError::Runtime("Rust output writer aggregate row count overflowed uint64.".to_string())
        })
    })?;
    Ok(OutputJobGeometry { chunk_count, row_count })
}

fn publish_or_reconcile_part<OutputIoType: OutputIo>(
    output_io: &OutputIoType,
    unpublished_part: &mut UnpublishedPart<'_, OutputIoType>,
    temporary_chunk_file_path: &Path,
    chunk_file_path: &Path,
    part_footer: &OutputPartFooter,
    part_integrity: &FileIntegrity,
) -> OutputResult<()> {
    let publication_result =
        output_io.publish_file_no_replace(temporary_chunk_file_path, chunk_file_path).map_err(|error| {
            OutputError::Runtime(format!(
                "Failed to publish temporary Parquet part '{}' as '{}' without replacement: {error}",
                temporary_chunk_file_path.display(),
                chunk_file_path.display()
            ))
        })?;
    if publication_result == NoReplacePublication::Created {
        unpublished_part.mark_published();
        return Ok(());
    }
    let existing_footer = read_part_footer(chunk_file_path)?;
    if existing_footer != *part_footer {
        return Err(OutputError::InvalidInput(format!(
            "Published Parquet part '{}' conflicts with the expected immutable footer.",
            chunk_file_path.display()
        )));
    }
    let existing_integrity = output_io
        .file_integrity(chunk_file_path)
        .map_err(|error| path_operation_error("hash existing Parquet part", chunk_file_path, &error))?;
    if existing_integrity != *part_integrity {
        return Err(OutputError::InvalidInput(format!(
            "Published Parquet part '{}' conflicts with the expected raw bytes.",
            chunk_file_path.display()
        )));
    }
    output_io.remove_file(temporary_chunk_file_path).map_err(|error| {
        path_operation_error("remove reconciled temporary Parquet part", temporary_chunk_file_path, &error)
    })?;
    unpublished_part.mark_published();
    Ok(())
}

fn build_output_job_timing(input: &OutputJobTimingInput) -> RegenieStep2ChunkWriteTiming {
    RegenieStep2ChunkWriteTiming {
        chunk_file_count: 1,
        chunk_count: input.geometry.chunk_count,
        row_count: input.geometry.row_count,
        record_batch_build_seconds: input.record_batch_build_seconds,
        metadata_array_build_seconds: input.record_batch_build_timing.metadata_array_build_seconds,
        statistic_array_build_seconds: input.record_batch_build_timing.statistic_array_build_seconds,
        result_array_build_seconds: input.record_batch_build_timing.result_array_build_seconds,
        record_batch_try_new_seconds: input.record_batch_build_timing.record_batch_try_new_seconds,
        parquet_file_write_seconds: input.parquet_file_write_timing.total_seconds(),
        parquet_file_create_seconds: input.parquet_file_write_timing.file_create,
        parquet_writer_init_seconds: input.parquet_file_write_timing.writer_init,
        parquet_batch_write_seconds: input.parquet_file_write_timing.batch_write,
        parquet_writer_finish_seconds: input.parquet_file_write_timing.writer_finish,
        parquet_file_sync_seconds: input.parquet_file_sync_seconds,
        parquet_file_hash_seconds: input.parquet_file_hash_seconds,
        parquet_file_publish_seconds: input.parquet_file_publish_seconds,
        parquet_directory_sync_seconds: input.parquet_directory_sync_seconds,
        receipt_publish_seconds: input.receipt_publish_seconds,
        arrow_array_memory_bytes: input.record_batch_build_timing.arrow_array_memory_bytes,
        parquet_file_bytes: input.parquet_file_bytes,
        total_seconds: input.total_start_time.as_ref().map_or(0.0, |start_time| start_time.elapsed().as_secs_f64()),
    }
}

fn transaction_temporary_part_path(
    parts_directory: &Path,
    chunk_file_name: &str,
    transaction_identifier: &crate::persistence::model::OutputTransactionIdentifier,
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
