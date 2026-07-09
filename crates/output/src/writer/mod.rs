#![allow(clippy::needless_pass_by_value)]

use std::path::Path;
use std::time::Instant;

use crate::error::OutputError;
use crate::schema::OutputStatisticDtype;

mod chunk_manifest;
mod record_batch;
mod regenie_text;
mod streams;
mod types;

pub(crate) use chunk_manifest::{build_output_file_name, build_regenie_text_metadata_sidecar_path};
pub(crate) use regenie_text::REGENIE_STEP2_TEXT_HEADER;
pub use types::OutputFileFormat;
use types::{RegenieStep2ArrowFileWriteTiming, RegenieStep2ChunkStreamWriteResult};
pub(crate) use types::{
    RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, RegenieStep2ChunkWriteResult, RegenieStep2ChunkWriteTiming,
    RegenieStep2RecordBatchBuildTiming,
};

type OutputWriterResult<T> = Result<T, OutputError>;

pub(crate) fn write_regenie_step2_chunk_job(
    chunks_directory: &Path,
    job: RegenieStep2ChunkWriteBatch,
    output_format: OutputFileFormat,
    output_statistic_dtype: OutputStatisticDtype,
    arrow_compression: &str,
    parquet_compression: &str,
) -> OutputWriterResult<RegenieStep2ChunkWriteResult> {
    let total_start_time = Instant::now();
    let chunk_file_path = chunks_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = match output_format {
        OutputFileFormat::Arrow => chunk_file_path.with_extension("arrow.tmp"),
        OutputFileFormat::Parquet => chunk_file_path.with_extension("parquet.tmp"),
        OutputFileFormat::Regenie => chunk_file_path.with_extension("regenie.tmp"),
    };
    let chunk_count = u64::try_from(job.chunks.len()).map_err(OutputError::runtime)?;
    let row_count = job
        .chunks
        .iter()
        .map(|chunk_job| u64::try_from(chunk_job.chunk_handle.row_count()).map_err(OutputError::runtime))
        .sum::<OutputWriterResult<u64>>()?;
    let compression = match output_format {
        OutputFileFormat::Arrow => arrow_compression,
        OutputFileFormat::Parquet => parquet_compression,
        OutputFileFormat::Regenie => "none",
    };
    let chunk_commits = chunk_manifest::build_run_manifest_chunk_commits(&job, output_format, compression)?;

    let schema_metadata_build_start_time = Instant::now();
    let chunk_schema = chunk_manifest::build_regenie_step2_chunk_file_schema(&chunk_commits, output_statistic_dtype)?;
    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming {
        schema_metadata_build_seconds: schema_metadata_build_start_time.elapsed().as_secs_f64(),
        ..RegenieStep2RecordBatchBuildTiming::default()
    };
    let stream_write_result = match output_format {
        OutputFileFormat::Arrow => streams::write_regenie_step2_chunks_to_arrow_file(
            job.chunks,
            chunk_schema,
            &temporary_chunk_file_path,
            arrow_compression,
        )?,
        OutputFileFormat::Parquet => streams::write_regenie_step2_chunks_to_parquet_file(
            job.chunks,
            chunk_schema,
            &temporary_chunk_file_path,
            parquet_compression,
            &chunk_commits,
        )?,
        OutputFileFormat::Regenie => regenie_text::write_regenie_step2_chunks_to_regenie_text_file(
            job.chunks,
            chunk_schema,
            &temporary_chunk_file_path,
        )?,
    };
    record_batch_build_timing.add(stream_write_result.record_batch_build_timing);
    let record_batch_build_seconds =
        record_batch_build_timing.schema_metadata_build_seconds + stream_write_result.record_batch_build_seconds;
    let arrow_file_write_timing = stream_write_result.arrow_file_write_timing;
    let arrow_file_rename_start_time = Instant::now();
    std::fs::rename(&temporary_chunk_file_path, &chunk_file_path).map_err(OutputError::runtime)?;
    let arrow_file_rename_seconds = arrow_file_rename_start_time.elapsed().as_secs_f64();
    if output_format == OutputFileFormat::Regenie {
        regenie_text::write_regenie_text_metadata_sidecar(&chunk_file_path, &chunk_commits)?;
    }
    let arrow_file_bytes = std::fs::metadata(&chunk_file_path).map_err(OutputError::runtime)?.len();
    let arrow_file_write_seconds = arrow_file_write_timing.total_seconds();

    Ok(RegenieStep2ChunkWriteResult {
        chunk_commits,
        timing: RegenieStep2ChunkWriteTiming {
            chunk_file_count: 1,
            chunk_count,
            row_count,
            record_batch_build_seconds,
            schema_metadata_build_seconds: record_batch_build_timing.schema_metadata_build_seconds,
            metadata_array_build_seconds: record_batch_build_timing.metadata_array_build_seconds,
            statistic_array_build_seconds: record_batch_build_timing.statistic_array_build_seconds,
            test_array_build_seconds: record_batch_build_timing.test_array_build_seconds,
            result_array_build_seconds: record_batch_build_timing.result_array_build_seconds,
            extra_array_build_seconds: record_batch_build_timing.extra_array_build_seconds,
            record_batch_try_new_seconds: record_batch_build_timing.record_batch_try_new_seconds,
            arrow_file_write_seconds,
            arrow_file_create_seconds: arrow_file_write_timing.file_create,
            arrow_writer_init_seconds: arrow_file_write_timing.writer_init,
            arrow_batch_write_seconds: arrow_file_write_timing.batch_write,
            arrow_writer_finish_seconds: arrow_file_write_timing.writer_finish,
            arrow_file_rename_seconds,
            arrow_array_memory_bytes: record_batch_build_timing.arrow_array_memory_bytes,
            arrow_file_bytes,
            total_seconds: total_start_time.elapsed().as_secs_f64(),
        },
    })
}
