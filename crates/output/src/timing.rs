use std::path::Path;
use std::time::Instant;

use serde_json::json;

use crate::error::OutputError;
use crate::writer::RegenieStep2ChunkWriteTiming;

const OUTPUT_STAGE_TIMING_FILE_NAME: &str = "output_stage_timings.json";

#[derive(Default)]
pub(crate) struct OutputStageTimingAccumulator {
    pub(crate) enqueue_seconds: f64,
    pub(crate) coordinator_flush_seconds: f64,
    pub(crate) writer_record_batch_build_seconds: f64,
    pub(crate) writer_metadata_array_build_seconds: f64,
    pub(crate) writer_statistic_array_build_seconds: f64,
    pub(crate) writer_result_array_build_seconds: f64,
    pub(crate) writer_record_batch_try_new_seconds: f64,
    pub(crate) writer_parquet_file_write_seconds: f64,
    pub(crate) writer_parquet_file_create_seconds: f64,
    pub(crate) writer_parquet_init_seconds: f64,
    pub(crate) writer_parquet_batch_write_seconds: f64,
    pub(crate) writer_parquet_finish_seconds: f64,
    pub(crate) writer_parquet_rename_seconds: f64,
    pub(crate) writer_total_seconds: f64,
    pub(crate) manifest_commit_seconds: f64,
    pub(crate) finish_total_seconds: f64,
    pub(crate) enqueue_count: u64,
    pub(crate) coordinator_flush_count: u64,
    pub(crate) writer_chunk_file_count: u64,
    pub(crate) writer_chunk_count: u64,
    pub(crate) writer_row_count: u64,
    pub(crate) writer_arrow_array_memory_bytes: u64,
    pub(crate) writer_parquet_file_bytes: u64,
    pub(crate) manifest_commit_count: u64,
    pub(crate) finish_count: u64,
}

impl OutputStageTimingAccumulator {
    pub(crate) fn add_writer_timing(&mut self, timing: RegenieStep2ChunkWriteTiming) {
        self.writer_record_batch_build_seconds += timing.record_batch_build_seconds;
        self.writer_metadata_array_build_seconds += timing.metadata_array_build_seconds;
        self.writer_statistic_array_build_seconds += timing.statistic_array_build_seconds;
        self.writer_result_array_build_seconds += timing.result_array_build_seconds;
        self.writer_record_batch_try_new_seconds += timing.record_batch_try_new_seconds;
        self.writer_parquet_file_write_seconds += timing.parquet_file_write_seconds;
        self.writer_parquet_file_create_seconds += timing.parquet_file_create_seconds;
        self.writer_parquet_init_seconds += timing.parquet_writer_init_seconds;
        self.writer_parquet_batch_write_seconds += timing.parquet_batch_write_seconds;
        self.writer_parquet_finish_seconds += timing.parquet_writer_finish_seconds;
        self.writer_parquet_rename_seconds += timing.parquet_file_rename_seconds;
        self.writer_total_seconds += timing.total_seconds;
        self.writer_chunk_file_count = self.writer_chunk_file_count.saturating_add(timing.chunk_file_count);
        self.writer_chunk_count = self.writer_chunk_count.saturating_add(timing.chunk_count);
        self.writer_row_count = self.writer_row_count.saturating_add(timing.row_count);
        self.writer_arrow_array_memory_bytes =
            self.writer_arrow_array_memory_bytes.saturating_add(timing.arrow_array_memory_bytes);
        self.writer_parquet_file_bytes = self.writer_parquet_file_bytes.saturating_add(timing.parquet_file_bytes);
    }
}

pub(crate) fn start_optional_timing(collect_stage_timings: bool) -> Option<Instant> {
    collect_stage_timings.then(Instant::now)
}

pub(crate) fn write_stage_timing_snapshot(
    run_directory: &Path,
    stage_timings: &OutputStageTimingAccumulator,
) -> Result<(), OutputError> {
    let payload = json!({
        "stage_totals_seconds": {
            "rust_output_enqueue": stage_timings.enqueue_seconds,
            "rust_output_coordinator_flush": stage_timings.coordinator_flush_seconds,
            "rust_output_writer_record_batch_build": stage_timings.writer_record_batch_build_seconds,
            "rust_output_writer_metadata_arrays": stage_timings.writer_metadata_array_build_seconds,
            "rust_output_writer_statistic_arrays": stage_timings.writer_statistic_array_build_seconds,
            "rust_output_writer_result_arrays": stage_timings.writer_result_array_build_seconds,
            "rust_output_writer_record_batch_try_new": stage_timings.writer_record_batch_try_new_seconds,
            "rust_output_writer_parquet_file_write": stage_timings.writer_parquet_file_write_seconds,
            "rust_output_writer_parquet_file_create": stage_timings.writer_parquet_file_create_seconds,
            "rust_output_writer_parquet_init": stage_timings.writer_parquet_init_seconds,
            "rust_output_writer_parquet_batch_write": stage_timings.writer_parquet_batch_write_seconds,
            "rust_output_writer_parquet_finish": stage_timings.writer_parquet_finish_seconds,
            "rust_output_writer_parquet_rename": stage_timings.writer_parquet_rename_seconds,
            "rust_output_writer_total": stage_timings.writer_total_seconds,
            "rust_output_manifest_commit": stage_timings.manifest_commit_seconds,
            "rust_output_finish_total": stage_timings.finish_total_seconds,
        },
        "stage_counts": {
            "rust_output_enqueue": stage_timings.enqueue_count,
            "rust_output_coordinator_flush": stage_timings.coordinator_flush_count,
            "rust_output_writer_record_batch_build": stage_timings.writer_chunk_file_count,
            "rust_output_writer_metadata_arrays": stage_timings.writer_chunk_count,
            "rust_output_writer_statistic_arrays": stage_timings.writer_chunk_count,
            "rust_output_writer_result_arrays": stage_timings.writer_chunk_count,
            "rust_output_writer_record_batch_try_new": stage_timings.writer_chunk_count,
            "rust_output_writer_parquet_file_write": stage_timings.writer_chunk_file_count,
            "rust_output_writer_parquet_file_create": stage_timings.writer_chunk_file_count,
            "rust_output_writer_parquet_init": stage_timings.writer_chunk_file_count,
            "rust_output_writer_parquet_batch_write": stage_timings.writer_chunk_count,
            "rust_output_writer_parquet_finish": stage_timings.writer_chunk_file_count,
            "rust_output_writer_parquet_rename": stage_timings.writer_chunk_file_count,
            "rust_output_writer_total": stage_timings.writer_chunk_file_count,
            "rust_output_manifest_commit": stage_timings.manifest_commit_count,
            "rust_output_finish_total": stage_timings.finish_count,
        },
        "output_metrics": {
            "writer_chunk_file_count": stage_timings.writer_chunk_file_count,
            "writer_chunk_count": stage_timings.writer_chunk_count,
            "writer_row_count": stage_timings.writer_row_count,
            "writer_arrow_array_memory_bytes": stage_timings.writer_arrow_array_memory_bytes,
            "writer_parquet_file_bytes": stage_timings.writer_parquet_file_bytes,
        },
    });
    let timing_path = run_directory.join(OUTPUT_STAGE_TIMING_FILE_NAME);
    let temporary_timing_path = timing_path.with_extension("json.tmp");
    let mut timing_text = serde_json::to_string_pretty(&payload).map_err(OutputError::runtime)?;
    timing_text.push('\n');
    std::fs::write(&temporary_timing_path, timing_text).map_err(OutputError::runtime)?;
    std::fs::rename(&temporary_timing_path, &timing_path).map_err(OutputError::runtime)
}
