use std::path::Path;

use serde_json::json;

use crate::error::OutputError;

use super::OutputStageTimingAccumulator;

const OUTPUT_STAGE_TIMING_FILE_NAME: &str = "output_stage_timings.json";

pub(crate) fn write_stage_timing_snapshot(
    run_directory: &Path,
    stage_timings: &OutputStageTimingAccumulator,
) -> Result<(), OutputError> {
    let payload = json!({
        "stage_totals_seconds": {
            "rust_output_metadata_clone": stage_timings.metadata_clone_seconds,
            "rust_output_result_buffer_copy": stage_timings.result_buffer_copy_seconds,
            "rust_output_enqueue": stage_timings.enqueue_seconds,
            "rust_output_coordinator_flush": stage_timings.coordinator_flush_seconds,
            "rust_output_writer_record_batch_build": stage_timings.writer_record_batch_build_seconds,
            "rust_output_writer_schema_metadata_build": stage_timings.writer_schema_metadata_build_seconds,
            "rust_output_writer_metadata_arrays": stage_timings.writer_metadata_array_build_seconds,
            "rust_output_writer_statistic_arrays": stage_timings.writer_statistic_array_build_seconds,
            "rust_output_writer_test_array": stage_timings.writer_test_array_build_seconds,
            "rust_output_writer_result_arrays": stage_timings.writer_result_array_build_seconds,
            "rust_output_writer_extra_array": stage_timings.writer_extra_array_build_seconds,
            "rust_output_writer_record_batch_try_new": stage_timings.writer_record_batch_try_new_seconds,
            "rust_output_writer_arrow_file_write": stage_timings.writer_arrow_file_write_seconds,
            "rust_output_writer_arrow_file_create": stage_timings.writer_arrow_file_create_seconds,
            "rust_output_writer_arrow_init": stage_timings.writer_arrow_init_seconds,
            "rust_output_writer_arrow_batch_write": stage_timings.writer_arrow_batch_write_seconds,
            "rust_output_writer_arrow_finish": stage_timings.writer_arrow_finish_seconds,
            "rust_output_writer_arrow_rename": stage_timings.writer_arrow_rename_seconds,
            "rust_output_writer_total": stage_timings.writer_total_seconds,
            "rust_output_manifest_commit": stage_timings.manifest_commit_seconds,
            "rust_output_finish_total": stage_timings.finish_total_seconds,
            "rust_output_finalization_list_chunk_files": stage_timings.finalization_list_chunk_files_seconds,
            "rust_output_finalization_parquet_writer_properties": stage_timings.finalization_parquet_writer_properties_seconds,
            "rust_output_finalization_parquet_file_create": stage_timings.finalization_parquet_file_create_seconds,
            "rust_output_finalization_parquet_writer_init": stage_timings.finalization_parquet_writer_init_seconds,
            "rust_output_finalization_arrow_file_open": stage_timings.finalization_arrow_file_open_seconds,
            "rust_output_finalization_arrow_reader_init": stage_timings.finalization_arrow_reader_init_seconds,
            "rust_output_finalization_arrow_batch_read": stage_timings.finalization_arrow_batch_read_seconds,
            "rust_output_finalization_read_arrow": stage_timings.finalization_read_arrow_seconds,
            "rust_output_finalization_project_batch": stage_timings.finalization_project_batch_seconds,
            "rust_output_finalization_write_parquet": stage_timings.finalization_write_parquet_seconds,
            "rust_output_finalization_footer_metadata": stage_timings.finalization_footer_metadata_seconds,
            "rust_output_finalization_close_writer": stage_timings.finalization_close_writer_seconds,
            "rust_output_finalization_manifest_update": stage_timings.finalization_manifest_update_seconds,
            "rust_output_finalization_total": stage_timings.finalization_total_seconds,
        },
        "stage_counts": {
            "rust_output_metadata_clone": stage_timings.metadata_clone_count,
            "rust_output_result_buffer_copy": stage_timings.result_buffer_copy_count,
            "rust_output_enqueue": stage_timings.enqueue_count,
            "rust_output_coordinator_flush": stage_timings.coordinator_flush_count,
            "rust_output_writer_record_batch_build": stage_timings.writer_chunk_file_count,
            "rust_output_writer_schema_metadata_build": stage_timings.writer_chunk_file_count,
            "rust_output_writer_metadata_arrays": stage_timings.writer_chunk_count,
            "rust_output_writer_statistic_arrays": stage_timings.writer_chunk_count,
            "rust_output_writer_test_array": stage_timings.writer_chunk_count,
            "rust_output_writer_result_arrays": stage_timings.writer_chunk_count,
            "rust_output_writer_extra_array": stage_timings.writer_chunk_count,
            "rust_output_writer_record_batch_try_new": stage_timings.writer_chunk_count,
            "rust_output_writer_arrow_file_write": stage_timings.writer_chunk_file_count,
            "rust_output_writer_arrow_file_create": stage_timings.writer_chunk_file_count,
            "rust_output_writer_arrow_init": stage_timings.writer_chunk_file_count,
            "rust_output_writer_arrow_batch_write": stage_timings.writer_chunk_count,
            "rust_output_writer_arrow_finish": stage_timings.writer_chunk_file_count,
            "rust_output_writer_arrow_rename": stage_timings.writer_chunk_file_count,
            "rust_output_writer_total": stage_timings.writer_chunk_file_count,
            "rust_output_manifest_commit": stage_timings.manifest_commit_count,
            "rust_output_finish_total": stage_timings.finish_count,
            "rust_output_finalization_list_chunk_files": stage_timings.finalization_count,
            "rust_output_finalization_parquet_writer_properties": stage_timings.finalization_count,
            "rust_output_finalization_parquet_file_create": stage_timings.finalization_count,
            "rust_output_finalization_parquet_writer_init": stage_timings.finalization_count,
            "rust_output_finalization_arrow_file_open": stage_timings.finalization_chunk_file_count,
            "rust_output_finalization_arrow_reader_init": stage_timings.finalization_chunk_file_count,
            "rust_output_finalization_arrow_batch_read": stage_timings.finalization_batch_count,
            "rust_output_finalization_read_arrow": stage_timings.finalization_chunk_file_count,
            "rust_output_finalization_project_batch": stage_timings.finalization_batch_count,
            "rust_output_finalization_write_parquet": stage_timings.finalization_batch_count,
            "rust_output_finalization_footer_metadata": stage_timings.finalization_count,
            "rust_output_finalization_close_writer": stage_timings.finalization_count,
            "rust_output_finalization_manifest_update": stage_timings.finalization_count,
            "rust_output_finalization_total": stage_timings.finalization_count,
        },
        "output_metrics": {
            "writer_chunk_file_count": stage_timings.writer_chunk_file_count,
            "writer_chunk_count": stage_timings.writer_chunk_count,
            "writer_row_count": stage_timings.writer_row_count,
            "writer_arrow_array_memory_bytes": stage_timings.writer_arrow_array_memory_bytes,
            "writer_arrow_file_bytes": stage_timings.writer_arrow_file_bytes,
            "finalization_chunk_file_count": stage_timings.finalization_chunk_file_count,
            "finalization_batch_count": stage_timings.finalization_batch_count,
            "finalization_row_count": stage_timings.finalization_row_count,
            "finalization_arrow_file_bytes": stage_timings.finalization_arrow_file_bytes,
            "finalization_parquet_file_bytes": stage_timings.finalization_parquet_file_bytes,
        },
    });
    let timing_path = run_directory.join(OUTPUT_STAGE_TIMING_FILE_NAME);
    let temporary_timing_path = timing_path.with_extension("json.tmp");
    let timing_text = serde_json::to_string_pretty(&payload).map_err(OutputError::runtime)?;
    std::fs::write(&temporary_timing_path, format!("{timing_text}\n")).map_err(OutputError::runtime)?;
    std::fs::rename(&temporary_timing_path, &timing_path).map_err(OutputError::runtime)
}
