use crate::finalization::RegenieStep2FinalizationTiming;
use crate::writer::RegenieStep2ChunkWriteTiming;

#[derive(Default)]
pub(crate) struct OutputStageTimingAccumulator {
    pub(crate) metadata_clone_seconds: f64,
    pub(crate) result_buffer_copy_seconds: f64,
    pub(crate) enqueue_seconds: f64,
    pub(crate) coordinator_flush_seconds: f64,
    pub(crate) writer_record_batch_build_seconds: f64,
    pub(crate) writer_schema_metadata_build_seconds: f64,
    pub(crate) writer_metadata_array_build_seconds: f64,
    pub(crate) writer_statistic_array_build_seconds: f64,
    pub(crate) writer_test_array_build_seconds: f64,
    pub(crate) writer_result_array_build_seconds: f64,
    pub(crate) writer_extra_array_build_seconds: f64,
    pub(crate) writer_record_batch_try_new_seconds: f64,
    pub(crate) writer_arrow_file_write_seconds: f64,
    pub(crate) writer_arrow_file_create_seconds: f64,
    pub(crate) writer_arrow_init_seconds: f64,
    pub(crate) writer_arrow_batch_write_seconds: f64,
    pub(crate) writer_arrow_finish_seconds: f64,
    pub(crate) writer_arrow_rename_seconds: f64,
    pub(crate) writer_total_seconds: f64,
    pub(crate) manifest_commit_seconds: f64,
    pub(crate) finish_total_seconds: f64,
    pub(crate) finalization_list_chunk_files_seconds: f64,
    pub(crate) finalization_parquet_writer_properties_seconds: f64,
    pub(crate) finalization_parquet_file_create_seconds: f64,
    pub(crate) finalization_parquet_writer_init_seconds: f64,
    pub(crate) finalization_arrow_file_open_seconds: f64,
    pub(crate) finalization_arrow_reader_init_seconds: f64,
    pub(crate) finalization_arrow_batch_read_seconds: f64,
    pub(crate) finalization_read_arrow_seconds: f64,
    pub(crate) finalization_project_batch_seconds: f64,
    pub(crate) finalization_write_parquet_seconds: f64,
    pub(crate) finalization_footer_metadata_seconds: f64,
    pub(crate) finalization_close_writer_seconds: f64,
    pub(crate) finalization_manifest_update_seconds: f64,
    pub(crate) finalization_total_seconds: f64,
    pub(crate) metadata_clone_count: u64,
    pub(crate) result_buffer_copy_count: u64,
    pub(crate) enqueue_count: u64,
    pub(crate) coordinator_flush_count: u64,
    pub(crate) writer_chunk_file_count: u64,
    pub(crate) writer_chunk_count: u64,
    pub(crate) writer_row_count: u64,
    pub(crate) writer_arrow_array_memory_bytes: u64,
    pub(crate) writer_arrow_file_bytes: u64,
    pub(crate) manifest_commit_count: u64,
    pub(crate) finish_count: u64,
    pub(crate) finalization_chunk_file_count: u64,
    pub(crate) finalization_batch_count: u64,
    pub(crate) finalization_row_count: u64,
    pub(crate) finalization_arrow_file_bytes: u64,
    pub(crate) finalization_parquet_file_bytes: u64,
    pub(crate) finalization_count: u64,
}

impl OutputStageTimingAccumulator {
    pub(crate) fn add_writer_timing(&mut self, timing: RegenieStep2ChunkWriteTiming) {
        self.writer_record_batch_build_seconds += timing.record_batch_build_seconds;
        self.writer_schema_metadata_build_seconds += timing.schema_metadata_build_seconds;
        self.writer_metadata_array_build_seconds += timing.metadata_array_build_seconds;
        self.writer_statistic_array_build_seconds += timing.statistic_array_build_seconds;
        self.writer_test_array_build_seconds += timing.test_array_build_seconds;
        self.writer_result_array_build_seconds += timing.result_array_build_seconds;
        self.writer_extra_array_build_seconds += timing.extra_array_build_seconds;
        self.writer_record_batch_try_new_seconds += timing.record_batch_try_new_seconds;
        self.writer_arrow_file_write_seconds += timing.arrow_file_write_seconds;
        self.writer_arrow_file_create_seconds += timing.arrow_file_create_seconds;
        self.writer_arrow_init_seconds += timing.arrow_writer_init_seconds;
        self.writer_arrow_batch_write_seconds += timing.arrow_batch_write_seconds;
        self.writer_arrow_finish_seconds += timing.arrow_writer_finish_seconds;
        self.writer_arrow_rename_seconds += timing.arrow_file_rename_seconds;
        self.writer_total_seconds += timing.total_seconds;
        self.writer_chunk_file_count = self.writer_chunk_file_count.saturating_add(timing.chunk_file_count);
        self.writer_chunk_count = self.writer_chunk_count.saturating_add(timing.chunk_count);
        self.writer_row_count = self.writer_row_count.saturating_add(timing.row_count);
        self.writer_arrow_array_memory_bytes =
            self.writer_arrow_array_memory_bytes.saturating_add(timing.arrow_array_memory_bytes);
        self.writer_arrow_file_bytes = self.writer_arrow_file_bytes.saturating_add(timing.arrow_file_bytes);
    }

    pub(crate) fn add_finalization_timing(&mut self, timing: RegenieStep2FinalizationTiming) {
        self.finalization_list_chunk_files_seconds += timing.list_chunk_files_seconds;
        self.finalization_parquet_writer_properties_seconds += timing.parquet_writer_properties_seconds;
        self.finalization_parquet_file_create_seconds += timing.parquet_file_create_seconds;
        self.finalization_parquet_writer_init_seconds += timing.parquet_writer_init_seconds;
        self.finalization_arrow_file_open_seconds += timing.arrow_file_open_seconds;
        self.finalization_arrow_reader_init_seconds += timing.arrow_reader_init_seconds;
        self.finalization_arrow_batch_read_seconds += timing.arrow_batch_read_seconds;
        self.finalization_read_arrow_seconds += timing.read_arrow_seconds;
        self.finalization_project_batch_seconds += timing.project_batch_seconds;
        self.finalization_write_parquet_seconds += timing.write_parquet_seconds;
        self.finalization_footer_metadata_seconds += timing.footer_metadata_seconds;
        self.finalization_close_writer_seconds += timing.close_writer_seconds;
        self.finalization_manifest_update_seconds += timing.manifest_update_seconds;
        self.finalization_total_seconds += timing.total_seconds;
        self.finalization_chunk_file_count = self.finalization_chunk_file_count.saturating_add(timing.chunk_file_count);
        self.finalization_batch_count = self.finalization_batch_count.saturating_add(timing.batch_count);
        self.finalization_row_count = self.finalization_row_count.saturating_add(timing.row_count);
        self.finalization_arrow_file_bytes = self.finalization_arrow_file_bytes.saturating_add(timing.arrow_file_bytes);
        self.finalization_parquet_file_bytes =
            self.finalization_parquet_file_bytes.saturating_add(timing.parquet_file_bytes);
        self.finalization_count = self.finalization_count.saturating_add(1);
    }
}
