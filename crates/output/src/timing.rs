use std::path::Path;
use std::time::Instant;

use serde_json::json;

use crate::error::OutputError;
use crate::persistence::io::write_json_atomic;
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
    pub(crate) writer_parquet_file_sync_seconds: f64,
    pub(crate) writer_parquet_file_hash_seconds: f64,
    pub(crate) writer_parquet_file_publish_seconds: f64,
    pub(crate) writer_parquet_directory_sync_seconds: f64,
    pub(crate) writer_receipt_publish_seconds: f64,
    pub(crate) writer_total_seconds: f64,
    pub(crate) finish_total_seconds: f64,
    pub(crate) enqueue_count: u64,
    pub(crate) coordinator_flush_count: u64,
    pub(crate) writer_chunk_file_count: u64,
    pub(crate) writer_chunk_count: u64,
    pub(crate) writer_row_count: u64,
    pub(crate) writer_arrow_array_memory_bytes: u64,
    pub(crate) writer_parquet_file_bytes: u64,
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
        self.writer_parquet_file_sync_seconds += timing.parquet_file_sync_seconds;
        self.writer_parquet_file_hash_seconds += timing.parquet_file_hash_seconds;
        self.writer_parquet_file_publish_seconds += timing.parquet_file_publish_seconds;
        self.writer_parquet_directory_sync_seconds += timing.parquet_directory_sync_seconds;
        self.writer_receipt_publish_seconds += timing.receipt_publish_seconds;
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
            "rust_output_writer_parquet_file_sync": stage_timings.writer_parquet_file_sync_seconds,
            "rust_output_writer_parquet_file_hash": stage_timings.writer_parquet_file_hash_seconds,
            "rust_output_writer_parquet_file_publish": stage_timings.writer_parquet_file_publish_seconds,
            "rust_output_writer_parquet_directory_sync": stage_timings.writer_parquet_directory_sync_seconds,
            "rust_output_writer_receipt_publish": stage_timings.writer_receipt_publish_seconds,
            "rust_output_writer_total": stage_timings.writer_total_seconds,
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
            "rust_output_writer_parquet_file_sync": stage_timings.writer_chunk_file_count,
            "rust_output_writer_parquet_file_hash": stage_timings.writer_chunk_file_count,
            "rust_output_writer_parquet_file_publish": stage_timings.writer_chunk_file_count,
            "rust_output_writer_parquet_directory_sync": stage_timings.writer_chunk_file_count,
            "rust_output_writer_receipt_publish": stage_timings.writer_chunk_file_count,
            "rust_output_writer_total": stage_timings.writer_chunk_file_count,
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
    write_json_atomic(&timing_path, &payload)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::Value;

    use super::{
        OUTPUT_STAGE_TIMING_FILE_NAME, OutputStageTimingAccumulator, start_optional_timing, write_stage_timing_snapshot,
    };
    use crate::writer::RegenieStep2ChunkWriteTiming;

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new() -> Self {
            static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);
            let sequence = DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
            let timestamp =
                SystemTime::now().duration_since(UNIX_EPOCH).expect("test time is after Unix epoch").as_nanos();
            let path =
                std::env::temp_dir().join(format!("g-output-timing-{}-{timestamp}-{sequence}", std::process::id()));
            std::fs::create_dir_all(&path).expect("test directory is created");
            Self { path }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn writer_timing() -> RegenieStep2ChunkWriteTiming {
        RegenieStep2ChunkWriteTiming {
            chunk_file_count: 2,
            chunk_count: 3,
            row_count: 4,
            record_batch_build_seconds: 0.1,
            metadata_array_build_seconds: 0.2,
            statistic_array_build_seconds: 0.3,
            result_array_build_seconds: 0.4,
            record_batch_try_new_seconds: 0.5,
            parquet_file_write_seconds: 0.6,
            parquet_file_create_seconds: 0.7,
            parquet_writer_init_seconds: 0.8,
            parquet_batch_write_seconds: 0.9,
            parquet_writer_finish_seconds: 1.0,
            parquet_file_sync_seconds: 1.1,
            parquet_file_hash_seconds: 1.2,
            parquet_file_publish_seconds: 1.3,
            parquet_directory_sync_seconds: 1.4,
            receipt_publish_seconds: 1.5,
            arrow_array_memory_bytes: 5,
            parquet_file_bytes: 6,
            total_seconds: 1.6,
        }
    }

    #[test]
    fn optional_timer_tracks_only_enabled_diagnostics() {
        assert!(start_optional_timing(false).is_none());
        assert!(start_optional_timing(true).is_some());
    }

    #[test]
    fn writer_timing_accumulates_seconds_and_saturates_counters() {
        let mut accumulator = OutputStageTimingAccumulator {
            writer_chunk_file_count: u64::MAX - 1,
            writer_chunk_count: u64::MAX - 2,
            writer_row_count: u64::MAX - 3,
            writer_arrow_array_memory_bytes: u64::MAX - 4,
            writer_parquet_file_bytes: u64::MAX - 5,
            ..OutputStageTimingAccumulator::default()
        };
        accumulator.add_writer_timing(writer_timing());

        for (observed, expected) in [
            (accumulator.writer_record_batch_build_seconds, 0.1),
            (accumulator.writer_metadata_array_build_seconds, 0.2),
            (accumulator.writer_statistic_array_build_seconds, 0.3),
            (accumulator.writer_result_array_build_seconds, 0.4),
            (accumulator.writer_record_batch_try_new_seconds, 0.5),
            (accumulator.writer_parquet_file_write_seconds, 0.6),
            (accumulator.writer_parquet_file_create_seconds, 0.7),
            (accumulator.writer_parquet_init_seconds, 0.8),
            (accumulator.writer_parquet_batch_write_seconds, 0.9),
            (accumulator.writer_parquet_finish_seconds, 1.0),
            (accumulator.writer_parquet_file_sync_seconds, 1.1),
            (accumulator.writer_parquet_file_hash_seconds, 1.2),
            (accumulator.writer_parquet_file_publish_seconds, 1.3),
            (accumulator.writer_parquet_directory_sync_seconds, 1.4),
            (accumulator.writer_receipt_publish_seconds, 1.5),
            (accumulator.writer_total_seconds, 1.6),
        ] {
            assert!((observed - expected).abs() < f64::EPSILON);
        }
        assert_eq!(accumulator.writer_chunk_file_count, u64::MAX);
        assert_eq!(accumulator.writer_chunk_count, u64::MAX);
        assert_eq!(accumulator.writer_row_count, u64::MAX);
        assert_eq!(accumulator.writer_arrow_array_memory_bytes, u64::MAX);
        assert_eq!(accumulator.writer_parquet_file_bytes, u64::MAX);
    }

    #[test]
    fn timing_snapshot_persists_stable_stage_and_metric_names_atomically() {
        let directory = TestDirectory::new();
        let stage_timings = OutputStageTimingAccumulator {
            enqueue_seconds: 0.25,
            finish_total_seconds: 0.75,
            writer_parquet_file_sync_seconds: 0.1,
            writer_parquet_file_hash_seconds: 0.2,
            writer_parquet_file_publish_seconds: 0.3,
            writer_parquet_directory_sync_seconds: 0.4,
            writer_receipt_publish_seconds: 0.5,
            enqueue_count: 2,
            writer_chunk_file_count: 3,
            writer_chunk_count: 4,
            writer_row_count: 5,
            writer_arrow_array_memory_bytes: 6,
            writer_parquet_file_bytes: 7,
            finish_count: 1,
            ..OutputStageTimingAccumulator::default()
        };
        write_stage_timing_snapshot(&directory.path, &stage_timings).expect("timing snapshot writes");

        let timing_path = directory.path.join(OUTPUT_STAGE_TIMING_FILE_NAME);
        let timing_text = std::fs::read_to_string(&timing_path).expect("timing snapshot reads");
        assert!(timing_text.ends_with('\n'));
        assert!(!timing_path.with_extension("json.tmp").exists());
        let payload: Value = serde_json::from_str(&timing_text).expect("timing snapshot is valid JSON");
        let enqueue = payload["stage_totals_seconds"]["rust_output_enqueue"].as_f64().expect("enqueue is float");
        let finish = payload["stage_totals_seconds"]["rust_output_finish_total"].as_f64().expect("finish is float");
        let hash =
            payload["stage_totals_seconds"]["rust_output_writer_parquet_file_hash"].as_f64().expect("hash is float");
        assert!((enqueue - 0.25).abs() < f64::EPSILON);
        assert!((finish - 0.75).abs() < f64::EPSILON);
        assert!((hash - 0.2).abs() < f64::EPSILON);
        assert_eq!(payload["stage_counts"]["rust_output_enqueue"], 2);
        assert_eq!(payload["stage_counts"]["rust_output_writer_total"], 3);
        assert_eq!(payload["stage_counts"]["rust_output_writer_parquet_file_sync"], 3);
        assert_eq!(payload["stage_counts"]["rust_output_writer_parquet_file_hash"], 3);
        assert_eq!(payload["stage_counts"]["rust_output_writer_parquet_file_publish"], 3);
        assert_eq!(payload["stage_counts"]["rust_output_writer_parquet_directory_sync"], 3);
        assert_eq!(payload["stage_counts"]["rust_output_writer_receipt_publish"], 3);
        assert_eq!(payload["stage_counts"]["rust_output_writer_metadata_arrays"], 4);
        assert_eq!(payload["output_metrics"]["writer_row_count"], 5);
        assert_eq!(payload["output_metrics"]["writer_arrow_array_memory_bytes"], 6);
        assert_eq!(payload["output_metrics"]["writer_parquet_file_bytes"], 7);
    }
}
