use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use arrow::array::ArrayRef;
use crossbeam_channel::{Sender, bounded};

use crate::OutputStatisticDtype;
use crate::chunk::NativeChunkHandle;
use crate::error::OutputError;
use crate::finalization;
use crate::manifest;
use crate::timing::{OutputStageTimingAccumulator, start_optional_timing, write_stage_timing_snapshot};
use crate::writer::{OutputFileFormat, RegenieStep2ChunkJob};

mod coordinator;
mod validation;
mod worker_pool;

use coordinator::{OutputCoordinatorJob, run_output_writer_coordinator};
use validation::{validate_column_lengths, validate_statistic_array_type};
use worker_pool::{OutputWriteCompletionTracker, get_output_writer_pool};

#[derive(Clone)]
struct OutputWriterConfig {
    run_directory: PathBuf,
    chunks_directory: PathBuf,
    association_mode: String,
    output_format: OutputFileFormat,
    output_statistic_dtype: OutputStatisticDtype,
    finalize_parquet: bool,
    chunks_per_arrow_file: usize,
    arrow_compression: String,
    parquet_compression: String,
    collect_stage_timings: bool,
}

pub struct OutputWriterSession {
    sender: Mutex<Option<Sender<OutputCoordinatorJob>>>,
    coordinator_handle: Mutex<Option<JoinHandle<()>>>,
    worker_errors: Arc<Mutex<Vec<String>>>,
    worker_commits: Arc<Mutex<Vec<manifest::RunManifestChunkCommit>>>,
    stage_timings: Arc<Mutex<OutputStageTimingAccumulator>>,
    completion_tracker: OutputWriteCompletionTracker,
    config: OutputWriterConfig,
}

#[allow(clippy::missing_errors_doc)]
impl OutputWriterSession {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_directory: String,
        chunks_directory: String,
        association_mode: String,
        writer_thread_count: usize,
        writer_queue_depth: usize,
        output_format: &str,
        output_statistic_dtype: &str,
        finalize_parquet: bool,
        chunks_per_arrow_file: usize,
        arrow_compression: String,
        parquet_compression: String,
        collect_stage_timings: bool,
    ) -> Result<Self, OutputError> {
        if writer_thread_count == 0 {
            return Err(OutputError::InvalidInput("Writer thread count must be at least 1.".to_string()));
        }
        if chunks_per_arrow_file == 0 {
            return Err(OutputError::InvalidInput("Chunks per Arrow file must be at least 1.".to_string()));
        }
        let config = OutputWriterConfig {
            run_directory: PathBuf::from(run_directory),
            chunks_directory: PathBuf::from(chunks_directory),
            association_mode,
            output_format: OutputFileFormat::parse(output_format)?,
            output_statistic_dtype: OutputStatisticDtype::parse(output_statistic_dtype)?,
            finalize_parquet,
            chunks_per_arrow_file,
            arrow_compression,
            parquet_compression,
            collect_stage_timings,
        };
        let (sender, receiver) = bounded(writer_queue_depth.max(1));
        let worker_errors = Arc::new(Mutex::new(Vec::new()));
        let worker_commits = Arc::new(Mutex::new(Vec::new()));
        let stage_timings = Arc::new(Mutex::new(OutputStageTimingAccumulator::default()));
        let writer_pool = get_output_writer_pool(writer_thread_count)?;
        let completion_tracker = OutputWriteCompletionTracker::new();
        let coordinator_worker_errors = Arc::clone(&worker_errors);
        let coordinator_stage_timings = Arc::clone(&stage_timings);
        let coordinator_config = config.clone();
        let coordinator_writer_pool = Arc::clone(&writer_pool);
        let coordinator_worker_commits = Arc::clone(&worker_commits);
        let coordinator_completion_tracker = completion_tracker.clone();
        let coordinator_handle = std::thread::spawn(move || {
            run_output_writer_coordinator(
                receiver,
                coordinator_writer_pool,
                coordinator_config,
                coordinator_worker_errors,
                coordinator_worker_commits,
                coordinator_stage_timings,
                coordinator_completion_tracker,
            );
        });
        Ok(Self {
            sender: Mutex::new(Some(sender)),
            coordinator_handle: Mutex::new(Some(coordinator_handle)),
            worker_errors,
            worker_commits,
            stage_timings,
            completion_tracker,
            config,
        })
    }

    pub fn finish(&self) -> Result<Option<PathBuf>, OutputError> {
        let finish_start_time = start_optional_timing(self.config.collect_stage_timings);
        self.close_writer_sender(OutputCoordinatorJob::Finish)?;
        self.join_coordinator_thread()?;
        self.wait_for_writer_tasks()?;
        self.raise_if_worker_failed()?;
        let chunk_commits = self.take_worker_commits()?;
        let manifest_commit_start_time = start_optional_timing(self.config.collect_stage_timings);
        manifest::record_run_manifest_chunk_commits(&self.config.run_directory, chunk_commits)?;
        if let Some(start_time) = manifest_commit_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.manifest_commit_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.manifest_commit_count += 1;
            })?;
        }
        if self.config.output_format == OutputFileFormat::Regenie {
            let final_regenie_path = self.config.run_directory.join("final.regenie");
            let finalization_timing = finalization::write_final_regenie_from_chunk_files_with_timing(
                &self.config.chunks_directory,
                &final_regenie_path,
                &self.config.association_mode,
                self.config.output_format,
            )?;
            self.record_stage_timing(|stage_timings| stage_timings.add_finalization_timing(finalization_timing))?;
            self.record_finish_timing(finish_start_time)?;
            self.write_stage_timing_snapshot()?;
            return Ok(Some(final_regenie_path));
        }
        if !self.config.finalize_parquet {
            self.record_finish_timing(finish_start_time)?;
            self.write_stage_timing_snapshot()?;
            return Ok(None);
        }
        let final_parquet_path = self.config.run_directory.join("final.parquet");
        let finalization_timing = finalization::write_final_parquet_from_chunk_files_with_timing_for_dtype(
            &self.config.chunks_directory,
            &final_parquet_path,
            &self.config.association_mode,
            self.config.output_format,
            self.config.output_statistic_dtype,
        )?;
        self.record_stage_timing(|stage_timings| stage_timings.add_finalization_timing(finalization_timing))?;
        self.record_finish_timing(finish_start_time)?;
        self.write_stage_timing_snapshot()?;
        Ok(Some(final_parquet_path))
    }

    pub fn finish_interrupted(&self, signal_name: &str) -> Result<(), OutputError> {
        let finish_start_time = start_optional_timing(self.config.collect_stage_timings);
        self.close_writer_sender(OutputCoordinatorJob::Finish)?;
        self.join_coordinator_thread()?;
        self.wait_for_writer_tasks()?;
        self.raise_if_worker_failed()?;
        let chunk_commits = self.take_worker_commits()?;
        let manifest_commit_start_time = start_optional_timing(self.config.collect_stage_timings);
        manifest::record_run_manifest_chunk_commits(&self.config.run_directory, chunk_commits)?;
        if let Some(start_time) = manifest_commit_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.manifest_commit_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.manifest_commit_count += 1;
            })?;
        }
        manifest::mark_run_manifest_interrupted(&self.config.run_directory, signal_name)?;
        self.record_finish_timing(finish_start_time)?;
        self.write_stage_timing_snapshot()
    }

    pub fn abort(&self) -> Result<(), OutputError> {
        self.close_writer_sender(OutputCoordinatorJob::Abort)?;
        self.join_coordinator_thread()?;
        self.wait_for_writer_tasks()?;
        Ok(())
    }

    fn take_worker_commits(&self) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
        let mut worker_commits = self
            .worker_commits
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer commit lock was poisoned.".to_string()))?;
        Ok(std::mem::take(&mut *worker_commits))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_regenie2_native_chunk_handle_arrays(
        &self,
        chunk_handle: NativeChunkHandle,
        beta: ArrayRef,
        standard_error: ArrayRef,
        chi_squared: ArrayRef,
        log10_p_value: ArrayRef,
        extra_code: Option<ArrayRef>,
    ) -> Result<(), OutputError> {
        if self.config.association_mode != "regenie2_linear" && self.config.association_mode != "regenie2_binary" {
            return Err(OutputError::InvalidInput(
                "Rust output backend only supports REGENIE step 2 quantitative and binary output.".to_string(),
            ));
        }
        let row_count = chunk_handle.row_count();
        let observed_lengths = [
            chunk_handle.metadata.chromosome.len(),
            chunk_handle.metadata.variant_identifier.len(),
            chunk_handle.metadata.allele_two.len(),
            chunk_handle.metadata.allele_one.len(),
            chunk_handle.stats.allele_one_frequency.len(),
            chunk_handle.stats.info_score.len(),
            chunk_handle.stats.observation_count.len(),
            beta.len(),
            standard_error.len(),
            chi_squared.len(),
            log10_p_value.len(),
        ];
        validate_column_lengths(row_count, observed_lengths.as_slice())?;
        validate_statistic_array_type("BETA", &beta, self.config.output_statistic_dtype)?;
        validate_statistic_array_type("SE", &standard_error, self.config.output_statistic_dtype)?;
        validate_statistic_array_type("CHISQ", &chi_squared, self.config.output_statistic_dtype)?;
        validate_statistic_array_type("LOG10P", &log10_p_value, self.config.output_statistic_dtype)?;
        if let Some(extra_code_values) = extra_code.as_ref() {
            validate_column_lengths(row_count, &[extra_code_values.len()])?;
        }
        let job = RegenieStep2ChunkJob {
            chunk_handle,
            beta,
            se: standard_error,
            chisq: chi_squared,
            log10p: log10_p_value,
            extra_code,
        };
        self.raise_if_worker_failed()?;
        let sender_guard = self
            .sender
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer sender lock was poisoned.".to_string()))?;
        let sender = sender_guard
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Rust output writer session is already closed.".to_string()))?;
        let enqueue_start_time = start_optional_timing(self.config.collect_stage_timings);
        sender.send(OutputCoordinatorJob::RegenieStep2(Box::new(job))).map_err(OutputError::runtime)?;
        if let Some(start_time) = enqueue_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.enqueue_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.enqueue_count += 1;
            })?;
        }
        Ok(())
    }

    fn record_finish_timing(&self, finish_start_time: Option<Instant>) -> Result<(), OutputError> {
        if let Some(start_time) = finish_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.finish_total_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.finish_count += 1;
            })?;
        }
        Ok(())
    }

    fn record_stage_timing(
        &self,
        update_stage_timings: impl FnOnce(&mut OutputStageTimingAccumulator),
    ) -> Result<(), OutputError> {
        if !self.config.collect_stage_timings {
            return Ok(());
        }
        let mut stage_timings = self
            .stage_timings
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer stage timing lock was poisoned.".to_string()))?;
        update_stage_timings(&mut stage_timings);
        Ok(())
    }

    fn write_stage_timing_snapshot(&self) -> Result<(), OutputError> {
        if !self.config.collect_stage_timings {
            return Ok(());
        }
        let stage_timings = self
            .stage_timings
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer stage timing lock was poisoned.".to_string()))?;
        write_stage_timing_snapshot(&self.config.run_directory, &stage_timings)
    }

    fn raise_if_worker_failed(&self) -> Result<(), OutputError> {
        let worker_errors = self
            .worker_errors
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer error lock was poisoned.".to_string()))?;
        if let Some(first_error) = worker_errors.first() {
            return Err(OutputError::Runtime(first_error.clone()));
        }
        Ok(())
    }

    fn close_writer_sender(&self, close_job: OutputCoordinatorJob) -> Result<(), OutputError> {
        let mut sender_guard = self
            .sender
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer sender lock was poisoned.".to_string()))?;
        if let Some(active_sender) = sender_guard.take() {
            active_sender.send(close_job).map_err(OutputError::runtime)?;
        }
        Ok(())
    }

    fn join_coordinator_thread(&self) -> Result<(), OutputError> {
        let mut coordinator_handle_guard = self.coordinator_handle.lock().map_err(|_| {
            OutputError::Runtime("Rust output writer coordinator handle lock was poisoned.".to_string())
        })?;
        if let Some(handle) = coordinator_handle_guard.take() {
            handle
                .join()
                .map_err(|_| OutputError::Runtime("Rust output writer coordinator thread panicked.".to_string()))?;
        }
        Ok(())
    }

    fn wait_for_writer_tasks(&self) -> Result<(), OutputError> {
        self.completion_tracker.wait()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finalization::RegenieStep2FinalizationTiming;

    #[test]
    fn stage_timing_accumulator_records_finalization_timing() {
        let mut stage_timings = OutputStageTimingAccumulator::default();

        stage_timings.add_finalization_timing(RegenieStep2FinalizationTiming {
            chunk_file_count: 2,
            batch_count: 3,
            row_count: 5,
            list_chunk_files_seconds: 0.1,
            parquet_writer_properties_seconds: 0.0,
            parquet_file_create_seconds: 0.0,
            parquet_writer_init_seconds: 0.0,
            arrow_file_open_seconds: 0.0,
            arrow_reader_init_seconds: 0.0,
            arrow_batch_read_seconds: 0.0,
            read_arrow_seconds: 0.2,
            project_batch_seconds: 0.3,
            write_parquet_seconds: 0.4,
            footer_metadata_seconds: 0.5,
            close_writer_seconds: 0.6,
            manifest_update_seconds: 0.7,
            arrow_file_bytes: 0,
            parquet_file_bytes: 0,
            total_seconds: 1.8,
        });

        assert_eq!(stage_timings.finalization_chunk_file_count, 2);
        assert_eq!(stage_timings.finalization_batch_count, 3);
        assert_eq!(stage_timings.finalization_row_count, 5);
        assert_eq!(stage_timings.finalization_count, 1);
        assert!(stage_timings.finalization_total_seconds > 1.0);
    }

    #[test]
    fn output_writer_pool_reuses_one_process_pool_at_max_requested_cap() {
        let first_pool = get_output_writer_pool(1).expect("first pool should open");
        let second_pool = get_output_writer_pool(2).expect("second pool should reuse and grow first pool");
        let third_pool = get_output_writer_pool(1).expect("third pool should reuse existing pool");

        assert!(std::sync::Arc::ptr_eq(&first_pool, &second_pool));
        assert!(std::sync::Arc::ptr_eq(&second_pool, &third_pool));
        assert!(third_pool.current_worker_count() >= 2);
    }
}
