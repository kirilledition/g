use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow::array::{ArrayRef, Float32Array, Float64Array};

use crate::chunk::NativeChunkHandle;
use crate::error::OutputError;
use crate::manifest;
use crate::timing::{OutputStageTimingAccumulator, start_optional_timing, write_stage_timing_snapshot};
use crate::writer::{RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, build_part_file_name};

use super::worker_pool::{OutputWriteCompletionTracker, OutputWriterPool};

pub(super) struct OutputWriterConfig {
    pub(super) run_directory: PathBuf,
    pub(super) parts_directory: PathBuf,
    pub(super) output_statistic_dtype: g_plan::FloatingPointDtype,
    pub(super) chunks_per_parquet_file: usize,
    pub(super) parquet_compression: g_plan::ParquetCompression,
    pub(super) collect_stage_timings: bool,
}

impl OutputWriterConfig {
    pub(crate) fn from_plan(
        run_directory: PathBuf,
        parts_directory: PathBuf,
        output_plan: &g_plan::OutputPlan,
        collect_stage_timings: bool,
    ) -> Result<Self, OutputError> {
        Ok(Self {
            run_directory,
            parts_directory,
            output_statistic_dtype: output_plan.output_statistic_dtype,
            chunks_per_parquet_file: usize::try_from(output_plan.chunks_per_parquet_file)
                .map_err(OutputError::runtime)?,
            parquet_compression: output_plan.parquet_compression,
            collect_stage_timings,
        })
    }
}

pub struct OutputWriterSession {
    pending_chunks: Mutex<Option<Vec<RegenieStep2ChunkJob>>>,
    writer_pool: Arc<OutputWriterPool>,
    worker_error: Arc<Mutex<Option<String>>>,
    worker_commits: Arc<Mutex<Vec<manifest::RunManifestChunkCommit>>>,
    stage_timings: Arc<Mutex<OutputStageTimingAccumulator>>,
    completion_tracker: OutputWriteCompletionTracker,
    config: Arc<OutputWriterConfig>,
}

#[allow(clippy::missing_errors_doc)]
impl OutputWriterSession {
    fn new_with_writer_pool(
        config: OutputWriterConfig,
        writer_pool: Arc<OutputWriterPool>,
    ) -> Result<Self, OutputError> {
        if config.chunks_per_parquet_file == 0 {
            return Err(OutputError::InvalidInput("Chunks per Parquet file must be at least 1.".to_string()));
        }
        let worker_error = Arc::new(Mutex::new(None));
        let worker_commits = Arc::new(Mutex::new(Vec::new()));
        let stage_timings = Arc::new(Mutex::new(OutputStageTimingAccumulator::default()));
        let completion_tracker = OutputWriteCompletionTracker::new();
        Ok(Self {
            pending_chunks: Mutex::new(Some(Vec::with_capacity(config.chunks_per_parquet_file))),
            writer_pool,
            worker_error,
            worker_commits,
            stage_timings,
            completion_tracker,
            config: Arc::new(config),
        })
    }

    pub fn finish(&self) -> Result<(), OutputError> {
        let finish_start_time = start_optional_timing(self.config.collect_stage_timings);
        self.flush_and_commit()?;
        manifest::mark_run_manifest_completed(&self.config.run_directory)?;
        self.record_finish_timing(finish_start_time)?;
        self.write_stage_timing_snapshot()
    }

    pub fn finish_interrupted(&self, signal_name: &str) -> Result<(), OutputError> {
        let finish_start_time = start_optional_timing(self.config.collect_stage_timings);
        self.flush_and_commit()?;
        manifest::mark_run_manifest_interrupted(&self.config.run_directory, signal_name)?;
        self.record_finish_timing(finish_start_time)?;
        self.write_stage_timing_snapshot()
    }

    pub fn abort(&self) -> Result<(), OutputError> {
        self.close_and_discard_pending_chunks()?;
        self.completion_tracker.wait()?;
        Ok(())
    }

    fn take_worker_commits(&self) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
        let mut worker_commits = self
            .worker_commits
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer commit lock was poisoned.".to_string()))?;
        Ok(std::mem::take(&mut *worker_commits))
    }

    fn flush_and_commit(&self) -> Result<(), OutputError> {
        self.close_and_flush_pending_chunks()?;
        self.completion_tracker.wait()?;
        self.raise_if_worker_failed()?;
        let manifest_commit_start_time = start_optional_timing(self.config.collect_stage_timings);
        manifest::record_run_manifest_chunk_commits(&self.config.run_directory, self.take_worker_commits()?)?;
        if let Some(start_time) = manifest_commit_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.manifest_commit_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.manifest_commit_count = stage_timings.manifest_commit_count.saturating_add(1);
            })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_regenie2_native_chunk_handle_arrays(
        &self,
        chunk_handle: NativeChunkHandle,
        beta: ArrayRef,
        standard_error: ArrayRef,
        chi_squared: ArrayRef,
        log10_p_value: ArrayRef,
        correction_code: Option<ArrayRef>,
    ) -> Result<(), OutputError> {
        let row_count = chunk_handle.row_count();
        validate_column_lengths(row_count, &chunk_handle.writer_arrays().column_lengths())?;
        validate_column_lengths(
            row_count,
            &[beta.len(), standard_error.len(), chi_squared.len(), log10_p_value.len()],
        )?;
        validate_statistic_array_type("BETA", &beta, self.config.output_statistic_dtype)?;
        validate_statistic_array_type("SE", &standard_error, self.config.output_statistic_dtype)?;
        validate_statistic_array_type("CHISQ", &chi_squared, self.config.output_statistic_dtype)?;
        validate_statistic_array_type("LOG10P", &log10_p_value, self.config.output_statistic_dtype)?;
        if let Some(correction_code_values) = correction_code.as_ref() {
            validate_column_lengths(row_count, &[correction_code_values.len()])?;
        }
        let job = RegenieStep2ChunkJob {
            chunk_handle,
            beta,
            se: standard_error,
            chisq: chi_squared,
            log10p: log10_p_value,
            correction_code,
        };
        self.raise_if_worker_failed()?;
        let enqueue_start_time = start_optional_timing(self.config.collect_stage_timings);
        let write_batch = {
            let mut pending_chunks_guard = self
                .pending_chunks
                .lock()
                .map_err(|_| OutputError::Runtime("Rust output writer pending chunk lock was poisoned.".to_string()))?;
            let pending_chunks = pending_chunks_guard
                .as_mut()
                .ok_or_else(|| OutputError::Runtime("Rust output writer session is already closed.".to_string()))?;
            pending_chunks.push(job);
            (pending_chunks.len() >= self.config.chunks_per_parquet_file)
                .then(|| take_pending_chunk_write_batch(pending_chunks))
        };
        if let Some(write_batch) = write_batch {
            self.enqueue_write_batch(write_batch)?;
        }
        if let Some(start_time) = enqueue_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.enqueue_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.enqueue_count = stage_timings.enqueue_count.saturating_add(1);
            })?;
        }
        Ok(())
    }

    fn close_and_flush_pending_chunks(&self) -> Result<(), OutputError> {
        let write_batch = {
            let mut pending_chunks_guard = self
                .pending_chunks
                .lock()
                .map_err(|_| OutputError::Runtime("Rust output writer pending chunk lock was poisoned.".to_string()))?;
            pending_chunks_guard.take().and_then(|mut pending_chunks| {
                (!pending_chunks.is_empty()).then(|| take_pending_chunk_write_batch(&mut pending_chunks))
            })
        };
        if let Some(write_batch) = write_batch {
            self.enqueue_write_batch(write_batch)?;
        }
        Ok(())
    }

    fn close_and_discard_pending_chunks(&self) -> Result<(), OutputError> {
        let mut pending_chunks_guard = self
            .pending_chunks
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer pending chunk lock was poisoned.".to_string()))?;
        pending_chunks_guard.take();
        Ok(())
    }

    fn enqueue_write_batch(&self, write_batch: RegenieStep2ChunkWriteBatch) -> Result<(), OutputError> {
        let flush_start_time = start_optional_timing(self.config.collect_stage_timings);
        self.writer_pool.enqueue_regenie_step2(
            write_batch,
            &self.config,
            &self.worker_error,
            &self.worker_commits,
            &self.stage_timings,
            &self.completion_tracker,
        )?;
        if let Some(start_time) = flush_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.coordinator_flush_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.coordinator_flush_count = stage_timings.coordinator_flush_count.saturating_add(1);
            })?;
        }
        Ok(())
    }

    fn record_finish_timing(&self, finish_start_time: Option<Instant>) -> Result<(), OutputError> {
        if let Some(start_time) = finish_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.finish_total_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.finish_count = stage_timings.finish_count.saturating_add(1);
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
        let worker_error = self
            .worker_error
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer error lock was poisoned.".to_string()))?;
        if let Some(error) = worker_error.as_ref() {
            return Err(OutputError::Runtime(error.clone()));
        }
        Ok(())
    }
}

fn take_pending_chunk_write_batch(pending_chunks: &mut Vec<RegenieStep2ChunkJob>) -> RegenieStep2ChunkWriteBatch {
    let first_chunk_identifier = pending_chunks.first().map_or(0, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    let last_chunk_identifier =
        pending_chunks.last().map_or(first_chunk_identifier, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    let chunk_file_name = build_part_file_name(first_chunk_identifier, last_chunk_identifier);
    RegenieStep2ChunkWriteBatch { chunk_file_name, chunks: std::mem::take(pending_chunks) }
}

/// Create one native output writer session per run/parts directory pair.
///
/// # Errors
///
/// Returns an error when the directory vector lengths differ, writer settings
/// are invalid, or a writer pool cannot be created.
pub fn create_output_writer_sessions(
    run_directories: Vec<PathBuf>,
    parts_directories: Vec<PathBuf>,
    output_plan: &g_plan::OutputPlan,
    collect_stage_timings: bool,
) -> Result<Vec<OutputWriterSession>, OutputError> {
    if run_directories.len() != parts_directories.len() {
        return Err(OutputError::InvalidInput(format!(
            "Output writer run directory count ({}) does not match parts directory count ({}).",
            run_directories.len(),
            parts_directories.len()
        )));
    }
    if run_directories.is_empty() {
        return Ok(Vec::new());
    }
    let configs = run_directories
        .into_iter()
        .zip(parts_directories)
        .map(|(run_directory, parts_directory)| {
            OutputWriterConfig::from_plan(run_directory, parts_directory, output_plan, collect_stage_timings)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let writer_thread_count = usize::try_from(output_plan.writer_thread_count).map_err(OutputError::runtime)?;
    let writer_queue_depth = usize::try_from(output_plan.writer_queue_depth).map_err(OutputError::runtime)?;
    create_output_writer_sessions_from_configs(configs, writer_thread_count, writer_queue_depth)
}

pub(crate) fn create_output_writer_sessions_from_configs(
    configs: Vec<OutputWriterConfig>,
    writer_thread_count: usize,
    writer_queue_depth: usize,
) -> Result<Vec<OutputWriterSession>, OutputError> {
    if configs.is_empty() {
        return Ok(Vec::new());
    }
    let writer_pool = OutputWriterPool::new(writer_thread_count, writer_queue_depth)?;
    configs
        .into_iter()
        .map(|config| OutputWriterSession::new_with_writer_pool(config, Arc::clone(&writer_pool)))
        .collect()
}

/// Finish output writer sessions, optionally in bounded parallel batches.
///
/// # Errors
///
/// Returns an error when a session cannot be closed, a writer task failed, a
/// manifest commit fails or a finish thread panics.
pub fn finish_output_writer_sessions(
    writer_sessions: &[&OutputWriterSession],
    thread_count: usize,
) -> Result<(), OutputError> {
    if thread_count <= 1 {
        for writer_session in writer_sessions {
            writer_session.finish()?;
        }
        return Ok(());
    }
    for writer_session_batch in writer_sessions.chunks(thread_count) {
        finish_output_writer_session_batch(writer_session_batch)?;
    }
    Ok(())
}

/// Flush interrupted output writer sessions and mark their manifests interrupted.
///
/// # Errors
///
/// Returns an error when a session cannot be closed, a writer task failed, a
/// manifest update fails, or an interrupted-finish thread panics.
pub fn finish_interrupted_output_writer_sessions(
    writer_sessions: &[&OutputWriterSession],
    thread_count: usize,
    signal_name: &str,
) -> Result<(), OutputError> {
    if thread_count <= 1 {
        for writer_session in writer_sessions {
            writer_session.finish_interrupted(signal_name)?;
        }
        return Ok(());
    }
    for writer_session_batch in writer_sessions.chunks(thread_count) {
        finish_interrupted_output_writer_session_batch(writer_session_batch, signal_name)?;
    }
    Ok(())
}

fn finish_output_writer_session_batch(writer_sessions: &[&OutputWriterSession]) -> Result<(), OutputError> {
    std::thread::scope(|scope| {
        let writer_handles = writer_sessions
            .iter()
            .map(|writer_session| scope.spawn(move || writer_session.finish()))
            .collect::<Vec<_>>();
        for writer_handle in writer_handles {
            writer_handle
                .join()
                .map_err(|_| OutputError::Runtime("Output writer finish thread panicked.".to_string()))??;
        }
        Ok(())
    })
}
fn finish_interrupted_output_writer_session_batch(
    writer_sessions: &[&OutputWriterSession],
    signal_name: &str,
) -> Result<(), OutputError> {
    std::thread::scope(|scope| {
        let writer_handles = writer_sessions
            .iter()
            .map(|writer_session| scope.spawn(move || writer_session.finish_interrupted(signal_name)))
            .collect::<Vec<_>>();
        for writer_handle in writer_handles {
            writer_handle
                .join()
                .map_err(|_| OutputError::Runtime("Interrupted output writer finish thread panicked.".to_string()))??;
        }
        Ok(())
    })
}

pub(super) fn validate_column_lengths(
    expected_row_count: usize,
    observed_lengths: &[usize],
) -> Result<(), OutputError> {
    if observed_lengths.iter().all(|observed_length| *observed_length == expected_row_count) {
        return Ok(());
    }
    Err(OutputError::InvalidInput(
        "Rust output writer batch column lengths do not all match the expected row count.".to_string(),
    ))
}

pub(super) fn validate_statistic_array_type(
    column_name: &str,
    array: &ArrayRef,
    output_statistic_dtype: g_plan::FloatingPointDtype,
) -> Result<(), OutputError> {
    let type_matches = match output_statistic_dtype {
        g_plan::FloatingPointDtype::Float32 => array.as_any().is::<Float32Array>(),
        g_plan::FloatingPointDtype::Float64 => array.as_any().is::<Float64Array>(),
    };
    if type_matches {
        return Ok(());
    }
    Err(OutputError::InvalidInput(format!(
        "Rust output writer column {column_name} must be {} for the configured output statistic dtype.",
        output_statistic_dtype.as_str(),
    )))
}
