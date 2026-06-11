use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Instant;

use arrow::array::{ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use serde_json::json;

use crate::genotype::common::{ChunkStats as NativeChunkStats, VariantMetadataColumns};
use crate::output::OutputStatisticDtype;
use crate::output::finalization;
use crate::output::manifest;
use crate::output::writer::{
    OutputFileFormat, OutputWriterError, RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch,
    RegenieStep2ChunkWriteTiming, build_output_file_name, write_regenie_step2_chunk_job,
};

const OUTPUT_STAGE_TIMING_FILE_NAME: &str = "output_stage_timings.json";

#[derive(Clone)]
pub(crate) struct NativeChunkHandle {
    pub(crate) metadata: Arc<VariantMetadataColumns>,
    pub(crate) stats: Arc<NativeChunkStats>,
    pub(crate) chunk_identifier: i64,
    writer_arrays: Arc<OnceLock<NativeChunkWriterArrays>>,
}

#[derive(Clone)]
pub(crate) struct NativeChunkWriterArrays {
    pub(crate) chromosome: ArrayRef,
    pub(crate) position: ArrayRef,
    pub(crate) variant_identifier: ArrayRef,
    pub(crate) allele_two: ArrayRef,
    pub(crate) allele_one: ArrayRef,
    pub(crate) allele_one_frequency: ArrayRef,
    pub(crate) info_score: ArrayRef,
    pub(crate) observation_count: ArrayRef,
}

impl NativeChunkWriterArrays {
    fn from_chunk_sources(metadata: &VariantMetadataColumns, stats: &NativeChunkStats) -> Self {
        Self {
            chromosome: Arc::new(StringArray::from(metadata.chromosome.clone())),
            position: Arc::new(Int64Array::from(metadata.position.clone())),
            variant_identifier: Arc::new(StringArray::from(metadata.variant_identifier.clone())),
            allele_two: Arc::new(StringArray::from(metadata.allele_two.clone())),
            allele_one: Arc::new(StringArray::from(metadata.allele_one.clone())),
            allele_one_frequency: Arc::new(Float32Array::from(stats.allele_one_frequency.clone())),
            info_score: Arc::new(Float32Array::from(stats.info_score.clone())),
            observation_count: Arc::new(Int32Array::from(stats.observation_count.clone())),
        }
    }
}

impl NativeChunkHandle {
    pub(crate) fn new(
        metadata: Arc<VariantMetadataColumns>,
        stats: Arc<NativeChunkStats>,
        chunk_identifier: i64,
    ) -> Self {
        Self { metadata, stats, chunk_identifier, writer_arrays: Arc::new(OnceLock::new()) }
    }

    pub(crate) fn row_count(&self) -> usize {
        self.metadata.position.len()
    }

    pub(crate) fn variant_start_index(&self) -> i64 {
        self.chunk_identifier
    }

    pub(crate) fn variant_stop_index(&self) -> Result<i64, OutputWriterError> {
        let row_count = i64::try_from(self.row_count()).map_err(|_| {
            OutputWriterError::InvalidInput("Rust output writer row count does not fit into int64.".to_string())
        })?;
        self.chunk_identifier.checked_add(row_count).ok_or_else(|| {
            OutputWriterError::InvalidInput(
                "Rust output writer variant stop index does not fit into int64.".to_string(),
            )
        })
    }

    pub(crate) fn writer_arrays(&self) -> &NativeChunkWriterArrays {
        self.writer_arrays.get_or_init(|| NativeChunkWriterArrays::from_chunk_sources(&self.metadata, &self.stats))
    }
}

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

#[derive(Default)]
struct OutputStageTimingAccumulator {
    metadata_clone_seconds: f64,
    result_buffer_copy_seconds: f64,
    enqueue_seconds: f64,
    coordinator_flush_seconds: f64,
    writer_record_batch_build_seconds: f64,
    writer_schema_metadata_build_seconds: f64,
    writer_metadata_array_build_seconds: f64,
    writer_statistic_array_build_seconds: f64,
    writer_test_array_build_seconds: f64,
    writer_result_array_build_seconds: f64,
    writer_extra_array_build_seconds: f64,
    writer_record_batch_try_new_seconds: f64,
    writer_arrow_file_write_seconds: f64,
    writer_arrow_file_create_seconds: f64,
    writer_arrow_init_seconds: f64,
    writer_arrow_batch_write_seconds: f64,
    writer_arrow_finish_seconds: f64,
    writer_arrow_rename_seconds: f64,
    writer_total_seconds: f64,
    manifest_commit_seconds: f64,
    finish_total_seconds: f64,
    finalization_list_chunk_files_seconds: f64,
    finalization_parquet_writer_properties_seconds: f64,
    finalization_parquet_file_create_seconds: f64,
    finalization_parquet_writer_init_seconds: f64,
    finalization_arrow_file_open_seconds: f64,
    finalization_arrow_reader_init_seconds: f64,
    finalization_arrow_batch_read_seconds: f64,
    finalization_read_arrow_seconds: f64,
    finalization_project_batch_seconds: f64,
    finalization_write_parquet_seconds: f64,
    finalization_footer_metadata_seconds: f64,
    finalization_close_writer_seconds: f64,
    finalization_manifest_update_seconds: f64,
    finalization_total_seconds: f64,
    metadata_clone_count: u64,
    result_buffer_copy_count: u64,
    enqueue_count: u64,
    coordinator_flush_count: u64,
    writer_chunk_file_count: u64,
    writer_chunk_count: u64,
    writer_row_count: u64,
    writer_arrow_array_memory_bytes: u64,
    writer_arrow_file_bytes: u64,
    manifest_commit_count: u64,
    finish_count: u64,
    finalization_chunk_file_count: u64,
    finalization_batch_count: u64,
    finalization_row_count: u64,
    finalization_arrow_file_bytes: u64,
    finalization_parquet_file_bytes: u64,
    finalization_count: u64,
}

impl OutputStageTimingAccumulator {
    fn add_writer_timing(&mut self, timing: RegenieStep2ChunkWriteTiming) {
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
        self.writer_chunk_file_count += timing.chunk_file_count;
        self.writer_chunk_count += timing.chunk_count;
        self.writer_row_count += timing.row_count;
        self.writer_arrow_array_memory_bytes =
            self.writer_arrow_array_memory_bytes.saturating_add(timing.arrow_array_memory_bytes);
        self.writer_arrow_file_bytes = self.writer_arrow_file_bytes.saturating_add(timing.arrow_file_bytes);
    }

    fn add_finalization_timing(&mut self, timing: finalization::RegenieStep2FinalizationTiming) {
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
        self.finalization_chunk_file_count += timing.chunk_file_count;
        self.finalization_batch_count += timing.batch_count;
        self.finalization_row_count += timing.row_count;
        self.finalization_arrow_file_bytes = self.finalization_arrow_file_bytes.saturating_add(timing.arrow_file_bytes);
        self.finalization_parquet_file_bytes =
            self.finalization_parquet_file_bytes.saturating_add(timing.parquet_file_bytes);
        self.finalization_count += 1;
    }
}

enum OutputCoordinatorJob {
    RegenieStep2(Box<RegenieStep2ChunkJob>),
    Finish,
    Abort,
}

enum OutputWriteJob {
    RegenieStep2(Box<OutputWriteTask>),
}

struct OutputWriteTask {
    write_batch: RegenieStep2ChunkWriteBatch,
    config: OutputWriterConfig,
    worker_errors: Arc<Mutex<Vec<String>>>,
    worker_commits: Arc<Mutex<Vec<manifest::RunManifestChunkCommit>>>,
    stage_timings: Arc<Mutex<OutputStageTimingAccumulator>>,
    completion_tracker: OutputWriteCompletionTracker,
}

struct OutputWriterPool {
    sender: Sender<OutputWriteJob>,
    receiver: Receiver<OutputWriteJob>,
    worker_count: Mutex<usize>,
}

#[derive(Clone)]
struct OutputWriteCompletionTracker {
    inner: Arc<(Mutex<usize>, Condvar)>,
}

struct OutputWriteCompletionGuard {
    completion_tracker: OutputWriteCompletionTracker,
}

impl OutputWriterPool {
    fn new() -> Self {
        let (sender, receiver) = unbounded();
        Self { sender, receiver, worker_count: Mutex::new(0) }
    }

    fn sender(&self) -> Sender<OutputWriteJob> {
        self.sender.clone()
    }

    fn ensure_worker_count(&self, requested_worker_count: usize) -> Result<(), OutputWriterError> {
        let mut worker_count = self
            .worker_count
            .lock()
            .map_err(|_| OutputWriterError::Runtime("Rust output writer pool lock was poisoned.".to_string()))?;
        while *worker_count < requested_worker_count {
            let receiver_clone = self.receiver.clone();
            std::thread::spawn(move || run_output_writer_worker(receiver_clone));
            *worker_count += 1;
        }
        Ok(())
    }

    #[cfg(test)]
    fn current_worker_count(&self) -> usize {
        *self.worker_count.lock().expect("pool worker count should not be poisoned")
    }
}

impl OutputWriteCompletionTracker {
    fn new() -> Self {
        Self { inner: Arc::new((Mutex::new(0), Condvar::new())) }
    }

    fn increment(&self) -> Result<(), OutputWriterError> {
        let (pending_write_count_lock, _) = &*self.inner;
        let mut pending_write_count = pending_write_count_lock
            .lock()
            .map_err(|_| OutputWriterError::Runtime("Rust output writer completion lock was poisoned.".to_string()))?;
        *pending_write_count += 1;
        Ok(())
    }

    fn decrement(&self) {
        let (pending_write_count_lock, pending_write_condvar) = &*self.inner;
        if let Ok(mut pending_write_count) = pending_write_count_lock.lock() {
            *pending_write_count = pending_write_count.saturating_sub(1);
            if *pending_write_count == 0 {
                pending_write_condvar.notify_all();
            }
        }
    }

    fn wait(&self) -> Result<(), OutputWriterError> {
        let (pending_write_count_lock, pending_write_condvar) = &*self.inner;
        let mut pending_write_count = pending_write_count_lock
            .lock()
            .map_err(|_| OutputWriterError::Runtime("Rust output writer completion lock was poisoned.".to_string()))?;
        while *pending_write_count > 0 {
            pending_write_count = pending_write_condvar.wait(pending_write_count).map_err(|_| {
                OutputWriterError::Runtime("Rust output writer completion lock was poisoned.".to_string())
            })?;
        }
        Ok(())
    }
}

impl Drop for OutputWriteCompletionGuard {
    fn drop(&mut self) {
        self.completion_tracker.decrement();
    }
}

fn output_writer_pool() -> Arc<OutputWriterPool> {
    static OUTPUT_WRITER_POOL: OnceLock<Arc<OutputWriterPool>> = OnceLock::new();
    Arc::clone(OUTPUT_WRITER_POOL.get_or_init(|| Arc::new(OutputWriterPool::new())))
}

fn get_output_writer_pool(worker_count: usize) -> Result<Arc<OutputWriterPool>, OutputWriterError> {
    if worker_count == 0 {
        return Err(OutputWriterError::InvalidInput("Writer thread count must be at least 1.".to_string()));
    }
    let pool = output_writer_pool();
    pool.ensure_worker_count(worker_count)?;
    Ok(pool)
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
    ) -> Result<Self, OutputWriterError> {
        if writer_thread_count == 0 {
            return Err(OutputWriterError::InvalidInput("Writer thread count must be at least 1.".to_string()));
        }
        if chunks_per_arrow_file == 0 {
            return Err(OutputWriterError::InvalidInput("Chunks per Arrow file must be at least 1.".to_string()));
        }
        let config = OutputWriterConfig {
            run_directory: PathBuf::from(run_directory),
            chunks_directory: PathBuf::from(chunks_directory),
            association_mode,
            output_format: OutputFileFormat::parse(output_format).map_err(OutputWriterError::InvalidInput)?,
            output_statistic_dtype: OutputStatisticDtype::parse(output_statistic_dtype)
                .map_err(OutputWriterError::InvalidInput)?,
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

    pub fn finish(&self) -> Result<Option<PathBuf>, OutputWriterError> {
        let finish_start_time = start_optional_timing(self.config.collect_stage_timings);
        self.close_writer_sender(OutputCoordinatorJob::Finish)?;
        self.join_coordinator_thread()?;
        self.wait_for_writer_tasks()?;
        self.raise_if_worker_failed()?;
        let chunk_commits = self.take_worker_commits()?;
        let manifest_commit_start_time = start_optional_timing(self.config.collect_stage_timings);
        manifest::record_run_manifest_chunk_commits(&self.config.run_directory, chunk_commits)
            .map_err(OutputWriterError::runtime)?;
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

    pub fn finish_interrupted(&self, signal_name: &str) -> Result<(), OutputWriterError> {
        let finish_start_time = start_optional_timing(self.config.collect_stage_timings);
        self.close_writer_sender(OutputCoordinatorJob::Finish)?;
        self.join_coordinator_thread()?;
        self.wait_for_writer_tasks()?;
        self.raise_if_worker_failed()?;
        let chunk_commits = self.take_worker_commits()?;
        let manifest_commit_start_time = start_optional_timing(self.config.collect_stage_timings);
        manifest::record_run_manifest_chunk_commits(&self.config.run_directory, chunk_commits)
            .map_err(OutputWriterError::runtime)?;
        if let Some(start_time) = manifest_commit_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.manifest_commit_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.manifest_commit_count += 1;
            })?;
        }
        manifest::mark_run_manifest_interrupted(&self.config.run_directory, signal_name)
            .map_err(OutputWriterError::runtime)?;
        self.record_finish_timing(finish_start_time)?;
        self.write_stage_timing_snapshot()
    }

    pub fn abort(&self) -> Result<(), OutputWriterError> {
        self.close_writer_sender(OutputCoordinatorJob::Abort)?;
        self.join_coordinator_thread()?;
        self.wait_for_writer_tasks()?;
        Ok(())
    }

    fn take_worker_commits(&self) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputWriterError> {
        let mut worker_commits = self
            .worker_commits
            .lock()
            .map_err(|_| OutputWriterError::Runtime("Rust output writer commit lock was poisoned.".to_string()))?;
        Ok(std::mem::take(&mut *worker_commits))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_regenie2_native_chunk(
        &self,
        variant_start_index: i64,
        variant_stop_index: i64,
        metadata: &VariantMetadataColumns,
        chunk_stats: &NativeChunkStats,
        beta: &[f32],
        standard_error: &[f32],
        chi_squared: &[f32],
        log10_p_value: &[f32],
        extra_code: Option<&[i32]>,
    ) -> Result<(), OutputWriterError> {
        let expected_variant_stop_index = variant_start_index
            .checked_add(i64::try_from(metadata.position.len()).map_err(|_| {
                OutputWriterError::InvalidInput("Rust output writer row count does not fit into int64.".to_string())
            })?)
            .ok_or_else(|| {
                OutputWriterError::InvalidInput(
                    "Rust output writer variant stop index does not fit into int64.".to_string(),
                )
            })?;
        if variant_stop_index != expected_variant_stop_index {
            return Err(OutputWriterError::InvalidInput(
                "Rust output writer metadata bounds do not match metadata row count.".to_string(),
            ));
        }
        let metadata_clone_start_time = start_optional_timing(self.config.collect_stage_timings);
        let chunk_handle =
            NativeChunkHandle::new(Arc::new(metadata.clone()), Arc::new(chunk_stats.clone()), variant_start_index);
        if let Some(start_time) = metadata_clone_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.metadata_clone_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.metadata_clone_count += 1;
            })?;
        }
        self.write_regenie2_native_chunk_handle(
            chunk_handle,
            beta,
            standard_error,
            chi_squared,
            log10_p_value,
            extra_code,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_regenie2_native_chunk_handle(
        &self,
        chunk_handle: NativeChunkHandle,
        beta: &[f32],
        standard_error: &[f32],
        chi_squared: &[f32],
        log10_p_value: &[f32],
        extra_code: Option<&[i32]>,
    ) -> Result<(), OutputWriterError> {
        if self.config.association_mode != "regenie2_linear" && self.config.association_mode != "regenie2_binary" {
            return Err(OutputWriterError::InvalidInput(
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
        if let Some(extra_code_values) = extra_code {
            validate_column_lengths(row_count, &[extra_code_values.len()])?;
        }
        let result_buffer_copy_start_time = start_optional_timing(self.config.collect_stage_timings);
        let beta_array = build_float32_result_array(beta);
        let standard_error_array = build_float32_result_array(standard_error);
        let chi_squared_array = build_float32_result_array(chi_squared);
        let log10_p_value_array = build_float32_result_array(log10_p_value);
        let extra_code_array = extra_code.map(build_int32_result_array);
        if let Some(start_time) = result_buffer_copy_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.result_buffer_copy_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.result_buffer_copy_count += 1;
            })?;
        }
        self.write_regenie2_native_chunk_handle_arrays(
            chunk_handle,
            beta_array,
            standard_error_array,
            chi_squared_array,
            log10_p_value_array,
            extra_code_array,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_regenie2_native_chunk_handle_arrays(
        &self,
        chunk_handle: NativeChunkHandle,
        beta: ArrayRef,
        standard_error: ArrayRef,
        chi_squared: ArrayRef,
        log10_p_value: ArrayRef,
        extra_code: Option<ArrayRef>,
    ) -> Result<(), OutputWriterError> {
        if self.config.association_mode != "regenie2_linear" && self.config.association_mode != "regenie2_binary" {
            return Err(OutputWriterError::InvalidInput(
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
            .map_err(|_| OutputWriterError::Runtime("Rust output writer sender lock was poisoned.".to_string()))?;
        let sender = sender_guard
            .as_ref()
            .ok_or_else(|| OutputWriterError::Runtime("Rust output writer session is already closed.".to_string()))?;
        let enqueue_start_time = start_optional_timing(self.config.collect_stage_timings);
        sender.send(OutputCoordinatorJob::RegenieStep2(Box::new(job))).map_err(OutputWriterError::runtime)?;
        if let Some(start_time) = enqueue_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.enqueue_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.enqueue_count += 1;
            })?;
        }
        Ok(())
    }

    fn record_finish_timing(&self, finish_start_time: Option<Instant>) -> Result<(), OutputWriterError> {
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
    ) -> Result<(), OutputWriterError> {
        if !self.config.collect_stage_timings {
            return Ok(());
        }
        let mut stage_timings = self.stage_timings.lock().map_err(|_| {
            OutputWriterError::Runtime("Rust output writer stage timing lock was poisoned.".to_string())
        })?;
        update_stage_timings(&mut stage_timings);
        Ok(())
    }

    fn write_stage_timing_snapshot(&self) -> Result<(), OutputWriterError> {
        if !self.config.collect_stage_timings {
            return Ok(());
        }
        let stage_timings = self.stage_timings.lock().map_err(|_| {
            OutputWriterError::Runtime("Rust output writer stage timing lock was poisoned.".to_string())
        })?;
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
        let timing_path = self.config.run_directory.join(OUTPUT_STAGE_TIMING_FILE_NAME);
        let temporary_timing_path = timing_path.with_extension("json.tmp");
        let timing_text = serde_json::to_string_pretty(&payload).map_err(OutputWriterError::runtime)?;
        std::fs::write(&temporary_timing_path, format!("{timing_text}\n")).map_err(OutputWriterError::runtime)?;
        std::fs::rename(&temporary_timing_path, &timing_path).map_err(OutputWriterError::runtime)
    }

    fn raise_if_worker_failed(&self) -> Result<(), OutputWriterError> {
        let worker_errors = self
            .worker_errors
            .lock()
            .map_err(|_| OutputWriterError::Runtime("Rust output writer error lock was poisoned.".to_string()))?;
        if let Some(first_error) = worker_errors.first() {
            return Err(OutputWriterError::Runtime(first_error.clone()));
        }
        Ok(())
    }

    fn close_writer_sender(&self, close_job: OutputCoordinatorJob) -> Result<(), OutputWriterError> {
        let mut sender_guard = self
            .sender
            .lock()
            .map_err(|_| OutputWriterError::Runtime("Rust output writer sender lock was poisoned.".to_string()))?;
        if let Some(active_sender) = sender_guard.take() {
            active_sender.send(close_job).map_err(OutputWriterError::runtime)?;
        }
        Ok(())
    }

    fn join_coordinator_thread(&self) -> Result<(), OutputWriterError> {
        let mut coordinator_handle_guard = self.coordinator_handle.lock().map_err(|_| {
            OutputWriterError::Runtime("Rust output writer coordinator handle lock was poisoned.".to_string())
        })?;
        if let Some(handle) = coordinator_handle_guard.take() {
            handle.join().map_err(|_| {
                OutputWriterError::Runtime("Rust output writer coordinator thread panicked.".to_string())
            })?;
        }
        Ok(())
    }

    fn wait_for_writer_tasks(&self) -> Result<(), OutputWriterError> {
        self.completion_tracker.wait()
    }
}

fn build_float32_result_array(values: &[f32]) -> ArrayRef {
    Arc::new(Float32Array::from(values.to_vec()))
}

fn build_int32_result_array(values: &[i32]) -> ArrayRef {
    Arc::new(Int32Array::from(values.to_vec()))
}

fn validate_column_lengths(expected_row_count: usize, observed_lengths: &[usize]) -> Result<(), OutputWriterError> {
    if observed_lengths.iter().all(|observed_length| *observed_length == expected_row_count) {
        return Ok(());
    }
    Err(OutputWriterError::InvalidInput(
        "Rust output writer batch column lengths do not all match the expected row count.".to_string(),
    ))
}

fn validate_statistic_array_type(
    column_name: &str,
    array: &ArrayRef,
    output_statistic_dtype: OutputStatisticDtype,
) -> Result<(), OutputWriterError> {
    let type_matches = match output_statistic_dtype {
        OutputStatisticDtype::Float32 => array.as_any().is::<Float32Array>(),
        OutputStatisticDtype::Float64 => array.as_any().is::<Float64Array>(),
    };
    if type_matches {
        return Ok(());
    }
    Err(OutputWriterError::InvalidInput(format!(
        "Rust output writer column {column_name} must be {} for the configured output statistic dtype.",
        output_statistic_dtype.as_str(),
    )))
}

fn start_optional_timing(collect_stage_timings: bool) -> Option<Instant> {
    collect_stage_timings.then(Instant::now)
}

#[allow(clippy::needless_pass_by_value)]
fn run_output_writer_coordinator(
    receiver: Receiver<OutputCoordinatorJob>,
    writer_pool: Arc<OutputWriterPool>,
    config: OutputWriterConfig,
    worker_errors: Arc<Mutex<Vec<String>>>,
    worker_commits: Arc<Mutex<Vec<manifest::RunManifestChunkCommit>>>,
    stage_timings: Arc<Mutex<OutputStageTimingAccumulator>>,
    completion_tracker: OutputWriteCompletionTracker,
) {
    let mut pending_chunks = Vec::with_capacity(config.chunks_per_arrow_file);
    while let Ok(job) = receiver.recv() {
        match job {
            OutputCoordinatorJob::RegenieStep2(chunk_job) => {
                pending_chunks.push(*chunk_job);
                if pending_chunks.len() >= config.chunks_per_arrow_file
                    && flush_pending_regenie_step2_chunks(
                        &writer_pool,
                        &mut pending_chunks,
                        &config,
                        &worker_errors,
                        &worker_commits,
                        &stage_timings,
                        &completion_tracker,
                    )
                    .is_err()
                {
                    break;
                }
            }
            OutputCoordinatorJob::Finish => {
                let _ = flush_pending_regenie_step2_chunks(
                    &writer_pool,
                    &mut pending_chunks,
                    &config,
                    &worker_errors,
                    &worker_commits,
                    &stage_timings,
                    &completion_tracker,
                );
                break;
            }
            OutputCoordinatorJob::Abort => break,
        }
    }
}

fn flush_pending_regenie_step2_chunks(
    writer_pool: &OutputWriterPool,
    pending_chunks: &mut Vec<RegenieStep2ChunkJob>,
    config: &OutputWriterConfig,
    worker_errors: &Arc<Mutex<Vec<String>>>,
    worker_commits: &Arc<Mutex<Vec<manifest::RunManifestChunkCommit>>>,
    stage_timings: &Arc<Mutex<OutputStageTimingAccumulator>>,
    completion_tracker: &OutputWriteCompletionTracker,
) -> Result<(), ()> {
    if pending_chunks.is_empty() {
        return Ok(());
    }
    let flush_start_time = start_optional_timing(config.collect_stage_timings);
    let first_chunk_identifier = pending_chunks.first().map_or(0, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    let last_chunk_identifier =
        pending_chunks.last().map_or(first_chunk_identifier, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    let chunk_file_name = build_output_file_name(config.output_format, first_chunk_identifier, last_chunk_identifier);
    let write_batch = RegenieStep2ChunkWriteBatch { chunk_file_name, chunks: std::mem::take(pending_chunks) };
    completion_tracker.increment().map_err(|error| {
        push_worker_error(worker_errors, error.to_string());
    })?;
    let write_task = OutputWriteTask {
        write_batch,
        config: config.clone(),
        worker_errors: Arc::clone(worker_errors),
        worker_commits: Arc::clone(worker_commits),
        stage_timings: Arc::clone(stage_timings),
        completion_tracker: completion_tracker.clone(),
    };
    writer_pool.sender().send(OutputWriteJob::RegenieStep2(Box::new(write_task))).map_err(|error| {
        completion_tracker.decrement();
        push_worker_error(worker_errors, error.to_string());
    })?;
    if let Some(start_time) = flush_start_time {
        let mut stage_timings_guard = stage_timings.lock().map_err(|_| {
            push_worker_error(worker_errors, "Rust output writer stage timing lock was poisoned.".to_string());
        })?;
        stage_timings_guard.coordinator_flush_seconds += start_time.elapsed().as_secs_f64();
        stage_timings_guard.coordinator_flush_count += 1;
    }
    Ok(())
}

fn push_worker_error(worker_errors: &Arc<Mutex<Vec<String>>>, error: String) {
    if let Ok(mut worker_errors_guard) = worker_errors.lock() {
        worker_errors_guard.push(error);
    }
}

#[allow(clippy::needless_pass_by_value)]
fn run_output_writer_worker(receiver: Receiver<OutputWriteJob>) {
    while let Ok(job) = receiver.recv() {
        match job {
            OutputWriteJob::RegenieStep2(output_write_task) => run_output_write_task(*output_write_task),
        }
    }
}

fn run_output_write_task(output_write_task: OutputWriteTask) {
    let _completion_guard =
        OutputWriteCompletionGuard { completion_tracker: output_write_task.completion_tracker.clone() };
    let write_result = write_regenie_step2_chunk_job(
        &output_write_task.config.chunks_directory,
        output_write_task.write_batch,
        output_write_task.config.output_format,
        output_write_task.config.output_statistic_dtype,
        &output_write_task.config.arrow_compression,
        &output_write_task.config.parquet_compression,
    );
    match write_result {
        Ok(write_result) => {
            if output_write_task.config.collect_stage_timings {
                let Ok(mut stage_timings_guard) = output_write_task.stage_timings.lock() else {
                    push_worker_error(
                        &output_write_task.worker_errors,
                        "Rust output writer stage timing lock was poisoned.".to_string(),
                    );
                    return;
                };
                stage_timings_guard.add_writer_timing(write_result.timing);
            }
            let Ok(mut worker_commits_guard) = output_write_task.worker_commits.lock() else {
                push_worker_error(
                    &output_write_task.worker_errors,
                    "Rust output writer commit lock was poisoned.".to_string(),
                );
                return;
            };
            worker_commits_guard.extend(write_result.chunk_commits);
        }
        Err(error) => {
            push_worker_error(&output_write_task.worker_errors, error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::finalization::RegenieStep2FinalizationTiming;

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
    fn result_array_builders_and_stage_timing_skip_path_are_covered() {
        let float_array = build_float32_result_array(&[1.0, 2.0]);
        let int_array = build_int32_result_array(&[1, 2]);
        assert_eq!(float_array.len(), 2);
        assert_eq!(int_array.len(), 2);
        assert!(validate_column_lengths(2, &[2, 2]).is_ok());

        let session = OutputWriterSession::new(
            "unused-run".to_string(),
            "unused-chunks".to_string(),
            "regenie2_linear".to_string(),
            1,
            1,
            "arrow",
            "float32",
            false,
            1,
            "none".to_string(),
            "none".to_string(),
            false,
        )
        .expect("session should open");
        session
            .record_stage_timing(|stage_timings| {
                stage_timings.enqueue_count += 1;
            })
            .expect("timing skip path should not lock");
        session.abort().expect("session should abort");
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
