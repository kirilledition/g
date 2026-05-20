use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::genotype::common::{ChunkStats as NativeChunkStats, VariantMetadataColumns};
use crate::output::finalization;
use crate::output::writer::{
    OutputWriterError, RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, build_chunk_file_name,
    write_regenie_step2_chunk_job,
};

#[derive(Clone)]
pub(crate) struct NativeChunkHandle {
    pub(crate) metadata: Arc<VariantMetadataColumns>,
    pub(crate) stats: Arc<NativeChunkStats>,
    pub(crate) chunk_identifier: i64,
}

impl NativeChunkHandle {
    pub(crate) fn new(
        metadata: Arc<VariantMetadataColumns>,
        stats: Arc<NativeChunkStats>,
        chunk_identifier: i64,
    ) -> Self {
        Self { metadata, stats, chunk_identifier }
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
}

#[derive(Clone)]
struct OutputWriterConfig {
    run_directory: PathBuf,
    chunks_directory: PathBuf,
    association_mode: String,
    finalize_parquet: bool,
    chunks_per_arrow_file: usize,
    arrow_compression: String,
}

enum OutputCoordinatorJob {
    RegenieStep2(Box<RegenieStep2ChunkJob>),
    Finish,
    Abort,
}

enum OutputWriteJob {
    RegenieStep2(Box<RegenieStep2ChunkWriteBatch>),
    Shutdown,
}

pub struct OutputWriterSession {
    sender: Mutex<Option<Sender<OutputCoordinatorJob>>>,
    coordinator_handle: Mutex<Option<JoinHandle<()>>>,
    worker_handles: Mutex<Vec<JoinHandle<()>>>,
    worker_errors: Arc<Mutex<Vec<String>>>,
    config: OutputWriterConfig,
}

#[allow(clippy::missing_errors_doc)]
impl OutputWriterSession {
    pub fn new(
        run_directory: String,
        chunks_directory: String,
        association_mode: String,
        writer_thread_count: usize,
        writer_queue_depth: usize,
        finalize_parquet: bool,
        chunks_per_arrow_file: usize,
        arrow_compression: String,
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
            finalize_parquet,
            chunks_per_arrow_file,
            arrow_compression,
        };
        let (sender, receiver) = bounded(writer_queue_depth.max(1));
        let (writer_sender, writer_receiver) = bounded(writer_queue_depth.max(1));
        let worker_errors = Arc::new(Mutex::new(Vec::new()));
        let mut worker_handles = Vec::with_capacity(writer_thread_count);
        for _ in 0..writer_thread_count {
            let receiver_clone = writer_receiver.clone();
            let config_clone = config.clone();
            let worker_errors_clone = Arc::clone(&worker_errors);
            worker_handles.push(std::thread::spawn(move || {
                run_output_writer_worker(receiver_clone, config_clone, worker_errors_clone);
            }));
        }
        let coordinator_worker_errors = Arc::clone(&worker_errors);
        let coordinator_chunks_per_arrow_file = config.chunks_per_arrow_file;
        let coordinator_handle = std::thread::spawn(move || {
            run_output_writer_coordinator(
                receiver,
                writer_sender,
                writer_thread_count,
                coordinator_chunks_per_arrow_file,
                coordinator_worker_errors,
            );
        });
        Ok(Self {
            sender: Mutex::new(Some(sender)),
            coordinator_handle: Mutex::new(Some(coordinator_handle)),
            worker_handles: Mutex::new(worker_handles),
            worker_errors,
            config,
        })
    }

    pub fn finish(&self) -> Result<Option<PathBuf>, OutputWriterError> {
        self.close_writer_sender(OutputCoordinatorJob::Finish)?;
        self.join_coordinator_thread()?;
        self.join_writer_threads()?;
        self.raise_if_worker_failed()?;
        if !self.config.finalize_parquet {
            return Ok(None);
        }
        let final_parquet_path = self.config.run_directory.join("final.parquet");
        finalization::write_final_parquet_from_chunk_files(
            &self.config.chunks_directory,
            &final_parquet_path,
            &self.config.association_mode,
        )?;
        Ok(Some(final_parquet_path))
    }

    pub fn abort(&self) -> Result<(), OutputWriterError> {
        self.close_writer_sender(OutputCoordinatorJob::Abort)?;
        self.join_coordinator_thread()?;
        self.join_writer_threads()?;
        Ok(())
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
        self.write_regenie2_native_chunk_handle(
            NativeChunkHandle::new(Arc::new(metadata.clone()), Arc::new(chunk_stats.clone()), variant_start_index),
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
        let job = RegenieStep2ChunkJob {
            chunk_handle,
            beta: beta.to_vec(),
            se: standard_error.to_vec(),
            chisq: chi_squared.to_vec(),
            log10p: log10_p_value.to_vec(),
            extra_code: extra_code.map(<[i32]>::to_vec),
        };
        self.raise_if_worker_failed()?;
        let sender_guard = self
            .sender
            .lock()
            .map_err(|_| OutputWriterError::Runtime("Rust output writer sender lock was poisoned.".to_string()))?;
        let sender = sender_guard
            .as_ref()
            .ok_or_else(|| OutputWriterError::Runtime("Rust output writer session is already closed.".to_string()))?;
        sender.send(OutputCoordinatorJob::RegenieStep2(Box::new(job))).map_err(OutputWriterError::runtime)?;
        Ok(())
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

    fn join_writer_threads(&self) -> Result<(), OutputWriterError> {
        let mut worker_handles_guard = self
            .worker_handles
            .lock()
            .map_err(|_| OutputWriterError::Runtime("Rust output writer handle lock was poisoned.".to_string()))?;
        while let Some(worker_handle) = worker_handles_guard.pop() {
            worker_handle
                .join()
                .map_err(|_| OutputWriterError::Runtime("Rust output writer worker thread panicked.".to_string()))?;
        }
        Ok(())
    }
}

fn validate_column_lengths(expected_row_count: usize, observed_lengths: &[usize]) -> Result<(), OutputWriterError> {
    if observed_lengths.iter().all(|observed_length| *observed_length == expected_row_count) {
        return Ok(());
    }
    Err(OutputWriterError::InvalidInput(
        "Rust output writer batch column lengths do not all match the expected row count.".to_string(),
    ))
}

#[allow(clippy::needless_pass_by_value)]
fn run_output_writer_coordinator(
    receiver: Receiver<OutputCoordinatorJob>,
    writer_sender: Sender<OutputWriteJob>,
    writer_thread_count: usize,
    chunks_per_arrow_file: usize,
    worker_errors: Arc<Mutex<Vec<String>>>,
) {
    let mut pending_chunks = Vec::with_capacity(chunks_per_arrow_file);
    while let Ok(job) = receiver.recv() {
        match job {
            OutputCoordinatorJob::RegenieStep2(chunk_job) => {
                pending_chunks.push(*chunk_job);
                if pending_chunks.len() >= chunks_per_arrow_file
                    && flush_pending_regenie_step2_chunks(&writer_sender, &mut pending_chunks, &worker_errors).is_err()
                {
                    break;
                }
            }
            OutputCoordinatorJob::Finish => {
                let _ = flush_pending_regenie_step2_chunks(&writer_sender, &mut pending_chunks, &worker_errors);
                break;
            }
            OutputCoordinatorJob::Abort => break,
        }
    }
    for _ in 0..writer_thread_count {
        if let Err(error) = writer_sender.send(OutputWriteJob::Shutdown) {
            push_worker_error(&worker_errors, error.to_string());
            return;
        }
    }
}

fn flush_pending_regenie_step2_chunks(
    writer_sender: &Sender<OutputWriteJob>,
    pending_chunks: &mut Vec<RegenieStep2ChunkJob>,
    worker_errors: &Arc<Mutex<Vec<String>>>,
) -> Result<(), ()> {
    if pending_chunks.is_empty() {
        return Ok(());
    }
    let first_chunk_identifier = pending_chunks.first().map_or(0, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    let last_chunk_identifier =
        pending_chunks.last().map_or(first_chunk_identifier, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    let chunk_file_name = build_chunk_file_name(first_chunk_identifier, last_chunk_identifier);
    let write_batch = RegenieStep2ChunkWriteBatch { chunk_file_name, chunks: std::mem::take(pending_chunks) };
    writer_sender.send(OutputWriteJob::RegenieStep2(Box::new(write_batch))).map_err(|error| {
        push_worker_error(worker_errors, error.to_string());
    })
}

fn push_worker_error(worker_errors: &Arc<Mutex<Vec<String>>>, error: String) {
    if let Ok(mut worker_errors_guard) = worker_errors.lock() {
        worker_errors_guard.push(error);
    }
}

#[allow(clippy::needless_pass_by_value)]
fn run_output_writer_worker(
    receiver: Receiver<OutputWriteJob>,
    config: OutputWriterConfig,
    worker_errors: Arc<Mutex<Vec<String>>>,
) {
    while let Ok(job) = receiver.recv() {
        let write_result = match job {
            OutputWriteJob::RegenieStep2(regenie_step2_job) => write_regenie_step2_chunk_job(
                &config.run_directory,
                &config.chunks_directory,
                *regenie_step2_job,
                &config.arrow_compression,
            ),
            OutputWriteJob::Shutdown => return,
        };
        if let Err(error) = write_result {
            push_worker_error(&worker_errors, error);
            return;
        }
    }
}
