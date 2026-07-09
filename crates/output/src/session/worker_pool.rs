use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::error::OutputError;
use crate::manifest;
use crate::timing::OutputStageTimingAccumulator;
use crate::writer::{RegenieStep2ChunkWriteBatch, write_regenie_step2_chunk_job};

use super::OutputWriterConfig;

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

pub(super) struct OutputWriterPool {
    sender: Sender<OutputWriteJob>,
    receiver: Receiver<OutputWriteJob>,
    worker_count: Mutex<usize>,
}

#[derive(Clone)]
pub(super) struct OutputWriteCompletionTracker {
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

    fn ensure_worker_count(&self, requested_worker_count: usize) -> Result<(), OutputError> {
        let mut worker_count = self
            .worker_count
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer pool lock was poisoned.".to_string()))?;
        while *worker_count < requested_worker_count {
            let receiver_clone = self.receiver.clone();
            std::thread::spawn(move || run_output_writer_worker(receiver_clone));
            *worker_count += 1;
        }
        Ok(())
    }

    pub(super) fn enqueue_regenie_step2(
        &self,
        write_batch: RegenieStep2ChunkWriteBatch,
        config: OutputWriterConfig,
        worker_errors: &Arc<Mutex<Vec<String>>>,
        worker_commits: &Arc<Mutex<Vec<manifest::RunManifestChunkCommit>>>,
        stage_timings: &Arc<Mutex<OutputStageTimingAccumulator>>,
        completion_tracker: &OutputWriteCompletionTracker,
    ) -> Result<(), ()> {
        completion_tracker.increment().map_err(|error| {
            push_worker_error(worker_errors, error.to_string());
        })?;
        let write_task = OutputWriteTask {
            write_batch,
            config,
            worker_errors: Arc::clone(worker_errors),
            worker_commits: Arc::clone(worker_commits),
            stage_timings: Arc::clone(stage_timings),
            completion_tracker: completion_tracker.clone(),
        };
        self.sender.send(OutputWriteJob::RegenieStep2(Box::new(write_task))).map_err(|error| {
            completion_tracker.decrement();
            push_worker_error(worker_errors, error.to_string());
        })
    }
}

impl OutputWriteCompletionTracker {
    pub(super) fn new() -> Self {
        Self { inner: Arc::new((Mutex::new(0), Condvar::new())) }
    }

    fn increment(&self) -> Result<(), OutputError> {
        let (pending_write_count_lock, _) = &*self.inner;
        let mut pending_write_count = pending_write_count_lock
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer completion lock was poisoned.".to_string()))?;
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

    pub(super) fn wait(&self) -> Result<(), OutputError> {
        let (pending_write_count_lock, pending_write_condvar) = &*self.inner;
        let mut pending_write_count = pending_write_count_lock
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer completion lock was poisoned.".to_string()))?;
        while *pending_write_count > 0 {
            pending_write_count = pending_write_condvar
                .wait(pending_write_count)
                .map_err(|_| OutputError::Runtime("Rust output writer completion lock was poisoned.".to_string()))?;
        }
        Ok(())
    }
}

impl Drop for OutputWriteCompletionGuard {
    fn drop(&mut self) {
        self.completion_tracker.decrement();
    }
}

pub(super) fn get_output_writer_pool(worker_count: usize) -> Result<Arc<OutputWriterPool>, OutputError> {
    if worker_count == 0 {
        return Err(OutputError::InvalidInput("Writer thread count must be at least 1.".to_string()));
    }
    let pool = output_writer_pool();
    pool.ensure_worker_count(worker_count)?;
    Ok(pool)
}

pub(super) fn push_worker_error(worker_errors: &Arc<Mutex<Vec<String>>>, error: String) {
    if let Ok(mut worker_errors_guard) = worker_errors.lock() {
        worker_errors_guard.push(error);
    }
}

fn output_writer_pool() -> Arc<OutputWriterPool> {
    static OUTPUT_WRITER_POOL: OnceLock<Arc<OutputWriterPool>> = OnceLock::new();
    Arc::clone(OUTPUT_WRITER_POOL.get_or_init(|| Arc::new(OutputWriterPool::new())))
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
            push_worker_error(&output_write_task.worker_errors, error.to_string());
        }
    }
}
