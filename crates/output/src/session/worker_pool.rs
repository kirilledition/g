use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::error::OutputError;
use crate::manifest;
use crate::timing::OutputStageTimingAccumulator;
use crate::writer::{RegenieStep2ChunkWriteBatch, write_regenie_step2_chunk_job};

use super::writer_session::OutputWriterConfig;

struct OutputWriteTask {
    write_batch: RegenieStep2ChunkWriteBatch,
    config: Arc<OutputWriterConfig>,
    worker_error: Arc<Mutex<Option<String>>>,
    worker_commits: Arc<Mutex<Vec<manifest::RunManifestChunkCommit>>>,
    stage_timings: Arc<Mutex<OutputStageTimingAccumulator>>,
    completion_tracker: OutputWriteCompletionTracker,
}

pub(super) struct OutputWriterPool {
    sender: Option<Sender<OutputWriteTask>>,
    worker_handles: Vec<JoinHandle<()>>,
}

#[derive(Clone)]
pub(super) struct OutputWriteCompletionTracker {
    inner: Arc<(Mutex<usize>, Condvar)>,
}

struct OutputWriteCompletionGuard {
    completion_tracker: OutputWriteCompletionTracker,
}

impl OutputWriterPool {
    pub(super) fn new(worker_count: usize, queue_depth: usize) -> Result<Arc<Self>, OutputError> {
        if worker_count == 0 {
            return Err(OutputError::InvalidInput("Writer thread count must be at least 1.".to_string()));
        }
        if queue_depth == 0 {
            return Err(OutputError::InvalidInput("Writer queue depth must be at least 1.".to_string()));
        }
        let (sender, receiver) = bounded(queue_depth);
        let mut worker_handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let receiver_clone = receiver.clone();
            worker_handles.push(std::thread::spawn(move || run_output_writer_worker(receiver_clone)));
        }
        Ok(Arc::new(Self { sender: Some(sender), worker_handles }))
    }

    pub(super) fn enqueue_regenie_step2(
        &self,
        write_batch: RegenieStep2ChunkWriteBatch,
        config: &Arc<OutputWriterConfig>,
        worker_error: &Arc<Mutex<Option<String>>>,
        worker_commits: &Arc<Mutex<Vec<manifest::RunManifestChunkCommit>>>,
        stage_timings: &Arc<Mutex<OutputStageTimingAccumulator>>,
        completion_tracker: &OutputWriteCompletionTracker,
    ) -> Result<(), OutputError> {
        completion_tracker.increment()?;
        let write_task = OutputWriteTask {
            write_batch,
            config: Arc::clone(config),
            worker_error: Arc::clone(worker_error),
            worker_commits: Arc::clone(worker_commits),
            stage_timings: Arc::clone(stage_timings),
            completion_tracker: completion_tracker.clone(),
        };
        let Some(sender) = self.sender.as_ref() else {
            completion_tracker.decrement();
            return Err(OutputError::Runtime("Rust output writer pool is closed.".to_string()));
        };
        sender.send(write_task).map_err(|error| {
            completion_tracker.decrement();
            OutputError::runtime(error)
        })
    }
}

impl Drop for OutputWriterPool {
    fn drop(&mut self) {
        self.sender.take();
        for worker_handle in self.worker_handles.drain(..) {
            let _ = worker_handle.join();
        }
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
        *pending_write_count = pending_write_count.checked_add(1).ok_or_else(|| {
            OutputError::Runtime("Rust output writer pending-write count overflowed platform capacity.".to_string())
        })?;
        Ok(())
    }

    fn decrement(&self) {
        let (pending_write_count_lock, pending_write_condvar) = &*self.inner;
        if let Ok(mut pending_write_count) = pending_write_count_lock.lock() {
            *pending_write_count =
                pending_write_count.checked_sub(1).expect("output write completion count invariant violated");
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

pub(super) fn record_worker_error(worker_error: &Arc<Mutex<Option<String>>>, error: String) {
    if let Ok(mut worker_error_guard) = worker_error.lock()
        && worker_error_guard.is_none()
    {
        *worker_error_guard = Some(error);
    }
}

#[allow(clippy::needless_pass_by_value)]
fn run_output_writer_worker(receiver: Receiver<OutputWriteTask>) {
    while let Ok(output_write_task) = receiver.recv() {
        run_output_write_task(output_write_task);
    }
}

fn run_output_write_task(output_write_task: OutputWriteTask) {
    let _completion_guard =
        OutputWriteCompletionGuard { completion_tracker: output_write_task.completion_tracker.clone() };
    let write_result = write_regenie_step2_chunk_job(
        &output_write_task.config.parts_directory,
        output_write_task.write_batch,
    );
    match write_result {
        Ok(write_result) => {
            if output_write_task.config.collect_stage_timings {
                let Ok(mut stage_timings_guard) = output_write_task.stage_timings.lock() else {
                    record_worker_error(
                        &output_write_task.worker_error,
                        "Rust output writer stage timing lock was poisoned.".to_string(),
                    );
                    return;
                };
                stage_timings_guard.add_writer_timing(write_result.timing);
            }
            let Ok(mut worker_commits_guard) = output_write_task.worker_commits.lock() else {
                record_worker_error(
                    &output_write_task.worker_error,
                    "Rust output writer commit lock was poisoned.".to_string(),
                );
                return;
            };
            worker_commits_guard.extend(write_result.chunk_commits);
        }
        Err(error) => {
            record_worker_error(&output_write_task.worker_error, error.to_string());
        }
    }
}
