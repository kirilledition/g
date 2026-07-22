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
    completion_guard: OutputWriteCompletionGuard,
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
    pub(super) fn validate_settings(worker_count: usize, queue_depth: usize) -> Result<(), OutputError> {
        if worker_count == 0 {
            return Err(OutputError::InvalidInput("Writer thread count must be at least 1.".to_string()));
        }
        if queue_depth == 0 {
            return Err(OutputError::InvalidInput("Writer queue depth must be at least 1.".to_string()));
        }
        Ok(())
    }

    pub(super) fn new(worker_count: usize, queue_depth: usize) -> Result<Arc<Self>, OutputError> {
        Self::validate_settings(worker_count, queue_depth)?;
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
        let Some(sender) = self.sender.as_ref() else {
            return Err(OutputError::Runtime("Rust output writer pool is closed.".to_string()));
        };
        completion_tracker.increment()?;
        let write_task = OutputWriteTask {
            write_batch,
            config: Arc::clone(config),
            worker_error: Arc::clone(worker_error),
            worker_commits: Arc::clone(worker_commits),
            stage_timings: Arc::clone(stage_timings),
            completion_guard: OutputWriteCompletionGuard { completion_tracker: completion_tracker.clone() },
        };
        sender.send(write_task).map_err(OutputError::runtime)
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

// The spawned worker owns its receiver for the full thread lifetime.
#[allow(clippy::needless_pass_by_value)]
fn run_output_writer_worker(receiver: Receiver<OutputWriteTask>) {
    while let Ok(output_write_task) = receiver.recv() {
        let worker_error = Arc::clone(&output_write_task.worker_error);
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_output_write_task(output_write_task))).is_err()
        {
            record_worker_error(&worker_error, "Rust output writer panicked.".to_string());
        }
    }
}

fn run_output_write_task(output_write_task: OutputWriteTask) {
    let _completion_guard = output_write_task.completion_guard;
    let write_result = write_regenie_step2_chunk_job(
        &output_write_task.config.parts_directory,
        output_write_task.write_batch,
        output_write_task.config.collect_stage_timings,
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crossbeam_channel::bounded;

    use super::{
        OutputWriteCompletionGuard, OutputWriteCompletionTracker, OutputWriteTask, OutputWriterPool,
        record_worker_error, run_output_writer_worker,
    };

    #[test]
    fn writer_pool_rejects_zero_workers_or_queue_depth() {
        let worker_error = OutputWriterPool::new(0, 1).err().expect("zero workers are rejected");
        assert!(worker_error.to_string().contains("Writer thread count must be at least 1"));
        let queue_error = OutputWriterPool::new(1, 0).err().expect("zero queue depth is rejected");
        assert!(queue_error.to_string().contains("Writer queue depth must be at least 1"));

        drop(OutputWriterPool::new(1, 1).expect("valid writer pool starts and stops"));
    }

    #[test]
    fn completion_guard_decrements_tracker_and_unblocks_wait() {
        let tracker = OutputWriteCompletionTracker::new();
        tracker.increment().expect("pending count increments");
        let guard = OutputWriteCompletionGuard { completion_tracker: tracker.clone() };
        drop(guard);
        tracker.wait().expect("zero pending writes return immediately");
    }

    #[test]
    fn worker_error_records_only_the_first_failure() {
        let worker_error = Arc::new(Mutex::new(None));
        record_worker_error(&worker_error, "first".to_string());
        record_worker_error(&worker_error, "second".to_string());

        assert_eq!(worker_error.lock().expect("error lock is available").as_deref(), Some("first"));
    }

    #[test]
    fn worker_exits_cleanly_when_task_channel_disconnects() {
        let (sender, receiver) = bounded::<OutputWriteTask>(1);
        drop(sender);

        run_output_writer_worker(receiver);
    }
}
