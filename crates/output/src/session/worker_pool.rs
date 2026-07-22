use std::sync::{Arc, Condvar, Mutex};
use std::thread::{Builder, JoinHandle};

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::error::OutputError;
use crate::persistence::model::OutputChunkCommit;
use crate::timing::OutputStageTimingAccumulator;
use crate::writer::{RegenieStep2ChunkWriteBatch, write_regenie_step2_chunk_job};

use super::writer_session::OutputWriterConfig;

struct OutputWriteTask {
    payload: OutputWriteTaskPayload,
    completion_ticket: OutputWriteCompletionTicket,
}

struct OutputWriteTaskPayload {
    write_batch: RegenieStep2ChunkWriteBatch,
    config: Arc<OutputWriterConfig>,
    worker_error: Arc<Mutex<Option<String>>>,
    worker_commits: Arc<Mutex<Vec<OutputChunkCommit>>>,
    stage_timings: Arc<Mutex<OutputStageTimingAccumulator>>,
}

struct OutputWriterAdmission {
    sender: Mutex<Option<Sender<OutputWriteTask>>>,
}

struct OutputWriterSendPermit {
    sender: Sender<OutputWriteTask>,
}

#[derive(Clone)]
pub(super) struct OutputWriterClient {
    admission: Arc<OutputWriterAdmission>,
}

pub(crate) struct OutputWriterResourceOwner {
    admission: Arc<OutputWriterAdmission>,
    worker_handles: Vec<JoinHandle<()>>,
}

pub(super) struct StartedOutputWriterPool {
    pub(super) client: OutputWriterClient,
    pub(super) resource_owner: OutputWriterResourceOwner,
}

#[derive(Clone)]
pub(super) struct OutputWriteCompletionTracker {
    inner: Arc<(Mutex<usize>, Condvar)>,
}

pub(super) struct OutputWriteCompletionTicket {
    completion_tracker: OutputWriteCompletionTracker,
}

impl OutputWriterResourceOwner {
    pub(super) fn validate_settings(worker_count: usize, queue_depth: usize) -> Result<(), OutputError> {
        if worker_count == 0 {
            return Err(OutputError::InvalidInput("Writer thread count must be at least 1.".to_string()));
        }
        if queue_depth == 0 {
            return Err(OutputError::InvalidInput("Writer queue depth must be at least 1.".to_string()));
        }
        Ok(())
    }

    pub(super) fn start(worker_count: usize, queue_depth: usize) -> Result<StartedOutputWriterPool, OutputError> {
        Self::start_with_spawner(worker_count, queue_depth, |worker_index, receiver| {
            Builder::new()
                .name(format!("g-output-writer-{worker_index}"))
                .spawn(move || run_output_writer_worker(receiver))
        })
    }

    fn start_with_spawner(
        worker_count: usize,
        queue_depth: usize,
        mut spawn_worker: impl FnMut(usize, Receiver<OutputWriteTask>) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<StartedOutputWriterPool, OutputError> {
        Self::validate_settings(worker_count, queue_depth)?;
        let (sender, receiver) = bounded(queue_depth);
        let admission = Arc::new(OutputWriterAdmission { sender: Mutex::new(Some(sender)) });
        let mut worker_handles = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let receiver_clone = receiver.clone();
            match spawn_worker(worker_index, receiver_clone) {
                Ok(worker_handle) => worker_handles.push(worker_handle),
                Err(error) => {
                    let _ = admission.close();
                    drop(receiver);
                    let _ = join_worker_handles(&mut worker_handles);
                    return Err(OutputError::Runtime(format!(
                        "Failed to spawn Rust output writer worker {worker_index}: {error}"
                    )));
                }
            }
        }
        drop(receiver);
        Ok(StartedOutputWriterPool {
            client: OutputWriterClient { admission: Arc::clone(&admission) },
            resource_owner: Self { admission, worker_handles },
        })
    }

    pub(crate) fn shutdown_and_join(&mut self) -> Result<(), OutputError> {
        let close_result = self.admission.close();
        let join_result = join_worker_handles(&mut self.worker_handles);
        close_result.and(join_result)
    }
}

impl Drop for OutputWriterResourceOwner {
    fn drop(&mut self) {
        let _ = self.admission.close();
        for worker_handle in self.worker_handles.drain(..) {
            if worker_handle.is_finished() {
                let _ = worker_handle.join();
            }
        }
    }
}

impl OutputWriterAdmission {
    fn acquire_send_permit(&self) -> Result<OutputWriterSendPermit, OutputError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer admission lock was poisoned.".to_string()))?;
        sender
            .as_ref()
            .cloned()
            .map(|sender| OutputWriterSendPermit { sender })
            .ok_or_else(|| OutputError::Runtime("Rust output writer pool is closed.".to_string()))
    }

    fn close(&self) -> Result<(), OutputError> {
        let (mut sender, lock_was_poisoned) = match self.sender.lock() {
            Ok(sender) => (sender, false),
            Err(poisoned_error) => (poisoned_error.into_inner(), true),
        };
        sender.take();
        if lock_was_poisoned {
            Err(OutputError::Runtime("Rust output writer admission lock was poisoned.".to_string()))
        } else {
            Ok(())
        }
    }
}

impl OutputWriterSendPermit {
    fn send(self, write_task: OutputWriteTask) -> Result<(), crossbeam_channel::SendError<OutputWriteTask>> {
        self.sender.send(write_task)
    }
}

impl OutputWriterClient {
    pub(super) fn enqueue_regenie_step2(
        &self,
        write_batch: RegenieStep2ChunkWriteBatch,
        config: &Arc<OutputWriterConfig>,
        worker_error: &Arc<Mutex<Option<String>>>,
        worker_commits: &Arc<Mutex<Vec<OutputChunkCommit>>>,
        stage_timings: &Arc<Mutex<OutputStageTimingAccumulator>>,
        completion_ticket: OutputWriteCompletionTicket,
    ) -> Result<(), OutputError> {
        let send_permit = self.admission.acquire_send_permit().inspect_err(|error| {
            record_worker_error(worker_error, error.to_string());
        })?;
        let write_task = OutputWriteTask {
            payload: OutputWriteTaskPayload {
                write_batch,
                config: Arc::clone(config),
                worker_error: Arc::clone(worker_error),
                worker_commits: Arc::clone(worker_commits),
                stage_timings: Arc::clone(stage_timings),
            },
            completion_ticket,
        };
        if let Err(send_error) = send_permit.send(write_task) {
            let error_message = format!("Rust output writer task queue disconnected: {send_error}");
            record_worker_error(worker_error, error_message.clone());
            drop(send_error);
            return Err(OutputError::Runtime(error_message));
        }
        Ok(())
    }
}

fn join_worker_handles(worker_handles: &mut Vec<JoinHandle<()>>) -> Result<(), OutputError> {
    let mut first_error = None;
    for worker_handle in worker_handles.drain(..) {
        let worker_name = worker_handle.thread().name().unwrap_or("unnamed-output-writer").to_string();
        if worker_handle.join().is_err() && first_error.is_none() {
            first_error = Some(OutputError::Runtime(format!("Rust output writer worker '{worker_name}' panicked.")));
        }
    }
    first_error.map_or(Ok(()), Err)
}

impl OutputWriteCompletionTracker {
    pub(super) fn new() -> Self {
        Self { inner: Arc::new((Mutex::new(0), Condvar::new())) }
    }

    pub(super) fn reserve(&self) -> Result<OutputWriteCompletionTicket, OutputError> {
        let (pending_write_count_lock, _) = &*self.inner;
        let mut pending_write_count = pending_write_count_lock
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer completion lock was poisoned.".to_string()))?;
        *pending_write_count = pending_write_count.checked_add(1).ok_or_else(|| {
            OutputError::Runtime("Rust output writer pending-write count overflowed platform capacity.".to_string())
        })?;
        Ok(OutputWriteCompletionTicket { completion_tracker: self.clone() })
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

impl Drop for OutputWriteCompletionTicket {
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
        let worker_error = Arc::clone(&output_write_task.payload.worker_error);
        let completion_ticket = output_write_task.completion_ticket;
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_output_write_task(output_write_task.payload);
        }))
        .is_err()
        {
            record_worker_error(&worker_error, "Rust output writer panicked.".to_string());
        }
        drop(completion_ticket);
    }
}

fn run_output_write_task(output_write_task: OutputWriteTaskPayload) {
    #[cfg(test)]
    output_write_task.config.pause_worker_before_write_for_test();
    let write_result = write_regenie_step2_chunk_job(
        &output_write_task.config.parts_directory,
        &output_write_task.config.transaction_identifier,
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
    use std::io;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::Builder;
    use std::time::Duration;

    use crossbeam_channel::bounded;

    use crate::persistence::model::OutputTransactionIdentifier;
    use crate::timing::OutputStageTimingAccumulator;
    use crate::writer::RegenieStep2ChunkWriteBatch;

    use super::{
        OutputWriteCompletionTracker, OutputWriteTask, OutputWriteTaskPayload, OutputWriterResourceOwner,
        StartedOutputWriterPool, record_worker_error, run_output_writer_worker,
    };
    use crate::session::writer_session::OutputWriterConfig;

    fn empty_write_task(completion_tracker: &OutputWriteCompletionTracker, chunk_file_name: &str) -> OutputWriteTask {
        OutputWriteTask {
            payload: OutputWriteTaskPayload {
                write_batch: RegenieStep2ChunkWriteBatch {
                    chunk_file_name: chunk_file_name.to_string(),
                    chunks: Vec::new(),
                },
                config: Arc::new(OutputWriterConfig {
                    run_directory: PathBuf::from("unused-run"),
                    parts_directory: PathBuf::from("unused-parts"),
                    transaction_identifier: OutputTransactionIdentifier::for_test("empty-write-task"),
                    collect_stage_timings: false,
                    worker_before_write_hook: Mutex::new(None),
                }),
                worker_error: Arc::new(Mutex::new(None)),
                worker_commits: Arc::new(Mutex::new(Vec::new())),
                stage_timings: Arc::new(Mutex::new(OutputStageTimingAccumulator::default())),
            },
            completion_ticket: completion_tracker.reserve().expect("test write ticket is reserved"),
        }
    }

    #[test]
    fn writer_pool_rejects_zero_workers_or_queue_depth() {
        let worker_error = OutputWriterResourceOwner::start(0, 1).err().expect("zero workers are rejected");
        assert!(worker_error.to_string().contains("Writer thread count must be at least 1"));
        let queue_error = OutputWriterResourceOwner::start(1, 0).err().expect("zero queue depth is rejected");
        assert!(queue_error.to_string().contains("Writer queue depth must be at least 1"));

        let mut writer_pool = OutputWriterResourceOwner::start(1, 1).expect("valid writer pool starts");
        assert_eq!(writer_pool.resource_owner.worker_handles[0].thread().name(), Some("g-output-writer-0"));
        writer_pool.resource_owner.shutdown_and_join().expect("valid writer pool stops");
    }

    #[test]
    fn completion_ticket_decrements_tracker_and_unblocks_wait() {
        let tracker = OutputWriteCompletionTracker::new();
        let ticket = tracker.reserve().expect("pending write is reserved");
        drop(ticket);
        tracker.wait().expect("zero pending writes return immediately");
    }

    #[test]
    fn central_shutdown_rejects_an_outliving_client_and_releases_its_ticket() {
        let StartedOutputWriterPool { client, mut resource_owner } =
            OutputWriterResourceOwner::start(1, 1).expect("writer pool starts");
        let retained_client = client.clone();
        resource_owner.shutdown_and_join().expect("resource owner closes admission and joins workers");
        let tracker = OutputWriteCompletionTracker::new();
        let ticket = tracker.reserve().expect("pending write is reserved");
        let config = Arc::new(OutputWriterConfig {
            run_directory: PathBuf::from("unused-run"),
            parts_directory: PathBuf::from("unused-parts"),
            transaction_identifier: OutputTransactionIdentifier::for_test("closed-client"),
            collect_stage_timings: false,
            worker_before_write_hook: Mutex::new(None),
        });
        let worker_error = Arc::new(Mutex::new(None));
        let worker_commits = Arc::new(Mutex::new(Vec::new()));
        let stage_timings = Arc::new(Mutex::new(OutputStageTimingAccumulator::default()));
        let write_batch =
            RegenieStep2ChunkWriteBatch { chunk_file_name: "unused.parquet".to_string(), chunks: Vec::new() };

        let error = retained_client
            .enqueue_regenie_step2(write_batch, &config, &worker_error, &worker_commits, &stage_timings, ticket)
            .expect_err("centrally closed admission rejects the task");

        assert!(error.to_string().contains("writer pool is closed"));
        assert!(
            worker_error
                .lock()
                .expect("worker error lock is available")
                .as_deref()
                .is_some_and(|message| message.contains("writer pool is closed"))
        );
        tracker.wait().expect("failed send releases its completion ticket");
    }

    #[test]
    fn nth_spawn_failure_closes_admission_and_joins_started_workers() {
        let exited_worker_count = Arc::new(AtomicUsize::new(0));
        let observed_exited_worker_count = Arc::clone(&exited_worker_count);
        let error = OutputWriterResourceOwner::start_with_spawner(4, 1, move |worker_index, receiver| {
            if worker_index == 2 {
                return Err(io::Error::other("deterministic spawn failure"));
            }
            let exited_worker_count = Arc::clone(&observed_exited_worker_count);
            Builder::new().name(format!("test-output-writer-{worker_index}")).spawn(move || {
                run_output_writer_worker(receiver);
                exited_worker_count.fetch_add(1, Ordering::SeqCst);
            })
        })
        .err()
        .expect("third worker spawn fails");

        assert!(error.to_string().contains("worker 2: deterministic spawn failure"));
        assert_eq!(exited_worker_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn spawn_failure_remains_primary_after_partial_cleanup_worker_panic() {
        let error = OutputWriterResourceOwner::start_with_spawner(2, 1, |worker_index, _receiver| {
            if worker_index == 1 {
                return Err(io::Error::other("primary spawn failure"));
            }
            Builder::new().name("panicking-partial-worker".to_string()).spawn(|| {
                panic!("cleanup join panic");
            })
        })
        .err()
        .expect("second worker spawn fails");

        assert!(error.to_string().contains("worker 1: primary spawn failure"));
    }

    #[test]
    fn explicit_shutdown_propagates_worker_join_panic_and_still_closes_admission() {
        let StartedOutputWriterPool { client, mut resource_owner } =
            OutputWriterResourceOwner::start_with_spawner(1, 1, |_worker_index, _receiver| {
                Builder::new().name("panicking-output-writer".to_string()).spawn(|| {
                    panic!("deterministic worker panic");
                })
            })
            .expect("panicking worker starts");

        let error = resource_owner.shutdown_and_join().expect_err("worker panic is propagated");
        assert!(error.to_string().contains("'panicking-output-writer' panicked"));
        assert!(client.admission.sender.lock().expect("admission lock remains available").is_none());
        assert!(resource_owner.worker_handles.is_empty());
    }

    #[test]
    fn resource_owner_drop_closes_admission_without_waiting_for_a_live_worker() {
        let (release_sender, release_receiver) = bounded::<()>(1);
        let (worker_exited_sender, worker_exited_receiver) = bounded::<()>(1);
        let StartedOutputWriterPool { client, resource_owner } =
            OutputWriterResourceOwner::start_with_spawner(1, 1, move |_worker_index, _receiver| {
                let release_receiver = release_receiver.clone();
                let worker_exited_sender = worker_exited_sender.clone();
                Builder::new().name("drop-test-output-writer".to_string()).spawn(move || {
                    release_receiver.recv().expect("test releases detached worker");
                    worker_exited_sender.send(()).expect("test observes detached worker exit");
                })
            })
            .expect("blocking test worker starts");
        let (drop_finished_sender, drop_finished_receiver) = bounded::<()>(1);
        let drop_handle = Builder::new()
            .name("output-resource-drop-test".to_string())
            .spawn(move || {
                drop(resource_owner);
                drop_finished_sender.send(()).expect("test observes resource drop");
            })
            .expect("drop test controller starts");

        drop_finished_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("resource drop does not wait for the live worker");
        assert!(client.admission.sender.lock().expect("admission lock remains available").is_none());
        release_sender.send(()).expect("detached worker is released");
        worker_exited_receiver.recv_timeout(Duration::from_secs(2)).expect("detached worker exits after release");
        drop_handle.join().expect("drop test controller does not panic");
    }

    #[test]
    fn full_queue_send_permit_does_not_block_close_and_late_permits_are_rejected() {
        let (release_worker_sender, release_worker_receiver) = bounded::<()>(1);
        let (worker_exited_sender, worker_exited_receiver) = bounded::<()>(1);
        let StartedOutputWriterPool { client, resource_owner } =
            OutputWriterResourceOwner::start_with_spawner(1, 1, move |_worker_index, receiver| {
                let release_worker_receiver = release_worker_receiver.clone();
                let worker_exited_sender = worker_exited_sender.clone();
                Builder::new().name("full-queue-output-writer".to_string()).spawn(move || {
                    release_worker_receiver.recv().expect("test releases queue consumer");
                    drop(receiver.recv().expect("worker receives the first admitted task"));
                    drop(receiver.recv().expect("worker receives the pre-close permitted task"));
                    worker_exited_sender.send(()).expect("test observes queue consumer exit");
                })
            })
            .expect("controlled writer pool starts");
        let completion_tracker = OutputWriteCompletionTracker::new();
        let observed_completion_tracker = completion_tracker.clone();
        let first_permit = client.admission.acquire_send_permit().expect("open admission grants first permit");
        assert!(first_permit.send(empty_write_task(&completion_tracker, "first.parquet")).is_ok());
        let second_permit = client.admission.acquire_send_permit().expect("open admission grants second permit");
        let (send_started_sender, send_started_receiver) = bounded::<()>(1);
        let (send_result_sender, send_result_receiver) = bounded::<bool>(1);
        let send_handle = Builder::new()
            .name("full-queue-output-send".to_string())
            .spawn(move || {
                send_started_sender.send(()).expect("test observes blocking send start");
                let send_succeeded =
                    second_permit.send(empty_write_task(&completion_tracker, "second.parquet")).is_ok();
                send_result_sender.send(send_succeeded).expect("test observes blocking send result");
            })
            .expect("blocking send controller starts");
        send_started_receiver.recv().expect("second permitted send starts");
        assert!(matches!(send_result_receiver.try_recv(), Err(crossbeam_channel::TryRecvError::Empty)));
        let (drop_finished_sender, drop_finished_receiver) = bounded::<()>(1);
        let drop_handle = Builder::new()
            .name("full-queue-output-drop".to_string())
            .spawn(move || {
                drop(resource_owner);
                drop_finished_sender.send(()).expect("test observes resource drop");
            })
            .expect("resource drop controller starts");

        drop_finished_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("resource close does not wait for the full queue send");
        let late_error =
            client.admission.acquire_send_permit().err().expect("post-close admission rejects a new sender permit");
        assert!(late_error.to_string().contains("writer pool is closed"));
        assert!(matches!(send_result_receiver.try_recv(), Err(crossbeam_channel::TryRecvError::Empty)));

        release_worker_sender.send(()).expect("queue consumer is released");
        assert!(
            send_result_receiver
                .recv_timeout(Duration::from_secs(2))
                .expect("pre-close permitted send completes after queue progress")
        );
        worker_exited_receiver.recv_timeout(Duration::from_secs(2)).expect("queue consumer exits");
        send_handle.join().expect("blocking send controller does not panic");
        drop_handle.join().expect("resource drop controller does not panic");
        observed_completion_tracker.wait().expect("both admitted task tickets are released");
    }

    #[test]
    fn worker_error_records_only_the_first_failure() {
        let worker_error = Arc::new(Mutex::new(None));
        record_worker_error(&worker_error, "first".to_string());
        record_worker_error(&worker_error, "second".to_string());

        assert_eq!(worker_error.lock().expect("error lock is available").as_deref(), Some("first"));
    }
}
