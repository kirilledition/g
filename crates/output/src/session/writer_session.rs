use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow::array::ArrayRef;

use crate::chunk::NativeChunkHandle;
use crate::error::OutputError;
use crate::manifest;
use crate::timing::{OutputStageTimingAccumulator, start_optional_timing, write_stage_timing_snapshot};
use crate::writer::{RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, build_part_file_name};

use super::worker_pool::{
    OutputWriteCompletionTicket, OutputWriteCompletionTracker, OutputWriterClient, OutputWriterResourceOwner,
};

#[derive(Clone, Copy)]
enum OutputWriterCloseKind {
    Finish,
    Interrupted,
    Abort,
}

impl OutputWriterCloseKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Finish => "finish",
            Self::Interrupted => "interrupted finish",
            Self::Abort => "abort",
        }
    }
}

enum OutputWriterSessionState {
    Open {
        pending_chunks: Vec<RegenieStep2ChunkJob>,
        #[cfg(test)]
        detached_before_send_hook: Option<Arc<DetachedBeforeSendTestHook>>,
    },
    Closing(OutputWriterCloseKind),
    Closed,
}

struct ReservedWriteBatch {
    write_batch: RegenieStep2ChunkWriteBatch,
    completion_ticket: OutputWriteCompletionTicket,
    #[cfg(test)]
    detached_before_send_hook: Option<Arc<DetachedBeforeSendTestHook>>,
}

#[cfg(test)]
struct DetachedBeforeSendTestHook {
    detached_sender: crossbeam_channel::Sender<()>,
    release_receiver: crossbeam_channel::Receiver<()>,
}

#[cfg(test)]
pub(crate) struct DetachedBeforeSendTestControl {
    detached_receiver: crossbeam_channel::Receiver<()>,
    release_sender: crossbeam_channel::Sender<()>,
}

#[cfg(test)]
pub(super) struct WorkerBeforeWriteTestHook {
    started_sender: crossbeam_channel::Sender<()>,
    release_receiver: crossbeam_channel::Receiver<()>,
}

#[cfg(test)]
pub(crate) struct WorkerBeforeWriteTestControl {
    started_receiver: crossbeam_channel::Receiver<()>,
    release_sender: crossbeam_channel::Sender<()>,
}

pub(super) struct OutputWriterConfig {
    pub(super) run_directory: PathBuf,
    pub(super) parts_directory: PathBuf,
    pub(super) collect_stage_timings: bool,
    #[cfg(test)]
    pub(super) worker_before_write_hook: Mutex<Option<Arc<WorkerBeforeWriteTestHook>>>,
}

pub struct OutputWriterSession {
    state: Mutex<OutputWriterSessionState>,
    writer_client: OutputWriterClient,
    worker_error: Arc<Mutex<Option<String>>>,
    worker_commits: Arc<Mutex<Vec<manifest::RunManifestChunkCommit>>>,
    stage_timings: Arc<Mutex<OutputStageTimingAccumulator>>,
    completion_tracker: OutputWriteCompletionTracker,
    config: Arc<OutputWriterConfig>,
}

impl OutputWriterSession {
    fn new_with_writer_client(config: OutputWriterConfig, writer_client: OutputWriterClient) -> Self {
        let worker_error = Arc::new(Mutex::new(None));
        let worker_commits = Arc::new(Mutex::new(Vec::new()));
        let stage_timings = Arc::new(Mutex::new(OutputStageTimingAccumulator::default()));
        let completion_tracker = OutputWriteCompletionTracker::new();
        Self {
            state: Mutex::new(OutputWriterSessionState::Open {
                pending_chunks: Vec::with_capacity(crate::CHUNKS_PER_PARQUET_FILE),
                #[cfg(test)]
                detached_before_send_hook: None,
            }),
            writer_client,
            worker_error,
            worker_commits,
            stage_timings,
            completion_tracker,
            config: Arc::new(config),
        }
    }

    pub(crate) fn finish(&self) -> Result<(), OutputError> {
        let finish_start_time = start_optional_timing(self.config.collect_stage_timings);
        let reserved_write_batch = self.begin_flush_close(OutputWriterCloseKind::Finish)?;
        let finish_result = (|| {
            self.flush_and_commit(reserved_write_batch)?;
            manifest::mark_run_manifest_completed(&self.config.run_directory)?;
            self.record_finish_timing(finish_start_time)?;
            self.write_stage_timing_snapshot()
        })();
        self.complete_close(finish_result)
    }

    pub(crate) fn finish_interrupted(&self, signal_name: &str) -> Result<(), OutputError> {
        let finish_start_time = start_optional_timing(self.config.collect_stage_timings);
        let reserved_write_batch = self.begin_flush_close(OutputWriterCloseKind::Interrupted)?;
        let finish_result = (|| {
            self.flush_and_commit(reserved_write_batch)?;
            manifest::mark_run_manifest_interrupted(&self.config.run_directory, signal_name)?;
            self.record_finish_timing(finish_start_time)?;
            self.write_stage_timing_snapshot()
        })();
        self.complete_close(finish_result)
    }

    pub(crate) fn abort(&self) -> Result<(), OutputError> {
        self.begin_abort_close()?;
        let abort_result = self.completion_tracker.wait();
        self.complete_close(abort_result)
    }

    fn take_worker_commits(&self) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
        let mut worker_commits = self
            .worker_commits
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer commit lock was poisoned.".to_string()))?;
        Ok(std::mem::take(&mut *worker_commits))
    }

    fn flush_and_commit(&self, reserved_write_batch: Option<ReservedWriteBatch>) -> Result<(), OutputError> {
        let enqueue_result = reserved_write_batch
            .map_or(Ok(()), |reserved_write_batch| self.enqueue_reserved_write_batch(reserved_write_batch));
        let wait_result = self.completion_tracker.wait();
        enqueue_result?;
        wait_result?;
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

    pub(crate) fn write_regenie2_native_chunk_handle_arrays(
        &self,
        chunk_handle: NativeChunkHandle,
        beta: ArrayRef,
        standard_error: ArrayRef,
        chi_squared: ArrayRef,
        log10_p_value: ArrayRef,
        correction_code: Option<ArrayRef>,
    ) -> Result<(), OutputError> {
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
        let reserved_write_batch = {
            let mut state = self.lock_state()?;
            match &mut *state {
                OutputWriterSessionState::Open {
                    pending_chunks,
                    #[cfg(test)]
                    detached_before_send_hook,
                } => {
                    let should_detach = pending_chunks.len().checked_add(1).ok_or_else(|| {
                        OutputError::Runtime("Rust output writer pending chunk count overflowed.".to_string())
                    })? >= crate::CHUNKS_PER_PARQUET_FILE;
                    let completion_ticket = should_detach.then(|| self.completion_tracker.reserve()).transpose()?;
                    pending_chunks.push(job);
                    completion_ticket.map(|completion_ticket| {
                        let chunks =
                            std::mem::replace(pending_chunks, Vec::with_capacity(crate::CHUNKS_PER_PARQUET_FILE));
                        ReservedWriteBatch {
                            write_batch: build_chunk_write_batch(chunks),
                            completion_ticket,
                            #[cfg(test)]
                            detached_before_send_hook: detached_before_send_hook.take(),
                        }
                    })
                }
                OutputWriterSessionState::Closing(close_kind) => {
                    return Err(session_closing_error(*close_kind));
                }
                OutputWriterSessionState::Closed => return Err(session_closed_error()),
            }
        };
        if let Some(reserved_write_batch) = reserved_write_batch {
            #[cfg(test)]
            reserved_write_batch.pause_before_send();
            self.enqueue_reserved_write_batch(reserved_write_batch)?;
        }
        if let Some(start_time) = enqueue_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.enqueue_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.enqueue_count = stage_timings.enqueue_count.saturating_add(1);
            })?;
        }
        Ok(())
    }

    fn enqueue_reserved_write_batch(&self, reserved_write_batch: ReservedWriteBatch) -> Result<(), OutputError> {
        let flush_start_time = start_optional_timing(self.config.collect_stage_timings);
        self.writer_client.enqueue_regenie_step2(
            reserved_write_batch.write_batch,
            &self.config,
            &self.worker_error,
            &self.worker_commits,
            &self.stage_timings,
            reserved_write_batch.completion_ticket,
        )?;
        if let Some(start_time) = flush_start_time {
            self.record_stage_timing(|stage_timings| {
                stage_timings.coordinator_flush_seconds += start_time.elapsed().as_secs_f64();
                stage_timings.coordinator_flush_count = stage_timings.coordinator_flush_count.saturating_add(1);
            })?;
        }
        Ok(())
    }

    fn begin_flush_close(&self, close_kind: OutputWriterCloseKind) -> Result<Option<ReservedWriteBatch>, OutputError> {
        let mut state = self.lock_state()?;
        let reserved_write_batch = match &mut *state {
            OutputWriterSessionState::Open { pending_chunks, .. } => {
                if pending_chunks.is_empty() {
                    None
                } else {
                    let completion_ticket = self.completion_tracker.reserve()?;
                    let chunks = std::mem::replace(pending_chunks, Vec::with_capacity(crate::CHUNKS_PER_PARQUET_FILE));
                    Some(ReservedWriteBatch {
                        write_batch: build_chunk_write_batch(chunks),
                        completion_ticket,
                        #[cfg(test)]
                        detached_before_send_hook: None,
                    })
                }
            }
            OutputWriterSessionState::Closing(active_close_kind) => {
                return Err(session_closing_error(*active_close_kind));
            }
            OutputWriterSessionState::Closed => return Err(session_closed_error()),
        };
        *state = OutputWriterSessionState::Closing(close_kind);
        Ok(reserved_write_batch)
    }

    fn begin_abort_close(&self) -> Result<(), OutputError> {
        let mut state = self.lock_state()?;
        match &mut *state {
            OutputWriterSessionState::Open { pending_chunks, .. } => pending_chunks.clear(),
            OutputWriterSessionState::Closing(close_kind) => return Err(session_closing_error(*close_kind)),
            OutputWriterSessionState::Closed => return Err(session_closed_error()),
        }
        *state = OutputWriterSessionState::Closing(OutputWriterCloseKind::Abort);
        Ok(())
    }

    fn complete_close(&self, close_result: Result<(), OutputError>) -> Result<(), OutputError> {
        let state_result = self.mark_closed();
        close_result.and(state_result)
    }

    fn mark_closed(&self) -> Result<(), OutputError> {
        let mut state = self.lock_state()?;
        match *state {
            OutputWriterSessionState::Closing(_) => {
                *state = OutputWriterSessionState::Closed;
                Ok(())
            }
            OutputWriterSessionState::Closed => Ok(()),
            OutputWriterSessionState::Open { .. } => Err(OutputError::Runtime(
                "Rust output writer close completed while its session was still open.".to_string(),
            )),
        }
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, OutputWriterSessionState>, OutputError> {
        self.state
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer session state lock was poisoned.".to_string()))
    }

    #[cfg(test)]
    pub(crate) fn install_detached_before_send_test_hook(&self) -> Result<DetachedBeforeSendTestControl, OutputError> {
        let (detached_sender, detached_receiver) = crossbeam_channel::bounded(1);
        let (release_sender, release_receiver) = crossbeam_channel::bounded(1);
        let mut state = self.lock_state()?;
        match &mut *state {
            OutputWriterSessionState::Open { detached_before_send_hook, .. } => {
                *detached_before_send_hook =
                    Some(Arc::new(DetachedBeforeSendTestHook { detached_sender, release_receiver }));
            }
            OutputWriterSessionState::Closing(close_kind) => return Err(session_closing_error(*close_kind)),
            OutputWriterSessionState::Closed => return Err(session_closed_error()),
        }
        Ok(DetachedBeforeSendTestControl { detached_receiver, release_sender })
    }

    #[cfg(test)]
    pub(crate) fn install_worker_before_write_test_hook(&self) -> Result<WorkerBeforeWriteTestControl, OutputError> {
        let (started_sender, started_receiver) = crossbeam_channel::bounded(1);
        let (release_sender, release_receiver) = crossbeam_channel::bounded(1);
        let mut worker_before_write_hook = self.config.worker_before_write_hook.lock().map_err(|_| {
            OutputError::Runtime("Rust output writer before-write test hook lock was poisoned.".to_string())
        })?;
        if worker_before_write_hook.is_some() {
            return Err(OutputError::Runtime(
                "Rust output writer before-write test hook is already installed.".to_string(),
            ));
        }
        *worker_before_write_hook = Some(Arc::new(WorkerBeforeWriteTestHook { started_sender, release_receiver }));
        Ok(WorkerBeforeWriteTestControl { started_receiver, release_sender })
    }

    #[cfg(test)]
    pub(crate) fn is_closing_for_test(&self) -> Result<bool, OutputError> {
        Ok(matches!(*self.lock_state()?, OutputWriterSessionState::Closing(_)))
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

#[cfg(test)]
impl OutputWriterConfig {
    pub(super) fn pause_worker_before_write_for_test(&self) {
        let worker_before_write_hook =
            self.worker_before_write_hook.lock().expect("before-write test hook lock remains available").take();
        if let Some(worker_before_write_hook) = worker_before_write_hook {
            worker_before_write_hook.started_sender.send(()).expect("before-write test control remains connected");
            worker_before_write_hook
                .release_receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("before-write test is released before its timeout");
        }
    }
}

#[cfg(test)]
impl ReservedWriteBatch {
    fn pause_before_send(&self) {
        if let Some(hook) = self.detached_before_send_hook.as_ref() {
            hook.detached_sender.send(()).expect("detach test control remains connected");
            hook.release_receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("detach test is released before its timeout");
        }
    }
}

#[cfg(test)]
impl DetachedBeforeSendTestControl {
    pub(crate) fn wait_until_detached(&self) {
        self.detached_receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("writer detaches a reserved batch before the test timeout");
    }

    pub(crate) fn release(&self) {
        self.release_sender.send(()).expect("paused writer remains connected");
    }
}

#[cfg(test)]
impl WorkerBeforeWriteTestControl {
    pub(crate) fn wait_until_worker_started(&self) {
        self.started_receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("worker reaches the before-write barrier before the test timeout");
    }

    pub(crate) fn release(&self) {
        self.release_sender.send(()).expect("paused output worker remains connected");
    }
}

fn session_closing_error(close_kind: OutputWriterCloseKind) -> OutputError {
    OutputError::Runtime(format!("Rust output writer session is already closing via {}.", close_kind.label()))
}

fn session_closed_error() -> OutputError {
    OutputError::Runtime("Rust output writer session is already closed.".to_string())
}

fn build_chunk_write_batch(chunks: Vec<RegenieStep2ChunkJob>) -> RegenieStep2ChunkWriteBatch {
    let first_chunk_identifier = chunks.first().map_or(0, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    let last_chunk_identifier =
        chunks.last().map_or(first_chunk_identifier, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
    let chunk_file_name = build_part_file_name(first_chunk_identifier, last_chunk_identifier);
    RegenieStep2ChunkWriteBatch { chunk_file_name, chunks }
}

pub(crate) fn validate_output_writer_settings(
    output_plan: &g_plan::OutputPlan,
    output_run_count: usize,
) -> Result<(), OutputError> {
    if output_run_count == 0 {
        return Ok(());
    }
    let writer_thread_count = usize::try_from(output_plan.writer_thread_count).map_err(OutputError::runtime)?;
    OutputWriterResourceOwner::validate_settings(writer_thread_count, crate::WRITER_QUEUE_DEPTH)
}

pub(crate) struct CreatedOutputWriterSessions {
    pub(crate) sessions: Vec<OutputWriterSession>,
    pub(crate) resource_owner: Option<OutputWriterResourceOwner>,
}

/// Create one native output writer session per run/parts directory pair.
///
/// # Errors
///
/// Returns an error when the directory vector lengths differ, writer settings
/// are invalid, or a writer pool cannot be created.
pub(crate) fn create_output_writer_sessions(
    run_directories: Vec<PathBuf>,
    parts_directories: Vec<PathBuf>,
    output_plan: &g_plan::OutputPlan,
    collect_stage_timings: bool,
) -> Result<CreatedOutputWriterSessions, OutputError> {
    if run_directories.len() != parts_directories.len() {
        return Err(OutputError::InvalidInput(format!(
            "Output writer run directory count ({}) does not match parts directory count ({}).",
            run_directories.len(),
            parts_directories.len()
        )));
    }
    validate_output_writer_settings(output_plan, run_directories.len())?;
    if run_directories.is_empty() {
        return Ok(CreatedOutputWriterSessions { sessions: Vec::new(), resource_owner: None });
    }
    let writer_thread_count = usize::try_from(output_plan.writer_thread_count).map_err(OutputError::runtime)?;
    let writer_pool = OutputWriterResourceOwner::start(writer_thread_count, crate::WRITER_QUEUE_DEPTH)?;
    let sessions = run_directories
        .into_iter()
        .zip(parts_directories)
        .map(|(run_directory, parts_directory)| {
            let config = OutputWriterConfig {
                run_directory,
                parts_directory,
                collect_stage_timings,
                #[cfg(test)]
                worker_before_write_hook: Mutex::new(None),
            };
            OutputWriterSession::new_with_writer_client(config, writer_pool.client.clone())
        })
        .collect();
    Ok(CreatedOutputWriterSessions { sessions, resource_owner: Some(writer_pool.resource_owner) })
}

/// Finish output writer sessions, optionally in bounded parallel batches.
///
/// # Errors
///
/// Returns an error when a session cannot be closed, a writer task failed, a
/// manifest commit fails or a finish thread panics.
pub(crate) fn finish_output_writer_sessions(
    writer_sessions: &[&OutputWriterSession],
    thread_count: usize,
) -> Result<(), OutputError> {
    if thread_count <= 1 {
        let mut first_error = None;
        for writer_session in writer_sessions {
            record_first_error(&mut first_error, writer_session.finish());
        }
        return first_error.map_or(Ok(()), Err);
    }
    let mut first_error = None;
    for writer_session_batch in writer_sessions.chunks(thread_count) {
        record_first_error(&mut first_error, finish_output_writer_session_batch(writer_session_batch));
    }
    first_error.map_or(Ok(()), Err)
}

/// Flush interrupted output writer sessions and mark their manifests interrupted.
///
/// # Errors
///
/// Returns an error when a session cannot be closed, a writer task failed, a
/// manifest update fails, or an interrupted-finish thread panics.
pub(crate) fn finish_interrupted_output_writer_sessions(
    writer_sessions: &[&OutputWriterSession],
    thread_count: usize,
    signal_name: &str,
) -> Result<(), OutputError> {
    if thread_count <= 1 {
        let mut first_error = None;
        for writer_session in writer_sessions {
            record_first_error(&mut first_error, writer_session.finish_interrupted(signal_name));
        }
        return first_error.map_or(Ok(()), Err);
    }
    let mut first_error = None;
    for writer_session_batch in writer_sessions.chunks(thread_count) {
        record_first_error(
            &mut first_error,
            finish_interrupted_output_writer_session_batch(writer_session_batch, signal_name),
        );
    }
    first_error.map_or(Ok(()), Err)
}

fn finish_output_writer_session_batch(writer_sessions: &[&OutputWriterSession]) -> Result<(), OutputError> {
    std::thread::scope(|scope| {
        let writer_handles = writer_sessions
            .iter()
            .map(|writer_session| scope.spawn(move || writer_session.finish()))
            .collect::<Vec<_>>();
        let mut first_error = None;
        for writer_handle in writer_handles {
            let finish_result = writer_handle
                .join()
                .map_err(|_| OutputError::Runtime("Output writer finish thread panicked.".to_string()))
                .and_then(|result| result);
            record_first_error(&mut first_error, finish_result);
        }
        first_error.map_or(Ok(()), Err)
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
        let mut first_error = None;
        for writer_handle in writer_handles {
            let finish_result = writer_handle
                .join()
                .map_err(|_| OutputError::Runtime("Interrupted output writer finish thread panicked.".to_string()))
                .and_then(|result| result);
            record_first_error(&mut first_error, finish_result);
        }
        first_error.map_or(Ok(()), Err)
    })
}

fn record_first_error(first_error: &mut Option<OutputError>, result: Result<(), OutputError>) {
    if let Err(error) = result
        && first_error.is_none()
    {
        *first_error = Some(error);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        OutputWriterSession, create_output_writer_sessions, finish_interrupted_output_writer_sessions,
        finish_output_writer_sessions,
    };

    fn output_plan(writer_thread_count: u32) -> g_plan::OutputPlan {
        g_plan::OutputPlan { output_run_root: "unused".to_string(), resume: false, writer_thread_count }
    }

    #[test]
    fn session_creation_validates_directory_geometry_and_writer_count() {
        let mismatch = create_output_writer_sessions(vec![PathBuf::from("run")], Vec::new(), &output_plan(1), false)
            .err()
            .expect("directory count mismatch is rejected");
        assert!(mismatch.to_string().contains("run directory count (1)"));

        let empty = create_output_writer_sessions(Vec::new(), Vec::new(), &output_plan(0), false)
            .expect("empty session set needs no workers");
        assert!(empty.sessions.is_empty());
        assert!(empty.resource_owner.is_none());

        let zero_workers = create_output_writer_sessions(
            vec![PathBuf::from("run")],
            vec![PathBuf::from("parts")],
            &output_plan(0),
            false,
        )
        .err()
        .expect("nonempty session set needs workers");
        assert!(zero_workers.to_string().contains("Writer thread count must be at least 1"));
    }

    #[test]
    fn empty_session_completion_is_valid_for_serial_and_parallel_paths() {
        let sessions: [&OutputWriterSession; 0] = [];
        finish_output_writer_sessions(&sessions, 1).expect("serial empty finish succeeds");
        finish_output_writer_sessions(&sessions, 2).expect("parallel empty finish succeeds");
        finish_interrupted_output_writer_sessions(&sessions, 1, "SIGTERM")
            .expect("serial empty interrupted finish succeeds");
        finish_interrupted_output_writer_sessions(&sessions, 2, "SIGTERM")
            .expect("parallel empty interrupted finish succeeds");
    }
}
