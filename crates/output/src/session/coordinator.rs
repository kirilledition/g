use std::sync::{Arc, Mutex};

use crossbeam_channel::Receiver;

use crate::manifest;
use crate::timing::{OutputStageTimingAccumulator, start_optional_timing};
use crate::writer::{RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, build_output_file_name};

use super::OutputWriterConfig;
use super::worker_pool::{OutputWriteCompletionTracker, OutputWriterPool, push_worker_error};

pub(super) enum OutputCoordinatorJob {
    RegenieStep2(Box<RegenieStep2ChunkJob>),
    Finish,
    Abort,
}

#[allow(clippy::needless_pass_by_value)]
pub(super) fn run_output_writer_coordinator(
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
    writer_pool.enqueue_regenie_step2(
        write_batch,
        config.clone(),
        worker_errors,
        worker_commits,
        stage_timings,
        completion_tracker,
    )?;
    if let Some(start_time) = flush_start_time {
        let mut stage_timings_guard = stage_timings.lock().map_err(|_| {
            push_worker_error(worker_errors, "Rust output writer stage timing lock was poisoned.".to_string());
        })?;
        stage_timings_guard.coordinator_flush_seconds += start_time.elapsed().as_secs_f64();
        stage_timings_guard.coordinator_flush_count += 1;
    }
    Ok(())
}
