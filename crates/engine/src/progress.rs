//! Throttled association-run progress reporting.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g_genotype::ChunkSpec;
use g_runtime::{TelemetryRunError, TelemetryRunSession};

const PROGRESS_EMIT_INTERVAL: Duration = Duration::from_secs(5);
const PROGRESS_EVENT_NAME: &str = "run_progress";

#[derive(Debug, thiserror::Error)]
pub enum RunProgressError {
    #[error("Progress counter overflowed uint64 capacity.")]
    CounterOverflow,
    #[error("Progress group '{group_name}' was not initialized.")]
    MissingGroup { group_name: String },
    #[error(transparent)]
    Telemetry(#[from] TelemetryRunError),
}

#[derive(Clone)]
pub(crate) struct DeliveryProgress {
    reporter: Arc<RunProgressReporter>,
    group_name: String,
    association_mode: String,
}

impl DeliveryProgress {
    pub(crate) fn new(
        reporter: Arc<RunProgressReporter>,
        group_name: String,
        association_mode: String,
    ) -> Self {
        Self { reporter, group_name, association_mode }
    }

    pub(crate) fn initialize(
        &self,
        all_chunk_specs: &[ChunkSpec],
        pending_chunk_specs: &[ChunkSpec],
    ) -> Result<(), RunProgressError> {
        self.reporter.initialize_group(&self.group_name, &self.association_mode, all_chunk_specs, pending_chunk_specs)
    }

    pub(crate) fn record_writer_accepted(&self, variant_count: usize) -> Result<(), RunProgressError> {
        self.reporter.record_writer_accepted(&self.group_name, variant_count)
    }
}

pub struct RunProgressReporter {
    telemetry_session: TelemetryRunSession,
    thread_name: String,
    started_at: Instant,
    groups: Mutex<BTreeMap<String, ProgressGroup>>,
}

struct ProgressGroup {
    association_mode: String,
    total_chunk_count: u64,
    completed_chunk_count: u64,
    total_variant_count: u64,
    completed_variant_count: u64,
    last_emit_at: Option<Instant>,
}

impl RunProgressReporter {
    #[must_use]
    pub fn new(telemetry_session: TelemetryRunSession, thread_name: String) -> Self {
        Self {
            telemetry_session,
            thread_name,
            started_at: Instant::now(),
            groups: Mutex::new(BTreeMap::new()),
        }
    }

    pub(crate) fn initialize_group(
        &self,
        group_name: &str,
        association_mode: &str,
        all_chunk_specs: &[ChunkSpec],
        pending_chunk_specs: &[ChunkSpec],
    ) -> Result<(), RunProgressError> {
        let (total_chunk_count, total_variant_count) = count_chunks(all_chunk_specs)?;
        let (pending_chunk_count, pending_variant_count) = count_chunks(pending_chunk_specs)?;
        let completed_chunk_count = total_chunk_count.checked_sub(pending_chunk_count).ok_or(RunProgressError::CounterOverflow)?;
        let completed_variant_count =
            total_variant_count.checked_sub(pending_variant_count).ok_or(RunProgressError::CounterOverflow)?;
        let mut groups = self.groups.lock().map_err(|_| RunProgressError::CounterOverflow)?;
        let progress_group = groups.entry(group_name.to_string()).or_insert_with(|| ProgressGroup {
            association_mode: association_mode.to_string(),
            total_chunk_count,
            completed_chunk_count,
            total_variant_count,
            completed_variant_count,
            last_emit_at: None,
        });
        progress_group.association_mode = association_mode.to_string();
        progress_group.total_chunk_count = total_chunk_count;
        progress_group.completed_chunk_count = completed_chunk_count;
        progress_group.total_variant_count = total_variant_count;
        progress_group.completed_variant_count = completed_variant_count;
        self.emit_if_due(group_name, progress_group, true)
    }

    pub(crate) fn record_writer_accepted(
        &self,
        group_name: &str,
        variant_count: usize,
    ) -> Result<(), RunProgressError> {
        let mut groups = self.groups.lock().map_err(|_| RunProgressError::CounterOverflow)?;
        let progress_group = groups
            .get_mut(group_name)
            .ok_or_else(|| RunProgressError::MissingGroup { group_name: group_name.to_string() })?;
        progress_group.completed_chunk_count =
            progress_group.completed_chunk_count.checked_add(1).ok_or(RunProgressError::CounterOverflow)?;
        progress_group.completed_variant_count = progress_group
            .completed_variant_count
            .checked_add(u64::try_from(variant_count).map_err(|_| RunProgressError::CounterOverflow)?)
            .ok_or(RunProgressError::CounterOverflow)?;
        self.emit_if_due(group_name, progress_group, false)
    }

    /// Emit a final update for every initialized phenotype group.
    ///
    /// # Errors
    ///
    /// Returns an error when telemetry output cannot be written.
    pub fn finish(&self) -> Result<(), RunProgressError> {
        let mut groups = self.groups.lock().map_err(|_| RunProgressError::CounterOverflow)?;
        for (group_name, progress_group) in groups.iter_mut() {
            self.emit_if_due(group_name, progress_group, true)?;
        }
        Ok(())
    }

    fn emit_if_due(
        &self,
        group_name: &str,
        progress_group: &mut ProgressGroup,
        force: bool,
    ) -> Result<(), RunProgressError> {
        let now = Instant::now();
        if !force && progress_group.last_emit_at.is_some_and(|last_emit_at| now.duration_since(last_emit_at) < PROGRESS_EMIT_INTERVAL)
        {
            return Ok(());
        }
        progress_group.last_emit_at = Some(now);
        let elapsed_seconds = self.started_at.elapsed().as_secs_f32();
        let percent = if progress_group.total_chunk_count == 0 {
            100.0_f32
        } else {
            (progress_group.completed_chunk_count as f32 / progress_group.total_chunk_count as f32) * 100.0_f32
        };
        let fields = serde_json::json!({
            "group": group_name,
            "mode": progress_group.association_mode,
            "completed_chunks": progress_group.completed_chunk_count,
            "total_chunks": progress_group.total_chunk_count,
            "completed_variants": progress_group.completed_variant_count,
            "total_variants": progress_group.total_variant_count,
            "percent": percent,
            "elapsed_seconds": elapsed_seconds,
            "final": force,
        });
        self.telemetry_session.emit_current_event(&self.thread_name, PROGRESS_EVENT_NAME, "info", &fields)?;
        tracing::info!(
            target: "g.progress",
            g_event = PROGRESS_EVENT_NAME,
            group = group_name,
            mode = progress_group.association_mode,
            completed_chunks = progress_group.completed_chunk_count,
            total_chunks = progress_group.total_chunk_count,
            completed_variants = progress_group.completed_variant_count,
            total_variants = progress_group.total_variant_count,
            percent,
            elapsed_seconds,
            final_update = force,
            "run progress"
        );
        Ok(())
    }
}

fn count_chunks(chunk_specs: &[ChunkSpec]) -> Result<(u64, u64), RunProgressError> {
    let mut total_variant_count = 0_u64;
    for chunk_spec in chunk_specs {
        let variant_count = chunk_spec
            .variant_stop_index
            .checked_sub(chunk_spec.variant_start_index)
            .ok_or(RunProgressError::CounterOverflow)?;
        total_variant_count = total_variant_count
            .checked_add(u64::try_from(variant_count).map_err(|_| RunProgressError::CounterOverflow)?)
            .ok_or(RunProgressError::CounterOverflow)?;
    }
    Ok((u64::try_from(chunk_specs.len()).map_err(|_| RunProgressError::CounterOverflow)?, total_variant_count))
}
