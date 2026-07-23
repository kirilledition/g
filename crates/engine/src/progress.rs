//! Throttled association-run progress reporting.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g_genotype::ChunkSpec;
use g_runtime::TelemetryRunSession;
use serde::Serialize;

const PROGRESS_EMIT_INTERVAL: Duration = Duration::from_secs(5);
const PROGRESS_EVENT_NAME: &str = "run_progress";
const PERCENT_SCALE_MICROUNITS: u64 = 100_000_000;
const MICROUNITS_PER_PERCENT: f64 = 1_000_000.0;

#[derive(Serialize)]
struct ProgressTelemetryFields<'fields> {
    group: &'fields str,
    mode: &'fields str,
    completed_chunks: u64,
    total_chunks: u64,
    completed_variants: u64,
    total_variants: u64,
    percent: f64,
    elapsed_seconds: f32,
    #[serde(rename = "final")]
    final_update: bool,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum RunProgressError {
    #[error("Progress counter overflowed uint64 capacity.")]
    CounterOverflow,
    #[error("Run progress state mutex was poisoned.")]
    StateLockPoisoned,
    #[error("Progress group '{group_name}' was not initialized.")]
    UninitializedGroup { group_name: String },
}

pub(crate) struct DeliveryProgress {
    reporter: Arc<RunProgressReporter>,
    group: Arc<ProgressGroupEntry>,
}

#[derive(Clone, Copy)]
pub(crate) struct ProgressTotals {
    chunk_count: u64,
    variant_count: u64,
}

impl DeliveryProgress {
    pub(crate) fn initialize(&self, pending_chunk_specs: &[ChunkSpec]) -> Result<(), RunProgressError> {
        let pending_totals = ProgressTotals::from_chunk_specs(pending_chunk_specs)?;
        let completed_chunk_count = self
            .group
            .totals
            .chunk_count
            .checked_sub(pending_totals.chunk_count)
            .ok_or(RunProgressError::CounterOverflow)?;
        let completed_variant_count = self
            .group
            .totals
            .variant_count
            .checked_sub(pending_totals.variant_count)
            .ok_or(RunProgressError::CounterOverflow)?;
        let mut progress = self.group.progress.lock().map_err(|_| RunProgressError::StateLockPoisoned)?;
        let progress =
            progress.insert(ProgressGroup { completed_chunk_count, completed_variant_count, last_emit_at: None });
        self.reporter.emit_if_due(&self.group, progress, false)
    }

    pub(crate) fn record_writer_accepted(&self, variant_count: usize) -> Result<(), RunProgressError> {
        let mut progress = self.group.progress.lock().map_err(|_| RunProgressError::StateLockPoisoned)?;
        let progress = progress
            .as_mut()
            .ok_or_else(|| RunProgressError::UninitializedGroup { group_name: self.group.display_name.clone() })?;
        progress.completed_chunk_count =
            progress.completed_chunk_count.checked_add(1).ok_or(RunProgressError::CounterOverflow)?;
        progress.completed_variant_count = progress
            .completed_variant_count
            .checked_add(u64::try_from(variant_count).map_err(|_| RunProgressError::CounterOverflow)?)
            .ok_or(RunProgressError::CounterOverflow)?;
        self.reporter.emit_if_due(&self.group, progress, false)
    }
}

pub(crate) struct RunProgressReporter {
    telemetry_session: TelemetryRunSession,
    thread_name: String,
    association_mode: g_plan::AssociationMode,
    started_at: Instant,
    groups: Mutex<Vec<Arc<ProgressGroupEntry>>>,
}

struct ProgressGroupEntry {
    display_name: String,
    totals: ProgressTotals,
    progress: Mutex<Option<ProgressGroup>>,
}

struct ProgressGroup {
    completed_chunk_count: u64,
    completed_variant_count: u64,
    last_emit_at: Option<Instant>,
}

impl RunProgressReporter {
    #[must_use]
    pub(crate) fn new(
        telemetry_session: TelemetryRunSession,
        thread_name: String,
        association_mode: g_plan::AssociationMode,
    ) -> Self {
        Self {
            telemetry_session,
            thread_name,
            association_mode,
            started_at: Instant::now(),
            groups: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn register_delivery(
        self: &Arc<Self>,
        display_name: String,
        totals: ProgressTotals,
    ) -> Result<DeliveryProgress, RunProgressError> {
        let group = Arc::new(ProgressGroupEntry { display_name, totals, progress: Mutex::new(None) });
        self.groups.lock().map_err(|_| RunProgressError::StateLockPoisoned)?.push(Arc::clone(&group));
        Ok(DeliveryProgress { reporter: Arc::clone(self), group })
    }

    /// Emit a final update for every initialized phenotype group.
    ///
    /// # Errors
    ///
    /// Returns an error when progress state is unavailable or counters cannot be represented.
    pub(crate) fn finish(&self) -> Result<(), RunProgressError> {
        let groups = self.groups.lock().map_err(|_| RunProgressError::StateLockPoisoned)?;
        for group in groups.iter() {
            let mut progress = group.progress.lock().map_err(|_| RunProgressError::StateLockPoisoned)?;
            let progress = progress
                .as_mut()
                .ok_or_else(|| RunProgressError::UninitializedGroup { group_name: group.display_name.clone() })?;
            self.emit_if_due(group, progress, true)?;
        }
        Ok(())
    }

    fn emit_if_due(
        &self,
        group: &ProgressGroupEntry,
        progress_group: &mut ProgressGroup,
        final_update: bool,
    ) -> Result<(), RunProgressError> {
        let now = Instant::now();
        if !final_update
            && progress_group
                .last_emit_at
                .is_some_and(|last_emit_at| now.duration_since(last_emit_at) < PROGRESS_EMIT_INTERVAL)
        {
            return Ok(());
        }
        progress_group.last_emit_at = Some(now);
        let elapsed_seconds = self.started_at.elapsed().as_secs_f32();
        let percent = if group.totals.chunk_count == 0 {
            100.0_f64
        } else {
            let completed_chunk_count = progress_group.completed_chunk_count.min(group.totals.chunk_count);
            let scaled_percent = u128::from(completed_chunk_count) * u128::from(PERCENT_SCALE_MICROUNITS)
                / u128::from(group.totals.chunk_count);
            f64::from(u32::try_from(scaled_percent).map_err(|_| RunProgressError::CounterOverflow)?)
                / MICROUNITS_PER_PERCENT
        };
        let fields = ProgressTelemetryFields {
            group: &group.display_name,
            mode: self.association_mode.as_str(),
            completed_chunks: progress_group.completed_chunk_count,
            total_chunks: group.totals.chunk_count,
            completed_variants: progress_group.completed_variant_count,
            total_variants: group.totals.variant_count,
            percent,
            elapsed_seconds,
            final_update,
        };
        if let Err(error) =
            self.telemetry_session.emit_current_event(&self.thread_name, PROGRESS_EVENT_NAME, "info", &fields)
        {
            tracing::warn!(
                target: "g.engine",
                error = %error,
                telemetry_event = PROGRESS_EVENT_NAME,
                "Failed to emit native run progress telemetry event."
            );
        }
        Ok(())
    }
}

impl ProgressTotals {
    pub(crate) fn from_chunk_specs(chunk_specs: &[ChunkSpec]) -> Result<Self, RunProgressError> {
        let mut variant_count = 0_u64;
        for chunk_spec in chunk_specs {
            let chunk_variant_count = chunk_spec
                .variant_stop_index
                .checked_sub(chunk_spec.variant_start_index)
                .ok_or(RunProgressError::CounterOverflow)?;
            variant_count = variant_count
                .checked_add(u64::try_from(chunk_variant_count).map_err(|_| RunProgressError::CounterOverflow)?)
                .ok_or(RunProgressError::CounterOverflow)?;
        }
        Ok(Self {
            chunk_count: u64::try_from(chunk_specs.len()).map_err(|_| RunProgressError::CounterOverflow)?,
            variant_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(start: usize, stop: usize) -> ChunkSpec {
        ChunkSpec { variant_start_index: start, variant_stop_index: stop }
    }

    fn reporter() -> Arc<RunProgressReporter> {
        Arc::new(RunProgressReporter::new(
            TelemetryRunSession::default(),
            "test-thread".to_string(),
            g_plan::AssociationMode::Regenie2Binary,
        ))
    }

    #[test]
    fn totals_accumulate_exact_chunk_and_variant_counts() {
        let totals = ProgressTotals::from_chunk_specs(&[chunk(0, 4), chunk(4, 9), chunk(12, 12)])
            .expect("valid chunk totals are computed");
        assert_eq!(totals.chunk_count, 3);
        assert_eq!(totals.variant_count, 9);
        assert!(matches!(ProgressTotals::from_chunk_specs(&[chunk(5, 4)]), Err(RunProgressError::CounterOverflow)));
    }

    #[test]
    fn delivery_progress_accounts_for_resumed_chunks_and_writer_acceptance() {
        let reporter = reporter();
        let totals =
            ProgressTotals::from_chunk_specs(&[chunk(0, 4), chunk(4, 6)]).expect("full delivery totals are valid");
        let delivery = reporter.register_delivery("trait".to_string(), totals).expect("delivery registration succeeds");
        assert!(matches!(
            delivery.record_writer_accepted(1),
            Err(RunProgressError::UninitializedGroup { group_name }) if group_name == "trait"
        ));

        delivery.initialize(&[chunk(4, 6)]).expect("resume-aware progress initializes");
        {
            let progress = delivery.group.progress.lock().expect("test progress lock is available");
            let progress = progress.as_ref().expect("progress is initialized");
            assert_eq!(progress.completed_chunk_count, 1);
            assert_eq!(progress.completed_variant_count, 4);
            assert!(progress.last_emit_at.is_some());
        }
        delivery.record_writer_accepted(2).expect("writer acceptance advances progress");
        {
            let progress = delivery.group.progress.lock().expect("test progress lock is available");
            let progress = progress.as_ref().expect("progress remains initialized");
            assert_eq!(progress.completed_chunk_count, 2);
            assert_eq!(progress.completed_variant_count, 6);
        }
        reporter.finish().expect("final progress update succeeds");
    }

    #[test]
    fn delivery_progress_rejects_pending_totals_above_the_registered_total() {
        let reporter = reporter();
        let totals = ProgressTotals::from_chunk_specs(&[chunk(0, 2)]).expect("full totals are valid");
        let delivery = reporter.register_delivery("trait".to_string(), totals).expect("delivery registration succeeds");
        assert!(matches!(delivery.initialize(&[chunk(0, 3)]), Err(RunProgressError::CounterOverflow)));
        assert!(
            matches!(reporter.finish(), Err(RunProgressError::UninitializedGroup { group_name }) if group_name == "trait")
        );
    }

    #[test]
    fn delivery_progress_detects_counter_overflow() {
        let reporter = reporter();
        let totals = ProgressTotals { chunk_count: u64::MAX, variant_count: u64::MAX };
        let delivery = reporter.register_delivery("trait".to_string(), totals).expect("delivery registration succeeds");
        delivery.initialize(&[]).expect("zero pending chunks initialize completed progress");
        assert!(matches!(delivery.record_writer_accepted(1), Err(RunProgressError::CounterOverflow)));
    }
}
