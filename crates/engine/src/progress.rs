//! Throttled association-run progress reporting.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g_genotype::ChunkSpec;
use g_runtime::{TelemetryRunError, TelemetryRunSession};
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
    #[error("Run progress observation panicked.")]
    ObserverPanicked,
    #[error(transparent)]
    Telemetry(#[from] TelemetryRunError),
}

pub(crate) struct DeliveryProgress {
    reporter: Arc<RunProgressReporter>,
    group: Arc<ProgressGroupEntry>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ProgressTotals {
    chunk_count: u64,
    variant_count: u64,
}

impl DeliveryProgress {
    pub(crate) fn initialize(&self, pending_chunk_specs: &[ChunkSpec]) {
        if self.reporter.is_disabled() {
            return;
        }
        self.reporter.observe_bookkeeping(|| self.try_initialize(pending_chunk_specs));
    }

    fn try_initialize(&self, pending_chunk_specs: &[ChunkSpec]) -> Result<(), RunProgressError> {
        let pending_totals = ProgressTotals::try_from_chunk_specs(pending_chunk_specs)?;
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
        if self.reporter.is_disabled() {
            return Ok(());
        }
        let progress =
            progress.insert(ProgressGroup { completed_chunk_count, completed_variant_count, last_emit_at: None });
        self.reporter.try_emit_if_due(&self.group, progress, false)
    }

    pub(crate) fn record_writer_accepted(&self, variant_count: usize) {
        if self.reporter.is_disabled() {
            return;
        }
        self.reporter.observe_bookkeeping(|| self.try_record_writer_accepted(variant_count));
    }

    fn try_record_writer_accepted(&self, variant_count: usize) -> Result<(), RunProgressError> {
        let mut progress = self.group.progress.lock().map_err(|_| RunProgressError::StateLockPoisoned)?;
        if self.reporter.is_disabled() {
            return Ok(());
        }
        let progress = progress
            .as_mut()
            .ok_or_else(|| RunProgressError::UninitializedGroup { group_name: self.group.display_name.clone() })?;
        let completed_chunk_count =
            progress.completed_chunk_count.checked_add(1).ok_or(RunProgressError::CounterOverflow)?;
        let completed_variant_count = progress
            .completed_variant_count
            .checked_add(u64::try_from(variant_count).map_err(|_| RunProgressError::CounterOverflow)?)
            .ok_or(RunProgressError::CounterOverflow)?;
        if completed_chunk_count > self.group.totals.chunk_count
            || completed_variant_count > self.group.totals.variant_count
        {
            return Err(RunProgressError::CounterOverflow);
        }
        progress.completed_chunk_count = completed_chunk_count;
        progress.completed_variant_count = completed_variant_count;
        self.reporter.try_emit_if_due(&self.group, progress, false)
    }
}

pub(crate) struct RunProgressReporter {
    telemetry_session: TelemetryRunSession,
    thread_name: String,
    association_mode: g_plan::AssociationMode,
    started_at: Instant,
    groups: Mutex<Vec<Arc<ProgressGroupEntry>>>,
    disabled: AtomicBool,
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
            disabled: AtomicBool::new(false),
        }
    }

    pub(crate) fn totals_from_chunk_specs(&self, chunk_specs: &[ChunkSpec]) -> ProgressTotals {
        if self.is_disabled() {
            return ProgressTotals::default();
        }
        self.observe_bookkeeping(|| ProgressTotals::try_from_chunk_specs(chunk_specs)).unwrap_or_default()
    }

    pub(crate) fn totals_from_chunk_plan<Error, Plan>(&self, plan: Plan) -> Option<ProgressTotals>
    where
        Error: fmt::Display,
        Plan: FnOnce() -> Result<Vec<ChunkSpec>, Error>,
    {
        if self.is_disabled() {
            return None;
        }
        match catch_unwind(AssertUnwindSafe(plan)) {
            Ok(Ok(chunk_specs)) => {
                let totals = self.totals_from_chunk_specs(&chunk_specs);
                (!self.is_disabled()).then_some(totals)
            }
            Ok(Err(error)) => {
                self.disable_after_error(&error);
                None
            }
            Err(_) => {
                self.disable_after_error(&RunProgressError::ObserverPanicked);
                None
            }
        }
    }

    pub(crate) fn register_delivery(
        self: &Arc<Self>,
        display_name: String,
        totals: ProgressTotals,
    ) -> DeliveryProgress {
        let group = Arc::new(ProgressGroupEntry { display_name, totals, progress: Mutex::new(None) });
        if !self.is_disabled() {
            self.observe_bookkeeping(|| self.try_register_delivery(&group));
        }
        DeliveryProgress { reporter: Arc::clone(self), group }
    }

    fn try_register_delivery(&self, group: &Arc<ProgressGroupEntry>) -> Result<(), RunProgressError> {
        let mut groups = self.groups.lock().map_err(|_| RunProgressError::StateLockPoisoned)?;
        if self.is_disabled() {
            return Ok(());
        }
        groups.push(Arc::clone(group));
        Ok(())
    }

    /// Emit a final update for every initialized phenotype group.
    pub(crate) fn finish(&self) {
        if self.is_disabled() {
            return;
        }
        self.observe_bookkeeping(|| self.try_finish());
    }

    fn try_finish(&self) -> Result<(), RunProgressError> {
        let groups = self.groups.lock().map_err(|_| RunProgressError::StateLockPoisoned)?;
        for group in groups.iter() {
            if self.is_disabled() {
                return Ok(());
            }
            let mut progress = group.progress.lock().map_err(|_| RunProgressError::StateLockPoisoned)?;
            let progress = progress
                .as_mut()
                .ok_or_else(|| RunProgressError::UninitializedGroup { group_name: group.display_name.clone() })?;
            self.try_emit_if_due(group, progress, true)?;
        }
        Ok(())
    }

    fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Acquire)
    }

    fn observe_bookkeeping<Value, Operation>(&self, operation: Operation) -> Option<Value>
    where
        Operation: FnOnce() -> Result<Value, RunProgressError>,
    {
        if self.is_disabled() {
            return None;
        }
        match catch_unwind(AssertUnwindSafe(operation)) {
            Ok(Ok(value)) => Some(value),
            Ok(Err(error)) => {
                self.disable_after_error(&error);
                None
            }
            Err(_) => {
                self.disable_after_error(&RunProgressError::ObserverPanicked);
                None
            }
        }
    }

    fn disable_after_error<Error>(&self, error: &Error)
    where
        Error: fmt::Display + ?Sized,
    {
        if !self.disabled.swap(true, Ordering::AcqRel) {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                tracing::warn!(
                    target: "g.engine",
                    error = %error,
                    "Run progress reporting failed and is disabled for the remainder of this run."
                );
            }));
        }
    }

    fn try_emit_if_due(
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
        self.telemetry_session.emit_current_event(&self.thread_name, PROGRESS_EVENT_NAME, "info", &fields)?;
        Ok(())
    }
}

impl ProgressTotals {
    fn try_from_chunk_specs(chunk_specs: &[ChunkSpec]) -> Result<Self, RunProgressError> {
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
        let progress_reporter = reporter();
        let totals = progress_reporter.totals_from_chunk_specs(&[chunk(0, 4), chunk(4, 9), chunk(12, 12)]);
        assert_eq!(totals.chunk_count, 3);
        assert_eq!(totals.variant_count, 9);
        assert!(!progress_reporter.is_disabled());

        let invalid_reporter = reporter();
        let invalid_totals = invalid_reporter.totals_from_chunk_specs(&[chunk(5, 4)]);
        assert_eq!(invalid_totals.chunk_count, 0);
        assert_eq!(invalid_totals.variant_count, 0);
        assert!(invalid_reporter.is_disabled());
    }

    #[test]
    fn delivery_progress_accounts_for_resumed_chunks_and_writer_acceptance() {
        let reporter = reporter();
        let totals = reporter.totals_from_chunk_specs(&[chunk(0, 4), chunk(4, 6)]);
        let delivery = reporter.register_delivery("trait".to_string(), totals);

        delivery.initialize(&[chunk(4, 6)]);
        {
            let progress = delivery.group.progress.lock().expect("test progress lock is available");
            let progress = progress.as_ref().expect("progress is initialized");
            assert_eq!(progress.completed_chunk_count, 1);
            assert_eq!(progress.completed_variant_count, 4);
            assert!(progress.last_emit_at.is_some());
        }
        delivery.record_writer_accepted(2);
        {
            let progress = delivery.group.progress.lock().expect("test progress lock is available");
            let progress = progress.as_ref().expect("progress remains initialized");
            assert_eq!(progress.completed_chunk_count, 2);
            assert_eq!(progress.completed_variant_count, 6);
        }
        reporter.finish();
        assert!(!reporter.is_disabled());
    }

    #[test]
    fn first_bookkeeping_failure_disables_all_later_progress_operations() {
        let reporter = reporter();
        let totals = reporter.totals_from_chunk_specs(&[chunk(0, 2)]);
        let delivery = reporter.register_delivery("trait".to_string(), totals);

        delivery.initialize(&[chunk(0, 3)]);
        assert!(reporter.is_disabled());

        delivery.initialize(&[chunk(0, 2)]);
        delivery.record_writer_accepted(2);
        reporter.finish();
        let ignored_delivery =
            reporter.register_delivery("ignored-after-disable".to_string(), ProgressTotals::default());

        assert!(delivery.group.progress.lock().expect("test progress lock is available").is_none());
        assert!(ignored_delivery.group.progress.lock().expect("ignored progress lock is available").is_none());
        assert_eq!(reporter.groups.lock().expect("test group lock is available").len(), 1);
    }

    #[test]
    fn delivery_progress_counter_overflow_disables_without_mutating_counts() {
        let reporter = reporter();
        let totals = ProgressTotals { chunk_count: u64::MAX, variant_count: u64::MAX };
        let delivery = reporter.register_delivery("trait".to_string(), totals);
        delivery.initialize(&[]);
        delivery.record_writer_accepted(1);

        assert!(reporter.is_disabled());
        let progress = delivery.group.progress.lock().expect("test progress lock is available");
        let progress = progress.as_ref().expect("progress initialized before overflow");
        assert_eq!(progress.completed_chunk_count, u64::MAX);
        assert_eq!(progress.completed_variant_count, u64::MAX);
    }

    #[test]
    fn final_progress_failure_disables_reporter_and_later_initialization_is_ignored() {
        let reporter = reporter();
        let delivery = reporter.register_delivery("trait".to_string(), ProgressTotals::default());

        reporter.finish();
        assert!(reporter.is_disabled());
        delivery.initialize(&[]);
        assert!(delivery.group.progress.lock().expect("test progress lock is available").is_none());
    }

    #[test]
    fn panicking_progress_operation_is_contained_and_disables_later_operations() {
        let reporter = reporter();
        let observation = reporter.observe_bookkeeping(|| -> Result<(), RunProgressError> {
            panic!("intentional progress observer panic");
        });
        assert!(observation.is_none());
        assert!(reporter.is_disabled());

        let mut later_operation_ran = false;
        let later_observation = reporter.observe_bookkeeping(|| {
            later_operation_ran = true;
            Ok(())
        });
        assert!(later_observation.is_none());
        assert!(!later_operation_ran);
    }
}
