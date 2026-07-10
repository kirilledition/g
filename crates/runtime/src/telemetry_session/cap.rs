use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use serde::Serialize;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TelemetryCapAction {
    Write,
    Drop,
}

#[derive(Debug)]
pub struct TelemetryEventCapState {
    path: PathBuf,
    event_cap: Option<usize>,
    lossy: bool,
    written_event_count: AtomicUsize,
    dropped_event_count: AtomicUsize,
    exceeded: AtomicBool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TelemetryWriterCounterSnapshot {
    pub accepted_event_count: usize,
    pub written_event_count: usize,
    pub dropped_event_count: usize,
    pub cap_dropped_event_count: usize,
    pub queue_dropped_event_count: usize,
    pub event_cap_exceeded: bool,
    pub lossy: bool,
    pub event_cap: Option<usize>,
}

impl TelemetryEventCapState {
    #[must_use]
    pub fn new(path: &Path, event_cap: Option<usize>, lossy: bool) -> Self {
        Self {
            path: path.to_path_buf(),
            event_cap,
            lossy,
            written_event_count: AtomicUsize::new(0),
            dropped_event_count: AtomicUsize::new(0),
            exceeded: AtomicBool::new(false),
        }
    }

    #[must_use]
    pub fn has_event_cap(&self) -> bool {
        self.event_cap.is_some()
    }

    pub fn record_uncapped_event_count(&self, event_count: usize) {
        if event_count > 0 {
            self.written_event_count.fetch_add(event_count, Ordering::Relaxed);
        }
    }

    /// Reserve one event under the configured trace cap.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when a lossless capped writer has already reached
    /// its configured event limit.
    pub fn reserve_event(&self) -> io::Result<TelemetryCapAction> {
        let Some(event_cap) = self.event_cap else {
            self.written_event_count.fetch_add(1, Ordering::Relaxed);
            return Ok(TelemetryCapAction::Write);
        };

        loop {
            let written_event_count = self.written_event_count.load(Ordering::Acquire);
            if written_event_count >= event_cap {
                self.mark_exceeded();
                if self.lossy {
                    self.dropped_event_count.fetch_add(1, Ordering::Relaxed);
                    return Ok(TelemetryCapAction::Drop);
                }
                return Err(io::Error::other(self.cap_exceeded_error_message()));
            }
            if self
                .written_event_count
                .compare_exchange_weak(
                    written_event_count,
                    written_event_count + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(TelemetryCapAction::Write);
            }
        }
    }

    #[must_use]
    pub fn should_fail_for_cap_exceeded(&self) -> bool {
        self.exceeded.load(Ordering::Acquire) && !self.lossy
    }

    #[must_use]
    pub fn counter_snapshot(&self, queue_dropped_event_count: usize) -> TelemetryWriterCounterSnapshot {
        let accepted_event_count = self.written_event_count.load(Ordering::Acquire);
        let cap_dropped_event_count = self.dropped_event_count.load(Ordering::Acquire);
        TelemetryWriterCounterSnapshot {
            accepted_event_count,
            written_event_count: accepted_event_count.saturating_sub(queue_dropped_event_count),
            dropped_event_count: cap_dropped_event_count.saturating_add(queue_dropped_event_count),
            cap_dropped_event_count,
            queue_dropped_event_count,
            event_cap_exceeded: self.exceeded.load(Ordering::Acquire),
            lossy: self.lossy,
            event_cap: self.event_cap,
        }
    }

    #[must_use]
    pub fn cap_exceeded_error_message(&self) -> String {
        let event_cap = self.event_cap.unwrap_or(0);
        format!(
            "Trace telemetry event cap exceeded at {event_cap} events for {}. \
             Increase [diagnostics].trace_event_cap or set it to 0 to disable the cap for intentional deep traces. \
             Set [diagnostics].log_lossy = true to drop events after the cap instead of failing.",
            self.path.display()
        )
    }

    #[must_use]
    pub fn cap_exceeded_drop_message(&self) -> String {
        let event_cap = self.event_cap.unwrap_or(0);
        format!(
            "Trace telemetry event cap reached at {event_cap} events for {}; dropping additional trace events because log_lossy is enabled.",
            self.path.display()
        )
    }

    fn mark_exceeded(&self) {
        if !self.exceeded.swap(true, Ordering::AcqRel) && self.lossy {
            tracing::warn!(
                target: "g.logging",
                g_event = "native_telemetry_event_cap_exceeded",
                event_cap = self.event_cap.unwrap_or(0),
                lossy = self.lossy,
                path = %self.path.display(),
                message = %self.cap_exceeded_drop_message(),
                "Tracing writer reached event cap and started dropping events."
            );
        }
    }
}

impl TelemetryWriterCounterSnapshot {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            accepted_event_count: 0,
            written_event_count: 0,
            dropped_event_count: 0,
            cap_dropped_event_count: 0,
            queue_dropped_event_count: 0,
            event_cap_exceeded: false,
            lossy: true,
            event_cap: None,
        }
    }
}
