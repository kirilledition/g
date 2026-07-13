use std::sync::atomic::{AtomicUsize, Ordering};

use serde::Serialize;

#[derive(Debug)]
pub(crate) struct TelemetryEventCounterState {
    accepted_event_count: AtomicUsize,
    lossy: bool,
}

#[derive(Debug, PartialEq, Serialize)]
pub(crate) struct TelemetryWriterCounterSnapshot {
    pub accepted_event_count: u64,
    pub written_event_count: u64,
    pub dropped_event_count: u64,
    pub queue_dropped_event_count: u64,
    pub lossy: bool,
}

impl TelemetryEventCounterState {
    #[must_use]
    pub(crate) const fn new(lossy: bool) -> Self {
        Self { accepted_event_count: AtomicUsize::new(0), lossy }
    }

    pub(crate) fn record_event_count(&self, event_count: usize) {
        if event_count > 0 {
            let _result = self
                .accepted_event_count
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| Some(value.saturating_add(event_count)));
        }
    }

    #[must_use]
    pub(crate) fn counter_snapshot(&self, queue_dropped_event_count: usize) -> TelemetryWriterCounterSnapshot {
        let accepted_event_count = self.accepted_event_count.load(Ordering::Acquire);
        TelemetryWriterCounterSnapshot {
            accepted_event_count: supported_usize_to_u64(accepted_event_count),
            written_event_count: supported_usize_to_u64(accepted_event_count.saturating_sub(queue_dropped_event_count)),
            dropped_event_count: supported_usize_to_u64(queue_dropped_event_count),
            queue_dropped_event_count: supported_usize_to_u64(queue_dropped_event_count),
            lossy: self.lossy,
        }
    }
}

impl TelemetryWriterCounterSnapshot {
    #[must_use]
    pub(crate) const fn empty() -> Self {
        Self {
            accepted_event_count: 0,
            written_event_count: 0,
            dropped_event_count: 0,
            queue_dropped_event_count: 0,
            lossy: true,
        }
    }
}

fn supported_usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
