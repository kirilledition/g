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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn counter_snapshot_tracks_written_and_dropped_events() {
        let state = TelemetryEventCounterState::new(true);
        state.record_event_count(0);
        state.record_event_count(5);
        assert_eq!(
            state.counter_snapshot(2),
            TelemetryWriterCounterSnapshot {
                accepted_event_count: 5,
                written_event_count: 3,
                dropped_event_count: 2,
                queue_dropped_event_count: 2,
                lossy: true,
            }
        );
        assert_eq!(state.counter_snapshot(20).written_event_count, 0);
        assert_eq!(TelemetryWriterCounterSnapshot::empty().accepted_event_count, 0);
        assert!(TelemetryWriterCounterSnapshot::empty().lossy);
    }

    #[test]
    fn accepted_counter_is_thread_safe_and_saturating() {
        let state = Arc::new(TelemetryEventCounterState::new(false));
        let mut workers = Vec::new();
        for _worker_index in 0..4 {
            let worker_state = Arc::clone(&state);
            workers.push(std::thread::spawn(move || {
                for _event_index in 0..25 {
                    worker_state.record_event_count(1);
                }
            }));
        }
        for worker in workers {
            worker.join().expect("counter worker should complete");
        }
        assert_eq!(state.counter_snapshot(0).accepted_event_count, 100);

        state.accepted_event_count.store(usize::MAX, Ordering::Relaxed);
        state.record_event_count(1);
        let saturated = state.counter_snapshot(0);
        assert_eq!(saturated.accepted_event_count, u64::try_from(usize::MAX).unwrap_or(u64::MAX));
    }
}
