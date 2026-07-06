mod accumulator;
mod snapshot;

use std::time::Instant;

pub(crate) use accumulator::OutputStageTimingAccumulator;
pub(crate) use snapshot::write_stage_timing_snapshot;

pub(crate) fn start_optional_timing(collect_stage_timings: bool) -> Option<Instant> {
    collect_stage_timings.then(Instant::now)
}
