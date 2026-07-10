//! Native stage timing recorder state and aggregate bookkeeping.

pub(crate) mod diagnostics;
mod final_outputs;
mod payloads;
mod queue_backpressure;
mod recorder;
mod state;
mod transfer_metadata;

pub use final_outputs::{
    FinalTimingOutputContext, FinalTimingOutputsWriteResultPayload, FinalTimingOutputsWriteStartedDiagnosticPayload,
    StageTimingRecorderPlan, TimingFileError, TimingFileWritePlan,
    build_final_timing_outputs_write_started_diagnostic_payload, plan_stage_timing_recorder, plan_timing_file_write,
    resolve_final_timing_output_context, serialize_final_timing_outputs_write_started_diagnostic_fields_json,
};
pub use payloads::{
    ChunkStageSummary, ChunkStageTiming, NullLogisticDiagnosticValue, NullLogisticSummary, NumericDiagnosticValue,
    ProfileSummaryPayload, StageTimingSnapshotPayload,
};
pub use queue_backpressure::{QueueBackpressureAccumulator, QueueBackpressureKey, QueueBackpressureSnapshot};
pub use recorder::StageTimingRecorder;
pub use state::StageTimingState;
pub use transfer_metadata::{
    TransferMetadataAccumulator, TransferMetadataError, TransferMetadataKey, TransferMetadataObservation,
    TransferMetadataSnapshot, build_transfer_metadata_observation,
};
