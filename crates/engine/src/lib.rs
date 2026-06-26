#![warn(clippy::pedantic)]

pub mod backend;
pub mod coordinator;
pub mod fake_backend;
pub mod phase;
pub mod pipeline;
pub mod preflight;
pub mod schedule;

pub use backend::{
    AssociationBackend, AssociationBatchResult, BackendError, GenotypeBatchView, PredictionView, PreparedGroupInput,
};
pub use coordinator::{EngineCoordinator, EngineError, EngineRunInput, EngineRunReport, InjectedCoordinatorFailure};
pub use fake_backend::{FakeBackend, FakeBackendFailure, FakeChromosomeState, FakeGroupState};
pub use phase::RunPhase;
pub use pipeline::Regenie2RunEngineCore;
pub use preflight::{
    PreflightError, PreflightReportPayload, build_preflight_report_payload, build_preflight_warnings,
    resolve_scanned_variant_count,
};
pub use schedule::{
    BgenDeliveryMethod, CallbackWorkerLifecycleState, CallbackWorkerShutdownTimeouts, DosageBufferPoolState,
    DosageBufferReusePlan, NativeCallbackQueueLimits, ResultInFlightSlotState, ScheduleError,
    VariantMajorDosageBatchHandoffPlan, callback_worker_shutdown_timeouts, intersect_committed_chunk_identifier_sets,
    plan_dosage_buffer_reuse, plan_variant_major_dosage_batch_handoff, resolve_bgen_delivery_method,
    resolve_delivery_callback_batch_size, resolve_grouped_union_callback_batch_size,
    resolve_native_callback_queue_limits, resolve_writer_finish_thread_count,
};
