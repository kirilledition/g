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
pub use schedule::intersect_committed_chunk_identifier_sets;
