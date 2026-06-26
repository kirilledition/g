#![warn(clippy::pedantic)]

pub mod backend;
pub mod coordinator;
pub mod fake_backend;
pub mod phase;
pub mod pipeline;

pub use backend::{
    AssociationBackend, AssociationBatchResult, BackendError, GenotypeBatchView, PredictionView, PreparedGroupInput,
};
pub use coordinator::{EngineCoordinator, EngineError, EngineRunInput, EngineRunReport, InjectedCoordinatorFailure};
pub use fake_backend::{FakeBackend, FakeBackendFailure, FakeChromosomeState, FakeGroupState};
pub use phase::RunPhase;
pub use pipeline::Regenie2RunEngineCore;
