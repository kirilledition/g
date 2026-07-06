//! Public engine error boundary.

use crate::backend::BackendError;
use crate::effects::{EngineEffectError, EngineEffectOperation};
use crate::phase::RunPhase;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum EngineError {
    #[error("backend failed during {phase}: {source}")]
    Backend { phase: RunPhase, source: BackendError },
    #[error("coordinator failed during {phase}: {message}")]
    Coordinator { phase: RunPhase, message: String },
    #[error("coordinator side effect {operation} failed during {phase}: {source}")]
    Effect { phase: RunPhase, operation: EngineEffectOperation, source: EngineEffectError },
    #[error("run interrupted during {phase}")]
    Interrupted { phase: RunPhase },
}

pub type EngineResult<T> = Result<T, EngineError>;
