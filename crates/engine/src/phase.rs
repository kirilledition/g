//! Explicit run state machine phases for native orchestration.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunPhase {
    Planned,
    InputsOpened,
    InputsAligned,
    PreflightValidated,
    OutputsInitialized,
    Running,
    Draining,
    Finalizing,
    Completed,
    Interrupted,
    Failed,
}

impl RunPhase {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::InputsOpened => "inputs_opened",
            Self::InputsAligned => "inputs_aligned",
            Self::PreflightValidated => "preflight_validated",
            Self::OutputsInitialized => "outputs_initialized",
            Self::Running => "running",
            Self::Draining => "draining",
            Self::Finalizing => "finalizing",
            Self::Completed => "completed",
            Self::Interrupted => "interrupted",
            Self::Failed => "failed",
        }
    }
}

impl fmt::Display for RunPhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}
