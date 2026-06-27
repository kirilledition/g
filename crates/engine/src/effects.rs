//! Native coordinator side-effect boundary.

use std::fmt;

use crate::backend::AssociationBatchResult;
use crate::phase::RunPhase;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineEffectOperation {
    TelemetryEvent,
    InputOpen,
    InputAlignment,
    PreflightValidation,
    OutputCompatibility,
    WriterConstruction,
    OutputWrite,
    WriterDrain,
    OutputFinalization,
}

impl EngineEffectOperation {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TelemetryEvent => "telemetry_event",
            Self::InputOpen => "input_open",
            Self::InputAlignment => "input_alignment",
            Self::PreflightValidation => "preflight_validation",
            Self::OutputCompatibility => "output_compatibility",
            Self::WriterConstruction => "writer_construction",
            Self::OutputWrite => "output_write",
            Self::WriterDrain => "writer_drain",
            Self::OutputFinalization => "output_finalization",
        }
    }
}

impl fmt::Display for EngineEffectOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct EngineEffectError {
    message: String,
}

impl EngineEffectError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

pub trait EngineRunEffects {
    /// Emit one phase transition event.
    ///
    /// # Errors
    ///
    /// Returns an error when telemetry cannot record the transition.
    fn emit_phase_event(&mut self, _phase: RunPhase) -> Result<(), EngineEffectError> {
        Ok(())
    }

    /// Open run inputs.
    ///
    /// # Errors
    ///
    /// Returns an error when input resources cannot be opened.
    fn open_inputs(&mut self) -> Result<(), EngineEffectError> {
        Ok(())
    }

    /// Align samples, phenotypes, covariates, and predictions.
    ///
    /// # Errors
    ///
    /// Returns an error when alignment fails.
    fn align_inputs(&mut self) -> Result<(), EngineEffectError> {
        Ok(())
    }

    /// Validate preflight constraints.
    ///
    /// # Errors
    ///
    /// Returns an error when preflight validation fails.
    fn validate_preflight(&mut self) -> Result<(), EngineEffectError> {
        Ok(())
    }

    /// Validate output compatibility before writer construction.
    ///
    /// # Errors
    ///
    /// Returns an error when output compatibility or resume validation fails.
    fn validate_output_compatibility(&mut self) -> Result<(), EngineEffectError> {
        Ok(())
    }

    /// Construct output writers.
    ///
    /// # Errors
    ///
    /// Returns an error when writers cannot be constructed.
    fn construct_writers(&mut self) -> Result<(), EngineEffectError> {
        Ok(())
    }

    /// Write one association batch result.
    ///
    /// # Errors
    ///
    /// Returns an error when the result cannot be written.
    fn write_batch_result(&mut self, _result: &AssociationBatchResult) -> Result<(), EngineEffectError> {
        Ok(())
    }

    /// Drain queued writer work.
    ///
    /// # Errors
    ///
    /// Returns an error when queued writer work cannot be drained.
    fn drain_writers(&mut self) -> Result<(), EngineEffectError> {
        Ok(())
    }

    /// Finalize output artifacts.
    ///
    /// # Errors
    ///
    /// Returns an error when final output artifacts cannot be finalized.
    fn finalize_outputs(&mut self) -> Result<(), EngineEffectError> {
        Ok(())
    }

    /// Abort initialized outputs after a failed or interrupted run.
    fn abort_outputs(&mut self, _phase: RunPhase) {}
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoopEngineRunEffects;

impl EngineRunEffects for NoopEngineRunEffects {}
