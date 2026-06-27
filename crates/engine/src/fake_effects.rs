//! Deterministic side-effect model for Rust-only coordinator tests.

use crate::backend::AssociationBatchResult;
use crate::effects::{EngineEffectError, EngineEffectOperation, EngineRunEffects};
use crate::phase::RunPhase;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FakeOutputState {
    pub written_results: Vec<AssociationBatchResult>,
    pub aborted_phase: Option<RunPhase>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FakeRunEffectState {
    pub phase_events: Vec<RunPhase>,
    pub completed_operations: Vec<EngineEffectOperation>,
    pub output: FakeOutputState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FakeEngineRunEffects {
    failure: Option<EngineEffectOperation>,
    state: FakeRunEffectState,
}

impl FakeEngineRunEffects {
    #[must_use]
    pub fn succeed() -> Self {
        Self { failure: None, state: FakeRunEffectState::default() }
    }

    #[must_use]
    pub fn fail_at(failure: EngineEffectOperation) -> Self {
        Self { failure: Some(failure), state: FakeRunEffectState::default() }
    }

    #[must_use]
    pub fn state(&self) -> &FakeRunEffectState {
        &self.state
    }

    fn maybe_fail(&self, operation: EngineEffectOperation) -> Result<(), EngineEffectError> {
        if self.failure == Some(operation) {
            return Err(EngineEffectError::new(format!("fake engine side effect failed during {operation}")));
        }
        Ok(())
    }
}

impl Default for FakeEngineRunEffects {
    fn default() -> Self {
        Self::succeed()
    }
}

impl EngineRunEffects for FakeEngineRunEffects {
    fn emit_phase_event(&mut self, phase: RunPhase) -> Result<(), EngineEffectError> {
        self.state.phase_events.push(phase);
        self.maybe_fail(EngineEffectOperation::TelemetryEvent)
    }

    fn open_inputs(&mut self) -> Result<(), EngineEffectError> {
        self.state.completed_operations.push(EngineEffectOperation::InputOpen);
        self.maybe_fail(EngineEffectOperation::InputOpen)
    }

    fn align_inputs(&mut self) -> Result<(), EngineEffectError> {
        self.state.completed_operations.push(EngineEffectOperation::InputAlignment);
        self.maybe_fail(EngineEffectOperation::InputAlignment)
    }

    fn validate_preflight(&mut self) -> Result<(), EngineEffectError> {
        self.state.completed_operations.push(EngineEffectOperation::PreflightValidation);
        self.maybe_fail(EngineEffectOperation::PreflightValidation)
    }

    fn validate_output_compatibility(&mut self) -> Result<(), EngineEffectError> {
        self.state.completed_operations.push(EngineEffectOperation::OutputCompatibility);
        self.maybe_fail(EngineEffectOperation::OutputCompatibility)
    }

    fn construct_writers(&mut self) -> Result<(), EngineEffectError> {
        self.state.completed_operations.push(EngineEffectOperation::WriterConstruction);
        self.maybe_fail(EngineEffectOperation::WriterConstruction)
    }

    fn write_batch_result(&mut self, result: &AssociationBatchResult) -> Result<(), EngineEffectError> {
        self.state.completed_operations.push(EngineEffectOperation::OutputWrite);
        self.state.output.written_results.push(result.clone());
        self.maybe_fail(EngineEffectOperation::OutputWrite)
    }

    fn drain_writers(&mut self) -> Result<(), EngineEffectError> {
        self.state.completed_operations.push(EngineEffectOperation::WriterDrain);
        self.maybe_fail(EngineEffectOperation::WriterDrain)
    }

    fn finalize_outputs(&mut self) -> Result<(), EngineEffectError> {
        self.state.completed_operations.push(EngineEffectOperation::OutputFinalization);
        self.maybe_fail(EngineEffectOperation::OutputFinalization)
    }

    fn abort_outputs(&mut self, phase: RunPhase) {
        self.state.output.aborted_phase = Some(phase);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_effects_record_successful_lifecycle_state() {
        let mut effects = FakeEngineRunEffects::succeed();
        let result = AssociationBatchResult::new("chr1".to_string(), 3, 5.0);

        effects.emit_phase_event(RunPhase::InputsOpened).unwrap();
        effects.open_inputs().unwrap();
        effects.align_inputs().unwrap();
        effects.validate_preflight().unwrap();
        effects.validate_output_compatibility().unwrap();
        effects.construct_writers().unwrap();
        effects.write_batch_result(&result).unwrap();
        effects.drain_writers().unwrap();
        effects.finalize_outputs().unwrap();

        assert_eq!(effects.state.phase_events, vec![RunPhase::InputsOpened]);
        assert_eq!(
            effects.state.completed_operations,
            vec![
                EngineEffectOperation::InputOpen,
                EngineEffectOperation::InputAlignment,
                EngineEffectOperation::PreflightValidation,
                EngineEffectOperation::OutputCompatibility,
                EngineEffectOperation::WriterConstruction,
                EngineEffectOperation::OutputWrite,
                EngineEffectOperation::WriterDrain,
                EngineEffectOperation::OutputFinalization,
            ],
        );
        assert_eq!(effects.state.output.written_results, vec![result]);
        assert_eq!(effects.state.output.aborted_phase, None);
    }

    #[test]
    fn fake_effects_inject_operation_failure_after_state_update() {
        let mut effects = FakeEngineRunEffects::fail_at(EngineEffectOperation::WriterConstruction);

        let error = effects.construct_writers().unwrap_err();

        assert_eq!(error.message(), "fake engine side effect failed during writer_construction");
        assert_eq!(effects.state.completed_operations, vec![EngineEffectOperation::WriterConstruction]);
    }
}
