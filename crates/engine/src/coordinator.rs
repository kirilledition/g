//! Native run coordinator scaffold with explicit phase transitions.

mod types;

pub use types::{
    EngineChromosomeRunInput, EngineChromosomeRunReport, EngineGroupChromosomeInput, EngineGroupRunInput,
    EngineGroupRunReport, EngineRunInput, EngineRunReport, InjectedCoordinatorFailure,
};

use crate::backend::{AssociationBackend, BackendError};
use crate::effects::{EngineEffectError, EngineEffectOperation, EngineRunEffects, NoopEngineRunEffects};
use crate::error::{EngineError, EngineResult};
use crate::phase::RunPhase;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineCoordinator<Backend> {
    backend: Backend,
    phase: RunPhase,
    phase_history: Vec<RunPhase>,
    injected_failure: Option<InjectedCoordinatorFailure>,
    interruption_phase: Option<RunPhase>,
}

impl<Backend> EngineCoordinator<Backend> {
    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            phase: RunPhase::Planned,
            phase_history: vec![RunPhase::Planned],
            injected_failure: None,
            interruption_phase: None,
        }
    }

    #[must_use]
    pub fn with_injected_failure(mut self, failure: InjectedCoordinatorFailure) -> Self {
        self.injected_failure = Some(failure);
        self
    }

    #[must_use]
    pub const fn with_interruption(mut self, phase: RunPhase) -> Self {
        self.interruption_phase = Some(phase);
        self
    }

    #[must_use]
    pub const fn phase(&self) -> RunPhase {
        self.phase
    }

    #[must_use]
    pub fn phase_history(&self) -> &[RunPhase] {
        &self.phase_history
    }

    #[must_use]
    pub fn into_backend(self) -> Backend {
        self.backend
    }

    fn enter_phase<Effects>(&mut self, phase: RunPhase, effects: &mut Effects) -> EngineResult<()>
    where
        Effects: EngineRunEffects,
    {
        self.record_phase(phase);
        if let Err(source) = effects.emit_phase_event(phase) {
            return Err(self.record_effect_error(phase, EngineEffectOperation::TelemetryEvent, source, effects));
        }
        if self.interruption_phase == Some(phase) {
            self.record_phase(RunPhase::Interrupted);
            Self::abort_outputs_after_failure(phase, effects);
            return Err(EngineError::Interrupted { phase });
        }
        if let Some(failure) = self.injected_failure.as_ref()
            && failure.phase == phase
        {
            let message = failure.message.clone();
            self.record_phase(RunPhase::Failed);
            Self::abort_outputs_after_failure(phase, effects);
            return Err(EngineError::Coordinator { phase, message });
        }
        Ok(())
    }

    fn record_phase(&mut self, phase: RunPhase) {
        self.phase = phase;
        self.phase_history.push(phase);
    }

    fn record_backend_error(&mut self, phase: RunPhase, source: BackendError) -> EngineError {
        self.record_phase(RunPhase::Failed);
        EngineError::Backend { phase, source }
    }

    fn record_effect_error<Effects>(
        &mut self,
        phase: RunPhase,
        operation: EngineEffectOperation,
        source: EngineEffectError,
        effects: &mut Effects,
    ) -> EngineError
    where
        Effects: EngineRunEffects,
    {
        self.record_phase(RunPhase::Failed);
        Self::abort_outputs_after_failure(phase, effects);
        EngineError::Effect { phase, operation, source }
    }

    fn abort_outputs_after_failure<Effects>(phase: RunPhase, effects: &mut Effects)
    where
        Effects: EngineRunEffects,
    {
        if should_abort_outputs_after_failure(phase) {
            effects.abort_outputs(phase);
        }
    }
}

impl<Backend> EngineCoordinator<Backend>
where
    Backend: AssociationBackend,
{
    /// Execute a tiny single-batch run through the native coordinator.
    ///
    /// # Errors
    ///
    /// Returns an error when a coordinator phase is interrupted, a phase-level
    /// fault is injected, or the backend fails while preparing or computing.
    pub fn run_single_batch(&mut self, input: &EngineRunInput<'_>) -> EngineResult<EngineRunReport> {
        self.run_single_batch_with_effects(input, &mut NoopEngineRunEffects)
    }

    /// Execute a tiny single-batch run with explicit native side-effect hooks.
    ///
    /// # Errors
    ///
    /// Returns an error when a phase transition, side-effect hook, or backend
    /// operation fails.
    pub fn run_single_batch_with_effects<Effects>(
        &mut self,
        input: &EngineRunInput<'_>,
        effects: &mut Effects,
    ) -> EngineResult<EngineRunReport>
    where
        Effects: EngineRunEffects,
    {
        let chromosome_input =
            EngineChromosomeRunInput::new(input.group.clone(), input.chromosome, input.predictions, vec![input.batch]);
        let chromosome_report = self.run_chromosome_batches_with_effects(&chromosome_input, effects)?;
        let mut results = chromosome_report.results.into_iter();
        let Some(result) = results.next() else {
            return Err(EngineError::Coordinator {
                phase: RunPhase::Running,
                message: "single-batch coordinator produced no batch result".to_string(),
            });
        };
        Ok(EngineRunReport::new(chromosome_report.phase_history, result))
    }

    /// Execute one chromosome through a sequence of genotype batches.
    ///
    /// # Errors
    ///
    /// Returns an error when a coordinator phase is interrupted, a phase-level
    /// fault is injected, or the backend fails while preparing or computing.
    pub fn run_chromosome_batches(
        &mut self,
        input: &EngineChromosomeRunInput<'_>,
    ) -> EngineResult<EngineChromosomeRunReport> {
        self.run_chromosome_batches_with_effects(input, &mut NoopEngineRunEffects)
    }

    /// Execute one chromosome through a sequence of genotype batches with
    /// explicit native side-effect hooks.
    ///
    /// # Errors
    ///
    /// Returns an error when a phase transition, side-effect hook, or backend
    /// operation fails.
    pub fn run_chromosome_batches_with_effects<Effects>(
        &mut self,
        input: &EngineChromosomeRunInput<'_>,
        effects: &mut Effects,
    ) -> EngineResult<EngineChromosomeRunReport>
    where
        Effects: EngineRunEffects,
    {
        let group_input = EngineGroupRunInput::new(
            input.group.clone(),
            vec![EngineGroupChromosomeInput::new(input.chromosome, input.predictions, input.batches.clone())],
        );
        let group_report = self.run_group_chromosomes_with_effects(&group_input, effects)?;
        Ok(EngineChromosomeRunReport::new(group_report.phase_history, group_report.results))
    }

    /// Execute one group through a sequence of chromosomes and genotype batches.
    ///
    /// # Errors
    ///
    /// Returns an error when a coordinator phase is interrupted, a phase-level
    /// fault is injected, or the backend fails while preparing or computing.
    pub fn run_group_chromosomes(&mut self, input: &EngineGroupRunInput<'_>) -> EngineResult<EngineGroupRunReport> {
        self.run_group_chromosomes_with_effects(input, &mut NoopEngineRunEffects)
    }

    /// Execute one group through a sequence of chromosomes and genotype batches
    /// with explicit native side-effect hooks.
    ///
    /// # Errors
    ///
    /// Returns an error when a phase transition, side-effect hook, or backend
    /// operation fails.
    pub fn run_group_chromosomes_with_effects<Effects>(
        &mut self,
        input: &EngineGroupRunInput<'_>,
        effects: &mut Effects,
    ) -> EngineResult<EngineGroupRunReport>
    where
        Effects: EngineRunEffects,
    {
        self.enter_phase(RunPhase::InputsOpened, effects)?;
        effects.open_inputs().map_err(|source| {
            self.record_effect_error(RunPhase::InputsOpened, EngineEffectOperation::InputOpen, source, effects)
        })?;
        self.enter_phase(RunPhase::InputsAligned, effects)?;
        effects.align_inputs().map_err(|source| {
            self.record_effect_error(RunPhase::InputsAligned, EngineEffectOperation::InputAlignment, source, effects)
        })?;
        self.enter_phase(RunPhase::PreflightValidated, effects)?;
        effects.validate_preflight().map_err(|source| {
            self.record_effect_error(
                RunPhase::PreflightValidated,
                EngineEffectOperation::PreflightValidation,
                source,
                effects,
            )
        })?;
        self.enter_phase(RunPhase::OutputsInitialized, effects)?;
        effects.validate_output_compatibility().map_err(|source| {
            self.record_effect_error(
                RunPhase::OutputsInitialized,
                EngineEffectOperation::OutputCompatibility,
                source,
                effects,
            )
        })?;
        effects.construct_writers().map_err(|source| {
            self.record_effect_error(
                RunPhase::OutputsInitialized,
                EngineEffectOperation::WriterConstruction,
                source,
                effects,
            )
        })?;
        self.enter_phase(RunPhase::Running, effects)?;

        let group_state = match self.backend.prepare_group(&input.group) {
            Ok(group_state) => group_state,
            Err(source) => {
                Self::abort_outputs_after_failure(RunPhase::Running, effects);
                return Err(self.record_backend_error(RunPhase::Running, source));
            }
        };
        let result_count = input.chromosomes.iter().map(|chromosome| chromosome.batches.len()).sum();
        let mut results = Vec::with_capacity(result_count);
        for chromosome_input in &input.chromosomes {
            let chromosome_state = match self.backend.prepare_chromosome(
                &group_state,
                chromosome_input.chromosome,
                chromosome_input.predictions,
            ) {
                Ok(chromosome_state) => chromosome_state,
                Err(source) => {
                    Self::abort_outputs_after_failure(RunPhase::Running, effects);
                    return Err(self.record_backend_error(RunPhase::Running, source));
                }
            };
            for batch in &chromosome_input.batches {
                let result = match self.backend.compute_batch(&chromosome_state, *batch) {
                    Ok(result) => result,
                    Err(source) => {
                        Self::abort_outputs_after_failure(RunPhase::Running, effects);
                        return Err(self.record_backend_error(RunPhase::Running, source));
                    }
                };
                effects.write_batch_result(&result).map_err(|source| {
                    self.record_effect_error(RunPhase::Running, EngineEffectOperation::OutputWrite, source, effects)
                })?;
                results.push(result);
            }
        }

        self.enter_phase(RunPhase::Draining, effects)?;
        effects.drain_writers().map_err(|source| {
            self.record_effect_error(RunPhase::Draining, EngineEffectOperation::WriterDrain, source, effects)
        })?;
        self.enter_phase(RunPhase::Finalizing, effects)?;
        effects.finalize_outputs().map_err(|source| {
            self.record_effect_error(RunPhase::Finalizing, EngineEffectOperation::OutputFinalization, source, effects)
        })?;
        self.enter_phase(RunPhase::Completed, effects)?;

        Ok(EngineGroupRunReport::new(self.phase_history.clone(), results))
    }
}

fn should_abort_outputs_after_failure(phase: RunPhase) -> bool {
    matches!(phase, RunPhase::OutputsInitialized | RunPhase::Running | RunPhase::Draining | RunPhase::Finalizing)
}
