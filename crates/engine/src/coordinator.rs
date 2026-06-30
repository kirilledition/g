//! Native run coordinator scaffold with explicit phase transitions.

use crate::backend::{
    AssociationBackend, AssociationBatchResult, BackendError, GenotypeBatchView, PredictionView, PreparedGroupInput,
};
use crate::effects::{EngineEffectError, EngineEffectOperation, EngineRunEffects, NoopEngineRunEffects};
use crate::phase::RunPhase;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InjectedCoordinatorFailure {
    pub phase: RunPhase,
    pub message: String,
}

impl InjectedCoordinatorFailure {
    #[must_use]
    pub fn new(phase: RunPhase, message: String) -> Self {
        Self { phase, message }
    }
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineRunInput<'view> {
    pub group: PreparedGroupInput,
    pub chromosome: &'view str,
    pub predictions: PredictionView<'view>,
    pub batch: GenotypeBatchView<'view>,
}

impl<'view> EngineRunInput<'view> {
    #[must_use]
    pub const fn new(
        group: PreparedGroupInput,
        chromosome: &'view str,
        predictions: PredictionView<'view>,
        batch: GenotypeBatchView<'view>,
    ) -> Self {
        Self { group, chromosome, predictions, batch }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineChromosomeRunInput<'view> {
    pub group: PreparedGroupInput,
    pub chromosome: &'view str,
    pub predictions: PredictionView<'view>,
    pub batches: Vec<GenotypeBatchView<'view>>,
}

impl<'view> EngineChromosomeRunInput<'view> {
    #[must_use]
    pub fn new(
        group: PreparedGroupInput,
        chromosome: &'view str,
        predictions: PredictionView<'view>,
        batches: Vec<GenotypeBatchView<'view>>,
    ) -> Self {
        Self { group, chromosome, predictions, batches }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineGroupChromosomeInput<'view> {
    pub chromosome: &'view str,
    pub predictions: PredictionView<'view>,
    pub batches: Vec<GenotypeBatchView<'view>>,
}

impl<'view> EngineGroupChromosomeInput<'view> {
    #[must_use]
    pub fn new(
        chromosome: &'view str,
        predictions: PredictionView<'view>,
        batches: Vec<GenotypeBatchView<'view>>,
    ) -> Self {
        Self { chromosome, predictions, batches }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineGroupRunInput<'view> {
    pub group: PreparedGroupInput,
    pub chromosomes: Vec<EngineGroupChromosomeInput<'view>>,
}

impl<'view> EngineGroupRunInput<'view> {
    #[must_use]
    pub fn new(group: PreparedGroupInput, chromosomes: Vec<EngineGroupChromosomeInput<'view>>) -> Self {
        Self { group, chromosomes }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EngineRunReport {
    pub phase_history: Vec<RunPhase>,
    pub result: AssociationBatchResult,
}

impl EngineRunReport {
    #[must_use]
    pub fn new(phase_history: Vec<RunPhase>, result: AssociationBatchResult) -> Self {
        Self { phase_history, result }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EngineChromosomeRunReport {
    pub phase_history: Vec<RunPhase>,
    pub results: Vec<AssociationBatchResult>,
}

impl EngineChromosomeRunReport {
    #[must_use]
    pub fn new(phase_history: Vec<RunPhase>, results: Vec<AssociationBatchResult>) -> Self {
        Self { phase_history, results }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EngineGroupRunReport {
    pub phase_history: Vec<RunPhase>,
    pub results: Vec<AssociationBatchResult>,
}

impl EngineGroupRunReport {
    #[must_use]
    pub fn new(phase_history: Vec<RunPhase>, results: Vec<AssociationBatchResult>) -> Self {
        Self { phase_history, results }
    }
}

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

    fn enter_phase<Effects>(&mut self, phase: RunPhase, effects: &mut Effects) -> Result<(), EngineError>
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
    pub fn run_single_batch(&mut self, input: &EngineRunInput<'_>) -> Result<EngineRunReport, EngineError> {
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
    ) -> Result<EngineRunReport, EngineError>
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
    ) -> Result<EngineChromosomeRunReport, EngineError> {
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
    ) -> Result<EngineChromosomeRunReport, EngineError>
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
    pub fn run_group_chromosomes(
        &mut self,
        input: &EngineGroupRunInput<'_>,
    ) -> Result<EngineGroupRunReport, EngineError> {
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
    ) -> Result<EngineGroupRunReport, EngineError>
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake_backend::{FakeBackend, FakeBackendFailure};
    use crate::fake_effects::{FakeEngineRunEffects, FakeOutputLifecycleState};

    fn build_input() -> EngineRunInput<'static> {
        EngineRunInput::new(
            PreparedGroupInput::new("binary".to_string(), 2),
            "chr2",
            PredictionView::new("chr2", 5),
            GenotypeBatchView::new("chr2", 4, 3),
        )
    }

    fn build_chromosome_input() -> EngineChromosomeRunInput<'static> {
        EngineChromosomeRunInput::new(
            PreparedGroupInput::new("binary".to_string(), 2),
            "chr2",
            PredictionView::new("chr2", 5),
            vec![GenotypeBatchView::new("chr2", 4, 3), GenotypeBatchView::new("chr2", 2, 7)],
        )
    }

    fn build_group_input() -> EngineGroupRunInput<'static> {
        EngineGroupRunInput::new(
            PreparedGroupInput::new("binary".to_string(), 2),
            vec![
                EngineGroupChromosomeInput::new(
                    "chr2",
                    PredictionView::new("chr2", 5),
                    vec![GenotypeBatchView::new("chr2", 4, 3), GenotypeBatchView::new("chr2", 2, 7)],
                ),
                EngineGroupChromosomeInput::new(
                    "chr3",
                    PredictionView::new("chr3", 8),
                    vec![GenotypeBatchView::new("chr3", 3, 1)],
                ),
            ],
        )
    }

    #[test]
    fn coordinator_executes_tiny_run_with_fake_backend() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::succeed());

        let report = coordinator.run_single_batch(&build_input()).unwrap();

        assert_eq!(
            report.phase_history,
            vec![
                RunPhase::Planned,
                RunPhase::InputsOpened,
                RunPhase::InputsAligned,
                RunPhase::PreflightValidated,
                RunPhase::OutputsInitialized,
                RunPhase::Running,
                RunPhase::Draining,
                RunPhase::Finalizing,
                RunPhase::Completed,
            ],
        );
        assert_eq!(report.result.chromosome, "chr2");
        assert_eq!(report.result.variant_count, 4);
        assert_eq!(report.result.statistic_sum.to_bits(), 20.0_f64.to_bits());
        assert_eq!(coordinator.phase(), RunPhase::Completed);
    }

    #[test]
    fn coordinator_executes_group_chromosomes_with_fake_backend() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::succeed());
        let mut effects = FakeEngineRunEffects::succeed();

        let report = coordinator.run_group_chromosomes_with_effects(&build_group_input(), &mut effects).unwrap();

        assert_eq!(report.results.len(), 3);
        assert_eq!(report.results[0].chromosome, "chr2");
        assert_eq!(report.results[0].statistic_sum.to_bits(), 20.0_f64.to_bits());
        assert_eq!(report.results[1].chromosome, "chr2");
        assert_eq!(report.results[1].statistic_sum.to_bits(), 20.0_f64.to_bits());
        assert_eq!(report.results[2].chromosome, "chr3");
        assert_eq!(report.results[2].statistic_sum.to_bits(), 19.0_f64.to_bits());
        assert_eq!(effects.state().output.written_results, report.results);
        assert_eq!(effects.state().output.lifecycle_state, FakeOutputLifecycleState::Finalized);
        assert_eq!(coordinator.phase(), RunPhase::Completed);
    }

    #[test]
    fn coordinator_executes_chromosome_batches_with_fake_backend() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::succeed());
        let mut effects = FakeEngineRunEffects::succeed();

        let report = coordinator.run_chromosome_batches_with_effects(&build_chromosome_input(), &mut effects).unwrap();

        assert_eq!(report.results.len(), 2);
        assert_eq!(report.results[0].variant_count, 4);
        assert_eq!(report.results[0].statistic_sum.to_bits(), 20.0_f64.to_bits());
        assert_eq!(report.results[1].variant_count, 2);
        assert_eq!(report.results[1].statistic_sum.to_bits(), 20.0_f64.to_bits());
        assert_eq!(effects.state().output.written_results, report.results);
        assert_eq!(effects.state().output.lifecycle_state, FakeOutputLifecycleState::Finalized);
        assert_eq!(coordinator.phase(), RunPhase::Completed);
    }

    #[test]
    fn coordinator_executes_native_side_effect_hooks() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::succeed());
        let mut effects = FakeEngineRunEffects::succeed();

        let report = coordinator.run_single_batch_with_effects(&build_input(), &mut effects).unwrap();

        assert_eq!(report.result.variant_count, 4);
        assert_eq!(effects.state().output.written_results, vec![report.result]);
        assert_eq!(effects.state().output.lifecycle_state, FakeOutputLifecycleState::Finalized);
        assert_eq!(effects.state().output.abort_count, 0);
        assert_eq!(effects.state().phase_events, report.phase_history[1..]);
        assert_eq!(
            effects.state().completed_operations,
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
        assert_eq!(effects.state().output.aborted_phase, None);
    }

    #[test]
    fn coordinator_records_backend_prepare_failure() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::fail_at(FakeBackendFailure::PrepareGroup));
        let mut effects = FakeEngineRunEffects::succeed();

        let error = coordinator.run_single_batch_with_effects(&build_input(), &mut effects).unwrap_err();

        assert!(matches!(error, EngineError::Backend { phase: RunPhase::Running, .. }));
        assert_eq!(coordinator.phase(), RunPhase::Failed);
        assert_eq!(coordinator.phase_history().last(), Some(&RunPhase::Failed));
        assert_eq!(effects.state().output.aborted_phase, Some(RunPhase::Running));
        assert_eq!(effects.state().output.lifecycle_state, FakeOutputLifecycleState::Aborted);
        assert_eq!(effects.state().output.abort_count, 1);
    }

    #[test]
    fn coordinator_records_backend_chromosome_failure() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::fail_at(FakeBackendFailure::PrepareChromosome));
        let mut effects = FakeEngineRunEffects::succeed();

        let error = coordinator.run_single_batch_with_effects(&build_input(), &mut effects).unwrap_err();

        assert!(matches!(error, EngineError::Backend { phase: RunPhase::Running, .. }));
        assert_eq!(coordinator.phase(), RunPhase::Failed);
        assert_eq!(coordinator.phase_history().last(), Some(&RunPhase::Failed));
        assert_eq!(effects.state().output.aborted_phase, Some(RunPhase::Running));
        assert_eq!(effects.state().output.lifecycle_state, FakeOutputLifecycleState::Aborted);
        assert_eq!(effects.state().output.abort_count, 1);
    }

    #[test]
    fn coordinator_records_backend_batch_failure() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::fail_at(FakeBackendFailure::ComputeBatch));
        let mut effects = FakeEngineRunEffects::succeed();

        let error = coordinator.run_single_batch_with_effects(&build_input(), &mut effects).unwrap_err();

        assert!(matches!(error, EngineError::Backend { phase: RunPhase::Running, .. }));
        assert_eq!(coordinator.phase(), RunPhase::Failed);
        assert_eq!(effects.state().output.aborted_phase, Some(RunPhase::Running));
        assert_eq!(effects.state().output.lifecycle_state, FakeOutputLifecycleState::Aborted);
        assert_eq!(effects.state().output.abort_count, 1);
    }

    #[test]
    fn coordinator_records_injected_phase_failure() {
        let failure =
            InjectedCoordinatorFailure::new(RunPhase::OutputsInitialized, "writer construction failed".to_string());
        let mut coordinator = EngineCoordinator::new(FakeBackend::succeed()).with_injected_failure(failure);

        let error = coordinator.run_single_batch(&build_input()).unwrap_err();

        assert_eq!(
            error,
            EngineError::Coordinator {
                phase: RunPhase::OutputsInitialized,
                message: "writer construction failed".to_string(),
            },
        );
        assert_eq!(
            coordinator.phase_history(),
            &[
                RunPhase::Planned,
                RunPhase::InputsOpened,
                RunPhase::InputsAligned,
                RunPhase::PreflightValidated,
                RunPhase::OutputsInitialized,
                RunPhase::Failed,
            ],
        );
    }

    #[test]
    fn coordinator_records_side_effect_failure_and_aborts_initialized_outputs() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::succeed());
        let mut effects = FakeEngineRunEffects::fail_at(EngineEffectOperation::OutputWrite);

        let error = coordinator.run_single_batch_with_effects(&build_input(), &mut effects).unwrap_err();

        assert_eq!(
            error,
            EngineError::Effect {
                phase: RunPhase::Running,
                operation: EngineEffectOperation::OutputWrite,
                source: EngineEffectError::new("fake engine side effect failed during output_write"),
            },
        );
        assert_eq!(coordinator.phase(), RunPhase::Failed);
        assert_eq!(effects.state().output.aborted_phase, Some(RunPhase::Running));
        assert_eq!(effects.state().output.lifecycle_state, FakeOutputLifecycleState::Aborted);
        assert_eq!(effects.state().output.abort_count, 1);
    }

    #[test]
    fn coordinator_records_side_effect_failures_at_each_operation() {
        for (operation, phase, aborts_outputs) in [
            (EngineEffectOperation::TelemetryEvent, RunPhase::InputsOpened, false),
            (EngineEffectOperation::InputOpen, RunPhase::InputsOpened, false),
            (EngineEffectOperation::InputAlignment, RunPhase::InputsAligned, false),
            (EngineEffectOperation::PreflightValidation, RunPhase::PreflightValidated, false),
            (EngineEffectOperation::OutputCompatibility, RunPhase::OutputsInitialized, true),
            (EngineEffectOperation::WriterConstruction, RunPhase::OutputsInitialized, true),
            (EngineEffectOperation::OutputWrite, RunPhase::Running, true),
            (EngineEffectOperation::WriterDrain, RunPhase::Draining, true),
            (EngineEffectOperation::OutputFinalization, RunPhase::Finalizing, true),
        ] {
            let mut coordinator = EngineCoordinator::new(FakeBackend::succeed());
            let mut effects = FakeEngineRunEffects::fail_at(operation);

            let error = coordinator.run_single_batch_with_effects(&build_input(), &mut effects).unwrap_err();

            assert_eq!(
                error,
                EngineError::Effect {
                    phase,
                    operation,
                    source: EngineEffectError::new(format!("fake engine side effect failed during {operation}")),
                }
            );
            assert_eq!(coordinator.phase(), RunPhase::Failed);
            assert_eq!(effects.state().output.aborted_phase.is_some(), aborts_outputs,);
            assert_eq!(effects.state().output.lifecycle_state == FakeOutputLifecycleState::Aborted, aborts_outputs,);
            assert_eq!(effects.state().output.abort_count, usize::from(aborts_outputs));
        }
    }

    #[test]
    fn coordinator_records_injected_failure_at_each_entered_phase() {
        for phase in [
            RunPhase::InputsOpened,
            RunPhase::InputsAligned,
            RunPhase::PreflightValidated,
            RunPhase::OutputsInitialized,
            RunPhase::Running,
            RunPhase::Draining,
            RunPhase::Finalizing,
            RunPhase::Completed,
        ] {
            let message = format!("failed at {}", phase.as_str());
            let failure = InjectedCoordinatorFailure::new(phase, message.clone());
            let mut coordinator = EngineCoordinator::new(FakeBackend::succeed()).with_injected_failure(failure);

            let error = coordinator.run_single_batch(&build_input()).unwrap_err();

            assert_eq!(error, EngineError::Coordinator { phase, message });
            assert_eq!(coordinator.phase(), RunPhase::Failed);
            assert!(coordinator.phase_history().contains(&phase));
            assert_eq!(coordinator.phase_history().last(), Some(&RunPhase::Failed));
        }
    }

    #[test]
    fn coordinator_records_interruption() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::succeed()).with_interruption(RunPhase::Running);

        let error = coordinator.run_single_batch(&build_input()).unwrap_err();

        assert_eq!(error, EngineError::Interrupted { phase: RunPhase::Running });
        assert_eq!(coordinator.phase(), RunPhase::Interrupted);
        assert_eq!(coordinator.phase_history().last(), Some(&RunPhase::Interrupted));
    }

    #[test]
    fn coordinator_aborts_outputs_for_injected_failures_after_initialization() {
        for phase in [RunPhase::OutputsInitialized, RunPhase::Running, RunPhase::Draining, RunPhase::Finalizing] {
            let mut coordinator = EngineCoordinator::new(FakeBackend::succeed())
                .with_injected_failure(InjectedCoordinatorFailure::new(phase, format!("failed at {}", phase.as_str())));
            let mut effects = FakeEngineRunEffects::succeed();

            let error = coordinator.run_single_batch_with_effects(&build_input(), &mut effects).unwrap_err();

            assert!(matches!(error, EngineError::Coordinator { phase: error_phase, .. } if error_phase == phase));
            assert_eq!(effects.state().output.lifecycle_state, FakeOutputLifecycleState::Aborted);
            assert_eq!(effects.state().output.aborted_phase, Some(phase));
            assert_eq!(effects.state().output.abort_count, 1);
        }
    }

    #[test]
    fn coordinator_aborts_outputs_for_interruption_after_initialization() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::succeed()).with_interruption(RunPhase::Running);
        let mut effects = FakeEngineRunEffects::succeed();

        let error = coordinator.run_single_batch_with_effects(&build_input(), &mut effects).unwrap_err();

        assert_eq!(error, EngineError::Interrupted { phase: RunPhase::Running });
        assert_eq!(effects.state().output.lifecycle_state, FakeOutputLifecycleState::Aborted);
        assert_eq!(effects.state().output.aborted_phase, Some(RunPhase::Running));
        assert_eq!(effects.state().output.abort_count, 1);
    }
}
