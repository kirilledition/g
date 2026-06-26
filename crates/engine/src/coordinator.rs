//! Native run coordinator scaffold with explicit phase transitions.

use crate::backend::{
    AssociationBackend, AssociationBatchResult, BackendError, GenotypeBatchView, PredictionView, PreparedGroupInput,
};
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

    fn enter_phase(&mut self, phase: RunPhase) -> Result<(), EngineError> {
        self.record_phase(phase);
        if self.interruption_phase == Some(phase) {
            self.record_phase(RunPhase::Interrupted);
            return Err(EngineError::Interrupted { phase });
        }
        if let Some(failure) = self.injected_failure.as_ref()
            && failure.phase == phase
        {
            let message = failure.message.clone();
            self.record_phase(RunPhase::Failed);
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
        self.enter_phase(RunPhase::InputsOpened)?;
        self.enter_phase(RunPhase::InputsAligned)?;
        self.enter_phase(RunPhase::PreflightValidated)?;
        self.enter_phase(RunPhase::OutputsInitialized)?;
        self.enter_phase(RunPhase::Running)?;

        let group_state = match self.backend.prepare_group(&input.group) {
            Ok(group_state) => group_state,
            Err(source) => return Err(self.record_backend_error(RunPhase::Running, source)),
        };
        let chromosome_state = match self.backend.prepare_chromosome(&group_state, input.chromosome, input.predictions)
        {
            Ok(chromosome_state) => chromosome_state,
            Err(source) => return Err(self.record_backend_error(RunPhase::Running, source)),
        };
        let result = match self.backend.compute_batch(&chromosome_state, input.batch) {
            Ok(result) => result,
            Err(source) => return Err(self.record_backend_error(RunPhase::Running, source)),
        };

        self.enter_phase(RunPhase::Draining)?;
        self.enter_phase(RunPhase::Finalizing)?;
        self.enter_phase(RunPhase::Completed)?;

        Ok(EngineRunReport::new(self.phase_history.clone(), result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake_backend::{FakeBackend, FakeBackendFailure};

    fn build_input() -> EngineRunInput<'static> {
        EngineRunInput::new(
            PreparedGroupInput::new("binary".to_string(), 2),
            "chr2",
            PredictionView::new("chr2", 5),
            GenotypeBatchView::new("chr2", 4, 3),
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
    fn coordinator_records_backend_prepare_failure() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::fail_at(FakeBackendFailure::PrepareGroup));

        let error = coordinator.run_single_batch(&build_input()).unwrap_err();

        assert!(matches!(error, EngineError::Backend { phase: RunPhase::Running, .. }));
        assert_eq!(coordinator.phase(), RunPhase::Failed);
        assert_eq!(coordinator.phase_history().last(), Some(&RunPhase::Failed));
    }

    #[test]
    fn coordinator_records_backend_chromosome_failure() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::fail_at(FakeBackendFailure::PrepareChromosome));

        let error = coordinator.run_single_batch(&build_input()).unwrap_err();

        assert!(matches!(error, EngineError::Backend { phase: RunPhase::Running, .. }));
        assert_eq!(coordinator.phase(), RunPhase::Failed);
        assert_eq!(coordinator.phase_history().last(), Some(&RunPhase::Failed));
    }

    #[test]
    fn coordinator_records_backend_batch_failure() {
        let mut coordinator = EngineCoordinator::new(FakeBackend::fail_at(FakeBackendFailure::ComputeBatch));

        let error = coordinator.run_single_batch(&build_input()).unwrap_err();

        assert!(matches!(error, EngineError::Backend { phase: RunPhase::Running, .. }));
        assert_eq!(coordinator.phase(), RunPhase::Failed);
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
}
