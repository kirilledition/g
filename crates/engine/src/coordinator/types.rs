use crate::backend::{AssociationBatchResult, GenotypeBatchView, PredictionView, PreparedGroupInput};
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
