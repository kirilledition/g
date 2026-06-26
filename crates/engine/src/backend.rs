//! Association backend contract used by the native engine coordinator.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedGroupInput {
    pub group_identifier: String,
    pub phenotype_count: usize,
}

impl PreparedGroupInput {
    #[must_use]
    pub fn new(group_identifier: String, phenotype_count: usize) -> Self {
        Self { group_identifier, phenotype_count }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PredictionView<'view> {
    pub chromosome: &'view str,
    pub row_count: usize,
}

impl<'view> PredictionView<'view> {
    #[must_use]
    pub const fn new(chromosome: &'view str, row_count: usize) -> Self {
        Self { chromosome, row_count }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenotypeBatchView<'view> {
    pub chromosome: &'view str,
    pub variant_count: usize,
    pub variant_offset: usize,
}

impl<'view> GenotypeBatchView<'view> {
    #[must_use]
    pub const fn new(chromosome: &'view str, variant_count: usize, variant_offset: usize) -> Self {
        Self { chromosome, variant_count, variant_offset }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssociationBatchResult {
    pub chromosome: String,
    pub variant_count: usize,
    pub statistic_sum: f64,
}

impl AssociationBatchResult {
    #[must_use]
    pub fn new(chromosome: String, variant_count: usize, statistic_sum: f64) -> Self {
        Self { chromosome, variant_count, statistic_sum }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct BackendError {
    message: String,
}

impl BackendError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

pub trait AssociationBackend {
    type GroupState;
    type ChromosomeState;

    /// Prepare association state for one phenotype group.
    ///
    /// # Errors
    ///
    /// Returns an error when group-level backend resources cannot be prepared.
    fn prepare_group(&mut self, input: &PreparedGroupInput) -> Result<Self::GroupState, BackendError>;

    /// Prepare association state for one chromosome.
    ///
    /// # Errors
    ///
    /// Returns an error when chromosome-level prediction state is invalid or
    /// backend resources cannot be prepared.
    fn prepare_chromosome(
        &mut self,
        group: &Self::GroupState,
        chromosome: &str,
        predictions: PredictionView<'_>,
    ) -> Result<Self::ChromosomeState, BackendError>;

    /// Compute association statistics for one genotype batch.
    ///
    /// # Errors
    ///
    /// Returns an error when backend execution fails for the batch.
    fn compute_batch(
        &mut self,
        chromosome: &Self::ChromosomeState,
        batch: GenotypeBatchView<'_>,
    ) -> Result<AssociationBatchResult, BackendError>;
}
