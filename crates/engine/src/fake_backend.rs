//! Deterministic association backend for Rust-only coordinator tests.

use crate::backend::{
    AssociationBackend, AssociationBatchResult, BackendError, GenotypeBatchView, PredictionView, PreparedGroupInput,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FakeBackendFailure {
    PrepareGroup,
    PrepareChromosome,
    ComputeBatch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FakeGroupState {
    pub group_identifier: String,
    pub phenotype_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FakeChromosomeState {
    pub chromosome: String,
    pub phenotype_count: usize,
    pub prediction_row_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FakeBackend {
    failure: Option<FakeBackendFailure>,
}

impl FakeBackend {
    #[must_use]
    pub const fn new(failure: Option<FakeBackendFailure>) -> Self {
        Self { failure }
    }

    #[must_use]
    pub const fn succeed() -> Self {
        Self { failure: None }
    }

    #[must_use]
    pub const fn fail_at(failure: FakeBackendFailure) -> Self {
        Self { failure: Some(failure) }
    }

    fn should_fail(&self, failure: FakeBackendFailure) -> bool {
        self.failure == Some(failure)
    }
}

impl AssociationBackend for FakeBackend {
    type GroupState = FakeGroupState;
    type ChromosomeState = FakeChromosomeState;

    fn prepare_group(&mut self, input: &PreparedGroupInput) -> Result<Self::GroupState, BackendError> {
        if self.should_fail(FakeBackendFailure::PrepareGroup) {
            return Err(BackendError::new("fake backend failed while preparing group"));
        }
        Ok(FakeGroupState { group_identifier: input.group_identifier.clone(), phenotype_count: input.phenotype_count })
    }

    fn prepare_chromosome(
        &mut self,
        group: &Self::GroupState,
        chromosome: &str,
        predictions: PredictionView<'_>,
    ) -> Result<Self::ChromosomeState, BackendError> {
        if self.should_fail(FakeBackendFailure::PrepareChromosome) {
            return Err(BackendError::new("fake backend failed while preparing chromosome"));
        }
        if predictions.chromosome != chromosome {
            return Err(BackendError::new(format!(
                "prediction chromosome {} does not match requested chromosome {chromosome}",
                predictions.chromosome,
            )));
        }
        Ok(FakeChromosomeState {
            chromosome: chromosome.to_string(),
            phenotype_count: group.phenotype_count,
            prediction_row_count: predictions.row_count,
        })
    }

    fn compute_batch(
        &mut self,
        chromosome: &Self::ChromosomeState,
        batch: GenotypeBatchView<'_>,
    ) -> Result<AssociationBatchResult, BackendError> {
        if self.should_fail(FakeBackendFailure::ComputeBatch) {
            return Err(BackendError::new("fake backend failed while computing batch"));
        }
        if batch.chromosome != chromosome.chromosome {
            return Err(BackendError::new(format!(
                "batch chromosome {} does not match prepared chromosome {}",
                batch.chromosome, chromosome.chromosome,
            )));
        }
        let statistic_total = fake_statistic_total(chromosome, batch)?;
        Ok(AssociationBatchResult::new(chromosome.chromosome.clone(), batch.variant_count, f64::from(statistic_total)))
    }
}

fn fake_statistic_total(chromosome: &FakeChromosomeState, batch: GenotypeBatchView<'_>) -> Result<u32, BackendError> {
    let phenotype_count = bounded_count("phenotype count", chromosome.phenotype_count)?;
    let variant_count = bounded_count("variant count", batch.variant_count)?;
    let prediction_row_count = bounded_count("prediction row count", chromosome.prediction_row_count)?;
    let variant_offset = bounded_count("variant offset", batch.variant_offset)?;
    let chromosome_name_length = bounded_count("chromosome name length", chromosome.chromosome.len())?;
    let phenotype_variant_total = phenotype_count.checked_mul(variant_count).ok_or_else(|| {
        BackendError::new("fake backend statistic overflowed while multiplying phenotypes and variants")
    })?;
    phenotype_variant_total
        .checked_add(prediction_row_count)
        .and_then(|subtotal| subtotal.checked_add(variant_offset))
        .and_then(|subtotal| subtotal.checked_add(chromosome_name_length))
        .ok_or_else(|| BackendError::new("fake backend statistic overflowed while accumulating fields"))
}

fn bounded_count(label: &str, count: usize) -> Result<u32, BackendError> {
    u32::try_from(count).map_err(|_| BackendError::new(format!("fake backend {label} exceeds u32 capacity")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_backend_produces_known_statistics() {
        let mut backend = FakeBackend::succeed();
        let group = backend.prepare_group(&PreparedGroupInput::new("binary".to_string(), 3)).unwrap();
        let chromosome = backend.prepare_chromosome(&group, "chr1", PredictionView::new("chr1", 11)).unwrap();
        let result = backend.compute_batch(&chromosome, GenotypeBatchView::new("chr1", 7, 5)).unwrap();

        assert_eq!(result.chromosome, "chr1");
        assert_eq!(result.variant_count, 7);
        assert_eq!(result.statistic_sum.to_bits(), 41.0_f64.to_bits());
    }

    #[test]
    fn fake_backend_rejects_misaligned_batch_chromosome() {
        let mut backend = FakeBackend::succeed();
        let group = backend.prepare_group(&PreparedGroupInput::new("binary".to_string(), 1)).unwrap();
        let chromosome = backend.prepare_chromosome(&group, "1", PredictionView::new("1", 1)).unwrap();

        let error = backend.compute_batch(&chromosome, GenotypeBatchView::new("2", 1, 0)).unwrap_err();

        assert!(error.message().contains("does not match prepared chromosome"));
    }
}
