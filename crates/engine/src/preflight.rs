//! Canonical aligned-group preflight validation.

use nalgebra::DMatrix;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PreflightError {
    #[error("BGEN input contains no variants.")]
    EmptyBgenInput,
    #[error("BGEN scan contains no variants.")]
    EmptyBgenScan,
    #[error("Phenotype matrix must contain at least one trait.")]
    EmptyPhenotypeTraitSet,
    #[error("Phenotype matrix must contain at least one sample.")]
    EmptyPhenotypeSampleSet,
    #[error("Phenotype matrix shape exceeds native capacity.")]
    PhenotypeMatrixShapeOverflow,
    #[error("Phenotype matrix value count does not match its shape.")]
    PhenotypeMatrixValueCountMismatch,
    #[error("Covariate matrix shape exceeds native capacity.")]
    CovariateMatrixShapeOverflow,
    #[error("Covariate matrix value count does not match its shape.")]
    CovariateMatrixValueCountMismatch,
    #[error("Covariate matrix sample count does not match phenotype sample count.")]
    CovariateSampleCountMismatch,
    #[error("Sample count must exceed the number of covariate degrees of freedom.")]
    NonPositiveResidualDegreesOfFreedom,
    #[error("{label} contains non-finite values.")]
    NonFiniteArray { label: String },
    #[error("Covariate matrix is rank deficient.")]
    CovariateMatrixRankDeficient,
    #[error("Binary phenotype must be coded as 0/1 after alignment.")]
    BinaryPhenotypeCoding,
    #[error("Binary phenotype must contain at least one case and one control.")]
    BinaryPhenotypeMissingClass,
    #[error("Prediction matrix shape for chromosome {chromosome} exceeds native capacity.")]
    PredictionMatrixShapeOverflow { chromosome: String },
    #[error(
        "Prediction matrix for chromosome {chromosome} contains {actual_value_count} values; expected {expected_value_count}."
    )]
    PredictionMatrixValueCountMismatch { chromosome: String, actual_value_count: usize, expected_value_count: usize },
}

pub(crate) fn validate_multi_trait_preflight_values(
    phenotype_trait_count: usize,
    phenotype_sample_count: usize,
    phenotype_values: &[f32],
    covariate_sample_count: usize,
    covariate_count: usize,
    covariate_values: &[f32],
    is_binary_trait: bool,
) -> Result<(), PreflightError> {
    if phenotype_trait_count == 0 {
        return Err(PreflightError::EmptyPhenotypeTraitSet);
    }
    if phenotype_sample_count == 0 {
        return Err(PreflightError::EmptyPhenotypeSampleSet);
    }
    if covariate_sample_count != phenotype_sample_count {
        return Err(PreflightError::CovariateSampleCountMismatch);
    }
    if phenotype_sample_count <= covariate_count {
        return Err(PreflightError::NonPositiveResidualDegreesOfFreedom);
    }
    let expected_phenotype_value_count = phenotype_trait_count
        .checked_mul(phenotype_sample_count)
        .ok_or(PreflightError::PhenotypeMatrixShapeOverflow)?;
    if phenotype_values.len() != expected_phenotype_value_count {
        return Err(PreflightError::PhenotypeMatrixValueCountMismatch);
    }
    let expected_covariate_value_count =
        covariate_sample_count.checked_mul(covariate_count).ok_or(PreflightError::CovariateMatrixShapeOverflow)?;
    if covariate_values.len() != expected_covariate_value_count {
        return Err(PreflightError::CovariateMatrixValueCountMismatch);
    }
    validate_finite_values("Phenotype matrix", phenotype_values)?;
    validate_finite_values("Covariate matrix", covariate_values)?;
    validate_covariate_matrix_rank(covariate_sample_count, covariate_count, covariate_values)?;
    if is_binary_trait {
        for phenotype_trait_values in phenotype_values.chunks_exact(phenotype_sample_count) {
            validate_binary_phenotype(phenotype_trait_values)?;
        }
    }
    Ok(())
}

pub(crate) fn validate_multi_prediction_values(
    chromosome: &str,
    prediction_values: &[f32],
    trait_count: usize,
    sample_count: usize,
) -> Result<(), PreflightError> {
    let expected_value_count = trait_count
        .checked_mul(sample_count)
        .ok_or_else(|| PreflightError::PredictionMatrixShapeOverflow { chromosome: chromosome.to_string() })?;
    if prediction_values.len() != expected_value_count {
        return Err(PreflightError::PredictionMatrixValueCountMismatch {
            chromosome: chromosome.to_string(),
            actual_value_count: prediction_values.len(),
            expected_value_count,
        });
    }
    validate_finite_values(&format!("Prediction matrix for chromosome {chromosome}"), prediction_values)
}

fn validate_finite_values(label: &str, values: &[f32]) -> Result<(), PreflightError> {
    if values.iter().copied().all(f32::is_finite) {
        return Ok(());
    }
    Err(PreflightError::NonFiniteArray { label: label.to_string() })
}

fn validate_binary_phenotype(values: &[f32]) -> Result<(), PreflightError> {
    let mut case_count = 0_usize;
    let mut control_count = 0_usize;
    for value in values {
        if matches!(value.classify(), std::num::FpCategory::Zero) {
            control_count += 1;
        } else if value.to_bits() == 1.0_f32.to_bits() {
            case_count += 1;
        } else {
            return Err(PreflightError::BinaryPhenotypeCoding);
        }
    }
    if case_count == 0 || control_count == 0 {
        return Err(PreflightError::BinaryPhenotypeMissingClass);
    }
    Ok(())
}

fn validate_covariate_matrix_rank(row_count: usize, column_count: usize, values: &[f32]) -> Result<(), PreflightError> {
    let rank_values = values.iter().copied().map(f64::from).collect::<Vec<_>>();
    let rank_matrix = DMatrix::from_row_slice(row_count, column_count, &rank_values);
    let singular_values = rank_matrix.svd(false, false).singular_values;
    let largest_singular_value = singular_values.iter().copied().fold(0.0_f64, f64::max);
    #[allow(clippy::cast_precision_loss)]
    let dimension_count = row_count.max(column_count) as f64;
    let tolerance = largest_singular_value * dimension_count * f64::from(f32::EPSILON);
    if singular_values.iter().filter(|singular_value| **singular_value > tolerance).count() < column_count {
        return Err(PreflightError::CovariateMatrixRankDeficient);
    }
    Ok(())
}
