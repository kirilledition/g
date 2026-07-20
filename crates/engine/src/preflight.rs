//! Canonical aligned-group preflight validation.

use nalgebra::DMatrix;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum PreflightError {
    #[error("BGEN input contains no variants.")]
    EmptyBgenInput,
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
    #[error("{label} exceeds the JAX int32 index domain.")]
    JaxIndexCapacityExceeded { label: &'static str },
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

pub(crate) fn validate_jax_index_capacity(
    trait_count: usize,
    sample_count: usize,
    chunk_size: usize,
    firth_candidate_capacity: usize,
    firth_batch_size: usize,
    is_binary_trait: bool,
) -> Result<(), PreflightError> {
    let maximum_index_count = usize::try_from(i32::MAX).expect("supported 64-bit targets represent i32::MAX");
    for (label, count) in [("trait count", trait_count), ("sample count", sample_count), ("chunk size", chunk_size)] {
        if count > maximum_index_count {
            return Err(PreflightError::JaxIndexCapacityExceeded { label });
        }
    }
    let _flattened_lane_count = trait_count
        .checked_mul(chunk_size)
        .filter(|count| *count <= maximum_index_count)
        .ok_or(PreflightError::JaxIndexCapacityExceeded { label: "flattened trait-by-chunk lane count" })?;
    if !is_binary_trait {
        return Ok(());
    }
    let candidate_count = firth_candidate_capacity
        .min(chunk_size)
        .checked_mul(trait_count)
        .ok_or(PreflightError::JaxIndexCapacityExceeded { label: "multi-trait Firth candidate capacity" })?;
    candidate_count
        .checked_add(firth_batch_size - 1)
        .map(|count| count / firth_batch_size)
        .and_then(|batch_count| batch_count.checked_mul(firth_batch_size))
        .filter(|count| *count <= maximum_index_count)
        .ok_or(PreflightError::JaxIndexCapacityExceeded { label: "padded Firth candidate capacity" })?;
    Ok(())
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
    let dimension_count = i32::try_from(row_count.max(column_count))
        .map(f64::from)
        .map_err(|_| PreflightError::JaxIndexCapacityExceeded { label: "covariate matrix dimension" })?;
    let rank_matrix = DMatrix::from_row_iterator(row_count, column_count, values.iter().copied().map(f64::from));
    let singular_values = rank_matrix.svd(false, false).singular_values;
    let largest_singular_value = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = largest_singular_value * dimension_count * f64::from(f32::EPSILON);
    if singular_values.iter().filter(|singular_value| **singular_value > tolerance).count() < column_count {
        return Err(PreflightError::CovariateMatrixRankDeficient);
    }
    Ok(())
}
