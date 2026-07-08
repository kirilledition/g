use super::common::validate_non_negative_count;
use super::error::PreflightError;
use super::payloads::{MultiTraitPreflightShapePayload, SingleTraitPreflightShapePayload};

/// Validate deterministic single-trait preflight dimensions.
///
/// # Errors
///
/// Returns an error when matrix dimensions, sample counts, or model degrees of freedom are invalid.
pub fn validate_single_trait_preflight_shape_payload(
    phenotype_sample_count: i64,
    covariate_dimension_count: i64,
    covariate_sample_count: i64,
    covariate_count: i64,
) -> Result<SingleTraitPreflightShapePayload, PreflightError> {
    validate_non_negative_count("phenotype sample count", phenotype_sample_count)?;
    validate_non_negative_count("covariate dimension count", covariate_dimension_count)?;
    validate_non_negative_count("covariate sample count", covariate_sample_count)?;
    validate_non_negative_count("covariate count", covariate_count)?;
    validate_covariate_shape(phenotype_sample_count, covariate_dimension_count, covariate_sample_count)?;
    validate_residual_degrees_of_freedom(phenotype_sample_count, covariate_count)?;
    Ok(SingleTraitPreflightShapePayload { sample_count: phenotype_sample_count, covariate_count })
}

/// Validate deterministic multi-trait preflight dimensions.
///
/// # Errors
///
/// Returns an error when phenotype dimensions, covariate dimensions, sample counts, or model degrees of freedom are
/// invalid.
pub fn validate_multi_trait_preflight_shape_payload(
    phenotype_dimension_count: i64,
    phenotype_trait_count: i64,
    phenotype_sample_count: i64,
    covariate_dimension_count: i64,
    covariate_sample_count: i64,
    covariate_count: i64,
) -> Result<MultiTraitPreflightShapePayload, PreflightError> {
    validate_non_negative_count("phenotype dimension count", phenotype_dimension_count)?;
    validate_non_negative_count("phenotype trait count", phenotype_trait_count)?;
    validate_non_negative_count("phenotype sample count", phenotype_sample_count)?;
    validate_non_negative_count("covariate dimension count", covariate_dimension_count)?;
    validate_non_negative_count("covariate sample count", covariate_sample_count)?;
    validate_non_negative_count("covariate count", covariate_count)?;
    if phenotype_dimension_count != 2 {
        return Err(PreflightError::PhenotypeMatrixDimension);
    }
    if phenotype_trait_count == 0 {
        return Err(PreflightError::EmptyPhenotypeTraitSet);
    }
    if phenotype_sample_count == 0 {
        return Err(PreflightError::EmptyPhenotypeSampleSet);
    }
    validate_covariate_shape(phenotype_sample_count, covariate_dimension_count, covariate_sample_count)?;
    validate_residual_degrees_of_freedom(phenotype_sample_count, covariate_count)?;
    Ok(MultiTraitPreflightShapePayload {
        trait_count: phenotype_trait_count,
        sample_count: phenotype_sample_count,
        covariate_count,
    })
}

/// Validate deterministic finite-array preflight policy.
///
/// # Errors
///
/// Returns an error when the caller reports non-finite array values.
pub fn validate_finite_array(label: &str, all_values_finite: bool) -> Result<(), PreflightError> {
    if all_values_finite {
        return Ok(());
    }
    Err(PreflightError::NonFiniteArray { label: label.to_string() })
}

/// Validate deterministic covariate matrix rank policy.
///
/// # Errors
///
/// Returns an error when the covariate matrix rank is smaller than the number of covariate columns.
pub fn validate_covariate_matrix_rank(covariate_rank: i64, covariate_count: i64) -> Result<(), PreflightError> {
    validate_non_negative_count("covariate matrix rank", covariate_rank)?;
    validate_non_negative_count("covariate count", covariate_count)?;
    if covariate_rank < covariate_count {
        return Err(PreflightError::CovariateMatrixRankDeficient);
    }
    Ok(())
}

/// Validate deterministic binary phenotype coding policy.
///
/// # Errors
///
/// Returns an error when a binary phenotype contains a value other than 0 or 1 after alignment.
pub fn validate_binary_phenotype_coding(is_binary_coded: bool) -> Result<(), PreflightError> {
    if is_binary_coded {
        return Ok(());
    }
    Err(PreflightError::BinaryPhenotypeCoding)
}

/// Validate deterministic binary phenotype case/control counts.
///
/// # Errors
///
/// Returns an error when either class is missing.
pub fn validate_binary_phenotype_case_control_counts(
    case_count: i64,
    control_count: i64,
) -> Result<(), PreflightError> {
    validate_non_negative_count("binary phenotype case count", case_count)?;
    validate_non_negative_count("binary phenotype control count", control_count)?;
    if case_count == 0 || control_count == 0 {
        return Err(PreflightError::BinaryPhenotypeMissingClass);
    }
    Ok(())
}

fn validate_covariate_shape(
    phenotype_sample_count: i64,
    covariate_dimension_count: i64,
    covariate_sample_count: i64,
) -> Result<(), PreflightError> {
    if covariate_dimension_count != 2 {
        return Err(PreflightError::CovariateMatrixDimension);
    }
    if covariate_sample_count != phenotype_sample_count {
        return Err(PreflightError::CovariateSampleCountMismatch);
    }
    Ok(())
}

fn validate_residual_degrees_of_freedom(sample_count: i64, covariate_count: i64) -> Result<(), PreflightError> {
    if sample_count <= covariate_count {
        return Err(PreflightError::NonPositiveResidualDegreesOfFreedom);
    }
    Ok(())
}
