use super::common::{format_python_shape, validate_non_negative_count, validate_shape_counts};
use super::error::PreflightError;

/// Validate deterministic single-trait prediction shape.
///
/// # Errors
///
/// Returns an error when prediction sample count does not match the phenotype sample count.
pub fn validate_single_prediction_preflight_shape(
    chromosome: &str,
    prediction_shape: &[i64],
    sample_count: i64,
) -> Result<(), PreflightError> {
    validate_shape_counts("prediction shape", prediction_shape)?;
    validate_non_negative_count("sample count", sample_count)?;
    let actual_sample_count = prediction_shape.first().copied().unwrap_or(0);
    if actual_sample_count != sample_count {
        return Err(PreflightError::PredictionSampleCountMismatch {
            chromosome: chromosome.to_string(),
            actual_sample_count,
            expected_sample_count: sample_count,
        });
    }
    Ok(())
}

/// Validate deterministic multi-trait prediction shape.
///
/// # Errors
///
/// Returns an error when prediction shape does not match the expected trait-major shape.
pub fn validate_multi_prediction_preflight_shape(
    chromosome: &str,
    prediction_shape: &[i64],
    trait_count: i64,
    sample_count: i64,
) -> Result<(), PreflightError> {
    validate_shape_counts("prediction shape", prediction_shape)?;
    validate_non_negative_count("trait count", trait_count)?;
    validate_non_negative_count("sample count", sample_count)?;
    let expected_shape = [trait_count, sample_count];
    if prediction_shape != expected_shape {
        return Err(PreflightError::PredictionMatrixShapeMismatch {
            chromosome: chromosome.to_string(),
            actual_shape: format_python_shape(prediction_shape),
            expected_shape: format_python_shape(&expected_shape),
        });
    }
    Ok(())
}
