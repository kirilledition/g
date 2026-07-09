use nalgebra::DMatrix;

use super::error::PreflightError;
use super::payloads::{MultiTraitPreflightShapePayload, SingleTraitPreflightShapePayload};
use super::prediction::{validate_multi_prediction_preflight_shape, validate_single_prediction_preflight_shape};
use super::shape::{
    validate_binary_phenotype_case_control_counts, validate_binary_phenotype_coding, validate_covariate_matrix_rank,
    validate_finite_array, validate_multi_trait_preflight_shape_payload, validate_single_trait_preflight_shape_payload,
};

#[derive(Default)]
struct BinaryPhenotypeSummary {
    case_count: i64,
    control_count: i64,
    is_binary_coded: bool,
}

/// Validate single-trait values and covariate shape before native BGEN delivery.
///
/// # Errors
///
/// Returns an error when values are non-finite, dimensions are inconsistent,
/// covariates are rank deficient, or binary phenotypes are not coded 0/1.
pub fn validate_single_trait_preflight_values(
    phenotype_values: &[f32],
    covariate_row_count: usize,
    covariate_column_count: usize,
    covariate_values: &[f32],
    is_binary_trait: bool,
) -> Result<SingleTraitPreflightShapePayload, PreflightError> {
    validate_finite_f32_values("Phenotype", phenotype_values)?;
    validate_finite_f32_values("Covariate matrix", covariate_values)?;
    let shape = validate_single_trait_preflight_shape_payload(
        usize_to_i64(phenotype_values.len(), "phenotype sample count")?,
        2,
        usize_to_i64(covariate_row_count, "covariate sample count")?,
        usize_to_i64(covariate_column_count, "covariate count")?,
    )?;
    validate_covariate_matrix_rank_values(covariate_row_count, covariate_column_count, covariate_values)?;
    if is_binary_trait {
        validate_binary_phenotype_values(phenotype_values)?;
    }
    Ok(shape)
}

/// Validate one single-trait prediction vector before native BGEN delivery.
///
/// # Errors
///
/// Returns an error when the prediction vector is non-finite or has the wrong sample count.
pub fn validate_single_prediction_values(
    chromosome: &str,
    prediction_values: &[f32],
    sample_count: i64,
) -> Result<(), PreflightError> {
    validate_single_prediction_preflight_shape(
        chromosome,
        &[usize_to_i64(prediction_values.len(), "prediction sample count")?],
        sample_count,
    )?;
    validate_finite_f32_values(&format!("Prediction values for chromosome {chromosome}"), prediction_values)
}

/// Validate multi-trait values and covariate shape before native BGEN delivery.
///
/// # Errors
///
/// Returns an error when values are non-finite, dimensions are inconsistent,
/// covariates are rank deficient, or binary phenotypes are not coded 0/1.
pub fn validate_multi_trait_preflight_values(
    phenotype_row_count: usize,
    phenotype_column_count: usize,
    phenotype_values: &[f32],
    covariate_row_count: usize,
    covariate_column_count: usize,
    covariate_values: &[f32],
    is_binary_trait: bool,
) -> Result<MultiTraitPreflightShapePayload, PreflightError> {
    validate_finite_f32_values("Phenotype matrix", phenotype_values)?;
    validate_finite_f32_values("Covariate matrix", covariate_values)?;
    let shape = validate_multi_trait_preflight_shape_payload(
        2,
        usize_to_i64(phenotype_row_count, "trait count")?,
        usize_to_i64(phenotype_column_count, "phenotype sample count")?,
        2,
        usize_to_i64(covariate_row_count, "covariate sample count")?,
        usize_to_i64(covariate_column_count, "covariate count")?,
    )?;
    validate_matrix_value_count(
        phenotype_row_count,
        phenotype_column_count,
        phenotype_values.len(),
        PreflightError::PhenotypeMatrixValueCountMismatch,
    )?;
    validate_covariate_matrix_rank_values(covariate_row_count, covariate_column_count, covariate_values)?;
    if is_binary_trait {
        for phenotype_values_for_trait in phenotype_values.chunks_exact(phenotype_column_count) {
            validate_binary_phenotype_values(phenotype_values_for_trait)?;
        }
    }
    Ok(shape)
}

/// Validate one multi-trait prediction matrix before native BGEN delivery.
///
/// # Errors
///
/// Returns an error when the prediction matrix is non-finite or has the wrong shape.
pub fn validate_multi_prediction_values(
    chromosome: &str,
    prediction_values: &[f32],
    trait_count: i64,
    sample_count: i64,
) -> Result<(), PreflightError> {
    validate_multi_prediction_preflight_shape(chromosome, &[trait_count, sample_count], trait_count, sample_count)?;
    validate_finite_f32_values(&format!("Prediction matrix for chromosome {chromosome}"), prediction_values)
}

fn validate_finite_f32_values(label: &str, values: &[f32]) -> Result<(), PreflightError> {
    let all_values_finite = values.iter().copied().all(f32::is_finite);
    validate_finite_array(label, all_values_finite)
}

fn validate_binary_phenotype_values(phenotype_values: &[f32]) -> Result<(), PreflightError> {
    let mut summary = BinaryPhenotypeSummary { is_binary_coded: true, ..BinaryPhenotypeSummary::default() };
    for value in phenotype_values {
        if matches!(value.classify(), std::num::FpCategory::Zero) {
            summary.control_count += 1;
        } else if value.to_bits() == 1.0_f32.to_bits() {
            summary.case_count += 1;
        } else {
            summary.is_binary_coded = false;
        }
    }
    validate_binary_phenotype_coding(summary.is_binary_coded)?;
    validate_binary_phenotype_case_control_counts(summary.case_count, summary.control_count)
}

fn validate_covariate_matrix_rank_values(
    row_count: usize,
    column_count: usize,
    values: &[f32],
) -> Result<(), PreflightError> {
    validate_matrix_value_count(
        row_count,
        column_count,
        values.len(),
        PreflightError::CovariateMatrixValueCountMismatch,
    )?;
    let rank_values = values.iter().copied().map(f64::from).collect::<Vec<_>>();
    let rank_matrix = DMatrix::from_row_slice(row_count, column_count, &rank_values);
    let singular_values = rank_matrix.svd(false, false).singular_values;
    let largest_singular_value = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance =
        largest_singular_value * dimension_count_as_f64(row_count.max(column_count)) * f64::from(f32::EPSILON);
    let covariate_rank = singular_values.iter().filter(|singular_value| **singular_value > tolerance).count();
    validate_covariate_matrix_rank(
        usize_to_i64(covariate_rank, "covariate rank")?,
        usize_to_i64(column_count, "covariate count")?,
    )
}

fn validate_matrix_value_count(
    row_count: usize,
    column_count: usize,
    observed_value_count: usize,
    mismatch_error: PreflightError,
) -> Result<(), PreflightError> {
    let expected_value_count =
        row_count.checked_mul(column_count).ok_or(PreflightError::CovariateMatrixShapeOverflow)?;
    if observed_value_count != expected_value_count {
        return Err(mismatch_error);
    }
    Ok(())
}

fn usize_to_i64(value: usize, value_name: &'static str) -> Result<i64, PreflightError> {
    i64::try_from(value).map_err(|_| PreflightError::CountOverflow { label: value_name })
}

#[allow(clippy::cast_precision_loss)]
fn dimension_count_as_f64(dimension_count: usize) -> f64 {
    dimension_count as f64
}
