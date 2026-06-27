//! Native preflight report and scan-shape helpers.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreflightReportPayload {
    pub sample_count: i64,
    pub covariate_count: i64,
    pub chromosome_count: i64,
    pub warning_messages: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SingleTraitPreflightShapePayload {
    pub sample_count: i64,
    pub covariate_count: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiTraitPreflightShapePayload {
    pub trait_count: i64,
    pub sample_count: i64,
    pub covariate_count: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PreflightError {
    #[error("BGEN input contains no variants.")]
    EmptyBgenInput,
    #[error("BGEN scan contains no variants.")]
    EmptyBgenScan,
    #[error("Could not read chromosome boundary metadata: {message}")]
    ChromosomeMetadata { message: String },
    #[error("{label} cannot be negative: {count}")]
    NegativeCount { label: &'static str, count: i64 },
    #[error("Phenotype matrix must be two-dimensional.")]
    PhenotypeMatrixDimension,
    #[error("Phenotype matrix must contain at least one trait.")]
    EmptyPhenotypeTraitSet,
    #[error("Phenotype matrix must contain at least one sample.")]
    EmptyPhenotypeSampleSet,
    #[error("Covariate matrix must be two-dimensional.")]
    CovariateMatrixDimension,
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
    #[error(
        "Prediction sample count for chromosome {chromosome} is {actual_sample_count}, expected {expected_sample_count}."
    )]
    PredictionSampleCountMismatch { chromosome: String, actual_sample_count: i64, expected_sample_count: i64 },
    #[error("Prediction matrix shape for chromosome {chromosome} is {actual_shape}, expected {expected_shape}.")]
    PredictionMatrixShapeMismatch { chromosome: String, actual_shape: String, expected_shape: String },
}

#[must_use]
pub fn build_preflight_warnings(
    sample_count: i64,
    covariate_count: i64,
    trusted_no_missing_diploid: bool,
) -> Vec<String> {
    let mut warning_messages = Vec::new();
    let residual_degrees_of_freedom = sample_count - covariate_count;
    if residual_degrees_of_freedom < 10 {
        warning_messages.push("REGENIE step 2 is running with fewer than 10 residual degrees of freedom.".to_string());
    }
    if trusted_no_missing_diploid {
        warning_messages
            .push("Trusted no-missing diploid BGEN path is enabled after compatibility validation.".to_string());
    }
    warning_messages
}

/// Build a native preflight report payload from already-validated dimensions.
///
/// # Errors
///
/// Returns an error when a dimension is negative.
pub fn build_preflight_report_payload(
    sample_count: i64,
    covariate_count: i64,
    chromosome_count: i64,
    trusted_no_missing_diploid: bool,
) -> Result<PreflightReportPayload, PreflightError> {
    validate_non_negative_count("sample count", sample_count)?;
    validate_non_negative_count("covariate count", covariate_count)?;
    validate_non_negative_count("chromosome count", chromosome_count)?;
    Ok(PreflightReportPayload {
        sample_count,
        covariate_count,
        chromosome_count,
        warning_messages: build_preflight_warnings(sample_count, covariate_count, trusted_no_missing_diploid),
    })
}

/// Resolve the number of variants that a preflight scan must cover.
///
/// # Errors
///
/// Returns an error when the BGEN input or requested scan contains no variants.
pub fn resolve_scanned_variant_count(
    variant_count: usize,
    variant_limit: Option<usize>,
) -> Result<usize, PreflightError> {
    if variant_count == 0 {
        return Err(PreflightError::EmptyBgenInput);
    }
    let scanned_variant_count = variant_limit.map_or(variant_count, |limit| limit.min(variant_count));
    if scanned_variant_count == 0 {
        return Err(PreflightError::EmptyBgenScan);
    }
    Ok(scanned_variant_count)
}

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

fn validate_shape_counts(label: &'static str, shape_counts: &[i64]) -> Result<(), PreflightError> {
    for &shape_count in shape_counts {
        validate_non_negative_count(label, shape_count)?;
    }
    Ok(())
}

fn format_python_shape(shape_counts: &[i64]) -> String {
    match shape_counts {
        [] => "()".to_string(),
        [shape_count] => format!("({shape_count},)"),
        _ => format!("({})", shape_counts.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(", ")),
    }
}

fn validate_non_negative_count(label: &'static str, count: i64) -> Result<(), PreflightError> {
    if count < 0 {
        return Err(PreflightError::NegativeCount { label, count });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_warning_messages_for_low_degrees_of_freedom_and_trusted_path() {
        let warning_messages = build_preflight_warnings(3, 2, true);

        assert_eq!(
            warning_messages,
            vec![
                "REGENIE step 2 is running with fewer than 10 residual degrees of freedom.".to_string(),
                "Trusted no-missing diploid BGEN path is enabled after compatibility validation.".to_string(),
            ],
        );
    }

    #[test]
    fn builds_preflight_report_payload() {
        let payload = build_preflight_report_payload(12, 2, 3, false).unwrap();

        assert_eq!(payload.sample_count, 12);
        assert_eq!(payload.covariate_count, 2);
        assert_eq!(payload.chromosome_count, 3);
        assert!(payload.warning_messages.is_empty());
    }

    #[test]
    fn rejects_negative_report_dimensions() {
        let error = build_preflight_report_payload(-1, 0, 0, false).unwrap_err();

        assert_eq!(error, PreflightError::NegativeCount { label: "sample count", count: -1 });
    }

    #[test]
    fn resolves_scanned_variant_count() {
        assert_eq!(resolve_scanned_variant_count(12, None).unwrap(), 12);
        assert_eq!(resolve_scanned_variant_count(12, Some(5)).unwrap(), 5);
        assert_eq!(resolve_scanned_variant_count(12, Some(50)).unwrap(), 12);
    }

    #[test]
    fn rejects_empty_variant_scans() {
        assert_eq!(resolve_scanned_variant_count(0, None).unwrap_err(), PreflightError::EmptyBgenInput);
        assert_eq!(resolve_scanned_variant_count(1, Some(0)).unwrap_err(), PreflightError::EmptyBgenScan);
    }

    #[test]
    fn validates_single_trait_shape_payload() {
        let payload = validate_single_trait_preflight_shape_payload(12, 2, 12, 3).unwrap();

        assert_eq!(payload.sample_count, 12);
        assert_eq!(payload.covariate_count, 3);
    }

    #[test]
    fn rejects_invalid_single_trait_shape_policy() {
        assert_eq!(
            validate_single_trait_preflight_shape_payload(12, 1, 12, 3).unwrap_err(),
            PreflightError::CovariateMatrixDimension,
        );
        assert_eq!(
            validate_single_trait_preflight_shape_payload(12, 2, 11, 3).unwrap_err(),
            PreflightError::CovariateSampleCountMismatch,
        );
        assert_eq!(
            validate_single_trait_preflight_shape_payload(3, 2, 3, 3).unwrap_err(),
            PreflightError::NonPositiveResidualDegreesOfFreedom,
        );
    }

    #[test]
    fn validates_multi_trait_shape_payload() {
        let payload = validate_multi_trait_preflight_shape_payload(2, 4, 12, 2, 12, 3).unwrap();

        assert_eq!(payload.trait_count, 4);
        assert_eq!(payload.sample_count, 12);
        assert_eq!(payload.covariate_count, 3);
    }

    #[test]
    fn rejects_invalid_multi_trait_shape_policy() {
        assert_eq!(
            validate_multi_trait_preflight_shape_payload(1, 4, 12, 2, 12, 3).unwrap_err(),
            PreflightError::PhenotypeMatrixDimension,
        );
        assert_eq!(
            validate_multi_trait_preflight_shape_payload(2, 0, 12, 2, 12, 3).unwrap_err(),
            PreflightError::EmptyPhenotypeTraitSet,
        );
        assert_eq!(
            validate_multi_trait_preflight_shape_payload(2, 4, 0, 2, 0, 0).unwrap_err(),
            PreflightError::EmptyPhenotypeSampleSet,
        );
    }

    #[test]
    fn validates_binary_case_control_counts() {
        validate_binary_phenotype_case_control_counts(1, 2).unwrap();

        assert_eq!(
            validate_binary_phenotype_case_control_counts(0, 2).unwrap_err(),
            PreflightError::BinaryPhenotypeMissingClass,
        );
        assert_eq!(
            validate_binary_phenotype_case_control_counts(1, 0).unwrap_err(),
            PreflightError::BinaryPhenotypeMissingClass,
        );
    }

    #[test]
    fn validates_array_finiteness_rank_and_binary_coding_policy() {
        validate_finite_array("Phenotype", true).unwrap();
        validate_covariate_matrix_rank(2, 2).unwrap();
        validate_binary_phenotype_coding(true).unwrap();

        assert_eq!(
            validate_finite_array("Prediction values for chromosome 2", false).unwrap_err(),
            PreflightError::NonFiniteArray { label: "Prediction values for chromosome 2".to_string() },
        );
        assert_eq!(validate_covariate_matrix_rank(1, 2).unwrap_err(), PreflightError::CovariateMatrixRankDeficient,);
        assert_eq!(validate_binary_phenotype_coding(false).unwrap_err(), PreflightError::BinaryPhenotypeCoding,);
    }

    #[test]
    fn validates_prediction_shapes() {
        validate_single_prediction_preflight_shape("1", &[12], 12).unwrap();
        validate_multi_prediction_preflight_shape("2", &[4, 12], 4, 12).unwrap();

        assert_eq!(
            validate_single_prediction_preflight_shape("1", &[11], 12).unwrap_err(),
            PreflightError::PredictionSampleCountMismatch {
                chromosome: "1".to_string(),
                actual_sample_count: 11,
                expected_sample_count: 12,
            },
        );
        assert_eq!(
            validate_multi_prediction_preflight_shape("2", &[4, 11], 4, 12).unwrap_err(),
            PreflightError::PredictionMatrixShapeMismatch {
                chromosome: "2".to_string(),
                actual_shape: "(4, 11)".to_string(),
                expected_shape: "(4, 12)".to_string(),
            },
        );
    }

    #[test]
    fn formats_python_shape_tuples() {
        assert_eq!(format_python_shape(&[]), "()");
        assert_eq!(format_python_shape(&[7]), "(7,)");
        assert_eq!(format_python_shape(&[2, 3]), "(2, 3)");
    }
}
