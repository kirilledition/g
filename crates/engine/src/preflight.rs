//! Native preflight report and scan-shape helpers.

mod common;
mod error;
mod payloads;
mod prediction;
mod report;
mod shape;
mod variants;

pub use error::PreflightError;
pub use payloads::{MultiTraitPreflightShapePayload, PreflightReportPayload, SingleTraitPreflightShapePayload};
pub use prediction::{validate_multi_prediction_preflight_shape, validate_single_prediction_preflight_shape};
pub use report::{build_preflight_report_payload, build_preflight_warnings};
pub use shape::{
    validate_binary_phenotype_case_control_counts, validate_binary_phenotype_coding, validate_covariate_matrix_rank,
    validate_finite_array, validate_multi_trait_preflight_shape_payload, validate_single_trait_preflight_shape_payload,
};
pub use variants::{resolve_preflight_variant_count, resolve_scanned_variant_count};

#[cfg(test)]
use common::format_python_shape;

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
    fn resolves_python_facing_preflight_variant_count() {
        assert_eq!(resolve_preflight_variant_count(12, None).unwrap(), 12);
        assert_eq!(resolve_preflight_variant_count(12, Some(5)).unwrap(), 5);
        assert_eq!(resolve_preflight_variant_count(12, Some(50)).unwrap(), 12);
        assert_eq!(resolve_preflight_variant_count(0, None).unwrap_err(), PreflightError::EmptyBgenInput);
        assert_eq!(resolve_preflight_variant_count(-1, None).unwrap_err(), PreflightError::EmptyBgenInput);
        assert_eq!(resolve_preflight_variant_count(12, Some(0)).unwrap_err(), PreflightError::EmptyBgenScan);
        assert_eq!(resolve_preflight_variant_count(12, Some(-1)).unwrap_err(), PreflightError::EmptyBgenScan);
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
