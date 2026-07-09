//! Native preflight report and scan-shape helpers.

mod arrays;
mod common;
mod error;
mod payloads;
mod prediction;
mod report;
mod shape;
mod variants;

pub use arrays::{
    validate_multi_prediction_values, validate_multi_trait_preflight_values, validate_single_prediction_values,
    validate_single_trait_preflight_values,
};
pub use error::PreflightError;
pub use payloads::{MultiTraitPreflightShapePayload, PreflightReportPayload, SingleTraitPreflightShapePayload};
pub use prediction::{validate_multi_prediction_preflight_shape, validate_single_prediction_preflight_shape};
pub use report::{build_preflight_report_payload, build_preflight_warnings};
pub use shape::{
    validate_binary_phenotype_case_control_counts, validate_binary_phenotype_coding, validate_covariate_matrix_rank,
    validate_finite_array, validate_multi_trait_preflight_shape_payload, validate_single_trait_preflight_shape_payload,
};
pub use variants::{resolve_preflight_variant_count, resolve_scanned_variant_count};
