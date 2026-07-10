//! Native preflight report and scan-shape helpers.

mod arrays;
mod common;
mod error;
mod payloads;
mod prediction;
mod report;
mod shape;

pub use arrays::{
    validate_multi_prediction_values, validate_multi_trait_preflight_values, validate_single_prediction_values,
    validate_single_trait_preflight_values,
};
pub use error::PreflightError;
pub use payloads::{MultiTraitPreflightShapePayload, PreflightReportPayload, SingleTraitPreflightShapePayload};
pub use report::build_preflight_report_payload;
