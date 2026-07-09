//! Output writer chunk-array validation.

use arrow::array::{ArrayRef, Float32Array, Float64Array};

use crate::OutputStatisticDtype;
use crate::error::OutputError;

pub(super) fn validate_column_lengths(
    expected_row_count: usize,
    observed_lengths: &[usize],
) -> Result<(), OutputError> {
    if observed_lengths.iter().all(|observed_length| *observed_length == expected_row_count) {
        return Ok(());
    }
    Err(OutputError::InvalidInput(
        "Rust output writer batch column lengths do not all match the expected row count.".to_string(),
    ))
}

pub(super) fn validate_statistic_array_type(
    column_name: &str,
    array: &ArrayRef,
    output_statistic_dtype: OutputStatisticDtype,
) -> Result<(), OutputError> {
    let type_matches = match output_statistic_dtype {
        OutputStatisticDtype::Float32 => array.as_any().is::<Float32Array>(),
        OutputStatisticDtype::Float64 => array.as_any().is::<Float64Array>(),
    };
    if type_matches {
        return Ok(());
    }
    Err(OutputError::InvalidInput(format!(
        "Rust output writer column {column_name} must be {} for the configured output statistic dtype.",
        output_statistic_dtype.as_str(),
    )))
}

/// Validate a trait-major statistic array shape before native output delivery.
///
/// # Errors
///
/// Returns an error when the observed shape is not `(trait_count, row_count)`.
pub fn validate_trait_major_statistic_shape(
    array_name: &str,
    observed_shape: &[usize],
    trait_count: usize,
    row_count: usize,
) -> Result<(), OutputError> {
    if observed_shape == [trait_count, row_count] {
        return Ok(());
    }
    Err(OutputError::InvalidInput(format!(
        "{array_name} must have shape ({trait_count}, {row_count}) for multi-trait output."
    )))
}
