//! Output writer adapter planning.

use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType, Float32Type, UInt8Type};

use crate::chunk::NativeChunkHandle;
use crate::error::{OutputError, OutputResult};
use crate::session::OutputWriterSession;

#[derive(Debug, PartialEq)]
pub struct Regenie2StatisticBatch {
    pub trait_count: usize,
    pub variant_count: usize,
    pub beta: Vec<f32>,
    pub standard_error: Vec<f32>,
    pub chi_squared: Vec<f32>,
    pub log10_p_value: Vec<f32>,
    pub correction_code: Option<Vec<u8>>,
}

struct Regenie2StatisticArrowArrays {
    beta: ArrayRef,
    standard_error: ArrayRef,
    chi_squared: ArrayRef,
    log10_p_value: ArrayRef,
    correction_code: Option<ArrayRef>,
}

/// Write one f32 REGENIE statistic row to each active trait writer.
///
/// `None` selects every writer in identity order without allocating an index
/// vector. A slice selects the corresponding subset for resumed output.
///
/// # Errors
///
/// Returns an error when an active trait index is out of bounds or a writer
/// rejects the chunk.
pub fn write_regenie2_multi_trait_chunk_f32(
    writer_sessions: &[Arc<OutputWriterSession>],
    active_trait_indices: Option<&[usize]>,
    chunk_handle: &NativeChunkHandle,
    statistic_batch: Regenie2StatisticBatch,
) -> OutputResult<()> {
    let row_count = chunk_handle.row_count();
    let active_trait_count = active_trait_indices.map_or(writer_sessions.len(), <[usize]>::len);
    let expected_value_count = row_count.checked_mul(active_trait_count).ok_or_else(|| {
        OutputError::InvalidInput("Trait-major output statistic value count exceeds platform capacity.".to_string())
    })?;
    if statistic_batch.trait_count != active_trait_count || statistic_batch.variant_count != row_count {
        return Err(OutputError::InvalidInput(format!(
            "Materialized statistic shape ({}, {}) does not match expected ({active_trait_count}, {row_count}).",
            statistic_batch.trait_count, statistic_batch.variant_count
        )));
    }
    validate_statistic_batch_lengths(&statistic_batch, expected_value_count)?;
    let statistic_arrays = build_statistic_arrow_arrays(statistic_batch);
    for active_trait_position in 0..active_trait_count {
        let trait_index = active_trait_indices.map_or(active_trait_position, |indices| indices[active_trait_position]);
        let writer_session = writer_sessions.get(trait_index).map(Arc::as_ref).ok_or_else(|| {
            OutputError::InvalidInput("Active trait index is out of bounds for writer sessions.".to_string())
        })?;
        let row_start = active_trait_position * row_count;
        writer_session.write_regenie2_native_chunk_handle_arrays(
            chunk_handle.clone(),
            statistic_arrays.beta.slice(row_start, row_count),
            statistic_arrays.standard_error.slice(row_start, row_count),
            statistic_arrays.chi_squared.slice(row_start, row_count),
            statistic_arrays.log10_p_value.slice(row_start, row_count),
            statistic_arrays
                .correction_code
                .as_ref()
                .map(|correction_code| correction_code.slice(row_start, row_count)),
        )?;
    }
    Ok(())
}

fn validate_statistic_batch_lengths(
    statistic_batch: &Regenie2StatisticBatch,
    expected_value_count: usize,
) -> OutputResult<()> {
    let observed_value_counts = [
        statistic_batch.beta.len(),
        statistic_batch.standard_error.len(),
        statistic_batch.chi_squared.len(),
        statistic_batch.log10_p_value.len(),
    ];
    if observed_value_counts.iter().any(|value_count| *value_count != expected_value_count) {
        return Err(OutputError::InvalidInput(format!(
            "Trait-major output statistic value counts {observed_value_counts:?} do not match expected count {expected_value_count}."
        )));
    }
    if let Some(correction_code) = statistic_batch.correction_code.as_ref()
        && correction_code.len() != expected_value_count
    {
        return Err(OutputError::InvalidInput(format!(
            "Trait-major correction-code value count {} does not match expected count {expected_value_count}.",
            correction_code.len()
        )));
    }
    Ok(())
}

fn build_statistic_arrow_arrays(statistic_batch: Regenie2StatisticBatch) -> Regenie2StatisticArrowArrays {
    Regenie2StatisticArrowArrays {
        beta: build_owned_arrow_array::<f32, Float32Type>(statistic_batch.beta),
        standard_error: build_owned_arrow_array::<f32, Float32Type>(statistic_batch.standard_error),
        chi_squared: build_owned_arrow_array::<f32, Float32Type>(statistic_batch.chi_squared),
        log10_p_value: build_owned_arrow_array::<f32, Float32Type>(statistic_batch.log10_p_value),
        correction_code: statistic_batch.correction_code.map(build_owned_arrow_array::<u8, UInt8Type>),
    }
}

fn build_owned_arrow_array<T, ArrowType>(values: Vec<T>) -> ArrayRef
where
    T: ArrowNativeType,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    Arc::new(PrimitiveArray::<ArrowType>::new(ScalarBuffer::from(values), None))
}

#[cfg(test)]
mod tests {
    use super::{Regenie2StatisticBatch, validate_statistic_batch_lengths};

    fn statistic_batch(value_count: usize) -> Regenie2StatisticBatch {
        Regenie2StatisticBatch {
            trait_count: 1,
            variant_count: value_count,
            beta: vec![0.1; value_count],
            standard_error: vec![0.2; value_count],
            chi_squared: vec![0.25; value_count],
            log10_p_value: vec![0.3; value_count],
            correction_code: Some(vec![0; value_count]),
        }
    }

    #[test]
    fn statistic_batch_accepts_exact_trait_major_shape() {
        validate_statistic_batch_lengths(&statistic_batch(4), 4).expect("matching columns are valid");
    }

    #[test]
    fn statistic_batch_rejects_each_mismatched_result_column() {
        for column_index in 0..4 {
            let mut batch = statistic_batch(4);
            match column_index {
                0 => {
                    batch.beta.pop();
                }
                1 => {
                    batch.standard_error.pop();
                }
                2 => {
                    batch.chi_squared.pop();
                }
                3 => {
                    batch.log10_p_value.pop();
                }
                _ => unreachable!("test column index is bounded"),
            }
            let error = validate_statistic_batch_lengths(&batch, 4).expect_err("mismatched result column must fail");
            assert!(error.to_string().contains("value counts"));
        }
    }

    #[test]
    fn statistic_batch_rejects_mismatched_correction_codes() {
        let mut batch = statistic_batch(4);
        batch.correction_code.as_mut().expect("test correction codes exist").pop();
        let error = validate_statistic_batch_lengths(&batch, 4).expect_err("mismatched correction codes must fail");
        assert!(error.to_string().contains("correction-code value count 3"));
    }
}
