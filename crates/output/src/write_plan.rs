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
