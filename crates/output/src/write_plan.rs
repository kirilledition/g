//! Output writer adapter planning.

use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType, Float32Type, Float64Type, Int32Type};

use crate::chunk::NativeChunkHandle;
use crate::error::{OutputError, OutputResult};
use crate::session::OutputWriterSession;

pub struct Regenie2StatisticBatch<T> {
    pub beta: Vec<T>,
    pub standard_error: Vec<T>,
    pub chi_squared: Vec<T>,
    pub log10_p_value: Vec<T>,
    pub correction_code: Option<Vec<i32>>,
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
/// # Errors
///
/// Returns an error when an active trait index is out of bounds or a writer
/// rejects the chunk.
pub fn write_regenie2_multi_trait_chunk_f32(
    writer_sessions: &[Arc<OutputWriterSession>],
    active_trait_indices: &[usize],
    chunk_handle: &NativeChunkHandle,
    statistic_batch: Regenie2StatisticBatch<f32>,
) -> OutputResult<()> {
    write_regenie2_multi_trait_chunk::<f32, Float32Type>(
        writer_sessions,
        active_trait_indices,
        chunk_handle,
        statistic_batch,
    )
}

/// Write one f64 REGENIE statistic row to each active trait writer.
///
/// # Errors
///
/// Returns an error when an active trait index is out of bounds or a writer
/// rejects the chunk.
pub fn write_regenie2_multi_trait_chunk_f64(
    writer_sessions: &[Arc<OutputWriterSession>],
    active_trait_indices: &[usize],
    chunk_handle: &NativeChunkHandle,
    statistic_batch: Regenie2StatisticBatch<f64>,
) -> OutputResult<()> {
    write_regenie2_multi_trait_chunk::<f64, Float64Type>(
        writer_sessions,
        active_trait_indices,
        chunk_handle,
        statistic_batch,
    )
}

fn write_regenie2_multi_trait_chunk<T, ArrowType>(
    writer_sessions: &[Arc<OutputWriterSession>],
    active_trait_indices: &[usize],
    chunk_handle: &NativeChunkHandle,
    statistic_batch: Regenie2StatisticBatch<T>,
) -> OutputResult<()>
where
    T: ArrowNativeType,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    let row_count = chunk_handle.row_count();
    let expected_value_count = row_count.checked_mul(active_trait_indices.len()).ok_or_else(|| {
        OutputError::InvalidInput("Trait-major output statistic value count exceeds platform capacity.".to_string())
    })?;
    validate_statistic_batch_lengths(&statistic_batch, expected_value_count)?;
    let statistic_arrays = build_statistic_arrow_arrays::<T, ArrowType>(statistic_batch);
    for (active_trait_position, trait_index) in active_trait_indices.iter().copied().enumerate() {
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

fn validate_statistic_batch_lengths<T>(
    statistic_batch: &Regenie2StatisticBatch<T>,
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

fn build_statistic_arrow_arrays<T, ArrowType>(
    statistic_batch: Regenie2StatisticBatch<T>,
) -> Regenie2StatisticArrowArrays
where
    T: ArrowNativeType,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    Regenie2StatisticArrowArrays {
        beta: build_owned_arrow_array::<T, ArrowType>(statistic_batch.beta),
        standard_error: build_owned_arrow_array::<T, ArrowType>(statistic_batch.standard_error),
        chi_squared: build_owned_arrow_array::<T, ArrowType>(statistic_batch.chi_squared),
        log10_p_value: build_owned_arrow_array::<T, ArrowType>(statistic_batch.log10_p_value),
        correction_code: statistic_batch.correction_code.map(build_owned_arrow_array::<i32, Int32Type>),
    }
}

fn build_owned_arrow_array<T, ArrowType>(values: Vec<T>) -> ArrayRef
where
    T: ArrowNativeType,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    Arc::new(PrimitiveArray::<ArrowType>::new(ScalarBuffer::from(values), None))
}
