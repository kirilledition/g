//! Output writer adapter planning.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType, Float32Type, Float64Type, Int32Type};

use crate::chunk::NativeChunkHandle;
use crate::error::{OutputError, OutputResult};
use crate::schema::OutputStatisticDtype;
use crate::session::{OutputWriterSession, finish_interrupted_output_writer_sessions, finish_output_writer_sessions};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SingleTraitOutputWritePlan {
    pub uses_float64_native_writer: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiTraitOutputWritePlan {
    pub active_trait_count: usize,
    pub use_native_multi_writer: bool,
    pub uses_float64_native_writer: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WriterFinishExecutionPlan {
    pub writer_session_count: usize,
    pub thread_count: usize,
}

pub struct Regenie2StatisticSliceBundle<'a, T> {
    pub beta: &'a [T],
    pub standard_error: &'a [T],
    pub chi_squared: &'a [T],
    pub log10_p_value: &'a [T],
    pub extra_code: Option<&'a [i32]>,
}

impl WriterFinishExecutionPlan {
    #[must_use]
    pub const fn has_writer_sessions(&self) -> bool {
        self.writer_session_count > 0
    }

    #[must_use]
    pub const fn uses_parallel_finish(&self) -> bool {
        self.thread_count > 1
    }
}

/// Plan the native writer array type for one single-trait output write.
///
/// # Errors
///
/// Returns an error when the output statistic dtype is unsupported.
pub fn plan_single_trait_output_write(output_statistic_dtype: &str) -> OutputResult<SingleTraitOutputWritePlan> {
    let output_statistic_dtype = OutputStatisticDtype::parse(output_statistic_dtype)?;
    Ok(SingleTraitOutputWritePlan {
        uses_float64_native_writer: output_statistic_dtype == OutputStatisticDtype::Float64,
    })
}

/// Plan the native bulk writer path for one multi-trait output write.
///
/// # Errors
///
/// Returns an error when the output statistic dtype is unsupported.
pub fn plan_multi_trait_output_write(
    active_trait_count: usize,
    output_statistic_dtype: &str,
) -> OutputResult<MultiTraitOutputWritePlan> {
    let output_statistic_dtype = OutputStatisticDtype::parse(output_statistic_dtype)?;
    let use_native_multi_writer = active_trait_count > 0;
    Ok(MultiTraitOutputWritePlan {
        active_trait_count,
        use_native_multi_writer,
        uses_float64_native_writer: use_native_multi_writer && output_statistic_dtype == OutputStatisticDtype::Float64,
    })
}

/// Plan how writer sessions should be finished.
///
/// # Errors
///
/// Returns an error when at least one writer must finish and the requested
/// thread count is non-positive or cannot fit in `usize`.
pub fn plan_writer_finish_execution(
    writer_session_count: i64,
    requested_thread_count: i64,
) -> OutputResult<WriterFinishExecutionPlan> {
    let thread_count = resolve_writer_finish_thread_count(writer_session_count, requested_thread_count)?;
    let writer_session_count = if writer_session_count <= 0 {
        0
    } else {
        usize::try_from(writer_session_count).map_err(|_| {
            OutputError::InvalidInput(format!("Writer session count exceeds platform capacity: {writer_session_count}"))
        })?
    };
    Ok(WriterFinishExecutionPlan { writer_session_count, thread_count })
}

/// Resolve the thread count used to finish output writer sessions.
///
/// # Errors
///
/// Returns an error when at least one writer must finish and the requested
/// thread count is non-positive or cannot fit in `usize`.
pub fn resolve_writer_finish_thread_count(
    writer_session_count: i64,
    requested_thread_count: i64,
) -> OutputResult<usize> {
    if writer_session_count <= 0 {
        return Ok(0);
    }
    if requested_thread_count <= 0 {
        return Err(OutputError::InvalidInput("Writer finish thread count must be positive.".to_string()));
    }
    let writer_session_count = usize::try_from(writer_session_count).map_err(|_| {
        OutputError::InvalidInput(format!("Writer session count exceeds platform capacity: {writer_session_count}"))
    })?;
    let requested_thread_count = usize::try_from(requested_thread_count).map_err(|_| {
        OutputError::InvalidInput(format!(
            "Writer finish thread count exceeds platform capacity: {requested_thread_count}"
        ))
    })?;
    Ok(writer_session_count.min(requested_thread_count))
}

/// Finish writer sessions after resolving the requested parallelism.
///
/// # Errors
///
/// Returns an error when finish planning or writer finalization fails.
pub fn finish_output_writer_sessions_with_requested_threads(
    writer_sessions: &[&OutputWriterSession],
    requested_thread_count: i64,
) -> OutputResult<Vec<Option<PathBuf>>> {
    let writer_session_count = writer_session_count_as_i64(writer_sessions.len())?;
    let finish_plan = plan_writer_finish_execution(writer_session_count, requested_thread_count)?;
    finish_output_writer_sessions(writer_sessions, finish_plan.thread_count)
}

/// Flush interrupted writer sessions after resolving the requested parallelism.
///
/// # Errors
///
/// Returns an error when finish planning or interrupted writer flushing fails.
pub fn finish_interrupted_output_writer_sessions_with_requested_threads(
    writer_sessions: &[&OutputWriterSession],
    requested_thread_count: i64,
    signal_name: &str,
) -> OutputResult<()> {
    let writer_session_count = writer_session_count_as_i64(writer_sessions.len())?;
    let finish_plan = plan_writer_finish_execution(writer_session_count, requested_thread_count)?;
    finish_interrupted_output_writer_sessions(writer_sessions, finish_plan.thread_count, signal_name)
}

/// Write one f32 REGENIE statistic row to each active trait writer.
///
/// # Errors
///
/// Returns an error when an active trait index is out of bounds or a writer
/// rejects the chunk.
pub fn write_regenie2_multi_trait_chunk_f32(
    writer_sessions: &[&OutputWriterSession],
    active_trait_indices: &[usize],
    chunk_handle: &NativeChunkHandle,
    active_statistic_rows: &[Regenie2StatisticSliceBundle<'_, f32>],
) -> OutputResult<()> {
    write_regenie2_multi_trait_chunk::<f32, Float32Type>(
        writer_sessions,
        active_trait_indices,
        chunk_handle,
        active_statistic_rows,
    )
}

/// Write one f64 REGENIE statistic row to each active trait writer.
///
/// # Errors
///
/// Returns an error when an active trait index is out of bounds or a writer
/// rejects the chunk.
pub fn write_regenie2_multi_trait_chunk_f64(
    writer_sessions: &[&OutputWriterSession],
    active_trait_indices: &[usize],
    chunk_handle: &NativeChunkHandle,
    active_statistic_rows: &[Regenie2StatisticSliceBundle<'_, f64>],
) -> OutputResult<()> {
    write_regenie2_multi_trait_chunk::<f64, Float64Type>(
        writer_sessions,
        active_trait_indices,
        chunk_handle,
        active_statistic_rows,
    )
}

fn writer_session_count_as_i64(writer_session_count: usize) -> OutputResult<i64> {
    i64::try_from(writer_session_count)
        .map_err(|_| OutputError::InvalidInput("Writer session count exceeds native int64 capacity.".to_string()))
}

fn write_regenie2_multi_trait_chunk<T, ArrowType>(
    writer_sessions: &[&OutputWriterSession],
    active_trait_indices: &[usize],
    chunk_handle: &NativeChunkHandle,
    active_statistic_rows: &[Regenie2StatisticSliceBundle<'_, T>],
) -> OutputResult<()>
where
    T: ArrowNativeType,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    if active_trait_indices.len() != active_statistic_rows.len() {
        return Err(OutputError::InvalidInput(
            "Active trait index count must match active statistic row count.".to_string(),
        ));
    }
    for (&trait_index, statistic_row) in active_trait_indices.iter().zip(active_statistic_rows.iter()) {
        let writer_session = writer_sessions.get(trait_index).ok_or_else(|| {
            OutputError::InvalidInput("Active trait index is out of bounds for writer sessions.".to_string())
        })?;
        writer_session.write_regenie2_native_chunk_handle_arrays(
            chunk_handle.clone(),
            build_copied_arrow_array::<T, ArrowType>(statistic_row.beta),
            build_copied_arrow_array::<T, ArrowType>(statistic_row.standard_error),
            build_copied_arrow_array::<T, ArrowType>(statistic_row.chi_squared),
            build_copied_arrow_array::<T, ArrowType>(statistic_row.log10_p_value),
            statistic_row.extra_code.map(build_copied_arrow_array::<i32, Int32Type>),
        )?;
    }
    Ok(())
}

fn build_copied_arrow_array<T, ArrowType>(values: &[T]) -> ArrayRef
where
    T: ArrowNativeType,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    Arc::new(PrimitiveArray::<ArrowType>::from_iter_values(values.iter().copied()))
}
