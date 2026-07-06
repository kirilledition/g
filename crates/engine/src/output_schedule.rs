//! Output writer scheduling and committed-chunk policy.

use std::collections::BTreeSet;

use crate::schedule::ScheduleError;

const OUTPUT_STATISTIC_DTYPE_FLOAT32: &str = "float32";
const OUTPUT_STATISTIC_DTYPE_FLOAT64: &str = "float64";
pub(crate) const REGENIE2_NATIVE_CHUNK_WRITE_METHOD: &str = "write_regenie2_native_chunk";
pub(crate) const REGENIE2_NATIVE_CHUNK_WRITE_F64_METHOD: &str = "write_regenie2_native_chunk_f64";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiTraitChunkWritePlan {
    pub active_trait_indices: Vec<usize>,
    pub total_trait_count: usize,
}

impl MultiTraitChunkWritePlan {
    #[must_use]
    pub fn active_trait_count(&self) -> usize {
        self.active_trait_indices.len()
    }

    #[must_use]
    pub fn all_traits_committed(&self) -> bool {
        self.active_trait_indices.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WriterFinishExecutionPlan {
    pub writer_session_count: usize,
    pub thread_count: usize,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SingleTraitOutputWritePlan {
    pub method_name: String,
    pub uses_float64_native_writer: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiTraitOutputWritePlan {
    pub active_trait_count: usize,
    pub use_native_multi_writer: bool,
    pub uses_float64_native_writer: bool,
}

#[must_use]
pub fn intersect_committed_chunk_identifier_sets<T>(committed_chunk_identifier_sets: &[BTreeSet<T>]) -> BTreeSet<T>
where
    T: Copy + Ord,
{
    let Some((first_set, remaining_sets)) = committed_chunk_identifier_sets.split_first() else {
        return BTreeSet::new();
    };
    let mut shared_chunk_identifiers = first_set.clone();
    for committed_chunk_identifier_set in remaining_sets {
        shared_chunk_identifiers.retain(|chunk_identifier| committed_chunk_identifier_set.contains(chunk_identifier));
    }
    shared_chunk_identifiers
}

/// Plan which multi-trait writer lanes still need one chunk.
///
/// # Errors
///
/// Returns an error when the committed chunk identifier set count does not
/// match the writer session count.
pub fn plan_multi_trait_chunk_write(
    writer_session_count: usize,
    chunk_identifier: usize,
    committed_chunk_identifier_sets: &[BTreeSet<usize>],
) -> Result<MultiTraitChunkWritePlan, ScheduleError> {
    if committed_chunk_identifier_sets.len() != writer_session_count {
        return Err(ScheduleError::MultiTraitCommittedChunkSetCountMismatch {
            writer_session_count,
            committed_set_count: committed_chunk_identifier_sets.len(),
        });
    }
    let active_trait_indices =
        committed_chunk_identifier_sets
            .iter()
            .enumerate()
            .filter_map(|(trait_index, committed_chunk_identifier_set)| {
                if committed_chunk_identifier_set.contains(&chunk_identifier) { None } else { Some(trait_index) }
            })
            .collect();
    Ok(MultiTraitChunkWritePlan { active_trait_indices, total_trait_count: writer_session_count })
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
) -> Result<usize, ScheduleError> {
    if writer_session_count <= 0 {
        return Ok(0);
    }
    if requested_thread_count <= 0 {
        return Err(ScheduleError::NonPositiveWriterFinishThreadCount);
    }
    let writer_session_count = usize::try_from(writer_session_count)
        .map_err(|_| ScheduleError::WriterSessionCountOverflow { session_count: writer_session_count })?;
    let requested_thread_count = usize::try_from(requested_thread_count)
        .map_err(|_| ScheduleError::WriterFinishThreadCountOverflow { thread_count: requested_thread_count })?;
    Ok(writer_session_count.min(requested_thread_count))
}

/// Plan how writer sessions should be finished by the transitional Python adapter.
///
/// # Errors
///
/// Returns an error when at least one writer must finish and the requested
/// thread count is non-positive or cannot fit in `usize`.
pub fn plan_writer_finish_execution(
    writer_session_count: i64,
    requested_thread_count: i64,
) -> Result<WriterFinishExecutionPlan, ScheduleError> {
    let thread_count = resolve_writer_finish_thread_count(writer_session_count, requested_thread_count)?;
    let writer_session_count = if writer_session_count <= 0 {
        0
    } else {
        usize::try_from(writer_session_count)
            .map_err(|_| ScheduleError::WriterSessionCountOverflow { session_count: writer_session_count })?
    };
    Ok(WriterFinishExecutionPlan { writer_session_count, thread_count })
}

fn output_statistic_dtype_is_float64(output_statistic_dtype: &str) -> Result<bool, ScheduleError> {
    match output_statistic_dtype {
        OUTPUT_STATISTIC_DTYPE_FLOAT32 => Ok(false),
        OUTPUT_STATISTIC_DTYPE_FLOAT64 => Ok(true),
        _ => Err(ScheduleError::UnsupportedOutputStatisticDtype {
            output_statistic_dtype: output_statistic_dtype.to_string(),
        }),
    }
}

/// Plan the Python method used for one single-trait output write.
///
/// # Errors
///
/// Returns an error when the output statistic dtype is unsupported.
pub fn plan_single_trait_output_write(
    is_native_writer_session: bool,
    output_statistic_dtype: &str,
) -> Result<SingleTraitOutputWritePlan, ScheduleError> {
    let is_float64_output_dtype = output_statistic_dtype_is_float64(output_statistic_dtype)?;
    let uses_float64_native_writer = is_native_writer_session && is_float64_output_dtype;
    let method_name = if uses_float64_native_writer {
        REGENIE2_NATIVE_CHUNK_WRITE_F64_METHOD
    } else {
        REGENIE2_NATIVE_CHUNK_WRITE_METHOD
    }
    .to_string();
    Ok(SingleTraitOutputWritePlan { method_name, uses_float64_native_writer })
}

/// Plan the native bulk writer path for one multi-trait output write.
///
/// # Errors
///
/// Returns an error when the output statistic dtype is unsupported.
pub fn plan_multi_trait_output_write(
    active_trait_count: usize,
    all_writer_sessions_native: bool,
    output_statistic_dtype: &str,
) -> Result<MultiTraitOutputWritePlan, ScheduleError> {
    let is_float64_output_dtype = output_statistic_dtype_is_float64(output_statistic_dtype)?;
    let use_native_multi_writer = active_trait_count > 0 && all_writer_sessions_native;
    let uses_float64_native_writer = use_native_multi_writer && is_float64_output_dtype;
    Ok(MultiTraitOutputWritePlan { active_trait_count, use_native_multi_writer, uses_float64_native_writer })
}
