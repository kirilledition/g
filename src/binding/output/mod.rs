#![allow(clippy::needless_pass_by_value)]

pub(crate) mod profile;

use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType, Float32Type, Float64Type, Int32Type};
use g_output::{
    NativeChunkHandle, NativeChunkStats as NativeOutputChunkStats, OutputWriterSession as NativeOutputWriterSession,
    VariantMetadataColumns as NativeOutputVariantMetadataColumns,
};
use g_runtime as native_run_events;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use super::{
    errors,
    genotype::{ChunkStats as PyChunkStats, VariantMetadata as PyVariantMetadata},
    run_events,
};

#[pyclass]
pub(crate) struct OutputWriterSession {
    inner: NativeOutputWriterSession,
}

struct Regenie2StatisticArrays {
    beta: ArrayRef,
    standard_error: ArrayRef,
    chi_squared: ArrayRef,
    log10_p_value: ArrayRef,
    extra_code: Option<ArrayRef>,
}

#[pymethods]
impl OutputWriterSession {
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (metadata, chunk_stats, beta, standard_error, chi_squared, log10_p_value, extra_code=None))]
    fn write_regenie2_native_chunk(
        &self,
        py: Python<'_>,
        metadata: PyRef<'_, PyVariantMetadata>,
        chunk_stats: PyRef<'_, PyChunkStats>,
        beta: PyReadonlyArray1<'_, f32>,
        standard_error: PyReadonlyArray1<'_, f32>,
        chi_squared: PyReadonlyArray1<'_, f32>,
        log10_p_value: PyReadonlyArray1<'_, f32>,
        extra_code: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<()> {
        let chunk_handle = build_native_chunk_handle_from_python(&metadata, &chunk_stats)?;
        let beta_slice = beta.as_slice()?;
        let standard_error_slice = standard_error.as_slice()?;
        let chi_squared_slice = chi_squared.as_slice()?;
        let log10_p_value_slice = log10_p_value.as_slice()?;
        let extra_code_slice = extra_code.as_ref().map(|array| array.as_slice()).transpose()?;
        let beta_array = build_copied_arrow_array::<f32, Float32Type>(beta_slice);
        let standard_error_array = build_copied_arrow_array::<f32, Float32Type>(standard_error_slice);
        let chi_squared_array = build_copied_arrow_array::<f32, Float32Type>(chi_squared_slice);
        let log10_p_value_array = build_copied_arrow_array::<f32, Float32Type>(log10_p_value_slice);
        let extra_code_array = match (extra_code.as_ref(), extra_code_slice) {
            (None, None) => None,
            (Some(_), Some(extra_code_slice_values)) => {
                Some(build_copied_arrow_array::<i32, Int32Type>(extra_code_slice_values))
            }
            _ => return Err(PyRuntimeError::new_err("Extra code array state was inconsistent.")),
        };
        let statistic_arrays = Regenie2StatisticArrays {
            beta: beta_array,
            standard_error: standard_error_array,
            chi_squared: chi_squared_array,
            log10_p_value: log10_p_value_array,
            extra_code: extra_code_array,
        };
        write_regenie2_chunk_arrays_detached(
            py,
            &self.inner,
            chunk_handle,
            statistic_arrays,
            "write_regenie2_native_chunk",
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (metadata, chunk_stats, beta, standard_error, chi_squared, log10_p_value, extra_code=None))]
    fn write_regenie2_native_chunk_f64(
        &self,
        py: Python<'_>,
        metadata: PyRef<'_, PyVariantMetadata>,
        chunk_stats: PyRef<'_, PyChunkStats>,
        beta: PyReadonlyArray1<'_, f64>,
        standard_error: PyReadonlyArray1<'_, f64>,
        chi_squared: PyReadonlyArray1<'_, f64>,
        log10_p_value: PyReadonlyArray1<'_, f64>,
        extra_code: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<()> {
        let chunk_handle = build_native_chunk_handle_from_python(&metadata, &chunk_stats)?;
        let beta_slice = beta.as_slice()?;
        let standard_error_slice = standard_error.as_slice()?;
        let chi_squared_slice = chi_squared.as_slice()?;
        let log10_p_value_slice = log10_p_value.as_slice()?;
        let extra_code_slice = extra_code.as_ref().map(|array| array.as_slice()).transpose()?;
        let beta_array = build_copied_arrow_array::<f64, Float64Type>(beta_slice);
        let standard_error_array = build_copied_arrow_array::<f64, Float64Type>(standard_error_slice);
        let chi_squared_array = build_copied_arrow_array::<f64, Float64Type>(chi_squared_slice);
        let log10_p_value_array = build_copied_arrow_array::<f64, Float64Type>(log10_p_value_slice);
        let extra_code_array = match (extra_code.as_ref(), extra_code_slice) {
            (None, None) => None,
            (Some(_), Some(extra_code_slice_values)) => {
                Some(build_copied_arrow_array::<i32, Int32Type>(extra_code_slice_values))
            }
            _ => return Err(PyRuntimeError::new_err("Extra code array state was inconsistent.")),
        };
        let statistic_arrays = Regenie2StatisticArrays {
            beta: beta_array,
            standard_error: standard_error_array,
            chi_squared: chi_squared_array,
            log10_p_value: log10_p_value_array,
            extra_code: extra_code_array,
        };
        write_regenie2_chunk_arrays_detached(
            py,
            &self.inner,
            chunk_handle,
            statistic_arrays,
            "write_regenie2_native_chunk_f64",
        )
    }

    fn finish(&self, py: Python<'_>) -> PyResult<Option<String>> {
        py.detach(|| self.inner.finish())
            .map(|maybe_path| maybe_path.map(|path| path.display().to_string()))
            .map_err(|error| errors::convert_output_error("finish_output_writer", error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn finish_interrupted(&self, py: Python<'_>, signal_name: String) -> PyResult<()> {
        py.detach(|| self.inner.finish_interrupted(&signal_name))
            .map_err(|error| errors::convert_output_error("finish_interrupted", error))
    }

    fn abort(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.abort()).map_err(|error| errors::convert_output_error("abort_output_writer", error))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_output_writer_session_batch<'py>(
    py: Python<'py>,
    run_directories: Vec<String>,
    chunks_directories: Vec<String>,
    association_mode: &str,
    writer_thread_count: usize,
    writer_queue_depth: usize,
    output_format: &str,
    output_statistic_dtype: &str,
    finalize_parquet: bool,
    chunks_per_arrow_file: usize,
    arrow_compression: &str,
    parquet_compression: &str,
    collect_stage_timings: bool,
) -> PyResult<Vec<Py<OutputWriterSession>>> {
    if run_directories.len() != chunks_directories.len() {
        return Err(PyValueError::new_err(format!(
            "Output writer run directory count ({}) does not match chunks directory count ({}).",
            run_directories.len(),
            chunks_directories.len()
        )));
    }
    let writer_session_count = writer_session_count_as_i64(run_directories.len())?;
    record_pipeline_output_writer_sessions_create_started(association_mode, writer_session_count)?;
    g_output::create_output_writer_sessions(
        run_directories,
        chunks_directories,
        association_mode,
        writer_thread_count,
        writer_queue_depth,
        output_format,
        output_statistic_dtype,
        finalize_parquet,
        chunks_per_arrow_file,
        arrow_compression,
        parquet_compression,
        collect_stage_timings,
    )
    .map_err(|error| errors::convert_output_error("create_output_writer_sessions", error))?
    .into_iter()
    .map(|inner| Py::new(py, OutputWriterSession { inner }))
    .collect()
}

fn extract_optional_readonly_array1_i32<'py>(
    extra_code: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<PyReadonlyArray1<'py, i32>>> {
    extra_code.map(|array| array.extract().map_err(Into::into)).transpose()
}

fn extract_optional_readonly_array2_i32<'py>(
    extra_code: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<PyReadonlyArray2<'py, i32>>> {
    extra_code.map(|array| array.extract().map_err(Into::into)).transpose()
}

#[pyfunction]
#[pyo3(signature = (writer_session, metadata, chunk_stats, output_statistic_dtype, beta, standard_error, chi_squared, log10_p_value, extra_code=None))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn write_regenie2_native_chunk_with_output_dtype<'py>(
    py: Python<'py>,
    writer_session: PyRef<'py, OutputWriterSession>,
    metadata: PyRef<'py, PyVariantMetadata>,
    chunk_stats: PyRef<'py, PyChunkStats>,
    output_statistic_dtype: String,
    beta: &Bound<'py, PyAny>,
    standard_error: &Bound<'py, PyAny>,
    chi_squared: &Bound<'py, PyAny>,
    log10_p_value: &Bound<'py, PyAny>,
    extra_code: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    let write_plan = g_output::plan_single_trait_output_write(&output_statistic_dtype)
        .map_err(|error| errors::convert_output_error("plan_single_trait_output_write", error))?;
    if write_plan.uses_float64_native_writer {
        return writer_session.write_regenie2_native_chunk_f64(
            py,
            metadata,
            chunk_stats,
            beta.extract()?,
            standard_error.extract()?,
            chi_squared.extract()?,
            log10_p_value.extract()?,
            extract_optional_readonly_array1_i32(extra_code)?,
        );
    }
    writer_session.write_regenie2_native_chunk(
        py,
        metadata,
        chunk_stats,
        beta.extract()?,
        standard_error.extract()?,
        chi_squared.extract()?,
        log10_p_value.extract()?,
        extract_optional_readonly_array1_i32(extra_code)?,
    )
}

#[pyfunction]
#[pyo3(signature = (writer_sessions, active_trait_indices, metadata, chunk_stats, output_statistic_dtype, beta, standard_error, chi_squared, log10_p_value, extra_code=None))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn write_regenie2_multi_native_chunk_with_output_dtype<'py>(
    py: Python<'py>,
    writer_sessions: Vec<PyRef<'py, OutputWriterSession>>,
    active_trait_indices: Vec<usize>,
    metadata: PyRef<'py, PyVariantMetadata>,
    chunk_stats: PyRef<'py, PyChunkStats>,
    output_statistic_dtype: String,
    beta: &Bound<'py, PyAny>,
    standard_error: &Bound<'py, PyAny>,
    chi_squared: &Bound<'py, PyAny>,
    log10_p_value: &Bound<'py, PyAny>,
    extra_code: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    let write_plan = g_output::plan_multi_trait_output_write(active_trait_indices.len(), &output_statistic_dtype)
        .map_err(|error| errors::convert_output_error("plan_multi_trait_output_write", error))?;
    if !write_plan.use_native_multi_writer {
        return Ok(());
    }
    if write_plan.uses_float64_native_writer {
        return write_regenie2_multi_native_chunk_f64(
            py,
            writer_sessions,
            active_trait_indices,
            metadata,
            chunk_stats,
            beta.extract()?,
            standard_error.extract()?,
            chi_squared.extract()?,
            log10_p_value.extract()?,
            extract_optional_readonly_array2_i32(extra_code)?,
        );
    }
    write_regenie2_multi_native_chunk(
        py,
        writer_sessions,
        active_trait_indices,
        metadata,
        chunk_stats,
        beta.extract()?,
        standard_error.extract()?,
        chi_squared.extract()?,
        log10_p_value.extract()?,
        extract_optional_readonly_array2_i32(extra_code)?,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn write_regenie2_multi_native_chunk(
    _py: Python<'_>,
    writer_sessions: Vec<PyRef<'_, OutputWriterSession>>,
    active_trait_indices: Vec<usize>,
    metadata: PyRef<'_, PyVariantMetadata>,
    chunk_stats: PyRef<'_, PyChunkStats>,
    beta: PyReadonlyArray2<'_, f32>,
    standard_error: PyReadonlyArray2<'_, f32>,
    chi_squared: PyReadonlyArray2<'_, f32>,
    log10_p_value: PyReadonlyArray2<'_, f32>,
    extra_code: Option<PyReadonlyArray2<'_, i32>>,
) -> PyResult<()> {
    let chunk_handle = build_native_chunk_handle_from_python(&metadata, &chunk_stats)?;
    let trait_count = writer_sessions.len();
    let row_count = chunk_handle.row_count();
    g_output::validate_trait_major_statistic_shape("beta", beta.as_array().shape(), trait_count, row_count)
        .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    g_output::validate_trait_major_statistic_shape(
        "standard_error",
        standard_error.as_array().shape(),
        trait_count,
        row_count,
    )
    .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    g_output::validate_trait_major_statistic_shape(
        "chi_squared",
        chi_squared.as_array().shape(),
        trait_count,
        row_count,
    )
    .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    g_output::validate_trait_major_statistic_shape(
        "log10_p_value",
        log10_p_value.as_array().shape(),
        trait_count,
        row_count,
    )
    .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    if let Some(extra_code_array) = extra_code.as_ref() {
        g_output::validate_trait_major_statistic_shape(
            "extra_code",
            extra_code_array.as_array().shape(),
            trait_count,
            row_count,
        )
        .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    }

    let beta_values = beta.as_array();
    let standard_error_values = standard_error.as_array();
    let chi_squared_values = chi_squared.as_array();
    let log10_p_value_values = log10_p_value.as_array();
    let extra_code_values = extra_code.as_ref().map(PyReadonlyArray2::as_array);
    let mut beta_rows = Vec::with_capacity(active_trait_indices.len());
    let mut standard_error_rows = Vec::with_capacity(active_trait_indices.len());
    let mut chi_squared_rows = Vec::with_capacity(active_trait_indices.len());
    let mut log10_p_value_rows = Vec::with_capacity(active_trait_indices.len());
    let mut extra_code_rows = Vec::with_capacity(active_trait_indices.len());
    for &trait_index in &active_trait_indices {
        if trait_index >= writer_sessions.len() {
            return Err(PyValueError::new_err("Active trait index is out of bounds for writer sessions."));
        }
        beta_rows.push(beta_values.row(trait_index));
        standard_error_rows.push(standard_error_values.row(trait_index));
        chi_squared_rows.push(chi_squared_values.row(trait_index));
        log10_p_value_rows.push(log10_p_value_values.row(trait_index));
        if let Some(extra_code_array) = extra_code_values.as_ref() {
            extra_code_rows.push(extra_code_array.row(trait_index));
        }
    }
    let mut active_statistic_rows = Vec::with_capacity(active_trait_indices.len());
    for active_row_index in 0..active_trait_indices.len() {
        let extra_code_slice = if extra_code_values.is_some() {
            Some(
                extra_code_rows[active_row_index]
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("extra_code row is not contiguous."))?,
            )
        } else {
            None
        };
        active_statistic_rows.push(g_output::Regenie2StatisticSliceBundle {
            beta: beta_rows[active_row_index]
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("beta row is not contiguous."))?,
            standard_error: standard_error_rows[active_row_index]
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("standard_error row is not contiguous."))?,
            chi_squared: chi_squared_rows[active_row_index]
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("chi_squared row is not contiguous."))?,
            log10_p_value: log10_p_value_rows[active_row_index]
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("log10_p_value row is not contiguous."))?,
            extra_code: extra_code_slice,
        });
    }
    let native_writer_sessions = writer_sessions.iter().map(|writer_session| &writer_session.inner).collect::<Vec<_>>();
    g_output::write_regenie2_multi_trait_chunk_f32(
        &native_writer_sessions,
        &active_trait_indices,
        &chunk_handle,
        &active_statistic_rows,
    )
    .map_err(|error| errors::convert_output_error("write_regenie2_multi_native_chunk", error))
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn write_regenie2_multi_native_chunk_f64(
    _py: Python<'_>,
    writer_sessions: Vec<PyRef<'_, OutputWriterSession>>,
    active_trait_indices: Vec<usize>,
    metadata: PyRef<'_, PyVariantMetadata>,
    chunk_stats: PyRef<'_, PyChunkStats>,
    beta: PyReadonlyArray2<'_, f64>,
    standard_error: PyReadonlyArray2<'_, f64>,
    chi_squared: PyReadonlyArray2<'_, f64>,
    log10_p_value: PyReadonlyArray2<'_, f64>,
    extra_code: Option<PyReadonlyArray2<'_, i32>>,
) -> PyResult<()> {
    let chunk_handle = build_native_chunk_handle_from_python(&metadata, &chunk_stats)?;
    let trait_count = writer_sessions.len();
    let row_count = chunk_handle.row_count();
    g_output::validate_trait_major_statistic_shape("beta", beta.as_array().shape(), trait_count, row_count)
        .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    g_output::validate_trait_major_statistic_shape(
        "standard_error",
        standard_error.as_array().shape(),
        trait_count,
        row_count,
    )
    .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    g_output::validate_trait_major_statistic_shape(
        "chi_squared",
        chi_squared.as_array().shape(),
        trait_count,
        row_count,
    )
    .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    g_output::validate_trait_major_statistic_shape(
        "log10_p_value",
        log10_p_value.as_array().shape(),
        trait_count,
        row_count,
    )
    .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    if let Some(extra_code_array) = extra_code.as_ref() {
        g_output::validate_trait_major_statistic_shape(
            "extra_code",
            extra_code_array.as_array().shape(),
            trait_count,
            row_count,
        )
        .map_err(|error| errors::convert_output_error("validate_trait_major_statistic_shape", error))?;
    }

    let beta_values = beta.as_array();
    let standard_error_values = standard_error.as_array();
    let chi_squared_values = chi_squared.as_array();
    let log10_p_value_values = log10_p_value.as_array();
    let extra_code_values = extra_code.as_ref().map(PyReadonlyArray2::as_array);
    let mut beta_rows = Vec::with_capacity(active_trait_indices.len());
    let mut standard_error_rows = Vec::with_capacity(active_trait_indices.len());
    let mut chi_squared_rows = Vec::with_capacity(active_trait_indices.len());
    let mut log10_p_value_rows = Vec::with_capacity(active_trait_indices.len());
    let mut extra_code_rows = Vec::with_capacity(active_trait_indices.len());
    for &trait_index in &active_trait_indices {
        if trait_index >= writer_sessions.len() {
            return Err(PyValueError::new_err("Active trait index is out of bounds for writer sessions."));
        }
        beta_rows.push(beta_values.row(trait_index));
        standard_error_rows.push(standard_error_values.row(trait_index));
        chi_squared_rows.push(chi_squared_values.row(trait_index));
        log10_p_value_rows.push(log10_p_value_values.row(trait_index));
        if let Some(extra_code_array) = extra_code_values.as_ref() {
            extra_code_rows.push(extra_code_array.row(trait_index));
        }
    }
    let mut active_statistic_rows = Vec::with_capacity(active_trait_indices.len());
    for active_row_index in 0..active_trait_indices.len() {
        let extra_code_slice = if extra_code_values.is_some() {
            Some(
                extra_code_rows[active_row_index]
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("extra_code row is not contiguous."))?,
            )
        } else {
            None
        };
        active_statistic_rows.push(g_output::Regenie2StatisticSliceBundle {
            beta: beta_rows[active_row_index]
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("beta row is not contiguous."))?,
            standard_error: standard_error_rows[active_row_index]
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("standard_error row is not contiguous."))?,
            chi_squared: chi_squared_rows[active_row_index]
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("chi_squared row is not contiguous."))?,
            log10_p_value: log10_p_value_rows[active_row_index]
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("log10_p_value row is not contiguous."))?,
            extra_code: extra_code_slice,
        });
    }
    let native_writer_sessions = writer_sessions.iter().map(|writer_session| &writer_session.inner).collect::<Vec<_>>();
    g_output::write_regenie2_multi_trait_chunk_f64(
        &native_writer_sessions,
        &active_trait_indices,
        &chunk_handle,
        &active_statistic_rows,
    )
    .map_err(|error| errors::convert_output_error("write_regenie2_multi_native_chunk_f64", error))
}

pub(crate) fn finish_output_writer_sessions_for_delivery(
    py: Python<'_>,
    writer_sessions: &[PyRef<'_, OutputWriterSession>],
    requested_thread_count: i64,
) -> PyResult<Vec<Option<String>>> {
    let writer_session_count = writer_session_count_as_i64(writer_sessions.len())?;
    record_writer_sessions_finish_started(requested_thread_count, writer_session_count)?;
    let native_writer_sessions = writer_sessions.iter().map(|writer_session| &writer_session.inner).collect::<Vec<_>>();
    py.detach(|| {
        g_output::finish_output_writer_sessions_with_requested_threads(&native_writer_sessions, requested_thread_count)
    })
    .map(optional_path_values_to_strings)
    .map_err(|error| errors::convert_output_error("finish_output_writer_sessions", error))
}

pub(crate) fn finish_interrupted_output_writer_sessions_for_delivery(
    py: Python<'_>,
    writer_sessions: &[PyRef<'_, OutputWriterSession>],
    requested_thread_count: i64,
    signal_exit_code: i64,
    signal_name: &str,
    signal_number: i64,
) -> PyResult<()> {
    let writer_session_count = writer_session_count_as_i64(writer_sessions.len())?;
    record_writer_sessions_interrupted_flush_started(
        requested_thread_count,
        signal_exit_code,
        signal_name,
        signal_number,
        writer_session_count,
    )?;
    let native_writer_sessions = writer_sessions.iter().map(|writer_session| &writer_session.inner).collect::<Vec<_>>();
    py.detach(|| {
        g_output::finish_interrupted_output_writer_sessions_with_requested_threads(
            &native_writer_sessions,
            requested_thread_count,
            signal_name,
        )
    })
    .map_err(|error| errors::convert_output_error("finish_interrupted_output_writer_sessions", error))
}

pub(crate) fn abort_output_writer_sessions_for_delivery(writer_sessions: &[PyRef<'_, OutputWriterSession>]) {
    for writer_session in writer_sessions {
        let _ = writer_session.inner.abort();
    }
}

fn record_writer_sessions_finish_started(requested_thread_count: i64, writer_session_count: i64) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_writer_sessions_finish_started_diagnostic_payload(
        requested_thread_count,
        writer_session_count,
    );
    run_events::emit_run_diagnostic_event_payload(&payload)
}

fn record_writer_sessions_interrupted_flush_started(
    requested_thread_count: i64,
    signal_exit_code: i64,
    signal_name: &str,
    signal_number: i64,
    writer_session_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload(
        requested_thread_count,
        signal_exit_code,
        signal_name,
        signal_number,
        writer_session_count,
    );
    run_events::emit_run_diagnostic_event_payload(&payload)
}

fn record_pipeline_output_writer_sessions_create_started(
    association_mode: &str,
    writer_session_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_output_writer_sessions_create_started_diagnostic_payload(
        association_mode,
        writer_session_count,
    );
    run_events::emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<OutputWriterSession>()?;
    module.add_function(wrap_pyfunction!(write_regenie2_native_chunk_with_output_dtype, module)?)?;
    module.add_function(wrap_pyfunction!(write_regenie2_multi_native_chunk_with_output_dtype, module)?)?;
    Ok(())
}

fn writer_session_count_as_i64(writer_session_count: usize) -> PyResult<i64> {
    i64::try_from(writer_session_count)
        .map_err(|_| PyValueError::new_err("Writer session count exceeds native int64 capacity."))
}

fn optional_path_values_to_strings(paths: Vec<Option<std::path::PathBuf>>) -> Vec<Option<String>> {
    paths.into_iter().map(|maybe_path| maybe_path.map(|path| path.display().to_string())).collect()
}

fn write_regenie2_chunk_arrays_detached(
    py: Python<'_>,
    writer_session: &NativeOutputWriterSession,
    chunk_handle: NativeChunkHandle,
    statistic_arrays: Regenie2StatisticArrays,
    operation: &str,
) -> PyResult<()> {
    py.detach(move || {
        writer_session.write_regenie2_native_chunk_handle_arrays(
            chunk_handle,
            statistic_arrays.beta,
            statistic_arrays.standard_error,
            statistic_arrays.chi_squared,
            statistic_arrays.log10_p_value,
            statistic_arrays.extra_code,
        )
    })
    .map_err(|error| errors::convert_output_error(operation, error))
}

fn build_native_chunk_handle_from_python(
    metadata: &PyVariantMetadata,
    chunk_stats: &PyChunkStats,
) -> PyResult<NativeChunkHandle> {
    let variant_start_index = i64::try_from(metadata.variant_start_index)
        .map_err(|_| PyValueError::new_err("Variant start index does not fit into int64 output."))?;
    let variant_stop_index = i64::try_from(metadata.variant_stop_index)
        .map_err(|_| PyValueError::new_err("Variant stop index does not fit into int64 output."))?;
    let expected_variant_stop_index = variant_start_index
        .checked_add(
            i64::try_from(metadata.metadata.position.len())
                .map_err(|_| PyValueError::new_err("Variant metadata row count does not fit into int64 output."))?,
        )
        .ok_or_else(|| PyValueError::new_err("Variant stop index does not fit into int64 output."))?;
    if variant_stop_index != expected_variant_stop_index {
        return Err(PyValueError::new_err("Variant metadata bounds do not match metadata row count."));
    }
    Ok(NativeChunkHandle::new(
        Arc::new(convert_variant_metadata_to_output(&metadata.metadata)),
        Arc::new(convert_chunk_stats_to_output(&chunk_stats.stats)),
        variant_start_index,
    ))
}

fn convert_variant_metadata_to_output(
    metadata: &g_genotype::VariantMetadataColumns,
) -> NativeOutputVariantMetadataColumns {
    NativeOutputVariantMetadataColumns {
        chromosome: metadata.chromosome.clone(),
        variant_identifier: metadata.variant_identifier.clone(),
        position: metadata.position.clone(),
        allele_one: metadata.allele_one.clone(),
        allele_two: metadata.allele_two.clone(),
    }
}

fn convert_chunk_stats_to_output(chunk_stats: &g_genotype::ChunkStats) -> NativeOutputChunkStats {
    NativeOutputChunkStats {
        allele_one_frequency: chunk_stats.allele_one_frequency.clone(),
        observation_count: chunk_stats.observation_count.clone(),
        has_missing_values: chunk_stats.has_missing_values,
        dosage_sum: Arc::clone(&chunk_stats.dosage_sum),
        dosage_square_sum: chunk_stats.dosage_square_sum.clone(),
        imputed_dosage_square_sum: chunk_stats.imputed_dosage_square_sum.clone(),
        dosage_variance_numerator: chunk_stats.dosage_variance_numerator.clone(),
        info_score: chunk_stats.info_score.clone(),
        allele_count: Arc::clone(&chunk_stats.allele_count),
        minor_allele_count: chunk_stats.minor_allele_count.clone(),
        zero_count: chunk_stats.zero_count.clone(),
        nonzero_count: chunk_stats.nonzero_count.clone(),
        homozygous_reference_count: chunk_stats.homozygous_reference_count.clone(),
        heterozygous_count: chunk_stats.heterozygous_count.clone(),
        homozygous_alternate_count: chunk_stats.homozygous_alternate_count.clone(),
        is_sparse_candidate: chunk_stats.is_sparse_candidate.clone(),
        is_rare_sparse_firth_candidate: chunk_stats.is_rare_sparse_firth_candidate.clone(),
    }
}

fn build_copied_arrow_array<T, ArrowType>(values: &[T]) -> ArrayRef
where
    T: ArrowNativeType,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    Arc::new(PrimitiveArray::<ArrowType>::from_iter_values(values.iter().copied()))
}
