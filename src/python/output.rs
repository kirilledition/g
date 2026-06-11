#![allow(clippy::needless_pass_by_value)]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType, Float32Type, Float64Type, Int32Type};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::output::{
    NativeChunkHandle, OutputResumeMode, OutputWriterError, OutputWriterSession as NativeOutputWriterSession,
    finalize_output_run_chunks as finalize_native_output_run_chunks,
    initialize_output_run as initialize_native_output_run, load_run_manifest_json as load_native_run_manifest_json,
    prepare_output_run as prepare_native_output_run,
    read_run_manifest_committed_chunk_identifiers_from_text as read_native_manifest_committed_chunk_identifiers,
    repair_strict_manifest_chunk_commits as repair_native_strict_manifest_chunk_commits,
    resolve_output_run_paths as resolve_native_output_run_paths,
    scan_committed_chunk_identifiers as scan_native_committed_chunk_identifiers,
    validate_run_manifest_compatibility as validate_native_run_manifest_compatibility,
    validate_strict_manifest_chunks as validate_native_strict_manifest_chunks,
    write_run_manifest_json as write_native_run_manifest_json,
};

use super::{ChunkStats as PyChunkStats, VariantMetadata as PyVariantMetadata};

#[pyclass]
pub(crate) struct OutputWriterSession {
    inner: NativeOutputWriterSession,
}

#[pyclass]
pub(crate) struct NativeOutputRunPaths {
    #[pyo3(get)]
    run_directory: String,
    #[pyo3(get)]
    chunks_directory: String,
}

#[pyclass]
pub(crate) struct NativePreparedOutputRun {
    #[pyo3(get)]
    run_directory: String,
    #[pyo3(get)]
    chunks_directory: String,
    #[pyo3(get)]
    existing_manifest_json: Option<String>,
}

#[pyclass]
pub(crate) struct NativeInitializedOutputRun {
    #[pyo3(get)]
    committed_chunk_identifiers: Vec<i64>,
}

#[pymethods]
impl OutputWriterSession {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        run_directory: String,
        chunks_directory: String,
        association_mode: String,
        writer_thread_count: usize,
        writer_queue_depth: usize,
        output_format: String,
        output_statistic_dtype: String,
        finalize_parquet: bool,
        chunks_per_arrow_file: usize,
        arrow_compression: String,
        parquet_compression: String,
        collect_stage_timings: bool,
    ) -> PyResult<Self> {
        let inner = NativeOutputWriterSession::new(
            run_directory,
            chunks_directory,
            association_mode,
            writer_thread_count,
            writer_queue_depth,
            &output_format,
            &output_statistic_dtype,
            finalize_parquet,
            chunks_per_arrow_file,
            arrow_compression,
            parquet_compression,
            collect_stage_timings,
        )
        .map_err(|error| output_writer_error_to_py(error, "new_output_writer_session"))?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (metadata, chunk_stats, beta, standard_error, chi_squared, log10_p_value, extra_code=None))]
    fn write_regenie2_native_chunk(
        &self,
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
        self.inner
            .write_regenie2_native_chunk_handle_arrays(
                chunk_handle,
                beta_array,
                standard_error_array,
                chi_squared_array,
                log10_p_value_array,
                extra_code_array,
            )
            .map_err(|error| output_writer_error_to_py(error, "write_regenie2_native_chunk"))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (metadata, chunk_stats, beta, standard_error, chi_squared, log10_p_value, extra_code=None))]
    fn write_regenie2_native_chunk_f64(
        &self,
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
        self.inner
            .write_regenie2_native_chunk_handle_arrays(
                chunk_handle,
                beta_array,
                standard_error_array,
                chi_squared_array,
                log10_p_value_array,
                extra_code_array,
            )
            .map_err(|error| output_writer_error_to_py(error, "write_regenie2_native_chunk_f64"))
    }

    fn finish(&self, py: Python<'_>) -> PyResult<Option<String>> {
        py.detach(|| self.inner.finish())
            .map(|maybe_path| maybe_path.map(|path| path.display().to_string()))
            .map_err(|error| output_writer_error_to_py(error, "finish_output_writer"))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn finish_interrupted(&self, py: Python<'_>, signal_name: String) -> PyResult<()> {
        py.detach(|| self.inner.finish_interrupted(&signal_name))
            .map_err(|error| output_writer_error_to_py(error, "finish_interrupted"))
    }

    fn abort(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.abort()).map_err(|error| output_writer_error_to_py(error, "abort_output_writer"))
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    writer_sessions,
    active_trait_indices,
    metadata,
    chunk_stats,
    beta,
    standard_error,
    chi_squared,
    log10_p_value,
    extra_code=None,
))]
pub(crate) fn write_regenie2_multi_native_chunk(
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
    validate_trait_major_shape("beta", beta.as_array().shape(), trait_count, row_count)?;
    validate_trait_major_shape("standard_error", standard_error.as_array().shape(), trait_count, row_count)?;
    validate_trait_major_shape("chi_squared", chi_squared.as_array().shape(), trait_count, row_count)?;
    validate_trait_major_shape("log10_p_value", log10_p_value.as_array().shape(), trait_count, row_count)?;
    if let Some(extra_code_array) = extra_code.as_ref() {
        validate_trait_major_shape("extra_code", extra_code_array.as_array().shape(), trait_count, row_count)?;
    }

    let beta_values = beta.as_array();
    let standard_error_values = standard_error.as_array();
    let chi_squared_values = chi_squared.as_array();
    let log10_p_value_values = log10_p_value.as_array();
    let extra_code_values = extra_code.as_ref().map(PyReadonlyArray2::as_array);
    for trait_index in active_trait_indices {
        if trait_index >= writer_sessions.len() {
            return Err(PyValueError::new_err("Active trait index is out of bounds for writer sessions."));
        }
        let beta_row = beta_values.row(trait_index);
        let standard_error_row = standard_error_values.row(trait_index);
        let chi_squared_row = chi_squared_values.row(trait_index);
        let log10_p_value_row = log10_p_value_values.row(trait_index);
        let beta_slice = beta_row.as_slice().ok_or_else(|| PyValueError::new_err("beta row is not contiguous."))?;
        let standard_error_slice = standard_error_row
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("standard_error row is not contiguous."))?;
        let chi_squared_slice =
            chi_squared_row.as_slice().ok_or_else(|| PyValueError::new_err("chi_squared row is not contiguous."))?;
        let log10_p_value_slice = log10_p_value_row
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("log10_p_value row is not contiguous."))?;
        let extra_code_row = extra_code_values.as_ref().map(|extra_code_array| extra_code_array.row(trait_index));
        let extra_code_slice = match extra_code_row.as_ref() {
            None => None,
            Some(extra_code_array_row) => Some(
                extra_code_array_row
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("extra_code row is not contiguous."))?,
            ),
        };
        let beta_array = build_copied_arrow_array::<f32, Float32Type>(beta_slice);
        let standard_error_array = build_copied_arrow_array::<f32, Float32Type>(standard_error_slice);
        let chi_squared_array = build_copied_arrow_array::<f32, Float32Type>(chi_squared_slice);
        let log10_p_value_array = build_copied_arrow_array::<f32, Float32Type>(log10_p_value_slice);
        let extra_code_array = extra_code_slice.map(build_copied_arrow_array::<i32, Int32Type>);
        writer_sessions[trait_index]
            .inner
            .write_regenie2_native_chunk_handle_arrays(
                chunk_handle.clone(),
                beta_array,
                standard_error_array,
                chi_squared_array,
                log10_p_value_array,
                extra_code_array,
            )
            .map_err(|error| output_writer_error_to_py(error, "write_regenie2_multi_native_chunk"))?;
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    writer_sessions,
    active_trait_indices,
    metadata,
    chunk_stats,
    beta,
    standard_error,
    chi_squared,
    log10_p_value,
    extra_code=None,
))]
pub(crate) fn write_regenie2_multi_native_chunk_f64(
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
    validate_trait_major_shape("beta", beta.as_array().shape(), trait_count, row_count)?;
    validate_trait_major_shape("standard_error", standard_error.as_array().shape(), trait_count, row_count)?;
    validate_trait_major_shape("chi_squared", chi_squared.as_array().shape(), trait_count, row_count)?;
    validate_trait_major_shape("log10_p_value", log10_p_value.as_array().shape(), trait_count, row_count)?;
    if let Some(extra_code_array) = extra_code.as_ref() {
        validate_trait_major_shape("extra_code", extra_code_array.as_array().shape(), trait_count, row_count)?;
    }

    let beta_values = beta.as_array();
    let standard_error_values = standard_error.as_array();
    let chi_squared_values = chi_squared.as_array();
    let log10_p_value_values = log10_p_value.as_array();
    let extra_code_values = extra_code.as_ref().map(PyReadonlyArray2::as_array);
    for trait_index in active_trait_indices {
        if trait_index >= writer_sessions.len() {
            return Err(PyValueError::new_err("Active trait index is out of bounds for writer sessions."));
        }
        let beta_row = beta_values.row(trait_index);
        let standard_error_row = standard_error_values.row(trait_index);
        let chi_squared_row = chi_squared_values.row(trait_index);
        let log10_p_value_row = log10_p_value_values.row(trait_index);
        let beta_slice = beta_row.as_slice().ok_or_else(|| PyValueError::new_err("beta row is not contiguous."))?;
        let standard_error_slice = standard_error_row
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("standard_error row is not contiguous."))?;
        let chi_squared_slice =
            chi_squared_row.as_slice().ok_or_else(|| PyValueError::new_err("chi_squared row is not contiguous."))?;
        let log10_p_value_slice = log10_p_value_row
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("log10_p_value row is not contiguous."))?;
        let extra_code_row = extra_code_values.as_ref().map(|extra_code_array| extra_code_array.row(trait_index));
        let extra_code_slice = match extra_code_row.as_ref() {
            None => None,
            Some(extra_code_array_row) => Some(
                extra_code_array_row
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("extra_code row is not contiguous."))?,
            ),
        };
        let beta_array = build_copied_arrow_array::<f64, Float64Type>(beta_slice);
        let standard_error_array = build_copied_arrow_array::<f64, Float64Type>(standard_error_slice);
        let chi_squared_array = build_copied_arrow_array::<f64, Float64Type>(chi_squared_slice);
        let log10_p_value_array = build_copied_arrow_array::<f64, Float64Type>(log10_p_value_slice);
        let extra_code_array = extra_code_slice.map(build_copied_arrow_array::<i32, Int32Type>);
        writer_sessions[trait_index]
            .inner
            .write_regenie2_native_chunk_handle_arrays(
                chunk_handle.clone(),
                beta_array,
                standard_error_array,
                chi_squared_array,
                log10_p_value_array,
                extra_code_array,
            )
            .map_err(|error| output_writer_error_to_py(error, "write_regenie2_multi_native_chunk_f64"))?;
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn finalize_output_run_chunks(
    run_directory: String,
    chunks_directory: String,
    association_mode: String,
    output_format: String,
) -> PyResult<String> {
    let native_output_format = crate::output::OutputFileFormat::parse(&output_format).map_err(PyValueError::new_err)?;
    finalize_native_output_run_chunks(
        Path::new(&run_directory),
        Path::new(&chunks_directory),
        &association_mode,
        native_output_format,
    )
    .map(|path| path.display().to_string())
    .map_err(|error| output_writer_error_to_py(error, "finalize_output_run_chunks"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn resolve_output_run_paths(
    output_root: String,
    association_mode: String,
    output_format: String,
) -> PyResult<NativeOutputRunPaths> {
    let native_output_format = crate::output::OutputFileFormat::parse(&output_format).map_err(PyValueError::new_err)?;
    let output_run_paths =
        resolve_native_output_run_paths(Path::new(&output_root), &association_mode, native_output_format);
    Ok(NativeOutputRunPaths {
        run_directory: output_run_paths.run_directory.display().to_string(),
        chunks_directory: output_run_paths.chunks_directory.display().to_string(),
    })
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn prepare_output_run(
    output_root: String,
    association_mode: String,
    output_format: String,
    resume: bool,
) -> PyResult<NativePreparedOutputRun> {
    let native_output_format = crate::output::OutputFileFormat::parse(&output_format).map_err(PyValueError::new_err)?;
    let prepared_output_run =
        prepare_native_output_run(Path::new(&output_root), &association_mode, native_output_format, resume)
            .map_err(|error| output_writer_error_to_py(error, "prepare_output_run"))?;
    Ok(NativePreparedOutputRun {
        run_directory: prepared_output_run.output_run_paths.run_directory.display().to_string(),
        chunks_directory: prepared_output_run.output_run_paths.chunks_directory.display().to_string(),
        existing_manifest_json: prepared_output_run.existing_manifest_json,
    })
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn load_run_manifest_json(run_directory: String) -> PyResult<Option<String>> {
    load_native_run_manifest_json(Path::new(&run_directory))
        .map_err(|error| output_writer_error_to_py(error, "load_run_manifest_json"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn write_run_manifest_json(run_directory: String, manifest_json: String) -> PyResult<()> {
    write_native_run_manifest_json(Path::new(&run_directory), &manifest_json)
        .map_err(|error| output_writer_error_to_py(error, "write_run_manifest_json"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn validate_run_manifest_compatibility(manifest_json: String, current_header_json: String) -> PyResult<()> {
    validate_native_run_manifest_compatibility(&manifest_json, &current_header_json)
        .map_err(|error| output_writer_error_to_py(error, "validate_run_manifest_compatibility"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn read_manifest_committed_chunk_identifiers(manifest_json: String) -> PyResult<Vec<i64>> {
    read_native_manifest_committed_chunk_identifiers(&manifest_json)
        .map_err(|error| output_writer_error_to_py(error, "read_manifest_committed_chunk_identifiers"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn initialize_output_run(
    run_directory: String,
    chunks_directory: String,
    existing_manifest_json: Option<String>,
    current_header_json: String,
    resume: bool,
    resume_mode: String,
) -> PyResult<NativeInitializedOutputRun> {
    let native_resume_mode = OutputResumeMode::parse(&resume_mode)
        .map_err(|error| output_writer_error_to_py(error, "parse_output_resume_mode"))?;
    let initialized_output_run = initialize_native_output_run(
        Path::new(&run_directory),
        Path::new(&chunks_directory),
        existing_manifest_json.as_deref(),
        &current_header_json,
        resume,
        native_resume_mode,
    )
    .map_err(|error| output_writer_error_to_py(error, "initialize_output_run"))?;
    Ok(NativeInitializedOutputRun { committed_chunk_identifiers: initialized_output_run.committed_chunk_identifiers })
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn scan_committed_chunk_identifiers(chunks_directory: String) -> PyResult<Vec<i64>> {
    scan_native_committed_chunk_identifiers(Path::new(&chunks_directory))
        .map_err(|error| output_writer_error_to_py(error, "scan_committed_chunk_identifiers"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn validate_strict_manifest_chunks(chunks_directory: String, manifest_json: String) -> PyResult<Vec<i64>> {
    validate_native_strict_manifest_chunks(Path::new(&chunks_directory), &manifest_json)
        .map_err(|error| output_writer_error_to_py(error, "validate_strict_manifest_chunks"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn repair_strict_manifest_chunk_commits(
    chunks_directory: String,
    manifest_json: String,
) -> PyResult<String> {
    let chunk_commits = repair_native_strict_manifest_chunk_commits(Path::new(&chunks_directory), &manifest_json)
        .map_err(|error| output_writer_error_to_py(error, "repair_strict_manifest_chunk_commits"))?;
    serde_json::to_string(
        &chunk_commits
            .into_iter()
            .map(|chunk_commit| {
                serde_json::json!({
                    "chunk_identifier": chunk_commit.chunk_identifier,
                    "output_format": chunk_commit.output_format,
                    "compression": chunk_commit.compression,
                    "variant_start_index": chunk_commit.variant_start_index,
                    "variant_stop_index": chunk_commit.variant_stop_index,
                    "row_count": chunk_commit.row_count,
                    "chunk_file_name": chunk_commit.chunk_file_name,
                })
            })
            .collect::<Vec<_>>(),
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn output_writer_error_to_py(error: OutputWriterError, operation: &str) -> PyErr {
    let (error_kind, message) = match &error {
        OutputWriterError::InvalidInput(message) => ("invalid_input", message.clone()),
        OutputWriterError::Runtime(message) => ("runtime", message.clone()),
    };
    tracing::warn!(
        target: "g.python.output",
        g_event = "native_output_writer_error",
        operation = operation,
        error_kind = error_kind,
        error_message = %message,
        "Native output writer conversion error."
    );
    match error {
        OutputWriterError::InvalidInput(message) => PyValueError::new_err(message),
        OutputWriterError::Runtime(message) => PyRuntimeError::new_err(message),
    }
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
    Ok(NativeChunkHandle::new(Arc::clone(&metadata.metadata), Arc::clone(&chunk_stats.stats), variant_start_index))
}

fn validate_trait_major_shape(
    array_name: &str,
    observed_shape: &[usize],
    trait_count: usize,
    row_count: usize,
) -> PyResult<()> {
    if observed_shape == [trait_count, row_count] {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{array_name} must have shape ({trait_count}, {row_count}) for multi-trait output."
    )))
}

fn build_copied_arrow_array<T, ArrowType>(values: &[T]) -> ArrayRef
where
    T: ArrowNativeType,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    Arc::new(PrimitiveArray::<ArrowType>::from_iter_values(values.iter().copied()))
}

#[cfg(test)]
mod tests {
    use arrow::array::Array;

    use super::{Float32Type, build_copied_arrow_array};

    #[test]
    fn copied_arrow_array_is_independent_from_source_slice() {
        let mut values = [1.25_f32, 2.5, 3.75];
        let array = build_copied_arrow_array::<f32, Float32Type>(values.as_slice());
        values.fill(99.0);

        assert_eq!(array.len(), 3);
        let typed_array = array
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .expect("array should preserve the f32 Arrow type");
        let observed_values =
            (0..typed_array.len()).map(|value_index| typed_array.value(value_index)).collect::<Vec<_>>();
        assert_eq!(observed_values, vec![1.25, 2.5, 3.75]);
    }

    #[test]
    fn copied_arrow_array_allows_empty_arrays() {
        let values: [f32; 0] = [];
        let array = build_copied_arrow_array::<f32, Float32Type>(values.as_slice());

        assert_eq!(array.len(), 0);
    }
}
