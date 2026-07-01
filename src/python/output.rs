#![allow(clippy::needless_pass_by_value)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType, Float32Type, Float64Type, Int32Type};
use g_input::regenie::{PredictionError, resolve_prediction_loco_paths as resolve_native_prediction_loco_paths};
use g_output::{
    CurrentRunManifestHeaderInput, ManifestFileFingerprint as NativeManifestFileFingerprint,
    ManifestFileFingerprintCache as NativeManifestFileFingerprintCacheState, NativeChunkHandle,
    NativeChunkStats as NativeOutputChunkStats, OutputFileFormat, OutputResumeMode, OutputWriterError,
    OutputWriterSession as NativeOutputWriterSession, VariantMetadataColumns as NativeOutputVariantMetadataColumns,
    build_current_run_manifest_header_json as build_native_current_run_manifest_header_json,
    build_current_run_manifest_header_json_with_cache as build_native_current_run_manifest_header_json_with_cache,
    build_file_content_sha256 as build_native_file_content_sha256,
    build_manifest_file_fingerprint as build_native_manifest_file_fingerprint,
    build_manifest_json_sha256 as build_native_manifest_json_sha256,
    build_prepared_run_manifest_header_json as build_native_prepared_run_manifest_header_json,
    build_prepared_run_manifest_header_json_from_current_header_json as build_native_prepared_run_manifest_header_json_from_current_header_json,
    build_prepared_run_plan_json_from_current_header_json as build_native_prepared_run_plan_json_from_current_header_json,
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
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde::de::DeserializeOwned;

use super::{
    errors::convert_prediction_error,
    genotype::{ChunkStats as PyChunkStats, VariantMetadata as PyVariantMetadata},
    runtime_state::NativeRuntimeCompatibilityToken,
};

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

#[pyclass]
pub(crate) struct NativeManifestFileFingerprintCache {
    inner: Mutex<NativeManifestFileFingerprintCacheState>,
}

struct Regenie2StatisticArrays {
    beta: ArrayRef,
    standard_error: ArrayRef,
    chi_squared: ArrayRef,
    log10_p_value: ArrayRef,
    extra_code: Option<ArrayRef>,
}

enum PredictionLocoFingerprintBuildError {
    Prediction(PredictionError),
    Output(OutputWriterError),
}

#[pymethods]
impl NativeManifestFileFingerprintCache {
    #[new]
    fn new() -> Self {
        Self { inner: Mutex::new(NativeManifestFileFingerprintCacheState::new()) }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn build_file_fingerprint_payload<'py>(
        &self,
        py: Python<'py>,
        path: String,
        include_content_hash: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let file_fingerprint = py
            .detach(|| {
                let mut fingerprint_cache = self.inner.lock().map_err(|_| {
                    OutputWriterError::Runtime("Manifest file fingerprint cache mutex was poisoned.".to_string())
                })?;
                fingerprint_cache.build_file_fingerprint(Path::new(&path), include_content_hash)
            })
            .map_err(|error| output_writer_error_to_py(error, "build_cached_manifest_file_fingerprint_payload"))?;
        manifest_file_fingerprint_to_dict(py, &file_fingerprint)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn build_current_run_manifest_header_json_from_input_json(
        &self,
        py: Python<'_>,
        current_header_input_json: String,
    ) -> PyResult<String> {
        let current_header_input = parse_json_argument::<CurrentRunManifestHeaderInput>(
            "current_header_input_json",
            &current_header_input_json,
        )?;
        py.detach(|| {
            let mut fingerprint_cache = self.inner.lock().map_err(|_| {
                OutputWriterError::Runtime("Manifest file fingerprint cache mutex was poisoned.".to_string())
            })?;
            build_native_current_run_manifest_header_json_with_cache(current_header_input, &mut fingerprint_cache)
        })
        .map_err(|error| output_writer_error_to_py(error, "build_cached_current_run_manifest_header_json"))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn build_prediction_loco_file_fingerprints_json(
        &self,
        py: Python<'_>,
        prediction_list_path: String,
        phenotype_names: Vec<String>,
    ) -> PyResult<String> {
        py.detach(|| {
            let mut fingerprint_cache = self.inner.lock().map_err(|_| {
                PredictionLocoFingerprintBuildError::Output(OutputWriterError::Runtime(
                    "Manifest file fingerprint cache mutex was poisoned.".to_string(),
                ))
            })?;
            build_prediction_loco_file_fingerprints_json_with_cache(
                &prediction_list_path,
                &phenotype_names,
                &mut fingerprint_cache,
            )
        })
        .map_err(prediction_loco_fingerprint_build_error_to_py)
    }
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
pub(crate) fn finish_output_writer_session(
    py: Python<'_>,
    writer_session: PyRef<'_, OutputWriterSession>,
) -> PyResult<Option<String>> {
    let native_writer_session = &writer_session.inner;
    py.detach(|| native_writer_session.finish())
        .map(|maybe_path| maybe_path.map(|path| path.display().to_string()))
        .map_err(|error| output_writer_error_to_py(error, "finish_output_writer"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn finish_output_writer_session_interrupted(
    py: Python<'_>,
    writer_session: PyRef<'_, OutputWriterSession>,
    signal_name: String,
) -> PyResult<()> {
    let native_writer_session = &writer_session.inner;
    py.detach(|| native_writer_session.finish_interrupted(&signal_name))
        .map_err(|error| output_writer_error_to_py(error, "finish_interrupted"))
}

#[pyfunction]
pub(crate) fn abort_output_writer_session(
    py: Python<'_>,
    writer_session: PyRef<'_, OutputWriterSession>,
) -> PyResult<()> {
    let native_writer_session = &writer_session.inner;
    py.detach(|| native_writer_session.abort()).map_err(|error| output_writer_error_to_py(error, "abort_output_writer"))
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
    py: Python<'_>,
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
        let statistic_arrays = Regenie2StatisticArrays {
            beta: beta_array,
            standard_error: standard_error_array,
            chi_squared: chi_squared_array,
            log10_p_value: log10_p_value_array,
            extra_code: extra_code_array,
        };
        write_regenie2_chunk_arrays_detached(
            py,
            &writer_sessions[trait_index].inner,
            chunk_handle.clone(),
            statistic_arrays,
            "write_regenie2_multi_native_chunk",
        )?;
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
    py: Python<'_>,
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
        let statistic_arrays = Regenie2StatisticArrays {
            beta: beta_array,
            standard_error: standard_error_array,
            chi_squared: chi_squared_array,
            log10_p_value: log10_p_value_array,
            extra_code: extra_code_array,
        };
        write_regenie2_chunk_arrays_detached(
            py,
            &writer_sessions[trait_index].inner,
            chunk_handle.clone(),
            statistic_arrays,
            "write_regenie2_multi_native_chunk_f64",
        )?;
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn finalize_output_run_chunks(
    py: Python<'_>,
    run_directory: String,
    chunks_directory: String,
    association_mode: String,
    output_format: String,
) -> PyResult<String> {
    let native_output_format = OutputFileFormat::parse(&output_format).map_err(PyValueError::new_err)?;
    py.detach(|| {
        finalize_native_output_run_chunks(
            Path::new(&run_directory),
            Path::new(&chunks_directory),
            &association_mode,
            native_output_format,
        )
    })
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
    let native_output_format = OutputFileFormat::parse(&output_format).map_err(PyValueError::new_err)?;
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
    py: Python<'_>,
    output_root: String,
    association_mode: String,
    output_format: String,
    resume: bool,
    runtime_compatibility_token: PyRef<'_, NativeRuntimeCompatibilityToken>,
) -> PyResult<NativePreparedOutputRun> {
    let _runtime_compatibility_token = runtime_compatibility_token.native_token();
    let native_output_format = OutputFileFormat::parse(&output_format).map_err(PyValueError::new_err)?;
    let prepared_output_run = py
        .detach(|| prepare_native_output_run(Path::new(&output_root), &association_mode, native_output_format, resume))
        .map_err(|error| output_writer_error_to_py(error, "prepare_output_run"))?;
    Ok(NativePreparedOutputRun {
        run_directory: prepared_output_run.output_run_paths.run_directory.display().to_string(),
        chunks_directory: prepared_output_run.output_run_paths.chunks_directory.display().to_string(),
        existing_manifest_json: prepared_output_run.existing_manifest_json,
    })
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn load_run_manifest_json(py: Python<'_>, run_directory: String) -> PyResult<Option<String>> {
    py.detach(|| load_native_run_manifest_json(Path::new(&run_directory)))
        .map_err(|error| output_writer_error_to_py(error, "load_run_manifest_json"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn write_run_manifest_json(py: Python<'_>, run_directory: String, manifest_json: String) -> PyResult<()> {
    py.detach(|| write_native_run_manifest_json(Path::new(&run_directory), &manifest_json))
        .map_err(|error| output_writer_error_to_py(error, "write_run_manifest_json"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_current_run_manifest_header_json_from_input_json(
    py: Python<'_>,
    current_header_input_json: String,
) -> PyResult<String> {
    let current_header_input =
        parse_json_argument::<CurrentRunManifestHeaderInput>("current_header_input_json", &current_header_input_json)?;
    py.detach(|| build_native_current_run_manifest_header_json(current_header_input))
        .map_err(|error| output_writer_error_to_py(error, "build_current_run_manifest_header_json_from_input_json"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_prepared_run_manifest_header_json(prepared_run_plan_json: String) -> PyResult<String> {
    let prepared_run_plan = serde_json::from_str::<g_plan::PreparedRunPlan>(&prepared_run_plan_json)
        .map_err(|error| PyValueError::new_err(format!("Invalid prepared run plan JSON: {error}")))?;
    build_native_prepared_run_manifest_header_json(&prepared_run_plan)
        .map_err(|error| output_writer_error_to_py(error, "build_prepared_run_manifest_header_json"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_prepared_run_manifest_header_json_from_current_header_json(
    current_header_json: String,
) -> PyResult<String> {
    build_native_prepared_run_manifest_header_json_from_current_header_json(&current_header_json).map_err(|error| {
        output_writer_error_to_py(error, "build_prepared_run_manifest_header_json_from_current_header_json")
    })
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_prepared_run_plan_json(prepared_run_plan_input_json: String) -> PyResult<String> {
    let prepared_run_plan_input = parse_json_argument::<g_plan::PreparedRunPlanInput>(
        "prepared_run_plan_input_json",
        &prepared_run_plan_input_json,
    )?;
    let prepared_run_plan = g_plan::build_prepared_run_plan(prepared_run_plan_input)
        .map_err(|error| PyValueError::new_err(format!("Invalid prepared run plan input: {error}")))?;
    serde_json::to_string(&prepared_run_plan)
        .map_err(|error| PyRuntimeError::new_err(format!("Could not serialize prepared run plan JSON: {error}")))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_prepared_run_plan_json_from_current_header_json(current_header_json: String) -> PyResult<String> {
    build_native_prepared_run_plan_json_from_current_header_json(&current_header_json)
        .map_err(|error| output_writer_error_to_py(error, "build_prepared_run_plan_json_from_current_header_json"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_prediction_loco_file_fingerprints_json(
    py: Python<'_>,
    prediction_list_path: String,
    phenotype_names: Vec<String>,
) -> PyResult<String> {
    py.detach(|| build_prediction_loco_file_fingerprints_json_detached(&prediction_list_path, &phenotype_names))
        .map_err(prediction_loco_fingerprint_build_error_to_py)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_file_content_sha256_value(py: Python<'_>, path: String) -> PyResult<String> {
    py.detach(|| build_native_file_content_sha256(Path::new(&path)))
        .map_err(|error| output_writer_error_to_py(error, "build_file_content_sha256_value"))
}

fn parse_json_argument<T>(argument_name: &str, argument_json: &str) -> PyResult<T>
where
    T: DeserializeOwned,
{
    serde_json::from_str(argument_json)
        .map_err(|error| PyValueError::new_err(format!("Invalid {argument_name}: {error}")))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_manifest_file_fingerprint_payload<'py>(
    py: Python<'py>,
    path: String,
    include_content_hash: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let file_fingerprint = py
        .detach(|| build_native_manifest_file_fingerprint(Path::new(&path), include_content_hash))
        .map_err(|error| output_writer_error_to_py(error, "build_manifest_file_fingerprint_payload"))?;
    manifest_file_fingerprint_to_dict(py, &file_fingerprint)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_manifest_file_fingerprint_mapping_payload<'py>(
    py: Python<'py>,
    path: String,
    size: u64,
    mtime_ns: i64,
    content_hash_algorithm: String,
    content_sha256: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    manifest_file_fingerprint_to_dict(
        py,
        &NativeManifestFileFingerprint { path, size, mtime_ns, content_hash_algorithm, content_sha256 },
    )
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_manifest_json_sha256(manifest_json: String) -> String {
    build_native_manifest_json_sha256(&manifest_json)
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
    py: Python<'_>,
    run_directory: String,
    chunks_directory: String,
    existing_manifest_json: Option<String>,
    current_header_json: String,
    resume: bool,
    resume_mode: String,
    runtime_compatibility_token: PyRef<'_, NativeRuntimeCompatibilityToken>,
) -> PyResult<NativeInitializedOutputRun> {
    let _runtime_compatibility_token = runtime_compatibility_token.native_token();
    let native_resume_mode = OutputResumeMode::parse(&resume_mode)
        .map_err(|error| output_writer_error_to_py(error, "parse_output_resume_mode"))?;
    let initialized_output_run = py
        .detach(|| {
            initialize_native_output_run(
                Path::new(&run_directory),
                Path::new(&chunks_directory),
                existing_manifest_json.as_deref(),
                &current_header_json,
                resume,
                native_resume_mode,
            )
        })
        .map_err(|error| output_writer_error_to_py(error, "initialize_output_run"))?;
    Ok(NativeInitializedOutputRun { committed_chunk_identifiers: initialized_output_run.committed_chunk_identifiers })
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn scan_committed_chunk_identifiers(py: Python<'_>, chunks_directory: String) -> PyResult<Vec<i64>> {
    py.detach(|| scan_native_committed_chunk_identifiers(Path::new(&chunks_directory)))
        .map_err(|error| output_writer_error_to_py(error, "scan_committed_chunk_identifiers"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn validate_strict_manifest_chunks(
    py: Python<'_>,
    chunks_directory: String,
    manifest_json: String,
) -> PyResult<Vec<i64>> {
    py.detach(|| validate_native_strict_manifest_chunks(Path::new(&chunks_directory), &manifest_json))
        .map_err(|error| output_writer_error_to_py(error, "validate_strict_manifest_chunks"))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn repair_strict_manifest_chunk_commits(
    py: Python<'_>,
    chunks_directory: String,
    manifest_json: String,
) -> PyResult<String> {
    let chunk_commits = py
        .detach(|| repair_native_strict_manifest_chunk_commits(Path::new(&chunks_directory), &manifest_json))
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

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeInitializedOutputRun>()?;
    module.add_class::<NativeManifestFileFingerprintCache>()?;
    module.add_class::<NativeOutputRunPaths>()?;
    module.add_class::<NativePreparedOutputRun>()?;
    module.add_class::<OutputWriterSession>()?;
    module.add_function(wrap_pyfunction!(abort_output_writer_session, module)?)?;
    module.add_function(wrap_pyfunction!(build_current_run_manifest_header_json_from_input_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_file_content_sha256_value, module)?)?;
    module.add_function(wrap_pyfunction!(build_manifest_file_fingerprint_mapping_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_manifest_file_fingerprint_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_manifest_json_sha256, module)?)?;
    module.add_function(wrap_pyfunction!(build_prepared_run_manifest_header_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_prepared_run_manifest_header_json_from_current_header_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_prepared_run_plan_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_prepared_run_plan_json_from_current_header_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_prediction_loco_file_fingerprints_json, module)?)?;
    module.add_function(wrap_pyfunction!(finalize_output_run_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(finish_output_writer_session, module)?)?;
    module.add_function(wrap_pyfunction!(finish_output_writer_session_interrupted, module)?)?;
    module.add_function(wrap_pyfunction!(initialize_output_run, module)?)?;
    module.add_function(wrap_pyfunction!(load_run_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_output_run, module)?)?;
    module.add_function(wrap_pyfunction!(read_manifest_committed_chunk_identifiers, module)?)?;
    module.add_function(wrap_pyfunction!(repair_strict_manifest_chunk_commits, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_output_run_paths, module)?)?;
    module.add_function(wrap_pyfunction!(scan_committed_chunk_identifiers, module)?)?;
    module.add_function(wrap_pyfunction!(validate_run_manifest_compatibility, module)?)?;
    module.add_function(wrap_pyfunction!(validate_strict_manifest_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(write_regenie2_multi_native_chunk, module)?)?;
    module.add_function(wrap_pyfunction!(write_regenie2_multi_native_chunk_f64, module)?)?;
    module.add_function(wrap_pyfunction!(write_run_manifest_json, module)?)?;
    Ok(())
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
    .map_err(|error| output_writer_error_to_py(error, operation))
}

fn manifest_file_fingerprint_to_dict<'py>(
    py: Python<'py>,
    file_fingerprint: &NativeManifestFileFingerprint,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("path", &file_fingerprint.path)?;
    payload.set_item("size", file_fingerprint.size)?;
    payload.set_item("mtime_ns", file_fingerprint.mtime_ns)?;
    payload.set_item("content_hash_algorithm", &file_fingerprint.content_hash_algorithm)?;
    payload.set_item("content_sha256", &file_fingerprint.content_sha256)?;
    Ok(payload)
}

fn build_prediction_loco_file_fingerprints_json_detached(
    prediction_list_path: &str,
    phenotype_names: &[String],
) -> Result<String, PredictionLocoFingerprintBuildError> {
    let mut fingerprint_cache = NativeManifestFileFingerprintCacheState::new();
    build_prediction_loco_file_fingerprints_json_with_cache(
        prediction_list_path,
        phenotype_names,
        &mut fingerprint_cache,
    )
}

fn build_prediction_loco_file_fingerprints_json_with_cache(
    prediction_list_path: &str,
    phenotype_names: &[String],
    fingerprint_cache: &mut NativeManifestFileFingerprintCacheState,
) -> Result<String, PredictionLocoFingerprintBuildError> {
    let resolved_loco_paths = resolve_native_prediction_loco_paths(Path::new(prediction_list_path), phenotype_names)
        .map_err(PredictionLocoFingerprintBuildError::Prediction)?;
    let mut loco_file_payloads = Vec::with_capacity(resolved_loco_paths.len());
    for resolved_loco_path in resolved_loco_paths {
        let file_fingerprint = fingerprint_cache
            .build_file_fingerprint(&resolved_loco_path.loco_file_path, true)
            .map_err(PredictionLocoFingerprintBuildError::Output)?;
        loco_file_payloads.push(serde_json::json!({
            "phenotype": resolved_loco_path.phenotype_name,
            "path": file_fingerprint.path,
            "size": file_fingerprint.size,
            "mtime_ns": file_fingerprint.mtime_ns,
            "content_hash_algorithm": file_fingerprint.content_hash_algorithm,
            "content_sha256": file_fingerprint.content_sha256,
        }));
    }
    serde_json::to_string(&loco_file_payloads)
        .map_err(|error| PredictionLocoFingerprintBuildError::Output(OutputWriterError::Runtime(error.to_string())))
}

fn prediction_loco_fingerprint_build_error_to_py(error: PredictionLocoFingerprintBuildError) -> PyErr {
    match error {
        PredictionLocoFingerprintBuildError::Prediction(prediction_error) => {
            convert_prediction_error("build_prediction_loco_file_fingerprints_json", &prediction_error)
        }
        PredictionLocoFingerprintBuildError::Output(output_error) => {
            output_writer_error_to_py(output_error, "build_prediction_loco_file_fingerprints_json")
        }
    }
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
    Ok(NativeChunkHandle::new(
        Arc::new(convert_variant_metadata_to_output(&metadata.metadata)),
        Arc::new(convert_chunk_stats_to_output(&chunk_stats.stats)),
        variant_start_index,
    ))
}

fn convert_variant_metadata_to_output(
    metadata: &g_genotype::common::VariantMetadataColumns,
) -> NativeOutputVariantMetadataColumns {
    NativeOutputVariantMetadataColumns {
        chromosome: metadata.chromosome.clone(),
        variant_identifier: metadata.variant_identifier.clone(),
        position: metadata.position.clone(),
        allele_one: metadata.allele_one.clone(),
        allele_two: metadata.allele_two.clone(),
    }
}

fn convert_chunk_stats_to_output(chunk_stats: &g_genotype::common::ChunkStats) -> NativeOutputChunkStats {
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
