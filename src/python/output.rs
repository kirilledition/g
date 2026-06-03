#![allow(clippy::needless_pass_by_value)]

use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType, Float32Type, Int32Type};
use numpy::{Element, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::output::{
    NativeChunkHandle, OutputWriterError, OutputWriterSession as NativeOutputWriterSession,
    finalize_output_run_chunks as finalize_native_output_run_chunks,
    repair_strict_manifest_chunk_commits as repair_native_strict_manifest_chunk_commits,
    scan_committed_chunk_identifiers as scan_native_committed_chunk_identifiers,
    validate_strict_manifest_chunks as validate_native_strict_manifest_chunks,
};

use super::{ChunkStats as PyChunkStats, VariantMetadata as PyVariantMetadata};

struct PythonArrayAllocation {
    _owner: Py<PyAny>,
}

impl std::panic::RefUnwindSafe for PythonArrayAllocation {}

#[pyclass]
pub(crate) struct OutputWriterSession {
    inner: NativeOutputWriterSession,
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
        finalize_parquet: bool,
        chunks_per_arrow_file: usize,
        arrow_compression: String,
        collect_stage_timings: bool,
    ) -> PyResult<Self> {
        let inner = NativeOutputWriterSession::new(
            run_directory,
            chunks_directory,
            association_mode,
            writer_thread_count,
            writer_queue_depth,
            finalize_parquet,
            chunks_per_arrow_file,
            arrow_compression,
            collect_stage_timings,
        )
        .map_err(output_writer_error_to_py)?;
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
        let beta_array = build_python_owned_arrow_array1::<f32, Float32Type>(&beta, beta_slice)?;
        let standard_error_array =
            build_python_owned_arrow_array1::<f32, Float32Type>(&standard_error, standard_error_slice)?;
        let chi_squared_array = build_python_owned_arrow_array1::<f32, Float32Type>(&chi_squared, chi_squared_slice)?;
        let log10_p_value_array =
            build_python_owned_arrow_array1::<f32, Float32Type>(&log10_p_value, log10_p_value_slice)?;
        let extra_code_array = match (extra_code.as_ref(), extra_code_slice) {
            (None, None) => None,
            (Some(extra_code_values), Some(extra_code_slice_values)) => {
                Some(build_python_owned_arrow_array1::<i32, Int32Type>(extra_code_values, extra_code_slice_values)?)
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
            .map_err(output_writer_error_to_py)
    }

    fn finish(&self) -> PyResult<Option<String>> {
        self.inner
            .finish()
            .map(|maybe_path| maybe_path.map(|path| path.display().to_string()))
            .map_err(output_writer_error_to_py)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn finish_interrupted(&self, signal_name: String) -> PyResult<()> {
        self.inner.finish_interrupted(&signal_name).map_err(output_writer_error_to_py)
    }

    fn abort(&self) -> PyResult<()> {
        self.inner.abort().map_err(output_writer_error_to_py)
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
        let beta_array = build_python_owned_arrow_array2::<f32, Float32Type>(&beta, beta_slice)?;
        let standard_error_array =
            build_python_owned_arrow_array2::<f32, Float32Type>(&standard_error, standard_error_slice)?;
        let chi_squared_array = build_python_owned_arrow_array2::<f32, Float32Type>(&chi_squared, chi_squared_slice)?;
        let log10_p_value_array =
            build_python_owned_arrow_array2::<f32, Float32Type>(&log10_p_value, log10_p_value_slice)?;
        let extra_code_array = match extra_code_slice {
            None => None,
            Some(extra_code_slice_values) => Some(build_python_owned_arrow_array2::<i32, Int32Type>(
                extra_code.as_ref().expect("extra code array should exist"),
                extra_code_slice_values,
            )?),
        };
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
            .map_err(output_writer_error_to_py)?;
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn finalize_output_run_chunks(
    run_directory: String,
    chunks_directory: String,
    association_mode: String,
) -> PyResult<String> {
    finalize_native_output_run_chunks(Path::new(&run_directory), Path::new(&chunks_directory), &association_mode)
        .map(|path| path.display().to_string())
        .map_err(output_writer_error_to_py)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn scan_committed_chunk_identifiers(chunks_directory: String) -> PyResult<Vec<i64>> {
    scan_native_committed_chunk_identifiers(Path::new(&chunks_directory)).map_err(output_writer_error_to_py)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn validate_strict_manifest_chunks(chunks_directory: String, manifest_json: String) -> PyResult<Vec<i64>> {
    validate_native_strict_manifest_chunks(Path::new(&chunks_directory), &manifest_json)
        .map_err(output_writer_error_to_py)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn repair_strict_manifest_chunk_commits(
    chunks_directory: String,
    manifest_json: String,
) -> PyResult<String> {
    let chunk_commits = repair_native_strict_manifest_chunk_commits(Path::new(&chunks_directory), &manifest_json)
        .map_err(output_writer_error_to_py)?;
    serde_json::to_string(
        &chunk_commits
            .into_iter()
            .map(|chunk_commit| {
                serde_json::json!({
                    "chunk_identifier": chunk_commit.chunk_identifier,
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

fn output_writer_error_to_py(error: OutputWriterError) -> PyErr {
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

fn build_python_owned_arrow_array1<T, ArrowType>(array: &PyReadonlyArray1<'_, T>, values: &[T]) -> PyResult<ArrayRef>
where
    T: ArrowNativeType + Element,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    build_python_owned_arrow_array_from_owner::<T, ArrowType>((**array).clone().into_any().unbind(), values)
}

fn build_python_owned_arrow_array2<T, ArrowType>(array: &PyReadonlyArray2<'_, T>, values: &[T]) -> PyResult<ArrayRef>
where
    T: ArrowNativeType + Element,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    build_python_owned_arrow_array_from_owner::<T, ArrowType>((**array).clone().into_any().unbind(), values)
}

fn build_python_owned_arrow_array_from_owner<T, ArrowType>(array_owner: Py<PyAny>, values: &[T]) -> PyResult<ArrayRef>
where
    T: ArrowNativeType + Element,
    ArrowType: ArrowPrimitiveType<Native = T>,
{
    let byte_length = std::mem::size_of_val(values);
    let pointer = NonNull::new(values.as_ptr().cast_mut().cast::<u8>())
        .ok_or_else(|| PyRuntimeError::new_err("NumPy result array has a null data pointer."))?;
    let buffer = unsafe {
        Buffer::from_custom_allocation(pointer, byte_length, Arc::new(PythonArrayAllocation { _owner: array_owner }))
    };
    Ok(Arc::new(PrimitiveArray::<ArrowType>::new(ScalarBuffer::new(buffer, 0, values.len()), None)))
}
