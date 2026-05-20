#![allow(clippy::needless_pass_by_value)]

use std::path::Path;
use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::output::{
    NativeChunkHandle, OutputWriterError, OutputWriterSession as NativeOutputWriterSession,
    finalize_output_run_chunks as finalize_native_output_run_chunks,
    scan_committed_chunk_identifiers as scan_native_committed_chunk_identifiers,
    validate_strict_manifest_chunks as validate_native_strict_manifest_chunks,
};

use super::{ChunkStats as PyChunkStats, VariantMetadata as PyVariantMetadata};

#[pyclass]
pub(crate) struct OutputWriterSession {
    inner: NativeOutputWriterSession,
}

#[pymethods]
impl OutputWriterSession {
    #[new]
    #[pyo3(signature = (
        run_directory,
        chunks_directory,
        association_mode,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=true,
        chunks_per_arrow_file=4,
        arrow_compression="zstd".to_string(),
    ))]
    fn new(
        run_directory: String,
        chunks_directory: String,
        association_mode: String,
        writer_thread_count: usize,
        writer_queue_depth: usize,
        finalize_parquet: bool,
        chunks_per_arrow_file: usize,
        arrow_compression: String,
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
        let extra_code_slice = extra_code.as_ref().map(|array| array.as_slice()).transpose()?;
        self.inner
            .write_regenie2_native_chunk_handle(
                NativeChunkHandle::new(
                    Arc::clone(&metadata.metadata),
                    Arc::clone(&chunk_stats.stats),
                    variant_start_index,
                ),
                beta.as_slice()?,
                standard_error.as_slice()?,
                chi_squared.as_slice()?,
                log10_p_value.as_slice()?,
                extra_code_slice,
            )
            .map_err(output_writer_error_to_py)
    }

    fn finish(&self) -> PyResult<Option<String>> {
        self.inner
            .finish()
            .map(|maybe_path| maybe_path.map(|path| path.display().to_string()))
            .map_err(output_writer_error_to_py)
    }

    fn abort(&self) -> PyResult<()> {
        self.inner.abort().map_err(output_writer_error_to_py)
    }
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

fn output_writer_error_to_py(error: OutputWriterError) -> PyErr {
    match error {
        OutputWriterError::InvalidInput(message) => PyValueError::new_err(message),
        OutputWriterError::Runtime(message) => PyRuntimeError::new_err(message),
    }
}
