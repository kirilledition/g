use std::path::Path;

use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::genotype::common::{ChunkStats as NativeChunkStats, VariantMetadataColumns};
use crate::output::{
    OutputWriterError, OutputWriterSession as NativeOutputWriterSession,
    finalize_output_run_chunks as finalize_native_output_run_chunks,
    scan_committed_chunk_identifiers as scan_native_committed_chunk_identifiers,
};

use super::{ChunkStats as PyChunkStats, VariantMetadata as PyVariantMetadata};

#[pyclass]
pub(crate) struct OutputWriterSession {
    inner: NativeOutputWriterSession,
}

#[pymethods]
impl OutputWriterSession {
    #[new]
    #[pyo3(signature = (run_directory, chunks_directory, association_mode, writer_thread_count=1, writer_queue_depth=1, finalize_parquet=true))]
    fn new(
        run_directory: String,
        chunks_directory: String,
        association_mode: String,
        writer_thread_count: usize,
        writer_queue_depth: usize,
        finalize_parquet: bool,
    ) -> PyResult<Self> {
        let inner = NativeOutputWriterSession::new(
            run_directory,
            chunks_directory,
            association_mode,
            writer_thread_count,
            writer_queue_depth,
            finalize_parquet,
        )
        .map_err(output_writer_error_to_py)?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (metadata, allele_one_frequency, observation_count, beta, standard_error, chi_squared, log10_p_value, extra_code=None))]
    fn write_regenie2_chunk(
        &self,
        metadata: &Bound<'_, PyAny>,
        allele_one_frequency: PyReadonlyArray1<'_, f32>,
        observation_count: PyReadonlyArray1<'_, i32>,
        beta: PyReadonlyArray1<'_, f32>,
        standard_error: PyReadonlyArray1<'_, f32>,
        chi_squared: PyReadonlyArray1<'_, f32>,
        log10_p_value: PyReadonlyArray1<'_, f32>,
        extra_code: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<()> {
        let variant_start_index = metadata.getattr("variant_start_index")?.extract::<i64>()?;
        let variant_stop_index = metadata.getattr("variant_stop_index")?.extract::<i64>()?;
        let position_object = metadata.getattr("position")?;
        let position = position_object.extract::<PyReadonlyArray1<'_, i64>>()?;
        let metadata_columns = VariantMetadataColumns {
            chromosome: extract_string_column(metadata, "chromosome")?,
            variant_identifier: extract_string_column(metadata, "variant_identifiers")?,
            position: position.as_slice()?.to_vec(),
            allele_one: extract_string_column(metadata, "allele_one")?,
            allele_two: extract_string_column(metadata, "allele_two")?,
        };
        let chunk_stats = NativeChunkStats {
            allele_one_frequency: allele_one_frequency.as_slice()?.to_vec(),
            observation_count: observation_count.as_slice()?.to_vec(),
            has_missing_values: false,
            dosage_sum: Vec::new(),
            dosage_variance_numerator: Vec::new(),
            info_score: vec![None; allele_one_frequency.len()],
            allele_count: Vec::new(),
            minor_allele_count: Vec::new(),
            zero_count: Vec::new(),
            nonzero_count: Vec::new(),
            homozygous_reference_count: Vec::new(),
            heterozygous_count: Vec::new(),
            homozygous_alternate_count: Vec::new(),
            is_sparse_candidate: Vec::new(),
            is_rare_sparse_firth_candidate: Vec::new(),
        };
        let extra_code_slice = extra_code.as_ref().map(|array| array.as_slice()).transpose()?;
        self.inner
            .write_regenie2_native_chunk(
                variant_start_index,
                variant_stop_index,
                &metadata_columns,
                &chunk_stats,
                beta.as_slice()?,
                standard_error.as_slice()?,
                chi_squared.as_slice()?,
                log10_p_value.as_slice()?,
                extra_code_slice,
            )
            .map_err(output_writer_error_to_py)
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
        let extra_code_slice = extra_code.as_ref().map(|array| array.as_slice()).transpose()?;
        self.inner
            .write_regenie2_native_chunk(
                variant_start_index,
                variant_stop_index,
                &metadata.metadata,
                &chunk_stats.stats,
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

fn extract_string_column(metadata: &Bound<'_, PyAny>, attribute_name: &str) -> PyResult<Vec<String>> {
    let column_object = metadata.getattr(attribute_name)?;
    if let Ok(values) = column_object.extract::<Vec<String>>() {
        return Ok(values);
    }
    column_object.call_method0("tolist")?.extract::<Vec<String>>()
}

fn output_writer_error_to_py(error: OutputWriterError) -> PyErr {
    match error {
        OutputWriterError::InvalidInput(message) => PyValueError::new_err(message),
        OutputWriterError::Runtime(message) => PyRuntimeError::new_err(message),
    }
}
