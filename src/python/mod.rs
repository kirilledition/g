use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_channel::{Receiver, bounded};
use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::genotype::bgen::{BgenError, BgenReaderCore, ReaderProfileSnapshot, VariantMetadataLists};
use crate::genotype::common::{
    ChunkSpec as NativeChunkSpec, ChunkStats as NativeChunkStats, GenotypeError, VariantMetadataColumns,
};
use crate::genotype::planner;
use crate::pipeline::Regenie2RunEngineCore;
use crate::regenie::{PredictionError, PredictionSource};
use crate::sample::{AlignedSampleData, AlignmentInputs};

mod output;

use output::{OutputWriterSession, finalize_output_run_chunks, scan_committed_chunk_identifiers};

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct ChunkSpec {
    chunk_spec: NativeChunkSpec,
}

#[pymethods]
impl ChunkSpec {
    #[getter]
    fn variant_start_index(&self) -> usize {
        self.chunk_spec.variant_start_index
    }

    #[getter]
    fn variant_stop_index(&self) -> usize {
        self.chunk_spec.variant_stop_index
    }
}

#[pyclass]
pub(crate) struct ChunkStats {
    pub(crate) stats: NativeChunkStats,
}

impl ChunkStats {
    fn new(stats: NativeChunkStats) -> Self {
        Self { stats }
    }
}

#[pymethods]
impl ChunkStats {
    #[getter]
    fn allele_one_frequency<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.allele_one_frequency.clone().into_pyarray(py)
    }

    #[getter]
    fn observation_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.stats.observation_count.clone().into_pyarray(py)
    }

    #[getter]
    fn has_missing_values(&self) -> bool {
        self.stats.has_missing_values
    }

    #[getter]
    fn info_score<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats
            .info_score
            .iter()
            .map(|maybe_info_score| maybe_info_score.unwrap_or(f32::NAN))
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[getter]
    fn minor_allele_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.minor_allele_count.clone().into_pyarray(py)
    }

    #[getter]
    fn zero_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.stats.zero_count.clone().into_pyarray(py)
    }

    #[getter]
    fn nonzero_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.stats.nonzero_count.clone().into_pyarray(py)
    }

    #[getter]
    fn is_sparse_candidate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.stats.is_sparse_candidate.clone().into_pyarray(py)
    }

    #[getter]
    fn is_rare_sparse_firth_candidate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.stats.is_rare_sparse_firth_candidate.clone().into_pyarray(py)
    }
}

#[pyclass]
pub(crate) struct VariantMetadata {
    pub(crate) variant_start_index: usize,
    pub(crate) variant_stop_index: usize,
    pub(crate) metadata: VariantMetadataColumns,
}

#[pyclass]
pub(crate) struct NativeAlignedSampleData {
    data: AlignedSampleData,
}

impl VariantMetadata {
    fn new(variant_start_index: usize, variant_stop_index: usize, metadata: VariantMetadataColumns) -> Self {
        Self { variant_start_index, variant_stop_index, metadata }
    }
}

impl NativeAlignedSampleData {
    fn new(data: AlignedSampleData) -> Self {
        Self { data }
    }
}

#[pymethods]
impl NativeAlignedSampleData {
    #[getter]
    fn sample_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.data.sample_indices.clone().into_pyarray(py)
    }

    #[getter]
    fn family_identifiers(&self) -> Vec<String> {
        self.data.family_identifiers.clone()
    }

    #[getter]
    fn individual_identifiers(&self) -> Vec<String> {
        self.data.individual_identifiers.clone()
    }

    #[getter]
    fn phenotype_name(&self) -> String {
        self.data.phenotype_name.clone()
    }

    #[getter]
    fn phenotype_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.data.phenotype_vector.clone().into_pyarray(py)
    }

    #[getter]
    fn covariate_names(&self) -> Vec<String> {
        self.data.covariate_names.clone()
    }

    #[getter]
    fn covariate_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let covariate_matrix = Array2::from_shape_vec(
            (self.data.covariate_row_count, self.data.covariate_column_count),
            self.data.covariate_matrix_values.clone(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(covariate_matrix.into_pyarray(py))
    }

    #[getter]
    fn is_binary_trait(&self) -> bool {
        self.data.is_binary_trait
    }
}

#[pymethods]
impl VariantMetadata {
    #[getter]
    fn variant_start_index(&self) -> usize {
        self.variant_start_index
    }

    #[getter]
    fn variant_stop_index(&self) -> usize {
        self.variant_stop_index
    }

    #[getter]
    fn chromosome(&self) -> Vec<String> {
        self.metadata.chromosome.clone()
    }

    #[getter]
    fn variant_identifiers(&self) -> Vec<String> {
        self.metadata.variant_identifier.clone()
    }

    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.metadata.position.clone().into_pyarray(py)
    }

    #[getter]
    fn allele_one(&self) -> Vec<String> {
        self.metadata.allele_one.clone()
    }

    #[getter]
    fn allele_two(&self) -> Vec<String> {
        self.metadata.allele_two.clone()
    }
}

#[pyclass]
struct BgenReader {
    reader: BgenReaderCore,
}

#[pymethods]
impl BgenReader {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (bgen_path, trusted_no_missing_diploid=false))]
    fn new(bgen_path: String, trusted_no_missing_diploid: bool) -> PyResult<Self> {
        let reader = BgenReaderCore::open(std::path::Path::new(&bgen_path), trusted_no_missing_diploid)
            .map_err(convert_bgen_error)?;
        Ok(Self { reader })
    }

    #[getter]
    fn sample_count(&self) -> usize {
        self.reader.sample_count()
    }

    #[getter]
    fn variant_count(&self) -> usize {
        self.reader.variant_count()
    }

    #[getter]
    fn contains_embedded_samples(&self) -> bool {
        self.reader.contains_embedded_samples()
    }

    #[getter]
    fn bgen_path(&self) -> String {
        self.reader.bgen_path().display().to_string()
    }

    fn sample_identifiers(&self) -> Vec<String> {
        self.reader.sample_identifiers()
    }

    fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.reader.chromosome_boundary_indices()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn prepare_sample_selection(&self, sample_indices: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        self.reader.prepare_sample_selection(sample_indices.as_slice()?).map_err(convert_bgen_error)
    }

    fn clear_prepared_sample_selection(&self) -> PyResult<()> {
        self.reader.clear_prepared_sample_selection().map_err(convert_bgen_error)
    }

    fn reset_profile(&self) {
        self.reader.reset_profile();
    }

    fn profile_snapshot(&self) -> HashMap<String, u64> {
        build_profile_snapshot_dict(&self.reader.profile_snapshot())
    }

    fn validate_trusted_no_missing_diploid(&self) -> PyResult<()> {
        self.reader.validate_trusted_no_missing_diploid().map_err(convert_bgen_error)
    }

    fn variant_metadata_slice(&self, variant_start: usize, variant_stop: usize) -> PyResult<VariantMetadataLists> {
        self.reader.variant_metadata_slice(variant_start, variant_stop).map_err(convert_bgen_error)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn read_dosage_f32<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        variant_start: usize,
        variant_stop: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let sample_index_values = sample_indices.as_slice()?;
        let selected_sample_count = sample_index_values.len();
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        let dosage_values = py
            .detach(|| self.reader.read_dosage_f32(sample_index_values, variant_start, variant_stop))
            .map_err(convert_bgen_error)?;
        let dosage_matrix = Array2::from_shape_vec((selected_sample_count, selected_variant_count), dosage_values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(dosage_matrix.into_pyarray(py))
    }

    fn read_dosage_f32_prepared<'py>(
        &self,
        py: Python<'py>,
        variant_start: usize,
        variant_stop: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        let dosage_values = py
            .detach(|| self.reader.read_dosage_f32_prepared(variant_start, variant_stop))
            .map_err(convert_bgen_error)?;
        let selected_sample_count = dosage_values.len().checked_div(selected_variant_count).unwrap_or(0);
        let dosage_matrix = Array2::from_shape_vec((selected_sample_count, selected_variant_count), dosage_values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(dosage_matrix.into_pyarray(py))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn read_dosage_f32_into<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        variant_start: usize,
        variant_stop: usize,
        mut output_array: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let sample_index_values = sample_indices.as_slice()?;
        let selected_sample_count = sample_index_values.len();
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        let output_shape = output_array.shape();
        if output_shape != [selected_sample_count, selected_variant_count] {
            return Err(PyValueError::new_err(format!(
                "Output array shape mismatch: expected ({selected_sample_count}, {selected_variant_count}), observed ({}, {}).",
                output_shape[0], output_shape[1],
            )));
        }
        if !output_array.is_c_contiguous() {
            return Err(PyValueError::new_err("Output array for BGEN dosage reads must be C-contiguous float32."));
        }

        let output_slice = output_array.as_slice_mut().map_err(|_| {
            PyValueError::new_err("Output array for BGEN dosage reads must expose a contiguous mutable slice.")
        })?;
        let output_pointer_address = output_slice.as_mut_ptr() as usize;
        let output_value_count = output_slice.len();

        py.detach(|| {
            self.reader.read_dosage_f32_into_address(
                sample_index_values,
                variant_start,
                variant_stop,
                output_pointer_address,
                output_value_count,
            )
        })
        .map_err(convert_bgen_error)
    }

    fn read_dosage_f32_into_prepared<'py>(
        &self,
        py: Python<'py>,
        variant_start: usize,
        variant_stop: usize,
        mut output_array: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let output_shape = output_array.shape();
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        if output_shape[1] != selected_variant_count {
            return Err(PyValueError::new_err(format!(
                "Output array shape mismatch: expected variant width {selected_variant_count}, observed {}.",
                output_shape[1],
            )));
        }
        if !output_array.is_c_contiguous() {
            return Err(PyValueError::new_err("Output array for BGEN dosage reads must be C-contiguous float32."));
        }

        let output_slice = output_array.as_slice_mut().map_err(|_| {
            PyValueError::new_err("Output array for BGEN dosage reads must expose a contiguous mutable slice.")
        })?;
        let output_pointer_address = output_slice.as_mut_ptr() as usize;
        let output_value_count = output_slice.len();

        py.detach(|| {
            self.reader.read_dosage_f32_into_address_prepared(
                variant_start,
                variant_stop,
                output_pointer_address,
                output_value_count,
            )
        })
        .map_err(convert_bgen_error)
    }

    fn read_preprocessed_dosage_f32_into_prepared<'py>(
        &self,
        py: Python<'py>,
        variant_start: usize,
        variant_stop: usize,
        mut output_array: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<ChunkStats> {
        let output_shape = output_array.shape();
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        if output_shape[1] != selected_variant_count {
            return Err(PyValueError::new_err(format!(
                "Output array shape mismatch: expected variant width {selected_variant_count}, observed {}.",
                output_shape[1],
            )));
        }
        if !output_array.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "Output array for preprocessed BGEN dosage reads must be C-contiguous float32.",
            ));
        }

        let output_slice = output_array.as_slice_mut().map_err(|_| {
            PyValueError::new_err(
                "Output array for preprocessed BGEN dosage reads must expose a contiguous mutable slice.",
            )
        })?;
        let output_pointer_address = output_slice.as_mut_ptr() as usize;
        let output_value_count = output_slice.len();

        let stats = py
            .detach(|| {
                self.reader.read_preprocessed_dosage_f32_into_address_prepared(
                    variant_start,
                    variant_stop,
                    output_pointer_address,
                    output_value_count,
                )
            })
            .map_err(convert_bgen_error)?;
        Ok(ChunkStats::new(stats))
    }

    #[allow(clippy::unused_self)]
    fn close(&self) {}
}

#[pyclass]
struct Regenie2RunEngine {
    engine: Regenie2RunEngineCore,
}

#[pyclass]
struct RegeniePredictionSource {
    source: PredictionSource,
}

struct StagedBgenDosageChunk {
    chunk_spec: NativeChunkSpec,
    metadata: VariantMetadataColumns,
    dosage_values: Vec<f32>,
}

#[pymethods]
impl Regenie2RunEngine {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (bgen_path, chunk_size, variant_limit=None, trusted_no_missing_diploid=false))]
    fn new(
        bgen_path: String,
        chunk_size: usize,
        variant_limit: Option<usize>,
        trusted_no_missing_diploid: bool,
    ) -> PyResult<Self> {
        let engine = Regenie2RunEngineCore::open_bgen(
            Path::new(&bgen_path),
            chunk_size,
            variant_limit,
            trusted_no_missing_diploid,
        )
        .map_err(convert_bgen_error)?;
        Ok(Self { engine })
    }

    #[getter]
    fn sample_count(&self) -> usize {
        self.engine.reader().sample_count()
    }

    #[getter]
    fn variant_count(&self) -> usize {
        self.engine.reader().variant_count()
    }

    #[getter]
    fn contains_embedded_samples(&self) -> bool {
        self.engine.reader().contains_embedded_samples()
    }

    fn sample_identifiers(&self) -> Vec<String> {
        self.engine.reader().sample_identifiers()
    }

    fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.engine.reader().chromosome_boundary_indices()
    }

    fn variant_metadata_slice(&self, variant_start: usize, variant_stop: usize) -> PyResult<VariantMetadataLists> {
        self.engine.reader().variant_metadata_slice(variant_start, variant_stop).map_err(convert_bgen_error)
    }

    fn reset_profile(&self) {
        self.engine.reader().reset_profile();
    }

    fn profile_snapshot(&self) -> HashMap<String, u64> {
        build_profile_snapshot_dict(&self.engine.reader().profile_snapshot())
    }

    fn validate_trusted_no_missing_diploid(&self) -> PyResult<()> {
        self.engine.reader().validate_trusted_no_missing_diploid().map_err(convert_bgen_error)
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (sample_indices, callback, committed_chunk_identifiers=None))]
    fn run_bgen_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let sample_index_values = sample_indices.as_slice()?.to_vec();
        self.engine.reader().prepare_sample_selection(&sample_index_values).map_err(convert_bgen_error)?;

        let run_result = self.run_prepared_bgen_chunks(py, &sample_index_values, callback, committed_chunk_identifiers);
        let clear_result = self.engine.reader().clear_prepared_sample_selection().map_err(convert_bgen_error);
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (sample_indices, callback, committed_chunk_identifiers=None, prefetch_chunks=1))]
    fn run_bgen_dosage_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        prefetch_chunks: usize,
    ) -> PyResult<usize> {
        let sample_index_values = sample_indices.as_slice()?.to_vec();
        self.engine.reader().prepare_sample_selection(&sample_index_values).map_err(convert_bgen_error)?;

        let run_result = self.run_prepared_bgen_dosage_chunks(
            py,
            &sample_index_values,
            callback,
            committed_chunk_identifiers,
            prefetch_chunks,
        );
        let clear_result = self.engine.reader().clear_prepared_sample_selection().map_err(convert_bgen_error);
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (sample_indices, callback, committed_chunk_identifiers=None))]
    fn run_bgen_dosage_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let sample_index_values = sample_indices.as_slice()?.to_vec();
        self.engine.reader().prepare_sample_selection(&sample_index_values).map_err(convert_bgen_error)?;

        let run_result = self.run_prepared_bgen_dosage_buffered_chunks(
            py,
            &sample_index_values,
            callback,
            committed_chunk_identifiers,
        );
        let clear_result = self.engine.reader().clear_prepared_sample_selection().map_err(convert_bgen_error);
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (sample_indices, callback, committed_chunk_identifiers=None))]
    fn run_bgen_variant_major_dosage_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let sample_index_values = sample_indices.as_slice()?.to_vec();
        self.engine.reader().prepare_sample_selection(&sample_index_values).map_err(convert_bgen_error)?;

        let run_result = self.run_prepared_bgen_variant_major_dosage_buffered_chunks(
            py,
            &sample_index_values,
            callback,
            committed_chunk_identifiers,
        );
        let clear_result = self.engine.reader().clear_prepared_sample_selection().map_err(convert_bgen_error);
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }
}

#[pymethods]
impl RegeniePredictionSource {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        prediction_list_path: String,
        phenotype_name: String,
        sample_family_identifiers: Vec<String>,
        sample_individual_identifiers: Vec<String>,
    ) -> PyResult<Self> {
        let source = PredictionSource::load(
            Path::new(&prediction_list_path),
            &phenotype_name,
            &sample_family_identifiers,
            &sample_individual_identifiers,
        )
        .map_err(convert_prediction_error)?;
        Ok(Self { source })
    }

    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    fn from_native_aligned_sample_data(
        prediction_list_path: String,
        phenotype_name: String,
        aligned_sample_data: PyRef<'_, NativeAlignedSampleData>,
    ) -> PyResult<Self> {
        let source = PredictionSource::load(
            Path::new(&prediction_list_path),
            &phenotype_name,
            &aligned_sample_data.data.family_identifiers,
            &aligned_sample_data.data.individual_identifiers,
        )
        .map_err(convert_prediction_error)?;
        Ok(Self { source })
    }

    #[allow(clippy::needless_pass_by_value)]
    fn get_chromosome_predictions<'py>(
        &self,
        py: Python<'py>,
        chromosome: String,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let prediction_values = self.source.chromosome_predictions(&chromosome).map_err(convert_prediction_error)?;
        Ok(Array1::from_vec(prediction_values.to_vec()).into_pyarray(py))
    }
}

impl Regenie2RunEngine {
    fn run_prepared_bgen_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self.engine.plan_chunks(&committed_identifier_set).map_err(convert_genotype_error)?;
        for chunk_spec in &chunk_specs {
            let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
            let mut dosage_values = vec![0.0_f32; sample_index_values.len() * selected_variant_count];
            let output_pointer_address = dosage_values.as_mut_ptr() as usize;
            let output_value_count = dosage_values.len();
            let stats = py
                .detach(|| {
                    self.engine.reader().read_preprocessed_dosage_f32_into_address_prepared(
                        chunk_spec.variant_start_index,
                        chunk_spec.variant_stop_index,
                        output_pointer_address,
                        output_value_count,
                    )
                })
                .map_err(convert_bgen_error)?;
            let metadata_tuple = self
                .engine
                .reader()
                .variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)
                .map_err(convert_bgen_error)?;
            let metadata = Py::new(
                py,
                VariantMetadata::new(
                    chunk_spec.variant_start_index,
                    chunk_spec.variant_stop_index,
                    convert_variant_metadata_tuple(metadata_tuple),
                ),
            )?;
            let genotype_matrix =
                Array2::from_shape_vec((sample_index_values.len(), selected_variant_count), dosage_values)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
                    .into_pyarray(py);
            let allele_one_frequency = stats.allele_one_frequency.into_pyarray(py);
            let observation_count = stats.observation_count.into_pyarray(py);
            callback
                .call_method1("compute_chunk", (metadata, genotype_matrix, allele_one_frequency, observation_count))?;
        }
        Ok(chunk_specs.len())
    }

    fn run_prepared_bgen_dosage_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        prefetch_chunks: usize,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self.engine.plan_chunks(&committed_identifier_set).map_err(convert_genotype_error)?;
        if prefetch_chunks > 0 {
            return self.run_prefetched_bgen_dosage_chunks(
                py,
                sample_index_values,
                callback,
                chunk_specs,
                prefetch_chunks,
            );
        }
        for chunk_spec in &chunk_specs {
            let staged_chunk =
                self.read_bgen_dosage_chunk(sample_index_values.len(), chunk_spec).map_err(convert_bgen_error)?;
            call_dosage_chunk_callback(py, sample_index_values.len(), callback, staged_chunk)?;
        }
        Ok(chunk_specs.len())
    }

    fn run_prefetched_bgen_dosage_chunks<'py>(
        &self,
        py: Python<'py>,
        selected_sample_count: &[i64],
        callback: &Bound<'py, PyAny>,
        chunk_specs: Vec<NativeChunkSpec>,
        prefetch_chunks: usize,
    ) -> PyResult<usize> {
        let (sender, receiver) = bounded(prefetch_chunks.max(1));
        let cancellation_flag = Arc::new(AtomicBool::new(false));
        let selected_sample_count = selected_sample_count.len();
        std::thread::scope(|thread_scope| {
            let reader = self.engine.reader();
            let cancellation_flag_clone = Arc::clone(&cancellation_flag);
            thread_scope.spawn(move || {
                for chunk_spec in chunk_specs {
                    if cancellation_flag_clone.load(Ordering::Relaxed) {
                        return;
                    }
                    let staged_chunk_result =
                        read_bgen_dosage_chunk_from_reader(reader, selected_sample_count, &chunk_spec)
                            .map_err(|error| error.to_string());
                    if sender.send(staged_chunk_result).is_err() {
                        return;
                    }
                }
            });
            consume_prefetched_bgen_dosage_chunks(py, selected_sample_count, callback, receiver, &cancellation_flag)
        })
    }

    fn read_bgen_dosage_chunk(
        &self,
        selected_sample_count: usize,
        chunk_spec: &NativeChunkSpec,
    ) -> Result<StagedBgenDosageChunk, BgenError> {
        read_bgen_dosage_chunk_from_reader(self.engine.reader(), selected_sample_count, chunk_spec)
    }

    fn run_prepared_bgen_dosage_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self.engine.plan_chunks(&committed_identifier_set).map_err(convert_genotype_error)?;
        for chunk_spec in &chunk_specs {
            let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
            let output_array_object =
                callback.call_method1("acquire_dosage_buffer", (sample_index_values.len(), selected_variant_count))?;
            let stats = {
                let mut output_array = output_array_object.extract::<PyReadwriteArray2<'_, f32>>()?;
                let output_shape = output_array.shape();
                if output_shape != [sample_index_values.len(), selected_variant_count] {
                    return Err(PyValueError::new_err(format!(
                        "Reusable BGEN dosage buffer shape mismatch: expected ({}, {selected_variant_count}), observed ({}, {}).",
                        sample_index_values.len(),
                        output_shape[0],
                        output_shape[1],
                    )));
                }
                if !output_array.is_c_contiguous() {
                    return Err(PyValueError::new_err("Reusable BGEN dosage buffer must be C-contiguous float32."));
                }
                let output_slice = output_array.as_slice_mut().map_err(|_| {
                    PyValueError::new_err("Reusable BGEN dosage buffer must expose a contiguous mutable slice.")
                })?;
                let output_pointer_address = output_slice.as_mut_ptr() as usize;
                let output_value_count = output_slice.len();
                let chunk_stats = py
                    .detach(|| {
                        self.engine.reader().read_preprocessed_dosage_f32_into_address_prepared(
                            chunk_spec.variant_start_index,
                            chunk_spec.variant_stop_index,
                            output_pointer_address,
                            output_value_count,
                        )
                    })
                    .map_err(convert_bgen_error)?;
                Py::new(py, ChunkStats::new(chunk_stats))?
            };
            let metadata_tuple = self
                .engine
                .reader()
                .variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)
                .map_err(convert_bgen_error)?;
            let metadata = Py::new(
                py,
                VariantMetadata::new(
                    chunk_spec.variant_start_index,
                    chunk_spec.variant_stop_index,
                    convert_variant_metadata_tuple(metadata_tuple),
                ),
            )?;
            callback.call_method1("compute_preprocessed_dosage_chunk", (metadata, output_array_object, stats))?;
        }
        Ok(chunk_specs.len())
    }

    fn run_prepared_bgen_variant_major_dosage_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self.engine.plan_chunks(&committed_identifier_set).map_err(convert_genotype_error)?;
        for chunk_spec in &chunk_specs {
            let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
            let output_array_object = callback.call_method1(
                "acquire_variant_major_dosage_buffer",
                (selected_variant_count, sample_index_values.len()),
            )?;
            let stats = {
                let mut output_array = output_array_object.extract::<PyReadwriteArray2<'_, f32>>()?;
                let output_shape = output_array.shape();
                if output_shape != [selected_variant_count, sample_index_values.len()] {
                    return Err(PyValueError::new_err(format!(
                        "Reusable variant-major BGEN dosage buffer shape mismatch: expected ({selected_variant_count}, {}), observed ({}, {}).",
                        sample_index_values.len(),
                        output_shape[0],
                        output_shape[1],
                    )));
                }
                if !output_array.is_c_contiguous() {
                    return Err(PyValueError::new_err(
                        "Reusable variant-major BGEN dosage buffer must be C-contiguous float32.",
                    ));
                }
                let output_slice = output_array.as_slice_mut().map_err(|_| {
                    PyValueError::new_err(
                        "Reusable variant-major BGEN dosage buffer must expose a contiguous mutable slice.",
                    )
                })?;
                let output_pointer_address = output_slice.as_mut_ptr() as usize;
                let output_value_count = output_slice.len();
                let chunk_stats = py
                    .detach(|| {
                        self.engine.reader().read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                            chunk_spec.variant_start_index,
                            chunk_spec.variant_stop_index,
                            output_pointer_address,
                            output_value_count,
                        )
                    })
                    .map_err(convert_bgen_error)?;
                Py::new(py, ChunkStats::new(chunk_stats))?
            };
            let metadata_tuple = self
                .engine
                .reader()
                .variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)
                .map_err(convert_bgen_error)?;
            let metadata = Py::new(
                py,
                VariantMetadata::new(
                    chunk_spec.variant_start_index,
                    chunk_spec.variant_stop_index,
                    convert_variant_metadata_tuple(metadata_tuple),
                ),
            )?;
            callback.call_method1(
                "compute_preprocessed_variant_major_dosage_chunk",
                (metadata, output_array_object, stats),
            )?;
        }
        Ok(chunk_specs.len())
    }
}

fn read_bgen_dosage_chunk_from_reader(
    reader: &BgenReaderCore,
    selected_sample_count: usize,
    chunk_spec: &NativeChunkSpec,
) -> Result<StagedBgenDosageChunk, BgenError> {
    let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
    let mut dosage_values = vec![0.0_f32; selected_sample_count * selected_variant_count];
    let output_pointer_address = dosage_values.as_mut_ptr() as usize;
    let output_value_count = dosage_values.len();
    reader.read_dosage_f32_into_address_prepared(
        chunk_spec.variant_start_index,
        chunk_spec.variant_stop_index,
        output_pointer_address,
        output_value_count,
    )?;
    let metadata_tuple =
        reader.variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)?;
    Ok(StagedBgenDosageChunk {
        chunk_spec: chunk_spec.clone(),
        metadata: convert_variant_metadata_tuple(metadata_tuple),
        dosage_values,
    })
}

fn consume_prefetched_bgen_dosage_chunks<'py>(
    py: Python<'py>,
    selected_sample_count: usize,
    callback: &Bound<'py, PyAny>,
    receiver: Receiver<Result<StagedBgenDosageChunk, String>>,
    cancellation_flag: &Arc<AtomicBool>,
) -> PyResult<usize> {
    let mut processed_chunk_count = 0;
    for staged_chunk_result in receiver {
        let staged_chunk = match staged_chunk_result {
            Ok(staged_chunk) => staged_chunk,
            Err(message) => {
                cancellation_flag.store(true, Ordering::Relaxed);
                return Err(PyRuntimeError::new_err(message));
            }
        };
        if let Err(error) = call_dosage_chunk_callback(py, selected_sample_count, callback, staged_chunk) {
            cancellation_flag.store(true, Ordering::Relaxed);
            return Err(error);
        }
        processed_chunk_count += 1;
    }
    Ok(processed_chunk_count)
}

fn call_dosage_chunk_callback<'py>(
    py: Python<'py>,
    selected_sample_count: usize,
    callback: &Bound<'py, PyAny>,
    staged_chunk: StagedBgenDosageChunk,
) -> PyResult<()> {
    let selected_variant_count =
        staged_chunk.chunk_spec.variant_stop_index - staged_chunk.chunk_spec.variant_start_index;
    let metadata = Py::new(
        py,
        VariantMetadata::new(
            staged_chunk.chunk_spec.variant_start_index,
            staged_chunk.chunk_spec.variant_stop_index,
            staged_chunk.metadata,
        ),
    )?;
    let genotype_matrix =
        Array2::from_shape_vec((selected_sample_count, selected_variant_count), staged_chunk.dosage_values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .into_pyarray(py);
    callback.call_method1("compute_dosage_chunk", (metadata, genotype_matrix))?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    sample_indices,
    family_identifiers,
    individual_identifiers,
    phenotype_path,
    phenotype_name,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false
))]
fn align_sample_data<'py>(
    py: Python<'py>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    family_identifiers: Vec<String>,
    individual_identifiers: Vec<String>,
    phenotype_path: String,
    phenotype_name: String,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
) -> PyResult<NativeAlignedSampleData> {
    let sample_index_values = sample_indices.as_slice()?.to_vec();
    let inputs = AlignmentInputs {
        sample_indices: sample_index_values,
        family_identifiers,
        individual_identifiers,
        phenotype_path,
        phenotype_name,
        covariate_path,
        covariate_names,
        is_binary_trait,
    };
    py.detach(|| crate::sample::align_sample_data(inputs))
        .map(NativeAlignedSampleData::new)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    sample_path,
    expected_sample_count,
    phenotype_path,
    phenotype_name,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false
))]
fn align_sample_data_from_sample_file<'py>(
    py: Python<'py>,
    sample_path: String,
    expected_sample_count: usize,
    phenotype_path: String,
    phenotype_name: String,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
) -> PyResult<NativeAlignedSampleData> {
    py.detach(move || {
        crate::sample::align_sample_data_from_sample_file(
            Path::new(&sample_path),
            expected_sample_count,
            phenotype_path,
            phenotype_name,
            covariate_path,
            covariate_names,
            is_binary_trait,
        )
    })
    .map(NativeAlignedSampleData::new)
    .map_err(PyValueError::new_err)
}

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from g!".to_string()
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (variant_count, chunk_size, chromosome_boundary_indices, variant_limit=None, committed_chunk_identifiers=None))]
fn plan_genotype_chunks(
    variant_count: usize,
    chunk_size: usize,
    chromosome_boundary_indices: Vec<usize>,
    variant_limit: Option<usize>,
    committed_chunk_identifiers: Option<Vec<usize>>,
) -> PyResult<Vec<ChunkSpec>> {
    let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
    let chunk_specs = planner::plan_chromosome_homogeneous_chunks(
        variant_count,
        chunk_size,
        variant_limit,
        &chromosome_boundary_indices,
        &committed_identifier_set,
    )
    .map_err(convert_genotype_error)?;
    Ok(chunk_specs.into_iter().map(|chunk_spec| ChunkSpec { chunk_spec }).collect())
}

fn validate_supported_layout(combination_count: usize, is_phased: bool) -> PyResult<()> {
    if matches!((combination_count, is_phased), (3, false) | (4, true)) {
        return Ok(());
    }
    Err(PyValueError::new_err(
        "Unsupported BGEN probability layout. Only diploid biallelic phased or unphased variants are supported.",
    ))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn convert_probability_tensor_to_dosage_f32<'py>(
    py: Python<'py>,
    probability_tensor: PyReadonlyArray3<'py, f32>,
    combination_count: usize,
    is_phased: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    validate_supported_layout(combination_count, is_phased)?;

    let probability_array = probability_tensor.as_array();
    let shape = probability_array.shape();
    let sample_count = shape[0];
    let variant_count = shape[1];
    let observed_combination_count = shape[2];

    if observed_combination_count != combination_count {
        return Err(PyValueError::new_err(format!(
            "Probability tensor combination count mismatch: expected {combination_count}, observed {observed_combination_count}.",
        )));
    }

    let dosage_values = py.detach(move || {
        let mut values = vec![0.0_f32; sample_count * variant_count];
        match (combination_count, is_phased) {
            (3, false) => {
                for sample_index in 0..sample_count {
                    for variant_index in 0..variant_count {
                        let dosage_value = probability_array[[sample_index, variant_index, 1]]
                            + (2.0 * probability_array[[sample_index, variant_index, 2]]);
                        values[(sample_index * variant_count) + variant_index] = dosage_value;
                    }
                }
            }
            (4, true) => {
                for sample_index in 0..sample_count {
                    for variant_index in 0..variant_count {
                        let dosage_value = probability_array[[sample_index, variant_index, 1]]
                            + probability_array[[sample_index, variant_index, 3]];
                        values[(sample_index * variant_count) + variant_index] = dosage_value;
                    }
                }
            }
            _ => unreachable!(),
        }
        values
    });

    let dosage_matrix = Array2::from_shape_vec((sample_count, variant_count), dosage_values)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(dosage_matrix.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn convert_probability_matrix_to_dosage_f32<'py>(
    py: Python<'py>,
    probability_matrix: PyReadonlyArray2<'py, f32>,
    combination_count: usize,
    is_phased: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    validate_supported_layout(combination_count, is_phased)?;

    let probability_array = probability_matrix.as_array();
    let shape = probability_array.shape();
    let sample_count = shape[0];
    let observed_combination_count = shape[1];

    if observed_combination_count != combination_count {
        return Err(PyValueError::new_err(format!(
            "Probability matrix combination count mismatch: expected {combination_count}, observed {observed_combination_count}.",
        )));
    }

    let dosage_values = py.detach(move || {
        let mut values = vec![0.0_f32; sample_count];
        match (combination_count, is_phased) {
            (3, false) => {
                for sample_index in 0..sample_count {
                    values[sample_index] =
                        probability_array[[sample_index, 1]] + (2.0 * probability_array[[sample_index, 2]]);
                }
            }
            (4, true) => {
                for sample_index in 0..sample_count {
                    values[sample_index] = probability_array[[sample_index, 1]] + probability_array[[sample_index, 3]];
                }
            }
            _ => unreachable!(),
        }
        values
    });

    Ok(Array1::from_vec(dosage_values).into_pyarray(py))
}

fn convert_bgen_error(error: BgenError) -> PyErr {
    match error {
        BgenError::InvalidFormat(message) | BgenError::UnsupportedFormat(message) | BgenError::Range(message) => {
            PyValueError::new_err(message)
        }
        BgenError::Io(io_error) => PyRuntimeError::new_err(io_error.to_string()),
    }
}

fn convert_genotype_error(error: GenotypeError) -> PyErr {
    match error {
        GenotypeError::InvalidInput(message) => PyValueError::new_err(message),
        GenotypeError::Reader(message) => PyRuntimeError::new_err(message),
    }
}

fn convert_prediction_error(error: PredictionError) -> PyErr {
    match error {
        PredictionError::PredictionListNotFound(path) => pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Prediction list file not found: {}",
            path.display()
        )),
        PredictionError::LocoFileNotFound(path) => {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!("LOCO file not found: {}", path.display()))
        }
        PredictionError::Io(io_error) => PyRuntimeError::new_err(io_error.to_string()),
        other_error => PyValueError::new_err(other_error.to_string()),
    }
}

fn build_committed_identifier_set(committed_chunk_identifiers: Option<Vec<usize>>) -> BTreeSet<usize> {
    committed_chunk_identifiers.unwrap_or_default().into_iter().collect()
}

fn convert_variant_metadata_tuple(variant_metadata: VariantMetadataLists) -> VariantMetadataColumns {
    let (chromosome, variant_identifier, position, allele_one, allele_two) = variant_metadata;
    VariantMetadataColumns { chromosome, variant_identifier, position, allele_one, allele_two }
}

fn build_profile_snapshot_dict(profile_snapshot: &ReaderProfileSnapshot) -> HashMap<String, u64> {
    HashMap::from([
        ("sample_selection_prepare_ns".to_string(), profile_snapshot.sample_selection_prepare_ns),
        ("sample_selection_prepare_count".to_string(), profile_snapshot.sample_selection_prepare_count),
        ("compressed_block_fetch_ns".to_string(), profile_snapshot.compressed_block_fetch_ns),
        ("compressed_block_fetch_count".to_string(), profile_snapshot.compressed_block_fetch_count),
        ("compressed_byte_count".to_string(), profile_snapshot.compressed_byte_count),
        ("decompression_ns".to_string(), profile_snapshot.decompression_ns),
        ("decompression_count".to_string(), profile_snapshot.decompression_count),
        ("uncompressed_byte_count".to_string(), profile_snapshot.uncompressed_byte_count),
        ("zlib_stream_count".to_string(), profile_snapshot.zlib_stream_count),
        ("probability_decode_ns".to_string(), profile_snapshot.probability_decode_ns),
        ("probability_decode_count".to_string(), profile_snapshot.probability_decode_count),
        ("variant_decode_count".to_string(), profile_snapshot.variant_decode_count),
        ("output_write_ns".to_string(), profile_snapshot.output_write_ns),
        ("output_write_count".to_string(), profile_snapshot.output_write_count),
        ("output_byte_count".to_string(), profile_snapshot.output_byte_count),
        ("decode_tile_count".to_string(), profile_snapshot.decode_tile_count),
        ("selected_sample_count".to_string(), profile_snapshot.selected_sample_count),
        ("metadata_slice_ns".to_string(), profile_snapshot.metadata_slice_ns),
        ("metadata_slice_count".to_string(), profile_snapshot.metadata_slice_count),
    ])
}

#[allow(clippy::missing_errors_doc)]
pub fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BgenReader>()?;
    module.add_class::<ChunkSpec>()?;
    module.add_class::<ChunkStats>()?;
    module.add_class::<NativeAlignedSampleData>()?;
    module.add_class::<OutputWriterSession>()?;
    module.add_class::<Regenie2RunEngine>()?;
    module.add_class::<RegeniePredictionSource>()?;
    module.add_class::<VariantMetadata>()?;
    module.add_function(wrap_pyfunction!(finalize_output_run_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(scan_committed_chunk_identifiers, module)?)?;
    module.add_function(wrap_pyfunction!(hello_from_bin, module)?)?;
    module.add_function(wrap_pyfunction!(plan_genotype_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(align_sample_data, module)?)?;
    module.add_function(wrap_pyfunction!(align_sample_data_from_sample_file, module)?)?;
    module.add_function(wrap_pyfunction!(convert_probability_tensor_to_dosage_f32, module)?)?;
    module.add_function(wrap_pyfunction!(convert_probability_matrix_to_dosage_f32, module)?)?;
    Ok(())
}
