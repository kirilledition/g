use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::genotype::bgen::{BgenError, ReaderProfileSnapshot, VariantMetadataLists};
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
struct Regenie2RunEngine {
    engine: Regenie2RunEngineCore,
}

#[pyclass]
struct RegeniePredictionSource {
    source: PredictionSource,
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
    Ok(())
}
