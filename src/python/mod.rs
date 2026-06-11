use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::genotype::bgen::{BgenError, ReaderProfileSnapshot, set_bgen_decode_tile_variant_count};
use crate::genotype::common::{
    ChunkSpec as NativeChunkSpec, ChunkStats as NativeChunkStats, GenotypeError, VariantMetadataColumns,
};
use crate::genotype::planner;
use crate::genotype::preprocess;
use crate::pipeline::Regenie2RunEngineCore;
use crate::regenie::{MultiPredictionSource as NativeMultiPredictionSource, PredictionError, PredictionSource};
use crate::sample::{
    AlignedPhenotypeGroup, AlignedSampleData, AlignmentInputs, GroupedAlignedSampleData, MultiAlignedSampleData,
    MultiAlignmentInputs, SampleKeyMode,
};

mod config;
mod logging;
mod output;

use logging::{NativeTelemetrySession, emit_diagnostic_event, initialize_logging, shutdown_logging};
use output::{
    NativeInitializedOutputRun, NativeOutputRunPaths, NativePreparedOutputRun, OutputWriterSession,
    finalize_output_run_chunks, initialize_output_run, load_run_manifest_json, prepare_output_run,
    read_manifest_committed_chunk_identifiers, repair_strict_manifest_chunk_commits, resolve_output_run_paths,
    scan_committed_chunk_identifiers, validate_run_manifest_compatibility, validate_strict_manifest_chunks,
    write_regenie2_multi_native_chunk, write_regenie2_multi_native_chunk_f64, write_run_manifest_json,
};

type VariantMetadataTuple = (Vec<String>, Vec<String>, Vec<i64>, Vec<String>, Vec<String>);

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
    pub(crate) stats: Arc<NativeChunkStats>,
}

impl ChunkStats {
    fn new(stats: NativeChunkStats) -> Self {
        Self { stats: Arc::new(stats) }
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn summarize_variant_major_dosage_chunk_stats(
    genotype_matrix_by_variant: PyReadonlyArray2<'_, f32>,
) -> PyResult<ChunkStats> {
    let genotype_array = genotype_matrix_by_variant.as_array();
    let genotype_shape = genotype_array.shape();
    let selected_variant_count = genotype_shape[0];
    let selected_sample_count = genotype_shape[1];
    let genotype_values = genotype_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Variant-major genotype matrix must be C-contiguous."))?;
    let chunk_stats = preprocess::summarize_variant_major_dosage_matrix(
        genotype_values,
        selected_sample_count,
        selected_variant_count,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ChunkStats::new(chunk_stats))
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
    fn dosage_sum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.dosage_sum.as_ref().to_vec().into_pyarray(py)
    }

    #[getter]
    fn allele_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.dosage_sum.as_ref().to_vec().into_pyarray(py)
    }

    #[getter]
    fn has_missing_values(&self) -> bool {
        self.stats.has_missing_values
    }

    #[getter]
    fn dosage_square_sum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.dosage_square_sum.clone().into_pyarray(py)
    }

    #[getter]
    fn imputed_dosage_square_sum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.imputed_dosage_square_sum.clone().into_pyarray(py)
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

    #[pyo3(signature = (*, include_imputed_dosage_square_sum = true, include_sparse_firth_candidate = true))]
    fn compute_arrays<'py>(
        &self,
        py: Python<'py>,
        include_imputed_dosage_square_sum: bool,
        include_sparse_firth_candidate: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let compute_arrays = PyDict::new(py);
        compute_arrays.set_item("dosage_sum", self.stats.dosage_sum.as_ref().to_vec().into_pyarray(py))?;
        compute_arrays.set_item("observation_count", self.stats.observation_count.clone().into_pyarray(py))?;
        if include_imputed_dosage_square_sum {
            compute_arrays
                .set_item("imputed_dosage_square_sum", self.stats.imputed_dosage_square_sum.clone().into_pyarray(py))?;
        }
        if include_sparse_firth_candidate {
            compute_arrays.set_item(
                "is_rare_sparse_firth_candidate",
                self.stats.is_rare_sparse_firth_candidate.clone().into_pyarray(py),
            )?;
        }
        Ok(compute_arrays)
    }
}

#[pyclass]
pub(crate) struct VariantMetadata {
    pub(crate) variant_start_index: usize,
    pub(crate) variant_stop_index: usize,
    pub(crate) metadata: Arc<VariantMetadataColumns>,
}

#[pyclass]
pub(crate) struct NativeAlignedSampleData {
    data: AlignedSampleData,
}

#[pyclass]
pub(crate) struct NativeMultiAlignedSampleData {
    data: MultiAlignedSampleData,
}

#[pyclass]
pub(crate) struct NativeAlignedPhenotypeGroup {
    data: AlignedPhenotypeGroup,
}

#[pyclass]
pub(crate) struct NativeGroupedAlignedSampleData {
    data: GroupedAlignedSampleData,
}

impl VariantMetadata {
    fn new(variant_start_index: usize, variant_stop_index: usize, metadata: VariantMetadataColumns) -> Self {
        Self { variant_start_index, variant_stop_index, metadata: Arc::new(metadata) }
    }
}

impl NativeAlignedSampleData {
    fn new(data: AlignedSampleData) -> Self {
        Self { data }
    }
}

impl NativeMultiAlignedSampleData {
    fn new(data: MultiAlignedSampleData) -> Self {
        Self { data }
    }
}

impl NativeAlignedPhenotypeGroup {
    fn new(data: AlignedPhenotypeGroup) -> Self {
        Self { data }
    }
}

impl NativeGroupedAlignedSampleData {
    fn new(data: GroupedAlignedSampleData) -> Self {
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
impl NativeMultiAlignedSampleData {
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
    fn phenotype_names(&self) -> Vec<String> {
        self.data.phenotype_names.clone()
    }

    #[getter]
    fn phenotype_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let phenotype_matrix = Array2::from_shape_vec(
            (self.data.phenotype_row_count, self.data.phenotype_column_count),
            self.data.phenotype_matrix_values.clone(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(phenotype_matrix.into_pyarray(py))
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
impl NativeAlignedPhenotypeGroup {
    #[getter]
    fn phenotype_indices(&self) -> Vec<usize> {
        self.data.phenotype_indices.clone()
    }

    #[getter]
    fn aligned_sample_data(&self) -> NativeMultiAlignedSampleData {
        NativeMultiAlignedSampleData::new(self.data.aligned_sample_data.clone())
    }
}

#[pymethods]
impl NativeGroupedAlignedSampleData {
    #[getter]
    fn groups(&self, py: Python<'_>) -> PyResult<Vec<Py<NativeAlignedPhenotypeGroup>>> {
        self.data.groups.iter().cloned().map(|group| Py::new(py, NativeAlignedPhenotypeGroup::new(group))).collect()
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
    fn chromosome_label(&self) -> PyResult<String> {
        self.metadata
            .chromosome
            .first()
            .cloned()
            .ok_or_else(|| PyValueError::new_err("Variant metadata contains no chromosome labels."))
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

#[pyclass]
struct MultiRegeniePredictionSource {
    source: NativeMultiPredictionSource,
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
        .map_err(|error| convert_bgen_error("open_bgen", error))?;
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

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_path,
        phenotype_path,
        phenotype_name,
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=false,
        sample_key_mode="iid".to_string()
    ))]
    fn align_sample_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
        phenotype_path: String,
        phenotype_name: String,
        covariate_path: Option<String>,
        covariate_names: Option<Vec<String>>,
        is_binary_trait: bool,
        sample_key_mode: String,
    ) -> PyResult<NativeAlignedSampleData> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    crate::sample::align_sample_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                        phenotype_path,
                        phenotype_name,
                        covariate_path,
                        covariate_names,
                        is_binary_trait,
                        parsed_sample_key_mode,
                    )
                })
                .map(NativeAlignedSampleData::new)
                .map_err(PyValueError::new_err);
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let inputs = AlignmentInputs {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
            phenotype_path,
            phenotype_name,
            covariate_path,
            covariate_names,
            is_binary_trait,
            sample_key_mode: parsed_sample_key_mode,
        };
        py.detach(move || crate::sample::align_sample_data(inputs))
            .map(NativeAlignedSampleData::new)
            .map_err(PyValueError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_path,
        phenotype_path,
        phenotype_names,
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=false,
        sample_key_mode="iid".to_string()
    ))]
    fn align_multi_sample_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
        phenotype_path: String,
        phenotype_names: Vec<String>,
        covariate_path: Option<String>,
        covariate_names: Option<Vec<String>>,
        is_binary_trait: bool,
        sample_key_mode: String,
    ) -> PyResult<NativeMultiAlignedSampleData> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    crate::sample::align_multi_sample_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                        phenotype_path,
                        phenotype_names,
                        covariate_path,
                        covariate_names,
                        is_binary_trait,
                        parsed_sample_key_mode,
                    )
                })
                .map(NativeMultiAlignedSampleData::new)
                .map_err(PyValueError::new_err);
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let inputs = MultiAlignmentInputs {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            sample_key_mode: parsed_sample_key_mode,
        };
        py.detach(move || crate::sample::align_multi_sample_data(inputs))
            .map(NativeMultiAlignedSampleData::new)
            .map_err(PyValueError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_path,
        phenotype_path,
        phenotype_names,
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=false,
        sample_key_mode="iid".to_string()
    ))]
    fn align_grouped_sample_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
        phenotype_path: String,
        phenotype_names: Vec<String>,
        covariate_path: Option<String>,
        covariate_names: Option<Vec<String>>,
        is_binary_trait: bool,
        sample_key_mode: String,
    ) -> PyResult<NativeGroupedAlignedSampleData> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    crate::sample::align_grouped_sample_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                        phenotype_path,
                        phenotype_names,
                        covariate_path,
                        covariate_names,
                        is_binary_trait,
                        parsed_sample_key_mode,
                    )
                })
                .map(NativeGroupedAlignedSampleData::new)
                .map_err(PyValueError::new_err);
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let inputs = MultiAlignmentInputs {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            sample_key_mode: parsed_sample_key_mode,
        };
        py.detach(move || crate::sample::align_grouped_sample_data(&inputs))
            .map(NativeGroupedAlignedSampleData::new)
            .map_err(PyValueError::new_err)
    }

    fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.engine.reader().chromosome_boundary_indices()
    }

    fn variant_metadata_slice(&self, variant_start: usize, variant_stop: usize) -> PyResult<VariantMetadataTuple> {
        self.engine
            .reader()
            .variant_metadata_slice(variant_start, variant_stop)
            .map(convert_variant_metadata_columns_to_tuple)
            .map_err(|error| convert_bgen_error("read_variant_metadata_slice", error))
    }

    #[pyo3(signature = (variant_limit=None))]
    fn required_chromosomes(&self, variant_limit: Option<usize>) -> PyResult<Vec<String>> {
        let variant_count = self.engine.reader().variant_count();
        let scanned_variant_count = variant_limit.map_or(variant_count, |limit| limit.min(variant_count));
        if scanned_variant_count == 0 {
            return Ok(Vec::new());
        }

        let mut chromosome_labels = Vec::new();
        for chromosome_boundaries in self.engine.reader().chromosome_boundary_indices().windows(2) {
            let chromosome_start_index = chromosome_boundaries[0];
            let chromosome_stop_index = chromosome_boundaries[1].min(scanned_variant_count);
            if chromosome_start_index >= chromosome_stop_index {
                continue;
            }
            let metadata = self
                .engine
                .reader()
                .variant_metadata_slice(chromosome_start_index, chromosome_start_index + 1)
                .map_err(|error| convert_bgen_error("read_variant_metadata_slice", error))?;
            let chromosome_label = metadata.chromosome.into_iter().next().ok_or_else(|| {
                PyRuntimeError::new_err("Chromosome boundary metadata contained no chromosome label.")
            })?;
            chromosome_labels.push(chromosome_label);
        }
        Ok(chromosome_labels)
    }

    fn reset_profile(&self) {
        self.engine.reader().reset_profile();
    }

    fn profile_snapshot(&self) -> HashMap<String, u64> {
        build_profile_snapshot_dict(&self.engine.reader().profile_snapshot())
    }

    fn validate_trusted_no_missing_diploid(&self) -> PyResult<()> {
        self.engine
            .reader()
            .validate_trusted_no_missing_diploid()
            .map_err(|error| convert_bgen_error("validate_trusted_no_missing_diploid", error))
    }

    fn mark_trusted_no_missing_diploid_validated(&self) -> PyResult<()> {
        self.engine
            .reader()
            .mark_trusted_no_missing_diploid_validated()
            .map_err(|error| convert_bgen_error("mark_trusted_no_missing_diploid_validated", error))
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
        self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
            py,
            &sample_index_values,
            callback,
            committed_chunk_identifiers,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_dosage_buffered_chunks_for_native_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeMultiAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (sample_indices, callback, committed_chunk_identifiers=None))]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let sample_index_values = sample_indices.as_slice()?.to_vec();
        self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
            py,
            &sample_index_values,
            callback,
            committed_chunk_identifiers,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_multi_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeMultiAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
        )
    }
}

#[pymethods]
impl RegeniePredictionSource {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        phenotype_name,
        sample_family_identifiers,
        sample_individual_identifiers,
        sample_key_mode="iid".to_string()
    ))]
    fn new(
        prediction_list_path: String,
        phenotype_name: String,
        sample_family_identifiers: Vec<String>,
        sample_individual_identifiers: Vec<String>,
        sample_key_mode: String,
    ) -> PyResult<Self> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let source = PredictionSource::load(
            Path::new(&prediction_list_path),
            &phenotype_name,
            &sample_family_identifiers,
            &sample_individual_identifiers,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_prediction_source", &error))?;
        Ok(Self { source })
    }

    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        phenotype_name,
        aligned_sample_data,
        sample_key_mode="iid".to_string()
    ))]
    fn from_native_aligned_sample_data(
        prediction_list_path: String,
        phenotype_name: String,
        aligned_sample_data: PyRef<'_, NativeAlignedSampleData>,
        sample_key_mode: String,
    ) -> PyResult<Self> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let source = PredictionSource::load(
            Path::new(&prediction_list_path),
            &phenotype_name,
            &aligned_sample_data.data.family_identifiers,
            &aligned_sample_data.data.individual_identifiers,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_prediction_source_from_native_aligned_sample_data", &error))?;
        Ok(Self { source })
    }

    #[allow(clippy::needless_pass_by_value)]
    fn get_chromosome_predictions<'py>(
        &self,
        py: Python<'py>,
        chromosome: String,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let prediction_values = self
            .source
            .chromosome_predictions(&chromosome)
            .map_err(|error| convert_prediction_error("chromosome_predictions", &error))?;
        Ok(Array1::from_vec(prediction_values.to_vec()).into_pyarray(py))
    }
}

#[pymethods]
impl MultiRegeniePredictionSource {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        phenotype_names,
        sample_family_identifiers,
        sample_individual_identifiers,
        sample_key_mode="iid".to_string()
    ))]
    fn new(
        prediction_list_path: String,
        phenotype_names: Vec<String>,
        sample_family_identifiers: Vec<String>,
        sample_individual_identifiers: Vec<String>,
        sample_key_mode: String,
    ) -> PyResult<Self> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let source = NativeMultiPredictionSource::load(
            Path::new(&prediction_list_path),
            &phenotype_names,
            &sample_family_identifiers,
            &sample_individual_identifiers,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_multi_prediction_source", &error))?;
        Ok(Self { source })
    }

    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        aligned_sample_data,
        sample_key_mode="iid".to_string()
    ))]
    fn from_native_multi_aligned_sample_data(
        prediction_list_path: String,
        aligned_sample_data: PyRef<'_, NativeMultiAlignedSampleData>,
        sample_key_mode: String,
    ) -> PyResult<Self> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let source = NativeMultiPredictionSource::load(
            Path::new(&prediction_list_path),
            &aligned_sample_data.data.phenotype_names,
            &aligned_sample_data.data.family_identifiers,
            &aligned_sample_data.data.individual_identifiers,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_multi_prediction_source_from_aligned_samples", &error))?;
        Ok(Self { source })
    }

    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        grouped_aligned_sample_data,
        sample_key_mode="iid".to_string()
    ))]
    fn from_native_grouped_aligned_sample_data(
        prediction_list_path: String,
        grouped_aligned_sample_data: PyRef<'_, NativeGroupedAlignedSampleData>,
        sample_key_mode: String,
    ) -> PyResult<Vec<Self>> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let aligned_sample_data_groups =
            grouped_aligned_sample_data.data.groups.iter().map(|group| &group.aligned_sample_data).collect::<Vec<_>>();
        let sources = NativeMultiPredictionSource::load_grouped(
            Path::new(&prediction_list_path),
            &aligned_sample_data_groups,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_multi_prediction_source_from_grouped_samples", &error))?;
        Ok(sources.into_iter().map(|source| Self { source }).collect())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn get_chromosome_predictions<'py>(
        &self,
        py: Python<'py>,
        chromosome: String,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (trait_count, sample_count, prediction_values) = self
            .source
            .chromosome_prediction_matrix(&chromosome)
            .map_err(|error| convert_prediction_error("chromosome_prediction_matrix", &error))?;
        let prediction_matrix = Array2::from_shape_vec((trait_count, sample_count), prediction_values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(prediction_matrix.into_pyarray(py))
    }
}

impl Regenie2RunEngine {
    fn run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.engine
            .reader()
            .prepare_sample_selection(sample_index_values)
            .map_err(|error| convert_bgen_error("prepare_sample_selection", error))?;

        let run_result = self.run_prepared_bgen_variant_major_dosage_buffered_chunks(
            py,
            sample_index_values.len(),
            callback,
            committed_chunk_identifiers,
        );
        let clear_result = self
            .engine
            .reader()
            .clear_prepared_sample_selection()
            .map_err(|error| convert_bgen_error("clear_prepared_sample_selection", error));
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }

    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.engine
            .reader()
            .prepare_sample_selection(sample_index_values)
            .map_err(|error| convert_bgen_error("prepare_sample_selection", error))?;

        let run_result = self.run_prepared_bgen_variant_major_packed8_probability_pair_buffered_chunks(
            py,
            sample_index_values.len(),
            callback,
            committed_chunk_identifiers,
        );
        let clear_result = self
            .engine
            .reader()
            .clear_prepared_sample_selection()
            .map_err(|error| convert_bgen_error("clear_prepared_sample_selection", error));
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }

    fn run_prepared_bgen_variant_major_dosage_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        selected_sample_count: usize,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self
            .engine
            .plan_chunks(&committed_identifier_set)
            .map_err(|error| convert_genotype_error("plan_chunks", error))?;
        for chunk_spec in &chunk_specs {
            py.check_signals()?;
            let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
            let output_array_object = callback
                .call_method1("acquire_variant_major_dosage_buffer", (selected_variant_count, selected_sample_count))?;
            let stats = {
                let mut output_array = output_array_object.extract::<PyReadwriteArray2<'_, f32>>()?;
                let output_shape = output_array.shape();
                if output_shape != [selected_variant_count, selected_sample_count] {
                    return Err(PyValueError::new_err(format!(
                        "Reusable variant-major BGEN dosage buffer shape mismatch: expected ({selected_variant_count}, {}), observed ({}, {}).",
                        selected_sample_count, output_shape[0], output_shape[1],
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
                    .map_err(|error| {
                        convert_bgen_error("read_preprocessed_variant_major_dosage_f32_into_address_prepared", error)
                    })?;
                Py::new(py, ChunkStats::new(chunk_stats))?
            };
            let metadata_columns = self
                .engine
                .reader()
                .variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)
                .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
            let metadata = Py::new(
                py,
                VariantMetadata::new(chunk_spec.variant_start_index, chunk_spec.variant_stop_index, metadata_columns),
            )?;
            callback.call_method1(
                "compute_preprocessed_variant_major_dosage_chunk",
                (metadata, output_array_object, stats),
            )?;
        }
        Ok(chunk_specs.len())
    }

    fn run_prepared_bgen_variant_major_packed8_probability_pair_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        selected_sample_count: usize,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self
            .engine
            .plan_chunks(&committed_identifier_set)
            .map_err(|error| convert_genotype_error("plan_chunks", error))?;
        for chunk_spec in &chunk_specs {
            py.check_signals()?;
            let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
            let output_array_object = callback.call_method1(
                "acquire_variant_major_packed8_probability_pair_buffer",
                (selected_variant_count, selected_sample_count),
            )?;
            let stats = {
                let mut output_array = output_array_object.extract::<PyReadwriteArray3<'_, u8>>()?;
                let output_shape = output_array.shape();
                if output_shape != [selected_variant_count, selected_sample_count, 2] {
                    return Err(PyValueError::new_err(format!(
                        "Reusable variant-major BGEN packed8 probability-pair buffer shape mismatch: expected ({selected_variant_count}, {}, 2), observed ({}, {}, {}).",
                        selected_sample_count, output_shape[0], output_shape[1], output_shape[2],
                    )));
                }
                if !output_array.is_c_contiguous() {
                    return Err(PyValueError::new_err(
                        "Reusable variant-major BGEN packed8 probability-pair buffer must be C-contiguous uint8.",
                    ));
                }
                let output_slice = output_array.as_slice_mut().map_err(|_| {
                    PyValueError::new_err(
                        "Reusable variant-major BGEN packed8 probability-pair buffer must expose a contiguous mutable slice.",
                    )
                })?;
                let output_pointer_address = output_slice.as_mut_ptr() as usize;
                let output_value_count = output_slice.len();
                let chunk_stats = py
                    .detach(|| {
                        self.engine
                            .reader()
                            .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
                                chunk_spec.variant_start_index,
                                chunk_spec.variant_stop_index,
                                output_pointer_address,
                                output_value_count,
                            )
                    })
                    .map_err(|error| {
                        convert_bgen_error(
                            "read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared",
                            error,
                        )
                    })?;
                Py::new(py, ChunkStats::new(chunk_stats))?
            };
            let metadata_columns = self
                .engine
                .reader()
                .variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)
                .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
            let metadata = Py::new(
                py,
                VariantMetadata::new(chunk_spec.variant_start_index, chunk_spec.variant_stop_index, metadata_columns),
            )?;
            callback.call_method1(
                "compute_preprocessed_variant_major_packed8_probability_pair_chunk",
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
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
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
    sample_key_mode: String,
) -> PyResult<NativeAlignedSampleData> {
    let sample_index_values = sample_indices.as_slice()?.to_vec();
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    let inputs = AlignmentInputs {
        sample_indices: sample_index_values,
        family_identifiers,
        individual_identifiers,
        phenotype_path,
        phenotype_name,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parsed_sample_key_mode,
    };
    py.detach(|| crate::sample::align_sample_data(inputs))
        .map(NativeAlignedSampleData::new)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    sample_indices,
    family_identifiers,
    individual_identifiers,
    phenotype_path,
    phenotype_names,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
))]
fn align_multi_sample_data<'py>(
    py: Python<'py>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    family_identifiers: Vec<String>,
    individual_identifiers: Vec<String>,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: String,
) -> PyResult<NativeMultiAlignedSampleData> {
    let sample_index_values = sample_indices.as_slice()?.to_vec();
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    let inputs = MultiAlignmentInputs {
        sample_indices: sample_index_values,
        family_identifiers,
        individual_identifiers,
        phenotype_path,
        phenotype_names,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parsed_sample_key_mode,
    };
    py.detach(|| crate::sample::align_multi_sample_data(inputs))
        .map(NativeMultiAlignedSampleData::new)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    sample_indices,
    family_identifiers,
    individual_identifiers,
    phenotype_path,
    phenotype_names,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
))]
fn align_grouped_sample_data<'py>(
    py: Python<'py>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    family_identifiers: Vec<String>,
    individual_identifiers: Vec<String>,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: String,
) -> PyResult<NativeGroupedAlignedSampleData> {
    let sample_index_values = sample_indices.as_slice()?.to_vec();
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    let inputs = MultiAlignmentInputs {
        sample_indices: sample_index_values,
        family_identifiers,
        individual_identifiers,
        phenotype_path,
        phenotype_names,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parsed_sample_key_mode,
    };
    py.detach(|| crate::sample::align_grouped_sample_data(&inputs))
        .map(NativeGroupedAlignedSampleData::new)
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
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
))]
fn align_sample_data_from_sample_file(
    py: Python<'_>,
    sample_path: String,
    expected_sample_count: usize,
    phenotype_path: String,
    phenotype_name: String,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: String,
) -> PyResult<NativeAlignedSampleData> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    py.detach(move || {
        crate::sample::align_sample_data_from_sample_file(
            Path::new(&sample_path),
            expected_sample_count,
            phenotype_path,
            phenotype_name,
            covariate_path,
            covariate_names,
            is_binary_trait,
            parsed_sample_key_mode,
        )
    })
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
    phenotype_names,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
))]
fn align_multi_sample_data_from_sample_file(
    py: Python<'_>,
    sample_path: String,
    expected_sample_count: usize,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: String,
) -> PyResult<NativeMultiAlignedSampleData> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    py.detach(move || {
        crate::sample::align_multi_sample_data_from_sample_file(
            Path::new(&sample_path),
            expected_sample_count,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            parsed_sample_key_mode,
        )
    })
    .map(NativeMultiAlignedSampleData::new)
    .map_err(PyValueError::new_err)
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
    .map_err(|error| convert_genotype_error("plan_chromosome_homogeneous_chunks", error))?;
    Ok(chunk_specs.into_iter().map(|chunk_spec| ChunkSpec { chunk_spec }).collect())
}

fn parse_sample_key_mode(sample_key_mode: &str) -> PyResult<SampleKeyMode> {
    match sample_key_mode {
        "iid" => Ok(SampleKeyMode::Iid),
        "fid_iid" => Ok(SampleKeyMode::FidIid),
        _ => Err(PyValueError::new_err(format!(
            "sample_key_mode must be 'iid' or 'fid_iid', found '{sample_key_mode}'."
        ))),
    }
}

fn convert_bgen_error(operation: &str, error: BgenError) -> PyErr {
    let (error_class, message) = match &error {
        BgenError::InvalidFormat(message) | BgenError::UnsupportedFormat(message) | BgenError::Range(message) => {
            ("bgen_input", message.clone())
        }
        BgenError::Io(io_error) => ("bgen_io", io_error.to_string()),
    };
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "bgen",
        operation = operation,
        error_class = error_class,
        error_message = %message,
        "Converting Rust BGEN error to Python."
    );
    match error {
        BgenError::InvalidFormat(message) | BgenError::UnsupportedFormat(message) | BgenError::Range(message) => {
            PyValueError::new_err(message)
        }
        BgenError::Io(io_error) => PyRuntimeError::new_err(io_error.to_string()),
    }
}

fn convert_genotype_error(operation: &str, error: GenotypeError) -> PyErr {
    let (error_class, message) = match &error {
        GenotypeError::InvalidInput(message) => ("genotype_input", message.clone()),
        GenotypeError::Reader(message) => ("genotype_reader", message.clone()),
    };
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "genotype",
        operation = operation,
        error_class = error_class,
        error_message = %message,
        "Converting Rust genotype error to Python."
    );
    match error {
        GenotypeError::InvalidInput(message) => PyValueError::new_err(message),
        GenotypeError::Reader(message) => PyRuntimeError::new_err(message),
    }
}

fn convert_prediction_error(operation: &str, error: &PredictionError) -> PyErr {
    let error_message = match error {
        PredictionError::PredictionListNotFound(path) => {
            let message = format!("Prediction list file not found: {}", path.display());
            tracing::warn!(
                target: "g.python",
                g_event = "native_boundary_error",
                subsystem = "prediction",
                operation = operation,
                error_class = "prediction_list_not_found",
                error_message = %message,
                "Converting Rust prediction error to Python."
            );
            return pyo3::exceptions::PyFileNotFoundError::new_err(message);
        }
        PredictionError::LocoFileNotFound(path) => {
            let message = format!("LOCO file not found: {}", path.display());
            tracing::warn!(
                target: "g.python",
                g_event = "native_boundary_error",
                subsystem = "prediction",
                operation = operation,
                error_class = "loco_file_not_found",
                error_message = %message,
                "Converting Rust prediction error to Python."
            );
            return pyo3::exceptions::PyFileNotFoundError::new_err(message);
        }
        PredictionError::Io(io_error) => {
            let message = io_error.to_string();
            tracing::warn!(
                target: "g.python",
                g_event = "native_boundary_error",
                subsystem = "prediction",
                operation = operation,
                error_class = "prediction_io",
                error_message = %message,
                "Converting Rust prediction error to Python."
            );
            return PyRuntimeError::new_err(message);
        }
        other_error => other_error.to_string(),
    };
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "prediction",
        operation = operation,
        error_class = "prediction_error",
        error_message = %error_message,
        "Converting Rust prediction error to Python."
    );
    PyValueError::new_err(error_message)
}

fn build_committed_identifier_set(committed_chunk_identifiers: Option<Vec<usize>>) -> BTreeSet<usize> {
    committed_chunk_identifiers.unwrap_or_default().into_iter().collect()
}

fn convert_variant_metadata_columns_to_tuple(variant_metadata: VariantMetadataColumns) -> VariantMetadataTuple {
    (
        variant_metadata.chromosome,
        variant_metadata.variant_identifier,
        variant_metadata.position,
        variant_metadata.allele_one,
        variant_metadata.allele_two,
    )
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

#[pyfunction]
#[allow(clippy::missing_errors_doc)]
fn configure_bgen_decode_tile_variant_count(tile_variant_count: usize) -> PyResult<()> {
    set_bgen_decode_tile_variant_count(tile_variant_count)
        .map_err(|error| convert_bgen_error("configure_bgen_decode_tile_variant_count", error))
}

#[pyfunction]
#[allow(clippy::missing_errors_doc)]
fn configure_rayon_global_thread_pool(thread_count: usize) -> PyResult<()> {
    if thread_count == 0 {
        return Err(PyValueError::new_err("Rayon thread count must be positive."));
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build_global()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[allow(clippy::missing_errors_doc)]
pub fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    config::register_module(module)?;
    module.add_class::<ChunkSpec>()?;
    module.add_class::<ChunkStats>()?;
    module.add_class::<NativeAlignedPhenotypeGroup>()?;
    module.add_class::<NativeAlignedSampleData>()?;
    module.add_class::<NativeGroupedAlignedSampleData>()?;
    module.add_class::<NativeInitializedOutputRun>()?;
    module.add_class::<NativeMultiAlignedSampleData>()?;
    module.add_class::<NativeOutputRunPaths>()?;
    module.add_class::<NativePreparedOutputRun>()?;
    module.add_class::<OutputWriterSession>()?;
    module.add_class::<Regenie2RunEngine>()?;
    module.add_class::<RegeniePredictionSource>()?;
    module.add_class::<MultiRegeniePredictionSource>()?;
    module.add_class::<NativeTelemetrySession>()?;
    module.add_class::<VariantMetadata>()?;
    module.add_function(wrap_pyfunction!(finalize_output_run_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(initialize_output_run, module)?)?;
    module.add_function(wrap_pyfunction!(load_run_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_output_run, module)?)?;
    module.add_function(wrap_pyfunction!(read_manifest_committed_chunk_identifiers, module)?)?;
    module.add_function(wrap_pyfunction!(repair_strict_manifest_chunk_commits, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_output_run_paths, module)?)?;
    module.add_function(wrap_pyfunction!(scan_committed_chunk_identifiers, module)?)?;
    module.add_function(wrap_pyfunction!(summarize_variant_major_dosage_chunk_stats, module)?)?;
    module.add_function(wrap_pyfunction!(validate_run_manifest_compatibility, module)?)?;
    module.add_function(wrap_pyfunction!(validate_strict_manifest_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(write_regenie2_multi_native_chunk, module)?)?;
    module.add_function(wrap_pyfunction!(write_regenie2_multi_native_chunk_f64, module)?)?;
    module.add_function(wrap_pyfunction!(write_run_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(configure_bgen_decode_tile_variant_count, module)?)?;
    module.add_function(wrap_pyfunction!(configure_rayon_global_thread_pool, module)?)?;
    module.add_function(wrap_pyfunction!(emit_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(initialize_logging, module)?)?;
    module.add_function(wrap_pyfunction!(shutdown_logging, module)?)?;
    module.add_function(wrap_pyfunction!(plan_genotype_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(align_sample_data, module)?)?;
    module.add_function(wrap_pyfunction!(align_grouped_sample_data, module)?)?;
    module.add_function(wrap_pyfunction!(align_multi_sample_data, module)?)?;
    module.add_function(wrap_pyfunction!(align_sample_data_from_sample_file, module)?)?;
    module.add_function(wrap_pyfunction!(align_multi_sample_data_from_sample_file, module)?)?;
    Ok(())
}
