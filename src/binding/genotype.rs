//! PyO3 adapters for genotype chunk metadata and statistics.

use std::collections::BTreeSet;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use g_genotype::internal as native_genotype;
use g_genotype::{ChunkStats as NativeChunkStats, VariantMetadataColumns};

pub(crate) type VariantMetadataTuple = (Vec<String>, Vec<String>, Vec<i64>, Vec<String>, Vec<String>);

#[pyclass]
pub(crate) struct ChunkStats {
    pub(crate) stats: Arc<NativeChunkStats>,
}

impl ChunkStats {
    pub(crate) fn new(stats: NativeChunkStats) -> Self {
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
    let chunk_stats = native_genotype::summarize_variant_major_dosage_matrix(
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

impl VariantMetadata {
    pub(crate) fn new(variant_start_index: usize, variant_stop_index: usize, metadata: VariantMetadataColumns) -> Self {
        Self { variant_start_index, variant_stop_index, metadata: Arc::new(metadata) }
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

pub(crate) fn convert_variant_metadata_columns_to_tuple(
    variant_metadata: VariantMetadataColumns,
) -> VariantMetadataTuple {
    (
        variant_metadata.chromosome,
        variant_metadata.variant_identifier,
        variant_metadata.position,
        variant_metadata.allele_one,
        variant_metadata.allele_two,
    )
}

pub(crate) fn build_committed_identifier_set(committed_chunk_identifiers: Option<Vec<usize>>) -> BTreeSet<usize> {
    committed_chunk_identifiers.unwrap_or_default().into_iter().collect()
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ChunkStats>()?;
    module.add_class::<VariantMetadata>()?;
    module.add_function(wrap_pyfunction!(summarize_variant_major_dosage_chunk_stats, module)?)?;
    Ok(())
}
