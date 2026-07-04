//! PyO3 adapters for sample alignment and phenotype compute groups.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_input::sample::{
    self, AlignedPhenotypeGroup, AlignedSampleData, GroupedAlignedSampleData, MultiAlignedSampleData,
    ResolvedPhenotypeComputeGroup, SampleKeyMode,
};

#[pyclass]
pub(crate) struct NativeAlignedSampleData {
    pub(crate) data: AlignedSampleData,
}

#[pyclass]
pub(crate) struct NativeMultiAlignedSampleData {
    pub(crate) data: MultiAlignedSampleData,
}

#[pyclass]
pub(crate) struct NativeAlignedPhenotypeGroup {
    pub(crate) data: AlignedPhenotypeGroup,
}

#[pyclass]
pub(crate) struct NativeGroupedAlignedSampleData {
    pub(crate) data: GroupedAlignedSampleData,
}

#[pyclass]
pub(crate) struct NativeResolvedPhenotypeComputeGroup {
    pub(crate) data: ResolvedPhenotypeComputeGroup,
}

impl NativeAlignedSampleData {
    pub(crate) fn new(data: AlignedSampleData) -> Self {
        Self { data }
    }
}

impl NativeMultiAlignedSampleData {
    pub(crate) fn new(data: MultiAlignedSampleData) -> Self {
        Self { data }
    }
}

impl NativeAlignedPhenotypeGroup {
    pub(crate) fn new(data: AlignedPhenotypeGroup) -> Self {
        Self { data }
    }
}

impl NativeGroupedAlignedSampleData {
    pub(crate) fn new(data: GroupedAlignedSampleData) -> Self {
        Self { data }
    }
}

impl NativeResolvedPhenotypeComputeGroup {
    pub(crate) fn new(data: ResolvedPhenotypeComputeGroup) -> Self {
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
impl NativeResolvedPhenotypeComputeGroup {
    #[getter]
    fn group_mode(&self) -> String {
        self.data.group_mode.clone()
    }

    #[getter]
    fn phenotype_indices(&self) -> Vec<usize> {
        self.data.phenotype_indices.clone()
    }

    #[getter]
    fn phenotype_names(&self) -> Vec<String> {
        self.data.phenotype_names.clone()
    }

    #[getter]
    fn sample_mode(&self) -> String {
        self.data.sample_mode.clone()
    }

    #[getter]
    fn sample_set_fingerprint(&self) -> String {
        self.data.sample_set_fingerprint.clone()
    }

    #[getter]
    fn covariate_design_fingerprint(&self) -> String {
        self.data.covariate_design_fingerprint.clone()
    }

    #[getter]
    fn prediction_alignment_fingerprint(&self) -> Option<String> {
        self.data.prediction_alignment_fingerprint.clone()
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn resolve_single_phenotype_compute_group(
    aligned_sample_data: PyRef<'_, NativeAlignedSampleData>,
    phenotype_name: String,
    prediction_list_path: Option<String>,
    sample_key_mode: String,
) -> PyResult<NativeResolvedPhenotypeComputeGroup> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    Ok(NativeResolvedPhenotypeComputeGroup::new(sample::resolve_single_phenotype_compute_group(
        &aligned_sample_data.data,
        phenotype_name,
        prediction_list_path.as_deref(),
        parsed_sample_key_mode,
    )))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn resolve_per_phenotype_compute_group(
    aligned_sample_data: PyRef<'_, NativeMultiAlignedSampleData>,
    phenotype_indices: Vec<usize>,
    phenotype_names: Vec<String>,
    prediction_list_path: Option<String>,
    sample_key_mode: String,
) -> PyResult<NativeResolvedPhenotypeComputeGroup> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    Ok(NativeResolvedPhenotypeComputeGroup::new(sample::resolve_per_phenotype_compute_group(
        &aligned_sample_data.data,
        phenotype_indices,
        phenotype_names,
        prediction_list_path.as_deref(),
        parsed_sample_key_mode,
    )))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn resolve_complete_case_compute_group(
    aligned_sample_data: PyRef<'_, NativeMultiAlignedSampleData>,
    phenotype_indices: Vec<usize>,
    phenotype_names: Vec<String>,
    prediction_list_path: Option<String>,
    sample_key_mode: String,
) -> PyResult<NativeResolvedPhenotypeComputeGroup> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    Ok(NativeResolvedPhenotypeComputeGroup::new(sample::resolve_complete_case_compute_group(
        &aligned_sample_data.data,
        phenotype_indices,
        phenotype_names,
        prediction_list_path.as_deref(),
        parsed_sample_key_mode,
    )))
}

pub(crate) fn parse_sample_key_mode(sample_key_mode: &str) -> PyResult<SampleKeyMode> {
    match sample_key_mode {
        "iid" => Ok(SampleKeyMode::Iid),
        "fid_iid" => Ok(SampleKeyMode::FidIid),
        _ => Err(PyValueError::new_err(format!(
            "sample_key_mode must be 'iid' or 'fid_iid', found '{sample_key_mode}'."
        ))),
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeAlignedPhenotypeGroup>()?;
    module.add_class::<NativeAlignedSampleData>()?;
    module.add_class::<NativeGroupedAlignedSampleData>()?;
    module.add_class::<NativeMultiAlignedSampleData>()?;
    module.add_class::<NativeResolvedPhenotypeComputeGroup>()?;
    module.add_function(wrap_pyfunction!(resolve_complete_case_compute_group, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_per_phenotype_compute_group, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_single_phenotype_compute_group, module)?)?;
    Ok(())
}
