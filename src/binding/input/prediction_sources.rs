//! PyO3 adapters for REGENIE prediction source loading.

use std::path::Path;

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_input::{MultiPredictionSource as NativeMultiPredictionSource, PredictionSource};

use super::{
    errors::convert_prediction_error,
    sample_alignment::{
        NativeAlignedSampleData, NativeGroupedAlignedSampleData, NativeMultiAlignedSampleData, parse_sample_key_mode,
    },
};

#[pyclass]
pub(crate) struct RegeniePredictionSource {
    pub(crate) source: PredictionSource,
}

#[pyclass]
struct MultiRegeniePredictionSource {
    source: NativeMultiPredictionSource,
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
        let prediction_matrix_data = self
            .source
            .chromosome_prediction_matrix(&chromosome)
            .map_err(|error| convert_prediction_error("chromosome_prediction_matrix", &error))?;
        let prediction_matrix = Array2::from_shape_vec(
            (prediction_matrix_data.trait_count, prediction_matrix_data.sample_count),
            prediction_matrix_data.prediction_values,
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(prediction_matrix.into_pyarray(py))
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RegeniePredictionSource>()?;
    module.add_class::<MultiRegeniePredictionSource>()?;
    Ok(())
}

pub(crate) fn load_regenie_prediction_source_from_aligned_sample_data(
    prediction_list_path: &str,
    phenotype_name: &str,
    aligned_sample_data: &NativeAlignedSampleData,
    sample_key_mode: &str,
) -> PyResult<RegeniePredictionSource> {
    let parsed_sample_key_mode = parse_sample_key_mode(sample_key_mode)?;
    let source = PredictionSource::load(
        Path::new(prediction_list_path),
        phenotype_name,
        &aligned_sample_data.data.family_identifiers,
        &aligned_sample_data.data.individual_identifiers,
        parsed_sample_key_mode,
    )
    .map_err(|error| convert_prediction_error("load_prediction_source_from_native_pipeline_bundle", &error))?;
    Ok(RegeniePredictionSource { source })
}
