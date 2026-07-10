//! Native input loading at the Python error and GIL boundary.

use std::path::Path;

use g_engine::Regenie2RunEngineCore;
use g_input::{
    AlignedPhenotypeGroup, AlignedSampleData, AlignmentInputs, MultiAlignedSampleData, MultiAlignmentInputs,
    MultiPredictionSource, PredictionSource, SampleIdentifierData, SampleKeyMode,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::errors::{convert_input_error, convert_prediction_error};

pub(crate) fn parse_sample_key_mode(sample_key_mode: &str) -> PyResult<SampleKeyMode> {
    SampleKeyMode::from_str_value(sample_key_mode).ok_or_else(|| {
        PyValueError::new_err(format!(
            "sample_key_mode must be one of {}, found '{sample_key_mode}'.",
            SampleKeyMode::accepted_values().join(", "),
        ))
    })
}

pub(crate) fn load_prediction_source(
    prediction_list_path: &str,
    phenotype_name: &str,
    aligned_sample_data: &AlignedSampleData,
    sample_key_mode: &str,
) -> PyResult<PredictionSource> {
    PredictionSource::load(
        Path::new(prediction_list_path),
        phenotype_name,
        &aligned_sample_data.family_identifiers,
        &aligned_sample_data.individual_identifiers,
        parse_sample_key_mode(sample_key_mode)?,
    )
    .map_err(|error| convert_prediction_error("load_prediction_source", &error))
}

pub(crate) fn load_multi_prediction_source(
    prediction_list_path: &str,
    aligned_sample_data: &MultiAlignedSampleData,
    sample_key_mode: &str,
) -> PyResult<MultiPredictionSource> {
    MultiPredictionSource::load(
        Path::new(prediction_list_path),
        &aligned_sample_data.phenotype_names,
        &aligned_sample_data.family_identifiers,
        &aligned_sample_data.individual_identifiers,
        parse_sample_key_mode(sample_key_mode)?,
    )
    .map_err(|error| convert_prediction_error("load_multi_prediction_source", &error))
}

pub(crate) fn load_grouped_prediction_sources(
    prediction_list_path: &str,
    grouped_aligned_sample_data: &[AlignedPhenotypeGroup],
    sample_key_mode: &str,
) -> PyResult<Vec<MultiPredictionSource>> {
    let aligned_sample_data_groups =
        grouped_aligned_sample_data.iter().map(|group| &group.aligned_sample_data).collect::<Vec<_>>();
    MultiPredictionSource::load_grouped(
        Path::new(prediction_list_path),
        &aligned_sample_data_groups,
        parse_sample_key_mode(sample_key_mode)?,
    )
    .map_err(|error| convert_prediction_error("load_grouped_prediction_sources", &error))
}

fn sample_identifier_data_for_engine(
    engine: &Regenie2RunEngineCore,
    py: Python<'_>,
    sample_path: Option<String>,
) -> PyResult<SampleIdentifierData> {
    if let Some(sample_path) = sample_path {
        let expected_sample_count = engine.reader().sample_count();
        return py
            .detach(move || {
                g_input::load_sample_identifier_data_from_sample_file(Path::new(&sample_path), expected_sample_count)
            })
            .map_err(|error| convert_input_error("load_sample_identifier_data_from_sample_file", error));
    }
    if !engine.reader().contains_embedded_samples() {
        return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
    }
    let sample_identifiers = engine.reader().sample_identifiers();
    let sample_indices = (0..sample_identifiers.len()).collect::<Vec<_>>();
    Ok(SampleIdentifierData {
        sample_indices,
        family_identifiers: sample_identifiers.clone(),
        individual_identifiers: sample_identifiers,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn align_sample_data_for_engine(
    engine: &Regenie2RunEngineCore,
    py: Python<'_>,
    sample_path: Option<String>,
    phenotype_path: String,
    phenotype_name: String,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: &str,
) -> PyResult<AlignedSampleData> {
    let sample_identifiers = sample_identifier_data_for_engine(engine, py, sample_path)?;
    let inputs = AlignmentInputs {
        sample_indices: sample_identifiers.sample_indices,
        family_identifiers: sample_identifiers.family_identifiers,
        individual_identifiers: sample_identifiers.individual_identifiers,
        phenotype_path,
        phenotype_name,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parse_sample_key_mode(sample_key_mode)?,
    };
    py.detach(move || g_input::align_sample_data(inputs))
        .map_err(|error| convert_input_error("align_sample_data", error))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn align_multi_sample_data_for_engine(
    engine: &Regenie2RunEngineCore,
    py: Python<'_>,
    sample_path: Option<String>,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: &str,
) -> PyResult<MultiAlignedSampleData> {
    let sample_identifiers = sample_identifier_data_for_engine(engine, py, sample_path)?;
    let inputs = MultiAlignmentInputs {
        sample_indices: sample_identifiers.sample_indices,
        family_identifiers: sample_identifiers.family_identifiers,
        individual_identifiers: sample_identifiers.individual_identifiers,
        phenotype_path,
        phenotype_names,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parse_sample_key_mode(sample_key_mode)?,
    };
    py.detach(move || g_input::align_multi_sample_data(inputs))
        .map_err(|error| convert_input_error("align_multi_sample_data", error))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn align_grouped_sample_data_for_engine(
    engine: &Regenie2RunEngineCore,
    py: Python<'_>,
    sample_path: Option<String>,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: &str,
) -> PyResult<Vec<AlignedPhenotypeGroup>> {
    let sample_identifiers = sample_identifier_data_for_engine(engine, py, sample_path)?;
    let inputs = MultiAlignmentInputs {
        sample_indices: sample_identifiers.sample_indices,
        family_identifiers: sample_identifiers.family_identifiers,
        individual_identifiers: sample_identifiers.individual_identifiers,
        phenotype_path,
        phenotype_names,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parse_sample_key_mode(sample_key_mode)?,
    };
    py.detach(move || g_input::align_grouped_sample_data(&inputs))
        .map_err(|error| convert_input_error("align_grouped_sample_data", error))
}
