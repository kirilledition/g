//! Input section compilation.

use g_plan as plan;

use crate::ConfigResult;
use crate::resolved::RegenieConfigData;

use super::conversion;
use super::require_config_path;

pub(super) fn build_input_request(config: &RegenieConfigData) -> ConfigResult<plan::InputRequest> {
    Ok(plan::InputRequest {
        bgen_path: require_config_path("--bgen", config.input.bgen.as_ref())?,
        sample_path: config.input.sample.clone(),
        phenotype_path: require_config_path("--phenoFile", config.input.pheno_file.as_ref())?,
        prediction_list_path: require_config_path("--pred", config.input.pred.as_ref())?,
        covariate_path: config.input.covar_file.clone(),
        covariate_names: config.input.covar_columns.clone(),
        sample_key_mode: conversion::plan_sample_key_mode(config.g_compute.sample_key_mode),
    })
}
