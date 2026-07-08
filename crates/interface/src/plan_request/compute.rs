//! Compute section compilation.

use g_plan as plan;

use crate::resolved::RegenieConfigData;

use super::conversion;

#[must_use]
pub(super) fn build_compute_request(config: &RegenieConfigData) -> plan::ComputeRequest {
    plan::ComputeRequest {
        device: conversion::plan_device(config.g_compute.device),
        staging_depth: config.g_compute.staging_depth.get(),
        native_callback_batch_size: config.g_compute.native_callback_batch_size.get(),
        result_in_flight_limit: config.g_compute.result_in_flight_limit.map(std::num::NonZeroU32::get),
        dosage_buffer_limit: config.g_compute.dosage_buffer_limit.map(std::num::NonZeroU32::get),
        variant_limit: config.g_compute.variant_limit.map(std::num::NonZeroU32::get),
        bgen_decode_tile_variant_count: config.g_compute.bgen_decode_tile_variant_count.get(),
        requested_gpu_genotype_format: conversion::plan_gpu_genotype_format(config.g_compute.gpu_genotype_format),
        trusted_no_missing_diploid: config.g_compute.trusted_no_missing_diploid,
        trusted_bgen_validation_mode: conversion::plan_trusted_bgen_validation_mode(
            config.g_compute.trusted_bgen_validation_mode,
        ),
        multi_phenotype_sample_mode: conversion::plan_multi_phenotype_sample_mode(
            config.g_compute.multi_phenotype_sample_mode,
        ),
        score_dtype: conversion::plan_floating_point_dtype(config.g_compute.score_dtype),
        firth_dtype: conversion::plan_floating_point_dtype(config.g_compute.firth_dtype),
    }
}
