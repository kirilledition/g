//! Output section compilation.

use std::path::Path;

use g_plan as plan;

use crate::ConfigResult;
use crate::resolved::RegenieConfigData;

use super::conversion;
use super::require_config_path;

pub(super) fn build_output_writer_plan(config: &RegenieConfigData) -> ConfigResult<plan::OutputWriterPlan> {
    let output_prefix = require_config_path("--out", config.g_output.out.as_ref())?;
    let output_run_root =
        config.g_output.output_run_directory.clone().unwrap_or_else(|| default_output_run_root(&output_prefix));
    Ok(plan::OutputWriterPlan {
        output_prefix,
        output_run_root,
        resume: config.g_output.resume,
        resume_mode: conversion::plan_resume_mode(config.g_output.resume_mode),
        finalize_parquet: config.g_output.finalize_parquet,
        writer_thread_count: config.g_output.writer_threads.get(),
        writer_queue_depth: config.g_output.writer_queue_depth.get(),
        chunks_per_arrow_file: config.g_output.chunks_per_arrow_file.get(),
        arrow_compression: conversion::plan_arrow_compression(config.g_output.arrow_compression),
        parquet_compression: conversion::plan_parquet_compression(config.g_output.parquet_compression),
        output_format: conversion::plan_output_format(config.g_output.format),
        output_statistic_dtype: conversion::plan_floating_point_dtype(config.g_output.output_statistic_dtype),
    })
}

fn default_output_run_root(output_prefix: &str) -> String {
    let output_prefix_path = Path::new(output_prefix);
    let output_name = output_prefix_path.file_name().and_then(std::ffi::OsStr::to_str).unwrap_or(output_prefix);
    output_prefix_path.with_file_name(format!("{output_name}.g")).display().to_string()
}
