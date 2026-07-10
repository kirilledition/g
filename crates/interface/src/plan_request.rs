use std::path::Path;

use g_plan as plan;

use super::resolved::RegenieConfigData;
use super::{ConfigError, ConfigResult};

/// Compile a resolved config into a native requested-run plan.
///
/// # Errors
///
/// Returns an error when required run inputs are absent or native planning
/// policy rejects the requested correction/grouping configuration.
pub fn compile_run_plan(config: &RegenieConfigData) -> ConfigResult<plan::RunPlan> {
    let trait_type = config.trait_config.trait_type;
    let association_mode = match trait_type {
        plan::RegenieTraitType::Binary => plan::AssociationMode::Regenie2Binary,
        plan::RegenieTraitType::Quantitative => plan::AssociationMode::Regenie2Linear,
    };
    let phenotype_names = config.input.pheno_columns.clone();
    Ok(plan::RunPlan {
        association_mode,
        input: build_input_plan(config)?,
        analysis: plan::AnalysisPlan {
            trait_type,
            chunk_size: config.trait_config.bsize.get(),
            thread_count: config.trait_config.threads.map(std::num::NonZeroU32::get),
        },
        compute: build_compute_plan(config),
        correction: build_correction_plan(config),
        output: build_output_plan(config)?,
        runtime: build_runtime_plan(config),
        diagnostics: build_diagnostics_plan(config),
        phenotype_runs: build_phenotype_run_plans(&phenotype_names)?,
        phenotype_compute_groups: plan::build_phenotype_compute_groups(
            &phenotype_names,
            config.g_compute.multi_phenotype_sample_mode,
        )
        .map_err(|error| ConfigError::new(error.clone()))?,
    })
}

// Input plan

fn build_input_plan(config: &RegenieConfigData) -> ConfigResult<plan::InputPlan> {
    Ok(plan::InputPlan {
        bgen_path: require_config_path("--bgen", config.input.bgen.as_ref())?,
        sample_path: config.input.sample.clone(),
        phenotype_path: require_config_path("--phenoFile", config.input.pheno_file.as_ref())?,
        prediction_list_path: require_config_path("--pred", config.input.pred.as_ref())?,
        covariate_path: config.input.covar_file.clone(),
        covariate_names: config.input.covar_columns.clone(),
        sample_key_mode: config.g_compute.sample_key_mode,
    })
}

fn require_config_path(option_name: &str, path: Option<&String>) -> ConfigResult<String> {
    path.cloned().ok_or_else(|| ConfigError::new(format!("{option_name} is required to build a run plan.")))
}

// Compute plan

#[must_use]
fn build_compute_plan(config: &RegenieConfigData) -> plan::ComputePlan {
    plan::ComputePlan {
        device: config.g_compute.device,
        staging_depth: config.g_compute.staging_depth.get(),
        result_in_flight_limit: config.g_compute.result_in_flight_limit.map(std::num::NonZeroU32::get),
        variant_limit: config.g_compute.variant_limit.map(std::num::NonZeroU32::get),
        bgen_decode_tile_variant_count: config.g_compute.bgen_decode_tile_variant_count.get(),
        requested_gpu_genotype_format: config.g_compute.gpu_genotype_format,
        trusted_no_missing_diploid: config.g_compute.trusted_no_missing_diploid,
        trusted_bgen_validation_mode: config.g_compute.trusted_bgen_validation_mode,
        multi_phenotype_sample_mode: config.g_compute.multi_phenotype_sample_mode,
        score_dtype: config.g_compute.score_dtype,
        kernels: plan::KernelPlan {
            linear: plan::LinearKernelPlan {
                minimum_variance: config.g_compute.linear_minimum_variance,
                relative_variance_tolerance: config.g_compute.linear_relative_variance_tolerance,
            },
            binary_null: plan::BinaryNullKernelPlan {
                maximum_iterations: config.g_compute.binary_null_maximum_iterations.get(),
                coefficient_tolerance: config.g_compute.binary_null_coefficient_tolerance,
                nonconvergence_policy: config.g_compute.null_logistic_nonconvergence_policy,
                minimum_probability: config.g_compute.binary_minimum_probability,
                minimum_variance: config.g_compute.binary_minimum_variance,
                relative_variance_tolerance: config.g_compute.binary_relative_variance_tolerance,
            },
            firth: plan::FirthKernelPlan {
                batch_size: config.g_compute.firth_batch_size.get(),
                candidate_capacity: config.g_compute.firth_candidate_capacity.get(),
                maximum_iterations: config.g_compute.firth_maximum_iterations.get(),
                gradient_tolerance: config.g_compute.firth_gradient_tolerance,
                coefficient_tolerance: config.g_compute.firth_coefficient_tolerance,
                likelihood_tolerance: config.g_compute.firth_likelihood_tolerance,
                maximum_step_size: config.g_compute.firth_maximum_step_size,
                pseudo_maximum_iterations: config.g_compute.firth_pseudo_maximum_iterations.get(),
                pseudo_inner_maximum_iterations: config.g_compute.firth_pseudo_inner_maximum_iterations.get(),
                newton_raphson_zero_start_iterations: config.g_compute.firth_newton_raphson_zero_start_iterations.get(),
                line_search_maximum_attempts: config.g_compute.firth_line_search_maximum_attempts.get(),
                step_halving_maximum_attempts: config.g_compute.firth_step_halving_maximum_attempts.get(),
                initial_response_scale: config.g_compute.firth_initial_response_scale,
                sparse_carrier_dosage_threshold: config.g_compute.firth_sparse_carrier_dosage_threshold,
                step_halving_scale: config.g_compute.firth_step_halving_scale,
                use_block_math: config.g_compute.use_block_firth_math,
            },
            null_firth: plan::NullFirthKernelPlan {
                maximum_iterations: config.g_compute.null_firth_maximum_iterations.get(),
                gradient_tolerance: config.g_compute.null_firth_gradient_tolerance,
                maximum_step_size: config.g_compute.null_firth_maximum_step_size,
                fallback_iteration_multiplier: config.g_compute.null_firth_fallback_iteration_multiplier.get(),
                fallback_step_divisor: config.g_compute.null_firth_fallback_step_divisor,
                line_search_maximum_attempts: config.g_compute.null_firth_line_search_maximum_attempts.get(),
                step_halving_scale: config.g_compute.null_firth_step_halving_scale,
            },
        },
    }
}

// Correction plan

#[must_use]
fn build_correction_plan(config: &RegenieConfigData) -> plan::CorrectionPlan {
    if config.trait_config.trait_type == plan::RegenieTraitType::Quantitative {
        return plan::CorrectionPlan {
            method: plan::BinaryFallbackMethod::ScoreOnly,
            p_threshold: config.binary.p_threshold,
            firth_se: false,
        };
    }
    plan::CorrectionPlan {
        method: config.binary.fallback_method,
        p_threshold: config.binary.p_threshold,
        firth_se: config.binary.firth_se,
    }
}

// Output plan

fn build_output_plan(config: &RegenieConfigData) -> ConfigResult<plan::OutputPlan> {
    let output_prefix = require_config_path("--out", config.g_output.out.as_ref())?;
    let output_run_root =
        config.g_output.output_run_directory.clone().unwrap_or_else(|| default_output_run_root(&output_prefix));
    Ok(plan::OutputPlan {
        output_prefix,
        output_run_root,
        resume: config.g_output.resume,
        resume_mode: config.g_output.resume_mode,
        writer_thread_count: config.g_output.writer_threads.get(),
        writer_queue_depth: config.g_output.writer_queue_depth.get(),
        chunks_per_parquet_file: config.g_output.chunks_per_parquet_file.get(),
        parquet_compression: config.g_output.parquet_compression,
        output_statistic_dtype: config.g_output.output_statistic_dtype,
    })
}

fn default_output_run_root(output_prefix: &str) -> String {
    let output_prefix_path = Path::new(output_prefix);
    let output_name = output_prefix_path.file_name().and_then(std::ffi::OsStr::to_str).unwrap_or(output_prefix);
    output_prefix_path.with_file_name(format!("{output_name}.g")).display().to_string()
}

// Runtime plan

#[must_use]
fn build_runtime_plan(config: &RegenieConfigData) -> plan::RuntimePlan {
    plan::RuntimePlan {
        jax_cache_directory: config.g_compute.jax_cache_dir.clone(),
        jax_matmul_precision: config.g_compute.jax_matmul_precision,
        persistent_cache_enabled: config.g_compute.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes: config.g_compute.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds: config.g_compute.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache_enabled: config.g_compute.jax_xla_autotune_cache,
        transfer_guard_enabled: config.g_compute.jax_transfer_guard,
    }
}

// Diagnostics plan

#[must_use]
fn build_diagnostics_plan(config: &RegenieConfigData) -> plan::DiagnosticsPlan {
    plan::DiagnosticsPlan {
        telemetry: config.g_diagnostics.telemetry,
        log_directory: config.g_diagnostics.log_dir.clone(),
        stage_timings_path: config.g_diagnostics.stage_timings_json.clone(),
        log_filter: config.g_diagnostics.log_filter.clone(),
        log_file: config.g_diagnostics.log_file.clone(),
        log_to_stderr: config.g_diagnostics.log_stderr,
        profile_summary_path: config.g_diagnostics.profile_summary_json.clone(),
        trace_file: config.g_diagnostics.trace_file.clone(),
        trace_filter: config.g_diagnostics.trace_filter.clone(),
        trace_event_cap: config.g_diagnostics.trace_event_cap,
        log_queue_size: config.g_diagnostics.log_queue_size.get(),
        lossy_logging: config.g_diagnostics.log_lossy,
        include_source_location: config.g_diagnostics.include_source_location,
        include_span_events: config.g_diagnostics.include_span_events,
    }
}

// Phenotype plans

fn build_phenotype_run_plans(phenotype_names: &[String]) -> ConfigResult<Vec<plan::PhenotypeRunPlan>> {
    phenotype_names
        .iter()
        .enumerate()
        .map(|(phenotype_index, phenotype_name)| {
            let output_index = u32::try_from(phenotype_index + 1)
                .map_err(|_| ConfigError::new("Phenotype count exceeds native u32 capacity."))?;
            Ok(plan::PhenotypeRunPlan {
                phenotype_index: output_index,
                phenotype_name: phenotype_name.clone(),
                output_directory_name: plan::build_phenotype_output_directory_name(output_index, phenotype_name),
            })
        })
        .collect()
}
