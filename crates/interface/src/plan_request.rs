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
pub(crate) fn compile_run_plan(config: &RegenieConfigData) -> ConfigResult<plan::RunPlan> {
    let trait_type = config.trait_config.trait_type;
    let association_mode = match trait_type {
        plan::RegenieTraitType::Binary => plan::AssociationMode::Regenie2Binary,
        plan::RegenieTraitType::Quantitative => plan::AssociationMode::Regenie2Linear,
    };
    let phenotype_names = config.input.pheno_columns.clone();
    Ok(plan::RunPlan {
        association_mode,
        chunk_size: config.trait_config.bsize.get(),
        input: build_input_plan(config)?,
        compute: build_compute_plan(config),
        correction: build_correction_plan(config),
        output: build_output_plan(config)?,
        telemetry: config.g_diagnostics.telemetry,
        phenotype_runs: build_phenotype_run_plans(&phenotype_names)?,
    })
}

// Input plan

fn build_input_plan(config: &RegenieConfigData) -> ConfigResult<plan::InputPlan> {
    Ok(plan::InputPlan {
        bgen_path: require_config_path("--bgen", config.input.bgen.as_ref())?,
        bgen_content_sha256: config.input.bgen_content_sha256,
        sample_path: require_config_path("--sample", config.input.sample.as_ref())?,
        phenotype_path: require_config_path("--phenoFile", config.input.pheno_file.as_ref())?,
        prediction_list_path: require_config_path("--pred", config.input.pred.as_ref())?,
        covariate_path: config.input.covar_file.clone(),
        covariate_names: config.input.covar_columns.clone(),
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
        cpu_thread_count: config.g_compute.cpu_threads.map(std::num::NonZeroU32::get),
        jax_cache_directory: config.g_compute.jax_cache_dir.clone(),
        multi_phenotype_sample_mode: config.g_compute.multi_phenotype_sample_mode,
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
                maximum_step_size: config.g_compute.firth_maximum_step_size,
                pseudo_maximum_iterations: config.g_compute.firth_pseudo_maximum_iterations.get(),
                pseudo_inner_maximum_iterations: config.g_compute.firth_pseudo_inner_maximum_iterations.get(),
                line_search_maximum_attempts: config.g_compute.firth_line_search_maximum_attempts.get(),
                sparse_carrier_dosage_threshold: config.g_compute.firth_sparse_carrier_dosage_threshold,
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
        output_run_root,
        resume: config.g_output.resume,
        writer_thread_count: config.g_output.writer_threads.get(),
    })
}

fn default_output_run_root(output_prefix: &str) -> String {
    let output_prefix_path = Path::new(output_prefix);
    let output_name = output_prefix_path.file_name().and_then(std::ffi::OsStr::to_str).unwrap_or(output_prefix);
    output_prefix_path.with_file_name(format!("{output_name}.g")).display().to_string()
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
                phenotype_name: phenotype_name.clone(),
                output_directory_name: plan::build_phenotype_output_directory_name(output_index, phenotype_name),
            })
        })
        .collect()
}
