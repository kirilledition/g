use g_plan as plan;

mod compute;
mod conversion;
mod correction;
mod input;
mod output;
mod phenotype;
mod runtime;

use compute::build_compute_request;
use conversion::{plan_error_to_config_error, plan_multi_phenotype_sample_mode, plan_trait_type};
use correction::build_correction_plan;
use input::build_input_request;
use output::build_output_writer_plan;
use phenotype::build_phenotype_run_plans;
use runtime::build_runtime_plan;

use super::resolved::RegenieConfigData;
use super::{ConfigError, ConfigResult};

/// Compile a resolved config into a native requested-run plan.
///
/// # Errors
///
/// Returns an error when required run inputs are absent or native planning
/// policy rejects the requested correction/grouping configuration.
pub fn compile_run_request(config: &RegenieConfigData) -> ConfigResult<plan::RunRequest> {
    let trait_type = plan_trait_type(config.trait_config.trait_type);
    let association_mode = match trait_type {
        plan::RegenieTraitType::Binary => plan::AssociationMode::Regenie2Binary,
        plan::RegenieTraitType::Quantitative => plan::AssociationMode::Regenie2Linear,
    };
    let phenotype_names = config.input.pheno_columns.clone();
    Ok(plan::RunRequest {
        association_mode,
        input: build_input_request(config)?,
        trait_request: plan::TraitRequest {
            trait_type,
            chunk_size: config.trait_config.bsize.get(),
            thread_count: config.trait_config.threads.map(std::num::NonZeroU32::get),
        },
        compute: build_compute_request(config),
        correction: build_correction_plan(config)?,
        output: build_output_writer_plan(config)?,
        runtime: build_runtime_plan(config),
        phenotype_runs: build_phenotype_run_plans(&phenotype_names),
        phenotype_compute_groups: plan::build_phenotype_compute_groups(
            &phenotype_names,
            plan_multi_phenotype_sample_mode(config.g_compute.multi_phenotype_sample_mode),
        )
        .map_err(plan_error_to_config_error)?,
        stage_timings_json: config.g_diagnostics.stage_timings_json.clone(),
    })
}

fn require_config_path(option_name: &str, path: Option<&String>) -> ConfigResult<String> {
    path.cloned().ok_or_else(|| ConfigError::new(format!("{option_name} is required to build a run request.")))
}
