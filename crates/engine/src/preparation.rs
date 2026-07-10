//! Native run output-header preparation.

use g_output::{CurrentRunManifestHeaderInput, ManifestFileFingerprintCache};

use crate::output_manifest::build_prediction_loco_file_fingerprints_with_cache;

#[derive(Debug, thiserror::Error)]
pub enum PipelineOutputPreparationError {
    #[error("Unknown planned phenotype '{phenotype_name}'.")]
    UnknownPlannedPhenotype { phenotype_name: String },
    #[error("Resolved GPU genotype format cannot remain auto during output preparation.")]
    UnresolvedGpuGenotypeFormat,
    #[error(transparent)]
    Output(#[from] g_output::OutputError),
}

pub(crate) struct RuntimeOutputPlan {
    pub variant_count: usize,
    pub effective_trusted_no_missing_diploid: bool,
    pub resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat,
}

pub(crate) struct RuntimeOutputGroupInput<'a> {
    pub phenotype_group: &'a g_plan::PhenotypeComputeGroup,
    pub covariate_names: &'a [String],
    pub sample_count: usize,
    pub output_sample_mode: g_plan::MultiPhenotypeSampleMode,
}

/// Build output preparation for one runtime output group.
///
/// # Errors
///
/// Returns an error when a grouped phenotype is unknown, manifest header
/// construction fails, or output preparation inputs are inconsistent.
pub(crate) fn build_runtime_output_initializations(
    run_plan: &g_plan::RunPlan,
    output_group: &RuntimeOutputGroupInput<'_>,
    runtime_plan: &RuntimeOutputPlan,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<Vec<CurrentRunManifestHeaderInput>, PipelineOutputPreparationError> {
    for phenotype_name in &output_group.phenotype_group.phenotype_names {
        if !run_plan.phenotype_runs.iter().any(|run| run.phenotype_name == *phenotype_name) {
            return Err(PipelineOutputPreparationError::UnknownPlannedPhenotype {
                phenotype_name: phenotype_name.clone(),
            });
        }
    }
    output_group
        .phenotype_group
        .phenotype_names
        .iter()
        .map(|phenotype_name| {
            build_current_header_input(run_plan, phenotype_name, output_group, runtime_plan, fingerprint_cache)
        })
        .collect()
}

fn build_current_header_input(
    run_plan: &g_plan::RunPlan,
    phenotype_name: &str,
    output_group: &RuntimeOutputGroupInput<'_>,
    runtime_plan: &RuntimeOutputPlan,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<CurrentRunManifestHeaderInput, PipelineOutputPreparationError> {
    if runtime_plan.resolved_gpu_genotype_format == g_plan::GpuGenotypeFormat::Auto {
        return Err(PipelineOutputPreparationError::UnresolvedGpuGenotypeFormat);
    }
    let phenotype_compute_group = output_group.phenotype_group;
    Ok(CurrentRunManifestHeaderInput {
        phenotype_name: phenotype_name.to_string(),
        covariate_names: output_group.covariate_names.to_vec(),
        prediction_loco_files: build_prediction_loco_file_fingerprints_with_cache(
            &run_plan.input.prediction_list_path,
            &phenotype_compute_group.phenotype_names,
            fingerprint_cache,
        )?,
        sample_count: output_group.sample_count,
        variant_count: runtime_plan.variant_count,
        effective_trusted_no_missing_diploid: runtime_plan.effective_trusted_no_missing_diploid,
        resolved_gpu_genotype_format: runtime_plan.resolved_gpu_genotype_format,
        output_sample_mode: output_group.output_sample_mode,
        phenotype_compute_group_id: g_plan::build_phenotype_compute_group_id(phenotype_compute_group),
        sample_set_fingerprint: phenotype_compute_group.sample_set_fingerprint.clone(),
        covariate_design_fingerprint: phenotype_compute_group.covariate_design_fingerprint.clone(),
        prediction_alignment_fingerprint: phenotype_compute_group.prediction_alignment_fingerprint.clone(),
    })
}
