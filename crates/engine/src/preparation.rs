//! Native run output-header preparation.

use std::sync::Arc;

use g_output::CurrentRunManifestHeaderInput;

#[derive(Debug, thiserror::Error)]
pub enum PipelineOutputPreparationError {
    #[error("Resolved GPU genotype format cannot remain auto during output preparation.")]
    UnresolvedGpuGenotypeFormat,
    #[error("Phenotype index {phenotype_index} has no resolved LOCO prediction file.")]
    MissingPredictionLocoFile { phenotype_index: u32 },
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
    output_group: &RuntimeOutputGroupInput<'_>,
    runtime_plan: &RuntimeOutputPlan,
    all_prediction_loco_files: &Arc<[g_output::PredictionLocoFileFingerprint]>,
) -> Result<Vec<CurrentRunManifestHeaderInput>, PipelineOutputPreparationError> {
    if runtime_plan.resolved_gpu_genotype_format == g_plan::GpuGenotypeFormat::Auto {
        return Err(PipelineOutputPreparationError::UnresolvedGpuGenotypeFormat);
    }
    let phenotype_compute_group = output_group.phenotype_group;
    let prediction_loco_files = if phenotype_compute_group.phenotype_indices.len() == all_prediction_loco_files.len()
        && phenotype_compute_group
            .phenotype_indices
            .iter()
            .enumerate()
            .all(|(expected_index, phenotype_index)| usize::try_from(*phenotype_index) == Ok(expected_index))
    {
        Arc::clone(all_prediction_loco_files)
    } else {
        phenotype_compute_group
            .phenotype_indices
            .iter()
            .map(|phenotype_index| {
                usize::try_from(*phenotype_index)
                    .ok()
                    .and_then(|index| all_prediction_loco_files.get(index))
                    .cloned()
                    .ok_or(PipelineOutputPreparationError::MissingPredictionLocoFile {
                        phenotype_index: *phenotype_index,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into()
    };
    let covariate_names: Arc<[String]> = output_group.covariate_names.to_vec().into();
    let phenotype_compute_group_id: Arc<str> = g_plan::build_phenotype_compute_group_id(phenotype_compute_group).into();
    let sample_set_fingerprint = phenotype_compute_group.sample_set_fingerprint.as_deref().map(Arc::from);
    let covariate_design_fingerprint = phenotype_compute_group.covariate_design_fingerprint.as_deref().map(Arc::from);
    let prediction_alignment_fingerprint =
        phenotype_compute_group.prediction_alignment_fingerprint.as_deref().map(Arc::from);
    Ok(phenotype_compute_group
        .phenotype_names
        .iter()
        .map(|phenotype_name| CurrentRunManifestHeaderInput {
            phenotype_name: phenotype_name.clone(),
            covariate_names: Arc::clone(&covariate_names),
            prediction_loco_files: Arc::clone(&prediction_loco_files),
            sample_count: output_group.sample_count,
            variant_count: runtime_plan.variant_count,
            effective_trusted_no_missing_diploid: runtime_plan.effective_trusted_no_missing_diploid,
            resolved_gpu_genotype_format: runtime_plan.resolved_gpu_genotype_format,
            output_sample_mode: output_group.output_sample_mode,
            phenotype_compute_group_id: Arc::clone(&phenotype_compute_group_id),
            sample_set_fingerprint: sample_set_fingerprint.as_ref().map(Arc::clone),
            covariate_design_fingerprint: covariate_design_fingerprint.as_ref().map(Arc::clone),
            prediction_alignment_fingerprint: prediction_alignment_fingerprint.as_ref().map(Arc::clone),
        })
        .collect())
}
