use std::collections::BTreeMap;
use std::path::PathBuf;

use g_output::{
    CurrentRunManifestHeaderInput, ManifestFileFingerprintCache, build_current_run_manifest_header_json_with_cache,
};

use crate::output_manifest::build_prediction_loco_files_json_with_cache;

use super::batch::PipelineOutputPreparationBatch;
use super::error::PipelineResumeCompatibilityError;
use super::initialization::PipelineOutputInitialization;

#[derive(Debug, thiserror::Error)]
pub enum PipelineOutputPreparationError {
    #[error("{field_name} is required for grouped output.")]
    MissingGroupedOutputField { field_name: &'static str },
    #[error("phenotype_compute_group_indices and phenotype_compute_group_names must have the same length.")]
    PhenotypeComputeGroupLengthMismatch,
    #[error("Phenotype compute group index does not fit into uint32.")]
    PhenotypeComputeGroupIndexOutOfRange,
    #[error("Unknown prepared phenotype '{phenotype_name}'.")]
    UnknownPreparedPhenotype { phenotype_name: String },
    #[error("Invalid GPU genotype format '{value}'.")]
    InvalidGpuGenotypeFormat { value: String },
    #[error("Committed chunk count exceeds native int64 capacity.")]
    CommittedChunkCountOutOfRange,
    #[error("Output index exceeds native int64 capacity.")]
    OutputIndexOutOfRange,
    #[error(transparent)]
    AssociationBackend(#[from] g_plan::PreparedPlanError),
    #[error(transparent)]
    Output(#[from] g_output::OutputError),
    #[error(transparent)]
    ResumeCompatibility(#[from] PipelineResumeCompatibilityError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeOutputPreparedRun {
    pub phenotype_name: String,
    pub run_directory: PathBuf,
    pub chunks_directory: PathBuf,
    pub existing_manifest_json: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeOutputPlan {
    pub variant_count: i64,
    pub effective_trusted_no_missing_diploid: bool,
    pub sample_key_mode: String,
    pub binary_kernel_config_json: Option<String>,
    pub requested_gpu_genotype_format: String,
    pub gpu_genotype_format: String,
    pub score_dtype: String,
    pub firth_dtype: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeOutputGroupInput {
    pub phenotype_names: Vec<String>,
    pub covariate_names: Vec<String>,
    pub sample_count: i64,
    pub output_sample_mode: String,
    pub phenotype_compute_group_mode: Option<String>,
    pub phenotype_compute_group_indices: Option<Vec<i64>>,
    pub phenotype_compute_group_names: Option<Vec<String>>,
    pub phenotype_compute_group_sample_mode: Option<String>,
    pub sample_set_fingerprint: Option<String>,
    pub covariate_design_fingerprint: Option<String>,
    pub prediction_alignment_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeOutputGroup {
    pub phenotype_names: Vec<String>,
    pub covariate_names: Vec<String>,
    pub sample_count: i64,
    pub output_sample_mode: String,
    pub phenotype_compute_group: Option<RuntimeOutputPhenotypeComputeGroup>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeOutputPhenotypeComputeGroup {
    pub group_mode: String,
    pub phenotype_indices: Vec<u32>,
    pub phenotype_names: Vec<String>,
    pub sample_mode: String,
    pub sample_set_fingerprint: Option<String>,
    pub covariate_design_fingerprint: Option<String>,
    pub prediction_alignment_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeOutputPreparationGroup {
    pub phenotype_names: Vec<String>,
    pub preparation_batch: PipelineOutputPreparationBatch,
}

impl RuntimeOutputGroup {
    /// Build a runtime output group from native boundary input.
    ///
    /// # Errors
    ///
    /// Returns an error when grouped-output fields are incomplete, group names
    /// and indices have inconsistent lengths, or an index does not fit into
    /// `u32`.
    pub fn from_input(input: RuntimeOutputGroupInput) -> Result<Self, PipelineOutputPreparationError> {
        let phenotype_compute_group = build_runtime_output_phenotype_compute_group(
            input.phenotype_compute_group_mode,
            input.phenotype_compute_group_indices,
            input.phenotype_compute_group_names,
            input.phenotype_compute_group_sample_mode,
            input.sample_set_fingerprint,
            input.covariate_design_fingerprint,
            input.prediction_alignment_fingerprint,
        )?;
        Ok(Self {
            phenotype_names: input.phenotype_names,
            covariate_names: input.covariate_names,
            sample_count: input.sample_count,
            output_sample_mode: input.output_sample_mode,
            phenotype_compute_group,
        })
    }
}

/// Build output preparation for one runtime output group.
///
/// # Errors
///
/// Returns an error when a grouped phenotype is unknown, manifest header
/// construction fails, or output preparation inputs are inconsistent.
pub fn build_runtime_output_preparation_group(
    run_request: &g_plan::RunRequest,
    prepared_runs: &[RuntimeOutputPreparedRun],
    output_group: RuntimeOutputGroup,
    runtime_plan: &RuntimeOutputPlan,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<RuntimeOutputPreparationGroup, PipelineOutputPreparationError> {
    let prepared_runs_by_name = prepared_runs
        .iter()
        .map(|prepared_run| (prepared_run.phenotype_name.as_str(), prepared_run))
        .collect::<BTreeMap<_, _>>();
    let group_prepared_runs = output_group
        .phenotype_names
        .iter()
        .map(|phenotype_name| {
            prepared_runs_by_name.get(phenotype_name.as_str()).copied().ok_or_else(|| {
                PipelineOutputPreparationError::UnknownPreparedPhenotype { phenotype_name: phenotype_name.clone() }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let current_header_json_values =
        build_current_header_json_values(run_request, &output_group, runtime_plan, fingerprint_cache)?;
    let preparation_batch = PipelineOutputPreparationBatch::new(
        group_prepared_runs.iter().map(|prepared_run| prepared_run.run_directory.clone()).collect(),
        group_prepared_runs.iter().map(|prepared_run| prepared_run.chunks_directory.clone()).collect(),
        group_prepared_runs.iter().map(|prepared_run| prepared_run.existing_manifest_json.clone()).collect(),
        current_header_json_values,
        run_request.output.resume,
        match run_request.output.resume_mode {
            g_plan::ResumeMode::Fast => g_output::OutputResumeMode::Fast,
            g_plan::ResumeMode::Strict => g_output::OutputResumeMode::Strict,
        },
    )?;
    Ok(RuntimeOutputPreparationGroup { phenotype_names: output_group.phenotype_names, preparation_batch })
}

/// Build diagnostic payloads for already committed resume chunks.
///
/// # Errors
///
/// Returns an error when an output index or committed chunk count does not fit
/// into the runtime event integer representation.
pub fn build_output_resume_committed_chunk_diagnostic_payloads(
    initialization: &PipelineOutputInitialization,
) -> Result<Vec<g_runtime::RunDiagnosticEventPayload>, PipelineOutputPreparationError> {
    initialization
        .committed_chunk_counts()
        .into_iter()
        .enumerate()
        .map(|(output_index, committed_chunk_count)| {
            let committed_chunk_count_value = i64::try_from(committed_chunk_count)
                .map_err(|_| PipelineOutputPreparationError::CommittedChunkCountOutOfRange)?;
            let output_index_value =
                i64::try_from(output_index).map_err(|_| PipelineOutputPreparationError::OutputIndexOutOfRange)?;
            Ok(g_runtime::build_pipeline_output_resume_committed_chunks_diagnostic_payload(
                committed_chunk_count_value,
                output_index_value,
            ))
        })
        .collect()
}

fn build_runtime_output_phenotype_compute_group(
    group_mode: Option<String>,
    phenotype_indices: Option<Vec<i64>>,
    phenotype_names: Option<Vec<String>>,
    sample_mode: Option<String>,
    sample_set_fingerprint: Option<String>,
    covariate_design_fingerprint: Option<String>,
    prediction_alignment_fingerprint: Option<String>,
) -> Result<Option<RuntimeOutputPhenotypeComputeGroup>, PipelineOutputPreparationError> {
    if group_mode.is_none()
        && phenotype_indices.is_none()
        && phenotype_names.is_none()
        && sample_mode.is_none()
        && sample_set_fingerprint.is_none()
        && covariate_design_fingerprint.is_none()
        && prediction_alignment_fingerprint.is_none()
    {
        return Ok(None);
    }
    let group_mode = group_mode.ok_or(PipelineOutputPreparationError::MissingGroupedOutputField {
        field_name: "phenotype_compute_group_mode",
    })?;
    let phenotype_indices = phenotype_indices.ok_or(PipelineOutputPreparationError::MissingGroupedOutputField {
        field_name: "phenotype_compute_group_indices",
    })?;
    let phenotype_names = phenotype_names.ok_or(PipelineOutputPreparationError::MissingGroupedOutputField {
        field_name: "phenotype_compute_group_names",
    })?;
    let sample_mode = sample_mode.ok_or(PipelineOutputPreparationError::MissingGroupedOutputField {
        field_name: "phenotype_compute_group_sample_mode",
    })?;
    if phenotype_indices.len() != phenotype_names.len() {
        return Err(PipelineOutputPreparationError::PhenotypeComputeGroupLengthMismatch);
    }
    Ok(Some(RuntimeOutputPhenotypeComputeGroup {
        group_mode,
        phenotype_indices: convert_phenotype_indices(phenotype_indices)?,
        phenotype_names,
        sample_mode,
        sample_set_fingerprint,
        covariate_design_fingerprint,
        prediction_alignment_fingerprint,
    }))
}

fn convert_phenotype_indices(phenotype_indices: Vec<i64>) -> Result<Vec<u32>, PipelineOutputPreparationError> {
    phenotype_indices
        .into_iter()
        .map(|phenotype_index| {
            u32::try_from(phenotype_index)
                .map_err(|_| PipelineOutputPreparationError::PhenotypeComputeGroupIndexOutOfRange)
        })
        .collect()
}

fn build_current_header_json_values(
    run_request: &g_plan::RunRequest,
    output_group: &RuntimeOutputGroup,
    runtime_plan: &RuntimeOutputPlan,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<Vec<String>, PipelineOutputPreparationError> {
    output_group
        .phenotype_names
        .iter()
        .map(|phenotype_name| {
            let current_header_input =
                build_current_header_input(run_request, phenotype_name, output_group, runtime_plan, fingerprint_cache)?;
            build_current_run_manifest_header_json_with_cache(current_header_input, fingerprint_cache)
                .map_err(PipelineOutputPreparationError::from)
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
fn build_current_header_input(
    run_request: &g_plan::RunRequest,
    phenotype_name: &str,
    output_group: &RuntimeOutputGroup,
    runtime_plan: &RuntimeOutputPlan,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<CurrentRunManifestHeaderInput, PipelineOutputPreparationError> {
    let resolved_gpu_genotype_format = g_plan::GpuGenotypeFormat::from_str_value(&runtime_plan.gpu_genotype_format)
        .ok_or_else(|| PipelineOutputPreparationError::InvalidGpuGenotypeFormat {
            value: runtime_plan.gpu_genotype_format.clone(),
        })?;
    let association_backend_plan = g_plan::plan_association_backend(
        run_request.association_mode,
        run_request.compute.device,
        resolved_gpu_genotype_format,
    )?;
    let prediction_input_phenotype_names = output_group
        .phenotype_compute_group
        .as_ref()
        .map_or_else(|| vec![phenotype_name.to_string()], |group| group.phenotype_names.clone());
    let phenotype_compute_group = output_group.phenotype_compute_group.as_ref();
    Ok(CurrentRunManifestHeaderInput {
        association_mode: run_request.association_mode.as_str().to_string(),
        association_backend_kind: association_backend_plan.kind.as_str().to_string(),
        bgen_path: PathBuf::from(&run_request.input.bgen_path),
        sample_path: run_request.input.sample_path.as_ref().map(PathBuf::from),
        phenotype_path: PathBuf::from(&run_request.input.phenotype_path),
        phenotype_name: phenotype_name.to_string(),
        covariate_path: run_request.input.covariate_path.as_ref().map(PathBuf::from),
        covariate_names: output_group.covariate_names.clone(),
        prediction_list_path: PathBuf::from(&run_request.input.prediction_list_path),
        prediction_loco_files_json: build_prediction_loco_files_json_with_cache(
            &run_request.input.prediction_list_path,
            &prediction_input_phenotype_names,
            fingerprint_cache,
        )?,
        sample_count: output_group.sample_count,
        variant_count: runtime_plan.variant_count,
        chunk_size: i64::from(run_request.trait_request.chunk_size),
        variant_limit: run_request.compute.variant_limit.map(i64::from),
        binary_correction_plan_method: run_request.correction.method.as_str().to_string(),
        binary_correction_plan_p_threshold: run_request.correction.p_threshold,
        binary_correction_plan_firth_se: run_request.correction.firth_se,
        trusted_no_missing_diploid: runtime_plan.effective_trusted_no_missing_diploid,
        sample_key_mode: runtime_plan.sample_key_mode.clone(),
        binary_kernel_config_json: runtime_plan.binary_kernel_config_json.clone(),
        bgen_decode_tile_variant_count: i64::from(run_request.compute.bgen_decode_tile_variant_count),
        trusted_bgen_validation_mode: run_request.compute.trusted_bgen_validation_mode.as_str().to_string(),
        jax_device: run_request.compute.device.as_str().to_string(),
        jax_enable_x64: true,
        jax_matmul_precision: run_request
            .runtime
            .jax_matmul_precision
            .map(g_plan::JaxMatmulPrecision::as_str)
            .map(str::to_string),
        requested_gpu_genotype_format: runtime_plan.requested_gpu_genotype_format.clone(),
        gpu_genotype_format: runtime_plan.gpu_genotype_format.clone(),
        score_dtype: runtime_plan.score_dtype.clone(),
        firth_dtype: runtime_plan.firth_dtype.clone(),
        multi_phenotype_sample_mode: output_group.output_sample_mode.clone(),
        phenotype_compute_group_mode: phenotype_compute_group.map(|group| group.group_mode.clone()),
        phenotype_compute_group_indices: phenotype_compute_group.map(|group| group.phenotype_indices.clone()),
        phenotype_compute_group_names: phenotype_compute_group.map(|group| group.phenotype_names.clone()),
        phenotype_compute_group_sample_mode: phenotype_compute_group.map(|group| group.sample_mode.clone()),
        sample_set_fingerprint: phenotype_compute_group.and_then(|group| group.sample_set_fingerprint.clone()),
        covariate_design_fingerprint: phenotype_compute_group
            .and_then(|group| group.covariate_design_fingerprint.clone()),
        prediction_alignment_fingerprint: phenotype_compute_group
            .and_then(|group| group.prediction_alignment_fingerprint.clone()),
        output_format: run_request.output.output_format.as_str().to_string(),
        finalize_parquet: run_request.output.finalize_parquet,
        writer_thread_count: i64::from(run_request.output.writer_thread_count),
        writer_queue_depth: i64::from(run_request.output.writer_queue_depth),
        chunks_per_arrow_file: i64::from(run_request.output.chunks_per_arrow_file),
        arrow_compression: run_request.output.arrow_compression.as_str().to_string(),
        parquet_compression: run_request.output.parquet_compression.as_str().to_string(),
        output_statistic_dtype: run_request.output.output_statistic_dtype.as_str().to_string(),
    })
}
