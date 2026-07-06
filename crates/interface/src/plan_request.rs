use std::path::Path;

use g_plan as plan;

use super::domain::{
    ArrowCompressionValue, DeviceValue, FloatingPointDtypeValue, GpuGenotypeFormatValue, JaxMatmulPrecisionValue,
    MultiPhenotypeSampleModeValue, OutputFormatValue, ParquetCompressionValue, RegenieTraitTypeValue, ResumeModeValue,
    SampleKeyModeValue, TrustedBgenValidationModeValue,
};
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
        phenotype_compute_groups: build_phenotype_compute_groups(
            &phenotype_names,
            config.g_compute.multi_phenotype_sample_mode,
        )?,
        stage_timings_json: config.g_diagnostics.stage_timings_json.clone(),
    })
}

fn build_input_request(config: &RegenieConfigData) -> ConfigResult<plan::InputRequest> {
    Ok(plan::InputRequest {
        bgen_path: require_config_path("--bgen", config.input.bgen.as_ref())?,
        sample_path: config.input.sample.clone(),
        phenotype_path: require_config_path("--phenoFile", config.input.pheno_file.as_ref())?,
        prediction_list_path: require_config_path("--pred", config.input.pred.as_ref())?,
        covariate_path: config.input.covar_file.clone(),
        covariate_names: config.input.covar_columns.clone(),
        sample_key_mode: plan_sample_key_mode(config.g_compute.sample_key_mode),
    })
}

fn build_compute_request(config: &RegenieConfigData) -> plan::ComputeRequest {
    plan::ComputeRequest {
        device: plan_device(config.g_compute.device),
        staging_depth: config.g_compute.staging_depth.get(),
        native_callback_batch_size: config.g_compute.native_callback_batch_size.get(),
        result_in_flight_limit: config.g_compute.result_in_flight_limit.map(std::num::NonZeroU32::get),
        dosage_buffer_limit: config.g_compute.dosage_buffer_limit.map(std::num::NonZeroU32::get),
        variant_limit: config.g_compute.variant_limit.map(std::num::NonZeroU32::get),
        bgen_decode_tile_variant_count: config.g_compute.bgen_decode_tile_variant_count.get(),
        requested_gpu_genotype_format: plan_gpu_genotype_format(config.g_compute.gpu_genotype_format),
        trusted_no_missing_diploid: config.g_compute.trusted_no_missing_diploid,
        trusted_bgen_validation_mode: plan_trusted_bgen_validation_mode(config.g_compute.trusted_bgen_validation_mode),
        multi_phenotype_sample_mode: plan_multi_phenotype_sample_mode(config.g_compute.multi_phenotype_sample_mode),
        score_dtype: plan_floating_point_dtype(config.g_compute.score_dtype),
        firth_dtype: plan_floating_point_dtype(config.g_compute.firth_dtype),
    }
}

fn build_correction_plan(config: &RegenieConfigData) -> ConfigResult<plan::CorrectionPlan> {
    if config.trait_config.trait_type == RegenieTraitTypeValue::Quantitative {
        return Ok(plan::CorrectionPlan {
            method: plan::BinaryFallbackMethod::ScoreOnly,
            p_threshold: 0.05,
            firth_se: false,
        });
    }
    plan::normalize_binary_correction(
        config.binary.firth,
        config.binary.approx,
        config.binary.spa,
        f64::from(config.binary.p_threshold),
        config.binary.firth_se,
    )
    .map_err(plan_error_to_config_error)
}

fn build_output_writer_plan(config: &RegenieConfigData) -> ConfigResult<plan::OutputWriterPlan> {
    let output_prefix = require_config_path("--out", config.g_output.out.as_ref())?;
    let output_run_root =
        config.g_output.output_run_directory.clone().unwrap_or_else(|| default_output_run_root(&output_prefix));
    Ok(plan::OutputWriterPlan {
        output_prefix,
        output_run_root,
        resume: config.g_output.resume,
        resume_mode: plan_resume_mode(config.g_output.resume_mode),
        finalize_parquet: config.g_output.finalize_parquet,
        writer_thread_count: config.g_output.writer_threads.get(),
        writer_queue_depth: config.g_output.writer_queue_depth.get(),
        chunks_per_arrow_file: config.g_output.chunks_per_arrow_file.get(),
        arrow_compression: plan_arrow_compression(config.g_output.arrow_compression),
        parquet_compression: plan_parquet_compression(config.g_output.parquet_compression),
        output_format: plan_output_format(config.g_output.format),
        output_statistic_dtype: plan_floating_point_dtype(config.g_output.output_statistic_dtype),
    })
}

fn build_runtime_plan(config: &RegenieConfigData) -> plan::RuntimePlan {
    plan::RuntimePlan {
        jax_cache_directory: config.g_compute.jax_cache_dir.clone(),
        jax_matmul_precision: config.g_compute.jax_matmul_precision.map(plan_jax_matmul_precision),
        persistent_cache_enabled: config.g_compute.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes: config.g_compute.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds: config.g_compute.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache_enabled: config.g_compute.jax_xla_autotune_cache,
        transfer_guard_enabled: config.g_compute.jax_transfer_guard,
    }
}

fn build_phenotype_run_plans(phenotype_names: &[String]) -> Vec<plan::PhenotypeRunPlan> {
    phenotype_names
        .iter()
        .enumerate()
        .map(|(phenotype_index, phenotype_name)| {
            let output_index = u32::try_from(phenotype_index + 1).expect("phenotype count must fit in u32");
            plan::PhenotypeRunPlan {
                phenotype_index: output_index,
                phenotype_name: phenotype_name.clone(),
                output_directory_name: plan::build_phenotype_output_directory_name(output_index, phenotype_name),
            }
        })
        .collect()
}

fn build_phenotype_compute_groups(
    phenotype_names: &[String],
    multi_phenotype_sample_mode: MultiPhenotypeSampleModeValue,
) -> ConfigResult<Vec<plan::PhenotypeComputeGroup>> {
    plan::build_phenotype_compute_groups(phenotype_names, plan_multi_phenotype_sample_mode(multi_phenotype_sample_mode))
        .map_err(plan_error_to_config_error)
}

fn default_output_run_root(output_prefix: &str) -> String {
    let output_prefix_path = Path::new(output_prefix);
    let output_name = output_prefix_path.file_name().and_then(std::ffi::OsStr::to_str).unwrap_or(output_prefix);
    output_prefix_path.with_file_name(format!("{output_name}.g")).display().to_string()
}

fn require_config_path(option_name: &str, path: Option<&String>) -> ConfigResult<String> {
    path.cloned().ok_or_else(|| ConfigError::new(format!("{option_name} is required to build a run request.")))
}

fn plan_error_to_config_error(error: plan::HostPolicyError) -> ConfigError {
    match error {
        plan::HostPolicyError::NotImplemented(message) | plan::HostPolicyError::Value(message) => {
            ConfigError::new(message)
        }
    }
}

fn plan_trait_type(value: RegenieTraitTypeValue) -> plan::RegenieTraitType {
    match value {
        RegenieTraitTypeValue::Quantitative => plan::RegenieTraitType::Quantitative,
        RegenieTraitTypeValue::Binary => plan::RegenieTraitType::Binary,
    }
}

fn plan_device(value: DeviceValue) -> plan::Device {
    match value {
        DeviceValue::Cpu => plan::Device::Cpu,
        DeviceValue::Gpu => plan::Device::Gpu,
    }
}

fn plan_trusted_bgen_validation_mode(value: TrustedBgenValidationModeValue) -> plan::TrustedBgenValidationMode {
    match value {
        TrustedBgenValidationModeValue::CacheOnMiss => plan::TrustedBgenValidationMode::CacheOnMiss,
        TrustedBgenValidationModeValue::ForceValidate => plan::TrustedBgenValidationMode::ForceValidate,
        TrustedBgenValidationModeValue::AssumeValidated => plan::TrustedBgenValidationMode::AssumeValidated,
    }
}

fn plan_sample_key_mode(value: SampleKeyModeValue) -> plan::SampleKeyMode {
    match value {
        SampleKeyModeValue::Iid => plan::SampleKeyMode::Iid,
        SampleKeyModeValue::FidIid => plan::SampleKeyMode::FidIid,
    }
}

fn plan_multi_phenotype_sample_mode(value: MultiPhenotypeSampleModeValue) -> plan::MultiPhenotypeSampleMode {
    match value {
        MultiPhenotypeSampleModeValue::PerPhenotype => plan::MultiPhenotypeSampleMode::PerPhenotype,
        MultiPhenotypeSampleModeValue::CompleteCase => plan::MultiPhenotypeSampleMode::CompleteCase,
    }
}

fn plan_gpu_genotype_format(value: GpuGenotypeFormatValue) -> plan::GpuGenotypeFormat {
    match value {
        GpuGenotypeFormatValue::Auto => plan::GpuGenotypeFormat::Auto,
        GpuGenotypeFormatValue::Dosage => plan::GpuGenotypeFormat::Dosage,
        GpuGenotypeFormatValue::Packed8 => plan::GpuGenotypeFormat::Packed8,
    }
}

fn plan_floating_point_dtype(value: FloatingPointDtypeValue) -> plan::FloatingPointDtype {
    match value {
        FloatingPointDtypeValue::Float32 => plan::FloatingPointDtype::Float32,
        FloatingPointDtypeValue::Float64 => plan::FloatingPointDtype::Float64,
    }
}

fn plan_jax_matmul_precision(value: JaxMatmulPrecisionValue) -> plan::JaxMatmulPrecision {
    match value {
        JaxMatmulPrecisionValue::Float32 => plan::JaxMatmulPrecision::Float32,
        JaxMatmulPrecisionValue::TensorFloat32 => plan::JaxMatmulPrecision::TensorFloat32,
        JaxMatmulPrecisionValue::BrainFloat16 => plan::JaxMatmulPrecision::BrainFloat16,
        JaxMatmulPrecisionValue::Highest => plan::JaxMatmulPrecision::Highest,
    }
}

fn plan_resume_mode(value: ResumeModeValue) -> plan::ResumeMode {
    match value {
        ResumeModeValue::Fast => plan::ResumeMode::Fast,
        ResumeModeValue::Strict => plan::ResumeMode::Strict,
    }
}

fn plan_arrow_compression(value: ArrowCompressionValue) -> plan::ArrowCompression {
    match value {
        ArrowCompressionValue::Zstd => plan::ArrowCompression::Zstd,
        ArrowCompressionValue::None => plan::ArrowCompression::None,
    }
}

fn plan_parquet_compression(value: ParquetCompressionValue) -> plan::ParquetCompression {
    match value {
        ParquetCompressionValue::Zstd => plan::ParquetCompression::Zstd,
        ParquetCompressionValue::None => plan::ParquetCompression::None,
    }
}

fn plan_output_format(value: OutputFormatValue) -> plan::OutputFormat {
    match value {
        OutputFormatValue::Parquet => plan::OutputFormat::Parquet,
        OutputFormatValue::Arrow => plan::OutputFormat::Arrow,
        OutputFormatValue::Regenie => plan::OutputFormat::Regenie,
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::resolved::{
        BinaryConfigData, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData, InputConfigData,
        TraitConfigData,
    };

    #[test]
    fn compiles_quantitative_config_to_run_request() {
        let config = build_config(RegenieTraitTypeValue::Quantitative);
        let request = compile_run_request(&config).expect("request should compile");

        assert_eq!(request.association_mode, plan::AssociationMode::Regenie2Linear);
        assert_eq!(request.input.bgen_path, "data/input.bgen");
        assert_eq!(request.output.output_run_root, "data/out.g");
        assert_eq!(request.phenotype_runs[0].output_directory_name, "trait_0001_height");
        assert_eq!(request.correction.method, plan::BinaryFallbackMethod::ScoreOnly);
        assert_eq!(request.phenotype_compute_groups[0].group_mode, plan::PhenotypeComputeGroupMode::SinglePhenotype);
    }

    #[test]
    fn compiles_binary_config_to_approximate_firth_request() {
        let mut config = build_config(RegenieTraitTypeValue::Binary);
        config.binary.firth = true;
        config.binary.approx = true;
        config.binary.firth_se = true;
        config.input.pheno_columns = vec!["case".to_string(), "control".to_string()];
        config.g_compute.multi_phenotype_sample_mode = MultiPhenotypeSampleModeValue::CompleteCase;
        config.g_compute.gpu_genotype_format = GpuGenotypeFormatValue::Auto;

        let request = compile_run_request(&config).expect("request should compile");

        assert_eq!(request.association_mode, plan::AssociationMode::Regenie2Binary);
        assert_eq!(request.compute.requested_gpu_genotype_format, plan::GpuGenotypeFormat::Auto);
        assert_eq!(request.correction.method, plan::BinaryFallbackMethod::FirthApproximate);
        assert_eq!(request.phenotype_compute_groups.len(), 1);
        assert_eq!(request.phenotype_compute_groups[0].phenotype_indices, vec![0, 1]);
    }

    fn build_config(trait_type: RegenieTraitTypeValue) -> RegenieConfigData {
        RegenieConfigData {
            input: InputConfigData {
                bgen: Some("data/input.bgen".to_string()),
                sample: Some("data/input.sample".to_string()),
                pheno_file: Some("data/pheno.tsv".to_string()),
                pheno_columns: vec!["height".to_string()],
                covar_file: Some("data/covar.tsv".to_string()),
                covar_columns: vec!["age".to_string()],
                pred: Some("data/pred.list".to_string()),
            },
            trait_config: TraitConfigData {
                step: 2,
                trait_type,
                bsize: non_zero_u32(128),
                threads: Some(non_zero_u32(4)),
            },
            binary: BinaryConfigData { firth: false, approx: false, spa: false, p_threshold: 0.05, firth_se: false },
            g_compute: build_compute_config(),
            g_output: GOutputConfigData {
                out: Some("data/out".to_string()),
                format: OutputFormatValue::Parquet,
                output_run_directory: None,
                writer_threads: non_zero_u32(2),
                writer_queue_depth: non_zero_u32(4),
                chunks_per_arrow_file: non_zero_u32(8),
                arrow_compression: ArrowCompressionValue::Zstd,
                parquet_compression: ParquetCompressionValue::Zstd,
                output_statistic_dtype: FloatingPointDtypeValue::Float32,
                resume: true,
                resume_mode: ResumeModeValue::Strict,
                finalize_parquet: true,
            },
            g_diagnostics: GDiagnosticsConfigData {
                telemetry: crate::domain::TelemetryModeValue::Off,
                log_dir: None,
                stage_timings_json: Some("data/timing.json".to_string()),
                log_filter: "info".to_string(),
                log_file: None,
                log_stderr: false,
                progress_interval_seconds: 1.0,
                progress_interval_chunks: non_zero_u32(1),
                profile_summary_json: None,
                trace_file: None,
                trace_filter: "info".to_string(),
                trace_event_cap: 0,
                log_queue_size: non_zero_u32(64),
                log_lossy: false,
                include_source_location: false,
                include_span_events: false,
            },
            provenance: crate::resolved::ConfigProvenance::default(),
            is_validated: true,
        }
    }

    fn build_compute_config() -> GComputeConfigData {
        GComputeConfigData {
            device: DeviceValue::Gpu,
            staging_depth: non_zero_u32(2),
            native_callback_batch_size: non_zero_u32(1),
            result_in_flight_limit: Some(non_zero_u32(3)),
            dosage_buffer_limit: Some(non_zero_u32(5)),
            variant_limit: Some(non_zero_u32(100)),
            trusted_no_missing_diploid: true,
            trusted_bgen_validation_mode: TrustedBgenValidationModeValue::CacheOnMiss,
            sample_key_mode: SampleKeyModeValue::FidIid,
            multi_phenotype_sample_mode: MultiPhenotypeSampleModeValue::PerPhenotype,
            firth_batch_size: non_zero_u32(16),
            firth_candidate_capacity: non_zero_u32(32),
            binary_null_maximum_iterations: non_zero_u32(25),
            binary_null_coefficient_tolerance: 0.0001,
            null_logistic_nonconvergence_policy: crate::domain::NullLogisticNonconvergencePolicyValue::Warn,
            binary_minimum_probability: 0.0001,
            binary_minimum_variance: 0.0001,
            binary_relative_variance_tolerance: 0.001,
            linear_minimum_variance: 0.0001,
            linear_relative_variance_tolerance: 0.001,
            firth_maximum_iterations: non_zero_u32(25),
            firth_gradient_tolerance: 0.0001,
            firth_coefficient_tolerance: 0.0001,
            firth_likelihood_tolerance: 0.0001,
            firth_maximum_step_size: 1.0,
            firth_pseudo_maximum_iterations: non_zero_u32(8),
            firth_pseudo_inner_maximum_iterations: non_zero_u32(4),
            firth_newton_raphson_zero_start_iterations: non_zero_u32(2),
            firth_line_search_maximum_attempts: non_zero_u32(8),
            firth_step_halving_maximum_attempts: non_zero_u32(8),
            firth_initial_response_scale: 0.5,
            firth_sparse_carrier_dosage_threshold: 0.1,
            firth_step_halving_scale: 0.5,
            null_firth_maximum_iterations: non_zero_u32(25),
            null_firth_gradient_tolerance: 0.0001,
            null_firth_maximum_step_size: 1.0,
            null_firth_fallback_iteration_multiplier: non_zero_u32(2),
            null_firth_fallback_step_divisor: 2.0,
            null_firth_line_search_maximum_attempts: non_zero_u32(8),
            null_firth_step_halving_scale: 0.5,
            use_block_firth_math: false,
            bgen_decode_tile_variant_count: non_zero_u32(64),
            gpu_genotype_format: GpuGenotypeFormatValue::Packed8,
            score_dtype: FloatingPointDtypeValue::Float32,
            firth_dtype: FloatingPointDtypeValue::Float64,
            jax_cache_dir: Some("data/jax-cache".to_string()),
            jax_matmul_precision: Some(JaxMatmulPrecisionValue::Highest),
            jax_persistent_cache: true,
            jax_persistent_cache_min_entry_size_bytes: 1024,
            jax_persistent_cache_min_compile_time_seconds: 5,
            jax_xla_autotune_cache: true,
            jax_transfer_guard: true,
        }
    }

    fn non_zero_u32(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("test value should be non-zero")
    }
}
