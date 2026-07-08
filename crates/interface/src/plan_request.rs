use g_plan as plan;

mod compute;
mod conversion;
mod correction;
mod input;
mod output;
mod phenotype;
mod runtime;

use compute::build_compute_request;
use conversion::plan_trait_type;
use correction::build_correction_plan;
use input::build_input_request;
use output::build_output_writer_plan;
use phenotype::{build_phenotype_compute_groups, build_phenotype_run_plans};
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
        phenotype_compute_groups: build_phenotype_compute_groups(
            &phenotype_names,
            config.g_compute.multi_phenotype_sample_mode,
        )?,
        stage_timings_json: config.g_diagnostics.stage_timings_json.clone(),
    })
}

fn require_config_path(option_name: &str, path: Option<&String>) -> ConfigResult<String> {
    path.cloned().ok_or_else(|| ConfigError::new(format!("{option_name} is required to build a run request.")))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::domain::{
        ArrowCompressionValue, DeviceValue, FloatingPointDtypeValue, GpuGenotypeFormatValue, JaxMatmulPrecisionValue,
        MultiPhenotypeSampleModeValue, OutputFormatValue, ParquetCompressionValue, RegenieTraitTypeValue,
        ResumeModeValue, SampleKeyModeValue, TrustedBgenValidationModeValue,
    };
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
