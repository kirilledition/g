use std::num::NonZeroU32;

use g_plan as plan;
use serde::Deserialize;

use super::domain::NameList;
use super::resolved::{
    BinaryConfigData, ConfigProvenance, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData, InputConfigData,
    RegenieConfigData, TraitConfigData,
};
use super::{ConfigError, ConfigResult};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialConfig {
    pub(crate) input: PartialInputConfig,
    #[serde(rename = "trait")]
    pub(crate) trait_config: PartialTraitConfig,
    pub(crate) binary: PartialBinaryConfig,
    pub(crate) compute: PartialComputeConfig,
    pub(crate) output: PartialOutputConfig,
    pub(crate) diagnostics: PartialDiagnosticsConfig,
    pub(crate) metadata: Option<toml::Table>,
}

impl PartialConfig {
    pub(crate) fn resolve(self, provenance: ConfigProvenance) -> ConfigResult<RegenieConfigData> {
        Ok(RegenieConfigData {
            input: self.input.resolve(),
            trait_config: self.trait_config.resolve()?,
            binary: self.binary.resolve()?,
            g_compute: self.compute.resolve()?,
            g_output: self.output.resolve()?,
            g_diagnostics: self.diagnostics.resolve()?,
            provenance,
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialInputConfig {
    pub(crate) bgen: Option<String>,
    pub(crate) sample: Option<String>,
    pub(crate) pheno_file: Option<String>,
    pub(crate) pheno_columns: Option<NameList>,
    pub(crate) covar_file: Option<String>,
    pub(crate) covar_columns: Option<NameList>,
    pub(crate) pred: Option<String>,
}

impl PartialInputConfig {
    fn resolve(self) -> InputConfigData {
        InputConfigData {
            bgen: self.bgen,
            sample: self.sample,
            pheno_file: self.pheno_file,
            pheno_columns: self.pheno_columns.map(NameList::into_vec).unwrap_or_default(),
            covar_file: self.covar_file,
            covar_columns: self.covar_columns.map(NameList::into_vec).unwrap_or_default(),
            pred: self.pred,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialTraitConfig {
    pub(crate) trait_type: Option<plan::RegenieTraitType>,
    pub(crate) qt: Option<bool>,
    pub(crate) bt: Option<bool>,
    pub(crate) bsize: Option<NonZeroU32>,
    pub(crate) threads: Option<NonZeroU32>,
}

impl PartialTraitConfig {
    fn resolve(self) -> ConfigResult<TraitConfigData> {
        Ok(TraitConfigData {
            trait_type: normalize_trait_type(self.trait_type, self.qt, self.bt)?,
            bsize: required("bsize", self.bsize)?,
            threads: self.threads,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialBinaryConfig {
    pub(crate) fallback_method: Option<plan::BinaryFallbackMethod>,
    pub(crate) p_threshold: Option<plan::Probability>,
    pub(crate) firth_se: Option<bool>,
}

impl PartialBinaryConfig {
    fn resolve(self) -> ConfigResult<BinaryConfigData> {
        Ok(BinaryConfigData {
            fallback_method: required("fallback_method", self.fallback_method)?,
            p_threshold: required("p_threshold", self.p_threshold)?,
            firth_se: required("firth-se", self.firth_se)?,
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialComputeConfig {
    pub(crate) device: Option<plan::Device>,
    pub(crate) staging_depth: Option<NonZeroU32>,
    pub(crate) result_in_flight_limit: Option<NonZeroU32>,
    pub(crate) variant_limit: Option<NonZeroU32>,
    pub(crate) trusted_no_missing_diploid: Option<bool>,
    pub(crate) trusted_bgen_validation_mode: Option<plan::TrustedBgenValidationMode>,
    pub(crate) sample_key_mode: Option<plan::SampleKeyMode>,
    pub(crate) multi_phenotype_sample_mode: Option<plan::MultiPhenotypeSampleMode>,
    pub(crate) firth_batch_size: Option<NonZeroU32>,
    pub(crate) firth_candidate_capacity: Option<NonZeroU32>,
    pub(crate) binary_null_maximum_iterations: Option<NonZeroU32>,
    pub(crate) binary_null_coefficient_tolerance: Option<plan::PositiveF64>,
    pub(crate) null_logistic_nonconvergence_policy: Option<plan::NullLogisticNonconvergencePolicy>,
    pub(crate) binary_minimum_probability: Option<plan::ProbabilityFloor>,
    pub(crate) binary_minimum_variance: Option<plan::PositiveF64>,
    pub(crate) binary_relative_variance_tolerance: Option<plan::PositiveF64>,
    pub(crate) linear_minimum_variance: Option<plan::PositiveF64>,
    pub(crate) linear_relative_variance_tolerance: Option<plan::PositiveF64>,
    pub(crate) firth_maximum_iterations: Option<NonZeroU32>,
    pub(crate) firth_gradient_tolerance: Option<plan::PositiveF64>,
    pub(crate) firth_coefficient_tolerance: Option<plan::PositiveF64>,
    pub(crate) firth_likelihood_tolerance: Option<plan::PositiveF64>,
    pub(crate) firth_maximum_step_size: Option<plan::PositiveF64>,
    pub(crate) firth_pseudo_maximum_iterations: Option<NonZeroU32>,
    pub(crate) firth_pseudo_inner_maximum_iterations: Option<NonZeroU32>,
    pub(crate) firth_newton_raphson_zero_start_iterations: Option<NonZeroU32>,
    pub(crate) firth_line_search_maximum_attempts: Option<NonZeroU32>,
    pub(crate) firth_step_halving_maximum_attempts: Option<NonZeroU32>,
    pub(crate) firth_initial_response_scale: Option<plan::PositiveF64>,
    pub(crate) firth_sparse_carrier_dosage_threshold: Option<plan::DosageThreshold>,
    pub(crate) firth_step_halving_scale: Option<plan::StepScale>,
    pub(crate) null_firth_maximum_iterations: Option<NonZeroU32>,
    pub(crate) null_firth_gradient_tolerance: Option<plan::PositiveF64>,
    pub(crate) null_firth_maximum_step_size: Option<plan::PositiveF64>,
    pub(crate) null_firth_fallback_iteration_multiplier: Option<NonZeroU32>,
    pub(crate) null_firth_fallback_step_divisor: Option<plan::PositiveF64>,
    pub(crate) null_firth_line_search_maximum_attempts: Option<NonZeroU32>,
    pub(crate) null_firth_step_halving_scale: Option<plan::StepScale>,
    pub(crate) use_block_firth_math: Option<bool>,
    pub(crate) bgen_decode_tile_variant_count: Option<NonZeroU32>,
    pub(crate) gpu_genotype_format: Option<plan::GpuGenotypeFormat>,
    pub(crate) score_dtype: Option<plan::FloatingPointDtype>,
    pub(crate) jax_cache_dir: Option<String>,
    pub(crate) jax_matmul_precision: Option<plan::JaxMatmulPrecision>,
    pub(crate) jax_persistent_cache: Option<bool>,
    pub(crate) jax_persistent_cache_min_entry_size_bytes: Option<i64>,
    pub(crate) jax_persistent_cache_min_compile_time_seconds: Option<u32>,
    pub(crate) jax_xla_autotune_cache: Option<bool>,
    pub(crate) jax_transfer_guard: Option<bool>,
}

impl PartialComputeConfig {
    fn resolve(&self) -> ConfigResult<GComputeConfigData> {
        let core = self.resolve_core_fields()?;
        let firth = self.resolve_firth_fields()?;
        let null_firth = self.resolve_null_firth_fields()?;
        let genotype = self.resolve_genotype_fields()?;
        let jax = self.resolve_jax_fields()?;
        Ok(GComputeConfigData {
            device: core.device,
            staging_depth: core.staging_depth,
            result_in_flight_limit: core.result_in_flight_limit,
            variant_limit: core.variant_limit,
            trusted_no_missing_diploid: core.trusted_no_missing_diploid,
            trusted_bgen_validation_mode: core.trusted_bgen_validation_mode,
            sample_key_mode: core.sample_key_mode,
            multi_phenotype_sample_mode: core.multi_phenotype_sample_mode,
            firth_batch_size: core.firth_batch_size,
            firth_candidate_capacity: core.firth_candidate_capacity,
            binary_null_maximum_iterations: core.binary_null_maximum_iterations,
            binary_null_coefficient_tolerance: core.binary_null_coefficient_tolerance,
            null_logistic_nonconvergence_policy: core.null_logistic_nonconvergence_policy,
            binary_minimum_probability: core.binary_minimum_probability,
            binary_minimum_variance: core.binary_minimum_variance,
            binary_relative_variance_tolerance: core.binary_relative_variance_tolerance,
            linear_minimum_variance: core.linear_minimum_variance,
            linear_relative_variance_tolerance: core.linear_relative_variance_tolerance,
            firth_maximum_iterations: firth.maximum_iterations,
            firth_gradient_tolerance: firth.gradient_tolerance,
            firth_coefficient_tolerance: firth.coefficient_tolerance,
            firth_likelihood_tolerance: firth.likelihood_tolerance,
            firth_maximum_step_size: firth.maximum_step_size,
            firth_pseudo_maximum_iterations: firth.pseudo_maximum_iterations,
            firth_pseudo_inner_maximum_iterations: firth.pseudo_inner_maximum_iterations,
            firth_newton_raphson_zero_start_iterations: firth.newton_raphson_zero_start_iterations,
            firth_line_search_maximum_attempts: firth.line_search_maximum_attempts,
            firth_step_halving_maximum_attempts: firth.step_halving_maximum_attempts,
            firth_initial_response_scale: firth.initial_response_scale,
            firth_sparse_carrier_dosage_threshold: firth.sparse_carrier_dosage_threshold,
            firth_step_halving_scale: firth.step_halving_scale,
            null_firth_maximum_iterations: null_firth.maximum_iterations,
            null_firth_gradient_tolerance: null_firth.gradient_tolerance,
            null_firth_maximum_step_size: null_firth.maximum_step_size,
            null_firth_fallback_iteration_multiplier: null_firth.fallback_iteration_multiplier,
            null_firth_fallback_step_divisor: null_firth.fallback_step_divisor,
            null_firth_line_search_maximum_attempts: null_firth.line_search_maximum_attempts,
            null_firth_step_halving_scale: null_firth.step_halving_scale,
            use_block_firth_math: genotype.use_block_firth_math,
            bgen_decode_tile_variant_count: genotype.bgen_decode_tile_variant_count,
            gpu_genotype_format: genotype.gpu_genotype_format,
            score_dtype: genotype.score_dtype,
            jax_cache_dir: jax.cache_dir,
            jax_matmul_precision: jax.matmul_precision,
            jax_persistent_cache: jax.persistent_cache,
            jax_persistent_cache_min_entry_size_bytes: jax.persistent_cache_min_entry_size_bytes,
            jax_persistent_cache_min_compile_time_seconds: jax.persistent_cache_min_compile_time_seconds,
            jax_xla_autotune_cache: jax.xla_autotune_cache,
            jax_transfer_guard: jax.transfer_guard,
        })
    }

    fn resolve_core_fields(&self) -> ConfigResult<ResolvedComputeCoreFields> {
        Ok(ResolvedComputeCoreFields {
            device: required("device", self.device)?,
            staging_depth: required("staging_depth", self.staging_depth)?,
            result_in_flight_limit: self.result_in_flight_limit,
            variant_limit: self.variant_limit,
            trusted_no_missing_diploid: required("trusted_no_missing_diploid", self.trusted_no_missing_diploid)?,
            trusted_bgen_validation_mode: required("trusted_bgen_validation_mode", self.trusted_bgen_validation_mode)?,
            sample_key_mode: required("sample_key_mode", self.sample_key_mode)?,
            multi_phenotype_sample_mode: required("multi_phenotype_sample_mode", self.multi_phenotype_sample_mode)?,
            firth_batch_size: required("firth_batch_size", self.firth_batch_size)?,
            firth_candidate_capacity: required("firth_candidate_capacity", self.firth_candidate_capacity)?,
            binary_null_maximum_iterations: required(
                "binary_null_maximum_iterations",
                self.binary_null_maximum_iterations,
            )?,
            binary_null_coefficient_tolerance: required(
                "binary_null_coefficient_tolerance",
                self.binary_null_coefficient_tolerance,
            )?,
            null_logistic_nonconvergence_policy: required(
                "null_logistic_nonconvergence_policy",
                self.null_logistic_nonconvergence_policy,
            )?,
            binary_minimum_probability: required("binary_minimum_probability", self.binary_minimum_probability)?,
            binary_minimum_variance: required("binary_minimum_variance", self.binary_minimum_variance)?,
            binary_relative_variance_tolerance: required(
                "binary_relative_variance_tolerance",
                self.binary_relative_variance_tolerance,
            )?,
            linear_minimum_variance: required("linear_minimum_variance", self.linear_minimum_variance)?,
            linear_relative_variance_tolerance: required(
                "linear_relative_variance_tolerance",
                self.linear_relative_variance_tolerance,
            )?,
        })
    }

    fn resolve_firth_fields(&self) -> ConfigResult<ResolvedFirthFields> {
        Ok(ResolvedFirthFields {
            maximum_iterations: required("firth_maximum_iterations", self.firth_maximum_iterations)?,
            gradient_tolerance: required("firth_gradient_tolerance", self.firth_gradient_tolerance)?,
            coefficient_tolerance: required("firth_coefficient_tolerance", self.firth_coefficient_tolerance)?,
            likelihood_tolerance: required("firth_likelihood_tolerance", self.firth_likelihood_tolerance)?,
            maximum_step_size: required("firth_maximum_step_size", self.firth_maximum_step_size)?,
            pseudo_maximum_iterations: required(
                "firth_pseudo_maximum_iterations",
                self.firth_pseudo_maximum_iterations,
            )?,
            pseudo_inner_maximum_iterations: required(
                "firth_pseudo_inner_maximum_iterations",
                self.firth_pseudo_inner_maximum_iterations,
            )?,
            newton_raphson_zero_start_iterations: required(
                "firth_newton_raphson_zero_start_iterations",
                self.firth_newton_raphson_zero_start_iterations,
            )?,
            line_search_maximum_attempts: required(
                "firth_line_search_maximum_attempts",
                self.firth_line_search_maximum_attempts,
            )?,
            step_halving_maximum_attempts: required(
                "firth_step_halving_maximum_attempts",
                self.firth_step_halving_maximum_attempts,
            )?,
            initial_response_scale: required("firth_initial_response_scale", self.firth_initial_response_scale)?,
            sparse_carrier_dosage_threshold: required(
                "firth_sparse_carrier_dosage_threshold",
                self.firth_sparse_carrier_dosage_threshold,
            )?,
            step_halving_scale: required("firth_step_halving_scale", self.firth_step_halving_scale)?,
        })
    }

    fn resolve_null_firth_fields(&self) -> ConfigResult<ResolvedNullFirthFields> {
        Ok(ResolvedNullFirthFields {
            maximum_iterations: required("null_firth_maximum_iterations", self.null_firth_maximum_iterations)?,
            gradient_tolerance: required("null_firth_gradient_tolerance", self.null_firth_gradient_tolerance)?,
            maximum_step_size: required("null_firth_maximum_step_size", self.null_firth_maximum_step_size)?,
            fallback_iteration_multiplier: required(
                "null_firth_fallback_iteration_multiplier",
                self.null_firth_fallback_iteration_multiplier,
            )?,
            fallback_step_divisor: required("null_firth_fallback_step_divisor", self.null_firth_fallback_step_divisor)?,
            line_search_maximum_attempts: required(
                "null_firth_line_search_maximum_attempts",
                self.null_firth_line_search_maximum_attempts,
            )?,
            step_halving_scale: required("null_firth_step_halving_scale", self.null_firth_step_halving_scale)?,
        })
    }

    fn resolve_genotype_fields(&self) -> ConfigResult<ResolvedGenotypeFields> {
        Ok(ResolvedGenotypeFields {
            use_block_firth_math: required("use_block_firth_math", self.use_block_firth_math)?,
            bgen_decode_tile_variant_count: required(
                "bgen_decode_tile_variant_count",
                self.bgen_decode_tile_variant_count,
            )?,
            gpu_genotype_format: required("gpu_genotype_format", self.gpu_genotype_format)?,
            score_dtype: required("score_dtype", self.score_dtype)?,
        })
    }

    fn resolve_jax_fields(&self) -> ConfigResult<ResolvedJaxFields> {
        Ok(ResolvedJaxFields {
            cache_dir: self.jax_cache_dir.clone(),
            matmul_precision: self.jax_matmul_precision,
            persistent_cache: required("jax_persistent_cache", self.jax_persistent_cache)?,
            persistent_cache_min_entry_size_bytes: required(
                "jax_persistent_cache_min_entry_size_bytes",
                self.jax_persistent_cache_min_entry_size_bytes,
            )?,
            persistent_cache_min_compile_time_seconds: required(
                "jax_persistent_cache_min_compile_time_seconds",
                self.jax_persistent_cache_min_compile_time_seconds,
            )?,
            xla_autotune_cache: required("jax_xla_autotune_cache", self.jax_xla_autotune_cache)?,
            transfer_guard: required("jax_transfer_guard", self.jax_transfer_guard)?,
        })
    }
}

struct ResolvedComputeCoreFields {
    device: plan::Device,
    staging_depth: NonZeroU32,
    result_in_flight_limit: Option<NonZeroU32>,
    variant_limit: Option<NonZeroU32>,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: plan::TrustedBgenValidationMode,
    sample_key_mode: plan::SampleKeyMode,
    multi_phenotype_sample_mode: plan::MultiPhenotypeSampleMode,
    firth_batch_size: NonZeroU32,
    firth_candidate_capacity: NonZeroU32,
    binary_null_maximum_iterations: NonZeroU32,
    binary_null_coefficient_tolerance: plan::PositiveF64,
    null_logistic_nonconvergence_policy: plan::NullLogisticNonconvergencePolicy,
    binary_minimum_probability: plan::ProbabilityFloor,
    binary_minimum_variance: plan::PositiveF64,
    binary_relative_variance_tolerance: plan::PositiveF64,
    linear_minimum_variance: plan::PositiveF64,
    linear_relative_variance_tolerance: plan::PositiveF64,
}

struct ResolvedFirthFields {
    maximum_iterations: NonZeroU32,
    gradient_tolerance: plan::PositiveF64,
    coefficient_tolerance: plan::PositiveF64,
    likelihood_tolerance: plan::PositiveF64,
    maximum_step_size: plan::PositiveF64,
    pseudo_maximum_iterations: NonZeroU32,
    pseudo_inner_maximum_iterations: NonZeroU32,
    newton_raphson_zero_start_iterations: NonZeroU32,
    line_search_maximum_attempts: NonZeroU32,
    step_halving_maximum_attempts: NonZeroU32,
    initial_response_scale: plan::PositiveF64,
    sparse_carrier_dosage_threshold: plan::DosageThreshold,
    step_halving_scale: plan::StepScale,
}

struct ResolvedNullFirthFields {
    maximum_iterations: NonZeroU32,
    gradient_tolerance: plan::PositiveF64,
    maximum_step_size: plan::PositiveF64,
    fallback_iteration_multiplier: NonZeroU32,
    fallback_step_divisor: plan::PositiveF64,
    line_search_maximum_attempts: NonZeroU32,
    step_halving_scale: plan::StepScale,
}

struct ResolvedGenotypeFields {
    use_block_firth_math: bool,
    bgen_decode_tile_variant_count: NonZeroU32,
    gpu_genotype_format: plan::GpuGenotypeFormat,
    score_dtype: plan::FloatingPointDtype,
}

struct ResolvedJaxFields {
    cache_dir: Option<String>,
    matmul_precision: Option<plan::JaxMatmulPrecision>,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: u32,
    xla_autotune_cache: bool,
    transfer_guard: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialOutputConfig {
    pub(crate) out: Option<String>,
    pub(crate) output_run_directory: Option<String>,
    pub(crate) writer_threads: Option<NonZeroU32>,
    pub(crate) writer_queue_depth: Option<NonZeroU32>,
    pub(crate) chunks_per_parquet_file: Option<NonZeroU32>,
    pub(crate) parquet_compression: Option<plan::ParquetCompression>,
    pub(crate) output_statistic_dtype: Option<plan::FloatingPointDtype>,
    pub(crate) resume: Option<bool>,
    pub(crate) resume_mode: Option<plan::ResumeMode>,
}

impl PartialOutputConfig {
    fn resolve(self) -> ConfigResult<GOutputConfigData> {
        Ok(GOutputConfigData {
            out: self.out,
            output_run_directory: self.output_run_directory,
            writer_threads: required("writer_threads", self.writer_threads)?,
            writer_queue_depth: required("writer_queue_depth", self.writer_queue_depth)?,
            chunks_per_parquet_file: required("chunks_per_parquet_file", self.chunks_per_parquet_file)?,
            parquet_compression: required("parquet_compression", self.parquet_compression)?,
            output_statistic_dtype: required("output_statistic_dtype", self.output_statistic_dtype)?,
            resume: required("resume", self.resume)?,
            resume_mode: required("resume_mode", self.resume_mode)?,
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialDiagnosticsConfig {
    pub(crate) telemetry: Option<plan::TelemetryMode>,
    pub(crate) log_dir: Option<String>,
    pub(crate) stage_timings_json: Option<String>,
    pub(crate) log_filter: Option<String>,
    pub(crate) log_file: Option<String>,
    pub(crate) log_stderr: Option<bool>,
    pub(crate) profile_summary_json: Option<String>,
    pub(crate) trace_file: Option<String>,
    pub(crate) trace_filter: Option<String>,
    pub(crate) trace_event_cap: Option<u32>,
    pub(crate) log_queue_size: Option<NonZeroU32>,
    pub(crate) log_lossy: Option<bool>,
    pub(crate) include_source_location: Option<bool>,
    pub(crate) include_span_events: Option<bool>,
}

impl PartialDiagnosticsConfig {
    fn resolve(self) -> ConfigResult<GDiagnosticsConfigData> {
        Ok(GDiagnosticsConfigData {
            telemetry: required("telemetry", self.telemetry)?,
            log_dir: self.log_dir,
            stage_timings_json: self.stage_timings_json,
            log_filter: required("log_filter", self.log_filter)?,
            log_file: self.log_file,
            log_stderr: required("log_stderr", self.log_stderr)?,
            profile_summary_json: self.profile_summary_json,
            trace_file: self.trace_file,
            trace_filter: required("trace_filter", self.trace_filter)?,
            trace_event_cap: required("trace_event_cap", self.trace_event_cap)?,
            log_queue_size: required("log_queue_size", self.log_queue_size)?,
            log_lossy: required("log_lossy", self.log_lossy)?,
            include_source_location: required("include_source_location", self.include_source_location)?,
            include_span_events: required("include_span_events", self.include_span_events)?,
        })
    }
}

fn required<ValueType>(option_name: &str, value: Option<ValueType>) -> ConfigResult<ValueType> {
    value.ok_or_else(|| ConfigError::new(format!("Default config is missing required default option {option_name:?}.")))
}

pub(crate) fn normalize_trait_type(
    trait_type: Option<plan::RegenieTraitType>,
    qt: Option<bool>,
    bt: Option<bool>,
) -> ConfigResult<plan::RegenieTraitType> {
    if qt == Some(true) && bt == Some(true) {
        return Err(ConfigError::new("--qt and --bt are mutually exclusive."));
    }
    if bt == Some(true) {
        return Ok(plan::RegenieTraitType::Binary);
    }
    if qt == Some(true) {
        return Ok(plan::RegenieTraitType::Quantitative);
    }
    Ok(trait_type.unwrap_or(plan::RegenieTraitType::Quantitative))
}
