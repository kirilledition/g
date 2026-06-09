use std::collections::BTreeSet;

use serde::Deserialize;

use super::data::{
    BinaryConfigData, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData, InputConfigData,
    RegenieConfigData, TraitConfigData,
};
use super::domain::{
    ArrowCompressionValue, DeviceValue, FloatingPointDtypeValue, GpuGenotypeFormatValue, JaxMatmulPrecisionValue,
    MultiPhenotypeSampleModeValue, NameList, NonNegativeU32, NullLogisticNonconvergencePolicyValue, OutputFormatValue,
    ParquetCompressionValue, PositiveF32, PositiveU32, Probability, ProbabilityFloor, ResumeModeValue,
    SampleKeyModeValue, TelemetryModeValue, TrustedBgenValidationModeValue,
};
use super::{ConfigError, ConfigResult};

macro_rules! overlay_option {
    ($target:expr, $source:expr, $field:ident) => {
        if $source.$field.is_some() {
            $target.$field = $source.$field;
        }
    };
}

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
    pub(crate) fn overlay(&mut self, override_config: Self) -> ConfigResult<()> {
        override_config.reject_trait_flag_conflict()?;
        self.input.overlay(override_config.input);
        self.trait_config.overlay(override_config.trait_config);
        self.binary.overlay(override_config.binary);
        self.compute.overlay(override_config.compute);
        self.output.overlay(override_config.output);
        self.diagnostics.overlay(override_config.diagnostics);
        if override_config.metadata.is_some() {
            self.metadata = override_config.metadata;
        }
        self.apply_trait_flag_precedence();
        Ok(())
    }

    pub(crate) fn resolve(self, explicit_options: BTreeSet<String>) -> ConfigResult<RegenieConfigData> {
        Ok(RegenieConfigData {
            input: self.input.resolve()?,
            trait_config: self.trait_config.resolve()?,
            binary: self.binary.resolve()?,
            g_compute: self.compute.resolve()?,
            g_output: self.output.resolve()?,
            g_diagnostics: self.diagnostics.resolve()?,
            explicit_options,
            is_validated: false,
        })
    }

    fn reject_trait_flag_conflict(&self) -> ConfigResult<()> {
        if self.trait_config.qt == Some(true) && self.trait_config.bt == Some(true) {
            return Err(ConfigError::new("--qt and --bt are mutually exclusive."));
        }
        Ok(())
    }

    fn apply_trait_flag_precedence(&mut self) {
        if self.trait_config.qt == Some(true) {
            self.trait_config.bt = Some(false);
        }
        if self.trait_config.bt == Some(true) {
            self.trait_config.qt = Some(false);
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialInputConfig {
    pub(crate) bgen: Option<String>,
    pub(crate) sample: Option<String>,
    #[serde(rename = "phenoFile")]
    pub(crate) pheno_file: Option<String>,
    #[serde(rename = "phenoCol")]
    pub(crate) pheno_col: Option<NameList>,
    #[serde(rename = "phenoColList")]
    pub(crate) pheno_col_list: Option<NameList>,
    #[serde(rename = "covarFile")]
    pub(crate) covar_file: Option<String>,
    #[serde(rename = "covarCol")]
    pub(crate) covar_col: Option<NameList>,
    #[serde(rename = "covarColList")]
    pub(crate) covar_col_list: Option<NameList>,
    pub(crate) pred: Option<String>,
}

impl PartialInputConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, bgen);
        overlay_option!(self, override_config, sample);
        overlay_option!(self, override_config, pheno_file);
        overlay_option!(self, override_config, pheno_col);
        overlay_option!(self, override_config, pheno_col_list);
        overlay_option!(self, override_config, covar_file);
        overlay_option!(self, override_config, covar_col);
        overlay_option!(self, override_config, covar_col_list);
        overlay_option!(self, override_config, pred);
    }

    fn resolve(self) -> ConfigResult<InputConfigData> {
        Ok(InputConfigData {
            bgen: self.bgen,
            sample: self.sample,
            pheno_file: self.pheno_file,
            pheno_columns: resolve_column_options(self.pheno_col, self.pheno_col_list, "phenoCol", "phenoColList")?,
            covar_file: self.covar_file,
            covar_columns: resolve_column_options(self.covar_col, self.covar_col_list, "covarCol", "covarColList")?,
            pred: self.pred,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialTraitConfig {
    pub(crate) step: Option<u8>,
    pub(crate) qt: Option<bool>,
    pub(crate) bt: Option<bool>,
    pub(crate) bsize: Option<PositiveU32>,
    pub(crate) threads: Option<PositiveU32>,
}

impl PartialTraitConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, step);
        overlay_option!(self, override_config, qt);
        overlay_option!(self, override_config, bt);
        overlay_option!(self, override_config, bsize);
        overlay_option!(self, override_config, threads);
    }

    fn resolve(self) -> ConfigResult<TraitConfigData> {
        Ok(TraitConfigData {
            step: required("step", self.step)?,
            trait_type: normalize_trait_type(self.qt, self.bt)?,
            bsize: required("bsize", self.bsize)?.get(),
            threads: self.threads.map(PositiveU32::get),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialBinaryConfig {
    pub(crate) firth: Option<bool>,
    pub(crate) approx: Option<bool>,
    #[serde(rename = "pThresh")]
    pub(crate) p_threshold: Option<Probability>,
    #[serde(rename = "firth-se")]
    pub(crate) firth_se: Option<bool>,
}

impl PartialBinaryConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, firth);
        overlay_option!(self, override_config, approx);
        overlay_option!(self, override_config, p_threshold);
        overlay_option!(self, override_config, firth_se);
    }

    fn resolve(self) -> ConfigResult<BinaryConfigData> {
        Ok(BinaryConfigData {
            firth: required("firth", self.firth)?,
            approx: required("approx", self.approx)?,
            spa: false,
            p_threshold: required("pThresh", self.p_threshold)?.get(),
            firth_se: required("firth-se", self.firth_se)?,
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialComputeConfig {
    pub(crate) device: Option<DeviceValue>,
    pub(crate) staging_depth: Option<PositiveU32>,
    pub(crate) variant_limit: Option<PositiveU32>,
    pub(crate) trusted_no_missing_diploid: Option<bool>,
    pub(crate) trusted_bgen_validation_mode: Option<TrustedBgenValidationModeValue>,
    pub(crate) sample_key_mode: Option<SampleKeyModeValue>,
    pub(crate) multi_phenotype_sample_mode: Option<MultiPhenotypeSampleModeValue>,
    pub(crate) firth_batch_size: Option<PositiveU32>,
    pub(crate) firth_candidate_capacity: Option<PositiveU32>,
    pub(crate) binary_null_maximum_iterations: Option<PositiveU32>,
    pub(crate) binary_null_coefficient_tolerance: Option<PositiveF32>,
    pub(crate) null_logistic_nonconvergence_policy: Option<NullLogisticNonconvergencePolicyValue>,
    pub(crate) binary_minimum_probability: Option<ProbabilityFloor>,
    pub(crate) binary_minimum_variance: Option<PositiveF32>,
    pub(crate) binary_relative_variance_tolerance: Option<PositiveF32>,
    pub(crate) linear_minimum_variance: Option<PositiveF32>,
    pub(crate) linear_relative_variance_tolerance: Option<PositiveF32>,
    pub(crate) firth_maximum_iterations: Option<PositiveU32>,
    pub(crate) firth_gradient_tolerance: Option<PositiveF32>,
    pub(crate) firth_coefficient_tolerance: Option<PositiveF32>,
    pub(crate) firth_likelihood_tolerance: Option<PositiveF32>,
    pub(crate) firth_maximum_step_size: Option<PositiveF32>,
    pub(crate) firth_pseudo_maximum_iterations: Option<PositiveU32>,
    pub(crate) firth_pseudo_inner_maximum_iterations: Option<PositiveU32>,
    pub(crate) firth_newton_raphson_zero_start_iterations: Option<PositiveU32>,
    pub(crate) firth_line_search_maximum_attempts: Option<PositiveU32>,
    pub(crate) firth_step_halving_maximum_attempts: Option<PositiveU32>,
    pub(crate) firth_initial_response_scale: Option<PositiveF32>,
    pub(crate) firth_sparse_carrier_dosage_threshold: Option<PositiveF32>,
    pub(crate) firth_step_halving_scale: Option<PositiveF32>,
    pub(crate) null_firth_maximum_iterations: Option<PositiveU32>,
    pub(crate) null_firth_gradient_tolerance: Option<PositiveF32>,
    pub(crate) null_firth_maximum_step_size: Option<PositiveF32>,
    pub(crate) null_firth_fallback_iteration_multiplier: Option<PositiveU32>,
    pub(crate) null_firth_fallback_step_divisor: Option<PositiveF32>,
    pub(crate) null_firth_line_search_maximum_attempts: Option<PositiveU32>,
    pub(crate) null_firth_step_halving_scale: Option<PositiveF32>,
    pub(crate) use_block_firth_math: Option<bool>,
    pub(crate) bgen_decode_tile_variant_count: Option<PositiveU32>,
    pub(crate) gpu_genotype_format: Option<GpuGenotypeFormatValue>,
    pub(crate) score_dtype: Option<FloatingPointDtypeValue>,
    pub(crate) firth_dtype: Option<FloatingPointDtypeValue>,
    pub(crate) jax_cache_dir: Option<String>,
    pub(crate) jax_matmul_precision: Option<JaxMatmulPrecisionValue>,
    pub(crate) jax_persistent_cache: Option<bool>,
    pub(crate) jax_persistent_cache_min_entry_size_bytes: Option<i64>,
    pub(crate) jax_persistent_cache_min_compile_time_seconds: Option<NonNegativeU32>,
    pub(crate) jax_xla_autotune_cache: Option<bool>,
    pub(crate) jax_transfer_guard: Option<bool>,
}

impl PartialComputeConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, device);
        overlay_option!(self, override_config, staging_depth);
        overlay_option!(self, override_config, variant_limit);
        overlay_option!(self, override_config, trusted_no_missing_diploid);
        overlay_option!(self, override_config, trusted_bgen_validation_mode);
        overlay_option!(self, override_config, sample_key_mode);
        overlay_option!(self, override_config, multi_phenotype_sample_mode);
        overlay_option!(self, override_config, firth_batch_size);
        overlay_option!(self, override_config, firth_candidate_capacity);
        overlay_option!(self, override_config, binary_null_maximum_iterations);
        overlay_option!(self, override_config, binary_null_coefficient_tolerance);
        overlay_option!(self, override_config, null_logistic_nonconvergence_policy);
        overlay_option!(self, override_config, binary_minimum_probability);
        overlay_option!(self, override_config, binary_minimum_variance);
        overlay_option!(self, override_config, binary_relative_variance_tolerance);
        overlay_option!(self, override_config, linear_minimum_variance);
        overlay_option!(self, override_config, linear_relative_variance_tolerance);
        overlay_option!(self, override_config, firth_maximum_iterations);
        overlay_option!(self, override_config, firth_gradient_tolerance);
        overlay_option!(self, override_config, firth_coefficient_tolerance);
        overlay_option!(self, override_config, firth_likelihood_tolerance);
        overlay_option!(self, override_config, firth_maximum_step_size);
        overlay_option!(self, override_config, firth_pseudo_maximum_iterations);
        overlay_option!(self, override_config, firth_pseudo_inner_maximum_iterations);
        overlay_option!(self, override_config, firth_newton_raphson_zero_start_iterations);
        overlay_option!(self, override_config, firth_line_search_maximum_attempts);
        overlay_option!(self, override_config, firth_step_halving_maximum_attempts);
        overlay_option!(self, override_config, firth_initial_response_scale);
        overlay_option!(self, override_config, firth_sparse_carrier_dosage_threshold);
        overlay_option!(self, override_config, firth_step_halving_scale);
        overlay_option!(self, override_config, null_firth_maximum_iterations);
        overlay_option!(self, override_config, null_firth_gradient_tolerance);
        overlay_option!(self, override_config, null_firth_maximum_step_size);
        overlay_option!(self, override_config, null_firth_fallback_iteration_multiplier);
        overlay_option!(self, override_config, null_firth_fallback_step_divisor);
        overlay_option!(self, override_config, null_firth_line_search_maximum_attempts);
        overlay_option!(self, override_config, null_firth_step_halving_scale);
        overlay_option!(self, override_config, use_block_firth_math);
        overlay_option!(self, override_config, bgen_decode_tile_variant_count);
        overlay_option!(self, override_config, gpu_genotype_format);
        overlay_option!(self, override_config, score_dtype);
        overlay_option!(self, override_config, firth_dtype);
        overlay_option!(self, override_config, jax_cache_dir);
        overlay_option!(self, override_config, jax_matmul_precision);
        overlay_option!(self, override_config, jax_persistent_cache);
        overlay_option!(self, override_config, jax_persistent_cache_min_entry_size_bytes);
        overlay_option!(self, override_config, jax_persistent_cache_min_compile_time_seconds);
        overlay_option!(self, override_config, jax_xla_autotune_cache);
        overlay_option!(self, override_config, jax_transfer_guard);
    }

    fn resolve(&self) -> ConfigResult<GComputeConfigData> {
        let core = self.resolve_core_fields()?;
        let firth = self.resolve_firth_fields()?;
        let null_firth = self.resolve_null_firth_fields()?;
        let genotype = self.resolve_genotype_fields()?;
        let jax = self.resolve_jax_fields()?;
        Ok(GComputeConfigData {
            device: core.device,
            staging_depth: core.staging_depth,
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
            firth_dtype: genotype.firth_dtype,
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
            device: required("device", self.device)?.as_str().to_string(),
            staging_depth: required("staging_depth", self.staging_depth)?.get(),
            variant_limit: self.variant_limit.map(PositiveU32::get),
            trusted_no_missing_diploid: required("trusted_no_missing_diploid", self.trusted_no_missing_diploid)?,
            trusted_bgen_validation_mode: required("trusted_bgen_validation_mode", self.trusted_bgen_validation_mode)?
                .as_str()
                .to_string(),
            sample_key_mode: required("sample_key_mode", self.sample_key_mode)?.as_str().to_string(),
            multi_phenotype_sample_mode: required("multi_phenotype_sample_mode", self.multi_phenotype_sample_mode)?
                .as_str()
                .to_string(),
            firth_batch_size: required("firth_batch_size", self.firth_batch_size)?.get(),
            firth_candidate_capacity: required("firth_candidate_capacity", self.firth_candidate_capacity)?.get(),
            binary_null_maximum_iterations: required(
                "binary_null_maximum_iterations",
                self.binary_null_maximum_iterations,
            )?
            .get(),
            binary_null_coefficient_tolerance: required(
                "binary_null_coefficient_tolerance",
                self.binary_null_coefficient_tolerance,
            )?
            .get(),
            null_logistic_nonconvergence_policy: required(
                "null_logistic_nonconvergence_policy",
                self.null_logistic_nonconvergence_policy,
            )?
            .as_str()
            .to_string(),
            binary_minimum_probability: required("binary_minimum_probability", self.binary_minimum_probability)?.get(),
            binary_minimum_variance: required("binary_minimum_variance", self.binary_minimum_variance)?.get(),
            binary_relative_variance_tolerance: required(
                "binary_relative_variance_tolerance",
                self.binary_relative_variance_tolerance,
            )?
            .get(),
            linear_minimum_variance: required("linear_minimum_variance", self.linear_minimum_variance)?.get(),
            linear_relative_variance_tolerance: required(
                "linear_relative_variance_tolerance",
                self.linear_relative_variance_tolerance,
            )?
            .get(),
        })
    }

    fn resolve_firth_fields(&self) -> ConfigResult<ResolvedFirthFields> {
        Ok(ResolvedFirthFields {
            maximum_iterations: required("firth_maximum_iterations", self.firth_maximum_iterations)?.get(),
            gradient_tolerance: required("firth_gradient_tolerance", self.firth_gradient_tolerance)?.get(),
            coefficient_tolerance: required("firth_coefficient_tolerance", self.firth_coefficient_tolerance)?.get(),
            likelihood_tolerance: required("firth_likelihood_tolerance", self.firth_likelihood_tolerance)?.get(),
            maximum_step_size: required("firth_maximum_step_size", self.firth_maximum_step_size)?.get(),
            pseudo_maximum_iterations: required(
                "firth_pseudo_maximum_iterations",
                self.firth_pseudo_maximum_iterations,
            )?
            .get(),
            pseudo_inner_maximum_iterations: required(
                "firth_pseudo_inner_maximum_iterations",
                self.firth_pseudo_inner_maximum_iterations,
            )?
            .get(),
            newton_raphson_zero_start_iterations: required(
                "firth_newton_raphson_zero_start_iterations",
                self.firth_newton_raphson_zero_start_iterations,
            )?
            .get(),
            line_search_maximum_attempts: required(
                "firth_line_search_maximum_attempts",
                self.firth_line_search_maximum_attempts,
            )?
            .get(),
            step_halving_maximum_attempts: required(
                "firth_step_halving_maximum_attempts",
                self.firth_step_halving_maximum_attempts,
            )?
            .get(),
            initial_response_scale: required("firth_initial_response_scale", self.firth_initial_response_scale)?.get(),
            sparse_carrier_dosage_threshold: required(
                "firth_sparse_carrier_dosage_threshold",
                self.firth_sparse_carrier_dosage_threshold,
            )?
            .get(),
            step_halving_scale: required("firth_step_halving_scale", self.firth_step_halving_scale)?.get(),
        })
    }

    fn resolve_null_firth_fields(&self) -> ConfigResult<ResolvedNullFirthFields> {
        Ok(ResolvedNullFirthFields {
            maximum_iterations: required("null_firth_maximum_iterations", self.null_firth_maximum_iterations)?.get(),
            gradient_tolerance: required("null_firth_gradient_tolerance", self.null_firth_gradient_tolerance)?.get(),
            maximum_step_size: required("null_firth_maximum_step_size", self.null_firth_maximum_step_size)?.get(),
            fallback_iteration_multiplier: required(
                "null_firth_fallback_iteration_multiplier",
                self.null_firth_fallback_iteration_multiplier,
            )?
            .get(),
            fallback_step_divisor: required("null_firth_fallback_step_divisor", self.null_firth_fallback_step_divisor)?
                .get(),
            line_search_maximum_attempts: required(
                "null_firth_line_search_maximum_attempts",
                self.null_firth_line_search_maximum_attempts,
            )?
            .get(),
            step_halving_scale: required("null_firth_step_halving_scale", self.null_firth_step_halving_scale)?.get(),
        })
    }

    fn resolve_genotype_fields(&self) -> ConfigResult<ResolvedGenotypeFields> {
        Ok(ResolvedGenotypeFields {
            use_block_firth_math: required("use_block_firth_math", self.use_block_firth_math)?,
            bgen_decode_tile_variant_count: required(
                "bgen_decode_tile_variant_count",
                self.bgen_decode_tile_variant_count,
            )?
            .get(),
            gpu_genotype_format: required("gpu_genotype_format", self.gpu_genotype_format)?.as_str().to_string(),
            score_dtype: required("score_dtype", self.score_dtype)?.as_str().to_string(),
            firth_dtype: required("firth_dtype", self.firth_dtype)?.as_str().to_string(),
        })
    }

    fn resolve_jax_fields(&self) -> ConfigResult<ResolvedJaxFields> {
        Ok(ResolvedJaxFields {
            cache_dir: self.jax_cache_dir.clone(),
            matmul_precision: self.jax_matmul_precision.map(|value| value.as_str().to_string()),
            persistent_cache: required("jax_persistent_cache", self.jax_persistent_cache)?,
            persistent_cache_min_entry_size_bytes: required(
                "jax_persistent_cache_min_entry_size_bytes",
                self.jax_persistent_cache_min_entry_size_bytes,
            )?,
            persistent_cache_min_compile_time_seconds: required(
                "jax_persistent_cache_min_compile_time_seconds",
                self.jax_persistent_cache_min_compile_time_seconds,
            )?
            .get(),
            xla_autotune_cache: required("jax_xla_autotune_cache", self.jax_xla_autotune_cache)?,
            transfer_guard: required("jax_transfer_guard", self.jax_transfer_guard)?,
        })
    }
}

struct ResolvedComputeCoreFields {
    device: String,
    staging_depth: u32,
    variant_limit: Option<u32>,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: String,
    sample_key_mode: String,
    multi_phenotype_sample_mode: String,
    firth_batch_size: u32,
    firth_candidate_capacity: u32,
    binary_null_maximum_iterations: u32,
    binary_null_coefficient_tolerance: f32,
    null_logistic_nonconvergence_policy: String,
    binary_minimum_probability: f32,
    binary_minimum_variance: f32,
    binary_relative_variance_tolerance: f32,
    linear_minimum_variance: f32,
    linear_relative_variance_tolerance: f32,
}

struct ResolvedFirthFields {
    maximum_iterations: u32,
    gradient_tolerance: f32,
    coefficient_tolerance: f32,
    likelihood_tolerance: f32,
    maximum_step_size: f32,
    pseudo_maximum_iterations: u32,
    pseudo_inner_maximum_iterations: u32,
    newton_raphson_zero_start_iterations: u32,
    line_search_maximum_attempts: u32,
    step_halving_maximum_attempts: u32,
    initial_response_scale: f32,
    sparse_carrier_dosage_threshold: f32,
    step_halving_scale: f32,
}

struct ResolvedNullFirthFields {
    maximum_iterations: u32,
    gradient_tolerance: f32,
    maximum_step_size: f32,
    fallback_iteration_multiplier: u32,
    fallback_step_divisor: f32,
    line_search_maximum_attempts: u32,
    step_halving_scale: f32,
}

struct ResolvedGenotypeFields {
    use_block_firth_math: bool,
    bgen_decode_tile_variant_count: u32,
    gpu_genotype_format: String,
    score_dtype: String,
    firth_dtype: String,
}

struct ResolvedJaxFields {
    cache_dir: Option<String>,
    matmul_precision: Option<String>,
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
    pub(crate) format: Option<OutputFormatValue>,
    pub(crate) output_run_directory: Option<String>,
    pub(crate) writer_threads: Option<PositiveU32>,
    pub(crate) writer_queue_depth: Option<PositiveU32>,
    pub(crate) chunks_per_arrow_file: Option<PositiveU32>,
    pub(crate) arrow_compression: Option<ArrowCompressionValue>,
    pub(crate) parquet_compression: Option<ParquetCompressionValue>,
    pub(crate) resume: Option<bool>,
    pub(crate) resume_mode: Option<ResumeModeValue>,
    pub(crate) finalize_parquet: Option<bool>,
}

impl PartialOutputConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, out);
        overlay_option!(self, override_config, format);
        overlay_option!(self, override_config, output_run_directory);
        overlay_option!(self, override_config, writer_threads);
        overlay_option!(self, override_config, writer_queue_depth);
        overlay_option!(self, override_config, chunks_per_arrow_file);
        overlay_option!(self, override_config, arrow_compression);
        overlay_option!(self, override_config, parquet_compression);
        overlay_option!(self, override_config, resume);
        overlay_option!(self, override_config, resume_mode);
        overlay_option!(self, override_config, finalize_parquet);
    }

    fn resolve(self) -> ConfigResult<GOutputConfigData> {
        Ok(GOutputConfigData {
            out: self.out,
            format: required("format", self.format)?.as_str().to_string(),
            output_run_directory: self.output_run_directory,
            writer_threads: required("writer_threads", self.writer_threads)?.get(),
            writer_queue_depth: required("writer_queue_depth", self.writer_queue_depth)?.get(),
            chunks_per_arrow_file: required("chunks_per_arrow_file", self.chunks_per_arrow_file)?.get(),
            arrow_compression: required("arrow_compression", self.arrow_compression)?.as_str().to_string(),
            parquet_compression: required("parquet_compression", self.parquet_compression)?.as_str().to_string(),
            resume: required("resume", self.resume)?,
            resume_mode: required("resume_mode", self.resume_mode)?.as_str().to_string(),
            finalize_parquet: required("finalize_parquet", self.finalize_parquet)?,
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct PartialDiagnosticsConfig {
    pub(crate) telemetry: Option<TelemetryModeValue>,
    pub(crate) log_dir: Option<String>,
    pub(crate) stage_timings_json: Option<String>,
    pub(crate) log_filter: Option<String>,
    pub(crate) log_file: Option<String>,
    pub(crate) log_stderr: Option<bool>,
    pub(crate) progress_interval_seconds: Option<PositiveF32>,
    pub(crate) progress_interval_chunks: Option<PositiveU32>,
    pub(crate) profile_summary_json: Option<String>,
    pub(crate) trace_file: Option<String>,
    pub(crate) trace_filter: Option<String>,
    pub(crate) trace_event_cap: Option<NonNegativeU32>,
    pub(crate) log_queue_size: Option<PositiveU32>,
    pub(crate) log_lossy: Option<bool>,
    pub(crate) include_source_location: Option<bool>,
    pub(crate) include_span_events: Option<bool>,
}

impl PartialDiagnosticsConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, telemetry);
        overlay_option!(self, override_config, log_dir);
        overlay_option!(self, override_config, stage_timings_json);
        overlay_option!(self, override_config, log_filter);
        overlay_option!(self, override_config, log_file);
        overlay_option!(self, override_config, log_stderr);
        overlay_option!(self, override_config, progress_interval_seconds);
        overlay_option!(self, override_config, progress_interval_chunks);
        overlay_option!(self, override_config, profile_summary_json);
        overlay_option!(self, override_config, trace_file);
        overlay_option!(self, override_config, trace_filter);
        overlay_option!(self, override_config, trace_event_cap);
        overlay_option!(self, override_config, log_queue_size);
        overlay_option!(self, override_config, log_lossy);
        overlay_option!(self, override_config, include_source_location);
        overlay_option!(self, override_config, include_span_events);
    }

    fn resolve(self) -> ConfigResult<GDiagnosticsConfigData> {
        Ok(GDiagnosticsConfigData {
            telemetry: required("telemetry", self.telemetry)?.as_str().to_string(),
            log_dir: self.log_dir,
            stage_timings_json: self.stage_timings_json,
            log_filter: required("log_filter", self.log_filter)?,
            log_file: self.log_file,
            log_stderr: required("log_stderr", self.log_stderr)?,
            progress_interval_seconds: required("progress_interval_seconds", self.progress_interval_seconds)?.get(),
            progress_interval_chunks: required("progress_interval_chunks", self.progress_interval_chunks)?.get(),
            profile_summary_json: self.profile_summary_json,
            trace_file: self.trace_file,
            trace_filter: required("trace_filter", self.trace_filter)?,
            trace_event_cap: required("trace_event_cap", self.trace_event_cap)?.get(),
            log_queue_size: required("log_queue_size", self.log_queue_size)?.get(),
            log_lossy: required("log_lossy", self.log_lossy)?,
            include_source_location: required("include_source_location", self.include_source_location)?,
            include_span_events: required("include_span_events", self.include_span_events)?,
        })
    }
}

fn required<ValueType>(option_name: &str, value: Option<ValueType>) -> ConfigResult<ValueType> {
    value.ok_or_else(|| ConfigError::new(format!("Default config is missing required default option {option_name:?}.")))
}

fn resolve_column_options(
    repeated_columns: Option<NameList>,
    list_columns: Option<NameList>,
    repeated_option_name: &str,
    list_option_name: &str,
) -> ConfigResult<Vec<String>> {
    let repeated_values = repeated_columns.map(NameList::into_vec).unwrap_or_default();
    let list_values = list_columns.map(NameList::into_vec).unwrap_or_default();
    if !repeated_values.is_empty() && !list_values.is_empty() {
        return Err(ConfigError::new(format!(
            "Use either --{repeated_option_name} or --{list_option_name}, not both."
        )));
    }
    if repeated_values.is_empty() { Ok(list_values) } else { Ok(repeated_values) }
}

pub(crate) fn normalize_trait_type(qt: Option<bool>, bt: Option<bool>) -> ConfigResult<String> {
    if qt == Some(true) && bt == Some(true) {
        return Err(ConfigError::new("--qt and --bt are mutually exclusive."));
    }
    if bt == Some(true) {
        return Ok("binary".to_string());
    }
    Ok("quantitative".to_string())
}
