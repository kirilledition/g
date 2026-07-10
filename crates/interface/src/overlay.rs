use super::defaults::load_default_config_data;
use super::partial::{
    PartialBinaryConfig, PartialComputeConfig, PartialConfig, PartialDiagnosticsConfig, PartialInputConfig,
    PartialOutputConfig, PartialTraitConfig,
};
use super::resolved::{ConfigProvenance, RegenieConfigData};
use super::validation::validate_config;
use super::{ConfigError, ConfigResult};

macro_rules! overlay_option {
    ($target:expr, $source:expr, $field:ident) => {
        if $source.$field.is_some() {
            $target.$field = $source.$field;
        }
    };
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ConfigLayer {
    partial_config: PartialConfig,
    provenance: ConfigProvenance,
}

impl ConfigLayer {
    pub(crate) fn from_partial_config(partial_config: PartialConfig) -> Self {
        let provenance = ConfigProvenance::from_partial_config(&partial_config);
        Self::from_partial_config_with_provenance(partial_config, provenance)
    }

    pub(crate) fn from_partial_config_with_provenance(
        partial_config: PartialConfig,
        provenance: ConfigProvenance,
    ) -> Self {
        Self { partial_config, provenance }
    }
}

pub(crate) fn resolve_config_layers(
    explicit_layers: impl IntoIterator<Item = ConfigLayer>,
) -> ConfigResult<RegenieConfigData> {
    let mut merged_config = load_default_config_data()?.partial_config.clone();
    let mut merged_provenance = ConfigProvenance::default();
    for explicit_layer in explicit_layers {
        merged_config.overlay(explicit_layer.partial_config)?;
        merged_provenance.overlay(explicit_layer.provenance);
    }
    resolve_partial_config(merged_config, merged_provenance, true)
}

pub(crate) fn resolve_partial_config(
    partial_config: PartialConfig,
    provenance: ConfigProvenance,
    validate: bool,
) -> ConfigResult<RegenieConfigData> {
    let config = partial_config.resolve(provenance)?;
    if validate {
        validate_config(&config)?;
    }
    Ok(config)
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

    fn reject_trait_flag_conflict(&self) -> ConfigResult<()> {
        if self.trait_config.qt == Some(true) && self.trait_config.bt == Some(true) {
            return Err(ConfigError::new("--qt and --bt are mutually exclusive."));
        }
        Ok(())
    }

    fn apply_trait_flag_precedence(&mut self) {
        if self.trait_config.bt == Some(true) {
            self.trait_config.qt = Some(false);
        } else if self.trait_config.qt == Some(true) {
            self.trait_config.bt = Some(false);
        }
    }
}

impl PartialInputConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, bgen);
        overlay_option!(self, override_config, sample);
        overlay_option!(self, override_config, pheno_file);
        overlay_option!(self, override_config, pheno_columns);
        overlay_option!(self, override_config, covar_file);
        overlay_option!(self, override_config, covar_columns);
        overlay_option!(self, override_config, pred);
    }
}

impl PartialTraitConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, trait_type);
        overlay_option!(self, override_config, qt);
        overlay_option!(self, override_config, bt);
        overlay_option!(self, override_config, bsize);
        overlay_option!(self, override_config, threads);
    }
}

impl PartialBinaryConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, fallback_method);
        overlay_option!(self, override_config, p_threshold);
        overlay_option!(self, override_config, firth_se);
    }
}

impl PartialComputeConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, device);
        overlay_option!(self, override_config, staging_depth);
        overlay_option!(self, override_config, result_in_flight_limit);
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
        overlay_option!(self, override_config, jax_cache_dir);
        overlay_option!(self, override_config, jax_matmul_precision);
        overlay_option!(self, override_config, jax_persistent_cache);
        overlay_option!(self, override_config, jax_persistent_cache_min_entry_size_bytes);
        overlay_option!(self, override_config, jax_persistent_cache_min_compile_time_seconds);
        overlay_option!(self, override_config, jax_xla_autotune_cache);
        overlay_option!(self, override_config, jax_transfer_guard);
    }
}

impl PartialOutputConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, out);
        overlay_option!(self, override_config, output_run_directory);
        overlay_option!(self, override_config, writer_threads);
        overlay_option!(self, override_config, writer_queue_depth);
        overlay_option!(self, override_config, chunks_per_parquet_file);
        overlay_option!(self, override_config, parquet_compression);
        overlay_option!(self, override_config, output_statistic_dtype);
        overlay_option!(self, override_config, resume);
        overlay_option!(self, override_config, resume_mode);
    }
}

impl PartialDiagnosticsConfig {
    fn overlay(&mut self, override_config: Self) {
        overlay_option!(self, override_config, telemetry);
        overlay_option!(self, override_config, log_dir);
        overlay_option!(self, override_config, stage_timings_json);
        overlay_option!(self, override_config, log_filter);
        overlay_option!(self, override_config, log_file);
        overlay_option!(self, override_config, log_stderr);
        overlay_option!(self, override_config, profile_summary_json);
        overlay_option!(self, override_config, trace_file);
        overlay_option!(self, override_config, trace_filter);
        overlay_option!(self, override_config, trace_event_cap);
        overlay_option!(self, override_config, log_queue_size);
        overlay_option!(self, override_config, log_lossy);
        overlay_option!(self, override_config, include_source_location);
        overlay_option!(self, override_config, include_span_events);
    }
}
