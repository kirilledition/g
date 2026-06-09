use super::super::domain::NameList;
use super::super::overlay::ConfigLayer;
use super::super::partial::{
    PartialBinaryConfig, PartialComputeConfig, PartialConfig, PartialDiagnosticsConfig, PartialInputConfig,
    PartialOutputConfig, PartialTraitConfig,
};
use super::super::{ConfigError, ConfigResult};
use super::parser::RegenieCli;

impl RegenieCli {
    pub(crate) fn into_config_layer(self) -> ConfigResult<ConfigLayer> {
        let partial_config = PartialConfig {
            input: self.input_config(),
            trait_config: self.trait_config()?,
            binary: self.binary_config()?,
            compute: self.compute_config()?,
            output: self.output_config()?,
            diagnostics: self.diagnostics_config()?,
            metadata: None,
        };
        Ok(ConfigLayer::from_partial_config(partial_config))
    }

    fn input_config(&self) -> PartialInputConfig {
        PartialInputConfig {
            bgen: self.input.bgen.clone(),
            sample: self.input.sample.clone(),
            pheno_file: self.input.pheno_file.clone(),
            pheno_columns: repeated_name_list(&self.input.pheno_col),
            pheno_col: None,
            pheno_col_list: self.input.pheno_col_list.clone(),
            covar_file: self.input.covar_file.clone(),
            covar_columns: repeated_name_list(&self.input.covar_col),
            covar_col: None,
            covar_col_list: self.input.covar_col_list.clone(),
            pred: self.input.pred.clone(),
        }
    }

    fn trait_config(&self) -> ConfigResult<PartialTraitConfig> {
        Ok(PartialTraitConfig {
            step: self.trait_options.step,
            trait_type: None,
            qt: optional_flag("qt", self.trait_options.qt, self.trait_options.no_qt)?,
            bt: optional_flag("bt", self.trait_options.bt, self.trait_options.no_bt)?,
            bsize: self.trait_options.bsize,
            threads: self.trait_options.threads,
        })
    }

    fn binary_config(&self) -> ConfigResult<PartialBinaryConfig> {
        Ok(PartialBinaryConfig {
            firth: optional_flag("firth", self.binary.firth, self.binary.no_firth)?,
            approx: optional_flag("approx", self.binary.approx, self.binary.no_approx)?,
            p_threshold: self.binary.p_threshold,
            firth_se: optional_flag("firth-se", self.binary.firth_se, self.binary.no_firth_se)?,
        })
    }

    fn output_config(&self) -> ConfigResult<PartialOutputConfig> {
        Ok(PartialOutputConfig {
            out: self.output.out.clone(),
            format: self.output.format,
            output_run_directory: self.output.output_run_directory.clone(),
            writer_threads: self.output.writer_threads,
            writer_queue_depth: self.output.writer_queue_depth,
            chunks_per_arrow_file: self.output.chunks_per_arrow_file,
            arrow_compression: self.output.arrow_compression,
            parquet_compression: self.output.parquet_compression,
            resume: optional_flag("resume", self.output.resume, self.output.no_resume)?,
            resume_mode: self.output.resume_mode,
            finalize_parquet: optional_flag(
                "finalize_parquet",
                self.output.finalize_parquet,
                self.output.no_finalize_parquet,
            )?,
        })
    }

    fn compute_config(&self) -> ConfigResult<PartialComputeConfig> {
        Ok(PartialComputeConfig {
            device: self.compute.device,
            staging_depth: self.compute.staging_depth,
            result_in_flight_limit: self.compute.result_in_flight_limit,
            dosage_buffer_limit: self.compute.dosage_buffer_limit,
            variant_limit: self.compute.variant_limit,
            trusted_no_missing_diploid: optional_flag(
                "trusted_no_missing_diploid",
                self.compute.trusted_no_missing_diploid,
                self.compute.no_trusted_no_missing_diploid,
            )?,
            trusted_bgen_validation_mode: self.compute.trusted_bgen_validation_mode,
            sample_key_mode: self.compute.sample_key_mode,
            multi_phenotype_sample_mode: self.compute.multi_phenotype_sample_mode,
            firth_batch_size: self.compute.firth_batch_size,
            firth_candidate_capacity: self.compute.firth_candidate_capacity,
            binary_null_maximum_iterations: self.compute.binary_null_maximum_iterations,
            binary_null_coefficient_tolerance: self.compute.binary_null_coefficient_tolerance,
            null_logistic_nonconvergence_policy: self.compute.null_logistic_nonconvergence_policy,
            binary_minimum_probability: self.compute.binary_minimum_probability,
            binary_minimum_variance: self.compute.binary_minimum_variance,
            binary_relative_variance_tolerance: self.compute.binary_relative_variance_tolerance,
            linear_minimum_variance: self.compute.linear_minimum_variance,
            linear_relative_variance_tolerance: self.compute.linear_relative_variance_tolerance,
            firth_maximum_iterations: self.compute.firth_maximum_iterations,
            firth_gradient_tolerance: self.compute.firth_gradient_tolerance,
            firth_coefficient_tolerance: self.compute.firth_coefficient_tolerance,
            firth_likelihood_tolerance: self.compute.firth_likelihood_tolerance,
            firth_maximum_step_size: self.compute.firth_maximum_step_size,
            firth_pseudo_maximum_iterations: self.compute.firth_pseudo_maximum_iterations,
            firth_pseudo_inner_maximum_iterations: self.compute.firth_pseudo_inner_maximum_iterations,
            firth_newton_raphson_zero_start_iterations: self.compute.firth_newton_raphson_zero_start_iterations,
            firth_line_search_maximum_attempts: self.compute.firth_line_search_maximum_attempts,
            firth_step_halving_maximum_attempts: self.compute.firth_step_halving_maximum_attempts,
            firth_initial_response_scale: self.compute.firth_initial_response_scale,
            firth_sparse_carrier_dosage_threshold: self.compute.firth_sparse_carrier_dosage_threshold,
            firth_step_halving_scale: self.compute.firth_step_halving_scale,
            null_firth_maximum_iterations: self.compute.null_firth_maximum_iterations,
            null_firth_gradient_tolerance: self.compute.null_firth_gradient_tolerance,
            null_firth_maximum_step_size: self.compute.null_firth_maximum_step_size,
            null_firth_fallback_iteration_multiplier: self.compute.null_firth_fallback_iteration_multiplier,
            null_firth_fallback_step_divisor: self.compute.null_firth_fallback_step_divisor,
            null_firth_line_search_maximum_attempts: self.compute.null_firth_line_search_maximum_attempts,
            null_firth_step_halving_scale: self.compute.null_firth_step_halving_scale,
            use_block_firth_math: optional_flag(
                "use_block_firth_math",
                self.compute.use_block_firth_math,
                self.compute.no_use_block_firth_math,
            )?,
            bgen_decode_tile_variant_count: self.compute.bgen_decode_tile_variant_count,
            gpu_genotype_format: self.compute.gpu_genotype_format,
            score_dtype: self.compute.score_dtype,
            firth_dtype: self.compute.firth_dtype,
            jax_cache_dir: self.compute.jax_cache_dir.clone(),
            jax_matmul_precision: self.compute.jax_matmul_precision,
            jax_persistent_cache: optional_flag(
                "jax_persistent_cache",
                self.compute.jax_persistent_cache,
                self.compute.no_jax_persistent_cache,
            )?,
            jax_persistent_cache_min_entry_size_bytes: self.compute.jax_persistent_cache_min_entry_size_bytes,
            jax_persistent_cache_min_compile_time_seconds: self.compute.jax_persistent_cache_min_compile_time_seconds,
            jax_xla_autotune_cache: optional_flag(
                "jax_xla_autotune_cache",
                self.compute.jax_xla_autotune_cache,
                self.compute.no_jax_xla_autotune_cache,
            )?,
            jax_transfer_guard: optional_flag(
                "jax_transfer_guard",
                self.compute.jax_transfer_guard,
                self.compute.no_jax_transfer_guard,
            )?,
        })
    }

    fn diagnostics_config(&self) -> ConfigResult<PartialDiagnosticsConfig> {
        Ok(PartialDiagnosticsConfig {
            telemetry: self.diagnostics.telemetry,
            log_dir: self.diagnostics.log_dir.clone(),
            stage_timings_json: self.diagnostics.stage_timings_json.clone(),
            log_filter: self.diagnostics.log_filter.clone(),
            log_file: self.diagnostics.log_file.clone(),
            log_stderr: optional_flag("log_stderr", self.diagnostics.log_stderr, self.diagnostics.no_log_stderr)?,
            progress_interval_seconds: self.diagnostics.progress_interval_seconds,
            progress_interval_chunks: self.diagnostics.progress_interval_chunks,
            profile_summary_json: self.diagnostics.profile_summary_json.clone(),
            trace_file: self.diagnostics.trace_file.clone(),
            trace_filter: self.diagnostics.trace_filter.clone(),
            trace_event_cap: self.diagnostics.trace_event_cap,
            log_queue_size: self.diagnostics.log_queue_size,
            log_lossy: optional_flag("log_lossy", self.diagnostics.log_lossy, self.diagnostics.no_log_lossy)?,
            include_source_location: optional_flag(
                "include_source_location",
                self.diagnostics.include_source_location,
                self.diagnostics.no_include_source_location,
            )?,
            include_span_events: optional_flag(
                "include_span_events",
                self.diagnostics.include_span_events,
                self.diagnostics.no_include_span_events,
            )?,
        })
    }
}

fn repeated_name_list(values: &[String]) -> Option<NameList> {
    (!values.is_empty()).then(|| NameList::from_values(values.to_vec()))
}

fn optional_flag(option_name: &str, positive_value: bool, negative_value: bool) -> ConfigResult<Option<bool>> {
    if positive_value && negative_value {
        return Err(ConfigError::new(format!("--{option_name} and --no-{option_name} cannot be used together.")));
    }
    Ok(if positive_value { Some(true) } else { negative_value.then_some(false) })
}
