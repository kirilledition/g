pub(super) const RUN_STARTED_EVENT_NAME: &str = "run_started";
pub(super) const RUN_COMPLETED_EVENT_NAME: &str = "run_completed";
pub(super) const RUN_FAILED_EVENT_NAME: &str = "run_failed";
pub(super) const EXECUTION_PLAN_PREPARED_EVENT_NAME: &str = "execution_plan_prepared";
pub(super) const EFFECTIVE_CONFIG_WRITTEN_EVENT_NAME: &str = "effective_config_written";
pub(super) const WRITER_FINISHED_EVENT_NAME: &str = "writer_finished";
pub(super) const PREFLIGHT_COMPLETED_EVENT_NAME: &str = "preflight_completed";
pub(super) const SAMPLE_ALIGNMENT_COMPLETED_EVENT_NAME: &str = "sample_alignment_completed";
pub(super) const PREDICTION_SOURCE_LOADED_EVENT_NAME: &str = "prediction_source_loaded";
pub(super) const MULTI_PHENOTYPE_SAMPLE_SUMMARY_EVENT_NAME: &str = "multi_phenotype_sample_summary";
pub(super) const GPU_GENOTYPE_FORMAT_RESOLVED_EVENT_NAME: &str = "gpu_genotype_format_resolved";
pub(super) const ASSOCIATION_BACKEND_SELECTED_EVENT_NAME: &str = "association_backend_selected";
pub(super) const BGEN_ENGINE_OPENED_EVENT_NAME: &str = "bgen_engine_opened";
pub(super) const BINARY_CORRECTION_SUMMARY_EVENT_NAME: &str = "binary_correction_summary";
pub(super) const RUN_LIFECYCLE_INFO_LEVEL: &str = "info";
pub(super) const RUN_LIFECYCLE_WARN_LEVEL: &str = "warn";
pub(super) const RUN_LIFECYCLE_ERROR_LEVEL: &str = "error";
pub(super) const NATIVE_CLI_STDOUT_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_stdout";
pub(super) const NATIVE_CLI_STDERR_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_stderr";
pub(super) const NATIVE_CLI_INTERRUPTED_LINE_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_interrupted_line";
pub(super) const NATIVE_CLI_FAILED_LINE_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_failed_line";
pub(super) const NATIVE_CLI_COMPLETED_LINE_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_completed_line";
pub(super) const RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_execution_plan_build_started";
pub(super) const RUNNER_EXECUTION_PLAN_PREPARED_DIAGNOSTIC_EVENT_NAME: &str = "runner_execution_plan_prepared";
pub(super) const RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_execution_plan_dispatch_started";
pub(super) const RUNNER_EXECUTION_PLAN_FINALIZATION_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_execution_plan_finalization_started";
pub(super) const RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_MESSAGE: &str = "Building REGENIE execution plan.";
pub(super) const RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching REGENIE execution plan.";
pub(super) const RUNNER_EXECUTION_PLAN_FINALIZATION_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Finalizing REGENIE execution plan.";
pub(super) const PIPELINE_GPU_GENOTYPE_FORMAT_RESOLVED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_gpu_genotype_format_resolved";
pub(super) const PIPELINE_BGEN_ENGINE_OPEN_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_bgen_engine_open_started";
pub(super) const PIPELINE_BGEN_ENGINE_OPENED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_bgen_engine_opened";
pub(super) const PIPELINE_PREVALIDATED_BGEN_ENGINE_USED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_prevalidated_bgen_engine_used";
pub(super) const PIPELINE_OUTPUT_RESUME_COMMITTED_CHUNKS_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_output_resume_committed_chunks";
pub(super) const PIPELINE_OUTPUT_WRITER_SESSIONS_CREATE_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_output_writer_sessions_create_started";
pub(super) const PIPELINE_MULTI_PHENOTYPE_SAMPLE_SUMMARY_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_multi_phenotype_sample_summary";
pub(super) const PIPELINE_MULTI_TRAIT_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_multi_trait_started";
pub(super) const PIPELINE_MULTI_TRAIT_INPUT_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_multi_trait_input_load_started";
pub(super) const PIPELINE_MULTI_TRAIT_INPUT_ALIGNED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_multi_trait_input_aligned";
pub(super) const PIPELINE_MULTI_TRAIT_PREDICTION_SOURCE_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_multi_trait_prediction_source_load_started";
pub(super) const PIPELINE_GROUPED_PER_PHENOTYPE_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_grouped_per_phenotype_started";
pub(super) const PIPELINE_GROUPED_PER_PHENOTYPE_GROUPS_PREPARED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_grouped_per_phenotype_groups_prepared";
pub(super) const PIPELINE_GROUPED_UNION_DELIVERY_SELECTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_grouped_union_delivery_selected";
pub(super) const PIPELINE_MULTI_GROUP_PREFLIGHT_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_multi_group_preflight_started";
pub(super) const PIPELINE_MULTI_GROUP_PREFLIGHT_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Running preflight validation for multi-phenotype pipeline.";
pub(super) const PIPELINE_MULTI_GROUP_PREFLIGHT_COMPLETED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_multi_group_preflight_completed";
pub(super) const PIPELINE_MULTI_GROUP_PREFLIGHT_COMPLETED_DIAGNOSTIC_MESSAGE: &str =
    "Preflight validation passed for multi-phenotype pipeline.";
pub(super) const PIPELINE_SINGLE_TRAIT_STARTED_DIAGNOSTIC_EVENT_NAME: &str = "pipeline_single_trait_started";
pub(super) const PIPELINE_SINGLE_TRAIT_INPUT_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_single_trait_input_load_started";
pub(super) const PIPELINE_SINGLE_TRAIT_INPUT_ALIGNED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_single_trait_input_aligned";
pub(super) const PIPELINE_SINGLE_TRAIT_PREDICTION_SOURCE_LOAD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_single_trait_prediction_source_load_started";
pub(super) const PIPELINE_SINGLE_TRAIT_PREFLIGHT_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_single_trait_preflight_started";
pub(super) const PIPELINE_SINGLE_TRAIT_PREFLIGHT_COMPLETED_DIAGNOSTIC_EVENT_NAME: &str =
    "pipeline_single_trait_preflight_completed";
pub(super) const NATIVE_DISPATCH_BGEN_ENGINE_CONSTRUCTING_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_bgen_engine_constructing";
pub(super) const NATIVE_DISPATCH_BGEN_ENGINE_CONSTRUCTING_DIAGNOSTIC_MESSAGE: &str =
    "Constructing native BGEN run engine.";
pub(super) const NATIVE_DISPATCH_TRUSTED_BGEN_VALIDATION_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_trusted_bgen_validation_started";
pub(super) const NATIVE_DISPATCH_TRUSTED_BGEN_VALIDATION_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Validating trusted no-missing diploid BGEN mode.";
pub(super) const NATIVE_DISPATCH_DELIVERY_FINISHED_DIAGNOSTIC_EVENT_NAME: &str = "native_dispatch_delivery_finished";
pub(super) const NATIVE_DISPATCH_PIPELINE_FINISHED_DIAGNOSTIC_EVENT_NAME: &str = "native_dispatch_pipeline_finished";
pub(super) const NATIVE_DISPATCH_WRITER_SESSIONS_FINISH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_writer_sessions_finish_started";
pub(super) const NATIVE_DISPATCH_WRITER_SESSIONS_FINISH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Finishing output writer(s) and optional Parquet finalization.";
pub(super) const NATIVE_DISPATCH_WRITER_SESSIONS_INTERRUPTED_FLUSH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "native_dispatch_writer_sessions_interrupted_flush_started";
pub(super) const NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_EVENT_NAME: &str = "native_runtime_knobs_configured";
pub(super) const NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_MESSAGE: &str = "Configuring native runtime knobs.";
pub(super) const PREFLIGHT_WARNING_DIAGNOSTIC_EVENT_NAME: &str = "preflight_warning";
pub(super) const RUNNER_MULTI_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_multi_phenotype_dispatch_started";
pub(super) const RUNNER_SINGLE_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_single_phenotype_dispatch_started";
pub(super) const RUNNER_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_binary_engine_dispatch_started";
pub(super) const RUNNER_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_linear_engine_dispatch_started";
pub(super) const RUNNER_MULTI_PHENOTYPE_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_multi_phenotype_binary_engine_dispatch_started";
pub(super) const RUNNER_MULTI_PHENOTYPE_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_multi_phenotype_linear_engine_dispatch_started";
pub(super) const RUNNER_MULTI_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching multi-phenotype native engine pipeline.";
pub(super) const RUNNER_SINGLE_PHENOTYPE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching single-phenotype native engine pipeline.";
pub(super) const RUNNER_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching binary native engine pipeline.";
pub(super) const RUNNER_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching linear native engine pipeline.";
pub(super) const RUNNER_METADATA_ARTIFACTS_FINALIZED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_metadata_artifacts_finalized";
pub(super) const RUNNER_MULTI_PHENOTYPE_BINARY_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching multi-phenotype binary native engine pipeline.";
pub(super) const RUNNER_MULTI_PHENOTYPE_LINEAR_ENGINE_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching multi-phenotype linear native engine pipeline.";
