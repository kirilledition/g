#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::fn_params_excessive_bools)]

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::regenie::{
    MultiPredictionSource as NativeMultiPredictionSource, PredictionSource,
    resolve_prediction_loco_paths as resolve_native_prediction_loco_paths,
};
use crate::sample::{
    AlignedPhenotypeGroup, AlignedSampleData, AlignmentInputs, GroupedAlignedSampleData, MultiAlignedSampleData,
    MultiAlignmentInputs, ResolvedPhenotypeComputeGroup, SampleKeyMode,
};
use g_genotype::common::{ChunkSpec as NativeChunkSpec, ChunkStats as NativeChunkStats, VariantMetadataColumns};
use g_genotype::planner;
use g_genotype::preprocess;

mod association_backend;
mod callback_diagnostics;
mod callback_progress;
mod callback_queue;
mod callback_runtime_resources;
mod callback_summary;
mod config;
mod errors;
mod host_policy;
mod jax_runtime;
mod logging;
mod output;
mod preflight;
mod preparation;
mod profile;
mod run_events;
mod run_metadata;
mod runtime;
mod runtime_paths;
mod runtime_policy;
mod runtime_state;
mod schedule;
mod shutdown;
mod telemetry_policy;
mod timing;
mod trusted_validation;

use association_backend::{
    NativeAssociationBatchResult, NativeAssociationChromosomeRunInput, NativeAssociationChromosomeRunReport,
    NativeAssociationEngineRunReport, NativeAssociationGroupRunReport, NativeGenotypeBatchView, NativePredictionView,
    NativePreparedGroupInput, NativePythonAssociationBackend, NativePythonEngineRunEffects,
};
use callback_diagnostics::{NativeNullLogisticNonconvergencePlan, plan_null_logistic_nonconvergence};
use callback_progress::{
    NativeCallbackChunkIdentity, NativeCallbackProgressCompletion, NativeCallbackProgressState,
    NativeCallbackProgressTelemetryEvent, NativeCallbackProgressTelemetryPlan, NativeCallbackProgressTelemetryRecord,
    NativeCallbackProgressUpdate, build_callback_chunk_identity,
};
use callback_queue::{
    NativeCallbackObjectQueue, NativeCallbackObjectQueueGetResult, NativeCallbackWaitSignal, NativeCallbackWorkerThread,
};
use callback_runtime_resources::{
    NativeCallbackQueueGetObservedResult, NativeCallbackQueuePutResult, NativeCallbackRuntimeResources,
    NativeCallbackWorkerFinishLifecycleResult, NativeDosageBufferAcquireResult, NativeDosageBufferPoolOperationResult,
    NativeDosageBufferReuseSelectionResult, NativeDosageWorkItemDrainResult, NativeDosageWorkItemGetResult,
    NativeDosageWorkItemStageDurationAttribution, NativeResultInFlightAcquireResult,
    NativeResultInFlightSlotReleaseResult, NativeResultWorkItemResourceReleaseResult, NativeResultWriteItemDrainResult,
    NativeResultWriteItemGetResult,
};
use callback_summary::{
    NativeBinaryCorrectionDiagnosticsRecordPlan, NativeBinaryCorrectionSummary, NativeBinaryCorrectionSummaryEmitPlan,
};
use errors::{convert_bgen_error, convert_genotype_error, convert_prediction_error};
use g_engine::Regenie2RunEngineCore;
use host_policy::{
    build_phenotype_compute_group_id_value, build_phenotype_compute_groups_payload,
    build_phenotype_output_directory_name, normalize_binary_correction_payload, plan_association_backend_payload,
    resolve_association_mode_value,
};
use jax_runtime::{
    NativeJaxRuntimeDiagnosticRecordPlan, NativeJaxRuntimeSetupSession, build_jax_runtime_setup_diagnostic_payloads,
    complete_jax_runtime_setup_validation_payload, nvidia_driver_files_are_visible_value,
    plan_jax_gpu_validation_payload, plan_jax_runtime_config_update_payloads, plan_jax_runtime_diagnostic_record,
    plan_jax_runtime_diagnostic_record_payload, plan_jax_runtime_setup_side_effects_payload,
    resolve_jax_runtime_setup_payload,
};
use logging::{
    NativeTelemetryClosePlan, NativeTelemetryEventEmissionPlan, NativeTelemetryProgressEmissionPlan,
    NativeTelemetryProgressThrottle, NativeTelemetryRunSession, NativeTelemetrySession,
    build_current_telemetry_event_payload, build_telemetry_event_payload, emit_diagnostic_event,
    emit_diagnostic_event_fields, generate_telemetry_run_id_value, initialize_logging, plan_telemetry_close,
    plan_telemetry_event_emission, plan_telemetry_progress_emission, shutdown_logging,
};
use output::{
    NativeInitializedOutputRun, NativeOutputRunPaths, NativePreparedOutputRun, OutputWriterSession,
    build_current_run_manifest_header_json, build_file_content_sha256_value,
    build_manifest_file_fingerprint_mapping_payload, build_manifest_file_fingerprint_payload,
    build_manifest_json_sha256, build_prepared_run_manifest_header_json, build_prepared_run_plan_json,
    finalize_output_run_chunks, initialize_output_run, load_run_manifest_json, prepare_output_run,
    read_manifest_committed_chunk_identifiers, repair_strict_manifest_chunk_commits, resolve_output_run_paths,
    scan_committed_chunk_identifiers, validate_run_manifest_compatibility, validate_strict_manifest_chunks,
    write_regenie2_multi_native_chunk, write_regenie2_multi_native_chunk_f64, write_run_manifest_json,
};
use preflight::{
    build_preflight_report_payload, resolve_preflight_variant_count, validate_binary_phenotype_case_control_counts,
    validate_binary_phenotype_coding, validate_covariate_matrix_rank, validate_finite_array,
    validate_multi_prediction_preflight_shape, validate_multi_trait_preflight_shape_payload,
    validate_single_prediction_preflight_shape, validate_single_trait_preflight_shape_payload,
};
use preparation::{
    NativePipelineOutputInitialization, NativePipelineOutputPreparationBatch, initialize_pipeline_output_run_batch,
    initialize_pipeline_output_runs, validate_pipeline_resume_compatibility,
};
use profile::build_profile_snapshot_dict;
use run_events::{
    attach_run_metadata_payload, build_callback_null_logistic_nonconvergence_warning_diagnostic_payload,
    build_io_output_resume_committed_chunks_diagnostic_payload,
    build_native_dispatch_bgen_engine_constructing_diagnostic_payload,
    build_native_dispatch_callback_drain_started_diagnostic_payload,
    build_native_dispatch_delivery_failed_diagnostic_payload,
    build_native_dispatch_delivery_finished_diagnostic_payload,
    build_native_dispatch_delivery_interrupted_diagnostic_payload,
    build_native_dispatch_delivery_started_diagnostic_payload,
    build_native_dispatch_pipeline_finished_diagnostic_payload,
    build_native_dispatch_trusted_bgen_validation_started_diagnostic_payload,
    build_native_dispatch_writer_session_finish_started_diagnostic_payload,
    build_native_dispatch_writer_session_interrupted_flush_started_diagnostic_payload,
    build_native_dispatch_writer_sessions_finish_started_diagnostic_payload,
    build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload,
    build_native_runtime_knobs_configured_diagnostic_payload,
    build_pipeline_gpu_genotype_format_resolved_diagnostic_payload, build_preflight_warning_diagnostic_payload,
    build_run_completed_event_payload, build_run_completed_telemetry_fields, build_run_failed_event_payload,
    build_run_failed_telemetry_fields, build_run_interrupted_event_payload, build_run_interrupted_telemetry_fields,
    build_runner_binary_engine_dispatch_started_diagnostic_payload,
    build_runner_execution_plan_build_started_diagnostic_payload,
    build_runner_execution_plan_dispatch_started_diagnostic_payload,
    build_runner_execution_plan_finalization_started_diagnostic_payload,
    build_runner_execution_plan_prepared_diagnostic_payload,
    build_runner_jax_runtime_configuration_started_diagnostic_payload,
    build_runner_linear_engine_dispatch_started_diagnostic_payload,
    build_runner_metadata_artifacts_finalized_diagnostic_payload,
    build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload,
    build_runner_multi_phenotype_dispatch_started_diagnostic_payload,
    build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload,
    build_runner_run_completed_diagnostic_payload, build_runner_run_failed_diagnostic_payload,
    build_runner_run_interrupted_diagnostic_payload, build_runner_run_started_diagnostic_payload,
    build_runner_single_phenotype_dispatch_started_diagnostic_payload, render_run_completed_lines,
    render_run_failed_lines, render_run_interrupted_lines,
};
use run_metadata::{
    build_execution_run_artifacts_payload, build_multi_run_artifacts_payload, build_phenotype_run_artifacts_payload,
    build_run_manifest_extension_payload,
};
use runtime::{
    configure_bgen_decode_tile_variant_count, configure_rayon_global_thread_pool,
    format_rayon_thread_pool_configuration_error_value,
};
use runtime_paths::build_default_local_cache_directory_value;
use runtime_policy::{build_logging_runtime_policy_payload, describe_logging_runtime_policy_value};
use runtime_state::{
    NativeJaxRuntimeSetupLifecyclePlan, NativeRayonThreadPoolConfigurationPlan, NativeRunRuntime,
    NativeRuntimeCompatibilityToken, NativeRuntimePolicy, NativeRuntimeState, build_jax_runtime_policy_payload,
    build_runtime_policy_handle,
};
use schedule::{
    NativeBgenDeliveryCleanupPlan, NativeBgenDeliveryInvocationPlan, NativeCallbackQueueBackpressureObservation,
    NativeCallbackQueueGetAttemptPlan, NativeCallbackQueueGetObservationPlan, NativeCallbackQueueLimits,
    NativeCallbackQueueOperationObservationPlan, NativeCallbackQueuePutAttemptPlan,
    NativeCallbackQueuePutObservationPlan, NativeCallbackQueueStageBackpressureObservation,
    NativeCallbackQueueStageObservationPlan, NativeCallbackSchedulerState, NativeCallbackWorkerAbortPlan,
    NativeCallbackWorkerErrorRaisePlan, NativeCallbackWorkerErrorUpdatePlan, NativeCallbackWorkerFinishPlan,
    NativeCallbackWorkerJoinPlan, NativeCallbackWorkerLifecycleState, NativeCallbackWorkerShutdownTimeouts,
    NativeCallbackWorkerStartAttemptPlan, NativeCallbackWorkerStartPlan, NativeCallbackWorkerStopPlan,
    NativeCallbackWorkerStopPollPlan, NativeDosageBufferAcquireAttemptPlan, NativeDosageBufferDiscardAttemptPlan,
    NativeDosageBufferPoolObservationPlan, NativeDosageBufferPoolState, NativeDosageBufferRegisterAttemptPlan,
    NativeDosageBufferReturnAttemptPlan, NativeDosageBufferReusePlan, NativeDosageWorkDrainCompletionPlan,
    NativeDosageWorkHandoffPlan, NativeDosageWorkItemDispatchPlan, NativeDosageWorkItemStageDurationPlan,
    NativeGpuGenotypeFormatResolutionPlan, NativeMultiTraitChunkWritePlan, NativeMultiTraitOutputWritePlan,
    NativeResultInFlightAcquireAttemptPlan, NativeResultInFlightAcquireObservationPlan,
    NativeResultInFlightReleaseAttemptPlan, NativeResultInFlightReleaseObservationPlan, NativeResultInFlightSlotState,
    NativeResultWriteDrainCompletionPlan, NativeResultWriteHandoffPlan, NativeResultWriteItemDispatchPlan,
    NativeResultWriteItemResourceReleasePlan, NativeSingleTraitOutputWritePlan,
    NativeVariantMajorDosageBatchHandoffPlan, NativeWriterFinishExecutionPlan,
    format_dosage_callback_worker_error_message, format_result_callback_worker_error_message,
    intersect_committed_chunk_identifier_sets, plan_auto_gpu_genotype_format_after_trusted_validation,
    plan_bgen_delivery_cleanup, plan_bgen_delivery_invocation, plan_callback_queue_backpressure_observation,
    plan_callback_queue_operation_observation, plan_callback_queue_stage_backpressure_observation,
    plan_callback_queue_stage_observation, plan_callback_worker_abort, plan_callback_worker_finish,
    plan_callback_worker_start, plan_callback_worker_stop_poll, plan_dosage_buffer_reuse,
    plan_dosage_callback_worker_join, plan_dosage_callback_worker_stop, plan_dosage_work_handoff,
    plan_dosage_work_item_dispatch, plan_dosage_work_item_stage_duration, plan_gpu_genotype_format_auto_to_dosage,
    plan_multi_trait_chunk_write, plan_multi_trait_output_write, plan_result_callback_worker_join,
    plan_result_callback_worker_stop, plan_result_write_handoff, plan_result_write_item_dispatch,
    plan_single_trait_binary_gpu_genotype_format_resolution, plan_single_trait_output_write,
    plan_variant_major_dosage_batch_handoff, plan_writer_finish_execution, resolve_bgen_delivery_method_value,
    resolve_callback_worker_backpressure_poll_timeout_seconds, resolve_callback_worker_stop_poll_timeout_seconds,
    resolve_delivery_callback_batch_size, resolve_effective_trusted_no_missing_diploid,
    resolve_grouped_union_callback_batch_size, resolve_manifest_gpu_genotype_format,
    resolve_native_callback_queue_limits, resolve_native_callback_worker_shutdown_timeouts,
    resolve_writer_finish_thread_count, should_attempt_callback_worker_stop,
};
use shutdown::{
    NativeSecondSignalExceptionPlan, NativeShutdownController, build_shutdown_signal_payload,
    default_shutdown_signal_numbers, plan_second_signal_exception, raise_second_signal_exception,
};
use telemetry_policy::{
    NativeTelemetrySessionPolicy, build_empty_telemetry_writer_counters_payload, format_telemetry_timestamp_value,
    paths_refer_to_same_file_value, resolve_telemetry_output_run_root_value, resolve_telemetry_paths_payload,
    resolve_telemetry_session_policy_payload, resolve_telemetry_stream_file_value,
};
use timing::{
    NativeStageTimingRecorder, NativeStageTimingRecorderPlan, NativeTimingFileWritePlan,
    build_final_timing_outputs_write_started_diagnostic_payload, plan_stage_timing_recorder, plan_timing_file_write,
};
use trusted_validation::{
    build_trusted_bgen_validation_cache_path_value, build_trusted_bgen_validation_cache_payload,
    build_trusted_bgen_validation_fingerprint_value,
};

type VariantMetadataTuple = (Vec<String>, Vec<String>, Vec<i64>, Vec<String>, Vec<String>);

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct ChunkSpec {
    chunk_spec: NativeChunkSpec,
}

#[pymethods]
impl ChunkSpec {
    #[getter]
    fn variant_start_index(&self) -> usize {
        self.chunk_spec.variant_start_index
    }

    #[getter]
    fn variant_stop_index(&self) -> usize {
        self.chunk_spec.variant_stop_index
    }
}

#[pyclass]
pub(crate) struct ChunkStats {
    pub(crate) stats: Arc<NativeChunkStats>,
}

impl ChunkStats {
    fn new(stats: NativeChunkStats) -> Self {
        Self { stats: Arc::new(stats) }
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn summarize_variant_major_dosage_chunk_stats(
    genotype_matrix_by_variant: PyReadonlyArray2<'_, f32>,
) -> PyResult<ChunkStats> {
    let genotype_array = genotype_matrix_by_variant.as_array();
    let genotype_shape = genotype_array.shape();
    let selected_variant_count = genotype_shape[0];
    let selected_sample_count = genotype_shape[1];
    let genotype_values = genotype_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Variant-major genotype matrix must be C-contiguous."))?;
    let chunk_stats = preprocess::summarize_variant_major_dosage_matrix(
        genotype_values,
        selected_sample_count,
        selected_variant_count,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ChunkStats::new(chunk_stats))
}

#[pymethods]
impl ChunkStats {
    #[getter]
    fn allele_one_frequency<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.allele_one_frequency.clone().into_pyarray(py)
    }

    #[getter]
    fn observation_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.stats.observation_count.clone().into_pyarray(py)
    }

    #[getter]
    fn dosage_sum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.dosage_sum.as_ref().to_vec().into_pyarray(py)
    }

    #[getter]
    fn allele_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.dosage_sum.as_ref().to_vec().into_pyarray(py)
    }

    #[getter]
    fn has_missing_values(&self) -> bool {
        self.stats.has_missing_values
    }

    #[getter]
    fn dosage_square_sum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.dosage_square_sum.clone().into_pyarray(py)
    }

    #[getter]
    fn imputed_dosage_square_sum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.imputed_dosage_square_sum.clone().into_pyarray(py)
    }

    #[getter]
    fn info_score<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats
            .info_score
            .iter()
            .map(|maybe_info_score| maybe_info_score.unwrap_or(f32::NAN))
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[getter]
    fn minor_allele_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.stats.minor_allele_count.clone().into_pyarray(py)
    }

    #[getter]
    fn zero_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.stats.zero_count.clone().into_pyarray(py)
    }

    #[getter]
    fn nonzero_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.stats.nonzero_count.clone().into_pyarray(py)
    }

    #[getter]
    fn is_sparse_candidate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.stats.is_sparse_candidate.clone().into_pyarray(py)
    }

    #[getter]
    fn is_rare_sparse_firth_candidate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.stats.is_rare_sparse_firth_candidate.clone().into_pyarray(py)
    }

    #[pyo3(signature = (*, include_imputed_dosage_square_sum = true, include_sparse_firth_candidate = true))]
    fn compute_arrays<'py>(
        &self,
        py: Python<'py>,
        include_imputed_dosage_square_sum: bool,
        include_sparse_firth_candidate: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let compute_arrays = PyDict::new(py);
        compute_arrays.set_item("dosage_sum", self.stats.dosage_sum.as_ref().to_vec().into_pyarray(py))?;
        compute_arrays.set_item("observation_count", self.stats.observation_count.clone().into_pyarray(py))?;
        if include_imputed_dosage_square_sum {
            compute_arrays
                .set_item("imputed_dosage_square_sum", self.stats.imputed_dosage_square_sum.clone().into_pyarray(py))?;
        }
        if include_sparse_firth_candidate {
            compute_arrays.set_item(
                "is_rare_sparse_firth_candidate",
                self.stats.is_rare_sparse_firth_candidate.clone().into_pyarray(py),
            )?;
        }
        Ok(compute_arrays)
    }
}

#[pyclass]
pub(crate) struct VariantMetadata {
    pub(crate) variant_start_index: usize,
    pub(crate) variant_stop_index: usize,
    pub(crate) metadata: Arc<VariantMetadataColumns>,
}

#[pyclass]
pub(crate) struct NativeAlignedSampleData {
    data: AlignedSampleData,
}

#[pyclass]
pub(crate) struct NativeMultiAlignedSampleData {
    data: MultiAlignedSampleData,
}

#[pyclass]
pub(crate) struct NativeAlignedPhenotypeGroup {
    data: AlignedPhenotypeGroup,
}

#[pyclass]
pub(crate) struct NativeGroupedAlignedSampleData {
    data: GroupedAlignedSampleData,
}

#[pyclass]
pub(crate) struct NativeResolvedPhenotypeComputeGroup {
    data: ResolvedPhenotypeComputeGroup,
}

impl VariantMetadata {
    fn new(variant_start_index: usize, variant_stop_index: usize, metadata: VariantMetadataColumns) -> Self {
        Self { variant_start_index, variant_stop_index, metadata: Arc::new(metadata) }
    }
}

impl NativeAlignedSampleData {
    fn new(data: AlignedSampleData) -> Self {
        Self { data }
    }
}

impl NativeMultiAlignedSampleData {
    fn new(data: MultiAlignedSampleData) -> Self {
        Self { data }
    }
}

impl NativeAlignedPhenotypeGroup {
    fn new(data: AlignedPhenotypeGroup) -> Self {
        Self { data }
    }
}

impl NativeGroupedAlignedSampleData {
    fn new(data: GroupedAlignedSampleData) -> Self {
        Self { data }
    }
}

impl NativeResolvedPhenotypeComputeGroup {
    fn new(data: ResolvedPhenotypeComputeGroup) -> Self {
        Self { data }
    }
}

#[pymethods]
impl NativeAlignedSampleData {
    #[getter]
    fn sample_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.data.sample_indices.clone().into_pyarray(py)
    }

    #[getter]
    fn family_identifiers(&self) -> Vec<String> {
        self.data.family_identifiers.clone()
    }

    #[getter]
    fn individual_identifiers(&self) -> Vec<String> {
        self.data.individual_identifiers.clone()
    }

    #[getter]
    fn phenotype_name(&self) -> String {
        self.data.phenotype_name.clone()
    }

    #[getter]
    fn phenotype_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.data.phenotype_vector.clone().into_pyarray(py)
    }

    #[getter]
    fn covariate_names(&self) -> Vec<String> {
        self.data.covariate_names.clone()
    }

    #[getter]
    fn covariate_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let covariate_matrix = Array2::from_shape_vec(
            (self.data.covariate_row_count, self.data.covariate_column_count),
            self.data.covariate_matrix_values.clone(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(covariate_matrix.into_pyarray(py))
    }

    #[getter]
    fn is_binary_trait(&self) -> bool {
        self.data.is_binary_trait
    }
}

#[pymethods]
impl NativeMultiAlignedSampleData {
    #[getter]
    fn sample_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.data.sample_indices.clone().into_pyarray(py)
    }

    #[getter]
    fn family_identifiers(&self) -> Vec<String> {
        self.data.family_identifiers.clone()
    }

    #[getter]
    fn individual_identifiers(&self) -> Vec<String> {
        self.data.individual_identifiers.clone()
    }

    #[getter]
    fn phenotype_names(&self) -> Vec<String> {
        self.data.phenotype_names.clone()
    }

    #[getter]
    fn phenotype_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let phenotype_matrix = Array2::from_shape_vec(
            (self.data.phenotype_row_count, self.data.phenotype_column_count),
            self.data.phenotype_matrix_values.clone(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(phenotype_matrix.into_pyarray(py))
    }

    #[getter]
    fn covariate_names(&self) -> Vec<String> {
        self.data.covariate_names.clone()
    }

    #[getter]
    fn covariate_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let covariate_matrix = Array2::from_shape_vec(
            (self.data.covariate_row_count, self.data.covariate_column_count),
            self.data.covariate_matrix_values.clone(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(covariate_matrix.into_pyarray(py))
    }

    #[getter]
    fn is_binary_trait(&self) -> bool {
        self.data.is_binary_trait
    }
}

#[pymethods]
impl NativeAlignedPhenotypeGroup {
    #[getter]
    fn phenotype_indices(&self) -> Vec<usize> {
        self.data.phenotype_indices.clone()
    }

    #[getter]
    fn aligned_sample_data(&self) -> NativeMultiAlignedSampleData {
        NativeMultiAlignedSampleData::new(self.data.aligned_sample_data.clone())
    }
}

#[pymethods]
impl NativeGroupedAlignedSampleData {
    #[getter]
    fn groups(&self, py: Python<'_>) -> PyResult<Vec<Py<NativeAlignedPhenotypeGroup>>> {
        self.data.groups.iter().cloned().map(|group| Py::new(py, NativeAlignedPhenotypeGroup::new(group))).collect()
    }
}

#[pymethods]
impl NativeResolvedPhenotypeComputeGroup {
    #[getter]
    fn group_mode(&self) -> String {
        self.data.group_mode.clone()
    }

    #[getter]
    fn phenotype_indices(&self) -> Vec<usize> {
        self.data.phenotype_indices.clone()
    }

    #[getter]
    fn phenotype_names(&self) -> Vec<String> {
        self.data.phenotype_names.clone()
    }

    #[getter]
    fn sample_mode(&self) -> String {
        self.data.sample_mode.clone()
    }

    #[getter]
    fn sample_set_fingerprint(&self) -> String {
        self.data.sample_set_fingerprint.clone()
    }

    #[getter]
    fn covariate_design_fingerprint(&self) -> String {
        self.data.covariate_design_fingerprint.clone()
    }

    #[getter]
    fn prediction_alignment_fingerprint(&self) -> Option<String> {
        self.data.prediction_alignment_fingerprint.clone()
    }
}

#[pymethods]
impl VariantMetadata {
    #[getter]
    fn variant_start_index(&self) -> usize {
        self.variant_start_index
    }

    #[getter]
    fn variant_stop_index(&self) -> usize {
        self.variant_stop_index
    }

    #[getter]
    fn chromosome(&self) -> Vec<String> {
        self.metadata.chromosome.clone()
    }

    #[getter]
    fn chromosome_label(&self) -> PyResult<String> {
        self.metadata
            .chromosome
            .first()
            .cloned()
            .ok_or_else(|| PyValueError::new_err("Variant metadata contains no chromosome labels."))
    }

    #[getter]
    fn variant_identifiers(&self) -> Vec<String> {
        self.metadata.variant_identifier.clone()
    }

    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.metadata.position.clone().into_pyarray(py)
    }

    #[getter]
    fn allele_one(&self) -> Vec<String> {
        self.metadata.allele_one.clone()
    }

    #[getter]
    fn allele_two(&self) -> Vec<String> {
        self.metadata.allele_two.clone()
    }
}

#[pyclass]
struct Regenie2RunEngine {
    engine: Regenie2RunEngineCore,
}

#[pyclass]
struct RegeniePredictionSource {
    source: PredictionSource,
}

#[pyclass]
struct MultiRegeniePredictionSource {
    source: NativeMultiPredictionSource,
}

#[pymethods]
impl Regenie2RunEngine {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (bgen_path, chunk_size, variant_limit=None, trusted_no_missing_diploid=false))]
    fn new(
        py: Python<'_>,
        bgen_path: String,
        chunk_size: usize,
        variant_limit: Option<usize>,
        trusted_no_missing_diploid: bool,
    ) -> PyResult<Self> {
        let engine = py
            .detach(|| {
                Regenie2RunEngineCore::open_bgen(
                    Path::new(&bgen_path),
                    chunk_size,
                    variant_limit,
                    trusted_no_missing_diploid,
                )
            })
            .map_err(|error| convert_bgen_error("open_bgen", error))?;
        Ok(Self { engine })
    }

    #[getter]
    fn sample_count(&self) -> usize {
        self.engine.reader().sample_count()
    }

    #[getter]
    fn variant_count(&self) -> usize {
        self.engine.reader().variant_count()
    }

    #[getter]
    fn contains_embedded_samples(&self) -> bool {
        self.engine.reader().contains_embedded_samples()
    }

    fn sample_identifiers(&self) -> Vec<String> {
        self.engine.reader().sample_identifiers()
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_path,
        phenotype_path,
        phenotype_name,
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=false,
        sample_key_mode="iid".to_string()
    ))]
    fn align_sample_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
        phenotype_path: String,
        phenotype_name: String,
        covariate_path: Option<String>,
        covariate_names: Option<Vec<String>>,
        is_binary_trait: bool,
        sample_key_mode: String,
    ) -> PyResult<NativeAlignedSampleData> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    crate::sample::align_sample_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                        phenotype_path,
                        phenotype_name,
                        covariate_path,
                        covariate_names,
                        is_binary_trait,
                        parsed_sample_key_mode,
                    )
                })
                .map(NativeAlignedSampleData::new)
                .map_err(PyValueError::new_err);
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let inputs = AlignmentInputs {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
            phenotype_path,
            phenotype_name,
            covariate_path,
            covariate_names,
            is_binary_trait,
            sample_key_mode: parsed_sample_key_mode,
        };
        py.detach(move || crate::sample::align_sample_data(inputs))
            .map(NativeAlignedSampleData::new)
            .map_err(PyValueError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_path,
        phenotype_path,
        phenotype_names,
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=false,
        sample_key_mode="iid".to_string()
    ))]
    fn align_multi_sample_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
        phenotype_path: String,
        phenotype_names: Vec<String>,
        covariate_path: Option<String>,
        covariate_names: Option<Vec<String>>,
        is_binary_trait: bool,
        sample_key_mode: String,
    ) -> PyResult<NativeMultiAlignedSampleData> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    crate::sample::align_multi_sample_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                        phenotype_path,
                        phenotype_names,
                        covariate_path,
                        covariate_names,
                        is_binary_trait,
                        parsed_sample_key_mode,
                    )
                })
                .map(NativeMultiAlignedSampleData::new)
                .map_err(PyValueError::new_err);
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let inputs = MultiAlignmentInputs {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            sample_key_mode: parsed_sample_key_mode,
        };
        py.detach(move || crate::sample::align_multi_sample_data(inputs))
            .map(NativeMultiAlignedSampleData::new)
            .map_err(PyValueError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_path,
        phenotype_path,
        phenotype_names,
        covariate_path=None,
        covariate_names=None,
        is_binary_trait=false,
        sample_key_mode="iid".to_string()
    ))]
    fn align_grouped_sample_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
        phenotype_path: String,
        phenotype_names: Vec<String>,
        covariate_path: Option<String>,
        covariate_names: Option<Vec<String>>,
        is_binary_trait: bool,
        sample_key_mode: String,
    ) -> PyResult<NativeGroupedAlignedSampleData> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    crate::sample::align_grouped_sample_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                        phenotype_path,
                        phenotype_names,
                        covariate_path,
                        covariate_names,
                        is_binary_trait,
                        parsed_sample_key_mode,
                    )
                })
                .map(NativeGroupedAlignedSampleData::new)
                .map_err(PyValueError::new_err);
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        let inputs = MultiAlignmentInputs {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            sample_key_mode: parsed_sample_key_mode,
        };
        py.detach(move || crate::sample::align_grouped_sample_data(&inputs))
            .map(NativeGroupedAlignedSampleData::new)
            .map_err(PyValueError::new_err)
    }

    fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.engine.reader().chromosome_boundary_indices()
    }

    fn variant_metadata_slice(
        &self,
        py: Python<'_>,
        variant_start: usize,
        variant_stop: usize,
    ) -> PyResult<VariantMetadataTuple> {
        py.detach(|| self.engine.reader().variant_metadata_slice(variant_start, variant_stop))
            .map(convert_variant_metadata_columns_to_tuple)
            .map_err(|error| convert_bgen_error("read_variant_metadata_slice", error))
    }

    #[pyo3(signature = (variant_limit=None))]
    fn required_chromosomes(&self, variant_limit: Option<usize>) -> PyResult<Vec<String>> {
        self.engine.required_chromosomes(variant_limit).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn reset_profile(&self) {
        self.engine.reader().reset_profile();
    }

    fn profile_snapshot(&self) -> HashMap<String, u64> {
        build_profile_snapshot_dict(&self.engine.reader().profile_snapshot())
    }

    fn validate_trusted_no_missing_diploid(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.engine.reader().validate_trusted_no_missing_diploid())
            .map_err(|error| convert_bgen_error("validate_trusted_no_missing_diploid", error))
    }

    fn mark_trusted_no_missing_diploid_validated(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.engine.reader().mark_trusted_no_missing_diploid_validated())
            .map_err(|error| convert_bgen_error("mark_trusted_no_missing_diploid_validated", error))
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (sample_indices, callback, committed_chunk_identifiers=None, callback_batch_size=1))]
    fn run_bgen_variant_major_dosage_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        let sample_index_values = sample_indices.as_slice()?.to_vec();
        self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
            py,
            &sample_index_values,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None, callback_batch_size=1))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_dosage_buffered_chunks_for_native_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None, callback_batch_size=1))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeMultiAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (sample_indices, callback, committed_chunk_identifiers=None))]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let sample_index_values = sample_indices.as_slice()?.to_vec();
        self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
            py,
            &sample_index_values,
            callback,
            committed_chunk_identifiers,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
        )
    }

    #[pyo3(signature = (aligned_sample_data, callback, committed_chunk_identifiers=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_multi_aligned_samples<'py>(
        &self,
        py: Python<'py>,
        aligned_sample_data: PyRef<'py, NativeMultiAlignedSampleData>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
            py,
            &aligned_sample_data.data.sample_indices,
            callback,
            committed_chunk_identifiers,
        )
    }
}

#[pymethods]
impl RegeniePredictionSource {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        phenotype_name,
        sample_family_identifiers,
        sample_individual_identifiers,
        sample_key_mode="iid".to_string()
    ))]
    fn new(
        prediction_list_path: String,
        phenotype_name: String,
        sample_family_identifiers: Vec<String>,
        sample_individual_identifiers: Vec<String>,
        sample_key_mode: String,
    ) -> PyResult<Self> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let source = PredictionSource::load(
            Path::new(&prediction_list_path),
            &phenotype_name,
            &sample_family_identifiers,
            &sample_individual_identifiers,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_prediction_source", &error))?;
        Ok(Self { source })
    }

    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        phenotype_name,
        aligned_sample_data,
        sample_key_mode="iid".to_string()
    ))]
    fn from_native_aligned_sample_data(
        prediction_list_path: String,
        phenotype_name: String,
        aligned_sample_data: PyRef<'_, NativeAlignedSampleData>,
        sample_key_mode: String,
    ) -> PyResult<Self> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let source = PredictionSource::load(
            Path::new(&prediction_list_path),
            &phenotype_name,
            &aligned_sample_data.data.family_identifiers,
            &aligned_sample_data.data.individual_identifiers,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_prediction_source_from_native_aligned_sample_data", &error))?;
        Ok(Self { source })
    }

    #[allow(clippy::needless_pass_by_value)]
    fn get_chromosome_predictions<'py>(
        &self,
        py: Python<'py>,
        chromosome: String,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let prediction_values = self
            .source
            .chromosome_predictions(&chromosome)
            .map_err(|error| convert_prediction_error("chromosome_predictions", &error))?;
        Ok(Array1::from_vec(prediction_values.to_vec()).into_pyarray(py))
    }
}

#[pymethods]
impl MultiRegeniePredictionSource {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        phenotype_names,
        sample_family_identifiers,
        sample_individual_identifiers,
        sample_key_mode="iid".to_string()
    ))]
    fn new(
        prediction_list_path: String,
        phenotype_names: Vec<String>,
        sample_family_identifiers: Vec<String>,
        sample_individual_identifiers: Vec<String>,
        sample_key_mode: String,
    ) -> PyResult<Self> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let source = NativeMultiPredictionSource::load(
            Path::new(&prediction_list_path),
            &phenotype_names,
            &sample_family_identifiers,
            &sample_individual_identifiers,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_multi_prediction_source", &error))?;
        Ok(Self { source })
    }

    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        aligned_sample_data,
        sample_key_mode="iid".to_string()
    ))]
    fn from_native_multi_aligned_sample_data(
        prediction_list_path: String,
        aligned_sample_data: PyRef<'_, NativeMultiAlignedSampleData>,
        sample_key_mode: String,
    ) -> PyResult<Self> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let source = NativeMultiPredictionSource::load(
            Path::new(&prediction_list_path),
            &aligned_sample_data.data.phenotype_names,
            &aligned_sample_data.data.family_identifiers,
            &aligned_sample_data.data.individual_identifiers,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_multi_prediction_source_from_aligned_samples", &error))?;
        Ok(Self { source })
    }

    #[staticmethod]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        prediction_list_path,
        grouped_aligned_sample_data,
        sample_key_mode="iid".to_string()
    ))]
    fn from_native_grouped_aligned_sample_data(
        prediction_list_path: String,
        grouped_aligned_sample_data: PyRef<'_, NativeGroupedAlignedSampleData>,
        sample_key_mode: String,
    ) -> PyResult<Vec<Self>> {
        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let aligned_sample_data_groups =
            grouped_aligned_sample_data.data.groups.iter().map(|group| &group.aligned_sample_data).collect::<Vec<_>>();
        let sources = NativeMultiPredictionSource::load_grouped(
            Path::new(&prediction_list_path),
            &aligned_sample_data_groups,
            parsed_sample_key_mode,
        )
        .map_err(|error| convert_prediction_error("load_multi_prediction_source_from_grouped_samples", &error))?;
        Ok(sources.into_iter().map(|source| Self { source }).collect())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn get_chromosome_predictions<'py>(
        &self,
        py: Python<'py>,
        chromosome: String,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (trait_count, sample_count, prediction_values) = self
            .source
            .chromosome_prediction_matrix(&chromosome)
            .map_err(|error| convert_prediction_error("chromosome_prediction_matrix", &error))?;
        let prediction_matrix = Array2::from_shape_vec((trait_count, sample_count), prediction_values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(prediction_matrix.into_pyarray(py))
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn resolve_prediction_loco_paths(
    py: Python<'_>,
    prediction_list_path: String,
    phenotype_names: Vec<String>,
) -> PyResult<Bound<'_, PyList>> {
    let resolved_loco_paths = resolve_native_prediction_loco_paths(Path::new(&prediction_list_path), &phenotype_names)
        .map_err(|error| convert_prediction_error("resolve_prediction_loco_paths", &error))?;
    let payloads = PyList::empty(py);
    for resolved_loco_path in resolved_loco_paths {
        let payload = PyDict::new(py);
        payload.set_item("phenotype", resolved_loco_path.phenotype_name)?;
        payload.set_item("path", resolved_loco_path.loco_file_path.display().to_string())?;
        payloads.append(payload)?;
    }
    Ok(payloads)
}

fn flush_variant_major_dosage_batch<'py>(
    compute_dosage_chunk_batch_method: &Bound<'py, PyAny>,
    metadata_batch: &mut Vec<Py<VariantMetadata>>,
    output_array_batch: &mut Vec<Py<PyAny>>,
    stats_batch: &mut Vec<Py<ChunkStats>>,
) -> PyResult<()> {
    if metadata_batch.is_empty() {
        return Ok(());
    }
    let metadata_values = std::mem::take(metadata_batch);
    let output_array_values = std::mem::take(output_array_batch);
    let stats_values = std::mem::take(stats_batch);
    compute_dosage_chunk_batch_method.call1((metadata_values, output_array_values, stats_values))?;
    Ok(())
}

impl Regenie2RunEngine {
    fn run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        if callback_batch_size == 0 {
            return Err(PyValueError::new_err("callback_batch_size must be positive."));
        }
        py.detach(|| self.engine.reader().prepare_sample_selection(sample_index_values))
            .map_err(|error| convert_bgen_error("prepare_sample_selection", error))?;

        let run_result = self.run_prepared_bgen_variant_major_dosage_buffered_chunks(
            py,
            sample_index_values.len(),
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        );
        let clear_result = py
            .detach(|| self.engine.reader().clear_prepared_sample_selection())
            .map_err(|error| convert_bgen_error("clear_prepared_sample_selection", error));
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }

    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices<'py>(
        &self,
        py: Python<'py>,
        sample_index_values: &[i64],
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        py.detach(|| self.engine.reader().prepare_sample_selection(sample_index_values))
            .map_err(|error| convert_bgen_error("prepare_sample_selection", error))?;

        let run_result = self.run_prepared_bgen_variant_major_packed8_probability_pair_buffered_chunks(
            py,
            sample_index_values.len(),
            callback,
            committed_chunk_identifiers,
        );
        let clear_result = py
            .detach(|| self.engine.reader().clear_prepared_sample_selection())
            .map_err(|error| convert_bgen_error("clear_prepared_sample_selection", error));
        match (run_result, clear_result) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
        }
    }

    fn run_prepared_bgen_variant_major_dosage_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        selected_sample_count: usize,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self
            .engine
            .plan_chunks(&committed_identifier_set)
            .map_err(|error| convert_genotype_error("plan_chunks", error))?;
        let acquire_dosage_buffer_method = callback.getattr("acquire_variant_major_dosage_buffer")?;
        if callback_batch_size > 1 {
            return self.run_prepared_bgen_variant_major_dosage_buffered_chunk_batches(
                py,
                selected_sample_count,
                callback,
                &chunk_specs,
                &acquire_dosage_buffer_method,
                callback_batch_size,
            );
        }
        let chunk_batch_plan = g_engine::plan_chunk_batches(&chunk_specs, callback_batch_size)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let processed_chunk_count = chunk_batch_plan.chunk_count();
        let compute_dosage_chunk_method = callback.getattr("compute_preprocessed_variant_major_dosage_chunk")?;
        for chunk_batch in chunk_batch_plan.into_chunk_batches() {
            for chunk_spec in &chunk_batch {
                py.check_signals()?;
                let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
                let output_array_object =
                    acquire_dosage_buffer_method.call1((selected_variant_count, selected_sample_count))?;
                let stats = {
                    let mut output_array = output_array_object.extract::<PyReadwriteArray2<'_, f32>>()?;
                    let output_shape = output_array.shape();
                    if output_shape != [selected_variant_count, selected_sample_count] {
                        return Err(PyValueError::new_err(format!(
                            "Reusable variant-major BGEN dosage buffer shape mismatch: expected ({selected_variant_count}, {}), observed ({}, {}).",
                            selected_sample_count, output_shape[0], output_shape[1],
                        )));
                    }
                    if !output_array.is_c_contiguous() {
                        return Err(PyValueError::new_err(
                            "Reusable variant-major BGEN dosage buffer must be C-contiguous float32.",
                        ));
                    }
                    let output_slice = output_array.as_slice_mut().map_err(|_| {
                        PyValueError::new_err(
                            "Reusable variant-major BGEN dosage buffer must expose a contiguous mutable slice.",
                        )
                    })?;
                    let output_pointer_address = output_slice.as_mut_ptr() as usize;
                    let output_value_count = output_slice.len();
                    let chunk_stats = py
                        .detach(|| {
                            self.engine.reader().read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                                chunk_spec.variant_start_index,
                                chunk_spec.variant_stop_index,
                                output_pointer_address,
                                output_value_count,
                            )
                        })
                        .map_err(|error| {
                            convert_bgen_error(
                                "read_preprocessed_variant_major_dosage_f32_into_address_prepared",
                                error,
                            )
                        })?;
                    Py::new(py, ChunkStats::new(chunk_stats))?
                };
                let variant_start_index = chunk_spec.variant_start_index;
                let variant_stop_index = chunk_spec.variant_stop_index;
                let metadata_columns = py
                    .detach(|| self.engine.reader().variant_metadata_slice(variant_start_index, variant_stop_index))
                    .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
                let metadata =
                    Py::new(py, VariantMetadata::new(variant_start_index, variant_stop_index, metadata_columns))?;
                compute_dosage_chunk_method.call1((metadata, output_array_object, stats))?;
            }
        }
        Ok(processed_chunk_count)
    }

    fn run_prepared_bgen_variant_major_dosage_buffered_chunk_batches<'py>(
        &self,
        py: Python<'py>,
        selected_sample_count: usize,
        callback: &Bound<'py, PyAny>,
        chunk_specs: &[NativeChunkSpec],
        acquire_dosage_buffer_method: &Bound<'py, PyAny>,
        callback_batch_size: usize,
    ) -> PyResult<usize> {
        let compute_dosage_chunk_batch_method =
            callback.getattr("compute_preprocessed_variant_major_dosage_chunk_batch")?;
        let chunk_batch_plan = g_engine::plan_chunk_batches(chunk_specs, callback_batch_size)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let processed_chunk_count = chunk_batch_plan.chunk_count();
        let mut metadata_batch: Vec<Py<VariantMetadata>> = Vec::with_capacity(callback_batch_size);
        let mut output_array_batch: Vec<Py<PyAny>> = Vec::with_capacity(callback_batch_size);
        let mut stats_batch: Vec<Py<ChunkStats>> = Vec::with_capacity(callback_batch_size);
        for chunk_batch in chunk_batch_plan.into_chunk_batches() {
            for chunk_spec in &chunk_batch {
                py.check_signals()?;
                let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
                let output_array_object =
                    acquire_dosage_buffer_method.call1((selected_variant_count, selected_sample_count))?;
                let stats = {
                    let mut output_array = output_array_object.extract::<PyReadwriteArray2<'_, f32>>()?;
                    let output_shape = output_array.shape();
                    if output_shape != [selected_variant_count, selected_sample_count] {
                        return Err(PyValueError::new_err(format!(
                            "Reusable variant-major BGEN dosage buffer shape mismatch: expected ({selected_variant_count}, {}), observed ({}, {}).",
                            selected_sample_count, output_shape[0], output_shape[1],
                        )));
                    }
                    if !output_array.is_c_contiguous() {
                        return Err(PyValueError::new_err(
                            "Reusable variant-major BGEN dosage buffer must be C-contiguous float32.",
                        ));
                    }
                    let output_slice = output_array.as_slice_mut().map_err(|_| {
                        PyValueError::new_err(
                            "Reusable variant-major BGEN dosage buffer must expose a contiguous mutable slice.",
                        )
                    })?;
                    let output_pointer_address = output_slice.as_mut_ptr() as usize;
                    let output_value_count = output_slice.len();
                    let chunk_stats = py
                        .detach(|| {
                            self.engine.reader().read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                                chunk_spec.variant_start_index,
                                chunk_spec.variant_stop_index,
                                output_pointer_address,
                                output_value_count,
                            )
                        })
                        .map_err(|error| {
                            convert_bgen_error(
                                "read_preprocessed_variant_major_dosage_f32_into_address_prepared",
                                error,
                            )
                        })?;
                    Py::new(py, ChunkStats::new(chunk_stats))?
                };
                let variant_start_index = chunk_spec.variant_start_index;
                let variant_stop_index = chunk_spec.variant_stop_index;
                let metadata_columns = py
                    .detach(|| self.engine.reader().variant_metadata_slice(variant_start_index, variant_stop_index))
                    .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
                let metadata =
                    Py::new(py, VariantMetadata::new(variant_start_index, variant_stop_index, metadata_columns))?;
                metadata_batch.push(metadata);
                output_array_batch.push(output_array_object.unbind());
                stats_batch.push(stats);
            }
            flush_variant_major_dosage_batch(
                &compute_dosage_chunk_batch_method,
                &mut metadata_batch,
                &mut output_array_batch,
                &mut stats_batch,
            )?;
        }
        Ok(processed_chunk_count)
    }

    fn run_prepared_bgen_variant_major_packed8_probability_pair_buffered_chunks<'py>(
        &self,
        py: Python<'py>,
        selected_sample_count: usize,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
    ) -> PyResult<usize> {
        let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
        let chunk_specs = self
            .engine
            .plan_chunks(&committed_identifier_set)
            .map_err(|error| convert_genotype_error("plan_chunks", error))?;
        let chunk_batch_plan =
            g_engine::plan_chunk_batches(&chunk_specs, 1).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let processed_chunk_count = chunk_batch_plan.chunk_count();
        let acquire_packed_buffer_method = callback.getattr("acquire_variant_major_packed8_probability_pair_buffer")?;
        let compute_packed_chunk_method =
            callback.getattr("compute_preprocessed_variant_major_packed8_probability_pair_chunk")?;
        for chunk_batch in chunk_batch_plan.into_chunk_batches() {
            for chunk_spec in &chunk_batch {
                py.check_signals()?;
                let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
                let output_array_object =
                    acquire_packed_buffer_method.call1((selected_variant_count, selected_sample_count))?;
                let stats = {
                    let mut output_array = output_array_object.extract::<PyReadwriteArray3<'_, u8>>()?;
                    let output_shape = output_array.shape();
                    if output_shape != [selected_variant_count, selected_sample_count, 2] {
                        return Err(PyValueError::new_err(format!(
                            "Reusable variant-major BGEN packed8 probability-pair buffer shape mismatch: expected ({selected_variant_count}, {}, 2), observed ({}, {}, {}).",
                            selected_sample_count, output_shape[0], output_shape[1], output_shape[2],
                        )));
                    }
                    if !output_array.is_c_contiguous() {
                        return Err(PyValueError::new_err(
                            "Reusable variant-major BGEN packed8 probability-pair buffer must be C-contiguous uint8.",
                        ));
                    }
                    let output_slice = output_array.as_slice_mut().map_err(|_| {
                        PyValueError::new_err(
                            "Reusable variant-major BGEN packed8 probability-pair buffer must expose a contiguous mutable slice.",
                        )
                    })?;
                    let output_pointer_address = output_slice.as_mut_ptr() as usize;
                    let output_value_count = output_slice.len();
                    let chunk_stats = py
                        .detach(|| {
                            self.engine
                                .reader()
                                .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
                                    chunk_spec.variant_start_index,
                                    chunk_spec.variant_stop_index,
                                    output_pointer_address,
                                    output_value_count,
                                )
                        })
                        .map_err(|error| {
                            convert_bgen_error(
                                "read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared",
                                error,
                            )
                        })?;
                    Py::new(py, ChunkStats::new(chunk_stats))?
                };
                let variant_start_index = chunk_spec.variant_start_index;
                let variant_stop_index = chunk_spec.variant_stop_index;
                let metadata_columns = py
                    .detach(|| self.engine.reader().variant_metadata_slice(variant_start_index, variant_stop_index))
                    .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
                let metadata =
                    Py::new(py, VariantMetadata::new(variant_start_index, variant_stop_index, metadata_columns))?;
                compute_packed_chunk_method.call1((metadata, output_array_object, stats))?;
            }
        }
        Ok(processed_chunk_count)
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    sample_indices,
    family_identifiers,
    individual_identifiers,
    phenotype_path,
    phenotype_name,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
))]
fn align_sample_data<'py>(
    py: Python<'py>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    family_identifiers: Vec<String>,
    individual_identifiers: Vec<String>,
    phenotype_path: String,
    phenotype_name: String,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: String,
) -> PyResult<NativeAlignedSampleData> {
    let sample_index_values = sample_indices.as_slice()?.to_vec();
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    let inputs = AlignmentInputs {
        sample_indices: sample_index_values,
        family_identifiers,
        individual_identifiers,
        phenotype_path,
        phenotype_name,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parsed_sample_key_mode,
    };
    py.detach(|| crate::sample::align_sample_data(inputs))
        .map(NativeAlignedSampleData::new)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    sample_indices,
    family_identifiers,
    individual_identifiers,
    phenotype_path,
    phenotype_names,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
))]
fn align_multi_sample_data<'py>(
    py: Python<'py>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    family_identifiers: Vec<String>,
    individual_identifiers: Vec<String>,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: String,
) -> PyResult<NativeMultiAlignedSampleData> {
    let sample_index_values = sample_indices.as_slice()?.to_vec();
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    let inputs = MultiAlignmentInputs {
        sample_indices: sample_index_values,
        family_identifiers,
        individual_identifiers,
        phenotype_path,
        phenotype_names,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parsed_sample_key_mode,
    };
    py.detach(|| crate::sample::align_multi_sample_data(inputs))
        .map(NativeMultiAlignedSampleData::new)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    sample_indices,
    family_identifiers,
    individual_identifiers,
    phenotype_path,
    phenotype_names,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
))]
fn align_grouped_sample_data<'py>(
    py: Python<'py>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    family_identifiers: Vec<String>,
    individual_identifiers: Vec<String>,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: String,
) -> PyResult<NativeGroupedAlignedSampleData> {
    let sample_index_values = sample_indices.as_slice()?.to_vec();
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    let inputs = MultiAlignmentInputs {
        sample_indices: sample_index_values,
        family_identifiers,
        individual_identifiers,
        phenotype_path,
        phenotype_names,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parsed_sample_key_mode,
    };
    py.detach(|| crate::sample::align_grouped_sample_data(&inputs))
        .map(NativeGroupedAlignedSampleData::new)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    sample_path,
    expected_sample_count,
    phenotype_path,
    phenotype_name,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
))]
fn align_sample_data_from_sample_file(
    py: Python<'_>,
    sample_path: String,
    expected_sample_count: usize,
    phenotype_path: String,
    phenotype_name: String,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: String,
) -> PyResult<NativeAlignedSampleData> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    py.detach(move || {
        crate::sample::align_sample_data_from_sample_file(
            Path::new(&sample_path),
            expected_sample_count,
            phenotype_path,
            phenotype_name,
            covariate_path,
            covariate_names,
            is_binary_trait,
            parsed_sample_key_mode,
        )
    })
    .map(NativeAlignedSampleData::new)
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (
    sample_path,
    expected_sample_count,
    phenotype_path,
    phenotype_names,
    covariate_path=None,
    covariate_names=None,
    is_binary_trait=false,
    sample_key_mode="iid".to_string()
))]
fn align_multi_sample_data_from_sample_file(
    py: Python<'_>,
    sample_path: String,
    expected_sample_count: usize,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: String,
) -> PyResult<NativeMultiAlignedSampleData> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    py.detach(move || {
        crate::sample::align_multi_sample_data_from_sample_file(
            Path::new(&sample_path),
            expected_sample_count,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            parsed_sample_key_mode,
        )
    })
    .map(NativeMultiAlignedSampleData::new)
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn resolve_single_phenotype_compute_group(
    aligned_sample_data: PyRef<'_, NativeAlignedSampleData>,
    phenotype_name: String,
    prediction_list_path: Option<String>,
    sample_key_mode: String,
) -> PyResult<NativeResolvedPhenotypeComputeGroup> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    Ok(NativeResolvedPhenotypeComputeGroup::new(crate::sample::resolve_single_phenotype_compute_group(
        &aligned_sample_data.data,
        phenotype_name,
        prediction_list_path.as_deref(),
        parsed_sample_key_mode,
    )))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn resolve_per_phenotype_compute_group(
    aligned_sample_data: PyRef<'_, NativeMultiAlignedSampleData>,
    phenotype_indices: Vec<usize>,
    phenotype_names: Vec<String>,
    prediction_list_path: Option<String>,
    sample_key_mode: String,
) -> PyResult<NativeResolvedPhenotypeComputeGroup> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    Ok(NativeResolvedPhenotypeComputeGroup::new(crate::sample::resolve_per_phenotype_compute_group(
        &aligned_sample_data.data,
        phenotype_indices,
        phenotype_names,
        prediction_list_path.as_deref(),
        parsed_sample_key_mode,
    )))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn resolve_complete_case_compute_group(
    aligned_sample_data: PyRef<'_, NativeMultiAlignedSampleData>,
    phenotype_indices: Vec<usize>,
    phenotype_names: Vec<String>,
    prediction_list_path: Option<String>,
    sample_key_mode: String,
) -> PyResult<NativeResolvedPhenotypeComputeGroup> {
    let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
    Ok(NativeResolvedPhenotypeComputeGroup::new(crate::sample::resolve_complete_case_compute_group(
        &aligned_sample_data.data,
        phenotype_indices,
        phenotype_names,
        prediction_list_path.as_deref(),
        parsed_sample_key_mode,
    )))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
#[pyo3(signature = (variant_count, chunk_size, chromosome_boundary_indices, variant_limit=None, committed_chunk_identifiers=None))]
fn plan_genotype_chunks(
    variant_count: usize,
    chunk_size: usize,
    chromosome_boundary_indices: Vec<usize>,
    variant_limit: Option<usize>,
    committed_chunk_identifiers: Option<Vec<usize>>,
) -> PyResult<Vec<ChunkSpec>> {
    let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
    let chunk_specs = planner::plan_chromosome_homogeneous_chunks(
        variant_count,
        chunk_size,
        variant_limit,
        &chromosome_boundary_indices,
        &committed_identifier_set,
    )
    .map_err(|error| convert_genotype_error("plan_chromosome_homogeneous_chunks", error))?;
    Ok(chunk_specs.into_iter().map(|chunk_spec| ChunkSpec { chunk_spec }).collect())
}

fn parse_sample_key_mode(sample_key_mode: &str) -> PyResult<SampleKeyMode> {
    match sample_key_mode {
        "iid" => Ok(SampleKeyMode::Iid),
        "fid_iid" => Ok(SampleKeyMode::FidIid),
        _ => Err(PyValueError::new_err(format!(
            "sample_key_mode must be 'iid' or 'fid_iid', found '{sample_key_mode}'."
        ))),
    }
}

fn build_committed_identifier_set(committed_chunk_identifiers: Option<Vec<usize>>) -> BTreeSet<usize> {
    committed_chunk_identifiers.unwrap_or_default().into_iter().collect()
}

fn convert_variant_metadata_columns_to_tuple(variant_metadata: VariantMetadataColumns) -> VariantMetadataTuple {
    (
        variant_metadata.chromosome,
        variant_metadata.variant_identifier,
        variant_metadata.position,
        variant_metadata.allele_one,
        variant_metadata.allele_two,
    )
}

#[allow(clippy::missing_errors_doc)]
#[allow(clippy::too_many_lines)]
pub fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    config::register_module(module)?;
    module.add_class::<ChunkSpec>()?;
    module.add_class::<ChunkStats>()?;
    module.add_class::<NativeAssociationBatchResult>()?;
    module.add_class::<NativeAssociationChromosomeRunInput>()?;
    module.add_class::<NativeAssociationChromosomeRunReport>()?;
    module.add_class::<NativeAssociationEngineRunReport>()?;
    module.add_class::<NativeAssociationGroupRunReport>()?;
    module.add_class::<NativeGenotypeBatchView>()?;
    module.add_class::<NativePredictionView>()?;
    module.add_class::<NativePreparedGroupInput>()?;
    module.add_class::<NativePythonAssociationBackend>()?;
    module.add_class::<NativePythonEngineRunEffects>()?;
    module.add_class::<NativeAlignedPhenotypeGroup>()?;
    module.add_class::<NativeAlignedSampleData>()?;
    module.add_class::<NativeBinaryCorrectionDiagnosticsRecordPlan>()?;
    module.add_class::<NativeBinaryCorrectionSummary>()?;
    module.add_class::<NativeBinaryCorrectionSummaryEmitPlan>()?;
    module.add_class::<NativeCallbackChunkIdentity>()?;
    module.add_class::<NativeCallbackObjectQueue>()?;
    module.add_class::<NativeCallbackObjectQueueGetResult>()?;
    module.add_class::<NativeCallbackQueueGetObservedResult>()?;
    module.add_class::<NativeCallbackRuntimeResources>()?;
    module.add_class::<NativeCallbackWaitSignal>()?;
    module.add_class::<NativeCallbackWorkerThread>()?;
    module.add_class::<NativeCallbackProgressCompletion>()?;
    module.add_class::<NativeCallbackProgressState>()?;
    module.add_class::<NativeCallbackProgressTelemetryEvent>()?;
    module.add_class::<NativeCallbackProgressTelemetryPlan>()?;
    module.add_class::<NativeCallbackProgressTelemetryRecord>()?;
    module.add_class::<NativeCallbackProgressUpdate>()?;
    module.add_class::<NativeCallbackQueueLimits>()?;
    module.add_class::<NativeCallbackQueueBackpressureObservation>()?;
    module.add_class::<NativeCallbackQueueGetAttemptPlan>()?;
    module.add_class::<NativeCallbackQueueGetObservationPlan>()?;
    module.add_class::<NativeCallbackQueueOperationObservationPlan>()?;
    module.add_class::<NativeCallbackQueuePutAttemptPlan>()?;
    module.add_class::<NativeCallbackQueuePutObservationPlan>()?;
    module.add_class::<NativeCallbackQueueStageBackpressureObservation>()?;
    module.add_class::<NativeCallbackQueueStageObservationPlan>()?;
    module.add_class::<NativeCallbackSchedulerState>()?;
    module.add_class::<NativeCallbackWorkerAbortPlan>()?;
    module.add_class::<NativeCallbackWorkerErrorRaisePlan>()?;
    module.add_class::<NativeCallbackWorkerErrorUpdatePlan>()?;
    module.add_class::<NativeCallbackWorkerFinishPlan>()?;
    module.add_class::<NativeCallbackWorkerJoinPlan>()?;
    module.add_class::<NativeCallbackWorkerStartPlan>()?;
    module.add_class::<NativeCallbackWorkerStartAttemptPlan>()?;
    module.add_class::<NativeBgenDeliveryCleanupPlan>()?;
    module.add_class::<NativeBgenDeliveryInvocationPlan>()?;
    module.add_class::<NativeCallbackWorkerLifecycleState>()?;
    module.add_class::<NativeCallbackWorkerShutdownTimeouts>()?;
    module.add_class::<NativeCallbackWorkerStopPlan>()?;
    module.add_class::<NativeCallbackWorkerStopPollPlan>()?;
    module.add_class::<NativeCallbackWorkerFinishLifecycleResult>()?;
    module.add_class::<NativeCallbackQueuePutResult>()?;
    module.add_class::<NativeDosageBufferAcquireResult>()?;
    module.add_class::<NativeDosageBufferPoolOperationResult>()?;
    module.add_class::<NativeDosageBufferReuseSelectionResult>()?;
    module.add_class::<NativeDosageWorkItemDrainResult>()?;
    module.add_class::<NativeDosageWorkItemGetResult>()?;
    module.add_class::<NativeDosageWorkItemStageDurationAttribution>()?;
    module.add_class::<NativeResultInFlightAcquireResult>()?;
    module.add_class::<NativeResultInFlightSlotReleaseResult>()?;
    module.add_class::<NativeResultWorkItemResourceReleaseResult>()?;
    module.add_class::<NativeResultWriteItemDrainResult>()?;
    module.add_class::<NativeResultWriteItemGetResult>()?;
    module.add_class::<NativeDosageBufferAcquireAttemptPlan>()?;
    module.add_class::<NativeDosageBufferDiscardAttemptPlan>()?;
    module.add_class::<NativeDosageBufferPoolObservationPlan>()?;
    module.add_class::<NativeDosageBufferPoolState>()?;
    module.add_class::<NativeDosageBufferRegisterAttemptPlan>()?;
    module.add_class::<NativeDosageBufferReturnAttemptPlan>()?;
    module.add_class::<NativeDosageBufferReusePlan>()?;
    module.add_class::<NativeDosageWorkDrainCompletionPlan>()?;
    module.add_class::<NativeDosageWorkHandoffPlan>()?;
    module.add_class::<NativeDosageWorkItemDispatchPlan>()?;
    module.add_class::<NativeDosageWorkItemStageDurationPlan>()?;
    module.add_class::<NativeGpuGenotypeFormatResolutionPlan>()?;
    module.add_class::<NativeMultiTraitChunkWritePlan>()?;
    module.add_class::<NativeMultiTraitOutputWritePlan>()?;
    module.add_class::<NativeNullLogisticNonconvergencePlan>()?;
    module.add_class::<NativeResultInFlightAcquireAttemptPlan>()?;
    module.add_class::<NativeResultInFlightAcquireObservationPlan>()?;
    module.add_class::<NativeResultInFlightReleaseAttemptPlan>()?;
    module.add_class::<NativeResultInFlightReleaseObservationPlan>()?;
    module.add_class::<NativeResultInFlightSlotState>()?;
    module.add_class::<NativeResultWriteDrainCompletionPlan>()?;
    module.add_class::<NativeResultWriteHandoffPlan>()?;
    module.add_class::<NativeResultWriteItemDispatchPlan>()?;
    module.add_class::<NativeResultWriteItemResourceReleasePlan>()?;
    module.add_class::<NativeSingleTraitOutputWritePlan>()?;
    module.add_class::<NativeVariantMajorDosageBatchHandoffPlan>()?;
    module.add_class::<NativeWriterFinishExecutionPlan>()?;
    module.add_class::<NativeGroupedAlignedSampleData>()?;
    module.add_class::<NativeInitializedOutputRun>()?;
    module.add_class::<NativeMultiAlignedSampleData>()?;
    module.add_class::<NativeOutputRunPaths>()?;
    module.add_class::<NativePreparedOutputRun>()?;
    module.add_class::<NativePipelineOutputInitialization>()?;
    module.add_class::<NativePipelineOutputPreparationBatch>()?;
    module.add_class::<NativeResolvedPhenotypeComputeGroup>()?;
    module.add_class::<NativeJaxRuntimeDiagnosticRecordPlan>()?;
    module.add_class::<NativeJaxRuntimeSetupLifecyclePlan>()?;
    module.add_class::<NativeJaxRuntimeSetupSession>()?;
    module.add_class::<NativeRayonThreadPoolConfigurationPlan>()?;
    module.add_class::<NativeRunRuntime>()?;
    module.add_class::<NativeRuntimeCompatibilityToken>()?;
    module.add_class::<NativeRuntimePolicy>()?;
    module.add_class::<NativeRuntimeState>()?;
    module.add_class::<NativeSecondSignalExceptionPlan>()?;
    module.add_class::<NativeShutdownController>()?;
    module.add_class::<NativeStageTimingRecorder>()?;
    module.add_class::<NativeStageTimingRecorderPlan>()?;
    module.add_class::<NativeTimingFileWritePlan>()?;
    module.add_function(wrap_pyfunction!(build_final_timing_outputs_write_started_diagnostic_payload, module)?)?;
    module.add_class::<OutputWriterSession>()?;
    module.add_class::<Regenie2RunEngine>()?;
    module.add_class::<RegeniePredictionSource>()?;
    module.add_class::<MultiRegeniePredictionSource>()?;
    module.add_class::<NativeTelemetryClosePlan>()?;
    module.add_class::<NativeTelemetryEventEmissionPlan>()?;
    module.add_class::<NativeTelemetryProgressEmissionPlan>()?;
    module.add_class::<NativeTelemetryProgressThrottle>()?;
    module.add_class::<NativeTelemetryRunSession>()?;
    module.add_class::<NativeTelemetrySessionPolicy>()?;
    module.add_class::<NativeTelemetrySession>()?;
    module.add_class::<VariantMetadata>()?;
    module.add_function(wrap_pyfunction!(resolve_prediction_loco_paths, module)?)?;
    module.add_function(wrap_pyfunction!(build_current_run_manifest_header_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_prepared_run_manifest_header_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_prepared_run_plan_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_file_content_sha256_value, module)?)?;
    module.add_function(wrap_pyfunction!(build_manifest_file_fingerprint_mapping_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_manifest_file_fingerprint_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_manifest_json_sha256, module)?)?;
    module.add_function(wrap_pyfunction!(build_empty_telemetry_writer_counters_payload, module)?)?;
    module.add_function(wrap_pyfunction!(attach_run_metadata_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_run_completed_event_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_run_interrupted_event_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_run_failed_event_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_run_completed_telemetry_fields, module)?)?;
    module.add_function(wrap_pyfunction!(build_run_failed_telemetry_fields, module)?)?;
    module.add_function(wrap_pyfunction!(build_run_interrupted_telemetry_fields, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_run_started_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_run_interrupted_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_run_failed_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_run_completed_diagnostic_payload, module)?)?;
    module
        .add_function(wrap_pyfunction!(build_runner_jax_runtime_configuration_started_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_execution_plan_build_started_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_execution_plan_prepared_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_execution_plan_dispatch_started_diagnostic_payload, module)?)?;
    module
        .add_function(wrap_pyfunction!(build_runner_execution_plan_finalization_started_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_multi_phenotype_dispatch_started_diagnostic_payload, module)?)?;
    module
        .add_function(wrap_pyfunction!(build_runner_single_phenotype_dispatch_started_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_binary_engine_dispatch_started_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_linear_engine_dispatch_started_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(
        build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_native_runtime_knobs_configured_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runner_metadata_artifacts_finalized_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_preflight_warning_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_io_output_resume_committed_chunks_diagnostic_payload, module)?)?;
    module
        .add_function(wrap_pyfunction!(build_native_dispatch_bgen_engine_constructing_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(
        build_native_dispatch_trusted_bgen_validation_started_diagnostic_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        build_callback_null_logistic_nonconvergence_warning_diagnostic_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_pipeline_gpu_genotype_format_resolved_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_native_dispatch_callback_drain_started_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_native_dispatch_delivery_started_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_native_dispatch_delivery_finished_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_native_dispatch_delivery_interrupted_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_native_dispatch_delivery_failed_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_native_dispatch_pipeline_finished_diagnostic_payload, module)?)?;
    module.add_function(wrap_pyfunction!(
        build_native_dispatch_writer_session_finish_started_diagnostic_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        build_native_dispatch_writer_sessions_finish_started_diagnostic_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        build_native_dispatch_writer_session_interrupted_flush_started_diagnostic_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_logging_runtime_policy_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_shutdown_signal_payload, module)?)?;
    module.add_function(wrap_pyfunction!(default_shutdown_signal_numbers, module)?)?;
    module.add_function(wrap_pyfunction!(plan_second_signal_exception, module)?)?;
    module.add_function(wrap_pyfunction!(raise_second_signal_exception, module)?)?;
    module.add_function(wrap_pyfunction!(build_execution_run_artifacts_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_multi_run_artifacts_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_phenotype_compute_group_id_value, module)?)?;
    module.add_function(wrap_pyfunction!(build_phenotype_compute_groups_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_phenotype_output_directory_name, module)?)?;
    module.add_function(wrap_pyfunction!(build_phenotype_run_artifacts_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_run_manifest_extension_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_runtime_policy_handle, module)?)?;
    module.add_function(wrap_pyfunction!(validate_pipeline_resume_compatibility, module)?)?;
    module.add_function(wrap_pyfunction!(initialize_pipeline_output_run_batch, module)?)?;
    module.add_function(wrap_pyfunction!(initialize_pipeline_output_runs, module)?)?;
    module.add_function(wrap_pyfunction!(build_preflight_report_payload, module)?)?;
    module.add_function(wrap_pyfunction!(validate_single_trait_preflight_shape_payload, module)?)?;
    module.add_function(wrap_pyfunction!(validate_multi_trait_preflight_shape_payload, module)?)?;
    module.add_function(wrap_pyfunction!(validate_binary_phenotype_case_control_counts, module)?)?;
    module.add_function(wrap_pyfunction!(validate_finite_array, module)?)?;
    module.add_function(wrap_pyfunction!(validate_covariate_matrix_rank, module)?)?;
    module.add_function(wrap_pyfunction!(validate_binary_phenotype_coding, module)?)?;
    module.add_function(wrap_pyfunction!(validate_single_prediction_preflight_shape, module)?)?;
    module.add_function(wrap_pyfunction!(validate_multi_prediction_preflight_shape, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_binary_correction_payload, module)?)?;
    module.add_function(wrap_pyfunction!(plan_association_backend_payload, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_association_mode_value, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_telemetry_output_run_root_value, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_telemetry_paths_payload, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_telemetry_session_policy_payload, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_telemetry_stream_file_value, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_jax_runtime_setup_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_default_local_cache_directory_value, module)?)?;
    module.add_function(wrap_pyfunction!(complete_jax_runtime_setup_validation_payload, module)?)?;
    module.add_function(wrap_pyfunction!(nvidia_driver_files_are_visible_value, module)?)?;
    module.add_function(wrap_pyfunction!(build_jax_runtime_setup_diagnostic_payloads, module)?)?;
    module.add_function(wrap_pyfunction!(build_jax_runtime_policy_payload, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_runtime_config_update_payloads, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_runtime_diagnostic_record, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_runtime_diagnostic_record_payload, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_runtime_setup_side_effects_payload, module)?)?;
    module.add_function(wrap_pyfunction!(plan_jax_gpu_validation_payload, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_preflight_variant_count, module)?)?;
    module.add_function(wrap_pyfunction!(build_callback_chunk_identity, module)?)?;
    module.add_function(wrap_pyfunction!(intersect_committed_chunk_identifier_sets, module)?)?;
    module.add_function(wrap_pyfunction!(plan_null_logistic_nonconvergence, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_manifest_gpu_genotype_format, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_effective_trusted_no_missing_diploid, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_bgen_delivery_method_value, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_callback_worker_backpressure_poll_timeout_seconds, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_callback_worker_stop_poll_timeout_seconds, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_delivery_callback_batch_size, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_grouped_union_callback_batch_size, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_native_callback_queue_limits, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_native_callback_worker_shutdown_timeouts, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_writer_finish_thread_count, module)?)?;
    module.add_function(wrap_pyfunction!(should_attempt_callback_worker_stop, module)?)?;
    module.add_function(wrap_pyfunction!(plan_callback_queue_backpressure_observation, module)?)?;
    module.add_function(wrap_pyfunction!(plan_callback_queue_operation_observation, module)?)?;
    module.add_function(wrap_pyfunction!(plan_callback_queue_stage_backpressure_observation, module)?)?;
    module.add_function(wrap_pyfunction!(plan_callback_queue_stage_observation, module)?)?;
    module.add_function(wrap_pyfunction!(plan_callback_worker_abort, module)?)?;
    module.add_function(wrap_pyfunction!(plan_callback_worker_finish, module)?)?;
    module.add_function(wrap_pyfunction!(plan_callback_worker_start, module)?)?;
    module.add_function(wrap_pyfunction!(plan_callback_worker_stop_poll, module)?)?;
    module.add_function(wrap_pyfunction!(plan_auto_gpu_genotype_format_after_trusted_validation, module)?)?;
    module.add_function(wrap_pyfunction!(format_dosage_callback_worker_error_message, module)?)?;
    module.add_function(wrap_pyfunction!(format_result_callback_worker_error_message, module)?)?;
    module.add_function(wrap_pyfunction!(plan_dosage_buffer_reuse, module)?)?;
    module.add_function(wrap_pyfunction!(plan_dosage_callback_worker_join, module)?)?;
    module.add_function(wrap_pyfunction!(plan_dosage_callback_worker_stop, module)?)?;
    module.add_function(wrap_pyfunction!(plan_gpu_genotype_format_auto_to_dosage, module)?)?;
    module.add_function(wrap_pyfunction!(plan_multi_trait_chunk_write, module)?)?;
    module.add_function(wrap_pyfunction!(plan_multi_trait_output_write, module)?)?;
    module.add_function(wrap_pyfunction!(plan_result_callback_worker_join, module)?)?;
    module.add_function(wrap_pyfunction!(plan_result_callback_worker_stop, module)?)?;
    module.add_function(wrap_pyfunction!(plan_result_write_handoff, module)?)?;
    module.add_function(wrap_pyfunction!(plan_result_write_item_dispatch, module)?)?;
    module.add_function(wrap_pyfunction!(plan_single_trait_binary_gpu_genotype_format_resolution, module)?)?;
    module.add_function(wrap_pyfunction!(plan_single_trait_output_write, module)?)?;
    module.add_function(wrap_pyfunction!(plan_dosage_work_handoff, module)?)?;
    module.add_function(wrap_pyfunction!(plan_dosage_work_item_dispatch, module)?)?;
    module.add_function(wrap_pyfunction!(plan_dosage_work_item_stage_duration, module)?)?;
    module.add_function(wrap_pyfunction!(plan_variant_major_dosage_batch_handoff, module)?)?;
    module.add_function(wrap_pyfunction!(plan_bgen_delivery_cleanup, module)?)?;
    module.add_function(wrap_pyfunction!(plan_bgen_delivery_invocation, module)?)?;
    module.add_function(wrap_pyfunction!(plan_writer_finish_execution, module)?)?;
    module.add_function(wrap_pyfunction!(plan_telemetry_close, module)?)?;
    module.add_function(wrap_pyfunction!(plan_telemetry_event_emission, module)?)?;
    module.add_function(wrap_pyfunction!(plan_telemetry_progress_emission, module)?)?;
    module.add_function(wrap_pyfunction!(plan_stage_timing_recorder, module)?)?;
    module.add_function(wrap_pyfunction!(plan_timing_file_write, module)?)?;
    module.add_function(wrap_pyfunction!(build_current_telemetry_event_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_telemetry_event_payload, module)?)?;
    module.add_function(wrap_pyfunction!(generate_telemetry_run_id_value, module)?)?;
    module.add_function(wrap_pyfunction!(format_telemetry_timestamp_value, module)?)?;
    module.add_function(wrap_pyfunction!(build_trusted_bgen_validation_cache_path_value, module)?)?;
    module.add_function(wrap_pyfunction!(build_trusted_bgen_validation_cache_payload, module)?)?;
    module.add_function(wrap_pyfunction!(build_trusted_bgen_validation_fingerprint_value, module)?)?;
    module.add_function(wrap_pyfunction!(finalize_output_run_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(initialize_output_run, module)?)?;
    module.add_function(wrap_pyfunction!(load_run_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_output_run, module)?)?;
    module.add_function(wrap_pyfunction!(read_manifest_committed_chunk_identifiers, module)?)?;
    module.add_function(wrap_pyfunction!(repair_strict_manifest_chunk_commits, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_output_run_paths, module)?)?;
    module.add_function(wrap_pyfunction!(scan_committed_chunk_identifiers, module)?)?;
    module.add_function(wrap_pyfunction!(summarize_variant_major_dosage_chunk_stats, module)?)?;
    module.add_function(wrap_pyfunction!(paths_refer_to_same_file_value, module)?)?;
    module.add_function(wrap_pyfunction!(validate_run_manifest_compatibility, module)?)?;
    module.add_function(wrap_pyfunction!(validate_strict_manifest_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(write_regenie2_multi_native_chunk, module)?)?;
    module.add_function(wrap_pyfunction!(write_regenie2_multi_native_chunk_f64, module)?)?;
    module.add_function(wrap_pyfunction!(write_run_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(configure_bgen_decode_tile_variant_count, module)?)?;
    module.add_function(wrap_pyfunction!(configure_rayon_global_thread_pool, module)?)?;
    module.add_function(wrap_pyfunction!(format_rayon_thread_pool_configuration_error_value, module)?)?;
    module.add_function(wrap_pyfunction!(describe_logging_runtime_policy_value, module)?)?;
    module.add_function(wrap_pyfunction!(emit_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(emit_diagnostic_event_fields, module)?)?;
    module.add_function(wrap_pyfunction!(initialize_logging, module)?)?;
    module.add_function(wrap_pyfunction!(shutdown_logging, module)?)?;
    module.add_function(wrap_pyfunction!(plan_genotype_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(align_sample_data, module)?)?;
    module.add_function(wrap_pyfunction!(align_grouped_sample_data, module)?)?;
    module.add_function(wrap_pyfunction!(align_multi_sample_data, module)?)?;
    module.add_function(wrap_pyfunction!(align_sample_data_from_sample_file, module)?)?;
    module.add_function(wrap_pyfunction!(align_multi_sample_data_from_sample_file, module)?)?;
    module.add_function(wrap_pyfunction!(render_run_completed_lines, module)?)?;
    module.add_function(wrap_pyfunction!(render_run_failed_lines, module)?)?;
    module.add_function(wrap_pyfunction!(render_run_interrupted_lines, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_complete_case_compute_group, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_per_phenotype_compute_group, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_single_phenotype_compute_group, module)?)?;
    Ok(())
}
