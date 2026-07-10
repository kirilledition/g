#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::fn_params_excessive_bools)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use crate::binding::convert::int::{optional_usize_to_i64 as option_usize_to_i64, usize_to_i64};
use g_engine as native_engine_debug;
use g_engine::Regenie2RunEngineCore;
use g_input as native_input;
use g_plan as native_plan;
use g_runtime as native_run_events;
use g_runtime as native_trusted_validation;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::backend::{JaxBackendConfig, PyJaxBackend};
use super::backend_delivery::{
    self, AssociationDeliveryRequest, AssociationDeliverySettings, GroupedUnionAssociationDeliveryRequest,
    MultiAssociationDeliveryRequest, SingleAssociationDeliveryRequest,
};
use super::errors::{
    convert_bgen_error, convert_prediction_error, convert_preflight_error, convert_schedule_error,
    convert_trusted_bgen_validation_error,
};
use super::input::{
    align_grouped_sample_data_for_engine, align_multi_sample_data_for_engine, align_sample_data_for_engine,
    load_grouped_prediction_sources, load_multi_prediction_source, load_prediction_source, parse_sample_key_mode,
};
use super::output::{self, OutputWriterSession};
use super::run_lifecycle::{NativePreparedOutputBundle, NativeRunLifecycleSession};
use super::timing::NativeStageTimingRecorder;
use crate::binding::telemetry::{
    run_events::{self, NativeRunArtifacts},
    session as telemetry_session,
};

struct NativeSingleTraitPipelineBundle {
    aligned_sample_data: native_input::AlignedSampleData,
    prediction_source: native_input::PredictionSource,
    writer_session: Arc<OutputWriterSession>,
    committed_chunk_identifiers: Vec<usize>,
}

pub(crate) struct NativeRunEngineSession {
    lifecycle: NativeRunLifecycleSession,
    engine: Mutex<Option<Regenie2RunEngineCore>>,
}

struct NativeRunResolvedExecution {
    backend_plan: native_plan::AssociationBackendPlan,
    requested_gpu_genotype_format: String,
    resolved_gpu_genotype_format: String,
    effective_trusted_no_missing_diploid: bool,
    binary_kernel_config_json: Option<String>,
    null_logistic_nonconvergence_policy: String,
}

struct NativeGroupedRunInputState {
    compute_group: native_input::ResolvedPhenotypeComputeGroup,
    phenotype_indices: Vec<usize>,
    aligned_sample_data: native_input::MultiAlignedSampleData,
    prediction_source: native_input::MultiPredictionSource,
    sample_indices: Vec<usize>,
    sample_count: i64,
}

struct OutputBundleDeliveryState {
    writer_sessions: Vec<Arc<OutputWriterSession>>,
    committed_chunk_identifier_sets: Vec<BTreeSet<usize>>,
}

struct OutputWriterAbortGuard {
    writer_sessions: Vec<Arc<OutputWriterSession>>,
    armed: bool,
}

impl Drop for OutputWriterAbortGuard {
    fn drop(&mut self) {
        if self.armed {
            output::abort_output_writer_sessions_for_delivery(&self.writer_sessions);
        }
    }
}

impl NativeRunEngineSession {
    #[allow(clippy::too_many_lines)]
    fn prepare_single_trait_pipeline_bundle<'py>(
        &self,
        py: Python<'py>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
        pipeline_label: &str,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    ) -> PyResult<NativeSingleTraitPipelineBundle> {
        let phenotype_name = run_request
            .phenotype_runs
            .first()
            .ok_or_else(|| PyRuntimeError::new_err("Single-trait run request has no phenotype run."))?
            .phenotype_name
            .clone();
        let covariate_names = Some(run_request.input.covariate_names.clone());
        let association_mode = run_request.association_mode.as_str().to_string();
        let association_backend_kind = resolved_execution.backend_plan.kind.as_str().to_string();
        let jax_device = resolved_execution.backend_plan.device.as_str().to_string();
        let genotype_format = resolved_execution.resolved_gpu_genotype_format.clone();
        let requested_gpu_genotype_format = resolved_execution.requested_gpu_genotype_format.clone();
        let score_dtype = run_request.compute.score_dtype.as_str().to_string();
        let firth_dtype = run_request.compute.firth_dtype.as_str().to_string();
        let binary_kernel_config_json = resolved_execution.binary_kernel_config_json.clone();
        let sample_key_mode = run_request.input.sample_key_mode.as_str().to_string();
        let is_binary_trait = run_request.trait_request.trait_type == native_plan::RegenieTraitType::Binary;
        let bgen_path = run_request.input.bgen_path.clone();
        let sample_path = run_request.input.sample_path.clone();
        let phenotype_path = run_request.input.phenotype_path.clone();
        let covariate_path = run_request.input.covariate_path.clone();
        let prediction_list_path = run_request.input.prediction_list_path.clone();
        let chunk_size = u32_value_as_usize(run_request.trait_request.chunk_size, "trait chunk size")?;
        let variant_limit =
            run_request.compute.variant_limit.map(|value| u32_value_as_usize(value, "variant limit")).transpose()?;
        let effective_trusted_no_missing_diploid = resolved_execution.effective_trusted_no_missing_diploid;
        let trusted_bgen_validation_mode = run_request.compute.trusted_bgen_validation_mode.as_str().to_string();
        let stage_timing_recorder_reference = stage_timing_recorder;
        run_events::record_pipeline_single_trait_started_diagnostic_event(
            &association_mode,
            &phenotype_name,
            pipeline_label,
        )?;
        let engine_was_open = self.lock_engine()?.is_some();
        if engine_was_open {
            run_events::record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
                None,
                Some(phenotype_name.as_str()),
                pipeline_label,
            )?;
            run_events::record_association_backend_selected_telemetry(
                telemetry_session,
                &association_mode,
                &association_backend_kind,
                &jax_device,
                &genotype_format,
                Some(phenotype_name.clone()),
                None,
            )?;
        } else {
            let engine_start_time = Instant::now();
            run_events::record_pipeline_bgen_engine_open_started_diagnostic_event(
                None,
                Some(phenotype_name.as_str()),
                pipeline_label,
                effective_trusted_no_missing_diploid,
                option_usize_to_i64(variant_limit, "BGEN variant limit")?,
            )?;
            run_events::record_association_backend_selected_telemetry(
                telemetry_session,
                &association_mode,
                &association_backend_kind,
                &jax_device,
                &genotype_format,
                Some(phenotype_name.clone()),
                None,
            )?;
            self.open_bgen_engine_internal(
                py,
                &bgen_path,
                chunk_size,
                variant_limit,
                effective_trusted_no_missing_diploid,
                Some(trusted_bgen_validation_mode.as_str()),
            )?;
            record_stage_duration(stage_timing_recorder_reference, "bgen_engine_open_index_setup", engine_start_time)?;
        }

        let (engine_sample_count, engine_variant_count) = self.with_open_engine(|engine| {
            Ok((
                usize_to_i64(engine.reader().sample_count(), "BGEN sample count")?,
                usize_to_i64(engine.reader().variant_count(), "BGEN variant count")?,
            ))
        })?;
        run_events::record_pipeline_bgen_engine_opened_diagnostic_event(
            None,
            Some(phenotype_name.as_str()),
            pipeline_label,
            engine_sample_count,
            engine_variant_count,
        )?;
        run_events::record_bgen_engine_opened_telemetry(
            telemetry_session,
            &association_mode,
            &association_backend_kind,
            engine_sample_count,
            engine_variant_count,
            Some(phenotype_name.clone()),
            None,
        )?;

        let alignment_start_time = Instant::now();
        run_events::record_pipeline_single_trait_input_load_started_diagnostic_event(&phenotype_name, pipeline_label)?;
        let aligned_sample_data = self.with_open_engine(|engine| {
            align_sample_data_for_engine(
                engine,
                py,
                sample_path,
                phenotype_path,
                phenotype_name.clone(),
                covariate_path,
                covariate_names,
                is_binary_trait,
                &sample_key_mode,
            )
        })?;
        record_stage_duration(
            stage_timing_recorder_reference,
            "sample_phenotype_covariate_alignment",
            alignment_start_time,
        )?;
        let sample_count = usize_to_i64(aligned_sample_data.sample_indices.len(), "Aligned sample count")?;
        let covariate_count = usize_to_i64(aligned_sample_data.covariate_names.len(), "Covariate count")?;
        run_events::record_pipeline_single_trait_input_aligned_diagnostic_event(
            covariate_count,
            &phenotype_name,
            pipeline_label,
            sample_count,
        )?;
        run_events::record_sample_alignment_completed_telemetry(
            telemetry_session,
            &association_mode,
            Some(phenotype_name.clone()),
            None,
            Some(sample_count),
            Some(covariate_count),
            None,
        )?;

        let prediction_start_time = Instant::now();
        run_events::record_pipeline_single_trait_prediction_source_load_started_diagnostic_event(
            &phenotype_name,
            pipeline_label,
        )?;
        let prediction_source =
            load_prediction_source(&prediction_list_path, &phenotype_name, &aligned_sample_data, &sample_key_mode)?;
        record_stage_duration(stage_timing_recorder_reference, "prediction_source_load", prediction_start_time)?;
        run_events::record_prediction_source_loaded_telemetry(
            telemetry_session,
            &association_mode,
            Some(phenotype_name.clone()),
            None,
        )?;

        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let phenotype_compute_group = native_input::resolve_single_phenotype_compute_group(
            &aligned_sample_data,
            phenotype_name.clone(),
            Some(prediction_list_path.as_str()),
            parsed_sample_key_mode,
        );

        let preflight_start_time = Instant::now();
        run_events::record_pipeline_single_trait_preflight_started_diagnostic_event(
            &phenotype_name,
            pipeline_label,
            effective_trusted_no_missing_diploid,
            option_usize_to_i64(variant_limit, "BGEN variant limit")?,
        )?;
        let preflight_shape = native_engine_debug::validate_single_trait_preflight_values(
            &aligned_sample_data.phenotype_vector,
            aligned_sample_data.covariate_row_count,
            aligned_sample_data.covariate_column_count,
            &aligned_sample_data.covariate_matrix_values,
            is_binary_trait,
        )
        .map_err(|error| convert_preflight_error(&error))?;
        let required_chromosomes = self.with_open_engine(|engine| {
            engine.required_chromosomes(variant_limit).map_err(|error| convert_preflight_error(&error))
        })?;
        for chromosome in &required_chromosomes {
            let prediction_values = prediction_source
                .chromosome_predictions(chromosome)
                .map_err(|error| convert_prediction_error("chromosome_predictions", &error))?;
            native_engine_debug::validate_single_prediction_values(
                chromosome,
                prediction_values,
                preflight_shape.sample_count,
            )
            .map_err(|error| convert_preflight_error(&error))?;
        }
        let chromosome_count = usize_to_i64(required_chromosomes.len(), "Chromosome count")?;
        let preflight_report = native_engine_debug::build_preflight_report_payload(
            preflight_shape.sample_count,
            preflight_shape.covariate_count,
            chromosome_count,
            effective_trusted_no_missing_diploid,
        )
        .map_err(|error| convert_preflight_error(&error))?;
        run_events::record_preflight_warning_diagnostic_events(
            preflight_report.warning_messages.clone(),
            preflight_report.chromosome_count,
            preflight_report.covariate_count,
            "single_trait",
            preflight_report.sample_count,
            effective_trusted_no_missing_diploid,
        )?;
        record_stage_duration(stage_timing_recorder_reference, "preflight_validation", preflight_start_time)?;
        run_events::record_pipeline_single_trait_preflight_completed_diagnostic_event(
            preflight_report.chromosome_count,
            preflight_report.covariate_count,
            &phenotype_name,
            pipeline_label,
            preflight_report.sample_count,
        )?;
        run_events::record_single_trait_preflight_completed_telemetry(
            telemetry_session,
            &association_mode,
            &phenotype_name,
            preflight_report.sample_count,
            preflight_report.covariate_count,
            preflight_report.chromosome_count,
        )?;

        let output_group = build_single_trait_output_group(
            &phenotype_name,
            &aligned_sample_data,
            &phenotype_compute_group,
            preflight_report.sample_count,
        )?;
        let mut output_bundles = self.lifecycle.prepare_output_bundles_from_runtime_plan_internal(
            py,
            vec![output_group],
            engine_variant_count,
            effective_trusted_no_missing_diploid,
            sample_key_mode,
            binary_kernel_config_json,
            requested_gpu_genotype_format,
            genotype_format,
            score_dtype,
            firth_dtype,
            stage_timing_recorder_reference,
        )?;
        let output_bundle = output_bundles
            .pop()
            .ok_or_else(|| PyRuntimeError::new_err("Single-trait output preparation returned no bundle."))?;
        let writer_session = output_bundle.writer_session_handle(0)?;
        let committed_chunk_identifiers = output_bundle.committed_chunk_identifiers_usize(0)?;
        Ok(NativeSingleTraitPipelineBundle {
            aligned_sample_data,
            prediction_source,
            writer_session,
            committed_chunk_identifiers,
        })
    }
}

impl NativeRunEngineSession {
    pub(crate) fn from_config_internal(py: Python<'_>, config: &g_interface::RegenieConfigData) -> PyResult<Self> {
        Ok(Self { lifecycle: NativeRunLifecycleSession::from_config(py, config)?, engine: Mutex::new(None) })
    }

    pub(crate) fn run_with_backend_internal<'py>(
        &self,
        py: Python<'py>,
        backend: Arc<PyJaxBackend>,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    ) -> PyResult<NativeRunArtifacts> {
        self.run_to_completion_internal(py, backend, telemetry_session, stage_timing_recorder)
    }

    fn lock_engine(&self) -> PyResult<MutexGuard<'_, Option<Regenie2RunEngineCore>>> {
        self.engine.lock().map_err(|_| PyRuntimeError::new_err("Native BGEN engine mutex was poisoned."))
    }

    fn open_bgen_engine_internal(
        &self,
        py: Python<'_>,
        bgen_path: &str,
        chunk_size: usize,
        variant_limit: Option<usize>,
        trusted_no_missing_diploid: bool,
        trusted_bgen_validation_mode: Option<&str>,
    ) -> PyResult<bool> {
        if self.lock_engine()?.is_some() {
            return Ok(false);
        }
        let engine = open_bgen_engine_core(py, bgen_path, chunk_size, variant_limit, trusted_no_missing_diploid)?;
        if trusted_no_missing_diploid {
            let validation_mode = trusted_bgen_validation_mode.ok_or_else(|| {
                PyValueError::new_err("trusted_bgen_validation_mode is required for trusted no-missing diploid BGEN.")
            })?;
            validate_trusted_no_missing_diploid_with_default_cache_for_engine(&engine, py, bgen_path, validation_mode)?;
        }
        *self.lock_engine()? = Some(engine);
        Ok(true)
    }

    fn with_open_engine<T>(&self, operation: impl FnOnce(&Regenie2RunEngineCore) -> PyResult<T>) -> PyResult<T> {
        let engine_guard = self.lock_engine()?;
        let engine = engine_guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Native run engine session has no open BGEN engine."))?;
        operation(engine)
    }

    #[allow(clippy::too_many_lines)]
    fn run_to_completion_internal<'py>(
        &self,
        py: Python<'py>,
        backend: Arc<PyJaxBackend>,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    ) -> PyResult<NativeRunArtifacts> {
        let output_start_time = Instant::now();
        run_events::record_runner_execution_plan_build_started_diagnostic_event()?;
        let run_request = self.lifecycle.run_request_data().clone();
        let phenotype_count = usize_to_i64(run_request.phenotype_runs.len(), "Phenotype count")?;
        let resolved_execution =
            self.resolve_run_execution(py, telemetry_session, stage_timing_recorder, &run_request)?;
        run_events::record_execution_plan_prepared_events(
            telemetry_session,
            run_request.association_mode.as_str(),
            run_request.trait_request.trait_type.as_str(),
            phenotype_count,
            i64::from(run_request.trait_request.chunk_size),
            run_request.compute.variant_limit.map(i64::from),
            run_request.compute.device.as_str(),
        )?;
        record_stage_duration(stage_timing_recorder, "output_run_preparation", output_start_time)?;
        run_events::record_runner_execution_plan_dispatch_started_diagnostic_event(
            phenotype_count,
            run_request.association_mode.as_str(),
        )?;
        self.lifecycle.mark_dispatch_started_internal()?;
        let final_output_paths = if run_request.phenotype_runs.len() > 1 {
            self.run_multi_trait_to_completion(
                py,
                Arc::clone(&backend),
                telemetry_session,
                stage_timing_recorder,
                &run_request,
                &resolved_execution,
            )?
        } else {
            self.run_single_trait_to_completion(
                py,
                backend,
                telemetry_session,
                stage_timing_recorder,
                &run_request,
                &resolved_execution,
            )?
        };
        run_events::record_runner_execution_plan_finalization_started_diagnostic_event(
            phenotype_count,
            run_request.association_mode.as_str(),
        )?;
        let artifacts = self.lifecycle.finalize_success_artifacts(final_output_paths)?;
        run_events::record_runner_metadata_artifacts_finalized_diagnostic_event(
            run_request.association_mode.as_str(),
            phenotype_count,
        )?;
        Ok(artifacts)
    }

    fn resolve_run_execution<'py>(
        &self,
        py: Python<'py>,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
    ) -> PyResult<NativeRunResolvedExecution> {
        let backend_config = JaxBackendConfig::new(self.lifecycle.config_data().clone())
            .map_err(super::errors::convert_host_policy_error)?;
        let binary_kernel_config_json = backend_config.binary_kernel_config_json()?;
        let resolved_gpu_genotype_format =
            self.resolve_gpu_genotype_format(py, telemetry_session, stage_timing_recorder, run_request)?;
        let backend_plan = native_plan::plan_association_backend(
            run_request.association_mode,
            run_request.compute.device,
            resolved_gpu_genotype_format,
        )
        .map_err(|error| super::errors::convert_prepared_plan_error(&error))?;
        let effective_trusted_no_missing_diploid = native_engine_debug::resolve_effective_trusted_no_missing_diploid(
            run_request.compute.trusted_no_missing_diploid,
            backend_plan.resolved_genotype_format == native_plan::GpuGenotypeFormat::Packed8,
        );
        Ok(NativeRunResolvedExecution {
            backend_plan,
            requested_gpu_genotype_format: run_request.compute.requested_gpu_genotype_format.as_str().to_string(),
            resolved_gpu_genotype_format: resolved_gpu_genotype_format.as_str().to_string(),
            effective_trusted_no_missing_diploid,
            binary_kernel_config_json,
            null_logistic_nonconvergence_policy: self
                .lifecycle
                .config_data()
                .g_compute
                .null_logistic_nonconvergence_policy
                .as_str()
                .to_string(),
        })
    }

    fn resolve_gpu_genotype_format<'py>(
        &self,
        py: Python<'py>,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
    ) -> PyResult<native_plan::GpuGenotypeFormat> {
        let is_single_binary = run_request.phenotype_runs.len() == 1
            && run_request.trait_request.trait_type == native_plan::RegenieTraitType::Binary;
        if !is_single_binary {
            let resolution_reason =
                if run_request.phenotype_runs.len() == 1 { "single_trait_linear" } else { "multi_phenotype" };
            let native_resolution_plan = native_engine_debug::plan_gpu_genotype_format_auto_to_dosage(
                run_request.compute.requested_gpu_genotype_format.as_str(),
                resolution_reason,
            )
            .map_err(|error| convert_schedule_error(&error))?;
            run_events::record_gpu_genotype_format_resolved_native_plan_events(
                telemetry_session,
                &native_resolution_plan,
            )?;
            return concrete_gpu_genotype_format_from_resolution_plan(&native_resolution_plan);
        }

        let phenotype_name = run_request
            .phenotype_runs
            .first()
            .ok_or_else(|| PyRuntimeError::new_err("Single-trait run request has no phenotype run."))?
            .phenotype_name
            .clone();
        let existing_manifest_json = self.lifecycle.prepared_run_existing_manifest_json(&phenotype_name)?;
        let manifest_fields = manifest_gpu_genotype_format_fields(existing_manifest_json.as_deref())?;
        let native_resolution_plan = native_engine_debug::plan_single_trait_binary_gpu_genotype_format_resolution(
            run_request.compute.requested_gpu_genotype_format.as_str(),
            manifest_fields.0.as_deref(),
            manifest_fields.1.as_deref(),
            self.lifecycle.output_resume_value(),
            run_request.compute.device.as_str(),
        )
        .map_err(|error| convert_schedule_error(&error))?;
        run_events::record_gpu_genotype_format_resolved_native_plan_events(telemetry_session, &native_resolution_plan)?;
        if !native_resolution_plan.requires_trusted_validation {
            return concrete_gpu_genotype_format_from_resolution_plan(&native_resolution_plan);
        }

        let trusted_resolution_plan =
            match self.try_open_trusted_bgen_engine_for_gpu_format_resolution(py, stage_timing_recorder, run_request) {
                Ok(()) => native_engine_debug::plan_auto_gpu_genotype_format_after_trusted_validation(None),
                Err(error) => {
                    let error_message = error.value(py).str()?.to_string_lossy().into_owned();
                    native_engine_debug::plan_auto_gpu_genotype_format_after_trusted_validation(Some(&error_message))
                }
            };
        run_events::record_gpu_genotype_format_resolved_native_plan_events(
            telemetry_session,
            &trusted_resolution_plan,
        )?;
        concrete_gpu_genotype_format_from_resolution_plan(&trusted_resolution_plan)
    }

    fn try_open_trusted_bgen_engine_for_gpu_format_resolution(
        &self,
        py: Python<'_>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
    ) -> PyResult<()> {
        let engine_start_time = Instant::now();
        self.open_bgen_engine_internal(
            py,
            &run_request.input.bgen_path,
            u32_value_as_usize(run_request.trait_request.chunk_size, "trait chunk size")?,
            run_request.compute.variant_limit.map(|value| u32_value_as_usize(value, "variant limit")).transpose()?,
            true,
            Some(run_request.compute.trusted_bgen_validation_mode.as_str()),
        )?;
        record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_single_trait_to_completion<'py>(
        &self,
        py: Python<'py>,
        backend: Arc<PyJaxBackend>,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
    ) -> PyResult<Vec<Option<String>>> {
        let phenotype_run = run_request
            .phenotype_runs
            .first()
            .ok_or_else(|| PyRuntimeError::new_err("Single-trait run request has no phenotype run."))?;
        run_events::record_runner_single_phenotype_dispatch_started_diagnostic_event(
            run_request.association_mode.as_str(),
            &phenotype_run.phenotype_name,
        )?;
        if run_request.trait_request.trait_type == native_plan::RegenieTraitType::Binary {
            run_events::record_runner_binary_engine_dispatch_started_diagnostic_event(&phenotype_run.phenotype_name)?;
        } else {
            run_events::record_runner_linear_engine_dispatch_started_diagnostic_event(&phenotype_run.phenotype_name)?;
        }
        let pipeline_label = if run_request.trait_request.trait_type == native_plan::RegenieTraitType::Binary {
            "binary"
        } else {
            "linear"
        };
        let bundle = self.prepare_single_trait_pipeline_bundle(
            py,
            run_request,
            resolved_execution,
            pipeline_label,
            telemetry_session,
            stage_timing_recorder,
        )?;
        let writer_sessions = vec![Arc::clone(&bundle.writer_session)];
        let settings = association_delivery_settings(
            run_request,
            resolved_execution,
            &writer_sessions,
            vec![bundle.committed_chunk_identifiers.into_iter().collect()],
            bundle.aligned_sample_data.sample_indices.clone(),
        )?;
        let request = AssociationDeliveryRequest::Single(SingleAssociationDeliveryRequest {
            aligned_sample_data: bundle.aligned_sample_data,
            prediction_source: bundle.prediction_source,
            settings,
        });
        let final_output_path = self
            .run_native_association_delivery(
                py,
                backend,
                request,
                &writer_sessions,
                stage_timing_recorder,
                "Native BGEN",
                i64::from(run_request.output.writer_thread_count),
            )?
            .into_iter()
            .next()
            .flatten();
        run_events::record_phenotype_writer_finished_telemetry(
            telemetry_session,
            run_request.association_mode.as_str(),
            &phenotype_run.phenotype_name,
            final_output_path.clone(),
        )?;
        Ok(vec![final_output_path])
    }

    #[allow(clippy::too_many_arguments)]
    fn run_multi_trait_to_completion<'py>(
        &self,
        py: Python<'py>,
        backend: Arc<PyJaxBackend>,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
    ) -> PyResult<Vec<Option<String>>> {
        let phenotype_count = usize_to_i64(run_request.phenotype_runs.len(), "Phenotype count")?;
        run_events::record_runner_multi_phenotype_dispatch_started_diagnostic_event(
            phenotype_count,
            run_request.association_mode.as_str(),
        )?;
        if run_request.trait_request.trait_type == native_plan::RegenieTraitType::Binary {
            run_events::record_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_event(phenotype_count)?;
        } else {
            run_events::record_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_event(phenotype_count)?;
        }
        let final_output_paths = match run_request.compute.multi_phenotype_sample_mode {
            native_plan::MultiPhenotypeSampleMode::CompleteCase => self.run_complete_case_multi_trait_to_completion(
                py,
                Arc::clone(&backend),
                telemetry_session,
                stage_timing_recorder,
                run_request,
                resolved_execution,
            )?,
            native_plan::MultiPhenotypeSampleMode::PerPhenotype => self.run_grouped_per_phenotype_to_completion(
                py,
                backend,
                telemetry_session,
                stage_timing_recorder,
                run_request,
                resolved_execution,
            )?,
        };
        run_events::record_multi_phenotype_writer_finished_telemetry(
            telemetry_session,
            run_request.association_mode.as_str(),
            phenotype_count,
            final_output_paths.clone(),
        )?;
        Ok(final_output_paths)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn run_complete_case_multi_trait_to_completion<'py>(
        &self,
        py: Python<'py>,
        backend: Arc<PyJaxBackend>,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
    ) -> PyResult<Vec<Option<String>>> {
        let planned_compute_group = complete_case_compute_group_from_request(run_request)?;
        let phenotype_count = usize_to_i64(planned_compute_group.phenotype_names.len(), "Phenotype count")?;
        run_events::record_pipeline_multi_trait_started_diagnostic_event(
            run_request.association_mode.as_str(),
            phenotype_count,
            native_plan::MultiPhenotypeSampleMode::CompleteCase.as_str(),
        )?;
        self.open_pipeline_bgen_engine_with_events(
            py,
            telemetry_session,
            stage_timing_recorder,
            run_request,
            resolved_execution,
            "multi-phenotype",
            None,
            Some(phenotype_count),
        )?;
        let alignment_start_time = Instant::now();
        run_events::record_pipeline_multi_trait_input_load_started_diagnostic_event(phenotype_count)?;
        let aligned_sample_data = self.with_open_engine(|engine| {
            align_multi_sample_data_for_engine(
                engine,
                py,
                run_request.input.sample_path.clone(),
                run_request.input.phenotype_path.clone(),
                planned_compute_group.phenotype_names.clone(),
                run_request.input.covariate_path.clone(),
                Some(run_request.input.covariate_names.clone()),
                run_request.trait_request.trait_type == native_plan::RegenieTraitType::Binary,
                run_request.input.sample_key_mode.as_str(),
            )
        })?;
        let resolved_compute_group = native_input::resolve_complete_case_compute_group(
            &aligned_sample_data,
            u32_indices_to_usize(&planned_compute_group.phenotype_indices, "phenotype compute group index")?,
            planned_compute_group.phenotype_names.clone(),
            Some(run_request.input.prediction_list_path.as_str()),
            parse_sample_key_mode(run_request.input.sample_key_mode.as_str())?,
        );
        record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)?;
        let sample_count = usize_to_i64(aligned_sample_data.sample_indices.len(), "Aligned sample count")?;
        let covariate_count = usize_to_i64(aligned_sample_data.covariate_names.len(), "Covariate count")?;
        run_events::record_pipeline_multi_trait_input_aligned_diagnostic_event(
            covariate_count,
            phenotype_count,
            sample_count,
        )?;
        run_events::record_sample_alignment_completed_telemetry(
            telemetry_session,
            run_request.association_mode.as_str(),
            None,
            Some(phenotype_count),
            Some(sample_count),
            Some(covariate_count),
            None,
        )?;
        Self::record_multi_phenotype_sample_summary(
            telemetry_session,
            run_request,
            native_plan::MultiPhenotypeSampleMode::CompleteCase,
            std::slice::from_ref(&resolved_compute_group),
            &[sample_count],
        )?;
        let prediction_start_time = Instant::now();
        run_events::record_pipeline_multi_trait_prediction_source_load_started_diagnostic_event(phenotype_count)?;
        let prediction_source = load_multi_prediction_source(
            &run_request.input.prediction_list_path,
            &aligned_sample_data,
            run_request.input.sample_key_mode.as_str(),
        )?;
        record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)?;
        self.run_multi_group_preflight(
            telemetry_session,
            stage_timing_recorder,
            run_request,
            resolved_execution,
            &aligned_sample_data,
            &prediction_source,
        )?;
        let output_bundle = self.prepare_multi_trait_output_bundle(
            py,
            stage_timing_recorder,
            run_request,
            resolved_execution,
            &aligned_sample_data,
            &resolved_compute_group,
            native_plan::MultiPhenotypeSampleMode::CompleteCase,
        )?;
        let OutputBundleDeliveryState { writer_sessions, committed_chunk_identifier_sets } =
            output_bundle_delivery_state(&output_bundle)?;
        let settings = association_delivery_settings(
            run_request,
            resolved_execution,
            &writer_sessions,
            committed_chunk_identifier_sets,
            aligned_sample_data.sample_indices.clone(),
        )?;
        let request = AssociationDeliveryRequest::Multi(MultiAssociationDeliveryRequest {
            aligned_sample_data,
            prediction_source,
            settings,
        });
        self.run_native_association_delivery(
            py,
            backend,
            request,
            &writer_sessions,
            stage_timing_recorder,
            "Multi-phenotype native BGEN",
            i64::from(run_request.output.writer_thread_count),
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn run_grouped_per_phenotype_to_completion<'py>(
        &self,
        py: Python<'py>,
        backend: Arc<PyJaxBackend>,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
    ) -> PyResult<Vec<Option<String>>> {
        let phenotype_names = phenotype_names_from_request(run_request);
        let phenotype_count = usize_to_i64(phenotype_names.len(), "Phenotype count")?;
        run_events::record_pipeline_grouped_per_phenotype_started_diagnostic_event(
            run_request.association_mode.as_str(),
            phenotype_count,
            native_plan::MultiPhenotypeSampleMode::PerPhenotype.as_str(),
        )?;
        self.open_pipeline_bgen_engine_with_events(
            py,
            telemetry_session,
            stage_timing_recorder,
            run_request,
            resolved_execution,
            "grouped per-phenotype",
            None,
            Some(phenotype_count),
        )?;
        let alignment_start_time = Instant::now();
        let grouped_aligned_sample_data = self.with_open_engine(|engine| {
            align_grouped_sample_data_for_engine(
                engine,
                py,
                run_request.input.sample_path.clone(),
                run_request.input.phenotype_path.clone(),
                phenotype_names.clone(),
                run_request.input.covariate_path.clone(),
                Some(run_request.input.covariate_names.clone()),
                run_request.trait_request.trait_type == native_plan::RegenieTraitType::Binary,
                run_request.input.sample_key_mode.as_str(),
            )
        })?;
        let prediction_sources = load_grouped_prediction_sources(
            &run_request.input.prediction_list_path,
            &grouped_aligned_sample_data,
            run_request.input.sample_key_mode.as_str(),
        )?;
        if grouped_aligned_sample_data.len() != prediction_sources.len() {
            return Err(PyValueError::new_err("Grouped prediction source count does not match aligned group count."));
        }
        let grouped_run_inputs =
            Self::build_grouped_run_inputs(run_request, grouped_aligned_sample_data, prediction_sources)?;
        record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)?;
        run_events::record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event(
            phenotype_count,
            usize_to_i64(grouped_run_inputs.len(), "Phenotype group count")?,
        )?;
        run_events::record_sample_alignment_completed_telemetry(
            telemetry_session,
            run_request.association_mode.as_str(),
            None,
            Some(phenotype_count),
            None,
            None,
            Some(usize_to_i64(grouped_run_inputs.len(), "Phenotype group count")?),
        )?;
        Self::record_multi_phenotype_sample_summary(
            telemetry_session,
            run_request,
            native_plan::MultiPhenotypeSampleMode::PerPhenotype,
            &grouped_run_inputs.iter().map(|group| group.compute_group.clone()).collect::<Vec<_>>(),
            &grouped_run_inputs.iter().map(|group| group.sample_count).collect::<Vec<_>>(),
        )?;
        for grouped_run_input in &grouped_run_inputs {
            self.run_multi_group_preflight(
                telemetry_session,
                stage_timing_recorder,
                run_request,
                resolved_execution,
                &grouped_run_input.aligned_sample_data,
                &grouped_run_input.prediction_source,
            )?;
        }
        let output_bundles = self.prepare_grouped_output_bundles(
            py,
            stage_timing_recorder,
            run_request,
            resolved_execution,
            &grouped_run_inputs,
        )?;
        let mut writer_abort_guard = OutputWriterAbortGuard {
            writer_sessions: output_bundles
                .iter()
                .flat_map(NativePreparedOutputBundle::writer_session_handles)
                .collect(),
            armed: true,
        };
        if should_use_union_grouped_bgen_delivery(resolved_execution, &grouped_run_inputs) {
            let sample_indices_by_group =
                grouped_run_inputs.iter().map(|group| group.sample_indices.clone()).collect::<Vec<_>>();
            let union_sample_indices = native_input::build_union_sample_indices(&sample_indices_by_group);
            let grouped_sample_count = sample_indices_by_group.iter().map(Vec::len).sum::<usize>();
            run_events::record_pipeline_grouped_union_delivery_selected_diagnostic_event(
                usize_to_i64(grouped_sample_count, "Grouped sample count")?,
                usize_to_i64(grouped_run_inputs.len(), "Phenotype group count")?,
                usize_to_i64(union_sample_indices.len(), "Union sample count")?,
            )?;
            let mut group_requests = Vec::with_capacity(grouped_run_inputs.len());
            let mut all_writer_sessions = Vec::new();
            let mut phenotype_indices_by_group = Vec::with_capacity(grouped_run_inputs.len());
            for (grouped_run_input, output_bundle) in grouped_run_inputs.into_iter().zip(output_bundles) {
                let OutputBundleDeliveryState { writer_sessions, committed_chunk_identifier_sets } =
                    output_bundle_delivery_state(&output_bundle)?;
                let settings = association_delivery_settings(
                    run_request,
                    resolved_execution,
                    &writer_sessions,
                    committed_chunk_identifier_sets,
                    grouped_run_input.sample_indices,
                )?;
                all_writer_sessions.extend(writer_sessions);
                phenotype_indices_by_group.push(grouped_run_input.phenotype_indices);
                group_requests.push(MultiAssociationDeliveryRequest {
                    aligned_sample_data: grouped_run_input.aligned_sample_data,
                    prediction_source: grouped_run_input.prediction_source,
                    settings,
                });
            }
            let final_paths = self.run_native_grouped_union_delivery(
                py,
                backend,
                GroupedUnionAssociationDeliveryRequest { groups: group_requests, union_sample_indices },
                &all_writer_sessions,
                stage_timing_recorder,
                i64::from(run_request.output.writer_thread_count),
            )?;
            let mut final_paths_by_index = vec![None; phenotype_names.len()];
            let mut path_offset = 0_usize;
            for phenotype_indices in phenotype_indices_by_group {
                let path_stop = path_offset
                    .checked_add(phenotype_indices.len())
                    .ok_or_else(|| PyValueError::new_err("Grouped output path offset overflowed usize."))?;
                let group_paths = final_paths.get(path_offset..path_stop).ok_or_else(|| {
                    PyValueError::new_err("Grouped output path count does not match phenotype groups.")
                })?;
                scatter_group_final_paths(&mut final_paths_by_index, &phenotype_indices, group_paths)?;
                path_offset = path_stop;
            }
            if path_offset != final_paths.len() {
                return Err(PyValueError::new_err("Grouped output path count exceeds the planned phenotype groups."));
            }
            writer_abort_guard.armed = false;
            return Ok(final_paths_by_index);
        }
        let mut final_paths_by_index = vec![None; phenotype_names.len()];
        for (grouped_run_input, output_bundle) in grouped_run_inputs.into_iter().zip(output_bundles) {
            let phenotype_indices = grouped_run_input.phenotype_indices.clone();
            let OutputBundleDeliveryState { writer_sessions, committed_chunk_identifier_sets } =
                output_bundle_delivery_state(&output_bundle)?;
            let settings = association_delivery_settings(
                run_request,
                resolved_execution,
                &writer_sessions,
                committed_chunk_identifier_sets,
                grouped_run_input.sample_indices.clone(),
            )?;
            let request = AssociationDeliveryRequest::Multi(MultiAssociationDeliveryRequest {
                aligned_sample_data: grouped_run_input.aligned_sample_data,
                prediction_source: grouped_run_input.prediction_source,
                settings,
            });
            let group_paths = self.run_native_association_delivery(
                py,
                Arc::clone(&backend),
                request,
                &writer_sessions,
                stage_timing_recorder,
                "Multi-phenotype native BGEN",
                i64::from(run_request.output.writer_thread_count),
            )?;
            scatter_group_final_paths(&mut final_paths_by_index, &phenotype_indices, &group_paths)?;
        }
        writer_abort_guard.armed = false;
        Ok(final_paths_by_index)
    }
}

impl NativeRunEngineSession {
    #[allow(clippy::too_many_arguments)]
    fn run_native_association_delivery(
        &self,
        py: Python<'_>,
        backend: Arc<PyJaxBackend>,
        request: AssociationDeliveryRequest,
        writer_sessions: &[Arc<OutputWriterSession>],
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        pipeline_label: &str,
        writer_finish_thread_count: i64,
    ) -> PyResult<Vec<Option<String>>> {
        let delivery_start_time = Instant::now();
        let engine_guard = self.lock_engine()?;
        let engine = engine_guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Native run engine session has no open BGEN engine."))?;
        let delivery_result = backend_delivery::run_association_delivery(py, engine, backend, request).and_then(
            |processed_chunk_count| {
                crate::binding::runtime::check_process_signals(py)?;
                Ok(processed_chunk_count)
            },
        );
        drop(engine_guard);
        match delivery_result {
            Ok(processed_chunk_count) => {
                if let Err(error) =
                    record_stage_duration(stage_timing_recorder, "native_engine_delivery", delivery_start_time)
                {
                    output::abort_output_writer_sessions_for_delivery(writer_sessions);
                    return Err(error);
                }
                let final_parquet_paths =
                    output::finish_output_writer_sessions_for_delivery(writer_sessions, writer_finish_thread_count)?;
                run_events::record_native_dispatch_pipeline_finished_diagnostic_event(
                    usize_to_i64(final_parquet_paths.len(), "Final Parquet path count")?,
                    pipeline_label,
                )?;
                run_events::record_native_dispatch_delivery_finished_diagnostic_event(
                    pipeline_label,
                    usize_to_i64(processed_chunk_count, "Processed chunk count")?,
                )?;
                Ok(final_parquet_paths)
            }
            Err(error) => {
                if let Some(interrupted_event) = maybe_shutdown_event_from_error(py, &error)? {
                    output::finish_interrupted_output_writer_sessions_for_delivery(
                        writer_sessions,
                        writer_finish_thread_count,
                        interrupted_event.exit_code,
                        &interrupted_event.signal_name,
                        interrupted_event.signal_number,
                    )?;
                    if error.is_instance_of::<PyKeyboardInterrupt>(py) {
                        return Err(crate::binding::runtime::flushed_interrupt_error());
                    }
                } else {
                    output::abort_output_writer_sessions_for_delivery(writer_sessions);
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_native_grouped_union_delivery(
        &self,
        py: Python<'_>,
        backend: Arc<PyJaxBackend>,
        request: GroupedUnionAssociationDeliveryRequest,
        writer_sessions: &[Arc<OutputWriterSession>],
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        writer_finish_thread_count: i64,
    ) -> PyResult<Vec<Option<String>>> {
        const PIPELINE_LABEL: &str = "Grouped per-phenotype union native BGEN";
        let delivery_start_time = Instant::now();
        let engine_guard = self.lock_engine()?;
        let engine = engine_guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Native run engine session has no open BGEN engine."))?;
        let delivery_result = backend_delivery::run_grouped_union_association_delivery(py, engine, backend, request)
            .and_then(|processed_chunk_count| {
                crate::binding::runtime::check_process_signals(py)?;
                Ok(processed_chunk_count)
            });
        drop(engine_guard);
        match delivery_result {
            Ok(processed_chunk_count) => {
                if let Err(error) =
                    record_stage_duration(stage_timing_recorder, "native_engine_delivery", delivery_start_time)
                {
                    output::abort_output_writer_sessions_for_delivery(writer_sessions);
                    return Err(error);
                }
                let final_parquet_paths =
                    output::finish_output_writer_sessions_for_delivery(writer_sessions, writer_finish_thread_count)?;
                run_events::record_native_dispatch_pipeline_finished_diagnostic_event(
                    usize_to_i64(final_parquet_paths.len(), "Final Parquet path count")?,
                    PIPELINE_LABEL,
                )?;
                run_events::record_native_dispatch_delivery_finished_diagnostic_event(
                    PIPELINE_LABEL,
                    usize_to_i64(processed_chunk_count, "Processed chunk count")?,
                )?;
                Ok(final_parquet_paths)
            }
            Err(error) => {
                if let Some(interrupted_event) = maybe_shutdown_event_from_error(py, &error)? {
                    output::finish_interrupted_output_writer_sessions_for_delivery(
                        writer_sessions,
                        writer_finish_thread_count,
                        interrupted_event.exit_code,
                        &interrupted_event.signal_name,
                        interrupted_event.signal_number,
                    )?;
                    if error.is_instance_of::<PyKeyboardInterrupt>(py) {
                        return Err(crate::binding::runtime::flushed_interrupt_error());
                    }
                } else {
                    output::abort_output_writer_sessions_for_delivery(writer_sessions);
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn open_pipeline_bgen_engine_with_events(
        &self,
        py: Python<'_>,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
        pipeline_label: &str,
        phenotype_name: Option<&str>,
        phenotype_count: Option<i64>,
    ) -> PyResult<()> {
        let engine_start_time = Instant::now();
        run_events::record_pipeline_bgen_engine_open_started_diagnostic_event(
            phenotype_count,
            phenotype_name,
            pipeline_label,
            resolved_execution.effective_trusted_no_missing_diploid,
            run_request.compute.variant_limit.map(i64::from),
        )?;
        run_events::record_association_backend_selected_telemetry(
            telemetry_session,
            run_request.association_mode.as_str(),
            resolved_execution.backend_plan.kind.as_str(),
            resolved_execution.backend_plan.device.as_str(),
            resolved_execution.backend_plan.resolved_genotype_format.as_str(),
            phenotype_name.map(str::to_string),
            phenotype_count,
        )?;
        self.open_bgen_engine_internal(
            py,
            &run_request.input.bgen_path,
            u32_value_as_usize(run_request.trait_request.chunk_size, "trait chunk size")?,
            run_request.compute.variant_limit.map(|value| u32_value_as_usize(value, "variant limit")).transpose()?,
            resolved_execution.effective_trusted_no_missing_diploid,
            Some(run_request.compute.trusted_bgen_validation_mode.as_str()),
        )?;
        record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)?;
        let (sample_count, variant_count) = self.with_open_engine(|engine| {
            Ok((
                usize_to_i64(engine.reader().sample_count(), "BGEN sample count")?,
                usize_to_i64(engine.reader().variant_count(), "BGEN variant count")?,
            ))
        })?;
        run_events::record_pipeline_bgen_engine_opened_diagnostic_event(
            phenotype_count,
            phenotype_name,
            pipeline_label,
            sample_count,
            variant_count,
        )?;
        run_events::record_bgen_engine_opened_telemetry(
            telemetry_session,
            run_request.association_mode.as_str(),
            resolved_execution.backend_plan.kind.as_str(),
            sample_count,
            variant_count,
            phenotype_name.map(str::to_string),
            phenotype_count,
        )
    }

    fn record_multi_phenotype_sample_summary(
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        run_request: &native_plan::RunRequest,
        sample_mode: native_plan::MultiPhenotypeSampleMode,
        compute_groups: &[native_input::ResolvedPhenotypeComputeGroup],
        sample_counts_by_group: &[i64],
    ) -> PyResult<()> {
        let mut sample_counts = Vec::new();
        let mut sample_set_fingerprints = Vec::new();
        for (compute_group, sample_count) in compute_groups.iter().zip(sample_counts_by_group) {
            for _ in &compute_group.phenotype_names {
                sample_counts.push(*sample_count);
                sample_set_fingerprints.push(Some(compute_group.sample_set_fingerprint.clone()));
            }
        }
        let sample_counts_differ = sample_counts.iter().any(|sample_count| Some(sample_count) != sample_counts.first());
        run_events::record_pipeline_multi_phenotype_sample_summary_diagnostic_event(
            usize_to_i64(sample_counts.len(), "Phenotype count")?,
            usize_to_i64(compute_groups.len(), "Phenotype group count")?,
            sample_counts_differ,
            sample_mode.as_str(),
        )?;
        run_events::record_multi_phenotype_sample_summary_telemetry(
            telemetry_session,
            run_request.association_mode.as_str(),
            sample_mode.as_str(),
            sample_counts,
            sample_set_fingerprints,
            usize_to_i64(compute_groups.len(), "Phenotype group count")?,
        )
    }

    fn run_multi_group_preflight(
        &self,
        telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
        aligned_sample_data: &native_input::MultiAlignedSampleData,
        prediction_source: &native_input::MultiPredictionSource,
    ) -> PyResult<()> {
        let phenotype_count = usize_to_i64(aligned_sample_data.phenotype_names.len(), "Phenotype count")?;
        let sample_count = usize_to_i64(aligned_sample_data.sample_indices.len(), "Aligned sample count")?;
        run_events::record_prediction_source_loaded_telemetry(
            telemetry_session,
            run_request.association_mode.as_str(),
            None,
            Some(phenotype_count),
        )?;
        let preflight_start_time = Instant::now();
        run_events::record_pipeline_multi_group_preflight_started_diagnostic_event(
            phenotype_count,
            sample_count,
            resolved_execution.effective_trusted_no_missing_diploid,
            run_request.compute.variant_limit.map(i64::from),
        )?;
        let preflight_shape = native_engine_debug::validate_multi_trait_preflight_values(
            aligned_sample_data.phenotype_row_count,
            aligned_sample_data.phenotype_column_count,
            &aligned_sample_data.phenotype_matrix_values,
            aligned_sample_data.covariate_row_count,
            aligned_sample_data.covariate_column_count,
            &aligned_sample_data.covariate_matrix_values,
            aligned_sample_data.is_binary_trait,
        )
        .map_err(|error| convert_preflight_error(&error))?;
        let required_chromosomes = self.with_open_engine(|engine| {
            engine
                .required_chromosomes(
                    run_request
                        .compute
                        .variant_limit
                        .map(|value| u32_value_as_usize(value, "variant limit"))
                        .transpose()?,
                )
                .map_err(|error| convert_preflight_error(&error))
        })?;
        for chromosome in &required_chromosomes {
            let prediction_matrix = prediction_source
                .chromosome_prediction_matrix(chromosome)
                .map_err(|error| convert_prediction_error("chromosome_prediction_matrix", &error))?;
            native_engine_debug::validate_multi_prediction_values(
                chromosome,
                &prediction_matrix.prediction_values,
                preflight_shape.trait_count,
                preflight_shape.sample_count,
            )
            .map_err(|error| convert_preflight_error(&error))?;
        }
        let chromosome_count = usize_to_i64(required_chromosomes.len(), "Chromosome count")?;
        let preflight_report = native_engine_debug::build_preflight_report_payload(
            preflight_shape.sample_count,
            preflight_shape.covariate_count,
            chromosome_count,
            resolved_execution.effective_trusted_no_missing_diploid,
        )
        .map_err(|error| convert_preflight_error(&error))?;
        run_events::record_preflight_warning_diagnostic_events(
            preflight_report.warning_messages.clone(),
            preflight_report.chromosome_count,
            preflight_report.covariate_count,
            "multi_trait",
            preflight_report.sample_count,
            resolved_execution.effective_trusted_no_missing_diploid,
        )?;
        record_stage_duration(stage_timing_recorder, "preflight_validation", preflight_start_time)?;
        run_events::record_pipeline_multi_group_preflight_completed_diagnostic_event(
            phenotype_count,
            sample_count,
            resolved_execution.effective_trusted_no_missing_diploid,
            run_request.compute.variant_limit.map(i64::from),
        )?;
        run_events::record_multi_phenotype_preflight_completed_telemetry(
            telemetry_session,
            run_request.association_mode.as_str(),
            phenotype_count,
            sample_count,
        )
    }

    fn prepare_multi_trait_output_bundle(
        &self,
        py: Python<'_>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
        aligned_sample_data: &native_input::MultiAlignedSampleData,
        compute_group: &native_input::ResolvedPhenotypeComputeGroup,
        output_sample_mode: native_plan::MultiPhenotypeSampleMode,
    ) -> PyResult<NativePreparedOutputBundle> {
        let engine_variant_count =
            self.with_open_engine(|engine| usize_to_i64(engine.reader().variant_count(), "BGEN variant count"))?;
        let sample_count = usize_to_i64(aligned_sample_data.sample_indices.len(), "Aligned sample count")?;
        let output_group = build_multi_trait_output_group(
            &aligned_sample_data.covariate_names,
            sample_count,
            output_sample_mode,
            compute_group,
        )?;
        let mut output_bundles = self.lifecycle.prepare_output_bundles_from_runtime_plan_internal(
            py,
            vec![output_group],
            engine_variant_count,
            resolved_execution.effective_trusted_no_missing_diploid,
            run_request.input.sample_key_mode.as_str().to_string(),
            resolved_execution.binary_kernel_config_json.clone(),
            resolved_execution.requested_gpu_genotype_format.clone(),
            resolved_execution.resolved_gpu_genotype_format.clone(),
            run_request.compute.score_dtype.as_str().to_string(),
            run_request.compute.firth_dtype.as_str().to_string(),
            stage_timing_recorder,
        )?;
        output_bundles
            .pop()
            .ok_or_else(|| PyRuntimeError::new_err("Multi-trait output preparation returned no bundle."))
    }

    fn build_grouped_run_inputs(
        run_request: &native_plan::RunRequest,
        grouped_aligned_sample_data: Vec<native_input::AlignedPhenotypeGroup>,
        prediction_sources: Vec<native_input::MultiPredictionSource>,
    ) -> PyResult<Vec<NativeGroupedRunInputState>> {
        let planned_names_by_index = planned_phenotype_names_by_index(run_request)?;
        let parsed_sample_key_mode = parse_sample_key_mode(run_request.input.sample_key_mode.as_str())?;
        grouped_aligned_sample_data
            .into_iter()
            .zip(prediction_sources)
            .map(|(group, prediction_source)| {
                let phenotype_indices = group.phenotype_indices.clone();
                let group_phenotype_names = phenotype_indices
                    .iter()
                    .map(|phenotype_index| {
                        planned_names_by_index.get(phenotype_index).cloned().ok_or_else(|| {
                            PyValueError::new_err(format!("No planned phenotype name for index {phenotype_index}."))
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let compute_group = native_input::resolve_per_phenotype_compute_group(
                    &group.aligned_sample_data,
                    phenotype_indices.clone(),
                    group_phenotype_names,
                    Some(run_request.input.prediction_list_path.as_str()),
                    parsed_sample_key_mode,
                );
                let sample_count =
                    usize_to_i64(group.aligned_sample_data.sample_indices.len(), "Aligned sample count")?;
                let sample_indices = group.aligned_sample_data.sample_indices.clone();
                Ok(NativeGroupedRunInputState {
                    compute_group,
                    phenotype_indices,
                    aligned_sample_data: group.aligned_sample_data,
                    prediction_source,
                    sample_indices,
                    sample_count,
                })
            })
            .collect()
    }

    fn prepare_grouped_output_bundles(
        &self,
        py: Python<'_>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
        grouped_run_inputs: &[NativeGroupedRunInputState],
    ) -> PyResult<Vec<NativePreparedOutputBundle>> {
        let engine_variant_count =
            self.with_open_engine(|engine| usize_to_i64(engine.reader().variant_count(), "BGEN variant count"))?;
        let mut output_groups = Vec::with_capacity(grouped_run_inputs.len());
        for grouped_run_input in grouped_run_inputs {
            output_groups.push(build_multi_trait_output_group(
                &grouped_run_input.aligned_sample_data.covariate_names,
                grouped_run_input.sample_count,
                native_plan::MultiPhenotypeSampleMode::PerPhenotype,
                &grouped_run_input.compute_group,
            )?);
        }
        self.lifecycle.prepare_output_bundles_from_runtime_plan_internal(
            py,
            output_groups,
            engine_variant_count,
            resolved_execution.effective_trusted_no_missing_diploid,
            run_request.input.sample_key_mode.as_str().to_string(),
            resolved_execution.binary_kernel_config_json.clone(),
            resolved_execution.requested_gpu_genotype_format.clone(),
            resolved_execution.resolved_gpu_genotype_format.clone(),
            run_request.compute.score_dtype.as_str().to_string(),
            run_request.compute.firth_dtype.as_str().to_string(),
            stage_timing_recorder,
        )
    }
}

fn record_stage_duration(
    stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    stage_name: &str,
    start_time: Instant,
) -> PyResult<()> {
    if let Some(stage_timing_recorder) = stage_timing_recorder {
        stage_timing_recorder.record_stage_duration(stage_name, start_time.elapsed().as_secs_f64())?;
    }
    Ok(())
}

fn maybe_shutdown_event_from_error(
    py: Python<'_>,
    error: &PyErr,
) -> PyResult<Option<native_run_events::RunInterruptedEventPayload>> {
    if crate::binding::runtime::is_sigterm_request(error, py) {
        let signal = native_run_events::build_shutdown_signal(15)
            .map_err(|signal_error| PyValueError::new_err(signal_error.to_string()))?;
        return Ok(Some(native_run_events::build_run_interrupted_event_payload(
            i64::from(signal.number),
            &signal.name,
            i64::from(signal.exit_code),
            true,
        )));
    }
    if error.is_instance_of::<PyKeyboardInterrupt>(py) {
        let signal = native_run_events::build_shutdown_signal(2)
            .map_err(|signal_error| PyValueError::new_err(signal_error.to_string()))?;
        return Ok(Some(native_run_events::build_run_interrupted_event_payload(
            i64::from(signal.number),
            &signal.name,
            i64::from(signal.exit_code),
            false,
        )));
    }
    Ok(None)
}

fn build_single_trait_output_group(
    phenotype_name: &str,
    aligned_sample_data: &native_input::AlignedSampleData,
    phenotype_compute_group: &native_input::ResolvedPhenotypeComputeGroup,
    sample_count: i64,
) -> PyResult<g_engine::RuntimeOutputGroupInput> {
    let phenotype_indices = phenotype_compute_group
        .phenotype_indices
        .iter()
        .copied()
        .map(|index| usize_to_i64(index, "Phenotype compute group index"))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(g_engine::RuntimeOutputGroupInput {
        phenotype_names: vec![phenotype_name.to_string()],
        covariate_names: aligned_sample_data.covariate_names.clone(),
        sample_count,
        output_sample_mode: "single-phenotype".to_string(),
        phenotype_compute_group_mode: Some(phenotype_compute_group.group_mode.clone()),
        phenotype_compute_group_indices: Some(phenotype_indices),
        phenotype_compute_group_names: Some(phenotype_compute_group.phenotype_names.clone()),
        phenotype_compute_group_sample_mode: Some(phenotype_compute_group.sample_mode.clone()),
        sample_set_fingerprint: Some(phenotype_compute_group.sample_set_fingerprint.clone()),
        covariate_design_fingerprint: Some(phenotype_compute_group.covariate_design_fingerprint.clone()),
        prediction_alignment_fingerprint: phenotype_compute_group.prediction_alignment_fingerprint.clone(),
    })
}

fn build_multi_trait_output_group(
    covariate_names: &[String],
    sample_count: i64,
    output_sample_mode: native_plan::MultiPhenotypeSampleMode,
    phenotype_compute_group: &native_input::ResolvedPhenotypeComputeGroup,
) -> PyResult<g_engine::RuntimeOutputGroupInput> {
    let phenotype_indices = phenotype_compute_group
        .phenotype_indices
        .iter()
        .copied()
        .map(|index| usize_to_i64(index, "Phenotype compute group index"))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(g_engine::RuntimeOutputGroupInput {
        phenotype_names: phenotype_compute_group.phenotype_names.clone(),
        covariate_names: covariate_names.to_vec(),
        sample_count,
        output_sample_mode: output_sample_mode.as_str().to_string(),
        phenotype_compute_group_mode: Some(phenotype_compute_group.group_mode.clone()),
        phenotype_compute_group_indices: Some(phenotype_indices),
        phenotype_compute_group_names: Some(phenotype_compute_group.phenotype_names.clone()),
        phenotype_compute_group_sample_mode: Some(phenotype_compute_group.sample_mode.clone()),
        sample_set_fingerprint: Some(phenotype_compute_group.sample_set_fingerprint.clone()),
        covariate_design_fingerprint: Some(phenotype_compute_group.covariate_design_fingerprint.clone()),
        prediction_alignment_fingerprint: phenotype_compute_group.prediction_alignment_fingerprint.clone(),
    })
}

fn output_bundle_delivery_state(output_bundle: &NativePreparedOutputBundle) -> PyResult<OutputBundleDeliveryState> {
    let writer_sessions = output_bundle.writer_session_handles();
    let committed_chunk_identifier_sets = (0..writer_sessions.len())
        .map(|output_index| {
            output_bundle
                .committed_chunk_identifiers_usize(output_index)
                .map(|identifiers| identifiers.into_iter().collect())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(OutputBundleDeliveryState { writer_sessions, committed_chunk_identifier_sets })
}

fn association_delivery_settings(
    run_request: &native_plan::RunRequest,
    resolved_execution: &NativeRunResolvedExecution,
    writer_sessions: &[Arc<OutputWriterSession>],
    committed_chunk_identifier_sets: Vec<BTreeSet<usize>>,
    sample_indices: Vec<usize>,
) -> PyResult<AssociationDeliverySettings> {
    let staging_depth = u32_value_as_usize(run_request.compute.staging_depth, "association staging depth")?;
    let result_in_flight_limit = run_request
        .compute
        .result_in_flight_limit
        .map(|value| u32_value_as_usize(value, "association result in-flight limit"))
        .transpose()?
        .map_or_else(
            || {
                staging_depth
                    .checked_add(1)
                    .ok_or_else(|| PyValueError::new_err("Association result in-flight default overflowed usize."))
            },
            Ok,
        )?;
    Ok(AssociationDeliverySettings {
        writer_sessions: writer_sessions.iter().map(Arc::clone).collect(),
        committed_chunk_identifier_sets,
        null_logistic_nonconvergence_policy: resolved_execution.null_logistic_nonconvergence_policy.clone(),
        staging_depth,
        result_in_flight_limit,
        output_statistic_dtype: run_request.output.output_statistic_dtype,
        sample_indices,
        use_packed8: resolved_execution.backend_plan.resolved_genotype_format
            == native_plan::GpuGenotypeFormat::Packed8,
    })
}

fn concrete_gpu_genotype_format_from_resolution_plan(
    native_resolution_plan: &native_engine_debug::GpuGenotypeFormatResolutionPlan,
) -> PyResult<native_plan::GpuGenotypeFormat> {
    let resolved_gpu_genotype_format = native_resolution_plan
        .resolved_gpu_genotype_format
        .as_deref()
        .ok_or_else(|| PyRuntimeError::new_err("Native GPU genotype-format resolution plan is not resolved."))?;
    native_plan::GpuGenotypeFormat::from_str_value(resolved_gpu_genotype_format).ok_or_else(|| {
        PyValueError::new_err(format!("Unsupported resolved GPU genotype format '{resolved_gpu_genotype_format}'."))
    })
}

fn manifest_gpu_genotype_format_fields(
    existing_manifest_json: Option<&str>,
) -> PyResult<(Option<String>, Option<String>)> {
    let Some(existing_manifest_json) = existing_manifest_json else {
        return Ok((None, None));
    };
    let manifest_value = serde_json::from_str::<serde_json::Value>(existing_manifest_json)
        .map_err(|error| PyValueError::new_err(format!("Existing run manifest JSON is invalid: {error}")))?;
    let manifest_gpu_genotype_format =
        manifest_value.get("gpu_genotype_format").and_then(serde_json::Value::as_str).map(str::to_string);
    let association_backend_genotype_format = if manifest_gpu_genotype_format.is_none() {
        manifest_value
            .get("association_backend")
            .and_then(|association_backend| association_backend.get("genotype_format"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    } else {
        None
    };
    Ok((manifest_gpu_genotype_format, association_backend_genotype_format))
}

fn complete_case_compute_group_from_request(
    run_request: &native_plan::RunRequest,
) -> PyResult<native_plan::PhenotypeComputeGroup> {
    run_request
        .phenotype_compute_groups
        .iter()
        .find(|group| group.group_mode == native_plan::PhenotypeComputeGroupMode::CompleteCase)
        .cloned()
        .ok_or_else(|| PyValueError::new_err("A complete-case phenotype compute group is required."))
}

fn phenotype_names_from_request(run_request: &native_plan::RunRequest) -> Vec<String> {
    run_request.phenotype_runs.iter().map(|phenotype_run| phenotype_run.phenotype_name.clone()).collect()
}

fn planned_phenotype_names_by_index(run_request: &native_plan::RunRequest) -> PyResult<BTreeMap<usize, String>> {
    run_request
        .phenotype_runs
        .iter()
        .map(|phenotype_run| {
            Ok((
                u32_value_as_usize(phenotype_run.phenotype_index, "phenotype index")?,
                phenotype_run.phenotype_name.clone(),
            ))
        })
        .collect()
}

fn u32_indices_to_usize(values: &[u32], value_name: &str) -> PyResult<Vec<usize>> {
    values.iter().copied().map(|value| u32_value_as_usize(value, value_name)).collect()
}

fn u32_value_as_usize(value: u32, field_name: &str) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| PyValueError::new_err(format!("{field_name} does not fit into usize.")))
}

fn should_use_union_grouped_bgen_delivery(
    resolved_execution: &NativeRunResolvedExecution,
    grouped_run_inputs: &[NativeGroupedRunInputState],
) -> bool {
    if grouped_run_inputs.len() <= 1 {
        return false;
    }
    if resolved_execution.backend_plan.resolved_genotype_format == native_plan::GpuGenotypeFormat::Packed8 {
        return false;
    }
    if !resolved_execution.effective_trusted_no_missing_diploid {
        return false;
    }
    let sample_indices_by_group =
        grouped_run_inputs.iter().map(|grouped_run_input| grouped_run_input.sample_indices.clone()).collect::<Vec<_>>();
    let union_sample_count = native_input::build_union_sample_indices(&sample_indices_by_group).len();
    let grouped_sample_count =
        grouped_run_inputs.iter().map(|grouped_run_input| grouped_run_input.sample_indices.len()).sum();
    union_sample_count < grouped_sample_count
}

fn scatter_group_final_paths(
    final_paths_by_index: &mut [Option<String>],
    phenotype_indices: &[usize],
    group_paths: &[Option<String>],
) -> PyResult<()> {
    if phenotype_indices.len() != group_paths.len() {
        return Err(PyValueError::new_err("Grouped final output path count does not match phenotype index count."));
    }
    for (phenotype_index, final_output_path) in phenotype_indices.iter().copied().zip(group_paths.iter().cloned()) {
        let target_path = final_paths_by_index.get_mut(phenotype_index).ok_or_else(|| {
            PyValueError::new_err(format!("Phenotype output index {phenotype_index} is outside the run."))
        })?;
        *target_path = final_output_path;
    }
    Ok(())
}

fn record_native_dispatch_bgen_engine_constructing(
    chunk_size: usize,
    source_path: &str,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<usize>,
) -> PyResult<()> {
    let chunk_size_value = i64::try_from(chunk_size)
        .map_err(|_| PyValueError::new_err("BGEN chunk size exceeds native int64 capacity."))?;
    let variant_limit_value = variant_limit
        .map(|value| {
            i64::try_from(value).map_err(|_| PyValueError::new_err("BGEN variant limit exceeds native int64 capacity."))
        })
        .transpose()?;
    let payload = native_run_events::build_native_dispatch_bgen_engine_constructing_diagnostic_payload(
        chunk_size_value,
        source_path,
        trusted_no_missing_diploid,
        variant_limit_value,
    );
    run_events::emit_run_diagnostic_event_payload(&payload)
}

fn record_native_dispatch_trusted_bgen_validation_started(
    source_path: &str,
    trusted_bgen_validation_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_trusted_bgen_validation_started_diagnostic_payload(
        source_path,
        trusted_bgen_validation_mode,
    );
    run_events::emit_run_diagnostic_event_payload(&payload)
}

fn open_bgen_engine_core(
    py: Python<'_>,
    bgen_path: &str,
    chunk_size: usize,
    variant_limit: Option<usize>,
    trusted_no_missing_diploid: bool,
) -> PyResult<Regenie2RunEngineCore> {
    record_native_dispatch_bgen_engine_constructing(chunk_size, bgen_path, trusted_no_missing_diploid, variant_limit)?;
    py.detach(|| {
        Regenie2RunEngineCore::open_bgen(Path::new(bgen_path), chunk_size, variant_limit, trusted_no_missing_diploid)
    })
    .map_err(|error| convert_bgen_error("open_bgen", error))
}

fn validate_trusted_no_missing_diploid_with_default_cache_for_engine(
    engine: &Regenie2RunEngineCore,
    py: Python<'_>,
    bgen_path: &str,
    validation_mode: &str,
) -> PyResult<()> {
    record_native_dispatch_trusted_bgen_validation_started(bgen_path, validation_mode)?;
    let cache_directory = py
        .detach(native_trusted_validation::default_trusted_bgen_validation_cache_directory)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    py.detach(|| {
        engine.validate_trusted_no_missing_diploid_with_cache_directory(
            Path::new(bgen_path),
            validation_mode,
            cache_directory.as_path(),
        )
    })
    .map_err(convert_trusted_bgen_validation_error)
}
