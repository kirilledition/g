#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::fn_params_excessive_bools)]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use crate::binding::convert::int::{
    optional_usize_to_py_i64 as option_usize_to_i64, py_i64_slice_to_usize, usize_slice_to_py_i64,
    usize_to_py_i64 as usize_to_i64,
};
use g_engine as native_engine_debug;
use g_engine::Regenie2RunEngineCore;
use g_genotype::{ChunkSpec as NativeChunkSpec, OutputBufferAddress, OutputValueCount};
use g_input::{self as native_input, AlignmentInputs, MultiAlignmentInputs};
use g_plan as native_plan;
use g_runtime as native_trusted_validation;
use g_runtime as native_run_events;
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadwriteArray2, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use super::config::{NativeRunRequest, RegenieConfig};
use super::errors::{
    convert_bgen_error, convert_genotype_error, convert_input_error, convert_prediction_error, convert_preflight_error,
    convert_schedule_error, convert_trusted_bgen_validation_error,
};
use super::genotype::{
    ChunkStats, VariantMetadata, VariantMetadataTuple, build_committed_identifier_set,
    convert_variant_metadata_columns_to_tuple,
};
use super::output::{self, OutputWriterSession};
use super::prediction_sources::{self, RegeniePredictionSource};
use super::profile::build_profile_snapshot_dict;
use crate::binding::telemetry::run_events::{self, NativeRunArtifacts};
use super::run_lifecycle::{
    NativeOutputRuntimeGroupInput, NativePreparedOutputBundle, NativeRunLifecyclePhenotypeRun,
    NativeRunLifecycleSession,
};
use super::runtime_state::NativeRuntimeCompatibilityToken;
use super::sample_alignment::{
    NativeAlignedSampleData, NativeGroupedAlignedSampleData, NativeMultiAlignedSampleData,
    NativeResolvedPhenotypeComputeGroup, parse_sample_key_mode,
};
use super::timing::NativeStageTimingRecorder;

#[pyclass(name = "NativeSingleTraitRunInput", skip_from_py_object)]
struct NativeSingleTraitRunInput {
    data: native_input::AlignedSampleData,
}

#[pyclass(name = "NativeMultiTraitRunInput", skip_from_py_object)]
struct NativeMultiTraitRunInput {
    data: native_input::MultiAlignedSampleData,
}

#[pyclass(name = "NativeRunCallbackContext", skip_from_py_object)]
#[derive(Clone)]
struct NativeRunCallbackContext {
    association_mode: String,
    trait_type: String,
    correction_method: String,
    correction_p_threshold: f64,
    correction_firth_se: bool,
    staging_depth: i64,
    native_callback_batch_size: i64,
    result_in_flight_limit: Option<i64>,
    dosage_buffer_limit: Option<i64>,
    score_dtype: String,
    firth_dtype: String,
    output_statistic_dtype: String,
    jax_device: String,
    gpu_genotype_format: String,
    requested_gpu_genotype_format: String,
}

#[pyclass(name = "NativeSingleTraitPipelineBundle", skip_from_py_object)]
struct NativeSingleTraitPipelineBundle {
    run_input: Py<NativeSingleTraitRunInput>,
    prediction_source: Py<RegeniePredictionSource>,
    phenotype_compute_group: NativeResolvedPhenotypeComputeGroup,
    output_bundle: Py<NativePreparedOutputBundle>,
    writer_session: Py<OutputWriterSession>,
    committed_chunk_identifiers: Vec<usize>,
}

#[pyclass(name = "NativeRunEngineSession", skip_from_py_object)]
struct NativeRunEngineSession {
    lifecycle: NativeRunLifecycleSession,
    engine: Mutex<Option<Regenie2RunEngineCore>>,
}

struct BgenDeliveryCleanupExecution {
    final_parquet_paths: Vec<Option<String>>,
    callback_finished: bool,
}

struct NativeRunResolvedExecution {
    backend_plan: native_plan::AssociationBackendPlan,
    requested_gpu_genotype_format: String,
    resolved_gpu_genotype_format: String,
    effective_trusted_no_missing_diploid: bool,
    binary_kernel_config_json: Option<String>,
}

struct NativeGroupedRunInputState {
    compute_group: native_input::ResolvedPhenotypeComputeGroup,
    phenotype_indices: Vec<usize>,
    run_input: Py<NativeMultiTraitRunInput>,
    aligned_sample_data: Py<NativeMultiAlignedSampleData>,
    prediction_source: Py<prediction_sources::MultiRegeniePredictionSource>,
    sample_indices: Vec<usize>,
    sample_count: i64,
}

struct NativePreparedMultiGroupDelivery {
    phenotype_indices: Vec<usize>,
    aligned_sample_data: Py<NativeMultiAlignedSampleData>,
    callback: Py<PyAny>,
    writer_sessions: Vec<Py<OutputWriterSession>>,
    output_bundle: Py<NativePreparedOutputBundle>,
    sample_indices: Vec<usize>,
}

#[pymethods]
impl NativeSingleTraitRunInput {
    #[getter]
    fn native_aligned_sample_data(&self) -> NativeAlignedSampleData {
        NativeAlignedSampleData::new(self.data.clone())
    }

    #[getter]
    #[allow(clippy::unused_self)]
    fn native_multi_aligned_sample_data(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn sample_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(usize_slice_to_py_i64(&self.data.sample_indices, "sample_indices")?.into_pyarray(py))
    }

    #[getter]
    fn phenotype_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.data.phenotype_vector.clone().into_pyarray(py)
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

    #[getter]
    fn family_identifiers(&self) -> Vec<String> {
        self.data.family_identifiers.clone()
    }

    #[getter]
    fn individual_identifiers(&self) -> Vec<String> {
        self.data.individual_identifiers.clone()
    }

    #[getter]
    fn covariate_names(&self) -> Vec<String> {
        self.data.covariate_names.clone()
    }
}

#[pymethods]
impl NativeMultiTraitRunInput {
    #[getter]
    #[allow(clippy::unused_self)]
    fn native_aligned_sample_data(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn native_multi_aligned_sample_data(&self) -> NativeMultiAlignedSampleData {
        NativeMultiAlignedSampleData::new(self.data.clone())
    }

    #[getter]
    fn phenotype_names(&self) -> Vec<String> {
        self.data.phenotype_names.clone()
    }

    #[getter]
    fn sample_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(usize_slice_to_py_i64(&self.data.sample_indices, "sample_indices")?.into_pyarray(py))
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

    #[getter]
    fn family_identifiers(&self) -> Vec<String> {
        self.data.family_identifiers.clone()
    }

    #[getter]
    fn individual_identifiers(&self) -> Vec<String> {
        self.data.individual_identifiers.clone()
    }

    #[getter]
    fn covariate_names(&self) -> Vec<String> {
        self.data.covariate_names.clone()
    }
}

#[pymethods]
impl NativeRunCallbackContext {
    #[getter]
    fn association_mode(&self) -> &str {
        &self.association_mode
    }

    #[getter]
    fn trait_type(&self) -> &str {
        &self.trait_type
    }

    #[getter]
    fn correction_method(&self) -> &str {
        &self.correction_method
    }

    #[getter]
    fn correction_p_threshold(&self) -> f64 {
        self.correction_p_threshold
    }

    #[getter]
    fn correction_firth_se(&self) -> bool {
        self.correction_firth_se
    }

    #[getter]
    fn staging_depth(&self) -> i64 {
        self.staging_depth
    }

    #[getter]
    fn native_callback_batch_size(&self) -> i64 {
        self.native_callback_batch_size
    }

    #[getter]
    fn result_in_flight_limit(&self) -> Option<i64> {
        self.result_in_flight_limit
    }

    #[getter]
    fn dosage_buffer_limit(&self) -> Option<i64> {
        self.dosage_buffer_limit
    }

    #[getter]
    fn score_dtype(&self) -> &str {
        &self.score_dtype
    }

    #[getter]
    fn firth_dtype(&self) -> &str {
        &self.firth_dtype
    }

    #[getter]
    fn output_statistic_dtype(&self) -> &str {
        &self.output_statistic_dtype
    }

    #[getter]
    fn jax_device(&self) -> &str {
        &self.jax_device
    }

    #[getter]
    fn gpu_genotype_format(&self) -> &str {
        &self.gpu_genotype_format
    }

    #[getter]
    fn requested_gpu_genotype_format(&self) -> &str {
        &self.requested_gpu_genotype_format
    }
}

#[pymethods]
impl NativeSingleTraitPipelineBundle {
    #[getter]
    fn run_input(&self, py: Python<'_>) -> Py<NativeSingleTraitRunInput> {
        self.run_input.clone_ref(py)
    }

    #[getter]
    fn prediction_source(&self, py: Python<'_>) -> Py<RegeniePredictionSource> {
        self.prediction_source.clone_ref(py)
    }

    #[getter]
    fn phenotype_compute_group(&self) -> NativeResolvedPhenotypeComputeGroup {
        NativeResolvedPhenotypeComputeGroup::new(self.phenotype_compute_group.data.clone())
    }

    #[getter]
    fn output_bundle(&self, py: Python<'_>) -> Py<NativePreparedOutputBundle> {
        self.output_bundle.clone_ref(py)
    }

    #[getter]
    fn writer_session(&self, py: Python<'_>) -> Py<OutputWriterSession> {
        self.writer_session.clone_ref(py)
    }

    #[getter]
    fn committed_chunk_identifiers(&self) -> Vec<usize> {
        self.committed_chunk_identifiers.clone()
    }
}

#[pymethods]
impl NativeRunEngineSession {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        py: Python<'_>,
        config: &RegenieConfig,
        runtime_compatibility_token: PyRef<'_, NativeRuntimeCompatibilityToken>,
    ) -> PyResult<Self> {
        Ok(Self {
            lifecycle: NativeRunLifecycleSession::from_config(py, config, &runtime_compatibility_token)?,
            engine: Mutex::new(None),
        })
    }

    fn run_to_completion<'py>(
        &self,
        py: Python<'py>,
        callback_factory: &Bound<'py, PyAny>,
        telemetry_session: Option<&Bound<'py, PyAny>>,
        stage_timing_recorder: Option<PyRef<'py, NativeStageTimingRecorder>>,
    ) -> PyResult<NativeRunArtifacts> {
        self.run_to_completion_internal(py, callback_factory, telemetry_session, stage_timing_recorder.as_deref())
    }
}

impl NativeRunEngineSession {
    fn phase(&self) -> PyResult<&'static str> {
        self.lifecycle.phase_label()
    }
    fn output_resume(&self) -> bool {
        self.lifecycle.output_resume_value()
    }
    fn run_request(&self) -> NativeRunRequest {
        self.lifecycle.run_request_handle()
    }
    fn sample_count(&self) -> PyResult<usize> {
        self.with_open_engine(|engine| Ok(engine.reader().sample_count()))
    }
    fn variant_count(&self) -> PyResult<usize> {
        self.with_open_engine(|engine| Ok(engine.reader().variant_count()))
    }
    fn contains_embedded_samples(&self) -> PyResult<bool> {
        self.with_open_engine(|engine| Ok(engine.reader().contains_embedded_samples()))
    }

    fn prepared_phenotype_runs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.lifecycle.prepared_phenotype_runs_tuple(py)
    }
    fn prepared_phenotype_run(&self, phenotype_name: String) -> PyResult<NativeRunLifecyclePhenotypeRun> {
        self.lifecycle.prepared_phenotype_run_handle(phenotype_name)
    }

    fn mark_dispatch_started(&self) -> PyResult<()> {
        self.lifecycle.mark_dispatch_started_internal()
    }

    fn has_open_bgen_engine(&self) -> PyResult<bool> {
        Ok(self.lock_engine()?.is_some())
    }
    fn open_bgen_engine(
        &self,
        py: Python<'_>,
        bgen_path: String,
        chunk_size: usize,
        variant_limit: Option<usize>,
        trusted_no_missing_diploid: bool,
        trusted_bgen_validation_mode: Option<String>,
    ) -> PyResult<bool> {
        self.open_bgen_engine_internal(
            py,
            &bgen_path,
            chunk_size,
            variant_limit,
            trusted_no_missing_diploid,
            trusted_bgen_validation_mode.as_deref(),
        )
    }

    fn sample_identifiers(&self) -> PyResult<Vec<String>> {
        self.with_open_engine(|engine| Ok(engine.reader().sample_identifiers()))
    }
    #[allow(clippy::needless_pass_by_value)]
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
        self.with_open_engine(|engine| {
            align_sample_data_for_engine(
                engine,
                py,
                sample_path,
                phenotype_path,
                phenotype_name,
                covariate_path,
                covariate_names,
                is_binary_trait,
                &sample_key_mode,
            )
        })
    }
    #[allow(clippy::needless_pass_by_value)]
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
        self.with_open_engine(|engine| {
            align_multi_sample_data_for_engine(
                engine,
                py,
                sample_path,
                phenotype_path,
                phenotype_names,
                covariate_path,
                covariate_names,
                is_binary_trait,
                &sample_key_mode,
            )
        })
    }
    #[allow(clippy::needless_pass_by_value)]
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
        self.with_open_engine(|engine| {
            align_grouped_sample_data_for_engine(
                engine,
                py,
                sample_path,
                phenotype_path,
                phenotype_names,
                covariate_path,
                covariate_names,
                is_binary_trait,
                &sample_key_mode,
            )
        })
    }

    fn chromosome_boundary_indices(&self) -> PyResult<Vec<usize>> {
        self.with_open_engine(|engine| Ok(engine.reader().chromosome_boundary_indices()))
    }

    fn variant_metadata_slice(
        &self,
        py: Python<'_>,
        variant_start: usize,
        variant_stop: usize,
    ) -> PyResult<VariantMetadataTuple> {
        self.with_open_engine(|engine| {
            py.detach(|| engine.reader().variant_metadata_slice(variant_start, variant_stop))
                .map(convert_variant_metadata_columns_to_tuple)
                .map_err(|error| convert_bgen_error("read_variant_metadata_slice", error))
        })
    }
    fn required_chromosomes(&self, variant_limit: Option<usize>) -> PyResult<Vec<String>> {
        self.with_open_engine(|engine| {
            engine.required_chromosomes(variant_limit).map_err(|error| convert_preflight_error(&error))
        })
    }

    fn reset_profile(&self) -> PyResult<()> {
        self.with_open_engine(|engine| {
            engine.reader().reset_profile();
            Ok(())
        })
    }

    fn profile_snapshot(&self) -> PyResult<HashMap<String, u64>> {
        self.with_open_engine(|engine| Ok(build_profile_snapshot_dict(&engine.reader().profile_snapshot())))
    }

    fn validate_trusted_no_missing_diploid(&self, py: Python<'_>) -> PyResult<()> {
        self.with_open_engine(|engine| {
            py.detach(|| engine.reader().validate_trusted_no_missing_diploid())
                .map_err(|error| convert_bgen_error("validate_trusted_no_missing_diploid", error))
        })
    }

    fn mark_trusted_no_missing_diploid_validated(&self, py: Python<'_>) -> PyResult<()> {
        self.with_open_engine(|engine| {
            py.detach(|| engine.reader().mark_trusted_no_missing_diploid_validated())
                .map_err(|error| convert_bgen_error("mark_trusted_no_missing_diploid_validated", error))
        })
    }
    fn validate_trusted_no_missing_diploid_with_default_cache(
        &self,
        py: Python<'_>,
        bgen_path: String,
        validation_mode: String,
    ) -> PyResult<()> {
        self.with_open_engine(|engine| {
            validate_trusted_no_missing_diploid_with_default_cache_for_engine(engine, py, &bgen_path, &validation_mode)
        })
    }
    #[allow(clippy::type_complexity)]
    fn prepare_output_bundles_from_runtime_plan<'py>(
        &self,
        py: Python<'py>,
        output_groups: Vec<NativeOutputRuntimeGroupInput>,
        variant_count: i64,
        effective_trusted_no_missing_diploid: bool,
        sample_key_mode: String,
        binary_kernel_config_json: Option<String>,
        requested_gpu_genotype_format: String,
        gpu_genotype_format: String,
        score_dtype: String,
        firth_dtype: String,
        stage_timing_recorder: Option<PyRef<'py, NativeStageTimingRecorder>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        self.lifecycle.prepare_output_bundles_from_runtime_plan_internal(
            py,
            output_groups,
            variant_count,
            effective_trusted_no_missing_diploid,
            sample_key_mode,
            binary_kernel_config_json,
            requested_gpu_genotype_format,
            gpu_genotype_format,
            score_dtype,
            firth_dtype,
            stage_timing_recorder.as_deref(),
        )
    }
    #[allow(clippy::too_many_lines)]
    fn prepare_single_trait_pipeline_bundle<'py>(
        &self,
        py: Python<'py>,
        phenotype_name: String,
        covariate_names: Option<Vec<String>>,
        association_mode: String,
        association_backend_kind: String,
        jax_device: String,
        genotype_format: String,
        requested_gpu_genotype_format: String,
        score_dtype: String,
        firth_dtype: String,
        binary_kernel_config_json: Option<String>,
        sample_key_mode: String,
        is_binary_trait: bool,
        pipeline_label: String,
        bgen_path: String,
        sample_path: Option<String>,
        phenotype_path: String,
        covariate_path: Option<String>,
        prediction_list_path: String,
        chunk_size: usize,
        variant_limit: Option<usize>,
        effective_trusted_no_missing_diploid: bool,
        trusted_bgen_validation_mode: String,
        telemetry_session: Option<&Bound<'py, PyAny>>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    ) -> PyResult<NativeSingleTraitPipelineBundle> {
        let stage_timing_recorder_reference = stage_timing_recorder;
        run_events::record_pipeline_single_trait_started_diagnostic_event(
            &association_mode,
            &phenotype_name,
            &pipeline_label,
        )?;
        let engine_was_open = self.lock_engine()?.is_some();
        if engine_was_open {
            run_events::record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
                None,
                Some(phenotype_name.as_str()),
                &pipeline_label,
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
                &pipeline_label,
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
            &pipeline_label,
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
        run_events::record_pipeline_single_trait_input_load_started_diagnostic_event(&phenotype_name, &pipeline_label)?;
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
        let sample_count = usize_to_i64(aligned_sample_data.data.sample_indices.len(), "Aligned sample count")?;
        let covariate_count = usize_to_i64(aligned_sample_data.data.covariate_names.len(), "Covariate count")?;
        run_events::record_pipeline_single_trait_input_aligned_diagnostic_event(
            covariate_count,
            &phenotype_name,
            &pipeline_label,
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
            &pipeline_label,
        )?;
        let prediction_source = prediction_sources::load_regenie_prediction_source_from_aligned_sample_data(
            &prediction_list_path,
            &phenotype_name,
            &aligned_sample_data,
            &sample_key_mode,
        )?;
        record_stage_duration(stage_timing_recorder_reference, "prediction_source_load", prediction_start_time)?;
        run_events::record_prediction_source_loaded_telemetry(
            telemetry_session,
            &association_mode,
            Some(phenotype_name.clone()),
            None,
        )?;

        let parsed_sample_key_mode = parse_sample_key_mode(&sample_key_mode)?;
        let phenotype_compute_group =
            NativeResolvedPhenotypeComputeGroup::new(native_input::resolve_single_phenotype_compute_group(
                &aligned_sample_data.data,
                phenotype_name.clone(),
                Some(prediction_list_path.as_str()),
                parsed_sample_key_mode,
            ));

        let preflight_start_time = Instant::now();
        run_events::record_pipeline_single_trait_preflight_started_diagnostic_event(
            &phenotype_name,
            &pipeline_label,
            effective_trusted_no_missing_diploid,
            option_usize_to_i64(variant_limit, "BGEN variant limit")?,
        )?;
        let preflight_shape = native_engine_debug::validate_single_trait_preflight_values(
            &aligned_sample_data.data.phenotype_vector,
            aligned_sample_data.data.covariate_row_count,
            aligned_sample_data.data.covariate_column_count,
            &aligned_sample_data.data.covariate_matrix_values,
            is_binary_trait,
        )
        .map_err(|error| convert_preflight_error(&error))?;
        let required_chromosomes = self.with_open_engine(|engine| {
            engine.required_chromosomes(variant_limit).map_err(|error| convert_preflight_error(&error))
        })?;
        for chromosome in &required_chromosomes {
            let prediction_values = prediction_source
                .source
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
            &pipeline_label,
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
            &aligned_sample_data.data,
            &phenotype_compute_group.data,
            preflight_report.sample_count,
        )?;
        let mut output_bundles = self.lifecycle.prepare_output_bundle_objects_from_runtime_plan_internal(
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
        let (writer_session, committed_chunk_identifiers) = {
            let output_bundle_reference = output_bundle.bind(py).borrow();
            (
                output_bundle_reference.writer_session_handle(py, 0)?,
                output_bundle_reference.committed_chunk_identifiers_usize(0)?,
            )
        };
        Ok(NativeSingleTraitPipelineBundle {
            run_input: Py::new(py, NativeSingleTraitRunInput { data: aligned_sample_data.data })?,
            prediction_source: Py::new(py, prediction_source)?,
            phenotype_compute_group,
            output_bundle,
            writer_session,
            committed_chunk_identifiers,
        })
    }
    #[allow(clippy::needless_pass_by_value)]
    fn run_single_trait_pipeline_bundle<'py>(
        &self,
        py: Python<'py>,
        bundle: PyRef<'py, NativeSingleTraitPipelineBundle>,
        callback: &Bound<'py, PyAny>,
        stage_timing_recorder: Option<PyRef<'py, NativeStageTimingRecorder>>,
        variant_major_packed8_probability_pairs: bool,
        pipeline_label: String,
    ) -> PyResult<Option<String>> {
        self.run_single_trait_pipeline_bundle_internal(
            py,
            &bundle,
            callback,
            stage_timing_recorder.as_deref(),
            variant_major_packed8_probability_pairs,
            &pipeline_label,
        )
    }
}


impl NativeRunEngineSession {
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
        callback_factory: &Bound<'py, PyAny>,
        telemetry_session: Option<&Bound<'py, PyAny>>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    ) -> PyResult<NativeRunArtifacts> {
        let output_start_time = Instant::now();
        run_events::record_runner_execution_plan_build_started_diagnostic_event()?;
        let run_request = self.lifecycle.run_request_data().clone();
        let phenotype_count = usize_to_i64(run_request.phenotype_runs.len(), "Phenotype count")?;
        let resolved_execution =
            self.resolve_run_execution(py, callback_factory, telemetry_session, stage_timing_recorder, &run_request)?;
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
                callback_factory,
                telemetry_session,
                stage_timing_recorder,
                &run_request,
                &resolved_execution,
            )?
        } else {
            self.run_single_trait_to_completion(
                py,
                callback_factory,
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
        callback_factory: &Bound<'py, PyAny>,
        telemetry_session: Option<&Bound<'py, PyAny>>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
    ) -> PyResult<NativeRunResolvedExecution> {
        let binary_kernel_config_json =
            callback_factory.call_method0("binary_kernel_config_json")?.extract::<Option<String>>()?;
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
        })
    }

    fn resolve_gpu_genotype_format<'py>(
        &self,
        py: Python<'py>,
        telemetry_session: Option<&Bound<'py, PyAny>>,
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
        callback_factory: &Bound<'py, PyAny>,
        telemetry_session: Option<&Bound<'py, PyAny>>,
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
            phenotype_run.phenotype_name.clone(),
            Some(run_request.input.covariate_names.clone()),
            run_request.association_mode.as_str().to_string(),
            resolved_execution.backend_plan.kind.as_str().to_string(),
            resolved_execution.backend_plan.device.as_str().to_string(),
            resolved_execution.resolved_gpu_genotype_format.clone(),
            resolved_execution.requested_gpu_genotype_format.clone(),
            run_request.compute.score_dtype.as_str().to_string(),
            run_request.compute.firth_dtype.as_str().to_string(),
            resolved_execution.binary_kernel_config_json.clone(),
            run_request.input.sample_key_mode.as_str().to_string(),
            run_request.trait_request.trait_type == native_plan::RegenieTraitType::Binary,
            pipeline_label.to_string(),
            run_request.input.bgen_path.clone(),
            run_request.input.sample_path.clone(),
            run_request.input.phenotype_path.clone(),
            run_request.input.covariate_path.clone(),
            run_request.input.prediction_list_path.clone(),
            u32_value_as_usize(run_request.trait_request.chunk_size, "trait chunk size")?,
            run_request.compute.variant_limit.map(|value| u32_value_as_usize(value, "variant limit")).transpose()?,
            resolved_execution.effective_trusted_no_missing_diploid,
            run_request.compute.trusted_bgen_validation_mode.as_str().to_string(),
            telemetry_session,
            stage_timing_recorder,
        )?;
        let callback_context = Py::new(py, callback_context_from_request(run_request, resolved_execution)?)?;
        let callback = callback_factory.call_method1(
            "build_single_trait_callback",
            (
                callback_context,
                bundle.run_input.clone_ref(py),
                bundle.prediction_source.clone_ref(py),
                bundle.writer_session.clone_ref(py),
            ),
        )?;
        let final_output_path = self.run_single_trait_pipeline_bundle_internal(
            py,
            &bundle,
            &callback,
            stage_timing_recorder,
            resolved_execution.backend_plan.resolved_genotype_format == native_plan::GpuGenotypeFormat::Packed8,
            "Native BGEN",
        )?;
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
        callback_factory: &Bound<'py, PyAny>,
        telemetry_session: Option<&Bound<'py, PyAny>>,
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
                callback_factory,
                telemetry_session,
                stage_timing_recorder,
                run_request,
                resolved_execution,
            )?,
            native_plan::MultiPhenotypeSampleMode::PerPhenotype => self.run_grouped_per_phenotype_to_completion(
                py,
                callback_factory,
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
    fn run_complete_case_multi_trait_to_completion<'py>(
        &self,
        py: Python<'py>,
        callback_factory: &Bound<'py, PyAny>,
        telemetry_session: Option<&Bound<'py, PyAny>>,
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
            &aligned_sample_data.data,
            u32_indices_to_usize(&planned_compute_group.phenotype_indices, "phenotype compute group index")?,
            planned_compute_group.phenotype_names.clone(),
            Some(run_request.input.prediction_list_path.as_str()),
            parse_sample_key_mode(run_request.input.sample_key_mode.as_str())?,
        );
        record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)?;
        let sample_count = usize_to_i64(aligned_sample_data.data.sample_indices.len(), "Aligned sample count")?;
        let covariate_count = usize_to_i64(aligned_sample_data.data.covariate_names.len(), "Covariate count")?;
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
        self.record_multi_phenotype_sample_summary(
            telemetry_session,
            run_request,
            native_plan::MultiPhenotypeSampleMode::CompleteCase,
            &[resolved_compute_group.clone()],
            &[sample_count],
        )?;
        let prediction_start_time = Instant::now();
        run_events::record_pipeline_multi_trait_prediction_source_load_started_diagnostic_event(phenotype_count)?;
        let prediction_source =
            prediction_sources::load_multi_regenie_prediction_source_from_multi_aligned_sample_data(
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
            &aligned_sample_data.data,
            &prediction_source,
        )?;
        let output_bundle = self.prepare_multi_trait_output_bundle(
            py,
            stage_timing_recorder,
            run_request,
            resolved_execution,
            &aligned_sample_data.data,
            &resolved_compute_group,
            native_plan::MultiPhenotypeSampleMode::CompleteCase,
        )?;
        let run_input = Py::new(py, NativeMultiTraitRunInput { data: aligned_sample_data.data.clone() })?;
        let aligned_sample_data_handle = Py::new(py, aligned_sample_data)?;
        let prediction_source_handle = Py::new(py, prediction_source)?;
        let prepared_delivery = self.prepare_multi_group_delivery(
            py,
            callback_factory,
            run_request,
            resolved_execution,
            run_input,
            aligned_sample_data_handle,
            prediction_source_handle,
            output_bundle,
            u32_indices_to_usize(&planned_compute_group.phenotype_indices, "phenotype compute group index")?,
        )?;
        self.run_prepared_multi_group_delivery(
            py,
            stage_timing_recorder,
            resolved_execution,
            prepared_delivery,
            "Multi-phenotype native BGEN",
            i64::from(run_request.output.writer_thread_count),
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn run_grouped_per_phenotype_to_completion<'py>(
        &self,
        py: Python<'py>,
        callback_factory: &Bound<'py, PyAny>,
        telemetry_session: Option<&Bound<'py, PyAny>>,
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
        let prediction_sources =
            prediction_sources::load_multi_regenie_prediction_sources_from_grouped_aligned_sample_data(
                &run_request.input.prediction_list_path,
                &grouped_aligned_sample_data,
                run_request.input.sample_key_mode.as_str(),
            )?;
        if grouped_aligned_sample_data.data.groups.len() != prediction_sources.len() {
            return Err(PyValueError::new_err("Grouped prediction source count does not match aligned group count."));
        }
        let grouped_run_inputs =
            self.build_grouped_run_inputs(py, run_request, grouped_aligned_sample_data, prediction_sources)?;
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
        self.record_multi_phenotype_sample_summary(
            telemetry_session,
            run_request,
            native_plan::MultiPhenotypeSampleMode::PerPhenotype,
            &grouped_run_inputs.iter().map(|group| group.compute_group.clone()).collect::<Vec<_>>(),
            &grouped_run_inputs.iter().map(|group| group.sample_count).collect::<Vec<_>>(),
        )?;
        for grouped_run_input in &grouped_run_inputs {
            let prediction_source_reference = grouped_run_input.prediction_source.bind(py).borrow();
            let aligned_sample_reference = grouped_run_input.aligned_sample_data.bind(py).borrow();
            self.run_multi_group_preflight(
                telemetry_session,
                stage_timing_recorder,
                run_request,
                resolved_execution,
                &aligned_sample_reference.data,
                &prediction_source_reference,
            )?;
        }
        let output_bundles = self.prepare_grouped_output_bundles(
            py,
            stage_timing_recorder,
            run_request,
            resolved_execution,
            &grouped_run_inputs,
        )?;
        if should_use_union_grouped_bgen_delivery(resolved_execution, &grouped_run_inputs) {
            return self.run_grouped_union_delivery(
                py,
                callback_factory,
                stage_timing_recorder,
                run_request,
                resolved_execution,
                grouped_run_inputs,
                output_bundles,
                phenotype_names.len(),
            );
        }
        let mut final_paths_by_index = vec![None; phenotype_names.len()];
        for (grouped_run_input, output_bundle) in grouped_run_inputs.into_iter().zip(output_bundles) {
            let phenotype_indices = grouped_run_input.phenotype_indices.clone();
            let prepared_delivery = self.prepare_multi_group_delivery(
                py,
                callback_factory,
                run_request,
                resolved_execution,
                grouped_run_input.run_input,
                grouped_run_input.aligned_sample_data,
                grouped_run_input.prediction_source,
                output_bundle,
                grouped_run_input.phenotype_indices,
            )?;
            let group_paths = self.run_prepared_multi_group_delivery(
                py,
                stage_timing_recorder,
                resolved_execution,
                prepared_delivery,
                "Multi-phenotype native BGEN",
                i64::from(run_request.output.writer_thread_count),
            )?;
            scatter_group_final_paths(&mut final_paths_by_index, &phenotype_indices, &group_paths)?;
        }
        Ok(final_paths_by_index)
    }
}

impl NativeRunEngineSession {
    fn run_single_trait_pipeline_bundle_internal<'py>(
        &self,
        py: Python<'py>,
        bundle: &NativeSingleTraitPipelineBundle,
        callback: &Bound<'py, PyAny>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        variant_major_packed8_probability_pairs: bool,
        pipeline_label: &str,
    ) -> PyResult<Option<String>> {
        let run_input_handle = bundle.run_input.clone_ref(py);
        let writer_session_handle = bundle.writer_session.clone_ref(py);
        let committed_chunk_identifiers = bundle.committed_chunk_identifiers.clone();

        let run_input_data = run_input_handle.bind(py).borrow().data.clone();
        let sample_indices_array =
            usize_slice_to_py_i64(&run_input_data.sample_indices, "sample_indices")?.into_pyarray(py);
        let sample_indices = sample_indices_array.readonly();
        let native_aligned_sample_data = Py::new(py, NativeAlignedSampleData::new(run_input_data))?;
        let native_aligned_sample_data_reference = native_aligned_sample_data.bind(py).borrow();
        let writer_session_reference = writer_session_handle.bind(py).borrow();
        let writer_sessions = vec![writer_session_reference];
        let engine_guard = self.lock_engine()?;
        let engine = engine_guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Native run engine session has no open BGEN engine."))?;
        let mut callback_finished = false;
        let delivery_result = run_bgen_delivery_attempt(
            py,
            engine,
            &sample_indices,
            Some(native_aligned_sample_data_reference),
            None,
            &writer_sessions,
            callback,
            stage_timing_recorder,
            1,
            Some(committed_chunk_identifiers),
            variant_major_packed8_probability_pairs,
            pipeline_label,
            &mut callback_finished,
        );
        match delivery_result {
            Ok(final_parquet_paths) => {
                run_events::record_native_dispatch_pipeline_finished_diagnostic_event(
                    usize_to_i64(final_parquet_paths.len(), "Final Parquet path count")?,
                    pipeline_label,
                )?;
                Ok(final_parquet_paths.into_iter().next().flatten())
            }
            Err(error) => handle_bgen_delivery_error(
                py,
                error,
                callback_finished,
                callback,
                &writer_sessions,
                stage_timing_recorder,
                1,
                pipeline_label,
            )
            .map(|final_parquet_paths| final_parquet_paths.into_iter().next().flatten()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn open_pipeline_bgen_engine_with_events(
        &self,
        py: Python<'_>,
        telemetry_session: Option<&Bound<'_, PyAny>>,
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
        &self,
        telemetry_session: Option<&Bound<'_, PyAny>>,
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
        telemetry_session: Option<&Bound<'_, PyAny>>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
        aligned_sample_data: &native_input::MultiAlignedSampleData,
        prediction_source: &prediction_sources::MultiRegeniePredictionSource,
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
                .source
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
    ) -> PyResult<Py<NativePreparedOutputBundle>> {
        let engine_variant_count =
            self.with_open_engine(|engine| usize_to_i64(engine.reader().variant_count(), "BGEN variant count"))?;
        let sample_count = usize_to_i64(aligned_sample_data.sample_indices.len(), "Aligned sample count")?;
        let output_group = build_multi_trait_output_group(
            &aligned_sample_data.covariate_names,
            sample_count,
            output_sample_mode,
            compute_group,
        )?;
        let mut output_bundles = self.lifecycle.prepare_output_bundle_objects_from_runtime_plan_internal(
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
        &self,
        py: Python<'_>,
        run_request: &native_plan::RunRequest,
        grouped_aligned_sample_data: NativeGroupedAlignedSampleData,
        prediction_sources: Vec<prediction_sources::MultiRegeniePredictionSource>,
    ) -> PyResult<Vec<NativeGroupedRunInputState>> {
        let planned_names_by_index = planned_phenotype_names_by_index(run_request)?;
        let parsed_sample_key_mode = parse_sample_key_mode(run_request.input.sample_key_mode.as_str())?;
        grouped_aligned_sample_data
            .data
            .groups
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
                    run_input: Py::new(py, NativeMultiTraitRunInput { data: group.aligned_sample_data.clone() })?,
                    aligned_sample_data: Py::new(py, NativeMultiAlignedSampleData::new(group.aligned_sample_data))?,
                    prediction_source: Py::new(py, prediction_source)?,
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
    ) -> PyResult<Vec<Py<NativePreparedOutputBundle>>> {
        let engine_variant_count =
            self.with_open_engine(|engine| usize_to_i64(engine.reader().variant_count(), "BGEN variant count"))?;
        let mut output_groups = Vec::with_capacity(grouped_run_inputs.len());
        for grouped_run_input in grouped_run_inputs {
            let aligned_sample_data = grouped_run_input.aligned_sample_data.bind(py).borrow();
            output_groups.push(build_multi_trait_output_group(
                &aligned_sample_data.data.covariate_names,
                grouped_run_input.sample_count,
                native_plan::MultiPhenotypeSampleMode::PerPhenotype,
                &grouped_run_input.compute_group,
            )?);
        }
        self.lifecycle.prepare_output_bundle_objects_from_runtime_plan_internal(
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

    #[allow(clippy::too_many_arguments)]
    fn prepare_multi_group_delivery<'py>(
        &self,
        py: Python<'py>,
        callback_factory: &Bound<'py, PyAny>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
        run_input: Py<NativeMultiTraitRunInput>,
        aligned_sample_data: Py<NativeMultiAlignedSampleData>,
        prediction_source: Py<prediction_sources::MultiRegeniePredictionSource>,
        output_bundle: Py<NativePreparedOutputBundle>,
        phenotype_indices: Vec<usize>,
    ) -> PyResult<NativePreparedMultiGroupDelivery> {
        let (writer_sessions, chunk_write_planner) = {
            let output_bundle_reference = output_bundle.bind(py).borrow();
            (
                output_bundle_reference.writer_session_handles(py),
                output_bundle_reference.multi_trait_chunk_write_planner_handle()?,
            )
        };
        let writer_sessions_tuple = PyTuple::new(py, &writer_sessions)?;
        let callback_context = Py::new(py, callback_context_from_request(run_request, resolved_execution)?)?;
        let callback = callback_factory
            .call_method1(
                "build_multi_trait_callback",
                (
                    callback_context,
                    run_input.clone_ref(py),
                    prediction_source.clone_ref(py),
                    writer_sessions_tuple,
                    chunk_write_planner,
                ),
            )?
            .unbind();
        let sample_indices = aligned_sample_data.bind(py).borrow().data.sample_indices.clone();
        Ok(NativePreparedMultiGroupDelivery {
            phenotype_indices,
            aligned_sample_data,
            callback,
            writer_sessions,
            output_bundle,
            sample_indices,
        })
    }

    fn run_prepared_multi_group_delivery(
        &self,
        py: Python<'_>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        resolved_execution: &NativeRunResolvedExecution,
        prepared_delivery: NativePreparedMultiGroupDelivery,
        pipeline_label: &str,
        writer_finish_thread_count: i64,
    ) -> PyResult<Vec<Option<String>>> {
        let sample_indices_array =
            usize_slice_to_py_i64(&prepared_delivery.sample_indices, "sample_indices")?.into_pyarray(py);
        let sample_indices = sample_indices_array.readonly();
        let aligned_sample_data_reference = prepared_delivery.aligned_sample_data.bind(py).borrow();
        let writer_session_references =
            prepared_delivery.writer_sessions.iter().map(|session| session.bind(py).borrow()).collect::<Vec<_>>();
        let committed_chunk_identifiers = {
            let output_bundle_reference = prepared_delivery.output_bundle.bind(py).borrow();
            output_bundle_reference.shared_committed_chunk_identifiers_usize()?
        };
        let callback = prepared_delivery.callback.bind(py);
        let engine_guard = self.lock_engine()?;
        let engine = engine_guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Native run engine session has no open BGEN engine."))?;
        let mut callback_finished = false;
        let delivery_result = run_bgen_delivery_attempt(
            py,
            engine,
            &sample_indices,
            None,
            Some(aligned_sample_data_reference),
            &writer_session_references,
            callback,
            stage_timing_recorder,
            writer_finish_thread_count,
            Some(committed_chunk_identifiers),
            resolved_execution.backend_plan.resolved_genotype_format == native_plan::GpuGenotypeFormat::Packed8,
            pipeline_label,
            &mut callback_finished,
        );
        match delivery_result {
            Ok(final_parquet_paths) => {
                run_events::record_native_dispatch_pipeline_finished_diagnostic_event(
                    usize_to_i64(final_parquet_paths.len(), "Final Parquet path count")?,
                    pipeline_label,
                )?;
                Ok(final_parquet_paths)
            }
            Err(error) => handle_bgen_delivery_error(
                py,
                error,
                callback_finished,
                callback,
                &writer_session_references,
                stage_timing_recorder,
                writer_finish_thread_count,
                pipeline_label,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_grouped_union_delivery<'py>(
        &self,
        py: Python<'py>,
        callback_factory: &Bound<'py, PyAny>,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
        run_request: &native_plan::RunRequest,
        resolved_execution: &NativeRunResolvedExecution,
        grouped_run_inputs: Vec<NativeGroupedRunInputState>,
        output_bundles: Vec<Py<NativePreparedOutputBundle>>,
        phenotype_count: usize,
    ) -> PyResult<Vec<Option<String>>> {
        native_engine_debug::resolve_grouped_union_callback_batch_size(i64::from(
            run_request.compute.native_callback_batch_size,
        ))
        .map_err(|error| convert_schedule_error(&error))?;
        let sample_indices_by_group =
            grouped_run_inputs.iter().map(|group| group.sample_indices.clone()).collect::<Vec<_>>();
        let union_sample_indices = native_input::build_union_sample_indices(&sample_indices_by_group);
        let grouped_sample_count = sample_indices_by_group.iter().map(Vec::len).sum::<usize>();
        run_events::record_pipeline_grouped_union_delivery_selected_diagnostic_event(
            usize_to_i64(grouped_sample_count, "Grouped sample count")?,
            usize_to_i64(grouped_run_inputs.len(), "Phenotype group count")?,
            usize_to_i64(union_sample_indices.len(), "Union sample count")?,
        )?;
        let mut prepared_deliveries = Vec::with_capacity(grouped_run_inputs.len());
        for (grouped_run_input, output_bundle) in grouped_run_inputs.into_iter().zip(output_bundles) {
            prepared_deliveries.push(self.prepare_multi_group_delivery(
                py,
                callback_factory,
                run_request,
                resolved_execution,
                grouped_run_input.run_input,
                grouped_run_input.aligned_sample_data,
                grouped_run_input.prediction_source,
                output_bundle,
                grouped_run_input.phenotype_indices,
            )?);
        }
        let callbacks = prepared_deliveries.iter().map(|delivery| delivery.callback.clone_ref(py)).collect::<Vec<_>>();
        let mut sample_position_arrays = Vec::with_capacity(prepared_deliveries.len());
        for prepared_delivery in &prepared_deliveries {
            let sample_position_values = native_input::build_group_sample_position_array(
                &union_sample_indices,
                &prepared_delivery.sample_indices,
            )
            .map_err(|error| convert_input_error("build_group_sample_position_array", error.into()))?;
            sample_position_arrays.push(sample_position_values.into_pyarray(py).unbind());
        }
        let callbacks_tuple = PyTuple::new(py, &callbacks)?;
        let sample_position_arrays_tuple = PyTuple::new(py, &sample_position_arrays)?;
        let fanout_callback = callback_factory
            .call_method1("build_grouped_fanout_callback", (callbacks_tuple, sample_position_arrays_tuple))?
            .unbind();
        let writer_sessions = prepared_deliveries
            .iter()
            .flat_map(|delivery| delivery.writer_sessions.iter().map(|session| session.clone_ref(py)))
            .collect::<Vec<_>>();
        let writer_session_references =
            writer_sessions.iter().map(|session| session.bind(py).borrow()).collect::<Vec<_>>();
        let bundle_references =
            prepared_deliveries.iter().map(|delivery| delivery.output_bundle.bind(py).borrow()).collect::<Vec<_>>();
        let committed_chunk_identifiers =
            NativePreparedOutputBundle::shared_committed_chunk_identifiers_across_bundles_usize(&bundle_references)?;
        let union_sample_indices_array =
            usize_slice_to_py_i64(&union_sample_indices, "union_sample_indices")?.into_pyarray(py);
        let union_sample_indices_reference = union_sample_indices_array.readonly();
        let callback = fanout_callback.bind(py);
        let engine_guard = self.lock_engine()?;
        let engine = engine_guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Native run engine session has no open BGEN engine."))?;
        let mut callback_finished = false;
        let delivery_result = run_bgen_delivery_attempt(
            py,
            engine,
            &union_sample_indices_reference,
            None,
            None,
            &writer_session_references,
            callback,
            stage_timing_recorder,
            1,
            Some(committed_chunk_identifiers),
            false,
            "Grouped per-phenotype union native BGEN",
            &mut callback_finished,
        );
        let final_parquet_paths = match delivery_result {
            Ok(final_parquet_paths) => {
                run_events::record_native_dispatch_pipeline_finished_diagnostic_event(
                    usize_to_i64(final_parquet_paths.len(), "Final Parquet path count")?,
                    "Grouped per-phenotype union native BGEN",
                )?;
                final_parquet_paths
            }
            Err(error) => handle_bgen_delivery_error(
                py,
                error,
                callback_finished,
                callback,
                &writer_session_references,
                stage_timing_recorder,
                1,
                "Grouped per-phenotype union native BGEN",
            )?,
        };
        let mut final_paths_by_index = vec![None; phenotype_count];
        let mut final_path_offset = 0;
        for prepared_delivery in prepared_deliveries {
            let group_path_count = prepared_delivery.phenotype_indices.len();
            let group_paths = final_parquet_paths[final_path_offset..final_path_offset + group_path_count].to_vec();
            final_path_offset += group_path_count;
            scatter_group_final_paths(&mut final_paths_by_index, &prepared_delivery.phenotype_indices, &group_paths)?;
        }
        Ok(final_paths_by_index)
    }
}

fn delivery_callback_batch_size(callback: &Bound<'_, PyAny>) -> PyResult<i64> {
    callback.getattr("native_callback_batch_size")?.extract::<i64>()
}

fn for_each_owned_callback_runtime<'py, Function>(callback: &Bound<'py, PyAny>, mut operation: Function) -> PyResult<()>
where
    Function: FnMut(Bound<'py, PyAny>, Bound<'py, PyAny>) -> PyResult<()>,
{
    if callback.hasattr("callback_runtime_resources")? {
        let resources = callback.getattr("callback_runtime_resources")?;
        return operation(callback.clone(), resources);
    }
    if !callback.hasattr("group_fanouts")? {
        return Err(PyRuntimeError::new_err(
            "Delivery callback is missing native callback runtime resources.",
        ));
    }
    let group_fanouts = callback.getattr("group_fanouts")?;
    for index in 0..group_fanouts.len()? {
        let fanout = group_fanouts.get_item(index)?;
        let child_callback = fanout.getattr("callback")?;
        if !child_callback.hasattr("callback_runtime_resources")? {
            return Err(PyRuntimeError::new_err(
                "Grouped fanout child callback is missing native callback runtime resources.",
            ));
        }
        let resources = child_callback.getattr("callback_runtime_resources")?;
        operation(child_callback, resources)?;
    }
    Ok(())
}

fn delivery_start_callback_runtime(py: Python<'_>, callback: &Bound<'_, PyAny>) -> PyResult<()> {
    let _ = py;
    for_each_owned_callback_runtime(callback, |_owner, resources| {
        resources.call_method0("start_workers")?;
        Ok(())
    })
}

fn delivery_abort_callback_runtime(py: Python<'_>, callback: &Bound<'_, PyAny>) -> PyResult<()> {
    let _ = py;
    for_each_owned_callback_runtime(callback, |_owner, resources| {
        resources.call_method0("abort_worker_lifecycle")?;
        Ok(())
    })
}

fn delivery_finish_callback_runtime(py: Python<'_>, callback: &Bound<'_, PyAny>) -> PyResult<()> {
    for_each_owned_callback_runtime(callback, |owner, resources| {
        let pending = if owner.hasattr("binary_correction_pending_diagnostics")? {
            owner.getattr("binary_correction_pending_diagnostics")?
        } else {
            pyo3::types::PyList::empty(py).into_any()
        };
        let finish_result =
            resources.call_method1("finish_worker_lifecycle_for_pending_diagnostics", (pending,))?;
        if finish_result.getattr("has_shutdown_timeout")?.extract::<bool>()? {
            let worker_name = finish_result.getattr("shutdown_worker_name")?;
            let timeout_seconds = finish_result.getattr("shutdown_timeout_seconds")?.extract::<f64>()?;
            let shared = py.import("g.engine.callbacks.shared")?;
            let error_type = shared.getattr("NativeBgenWorkerShutdownError")?;
            let error = error_type.call((), Some(&{
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("worker_name", worker_name)?;
                kwargs.set_item("timeout_seconds", timeout_seconds)?;
                kwargs
            }))?;
            return Err(PyErr::from_value(error));
        }
        if finish_result.getattr("raise_worker_error")?.extract::<bool>()?
            && owner.hasattr("raise_worker_error_if_present")?
        {
            owner.call_method0("raise_worker_error_if_present")?;
        }
        let progress_event = finish_result.getattr("progress_completion_event")?;
        let telemetry_session = if owner.hasattr("telemetry_session")? {
            Some(owner.getattr("telemetry_session")?)
        } else {
            None
        };
        let progress_event_ref = if progress_event.is_none() {
            None
        } else {
            Some(progress_event.extract()?)
        };
        run_events::record_callback_progress_event_telemetry(
            py,
            telemetry_session.as_ref(),
            progress_event_ref,
        )?;
        if finish_result.getattr("emit_binary_correction_summary")?.extract::<bool>()? {
            let mut summary_payload = finish_result.getattr("binary_correction_summary_payload")?;
            if summary_payload.is_none()
                && finish_result
                    .getattr("flush_binary_correction_pending_diagnostics")?
                    .extract::<bool>()?
                && owner.hasattr("materialize_binary_correction_pending_diagnostics")?
            {
                owner.call_method0("materialize_binary_correction_pending_diagnostics")?;
                summary_payload = resources.call_method0("binary_correction_summary_payload")?;
            }
            if !summary_payload.is_none() {
                let summary_dict = summary_payload.cast::<pyo3::types::PyDict>()?;
                run_events::record_binary_correction_summary_telemetry(
                    telemetry_session.as_ref(),
                    Some(summary_dict),
                )?;
            }
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn run_bgen_delivery_attempt<'py>(
    py: Python<'py>,
    engine: &Regenie2RunEngineCore,
    sample_indices: &PyReadonlyArray1<'py, i64>,
    native_aligned_sample_data: Option<PyRef<'py, NativeAlignedSampleData>>,
    native_multi_aligned_sample_data: Option<PyRef<'py, NativeMultiAlignedSampleData>>,
    writer_sessions: &[PyRef<'py, OutputWriterSession>],
    callback: &Bound<'py, PyAny>,
    stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    writer_finish_thread_count: i64,
    committed_chunk_identifiers: Option<Vec<usize>>,
    variant_major_packed8_probability_pairs: bool,
    pipeline_label: &str,
    callback_finished: &mut bool,
) -> PyResult<Vec<Option<String>>> {
    if stage_timing_recorder.is_some() {
        engine.reader().reset_profile();
    }
    let delivery_start_time = Instant::now();
    let committed_chunk_count = committed_chunk_identifiers.as_ref().map_or(0, Vec::len);
    delivery_start_callback_runtime(py, callback)?;
    let callback_batch_size = delivery_callback_batch_size(callback)?;
    let attempt_plan = plan_bgen_delivery_attempt_for_binding(
        Some(callback_batch_size),
        variant_major_packed8_probability_pairs,
        native_multi_aligned_sample_data.is_some(),
        native_aligned_sample_data.is_some(),
        committed_chunk_count,
    )?;
    run_events::record_native_dispatch_delivery_started_diagnostic_event(
        usize_to_i64(attempt_plan.committed_chunk_count, "Committed chunk count")?,
        pipeline_label,
        variant_major_packed8_probability_pairs,
    )?;
    let processed_chunk_count = run_bgen_delivery_invocation(
        engine,
        py,
        sample_indices,
        native_aligned_sample_data,
        native_multi_aligned_sample_data,
        callback,
        committed_chunk_identifiers,
        attempt_plan.invocation_plan,
    )?;
    record_stage_duration(stage_timing_recorder, "native_engine_delivery", delivery_start_time)?;
    run_events::record_native_dispatch_delivery_finished_diagnostic_event(
        pipeline_label,
        usize_to_i64(processed_chunk_count, "Processed chunk count")?,
    )?;
    if let Some(stage_timing_recorder) = stage_timing_recorder {
        stage_timing_recorder.set_native_bgen_profile_snapshot(native_bgen_profile_snapshot_as_i64(
            build_profile_snapshot_dict(&engine.reader().profile_snapshot()),
        )?)?;
    }
    let cleanup_execution = execute_bgen_delivery_cleanup_actions(
        py,
        native_engine_debug::BgenDeliveryCleanupOutcome::Success,
        *callback_finished,
        callback,
        writer_sessions,
        writer_finish_thread_count,
        stage_timing_recorder,
        None,
    )?;
    *callback_finished = cleanup_execution.callback_finished;
    Ok(cleanup_execution.final_parquet_paths)
}

#[allow(clippy::too_many_arguments)]
fn handle_bgen_delivery_error<'py>(
    py: Python<'py>,
    error: PyErr,
    callback_finished: bool,
    callback: &Bound<'py, PyAny>,
    writer_sessions: &[PyRef<'py, OutputWriterSession>],
    stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    writer_finish_thread_count: i64,
    pipeline_label: &str,
) -> PyResult<Vec<Option<String>>> {
    if let Some(interrupted_event) = maybe_shutdown_event_from_error(py, &error)? {
        let error_handling_plan = native_engine_debug::plan_bgen_delivery_error_handling(
            native_engine_debug::BgenDeliveryErrorKind::Interrupted,
        );
        run_events::record_native_dispatch_delivery_interrupted_diagnostic_event(
            pipeline_label,
            interrupted_event.exit_code,
            &interrupted_event.signal_name,
            interrupted_event.signal_number,
        )?;
        let cleanup_result = execute_bgen_delivery_cleanup_actions(
            py,
            error_handling_plan.cleanup_outcome,
            callback_finished,
            callback,
            writer_sessions,
            writer_finish_thread_count,
            stage_timing_recorder,
            Some(&interrupted_event),
        );
        return match cleanup_result {
            Ok(_) => Err(error),
            Err(cleanup_error) => {
                if let Some(fallback_cleanup_outcome) = error_handling_plan.fallback_cleanup_outcome {
                    execute_bgen_delivery_cleanup_actions(
                        py,
                        fallback_cleanup_outcome,
                        callback_finished,
                        callback,
                        writer_sessions,
                        writer_finish_thread_count,
                        stage_timing_recorder,
                        Some(&interrupted_event),
                    )?;
                }
                Err(cleanup_error)
            }
        };
    }

    let exception = error.value(py);
    let exception_type = exception.get_type().name()?.to_string_lossy().into_owned();
    let exception_message = exception.str()?.to_string_lossy().into_owned();
    run_events::record_native_dispatch_delivery_failed_diagnostic_event(
        &exception_message,
        &exception_type,
        pipeline_label,
    )?;
    let error_handling_plan =
        native_engine_debug::plan_bgen_delivery_error_handling(native_engine_debug::BgenDeliveryErrorKind::Failure);
    let cleanup_result = execute_bgen_delivery_cleanup_actions(
        py,
        error_handling_plan.cleanup_outcome,
        callback_finished,
        callback,
        writer_sessions,
        writer_finish_thread_count,
        stage_timing_recorder,
        None,
    );
    match cleanup_result {
        Ok(_) => Err(error),
        Err(cleanup_error) => Err(cleanup_error),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_bgen_delivery_cleanup_actions<'py>(
    py: Python<'py>,
    cleanup_outcome: native_engine_debug::BgenDeliveryCleanupOutcome,
    callback_finished: bool,
    callback: &Bound<'py, PyAny>,
    writer_sessions: &[PyRef<'py, OutputWriterSession>],
    writer_finish_thread_count: i64,
    stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    interrupted_event: Option<&native_run_events::RunInterruptedEventPayload>,
) -> PyResult<BgenDeliveryCleanupExecution> {
    let cleanup_plan = native_engine_debug::plan_bgen_delivery_cleanup(cleanup_outcome, callback_finished);
    let mut final_parquet_paths = Vec::new();
    let mut resolved_callback_finished = callback_finished;
    for cleanup_action in cleanup_plan.cleanup_actions() {
        match cleanup_action {
            native_engine_debug::BgenDeliveryCleanupAction::DrainCallback => {
                let callback_finish_start_time = Instant::now();
                run_events::record_native_dispatch_callback_drain_started_diagnostic_event()?;
                delivery_finish_callback_runtime(py, callback)?;
                record_stage_duration(stage_timing_recorder, "callback_drain", callback_finish_start_time)?;
                resolved_callback_finished = true;
            }
            native_engine_debug::BgenDeliveryCleanupAction::FinishWriterSessions => {
                let writer_finish_start_time = Instant::now();
                final_parquet_paths = output::finish_output_writer_sessions_for_delivery(
                    py,
                    writer_sessions,
                    writer_finish_thread_count,
                )?;
                record_stage_duration(
                    stage_timing_recorder,
                    "writer_finish_and_parquet_finalization",
                    writer_finish_start_time,
                )?;
            }
            native_engine_debug::BgenDeliveryCleanupAction::FinishInterruptedWriterSessions => {
                let Some(interrupted_event) = interrupted_event else {
                    return Err(PyRuntimeError::new_err("Interrupted writer cleanup requires a shutdown request."));
                };
                let writer_finish_start_time = Instant::now();
                output::finish_interrupted_output_writer_sessions_for_delivery(
                    py,
                    writer_sessions,
                    writer_finish_thread_count,
                    interrupted_event.exit_code,
                    &interrupted_event.signal_name,
                    interrupted_event.signal_number,
                )?;
                record_stage_duration(stage_timing_recorder, "writer_finish_interrupted", writer_finish_start_time)?;
            }
            native_engine_debug::BgenDeliveryCleanupAction::AbortCallback => {
                let _ = delivery_abort_callback_runtime(py, callback);
            }
            native_engine_debug::BgenDeliveryCleanupAction::AbortWriterSessions => {
                output::abort_output_writer_sessions_for_delivery(writer_sessions);
            }
            native_engine_debug::BgenDeliveryCleanupAction::WriteStageTimingSnapshot => {}
        }
    }
    Ok(BgenDeliveryCleanupExecution { final_parquet_paths, callback_finished: resolved_callback_finished })
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
    match run_events::run_interrupted_event_payload_from_shutdown_request(error.value(py)) {
        Ok(interrupted_event) => Ok(Some(interrupted_event)),
        Err(interrupted_error) if interrupted_error.is_instance_of::<PyAttributeError>(py) => Ok(None),
        Err(interrupted_error) => Err(interrupted_error),
    }
}

fn native_bgen_profile_snapshot_as_i64(profile_snapshot: HashMap<String, u64>) -> PyResult<BTreeMap<String, i64>> {
    profile_snapshot
        .into_iter()
        .map(|(key, value)| {
            let converted_value =
                i64::try_from(value).map_err(|_| PyValueError::new_err("Native BGEN profile counter overflowed."))?;
            Ok((key, converted_value))
        })
        .collect()
}

fn build_single_trait_output_group(
    phenotype_name: &str,
    aligned_sample_data: &native_input::AlignedSampleData,
    phenotype_compute_group: &native_input::ResolvedPhenotypeComputeGroup,
    sample_count: i64,
) -> PyResult<NativeOutputRuntimeGroupInput> {
    let phenotype_indices = phenotype_compute_group
        .phenotype_indices
        .iter()
        .copied()
        .map(|index| usize_to_i64(index, "Phenotype compute group index"))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((
        vec![phenotype_name.to_string()],
        aligned_sample_data.covariate_names.clone(),
        sample_count,
        "single-phenotype".to_string(),
        Some(phenotype_compute_group.group_mode.clone()),
        Some(phenotype_indices),
        Some(phenotype_compute_group.phenotype_names.clone()),
        Some(phenotype_compute_group.sample_mode.clone()),
        Some(phenotype_compute_group.sample_set_fingerprint.clone()),
        Some(phenotype_compute_group.covariate_design_fingerprint.clone()),
        phenotype_compute_group.prediction_alignment_fingerprint.clone(),
    ))
}

fn build_multi_trait_output_group(
    covariate_names: &[String],
    sample_count: i64,
    output_sample_mode: native_plan::MultiPhenotypeSampleMode,
    phenotype_compute_group: &native_input::ResolvedPhenotypeComputeGroup,
) -> PyResult<NativeOutputRuntimeGroupInput> {
    let phenotype_indices = phenotype_compute_group
        .phenotype_indices
        .iter()
        .copied()
        .map(|index| usize_to_i64(index, "Phenotype compute group index"))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((
        phenotype_compute_group.phenotype_names.clone(),
        covariate_names.to_vec(),
        sample_count,
        output_sample_mode.as_str().to_string(),
        Some(phenotype_compute_group.group_mode.clone()),
        Some(phenotype_indices),
        Some(phenotype_compute_group.phenotype_names.clone()),
        Some(phenotype_compute_group.sample_mode.clone()),
        Some(phenotype_compute_group.sample_set_fingerprint.clone()),
        Some(phenotype_compute_group.covariate_design_fingerprint.clone()),
        phenotype_compute_group.prediction_alignment_fingerprint.clone(),
    ))
}

fn callback_context_from_request(
    run_request: &native_plan::RunRequest,
    resolved_execution: &NativeRunResolvedExecution,
) -> PyResult<NativeRunCallbackContext> {
    Ok(NativeRunCallbackContext {
        association_mode: run_request.association_mode.as_str().to_string(),
        trait_type: run_request.trait_request.trait_type.as_str().to_string(),
        correction_method: run_request.correction.method.as_str().to_string(),
        correction_p_threshold: run_request.correction.p_threshold,
        correction_firth_se: run_request.correction.firth_se,
        staging_depth: i64::from(run_request.compute.staging_depth),
        native_callback_batch_size: i64::from(run_request.compute.native_callback_batch_size),
        result_in_flight_limit: run_request.compute.result_in_flight_limit.map(i64::from),
        dosage_buffer_limit: run_request.compute.dosage_buffer_limit.map(i64::from),
        score_dtype: run_request.compute.score_dtype.as_str().to_string(),
        firth_dtype: run_request.compute.firth_dtype.as_str().to_string(),
        output_statistic_dtype: run_request.output.output_statistic_dtype.as_str().to_string(),
        jax_device: run_request.compute.device.as_str().to_string(),
        gpu_genotype_format: resolved_execution.resolved_gpu_genotype_format.clone(),
        requested_gpu_genotype_format: resolved_execution.requested_gpu_genotype_format.clone(),
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

fn flush_variant_major_dosage_batch_native(
    resources: &Bound<'_, PyAny>,
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
    resources.call_method1(
        "enqueue_variant_major_dosage_chunk_batch",
        (metadata_values, output_array_values, stats_values),
    )?;
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

fn sample_identifier_data_for_engine(
    engine: &Regenie2RunEngineCore,
    py: Python<'_>,
    sample_path: Option<String>,
) -> PyResult<native_input::SampleIdentifierData> {
    if let Some(sample_path) = sample_path {
        let expected_sample_count = engine.reader().sample_count();
        return py
            .detach(move || {
                native_input::load_sample_identifier_data_from_sample_file(
                    Path::new(&sample_path),
                    expected_sample_count,
                )
            })
            .map_err(|error| convert_input_error("load_sample_identifier_data_from_sample_file", error.into()));
    }
    if !engine.reader().contains_embedded_samples() {
        return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
    }
    let sample_identifiers = engine.reader().sample_identifiers();
    let sample_indices = (0..sample_identifiers.len()).collect::<Vec<_>>();
    Ok(native_input::SampleIdentifierData {
        sample_indices,
        family_identifiers: sample_identifiers.clone(),
        individual_identifiers: sample_identifiers,
    })
}

#[allow(clippy::too_many_arguments)]
fn align_sample_data_for_engine(
    engine: &Regenie2RunEngineCore,
    py: Python<'_>,
    sample_path: Option<String>,
    phenotype_path: String,
    phenotype_name: String,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: &str,
) -> PyResult<NativeAlignedSampleData> {
    let parsed_sample_key_mode = parse_sample_key_mode(sample_key_mode)?;
    let sample_identifier_data = sample_identifier_data_for_engine(engine, py, sample_path)?;
    let inputs = AlignmentInputs {
        sample_indices: sample_identifier_data.sample_indices,
        family_identifiers: sample_identifier_data.family_identifiers,
        individual_identifiers: sample_identifier_data.individual_identifiers,
        phenotype_path,
        phenotype_name,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parsed_sample_key_mode,
    };
    py.detach(move || native_input::align_sample_data(inputs))
        .map(NativeAlignedSampleData::new)
        .map_err(|error| convert_input_error("align_sample_data", error.into()))
}

#[allow(clippy::too_many_arguments)]
fn align_multi_sample_data_for_engine(
    engine: &Regenie2RunEngineCore,
    py: Python<'_>,
    sample_path: Option<String>,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: &str,
) -> PyResult<NativeMultiAlignedSampleData> {
    let parsed_sample_key_mode = parse_sample_key_mode(sample_key_mode)?;
    let sample_identifier_data = sample_identifier_data_for_engine(engine, py, sample_path)?;
    let inputs = MultiAlignmentInputs {
        sample_indices: sample_identifier_data.sample_indices,
        family_identifiers: sample_identifier_data.family_identifiers,
        individual_identifiers: sample_identifier_data.individual_identifiers,
        phenotype_path,
        phenotype_names,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parsed_sample_key_mode,
    };
    py.detach(move || native_input::align_multi_sample_data(inputs))
        .map(NativeMultiAlignedSampleData::new)
        .map_err(|error| convert_input_error("align_multi_sample_data", error.into()))
}

#[allow(clippy::too_many_arguments)]
fn align_grouped_sample_data_for_engine(
    engine: &Regenie2RunEngineCore,
    py: Python<'_>,
    sample_path: Option<String>,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: &str,
) -> PyResult<NativeGroupedAlignedSampleData> {
    let parsed_sample_key_mode = parse_sample_key_mode(sample_key_mode)?;
    let sample_identifier_data = sample_identifier_data_for_engine(engine, py, sample_path)?;
    let inputs = MultiAlignmentInputs {
        sample_indices: sample_identifier_data.sample_indices,
        family_identifiers: sample_identifier_data.family_identifiers,
        individual_identifiers: sample_identifier_data.individual_identifiers,
        phenotype_path,
        phenotype_names,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode: parsed_sample_key_mode,
    };
    py.detach(move || native_input::align_grouped_sample_data(&inputs))
        .map(NativeGroupedAlignedSampleData::new)
        .map_err(|error| convert_input_error("align_grouped_sample_data", error.into()))
}

fn plan_bgen_delivery_attempt_for_binding(
    callback_batch_size: Option<i64>,
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
    committed_chunk_count: usize,
) -> PyResult<native_engine_debug::BgenDeliveryAttemptPlan> {
    native_engine_debug::plan_bgen_delivery_attempt(
        callback_batch_size,
        variant_major_packed8_probability_pairs,
        has_native_multi_aligned_sample_data,
        has_native_aligned_sample_data,
        committed_chunk_count,
    )
    .map_err(|error| convert_schedule_error(&error))
}

#[allow(clippy::too_many_arguments)]
fn run_bgen_delivery_invocation<'py>(
    engine: &Regenie2RunEngineCore,
    py: Python<'py>,
    sample_indices: &PyReadonlyArray1<'py, i64>,
    native_aligned_sample_data: Option<PyRef<'py, NativeAlignedSampleData>>,
    native_multi_aligned_sample_data: Option<PyRef<'py, NativeMultiAlignedSampleData>>,
    callback: &Bound<'py, PyAny>,
    committed_chunk_identifiers: Option<Vec<usize>>,
    invocation_plan: native_engine_debug::BgenDeliveryInvocationPlan,
) -> PyResult<usize> {
    match invocation_plan.delivery_method {
        native_engine_debug::BgenDeliveryMethod::DosageNativeMultiAlignedSamples => {
            let aligned_sample_data = native_multi_aligned_sample_data.ok_or_else(|| {
                PyRuntimeError::new_err("Native BGEN delivery plan selected missing multi-aligned sample data.")
            })?;
            run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
                engine,
                py,
                &aligned_sample_data.data.sample_indices,
                callback,
                committed_chunk_identifiers,
                invocation_plan.callback_batch_size,
            )
        }
        native_engine_debug::BgenDeliveryMethod::DosageNativeAlignedSamples => {
            let aligned_sample_data = native_aligned_sample_data.ok_or_else(|| {
                PyRuntimeError::new_err("Native BGEN delivery plan selected missing aligned sample data.")
            })?;
            run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
                engine,
                py,
                &aligned_sample_data.data.sample_indices,
                callback,
                committed_chunk_identifiers,
                invocation_plan.callback_batch_size,
            )
        }
        native_engine_debug::BgenDeliveryMethod::DosageSampleIndices => {
            let sample_index_values = py_i64_slice_to_usize(sample_indices.as_slice()?, "sample_indices")?;
            run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
                engine,
                py,
                &sample_index_values,
                callback,
                committed_chunk_identifiers,
                invocation_plan.callback_batch_size,
            )
        }
        native_engine_debug::BgenDeliveryMethod::Packed8NativeMultiAlignedSamples => {
            let aligned_sample_data = native_multi_aligned_sample_data.ok_or_else(|| {
                PyRuntimeError::new_err("Native BGEN delivery plan selected missing multi-aligned sample data.")
            })?;
            run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
                engine,
                py,
                &aligned_sample_data.data.sample_indices,
                callback,
                committed_chunk_identifiers,
            )
        }
        native_engine_debug::BgenDeliveryMethod::Packed8NativeAlignedSamples => {
            let aligned_sample_data = native_aligned_sample_data.ok_or_else(|| {
                PyRuntimeError::new_err("Native BGEN delivery plan selected missing aligned sample data.")
            })?;
            run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
                engine,
                py,
                &aligned_sample_data.data.sample_indices,
                callback,
                committed_chunk_identifiers,
            )
        }
        native_engine_debug::BgenDeliveryMethod::Packed8SampleIndices => {
            let sample_index_values = py_i64_slice_to_usize(sample_indices.as_slice()?, "sample_indices")?;
            run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
                engine,
                py,
                &sample_index_values,
                callback,
                committed_chunk_identifiers,
            )
        }
    }
}

fn run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices<'py>(
    engine: &Regenie2RunEngineCore,
    py: Python<'py>,
    sample_indices: &[usize],
    callback: &Bound<'py, PyAny>,
    committed_chunk_identifiers: Option<Vec<usize>>,
    callback_batch_size: usize,
) -> PyResult<usize> {
    if callback_batch_size == 0 {
        return Err(PyValueError::new_err("callback_batch_size must be positive."));
    }
    py.detach(|| engine.reader().prepare_sample_selection(sample_indices))
        .map_err(|error| convert_bgen_error("prepare_sample_selection", error))?;

    let run_result = run_prepared_bgen_variant_major_dosage_buffered_chunks(
        engine,
        py,
        sample_indices.len(),
        callback,
        committed_chunk_identifiers,
        callback_batch_size,
    );
    let clear_result = py
        .detach(|| engine.reader().clear_prepared_sample_selection())
        .map_err(|error| convert_bgen_error("clear_prepared_sample_selection", error));
    match (run_result, clear_result) {
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
    }
}

fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices<'py>(
    engine: &Regenie2RunEngineCore,
    py: Python<'py>,
    sample_indices: &[usize],
    callback: &Bound<'py, PyAny>,
    committed_chunk_identifiers: Option<Vec<usize>>,
) -> PyResult<usize> {
    py.detach(|| engine.reader().prepare_sample_selection(sample_indices))
        .map_err(|error| convert_bgen_error("prepare_sample_selection", error))?;

    let run_result = run_prepared_bgen_variant_major_packed8_probability_pair_buffered_chunks(
        engine,
        py,
        sample_indices.len(),
        callback,
        committed_chunk_identifiers,
    );
    let clear_result = py
        .detach(|| engine.reader().clear_prepared_sample_selection())
        .map_err(|error| convert_bgen_error("clear_prepared_sample_selection", error));
    match (run_result, clear_result) {
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
    }
}

fn run_prepared_bgen_variant_major_dosage_buffered_chunks<'py>(
    engine: &Regenie2RunEngineCore,
    py: Python<'py>,
    selected_sample_count: usize,
    callback: &Bound<'py, PyAny>,
    committed_chunk_identifiers: Option<Vec<usize>>,
    callback_batch_size: usize,
) -> PyResult<usize> {
    let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
    let chunk_specs =
        engine.plan_chunks(&committed_identifier_set).map_err(|error| convert_genotype_error("plan_chunks", error))?;
    if callback_batch_size > 1 {
        return run_prepared_bgen_variant_major_dosage_buffered_chunk_batches(
            engine,
            py,
            selected_sample_count,
            callback,
            &chunk_specs,
            callback_batch_size,
        );
    }
    let chunk_batch_plan = native_engine_debug::plan_chunk_batches(&chunk_specs, callback_batch_size)
        .map_err(|error| convert_schedule_error(&error))?;
    let processed_chunk_count = chunk_batch_plan.chunk_count();
    let resources = optional_callback_runtime_resources(callback)?;
    let acquire_dosage_buffer_method = if resources.is_none() {
        Some(callback.getattr("acquire_variant_major_dosage_buffer")?)
    } else {
        None
    };
    let compute_dosage_chunk_method = if resources.is_none() {
        Some(callback.getattr("compute_preprocessed_variant_major_dosage_chunk")?)
    } else {
        None
    };
    for chunk_batch in chunk_batch_plan.into_chunk_batches() {
        for chunk_spec in &chunk_batch {
            py.check_signals()?;
            let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
            let output_array_object = if let Some(resources) = resources.as_ref() {
                resources.call_method1(
                    "acquire_variant_major_dosage_buffer",
                    (selected_variant_count, selected_sample_count),
                )?
            } else {
                acquire_dosage_buffer_method
                    .as_ref()
                    .expect("grouped acquire method")
                    .call1((selected_variant_count, selected_sample_count))?
            };
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
                let output_buffer_address = OutputBufferAddress::from_mut_ptr(output_slice.as_mut_ptr());
                let output_value_count = OutputValueCount::new(output_slice.len());
                let chunk_stats = py
                    .detach(|| {
                        engine.reader().read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                            chunk_spec.variant_start_index,
                            chunk_spec.variant_stop_index,
                            output_buffer_address,
                            output_value_count,
                        )
                    })
                    .map_err(|error| {
                        convert_bgen_error("read_preprocessed_variant_major_dosage_f32_into_address_prepared", error)
                    })?;
                Py::new(py, ChunkStats::new(chunk_stats))?
            };
            let variant_start_index = chunk_spec.variant_start_index;
            let variant_stop_index = chunk_spec.variant_stop_index;
            let metadata_columns = py
                .detach(|| engine.reader().variant_metadata_slice(variant_start_index, variant_stop_index))
                .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
            let metadata =
                Py::new(py, VariantMetadata::new(variant_start_index, variant_stop_index, metadata_columns))?;
            if let Some(resources) = resources.as_ref() {
                resources.call_method1(
                    "enqueue_variant_major_dosage_chunk",
                    (metadata, output_array_object, stats),
                )?;
            } else {
                compute_dosage_chunk_method
                    .as_ref()
                    .expect("grouped compute method")
                    .call1((metadata, output_array_object, stats))?;
            }
        }
    }
    Ok(processed_chunk_count)
}

fn optional_callback_runtime_resources<'py>(callback: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if callback.hasattr("callback_runtime_resources")? {
        return Ok(Some(callback.getattr("callback_runtime_resources")?));
    }
    Ok(None)
}

fn run_prepared_bgen_variant_major_dosage_buffered_chunk_batches<'py>(
    engine: &Regenie2RunEngineCore,
    py: Python<'py>,
    selected_sample_count: usize,
    callback: &Bound<'py, PyAny>,
    chunk_specs: &[NativeChunkSpec],
    callback_batch_size: usize,
) -> PyResult<usize> {
    let resources = optional_callback_runtime_resources(callback)?;
    let acquire_dosage_buffer_method = if resources.is_none() {
        Some(callback.getattr("acquire_variant_major_dosage_buffer")?)
    } else {
        None
    };
    let compute_dosage_chunk_batch_method = if resources.is_none() {
        Some(callback.getattr("compute_preprocessed_variant_major_dosage_chunk_batch")?)
    } else {
        None
    };
    let chunk_batch_plan = native_engine_debug::plan_chunk_batches(chunk_specs, callback_batch_size)
        .map_err(|error| convert_schedule_error(&error))?;
    let processed_chunk_count = chunk_batch_plan.chunk_count();
    let mut metadata_batch: Vec<Py<VariantMetadata>> = Vec::with_capacity(callback_batch_size);
    let mut output_array_batch: Vec<Py<PyAny>> = Vec::with_capacity(callback_batch_size);
    let mut stats_batch: Vec<Py<ChunkStats>> = Vec::with_capacity(callback_batch_size);
    for chunk_batch in chunk_batch_plan.into_chunk_batches() {
        for chunk_spec in &chunk_batch {
            py.check_signals()?;
            let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
            let output_array_object = if let Some(resources) = resources.as_ref() {
                resources.call_method1(
                    "acquire_variant_major_dosage_buffer",
                    (selected_variant_count, selected_sample_count),
                )?
            } else {
                acquire_dosage_buffer_method
                    .as_ref()
                    .expect("grouped acquire method")
                    .call1((selected_variant_count, selected_sample_count))?
            };
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
                let output_buffer_address = OutputBufferAddress::from_mut_ptr(output_slice.as_mut_ptr());
                let output_value_count = OutputValueCount::new(output_slice.len());
                let chunk_stats = py
                    .detach(|| {
                        engine.reader().read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                            chunk_spec.variant_start_index,
                            chunk_spec.variant_stop_index,
                            output_buffer_address,
                            output_value_count,
                        )
                    })
                    .map_err(|error| {
                        convert_bgen_error("read_preprocessed_variant_major_dosage_f32_into_address_prepared", error)
                    })?;
                Py::new(py, ChunkStats::new(chunk_stats))?
            };
            let variant_start_index = chunk_spec.variant_start_index;
            let variant_stop_index = chunk_spec.variant_stop_index;
            let metadata_columns = py
                .detach(|| engine.reader().variant_metadata_slice(variant_start_index, variant_stop_index))
                .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
            let metadata =
                Py::new(py, VariantMetadata::new(variant_start_index, variant_stop_index, metadata_columns))?;
            metadata_batch.push(metadata);
            output_array_batch.push(output_array_object.unbind());
            stats_batch.push(stats);
        }
        if let Some(resources) = resources.as_ref() {
            flush_variant_major_dosage_batch_native(
                resources,
                &mut metadata_batch,
                &mut output_array_batch,
                &mut stats_batch,
            )?;
        } else {
            flush_variant_major_dosage_batch(
                compute_dosage_chunk_batch_method.as_ref().expect("grouped batch compute"),
                &mut metadata_batch,
                &mut output_array_batch,
                &mut stats_batch,
            )?;
        }
    }
    Ok(processed_chunk_count)
}

fn run_prepared_bgen_variant_major_packed8_probability_pair_buffered_chunks<'py>(
    engine: &Regenie2RunEngineCore,
    py: Python<'py>,
    selected_sample_count: usize,
    callback: &Bound<'py, PyAny>,
    committed_chunk_identifiers: Option<Vec<usize>>,
) -> PyResult<usize> {
    let committed_identifier_set = build_committed_identifier_set(committed_chunk_identifiers);
    let chunk_specs =
        engine.plan_chunks(&committed_identifier_set).map_err(|error| convert_genotype_error("plan_chunks", error))?;
    let chunk_batch_plan =
        native_engine_debug::plan_chunk_batches(&chunk_specs, 1).map_err(|error| convert_schedule_error(&error))?;
    let processed_chunk_count = chunk_batch_plan.chunk_count();
    let resources = optional_callback_runtime_resources(callback)?;
    let acquire_packed_buffer_method = if resources.is_none() {
        Some(callback.getattr("acquire_variant_major_packed8_probability_pair_buffer")?)
    } else {
        None
    };
    let compute_packed_chunk_method = if resources.is_none() {
        Some(callback.getattr("compute_preprocessed_variant_major_packed8_probability_pair_chunk")?)
    } else {
        None
    };
    for chunk_batch in chunk_batch_plan.into_chunk_batches() {
        for chunk_spec in &chunk_batch {
            py.check_signals()?;
            let selected_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
            let output_array_object = if let Some(resources) = resources.as_ref() {
                resources.call_method1(
                    "acquire_variant_major_packed8_probability_pair_buffer",
                    (selected_variant_count, selected_sample_count),
                )?
            } else {
                acquire_packed_buffer_method
                    .as_ref()
                    .expect("grouped packed acquire")
                    .call1((selected_variant_count, selected_sample_count))?
            };
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
                let output_buffer_address = OutputBufferAddress::from_mut_ptr(output_slice.as_mut_ptr());
                let output_value_count = OutputValueCount::new(output_slice.len());
                let chunk_stats = py
                    .detach(|| {
                        engine.reader().read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
                            chunk_spec.variant_start_index,
                            chunk_spec.variant_stop_index,
                            output_buffer_address,
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
                .detach(|| engine.reader().variant_metadata_slice(variant_start_index, variant_stop_index))
                .map_err(|error| convert_bgen_error("variant_metadata_slice", error))?;
            let metadata =
                Py::new(py, VariantMetadata::new(variant_start_index, variant_stop_index, metadata_columns))?;
            if let Some(resources) = resources.as_ref() {
                resources.call_method1(
                    "enqueue_variant_major_packed8_probability_pair_chunk",
                    (metadata, output_array_object, stats),
                )?;
            } else {
                compute_packed_chunk_method
                    .as_ref()
                    .expect("grouped packed compute")
                    .call1((metadata, output_array_object, stats))?;
            }
        }
    }
    Ok(processed_chunk_count)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRunCallbackContext>()?;
    module.add_class::<NativeRunEngineSession>()?;
    Ok(())
}
