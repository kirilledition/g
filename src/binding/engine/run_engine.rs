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
use super::preflight;
use super::profile::build_profile_snapshot_dict;
use super::run_events::{self, NativeRunArtifacts};
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

#[pyclass]
struct Regenie2RunEngine {
    engine: Regenie2RunEngineCore,
}

#[pyclass(name = "NativeSingleTraitRunInput", skip_from_py_object)]
struct NativeSingleTraitRunInput {
    data: native_input::AlignedSampleData,
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

    #[getter]
    fn phase(&self) -> PyResult<&'static str> {
        self.lifecycle.phase_label()
    }

    #[getter]
    fn output_resume(&self) -> bool {
        self.lifecycle.output_resume_value()
    }

    #[getter]
    fn run_request(&self) -> NativeRunRequest {
        self.lifecycle.run_request_handle()
    }

    #[getter]
    fn sample_count(&self) -> PyResult<usize> {
        self.with_open_engine(|engine| Ok(engine.reader().sample_count()))
    }

    #[getter]
    fn variant_count(&self) -> PyResult<usize> {
        self.with_open_engine(|engine| Ok(engine.reader().variant_count()))
    }

    #[getter]
    fn contains_embedded_samples(&self) -> PyResult<bool> {
        self.with_open_engine(|engine| Ok(engine.reader().contains_embedded_samples()))
    }

    fn prepared_phenotype_runs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.lifecycle.prepared_phenotype_runs_tuple(py)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn prepared_phenotype_run(&self, phenotype_name: String) -> PyResult<NativeRunLifecyclePhenotypeRun> {
        self.lifecycle.prepared_phenotype_run_handle(phenotype_name)
    }

    fn mark_dispatch_started(&self) -> PyResult<()> {
        self.lifecycle.mark_dispatch_started_internal()
    }

    fn has_open_bgen_engine(&self) -> PyResult<bool> {
        Ok(self.lock_engine()?.is_some())
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        bgen_path,
        chunk_size,
        variant_limit=None,
        trusted_no_missing_diploid=false,
        trusted_bgen_validation_mode=None,
    ))]
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

    #[pyo3(signature = (variant_limit=None))]
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

    #[allow(clippy::needless_pass_by_value)]
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

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    #[allow(clippy::needless_pass_by_value)]
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

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::needless_pass_by_value)]
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
        stage_timing_recorder: Option<PyRef<'py, NativeStageTimingRecorder>>,
    ) -> PyResult<NativeSingleTraitPipelineBundle> {
        let stage_timing_recorder_reference = stage_timing_recorder.as_deref();
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
        let preflight_shape = preflight::validate_single_trait_preflight_values(
            &aligned_sample_data.data.phenotype_vector,
            aligned_sample_data.data.covariate_row_count,
            aligned_sample_data.data.covariate_column_count,
            &aligned_sample_data.data.covariate_matrix_values,
            is_binary_trait,
        )?;
        let required_chromosomes = self.with_open_engine(|engine| {
            engine.required_chromosomes(variant_limit).map_err(|error| convert_preflight_error(&error))
        })?;
        for chromosome in &required_chromosomes {
            let prediction_values = prediction_source
                .source
                .chromosome_predictions(chromosome)
                .map_err(|error| convert_prediction_error("chromosome_predictions", &error))?;
            preflight::validate_single_prediction_values(chromosome, prediction_values, preflight_shape.sample_count)?;
        }
        let chromosome_count = usize_to_i64(required_chromosomes.len(), "Chromosome count")?;
        let preflight_report = preflight::build_preflight_report(
            preflight_shape.sample_count,
            preflight_shape.covariate_count,
            chromosome_count,
            effective_trusted_no_missing_diploid,
        )?;
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

    #[allow(clippy::too_many_arguments)]
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
        let run_input_handle = bundle.run_input.clone_ref(py);
        let writer_session_handle = bundle.writer_session.clone_ref(py);
        let committed_chunk_identifiers = bundle.committed_chunk_identifiers.clone();
        drop(bundle);

        let run_input_data = run_input_handle.bind(py).borrow().data.clone();
        let sample_indices_array =
            usize_slice_to_py_i64(&run_input_data.sample_indices, "sample_indices")?.into_pyarray(py);
        let sample_indices = sample_indices_array.readonly();
        let native_aligned_sample_data = Py::new(py, NativeAlignedSampleData::new(run_input_data))?;
        let native_aligned_sample_data_reference = native_aligned_sample_data.bind(py).borrow();
        let writer_session_reference = writer_session_handle.bind(py).borrow();
        let writer_sessions = vec![writer_session_reference];
        let stage_timing_recorder_reference = stage_timing_recorder.as_deref();
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
            stage_timing_recorder_reference,
            1,
            Some(committed_chunk_identifiers),
            variant_major_packed8_probability_pairs,
            &pipeline_label,
            &mut callback_finished,
        );
        match delivery_result {
            Ok(final_parquet_paths) => {
                run_events::record_native_dispatch_pipeline_finished_diagnostic_event(
                    usize_to_i64(final_parquet_paths.len(), "Final Parquet path count")?,
                    &pipeline_label,
                )?;
                Ok(final_parquet_paths.into_iter().next().flatten())
            }
            Err(error) => handle_bgen_delivery_error(
                py,
                error,
                callback_finished,
                callback,
                &writer_sessions,
                stage_timing_recorder_reference,
                1,
                &pipeline_label,
            )
            .map(|final_parquet_paths| final_parquet_paths.into_iter().next().flatten()),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn finalize_success(&self, final_output_paths: Vec<Option<String>>) -> PyResult<NativeRunArtifacts> {
        self.lifecycle.finalize_success_artifacts(final_output_paths)
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
        Ok(Self {
            engine: open_bgen_engine_core(py, &bgen_path, chunk_size, variant_limit, trusted_no_missing_diploid)?,
        })
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
        align_sample_data_for_engine(
            &self.engine,
            py,
            sample_path,
            phenotype_path,
            phenotype_name,
            covariate_path,
            covariate_names,
            is_binary_trait,
            &sample_key_mode,
        )
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
        align_multi_sample_data_for_engine(
            &self.engine,
            py,
            sample_path,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            &sample_key_mode,
        )
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
        align_grouped_sample_data_for_engine(
            &self.engine,
            py,
            sample_path,
            phenotype_path,
            phenotype_names,
            covariate_path,
            covariate_names,
            is_binary_trait,
            &sample_key_mode,
        )
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
        self.engine.required_chromosomes(variant_limit).map_err(|error| convert_preflight_error(&error))
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
    fn validate_trusted_no_missing_diploid_with_default_cache(
        &self,
        py: Python<'_>,
        bgen_path: String,
        validation_mode: String,
    ) -> PyResult<()> {
        validate_trusted_no_missing_diploid_with_default_cache_for_engine(
            &self.engine,
            py,
            &bgen_path,
            &validation_mode,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_indices,
        native_aligned_sample_data,
        native_multi_aligned_sample_data,
        callback,
        committed_chunk_identifiers=None,
        callback_batch_size=1,
    ))]
    fn run_bgen_variant_major_dosage_buffered_chunks_for_best_sample_source<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        native_aligned_sample_data: Option<PyRef<'py, NativeAlignedSampleData>>,
        native_multi_aligned_sample_data: Option<PyRef<'py, NativeMultiAlignedSampleData>>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: i64,
    ) -> PyResult<usize> {
        run_bgen_variant_major_dosage_buffered_chunks_for_best_sample_source(
            &self.engine,
            py,
            &sample_indices,
            native_aligned_sample_data,
            native_multi_aligned_sample_data,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        sample_indices,
        native_aligned_sample_data,
        native_multi_aligned_sample_data,
        callback,
        committed_chunk_identifiers=None,
        callback_batch_size=1,
    ))]
    fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_best_sample_source<'py>(
        &self,
        py: Python<'py>,
        sample_indices: PyReadonlyArray1<'py, i64>,
        native_aligned_sample_data: Option<PyRef<'py, NativeAlignedSampleData>>,
        native_multi_aligned_sample_data: Option<PyRef<'py, NativeMultiAlignedSampleData>>,
        callback: &Bound<'py, PyAny>,
        committed_chunk_identifiers: Option<Vec<usize>>,
        callback_batch_size: i64,
    ) -> PyResult<usize> {
        run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_best_sample_source(
            &self.engine,
            py,
            &sample_indices,
            native_aligned_sample_data,
            native_multi_aligned_sample_data,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
fn run_bgen_delivery_with_writer_sessions<'py>(
    py: Python<'py>,
    engine: PyRef<'py, Regenie2RunEngine>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    native_aligned_sample_data: Option<PyRef<'py, NativeAlignedSampleData>>,
    native_multi_aligned_sample_data: Option<PyRef<'py, NativeMultiAlignedSampleData>>,
    writer_sessions: Vec<PyRef<'py, OutputWriterSession>>,
    callback: &Bound<'py, PyAny>,
    stage_timing_recorder: Option<PyRef<'py, NativeStageTimingRecorder>>,
    writer_finish_thread_count: i64,
    committed_chunk_identifiers: Option<Vec<usize>>,
    variant_major_packed8_probability_pairs: bool,
    pipeline_label: String,
) -> PyResult<Vec<Option<String>>> {
    let stage_timing_recorder_reference = stage_timing_recorder.as_deref();
    let mut callback_finished = false;
    let delivery_result = run_bgen_delivery_attempt(
        py,
        &engine.engine,
        &sample_indices,
        native_aligned_sample_data,
        native_multi_aligned_sample_data,
        &writer_sessions,
        callback,
        stage_timing_recorder_reference,
        writer_finish_thread_count,
        committed_chunk_identifiers,
        variant_major_packed8_probability_pairs,
        &pipeline_label,
        &mut callback_finished,
    );
    match delivery_result {
        Ok(final_parquet_paths) => {
            run_events::record_native_dispatch_pipeline_finished_diagnostic_event(
                usize_to_i64(final_parquet_paths.len(), "Final Parquet path count")?,
                &pipeline_label,
            )?;
            Ok(final_parquet_paths)
        }
        Err(error) => handle_bgen_delivery_error(
            py,
            error,
            callback_finished,
            callback,
            &writer_sessions,
            stage_timing_recorder_reference,
            writer_finish_thread_count,
            &pipeline_label,
        ),
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
fn run_bgen_session_delivery_with_writer_sessions<'py>(
    py: Python<'py>,
    engine_session: PyRef<'py, NativeRunEngineSession>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    native_aligned_sample_data: Option<PyRef<'py, NativeAlignedSampleData>>,
    native_multi_aligned_sample_data: Option<PyRef<'py, NativeMultiAlignedSampleData>>,
    writer_sessions: Vec<PyRef<'py, OutputWriterSession>>,
    callback: &Bound<'py, PyAny>,
    stage_timing_recorder: Option<PyRef<'py, NativeStageTimingRecorder>>,
    writer_finish_thread_count: i64,
    committed_chunk_identifiers: Option<Vec<usize>>,
    variant_major_packed8_probability_pairs: bool,
    pipeline_label: String,
) -> PyResult<Vec<Option<String>>> {
    let stage_timing_recorder_reference = stage_timing_recorder.as_deref();
    let engine_guard = engine_session.lock_engine()?;
    let engine = engine_guard
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("Native run engine session has no open BGEN engine."))?;
    let mut callback_finished = false;
    let delivery_result = run_bgen_delivery_attempt(
        py,
        engine,
        &sample_indices,
        native_aligned_sample_data,
        native_multi_aligned_sample_data,
        &writer_sessions,
        callback,
        stage_timing_recorder_reference,
        writer_finish_thread_count,
        committed_chunk_identifiers,
        variant_major_packed8_probability_pairs,
        &pipeline_label,
        &mut callback_finished,
    );
    match delivery_result {
        Ok(final_parquet_paths) => {
            run_events::record_native_dispatch_pipeline_finished_diagnostic_event(
                usize_to_i64(final_parquet_paths.len(), "Final Parquet path count")?,
                &pipeline_label,
            )?;
            Ok(final_parquet_paths)
        }
        Err(error) => handle_bgen_delivery_error(
            py,
            error,
            callback_finished,
            callback,
            &writer_sessions,
            stage_timing_recorder_reference,
            writer_finish_thread_count,
            &pipeline_label,
        ),
    }
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
    run_events::record_native_dispatch_delivery_started_diagnostic_event(
        usize_to_i64(committed_chunk_count, "Committed chunk count")?,
        pipeline_label,
        variant_major_packed8_probability_pairs,
    )?;
    callback.call_method0("start")?;
    let callback_batch_size = callback.getattr("native_callback_batch_size")?.extract::<i64>()?;
    let processed_chunk_count = if variant_major_packed8_probability_pairs {
        run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_best_sample_source(
            engine,
            py,
            sample_indices,
            native_aligned_sample_data,
            native_multi_aligned_sample_data,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )?
    } else {
        run_bgen_variant_major_dosage_buffered_chunks_for_best_sample_source(
            engine,
            py,
            sample_indices,
            native_aligned_sample_data,
            native_multi_aligned_sample_data,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )?
    };
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
        run_events::record_native_dispatch_delivery_interrupted_diagnostic_event(
            pipeline_label,
            interrupted_event.exit_code,
            &interrupted_event.signal_name,
            interrupted_event.signal_number,
        )?;
        let cleanup_result = execute_bgen_delivery_cleanup_actions(
            py,
            native_engine_debug::BgenDeliveryCleanupOutcome::Interrupted,
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
                execute_bgen_delivery_cleanup_actions(
                    py,
                    native_engine_debug::BgenDeliveryCleanupOutcome::InterruptedCleanupFailure,
                    callback_finished,
                    callback,
                    writer_sessions,
                    writer_finish_thread_count,
                    stage_timing_recorder,
                    Some(&interrupted_event),
                )?;
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
    let cleanup_result = execute_bgen_delivery_cleanup_actions(
        py,
        native_engine_debug::BgenDeliveryCleanupOutcome::Failure,
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
                callback.call_method0("finish")?;
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
                let _ = callback.call_method0("abort");
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

#[allow(clippy::too_many_arguments)]
fn run_bgen_variant_major_dosage_buffered_chunks_for_best_sample_source<'py>(
    engine: &Regenie2RunEngineCore,
    py: Python<'py>,
    sample_indices: &PyReadonlyArray1<'py, i64>,
    native_aligned_sample_data: Option<PyRef<'py, NativeAlignedSampleData>>,
    native_multi_aligned_sample_data: Option<PyRef<'py, NativeMultiAlignedSampleData>>,
    callback: &Bound<'py, PyAny>,
    committed_chunk_identifiers: Option<Vec<usize>>,
    callback_batch_size: i64,
) -> PyResult<usize> {
    let invocation_plan = native_engine_debug::plan_bgen_delivery_invocation(
        Some(callback_batch_size),
        false,
        native_multi_aligned_sample_data.is_some(),
        native_aligned_sample_data.is_some(),
    )
    .map_err(|error| convert_schedule_error(&error))?;
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
        _ => Err(PyRuntimeError::new_err("Native BGEN delivery plan selected a packed8 method for dosage.")),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_best_sample_source<'py>(
    engine: &Regenie2RunEngineCore,
    py: Python<'py>,
    sample_indices: &PyReadonlyArray1<'py, i64>,
    native_aligned_sample_data: Option<PyRef<'py, NativeAlignedSampleData>>,
    native_multi_aligned_sample_data: Option<PyRef<'py, NativeMultiAlignedSampleData>>,
    callback: &Bound<'py, PyAny>,
    committed_chunk_identifiers: Option<Vec<usize>>,
    callback_batch_size: i64,
) -> PyResult<usize> {
    let invocation_plan = native_engine_debug::plan_bgen_delivery_invocation(
        Some(callback_batch_size),
        true,
        native_multi_aligned_sample_data.is_some(),
        native_aligned_sample_data.is_some(),
    )
    .map_err(|error| convert_schedule_error(&error))?;
    match invocation_plan.delivery_method {
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
        _ => Err(PyRuntimeError::new_err("Native BGEN delivery plan selected a dosage method for packed8.")),
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
    let acquire_dosage_buffer_method = callback.getattr("acquire_variant_major_dosage_buffer")?;
    if callback_batch_size > 1 {
        return run_prepared_bgen_variant_major_dosage_buffered_chunk_batches(
            engine,
            py,
            selected_sample_count,
            callback,
            &chunk_specs,
            &acquire_dosage_buffer_method,
            callback_batch_size,
        );
    }
    let chunk_batch_plan = native_engine_debug::plan_chunk_batches(&chunk_specs, callback_batch_size)
        .map_err(|error| convert_schedule_error(&error))?;
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
            compute_dosage_chunk_method.call1((metadata, output_array_object, stats))?;
        }
    }
    Ok(processed_chunk_count)
}

fn run_prepared_bgen_variant_major_dosage_buffered_chunk_batches<'py>(
    engine: &Regenie2RunEngineCore,
    py: Python<'py>,
    selected_sample_count: usize,
    callback: &Bound<'py, PyAny>,
    chunk_specs: &[NativeChunkSpec],
    acquire_dosage_buffer_method: &Bound<'py, PyAny>,
    callback_batch_size: usize,
) -> PyResult<usize> {
    let compute_dosage_chunk_batch_method =
        callback.getattr("compute_preprocessed_variant_major_dosage_chunk_batch")?;
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
            compute_packed_chunk_method.call1((metadata, output_array_object, stats))?;
        }
    }
    Ok(processed_chunk_count)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRunEngineSession>()?;
    module.add_class::<NativeSingleTraitPipelineBundle>()?;
    module.add_class::<NativeSingleTraitRunInput>()?;
    module.add_class::<Regenie2RunEngine>()?;
    module.add_function(wrap_pyfunction!(run_bgen_delivery_with_writer_sessions, module)?)?;
    module.add_function(wrap_pyfunction!(run_bgen_session_delivery_with_writer_sessions, module)?)?;
    Ok(())
}
