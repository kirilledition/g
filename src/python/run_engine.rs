#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::fn_params_excessive_bools)]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use g_engine::Regenie2RunEngineCore;
use g_genotype::ChunkSpec as NativeChunkSpec;
use g_input::{self as native_input, AlignmentInputs, MultiAlignmentInputs};
use g_runtime as native_run_events;
use g_runtime as native_trusted_validation;
use numpy::{PyReadonlyArray1, PyReadwriteArray2, PyReadwriteArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::errors::{
    convert_bgen_error, convert_genotype_error, convert_input_error, convert_preflight_error, convert_schedule_error,
    convert_trusted_bgen_validation_error,
};
use super::genotype::{
    ChunkStats, VariantMetadata, VariantMetadataTuple, build_committed_identifier_set,
    convert_variant_metadata_columns_to_tuple,
};
use super::output::{self, OutputWriterSession};
use super::profile::build_profile_snapshot_dict;
use super::run_events;
use super::sample_alignment::{
    NativeAlignedSampleData, NativeGroupedAlignedSampleData, NativeMultiAlignedSampleData, parse_sample_key_mode,
};
use super::timing::NativeStageTimingRecorder;

#[pyclass]
struct Regenie2RunEngine {
    engine: Regenie2RunEngineCore,
}

struct BgenDeliveryCleanupExecution {
    final_parquet_paths: Vec<Option<String>>,
    callback_finished: bool,
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
        record_native_dispatch_bgen_engine_constructing(
            chunk_size,
            &bgen_path,
            trusted_no_missing_diploid,
            variant_limit,
        )?;
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
        let sample_identifier_data = self.sample_identifier_data(py, sample_path)?;
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
        let sample_identifier_data = self.sample_identifier_data(py, sample_path)?;
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
        let sample_identifier_data = self.sample_identifier_data(py, sample_path)?;
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
        record_native_dispatch_trusted_bgen_validation_started(&bgen_path, &validation_mode)?;
        let cache_directory = py
            .detach(native_trusted_validation::default_trusted_bgen_validation_cache_directory)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        py.detach(|| {
            self.engine.validate_trusted_no_missing_diploid_with_cache_directory(
                Path::new(&bgen_path),
                &validation_mode,
                cache_directory.as_path(),
            )
        })
        .map_err(convert_trusted_bgen_validation_error)
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
        let invocation_plan = g_engine::plan_bgen_delivery_invocation(
            Some(callback_batch_size),
            false,
            native_multi_aligned_sample_data.is_some(),
            native_aligned_sample_data.is_some(),
        )
        .map_err(|error| convert_schedule_error(&error))?;
        match invocation_plan.delivery_method {
            g_engine::BgenDeliveryMethod::DosageNativeMultiAlignedSamples => {
                let aligned_sample_data = native_multi_aligned_sample_data.ok_or_else(|| {
                    PyRuntimeError::new_err("Native BGEN delivery plan selected missing multi-aligned sample data.")
                })?;
                self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
                    py,
                    &aligned_sample_data.data.sample_indices,
                    callback,
                    committed_chunk_identifiers,
                    invocation_plan.callback_batch_size,
                )
            }
            g_engine::BgenDeliveryMethod::DosageNativeAlignedSamples => {
                let aligned_sample_data = native_aligned_sample_data.ok_or_else(|| {
                    PyRuntimeError::new_err("Native BGEN delivery plan selected missing aligned sample data.")
                })?;
                self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
                    py,
                    &aligned_sample_data.data.sample_indices,
                    callback,
                    committed_chunk_identifiers,
                    invocation_plan.callback_batch_size,
                )
            }
            g_engine::BgenDeliveryMethod::DosageSampleIndices => {
                let sample_index_values = sample_indices.as_slice()?.to_vec();
                self.run_bgen_variant_major_dosage_buffered_chunks_for_sample_indices(
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
        let invocation_plan = g_engine::plan_bgen_delivery_invocation(
            Some(callback_batch_size),
            true,
            native_multi_aligned_sample_data.is_some(),
            native_aligned_sample_data.is_some(),
        )
        .map_err(|error| convert_schedule_error(&error))?;
        match invocation_plan.delivery_method {
            g_engine::BgenDeliveryMethod::Packed8NativeMultiAlignedSamples => {
                let aligned_sample_data = native_multi_aligned_sample_data.ok_or_else(|| {
                    PyRuntimeError::new_err("Native BGEN delivery plan selected missing multi-aligned sample data.")
                })?;
                self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
                    py,
                    &aligned_sample_data.data.sample_indices,
                    callback,
                    committed_chunk_identifiers,
                )
            }
            g_engine::BgenDeliveryMethod::Packed8NativeAlignedSamples => {
                let aligned_sample_data = native_aligned_sample_data.ok_or_else(|| {
                    PyRuntimeError::new_err("Native BGEN delivery plan selected missing aligned sample data.")
                })?;
                self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
                    py,
                    &aligned_sample_data.data.sample_indices,
                    callback,
                    committed_chunk_identifiers,
                )
            }
            g_engine::BgenDeliveryMethod::Packed8SampleIndices => {
                let sample_index_values = sample_indices.as_slice()?.to_vec();
                self.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_sample_indices(
                    py,
                    &sample_index_values,
                    callback,
                    committed_chunk_identifiers,
                )
            }
            _ => Err(PyRuntimeError::new_err("Native BGEN delivery plan selected a dosage method for packed8.")),
        }
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
        &engine,
        sample_indices,
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
    engine: &Regenie2RunEngine,
    sample_indices: PyReadonlyArray1<'py, i64>,
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
        engine.engine.reader().reset_profile();
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
        engine.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_best_sample_source(
            py,
            sample_indices,
            native_aligned_sample_data,
            native_multi_aligned_sample_data,
            callback,
            committed_chunk_identifiers,
            callback_batch_size,
        )?
    } else {
        engine.run_bgen_variant_major_dosage_buffered_chunks_for_best_sample_source(
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
        stage_timing_recorder
            .set_native_bgen_profile_snapshot(native_bgen_profile_snapshot_as_i64(engine.profile_snapshot())?)?;
    }
    let cleanup_execution = execute_bgen_delivery_cleanup_actions(
        py,
        g_engine::BgenDeliveryCleanupOutcome::Success,
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
            g_engine::BgenDeliveryCleanupOutcome::Interrupted,
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
                    g_engine::BgenDeliveryCleanupOutcome::InterruptedCleanupFailure,
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
        g_engine::BgenDeliveryCleanupOutcome::Failure,
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
    cleanup_outcome: g_engine::BgenDeliveryCleanupOutcome,
    callback_finished: bool,
    callback: &Bound<'py, PyAny>,
    writer_sessions: &[PyRef<'py, OutputWriterSession>],
    writer_finish_thread_count: i64,
    stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    interrupted_event: Option<&native_run_events::RunInterruptedEventPayload>,
) -> PyResult<BgenDeliveryCleanupExecution> {
    let cleanup_plan = g_engine::plan_bgen_delivery_cleanup(cleanup_outcome, callback_finished);
    let mut final_parquet_paths = Vec::new();
    let mut resolved_callback_finished = callback_finished;
    for cleanup_action in cleanup_plan.cleanup_actions() {
        match cleanup_action {
            g_engine::BgenDeliveryCleanupAction::DrainCallback => {
                let callback_finish_start_time = Instant::now();
                run_events::record_native_dispatch_callback_drain_started_diagnostic_event()?;
                callback.call_method0("finish")?;
                record_stage_duration(stage_timing_recorder, "callback_drain", callback_finish_start_time)?;
                resolved_callback_finished = true;
            }
            g_engine::BgenDeliveryCleanupAction::FinishWriterSessions => {
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
            g_engine::BgenDeliveryCleanupAction::FinishInterruptedWriterSessions => {
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
            g_engine::BgenDeliveryCleanupAction::AbortCallback => {
                let _ = callback.call_method0("abort");
            }
            g_engine::BgenDeliveryCleanupAction::AbortWriterSessions => {
                output::abort_output_writer_sessions_for_delivery(writer_sessions);
            }
            g_engine::BgenDeliveryCleanupAction::WriteStageTimingSnapshot => {}
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

fn usize_to_i64(value: usize, value_name: &str) -> PyResult<i64> {
    i64::try_from(value).map_err(|_| PyValueError::new_err(format!("{value_name} exceeds native int64 capacity.")))
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

impl Regenie2RunEngine {
    fn sample_identifier_data(
        &self,
        py: Python<'_>,
        sample_path: Option<String>,
    ) -> PyResult<native_input::SampleIdentifierData> {
        if let Some(sample_path) = sample_path {
            let expected_sample_count = self.engine.reader().sample_count();
            return py
                .detach(move || {
                    native_input::load_sample_identifier_data_from_sample_file(
                        Path::new(&sample_path),
                        expected_sample_count,
                    )
                })
                .map_err(|error| convert_input_error("load_sample_identifier_data_from_sample_file", error.into()));
        }
        if !self.engine.reader().contains_embedded_samples() {
            return Err(PyValueError::new_err("BGEN file does not contain samples and no .sample file was found."));
        }
        let sample_identifiers = self.engine.reader().sample_identifiers();
        let sample_indices = (0..sample_identifiers.len())
            .map(|sample_index| i64::try_from(sample_index).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok(native_input::SampleIdentifierData {
            sample_indices,
            family_identifiers: sample_identifiers.clone(),
            individual_identifiers: sample_identifiers,
        })
    }

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
            g_engine::plan_chunk_batches(&chunk_specs, 1).map_err(|error| convert_schedule_error(&error))?;
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

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Regenie2RunEngine>()?;
    module.add_function(wrap_pyfunction!(run_bgen_delivery_with_writer_sessions, module)?)?;
    Ok(())
}
