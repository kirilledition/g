//! Unified tracing setup for Rust and Python diagnostics.

#![allow(clippy::missing_errors_doc)]

use std::ffi::CString;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::callback_progress::NativeCallbackProgressTelemetryEvent;
use super::errors;
use super::jax_runtime;
use super::json_bridge;
use super::run_events;
use super::telemetry_policy;
use g_runtime as native_logging_sink;
use g_runtime as native_run_events;
use g_runtime as native_telemetry_session;
use g_runtime as native_telemetry_writer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

const PYTHON_LOGGING_TARGET: &str = "g.python";

static PYTHON_LOGGING_INSTALLED: AtomicBool = AtomicBool::new(false);

#[pyclass]
pub struct NativeTelemetryRunSession {
    progress_start_time: Instant,
    state: Mutex<native_telemetry_session::TelemetryRunSessionState>,
    native_telemetry_session: Option<NativeTelemetrySession>,
}

#[pymethods]
impl NativeTelemetryRunSession {
    #[new]
    #[pyo3(signature = (
        telemetry_mode,
        stream_file,
        progress_interval_seconds,
        progress_interval_chunks,
        queue_size=65536,
        lossy=true,
        trace_event_cap=0,
        run_id=None,
    ))]
    pub fn new(
        telemetry_mode: &str,
        stream_file: Option<String>,
        progress_interval_seconds: f64,
        progress_interval_chunks: i64,
        queue_size: usize,
        lossy: bool,
        trace_event_cap: i64,
        run_id: Option<String>,
    ) -> PyResult<Self> {
        let parsed_telemetry_mode = telemetry_policy::parse_telemetry_mode(telemetry_mode)?;
        let state = native_telemetry_session::TelemetryRunSessionState::new(
            parsed_telemetry_mode,
            trace_event_cap,
            progress_interval_seconds,
            progress_interval_chunks,
            run_id,
        );
        let writer_plan = state.writer_plan(stream_file.is_some());
        let native_telemetry_session = if writer_plan.should_open_writer {
            let stream_file = stream_file
                .ok_or_else(|| PyValueError::new_err("Telemetry stream file is required when telemetry is enabled."))?;
            Some(NativeTelemetrySession::new(
                &stream_file,
                queue_size,
                lossy,
                telemetry_event_cap_to_usize(writer_plan.event_cap)?,
            )?)
        } else {
            None
        };

        Ok(Self { progress_start_time: Instant::now(), state: Mutex::new(state), native_telemetry_session })
    }

    #[getter]
    fn run_id(&self) -> PyResult<String> {
        self.run_id_value()
    }

    #[getter]
    fn enabled(&self) -> PyResult<bool> {
        Ok(self.state_guard()?.enabled())
    }

    #[getter]
    fn profile_enabled(&self) -> PyResult<bool> {
        Ok(self.state_guard()?.profile_enabled())
    }

    #[getter]
    fn event_cap(&self) -> PyResult<Option<i64>> {
        Ok(self.state_guard()?.event_cap())
    }

    #[getter]
    fn has_native_telemetry_session(&self) -> bool {
        self.native_telemetry_session.is_some()
    }

    pub fn should_emit_progress(&self, processed_chunk_count: i64) -> PyResult<bool> {
        let current_time_seconds = self.progress_start_time.elapsed().as_secs_f64();
        let mut state = self.state_guard()?;
        Ok(state.should_emit_progress_at(processed_chunk_count, current_time_seconds))
    }

    pub fn emit_current_event<'py>(
        &self,
        py: Python<'py>,
        event: &str,
        level: &str,
        fields: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        self.emit_current_event_fields(py, event, level, fields)
    }

    pub fn emit_run_completed_event<'py>(&self, py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<()> {
        let event_payload = run_events::run_completed_event_from_py(event)?;
        let telemetry_fields = native_run_events::build_run_completed_telemetry_fields(&event_payload);
        let fields = run_events::run_completed_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::RunCompleted, &fields)
    }

    pub fn emit_run_interrupted_event<'py>(&self, py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<()> {
        let event_payload = run_events::run_interrupted_event_from_py(event)?;
        let telemetry_fields = native_run_events::build_run_interrupted_telemetry_fields(&event_payload);
        let fields = run_events::run_interrupted_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::RunInterrupted, &fields)
    }

    pub fn emit_run_failed_event<'py>(&self, py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<()> {
        let event_payload = run_events::run_failed_event_from_py(event)?;
        let telemetry_fields = native_run_events::build_run_failed_telemetry_fields(&event_payload);
        let fields = run_events::run_failed_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::RunFailed, &fields)
    }

    pub fn emit_run_started_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        trait_type: &str,
        phenotype_count: i64,
        output_run_root: &str,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_run_started_telemetry_fields(
            association_mode,
            trait_type,
            phenotype_count,
            output_run_root,
        );
        let fields = run_events::run_started_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::RunStarted, &fields)
    }

    pub fn emit_execution_plan_prepared_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        trait_type: &str,
        phenotype_count: i64,
        chunk_size: i64,
        variant_limit: Option<i64>,
        device: &str,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_execution_plan_prepared_telemetry_fields(
            association_mode,
            trait_type,
            phenotype_count,
            chunk_size,
            variant_limit,
            device,
        );
        let fields = run_events::execution_plan_prepared_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::ExecutionPlanPrepared, &fields)
    }

    pub fn emit_effective_config_written_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        phenotype: &str,
        effective_config: &str,
        output_run_directory: &str,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_effective_config_written_telemetry_fields(
            association_mode,
            phenotype,
            effective_config,
            output_run_directory,
        );
        let fields = run_events::effective_config_written_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::EffectiveConfigWritten, &fields)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn emit_phenotype_writer_finished_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        phenotype: &str,
        final_output_path: Option<String>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_phenotype_writer_finished_telemetry_fields(
            association_mode,
            phenotype,
            final_output_path.as_deref(),
        );
        let fields = run_events::phenotype_writer_finished_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::WriterFinished, &fields)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn emit_multi_phenotype_writer_finished_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        phenotype_count: i64,
        final_output_paths: Vec<Option<String>>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_multi_phenotype_writer_finished_telemetry_fields(
            association_mode,
            phenotype_count,
            &final_output_paths,
        );
        let fields = run_events::multi_phenotype_writer_finished_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::WriterFinished, &fields)
    }

    pub fn emit_single_trait_preflight_completed_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        phenotype: &str,
        sample_count: i64,
        covariate_count: i64,
        chromosome_count: i64,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_single_trait_preflight_completed_telemetry_fields(
            association_mode,
            phenotype,
            sample_count,
            covariate_count,
            chromosome_count,
        );
        let fields = run_events::single_trait_preflight_completed_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::PreflightCompleted, &fields)
    }

    pub fn emit_multi_phenotype_preflight_completed_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        phenotype_count: i64,
        sample_count: i64,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_multi_phenotype_preflight_completed_telemetry_fields(
            association_mode,
            phenotype_count,
            sample_count,
        );
        let fields =
            run_events::multi_phenotype_preflight_completed_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::PreflightCompleted, &fields)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn emit_sample_alignment_completed_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        phenotype: Option<String>,
        phenotype_count: Option<i64>,
        sample_count: Option<i64>,
        covariate_count: Option<i64>,
        phenotype_group_count: Option<i64>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_sample_alignment_completed_telemetry_fields(
            association_mode,
            phenotype.as_deref(),
            phenotype_count,
            sample_count,
            covariate_count,
            phenotype_group_count,
        );
        let fields = run_events::sample_alignment_completed_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(
            py,
            native_run_events::RunTelemetryEventKind::SampleAlignmentCompleted,
            &fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn emit_prediction_source_loaded_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        phenotype: Option<String>,
        phenotype_count: Option<i64>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_prediction_source_loaded_telemetry_fields(
            association_mode,
            phenotype.as_deref(),
            phenotype_count,
        );
        let fields = run_events::prediction_source_loaded_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::PredictionSourceLoaded, &fields)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn emit_multi_phenotype_sample_summary_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        multi_phenotype_sample_mode: &str,
        sample_counts: Vec<i64>,
        sample_set_fingerprints: Vec<Option<String>>,
        phenotype_group_count: i64,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_multi_phenotype_sample_summary_telemetry_fields(
            association_mode,
            multi_phenotype_sample_mode,
            &sample_counts,
            &sample_set_fingerprints,
            phenotype_group_count,
        );
        let fields = run_events::multi_phenotype_sample_summary_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(
            py,
            native_run_events::RunTelemetryEventKind::MultiPhenotypeSampleSummary,
            &fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn emit_gpu_genotype_format_resolved_event<'py>(
        &self,
        py: Python<'py>,
        requested_gpu_genotype_format: &str,
        resolved_gpu_genotype_format: &str,
        resolution_reason: &str,
        fallback_error: Option<String>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_gpu_genotype_format_resolved_telemetry_fields(
            requested_gpu_genotype_format,
            resolved_gpu_genotype_format,
            resolution_reason,
            fallback_error.as_deref(),
        );
        let fields = run_events::gpu_genotype_format_resolved_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(
            py,
            native_run_events::RunTelemetryEventKind::GpuGenotypeFormatResolved,
            &fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn emit_association_backend_selected_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        association_backend_kind: &str,
        device: &str,
        genotype_format: &str,
        phenotype: Option<String>,
        phenotype_count: Option<i64>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_association_backend_selected_telemetry_fields(
            association_mode,
            association_backend_kind,
            device,
            genotype_format,
            phenotype.as_deref(),
            phenotype_count,
        );
        let fields = run_events::association_backend_selected_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(
            py,
            native_run_events::RunTelemetryEventKind::AssociationBackendSelected,
            &fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn emit_bgen_engine_opened_event<'py>(
        &self,
        py: Python<'py>,
        association_mode: &str,
        association_backend_kind: &str,
        sample_count: i64,
        variant_count: i64,
        phenotype: Option<String>,
        phenotype_count: Option<i64>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_bgen_engine_opened_telemetry_fields(
            association_mode,
            association_backend_kind,
            sample_count,
            variant_count,
            phenotype.as_deref(),
            phenotype_count,
        );
        let fields = run_events::bgen_engine_opened_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::BgenEngineOpened, &fields)
    }

    fn emit_callback_progress_event<'py>(
        &self,
        py: Python<'py>,
        progress_event: &NativeCallbackProgressTelemetryEvent,
    ) -> PyResult<()> {
        let fields = PyDict::new(py);
        fields.set_item("chromosome", progress_event.chromosome_value())?;
        fields.set_item("processed_chunk_count", progress_event.processed_chunk_count_value())?;
        self.emit_current_event_fields(py, progress_event.event_name_value(), progress_event.level_value(), &fields)
    }

    pub fn emit_binary_correction_summary_event<'py>(
        &self,
        py: Python<'py>,
        fields: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        self.emit_telemetry_event_fields(py, native_run_events::RunTelemetryEventKind::BinaryCorrectionSummary, fields)
    }

    pub fn emit_jax_runtime_diagnostic_event<'py>(
        &self,
        py: Python<'py>,
        event: &Bound<'py, PyAny>,
        telemetry_level: &str,
    ) -> PyResult<()> {
        let (event_name, fields) = jax_runtime::jax_runtime_diagnostic_event_fields_to_py_dict(py, event)?;
        self.emit_current_event_fields(py, &event_name, telemetry_level, &fields)
    }

    pub fn emit_progress<'py>(
        &self,
        py: Python<'py>,
        processed_chunk_count: i64,
        fields: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let current_time_seconds = self.progress_start_time.elapsed().as_secs_f64();
        let emission_plan = self.state_guard()?.plan_progress_emission_at(
            processed_chunk_count,
            current_time_seconds,
            self.native_telemetry_session.is_some(),
        );
        if !emission_plan.should_emit {
            return Ok(());
        }
        let Some(native_telemetry_session) = self.native_telemetry_session.as_ref() else {
            return Ok(());
        };
        let progress_fields = PyDict::new(py);
        progress_fields.set_item("processed_chunk_count", processed_chunk_count)?;
        for (key, value) in fields {
            progress_fields.set_item(key, value)?;
        }
        native_telemetry_session.emit_current_event(
            py,
            &self.run_id_value()?,
            &emission_plan.event_name,
            &emission_plan.level,
            &progress_fields,
        )
    }

    pub fn build_current_event_payload<'py>(
        &self,
        py: Python<'py>,
        event: &str,
        level: &str,
        fields: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        build_current_telemetry_event_payload(py, &self.run_id_value()?, event, level, fields)
    }

    pub fn emit_payload(&self, py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<()> {
        let Some(native_telemetry_session) = self.native_telemetry_session.as_ref() else {
            return Ok(());
        };
        native_telemetry_session.emit_payload(py, payload)
    }

    pub fn counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let Some(native_telemetry_session) = self.native_telemetry_session.as_ref() else {
            return telemetry_writer_counter_snapshot_to_py_dict(
                py,
                &native_telemetry_session::TelemetryWriterCounterSnapshot::empty(),
            );
        };
        native_telemetry_session.counters(py)
    }

    pub fn close_metadata<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(native_telemetry_session) = self.native_telemetry_session.as_ref() else {
            return Ok(None);
        };
        native_telemetry_session.close_metadata(py)
    }

    pub fn finish_close_metadata<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(native_telemetry_session) = self.native_telemetry_session.as_ref() else {
            return Ok(None);
        };
        native_telemetry_session.finish_close_metadata(py).map(Some)
    }

    pub fn finish_with_current_close_event_metadata<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(native_telemetry_session) = self.native_telemetry_session.as_ref() else {
            return Ok(None);
        };
        native_telemetry_session.finish_with_current_close_event_metadata(py, &self.run_id_value()?).map(Some)
    }
}

struct NativeTelemetrySession {
    writer: Mutex<native_telemetry_writer::TelemetrySessionWriter>,
}

impl NativeTelemetrySession {
    fn new(stream_file: &str, queue_size: usize, lossy: bool, event_cap: Option<usize>) -> PyResult<Self> {
        let writer = native_telemetry_writer::TelemetrySessionWriter::new(
            Path::new(stream_file).to_path_buf(),
            queue_size,
            lossy,
            event_cap,
        )
        .map_err(|error| errors::convert_telemetry_writer_error(&error))?;
        Ok(Self { writer: Mutex::new(writer) })
    }

    fn emit_json_line(&self, json_line: &str) -> PyResult<()> {
        self.lock_writer()?
            .write_json_line(json_line)
            .map_err(|error| errors::convert_telemetry_writer_error(&error))?;
        Ok(())
    }

    fn emit_payload(&self, _py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<()> {
        let json_line = serialize_telemetry_payload_json_line(payload)?;
        self.emit_json_line(&json_line)
    }

    fn emit_current_event<'py>(
        &self,
        py: Python<'py>,
        run_id: &str,
        event: &str,
        level: &str,
        fields: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let payload = build_current_telemetry_event_payload(py, run_id, event, level, fields)?;
        self.emit_payload(py, &payload)
    }

    fn counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        telemetry_writer_counter_snapshot_to_py_dict(py, &self.counter_snapshot()?)
    }

    fn close_metadata<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.lock_writer()?
            .close_metadata()
            .map(|metadata| telemetry_close_metadata_to_py_dict(py, &metadata))
            .transpose()
    }

    fn finish_close_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metadata = self
            .lock_writer()?
            .finish_close_metadata()
            .map_err(|error| errors::convert_telemetry_writer_error(&error))?;
        telemetry_close_metadata_to_py_dict(py, &metadata)
    }

    fn finish_with_current_close_event_metadata<'py>(
        &self,
        py: Python<'py>,
        run_id: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let _ = self.emit_current_close_event(py, run_id);
        self.finish_close_metadata(py)
    }

    fn counter_snapshot(&self) -> PyResult<native_telemetry_session::TelemetryWriterCounterSnapshot> {
        Ok(self.lock_writer()?.counter_snapshot())
    }

    fn emit_current_close_event<'py>(&self, py: Python<'py>, run_id: &str) -> PyResult<()> {
        let close_event_payload =
            native_telemetry_session::build_telemetry_close_event_payload(self.counter_snapshot()?);
        let fields = telemetry_close_event_fields_to_py_dict(py, &close_event_payload)?;
        self.emit_current_event(py, run_id, &close_event_payload.event_name, &close_event_payload.level, &fields)
    }

    fn lock_writer(&self) -> PyResult<std::sync::MutexGuard<'_, native_telemetry_writer::TelemetrySessionWriter>> {
        self.writer.lock().map_err(|_| PyRuntimeError::new_err("Telemetry writer mutex was poisoned."))
    }
}

impl NativeTelemetryRunSession {
    fn state_guard(&self) -> PyResult<std::sync::MutexGuard<'_, native_telemetry_session::TelemetryRunSessionState>> {
        self.state.lock().map_err(|_| PyRuntimeError::new_err("Telemetry run session mutex was poisoned."))
    }

    fn run_id_value(&self) -> PyResult<String> {
        Ok(self.state_guard()?.run_id().to_string())
    }

    fn emit_current_event_fields<'py>(
        &self,
        py: Python<'py>,
        event: &str,
        level: &str,
        fields: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let emission_plan = self.state_guard()?.plan_event_emission(self.native_telemetry_session.is_some());
        if !emission_plan.should_emit {
            return Ok(());
        }
        let Some(native_telemetry_session) = self.native_telemetry_session.as_ref() else {
            return Ok(());
        };
        native_telemetry_session.emit_current_event(py, &self.run_id_value()?, event, level, fields)
    }

    fn emit_telemetry_event_fields<'py>(
        &self,
        py: Python<'py>,
        event_kind: native_run_events::RunTelemetryEventKind,
        fields: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        self.emit_current_event_fields(py, event_kind.event_name(), event_kind.level(), fields)
    }
}

fn build_current_telemetry_event_payload<'py>(
    py: Python<'py>,
    run_id: &str,
    event: &str,
    level: &str,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let thread_name = current_python_thread_name(py)?;
    let envelope = native_telemetry_session::build_current_telemetry_event_envelope(run_id, event, level, &thread_name);
    telemetry_event_envelope_to_py_dict(py, &envelope, fields)
}

fn telemetry_event_envelope_to_py_dict<'py>(
    py: Python<'py>,
    envelope: &native_telemetry_session::TelemetryEventEnvelope,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("schema_version", envelope.schema_version)?;
    payload.set_item("run_id", &envelope.run_id)?;
    payload.set_item("ts", &envelope.timestamp)?;
    payload.set_item("level", &envelope.level)?;
    payload.set_item("source", envelope.source)?;
    payload.set_item("target", envelope.target)?;
    payload.set_item("event", &envelope.event)?;
    payload.set_item("pid", envelope.process_identifier)?;
    payload.set_item("thread_name", &envelope.thread_name)?;
    for (key, value) in fields {
        if !value.is_none() {
            payload.set_item(key, value)?;
        }
    }
    Ok(payload)
}

fn current_python_thread_name(py: Python<'_>) -> PyResult<String> {
    let threading_module = PyModule::import(py, "threading")?;
    threading_module.call_method0("current_thread")?.getattr("name")?.extract::<String>()
}

fn telemetry_writer_counter_snapshot_to_py_dict<'py>(
    py: Python<'py>,
    snapshot: &native_telemetry_session::TelemetryWriterCounterSnapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let counters = PyDict::new(py);
    counters.set_item("accepted_event_count", snapshot.accepted_event_count)?;
    counters.set_item("written_event_count", snapshot.written_event_count)?;
    counters.set_item("dropped_event_count", snapshot.dropped_event_count)?;
    counters.set_item("cap_dropped_event_count", snapshot.cap_dropped_event_count)?;
    counters.set_item("queue_dropped_event_count", snapshot.queue_dropped_event_count)?;
    counters.set_item("event_cap_exceeded", snapshot.event_cap_exceeded)?;
    counters.set_item("lossy", snapshot.lossy)?;
    counters.set_item("event_cap", snapshot.event_cap)?;
    counters.set_item("finish_flush_duration_seconds", snapshot.finish_flush_duration_seconds)?;
    Ok(counters)
}

fn telemetry_close_metadata_to_py_dict<'py>(
    py: Python<'py>,
    metadata: &native_telemetry_session::TelemetryCloseMetadataPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload
        .set_item("writer_counters", telemetry_writer_counter_snapshot_to_py_dict(py, &metadata.writer_counters)?)?;
    Ok(payload)
}

fn telemetry_close_event_fields_to_py_dict<'py>(
    py: Python<'py>,
    payload: &native_telemetry_session::TelemetryCloseEventPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let fields = PyDict::new(py);
    fields.set_item("writer_counters", telemetry_writer_counter_snapshot_to_py_dict(py, &payload.writer_counters)?)?;
    Ok(fields)
}

fn serialize_telemetry_payload_json_line(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    let json_value = json_bridge::json_value_from_py_any(payload.as_any())?;
    native_telemetry_session::serialize_telemetry_payload_json_line(&json_value)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub fn emit_diagnostic_event(level: &str, event: &str, message: &str, fields_json: Option<String>) -> PyResult<()> {
    let fields_json = fields_json.unwrap_or_else(|| "{}".to_string());
    match level {
        "error" => {
            tracing::error!(target: "g.python.diagnostic", g_event = event, g_fields = %fields_json, "{}", message);
        }
        "warn" | "warning" => {
            tracing::warn!(target: "g.python.diagnostic", g_event = event, g_fields = %fields_json, "{}", message);
        }
        "info" => {
            tracing::info!(target: "g.python.diagnostic", g_event = event, g_fields = %fields_json, "{}", message);
        }
        "debug" => {
            tracing::debug!(target: "g.python.diagnostic", g_event = event, g_fields = %fields_json, "{}", message);
        }
        "trace" => {
            tracing::trace!(target: "g.python.diagnostic", g_event = event, g_fields = %fields_json, "{}", message);
        }
        other_level => {
            return Err(PyValueError::new_err(format!("Unsupported diagnostic event level: {other_level}")));
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    clippy::needless_pass_by_value,
    reason = "Runtime logging policy forwards concrete sink fields directly."
)]
pub(crate) fn initialize_logging(
    py: Python<'_>,
    log_filter: Option<String>,
    log_file: Option<String>,
    log_stderr: bool,
    log_queue_size: usize,
    log_lossy: bool,
    include_source_location: bool,
    include_span_events: bool,
    trace_file: Option<String>,
    trace_filter: Option<String>,
    trace_event_cap: Option<usize>,
) -> PyResult<bool> {
    let config = native_logging_sink::LoggingSinkConfig {
        log_filter: log_filter.as_deref(),
        log_file: log_file.as_deref().map(Path::new),
        log_stderr,
        log_queue_size,
        log_lossy,
        include_source_location,
        include_span_events,
        trace_file: trace_file.as_deref().map(Path::new),
        trace_filter: trace_filter.as_deref(),
        trace_event_cap,
    };
    native_logging_sink::initialize_logging_sinks(config, || setup_python_logging(py))
        .map_err(errors::convert_logging_sink_initialization_error)
}

pub(crate) fn shutdown_logging() -> PyResult<()> {
    native_logging_sink::shutdown_logging_sinks().map_err(|error| errors::convert_logging_sink_error(&error))
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTelemetryRunSession>()?;
    Ok(())
}

fn setup_python_logging(py: Python<'_>) -> PyResult<()> {
    if PYTHON_LOGGING_INSTALLED
        .try_update(Ordering::AcqRel, Ordering::Acquire, |installed| (!installed).then_some(true))
        .is_err()
    {
        return Ok(());
    }

    if let Err(error) = install_python_logging(py) {
        PYTHON_LOGGING_INSTALLED.store(false, Ordering::Release);
        return Err(error);
    }
    Ok(())
}

fn install_python_logging(py: Python<'_>) -> PyResult<()> {
    pyo3_pylogger::setup_logging(py, PYTHON_LOGGING_TARGET)?;
    install_python_host_handler(py)?;
    register_shutdown_logging(py)?;
    Ok(())
}

fn install_python_host_handler(py: Python<'_>) -> PyResult<()> {
    let logging = py.import("logging")?;
    let code = CString::new(
        r#"
root_logger = getLogger()
if not any(handler.__class__.__name__ == "HostHandler" for handler in root_logger.handlers):
    root_logger.addHandler(HostHandler())
root_logger.setLevel(NOTSET)
"#,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    py.run(&code, Some(&logging.dict()), None)
}

fn register_shutdown_logging(py: Python<'_>) -> PyResult<()> {
    let core_module = py.import("g._core")?;
    let runtime_state_type = core_module.getattr("NativeRuntimeState")?;
    let runtime_state = runtime_state_type.call_method0("global_process_runtime_state")?;
    let shutdown_logging_function = runtime_state.getattr("shutdown_logging_runtime")?;
    let atexit = py.import("atexit")?;
    atexit.call_method1("register", (shutdown_logging_function,))?;
    Ok(())
}

fn telemetry_event_cap_to_usize(event_cap: Option<i64>) -> PyResult<Option<usize>> {
    event_cap
        .map(|value| {
            usize::try_from(value).map_err(|_| PyValueError::new_err("Telemetry event cap must be non-negative."))
        })
        .transpose()
}
