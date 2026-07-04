//! Unified tracing setup for Rust and Python diagnostics.

#![allow(clippy::missing_errors_doc)]

use std::ffi::CString;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::callback_progress::NativeCallbackProgressTelemetryEvent;
use super::jax_runtime;
use super::run_events;
use g_runtime::logging_sink as native_logging_sink;
use g_runtime::run_events as native_run_events;
use g_runtime::telemetry_session as native_telemetry_session;
use g_runtime::telemetry_writer as native_telemetry_writer;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyModule, PyString, PyTuple};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

const PYTHON_LOGGING_TARGET: &str = "g.python";

static PYTHON_LOGGING_INSTALLED: AtomicBool = AtomicBool::new(false);

#[pyclass]
pub struct NativeTelemetryRunSession {
    progress_start_time: Instant,
    state: Mutex<native_telemetry_session::TelemetryRunSessionState>,
    native_telemetry_session: Option<NativeTelemetrySession>,
}

#[pyclass]
pub struct NativeTelemetryClosePolicy;

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
        let state = native_telemetry_session::TelemetryRunSessionState::new(
            telemetry_mode,
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
        self.emit_current_event_fields(
            py,
            native_run_events::RUN_COMPLETED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
    }

    pub fn emit_run_interrupted_event<'py>(&self, py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<()> {
        let event_payload = run_events::run_interrupted_event_from_py(event)?;
        let telemetry_fields = native_run_events::build_run_interrupted_telemetry_fields(&event_payload);
        let fields = run_events::run_interrupted_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_current_event_fields(
            py,
            native_run_events::RUN_FAILED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_WARN_LEVEL,
            &fields,
        )
    }

    pub fn emit_run_failed_event<'py>(&self, py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<()> {
        let event_payload = run_events::run_failed_event_from_py(event)?;
        let telemetry_fields = native_run_events::build_run_failed_telemetry_fields(&event_payload);
        let fields = run_events::run_failed_telemetry_fields_to_py_dict(py, &telemetry_fields)?;
        self.emit_current_event_fields(
            py,
            native_run_events::RUN_FAILED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_ERROR_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::RUN_STARTED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::EXECUTION_PLAN_PREPARED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::EFFECTIVE_CONFIG_WRITTEN_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::WRITER_FINISHED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::WRITER_FINISHED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::PREFLIGHT_COMPLETED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::PREFLIGHT_COMPLETED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::SAMPLE_ALIGNMENT_COMPLETED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
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
        self.emit_current_event_fields(
            py,
            native_run_events::PREDICTION_SOURCE_LOADED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::MULTI_PHENOTYPE_SAMPLE_SUMMARY_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
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
        self.emit_current_event_fields(
            py,
            native_run_events::GPU_GENOTYPE_FORMAT_RESOLVED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
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
        self.emit_current_event_fields(
            py,
            native_run_events::ASSOCIATION_BACKEND_SELECTED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
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
        self.emit_current_event_fields(
            py,
            native_run_events::BGEN_ENGINE_OPENED_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            &fields,
        )
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
        self.emit_current_event_fields(
            py,
            native_run_events::BINARY_CORRECTION_SUMMARY_EVENT_NAME,
            native_run_events::RUN_LIFECYCLE_INFO_LEVEL,
            fields,
        )
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

#[pymethods]
#[allow(clippy::unused_self)]
impl NativeTelemetryClosePolicy {
    #[new]
    fn new() -> Self {
        Self
    }

    fn close_telemetry_session_with_event(&self, py: Python<'_>, telemetry_session: &Bound<'_, PyAny>) -> PyResult<()> {
        close_telemetry_session_with_event(py, telemetry_session)
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
        .map_err(telemetry_writer_error_to_py)?;
        Ok(Self { writer: Mutex::new(writer) })
    }

    fn emit_json_line(&self, json_line: &str) -> PyResult<()> {
        self.lock_writer()?.write_json_line(json_line).map_err(telemetry_writer_error_to_py)?;
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
        let metadata = self.lock_writer()?.finish_close_metadata().map_err(telemetry_writer_error_to_py)?;
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
}

fn close_telemetry_session_with_event(py: Python<'_>, telemetry_session: &Bound<'_, PyAny>) -> PyResult<()> {
    if telemetry_session.is_none() {
        return Ok(());
    }
    let native_telemetry_session = optional_native_telemetry_session(py, telemetry_session)?;
    let close_plan = native_telemetry_session::plan_telemetry_close(true, native_telemetry_session.is_some());
    if !close_plan.should_close {
        return Ok(());
    }
    if !close_plan.use_native_close_with_event {
        return Err(PyTypeError::new_err(
            "telemetry close requires a TelemetrySession with a native telemetry session handle.",
        ));
    }
    let active_native_telemetry_session = native_telemetry_session
        .ok_or_else(|| PyRuntimeError::new_err("Native telemetry close plan selected a missing native session."))?;
    active_native_telemetry_session.call_method0("finish_with_current_close_event_metadata")?;
    Ok(())
}

fn optional_native_telemetry_session<'py>(
    py: Python<'py>,
    telemetry_session: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match telemetry_session.getattr("native_telemetry_session") {
        Ok(native_telemetry_session) if native_telemetry_session.is_none() => Ok(None),
        Ok(native_telemetry_session) => Ok(Some(native_telemetry_session)),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Err(PyTypeError::new_err(
            "telemetry close requires a TelemetrySession with a native telemetry session handle.",
        )),
        Err(error) => Err(error),
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
    let json_value = telemetry_json_value_from_py_any(payload.as_any())?;
    native_telemetry_session::serialize_telemetry_payload_json_line(&json_value)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn telemetry_json_value_from_py_any(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if value.is_none() {
        return Ok(JsonValue::Null);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(JsonValue::Bool(value.extract::<bool>()?));
    }
    if let Ok(mapping) = value.cast::<PyMapping>() {
        return telemetry_json_object_from_py_mapping(mapping).map(JsonValue::Object);
    }
    if let Ok(list) = value.cast::<PyList>() {
        return telemetry_json_array_from_py_iter(list.as_any()).map(JsonValue::Array);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return telemetry_json_array_from_py_iter(tuple.as_any()).map(JsonValue::Array);
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(signed_integer) = value.extract::<i64>() {
            return Ok(JsonValue::Number(JsonNumber::from(signed_integer)));
        }
        if let Ok(unsigned_integer) = value.extract::<u64>() {
            return Ok(JsonValue::Number(JsonNumber::from(unsigned_integer)));
        }
        return Ok(JsonValue::String(value.str()?.to_string_lossy().into_owned()));
    }
    if value.is_instance_of::<PyFloat>() {
        let float_value = value.extract::<f64>()?;
        let Some(number) = JsonNumber::from_f64(float_value) else {
            return Ok(JsonValue::String(value.str()?.to_string_lossy().into_owned()));
        };
        return Ok(JsonValue::Number(number));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(JsonValue::String(value.extract::<String>()?));
    }
    if let Some(path_text) = telemetry_path_string(value)? {
        return Ok(JsonValue::String(path_text));
    }
    Ok(JsonValue::String(value.str()?.to_string_lossy().into_owned()))
}

fn telemetry_json_object_from_py_mapping(mapping: &Bound<'_, PyMapping>) -> PyResult<JsonMap<String, JsonValue>> {
    let items = mapping.call_method0("items")?;
    let mut json_object = JsonMap::new();
    for item in items.try_iter()? {
        let item = item?;
        let tuple = item.cast::<PyTuple>()?;
        let key = tuple.get_item(0)?;
        let value = tuple.get_item(1)?;
        json_object.insert(telemetry_json_key_from_py_any(&key)?, telemetry_json_value_from_py_any(&value)?);
    }
    Ok(json_object)
}

fn telemetry_json_array_from_py_iter(value: &Bound<'_, PyAny>) -> PyResult<Vec<JsonValue>> {
    let mut values = Vec::new();
    for item in value.try_iter()? {
        values.push(telemetry_json_value_from_py_any(&item?)?);
    }
    Ok(values)
}

fn telemetry_json_key_from_py_any(key: &Bound<'_, PyAny>) -> PyResult<String> {
    if key.is_none() {
        return Ok("null".to_string());
    }
    if key.is_instance_of::<PyBool>() {
        return Ok(if key.extract::<bool>()? { "true" } else { "false" }.to_string());
    }
    if key.is_instance_of::<PyString>() {
        return key.extract::<String>();
    }
    Ok(key.str()?.to_string_lossy().into_owned())
}

fn telemetry_path_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let Ok(file_system_path_method) = value.getattr("__fspath__") else {
        return Ok(None);
    };
    let path_value = file_system_path_method.call0()?;
    if let Ok(path_text) = path_value.extract::<String>() {
        return Ok(Some(path_text));
    }
    let type_name = path_value.get_type().name()?.to_string_lossy().into_owned();
    Err(PyValueError::new_err(format!("__fspath__ returned unsupported type '{type_name}'; expected str.")))
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
        .map_err(logging_sink_initialization_error_to_py)
}

pub(crate) fn shutdown_logging() -> PyResult<()> {
    native_logging_sink::shutdown_logging_sinks().map_err(logging_sink_error_to_py)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTelemetryRunSession>()?;
    module.add_class::<NativeTelemetryClosePolicy>()?;
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

fn logging_sink_initialization_error_to_py(error: native_logging_sink::LoggingSinkInitializationError<PyErr>) -> PyErr {
    match error {
        native_logging_sink::LoggingSinkInitializationError::Sink(sink_error) => logging_sink_error_to_py(sink_error),
        native_logging_sink::LoggingSinkInitializationError::HostLogging(host_logging_error) => host_logging_error,
    }
}

#[expect(clippy::needless_pass_by_value, reason = "map_err passes the native logging sink error by value.")]
fn logging_sink_error_to_py(error: native_logging_sink::LoggingSinkError) -> PyErr {
    match error {
        native_logging_sink::LoggingSinkError::InvalidLogFilter { .. }
        | native_logging_sink::LoggingSinkError::InvalidTraceFilter { .. } => PyValueError::new_err(error.to_string()),
        native_logging_sink::LoggingSinkError::Writer(_)
        | native_logging_sink::LoggingSinkError::LoggingGuardMutexPoisoned => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}

#[expect(clippy::needless_pass_by_value, reason = "map_err passes the native writer error by value.")]
fn telemetry_writer_error_to_py(error: std::io::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
