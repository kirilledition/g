//! Unified tracing setup for Rust and Python diagnostics.

#![allow(clippy::missing_errors_doc)]

use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::run_events;
use g_runtime::run_events as native_run_events;
use g_runtime::telemetry_session as native_telemetry_session;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use tracing_appender::non_blocking::{NonBlocking, NonBlockingBuilder, WorkerGuard};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::fmt::writer::MakeWriter;
use tracing_subscriber::prelude::*;

const DEFAULT_LOG_FILTER: &str = "info";
const PYTHON_LOGGING_TARGET: &str = "g.python";

static LOGGING_GUARDS: Mutex<Option<Vec<WorkerGuard>>> = Mutex::new(None);
static PYTHON_LOGGING_INSTALLED: AtomicBool = AtomicBool::new(false);
static TELEMETRY_WRITER: Mutex<Option<SharedTelemetryWriter>> = Mutex::new(None);

#[derive(Clone)]
struct SharedTelemetryWriter {
    path: PathBuf,
    writer: TelemetryWriterFactory,
}

#[derive(Clone)]
struct TelemetryWriterFactory {
    writer: NonBlocking,
    event_cap_state: Arc<native_telemetry_session::TelemetryEventCapState>,
}

struct TelemetryLineWriter {
    writer: NonBlocking,
    event_cap_state: Arc<native_telemetry_session::TelemetryEventCapState>,
    line_buffer: Vec<u8>,
}

#[pyclass]
pub struct NativeTelemetryProgressThrottle {
    start_time: Instant,
    state: Mutex<native_telemetry_session::TelemetryProgressThrottleState>,
}

#[pyclass]
pub struct NativeTelemetryEventEmissionPlan {
    inner: native_telemetry_session::TelemetryEventEmissionPlan,
}

#[pyclass]
pub struct NativeTelemetryProgressEmissionPlan {
    inner: native_telemetry_session::TelemetryProgressEmissionPlan,
}

#[pyclass]
pub struct NativeTelemetryClosePlan {
    inner: native_telemetry_session::TelemetryClosePlan,
}

#[pyclass]
pub struct NativeTelemetryRunSession {
    progress_start_time: Instant,
    state: Mutex<native_telemetry_session::TelemetryRunSessionState>,
    native_telemetry_session: Option<NativeTelemetrySession>,
}

impl TelemetryWriterFactory {
    fn new(writer: NonBlocking, event_cap_state: native_telemetry_session::TelemetryEventCapState) -> Self {
        Self { writer, event_cap_state: Arc::new(event_cap_state) }
    }

    fn write_json_line(&self, json_line: &str) -> io::Result<()> {
        let mut line_writer = self.make_writer();
        line_writer.write_all(json_line.as_bytes())?;
        if !json_line.ends_with('\n') {
            line_writer.write_all(b"\n")?;
        }
        line_writer.flush()
    }

    fn fail_if_lossless_cap_exceeded(&self) -> PyResult<()> {
        if self.event_cap_state.should_fail_for_cap_exceeded() {
            return Err(PyRuntimeError::new_err(self.event_cap_state.cap_exceeded_error_message()));
        }
        Ok(())
    }

    fn counter_snapshot(
        &self,
        finish_flush_duration_seconds: Option<f64>,
    ) -> native_telemetry_session::TelemetryWriterCounterSnapshot {
        self.event_cap_state
            .counter_snapshot(self.writer.error_counter().dropped_lines(), finish_flush_duration_seconds)
    }
}

impl<'a> MakeWriter<'a> for TelemetryWriterFactory {
    type Writer = TelemetryLineWriter;

    fn make_writer(&'a self) -> Self::Writer {
        TelemetryLineWriter {
            writer: self.writer.clone(),
            event_cap_state: Arc::clone(&self.event_cap_state),
            line_buffer: Vec::new(),
        }
    }
}

impl TelemetryLineWriter {
    fn write_complete_line(&mut self, line: &[u8]) -> io::Result<()> {
        match self.event_cap_state.reserve_event()? {
            native_telemetry_session::TelemetryCapAction::Write => self.writer.write_all(line),
            native_telemetry_session::TelemetryCapAction::Drop => Ok(()),
        }
    }
}

impl io::Write for TelemetryLineWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if !self.event_cap_state.has_event_cap() {
            let event_count = memchr::memchr_iter(b'\n', buffer).count();
            self.event_cap_state.record_uncapped_event_count(event_count);
            self.writer.write_all(buffer)?;
            return Ok(buffer.len());
        }

        self.line_buffer.extend_from_slice(buffer);
        while let Some(newline_index) = self.line_buffer.iter().position(|byte| *byte == b'\n') {
            let complete_line = self.line_buffer.drain(..=newline_index).collect::<Vec<_>>();
            self.write_complete_line(&complete_line)?;
        }
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.line_buffer.is_empty() {
            let complete_line = std::mem::take(&mut self.line_buffer);
            self.write_complete_line(&complete_line)?;
        }
        self.writer.flush()
    }
}

#[pymethods]
impl NativeTelemetryProgressThrottle {
    #[new]
    pub fn new(progress_interval_seconds: f64, progress_interval_chunks: i64) -> Self {
        Self {
            start_time: Instant::now(),
            state: Mutex::new(native_telemetry_session::TelemetryProgressThrottleState::new(
                progress_interval_seconds,
                progress_interval_chunks,
            )),
        }
    }

    pub fn should_emit_progress(&self, processed_chunk_count: i64) -> PyResult<bool> {
        let current_time_seconds = self.start_time.elapsed().as_secs_f64();
        let mut state =
            self.state.lock().map_err(|_| PyRuntimeError::new_err("Telemetry progress mutex was poisoned."))?;
        Ok(state.should_emit_progress_at(processed_chunk_count, current_time_seconds))
    }
}

#[pymethods]
impl NativeTelemetryEventEmissionPlan {
    #[getter]
    fn should_emit(&self) -> bool {
        self.inner.should_emit
    }
}

#[pymethods]
impl NativeTelemetryProgressEmissionPlan {
    #[getter]
    fn should_emit(&self) -> bool {
        self.inner.should_emit
    }

    #[getter]
    fn event_name(&self) -> &str {
        &self.inner.event_name
    }

    #[getter]
    fn level(&self) -> &str {
        &self.inner.level
    }
}

#[pymethods]
impl NativeTelemetryClosePlan {
    #[getter]
    fn should_close(&self) -> bool {
        self.inner.should_close
    }

    #[getter]
    fn use_native_close_with_event(&self) -> bool {
        self.inner.use_native_close_with_event
    }

    #[getter]
    fn should_emit_legacy_close_event(&self) -> bool {
        self.inner.should_emit_legacy_close_event
    }

    #[getter]
    fn legacy_close_event_name(&self) -> &str {
        &self.inner.legacy_close_event_name
    }

    #[getter]
    fn legacy_close_event_level(&self) -> &str {
        &self.inner.legacy_close_event_level
    }
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
                stream_file,
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

#[pyclass]
pub struct NativeTelemetrySession {
    path: PathBuf,
    writer: Mutex<Option<TelemetryWriterFactory>>,
    guard: Mutex<Option<WorkerGuard>>,
    last_counter_snapshot: Mutex<Option<native_telemetry_session::TelemetryWriterCounterSnapshot>>,
}

#[pymethods]
impl NativeTelemetrySession {
    #[new]
    #[pyo3(signature = (stream_file, queue_size=65536, lossy=true, event_cap=None))]
    pub fn new(stream_file: String, queue_size: usize, lossy: bool, event_cap: Option<usize>) -> PyResult<Self> {
        let path = PathBuf::from(stream_file);
        let (writer, guard) = build_telemetry_file_writer(&path, queue_size, lossy, normalize_event_cap(event_cap))?;
        replace_shared_telemetry_writer(path.clone(), writer.clone())?;
        Ok(Self {
            path,
            writer: Mutex::new(Some(writer)),
            guard: Mutex::new(Some(guard)),
            last_counter_snapshot: Mutex::new(None),
        })
    }

    pub fn emit_json_line(&self, json_line: &str) -> PyResult<()> {
        let writer_guard =
            self.writer.lock().map_err(|_| PyRuntimeError::new_err("Telemetry writer mutex was poisoned."))?;
        let Some(writer) = writer_guard.as_ref() else {
            return Ok(());
        };
        writer.write_json_line(json_line).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(())
    }

    pub fn emit_payload(&self, py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<()> {
        let json_line = serialize_telemetry_payload_json_line(py, payload)?;
        self.emit_json_line(&json_line)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_event<'py>(
        &self,
        py: Python<'py>,
        run_id: &str,
        event: &str,
        level: &str,
        timestamp: &str,
        process_identifier: u32,
        thread_name: &str,
        fields: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let payload = build_telemetry_event_payload(
            py,
            run_id,
            event,
            level,
            timestamp,
            process_identifier,
            thread_name,
            fields,
        )?;
        self.emit_payload(py, &payload)
    }

    pub fn emit_current_event<'py>(
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

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::unused_self)]
    pub fn build_event_payload<'py>(
        &self,
        py: Python<'py>,
        run_id: &str,
        event: &str,
        level: &str,
        timestamp: &str,
        process_identifier: u32,
        thread_name: &str,
        fields: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        build_telemetry_event_payload(py, run_id, event, level, timestamp, process_identifier, thread_name, fields)
    }

    pub fn counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let writer_guard =
            self.writer.lock().map_err(|_| PyRuntimeError::new_err("Telemetry writer mutex was poisoned."))?;
        if let Some(writer) = writer_guard.as_ref() {
            return telemetry_writer_counter_snapshot_to_py_dict(py, &writer.counter_snapshot(None));
        }
        let last_counter_snapshot = self
            .last_counter_snapshot
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Telemetry counter snapshot mutex was poisoned."))?;
        let Some(counter_snapshot) = last_counter_snapshot.as_ref() else {
            return telemetry_writer_counter_snapshot_to_py_dict(
                py,
                &native_telemetry_session::TelemetryWriterCounterSnapshot::empty(),
            );
        };
        telemetry_writer_counter_snapshot_to_py_dict(py, counter_snapshot)
    }

    pub fn close_metadata<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let last_counter_snapshot = self
            .last_counter_snapshot
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Telemetry counter snapshot mutex was poisoned."))?;
        let Some(counter_snapshot) = last_counter_snapshot.as_ref() else {
            return Ok(None);
        };
        let metadata = native_telemetry_session::build_telemetry_close_metadata(counter_snapshot.clone());
        telemetry_close_metadata_to_py_dict(py, &metadata).map(Some)
    }

    pub fn finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let counter_snapshot = self.finish_counter_snapshot()?;
        telemetry_writer_counter_snapshot_to_py_dict(py, &counter_snapshot)
    }

    pub fn finish_close_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let counter_snapshot = self.finish_counter_snapshot()?;
        let metadata = native_telemetry_session::build_telemetry_close_metadata(counter_snapshot);
        telemetry_close_metadata_to_py_dict(py, &metadata)
    }

    pub fn finish_with_close_event<'py>(
        &self,
        py: Python<'py>,
        run_id: &str,
        timestamp: &str,
        process_identifier: u32,
        thread_name: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fields = PyDict::new(py);
        fields.set_item("writer_counters", self.counters(py)?)?;
        let _ = self.emit_event(
            py,
            run_id,
            "telemetry_session_closed",
            "debug",
            timestamp,
            process_identifier,
            thread_name,
            &fields,
        );
        self.finish(py)
    }

    pub fn finish_with_close_event_metadata<'py>(
        &self,
        py: Python<'py>,
        run_id: &str,
        timestamp: &str,
        process_identifier: u32,
        thread_name: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fields = PyDict::new(py);
        fields.set_item("writer_counters", self.counters(py)?)?;
        let _ = self.emit_event(
            py,
            run_id,
            "telemetry_session_closed",
            "debug",
            timestamp,
            process_identifier,
            thread_name,
            &fields,
        );
        self.finish_close_metadata(py)
    }

    pub fn finish_with_current_close_event<'py>(&self, py: Python<'py>, run_id: &str) -> PyResult<Bound<'py, PyDict>> {
        let fields = PyDict::new(py);
        fields.set_item("writer_counters", self.counters(py)?)?;
        let _ = self.emit_current_event(py, run_id, "telemetry_session_closed", "debug", &fields);
        self.finish(py)
    }

    pub fn finish_with_current_close_event_metadata<'py>(
        &self,
        py: Python<'py>,
        run_id: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fields = PyDict::new(py);
        fields.set_item("writer_counters", self.counters(py)?)?;
        let _ = self.emit_current_event(py, run_id, "telemetry_session_closed", "debug", &fields);
        self.finish_close_metadata(py)
    }
}

impl NativeTelemetrySession {
    fn finish_counter_snapshot(&self) -> PyResult<native_telemetry_session::TelemetryWriterCounterSnapshot> {
        let finish_start_time = Instant::now();
        let mut writer_guard =
            self.writer.lock().map_err(|_| PyRuntimeError::new_err("Telemetry writer mutex was poisoned."))?;
        let dropped_writer = writer_guard.take();
        let mut guard =
            self.guard.lock().map_err(|_| PyRuntimeError::new_err("Telemetry guard mutex was poisoned."))?;
        let dropped_guard = guard.take();
        drop(dropped_guard);
        let finish_flush_duration_seconds = finish_start_time.elapsed().as_secs_f64();
        clear_shared_telemetry_writer(&self.path)?;
        let Some(writer) = dropped_writer.as_ref() else {
            let last_counter_snapshot = self
                .last_counter_snapshot
                .lock()
                .map_err(|_| PyRuntimeError::new_err("Telemetry counter snapshot mutex was poisoned."))?;
            let Some(counter_snapshot) = last_counter_snapshot.as_ref() else {
                return Ok(native_telemetry_session::TelemetryWriterCounterSnapshot::empty());
            };
            return Ok(counter_snapshot.clone());
        };

        let counter_snapshot = writer.counter_snapshot(Some(finish_flush_duration_seconds));
        let mut last_counter_snapshot = self
            .last_counter_snapshot
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Telemetry counter snapshot mutex was poisoned."))?;
        *last_counter_snapshot = Some(counter_snapshot.clone());
        writer.fail_if_lossless_cap_exceeded()?;
        Ok(counter_snapshot)
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

impl From<native_telemetry_session::TelemetryEventEmissionPlan> for NativeTelemetryEventEmissionPlan {
    fn from(emission_plan: native_telemetry_session::TelemetryEventEmissionPlan) -> Self {
        Self { inner: emission_plan }
    }
}

impl From<native_telemetry_session::TelemetryProgressEmissionPlan> for NativeTelemetryProgressEmissionPlan {
    fn from(emission_plan: native_telemetry_session::TelemetryProgressEmissionPlan) -> Self {
        Self { inner: emission_plan }
    }
}

impl From<native_telemetry_session::TelemetryClosePlan> for NativeTelemetryClosePlan {
    fn from(close_plan: native_telemetry_session::TelemetryClosePlan) -> Self {
        Self { inner: close_plan }
    }
}

#[pyfunction]
pub fn plan_telemetry_event_emission(
    telemetry_enabled: bool,
    has_native_telemetry_session: bool,
) -> NativeTelemetryEventEmissionPlan {
    native_telemetry_session::plan_telemetry_event_emission(telemetry_enabled, has_native_telemetry_session).into()
}

#[pyfunction]
pub fn plan_telemetry_progress_emission(
    telemetry_enabled: bool,
    has_native_telemetry_session: bool,
    should_emit_progress: bool,
) -> NativeTelemetryProgressEmissionPlan {
    native_telemetry_session::plan_telemetry_progress_emission(
        telemetry_enabled,
        has_native_telemetry_session,
        should_emit_progress,
    )
    .into()
}

#[pyfunction]
pub fn plan_telemetry_close(
    has_telemetry_session: bool,
    is_native_telemetry_session: bool,
) -> NativeTelemetryClosePlan {
    native_telemetry_session::plan_telemetry_close(has_telemetry_session, is_native_telemetry_session).into()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_telemetry_event_payload<'py>(
    py: Python<'py>,
    run_id: &str,
    event: &str,
    level: &str,
    timestamp: &str,
    process_identifier: u32,
    thread_name: &str,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let envelope = native_telemetry_session::build_telemetry_event_envelope(
        run_id,
        event,
        level,
        timestamp,
        process_identifier,
        thread_name,
    );
    telemetry_event_envelope_to_py_dict(py, &envelope, fields)
}

#[pyfunction]
pub fn build_current_telemetry_event_payload<'py>(
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

#[pyfunction]
pub fn generate_telemetry_run_id_value() -> String {
    native_telemetry_session::generate_run_id()
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

fn serialize_telemetry_payload_json_line(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<String> {
    let json_module = PyModule::import(py, "json")?;
    let builtins_module = PyModule::import(py, "builtins")?;
    let keyword_arguments = PyDict::new(py);
    keyword_arguments.set_item("sort_keys", true)?;
    keyword_arguments.set_item("default", builtins_module.getattr("str")?)?;
    let json_text = json_module.call_method("dumps", (payload,), Some(&keyword_arguments))?.extract::<String>()?;
    Ok(format!("{json_text}\n"))
}

#[pyfunction]
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

#[pyfunction]
#[expect(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    reason = "PyO3 exposes documented Python logging keyword arguments directly."
)]
#[pyo3(signature = (
    log_filter=None,
    log_file=None,
    log_stderr=true,
    log_queue_size=65536,
    log_lossy=true,
    include_source_location=false,
    include_span_events=false,
    trace_file=None,
    trace_filter=None,
    trace_event_cap=None
))]
pub fn initialize_logging(
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
    let mut logging_guards = lock_logging_guards()?;
    if logging_guards.is_some() {
        setup_python_logging(py)?;
        return Ok(false);
    }

    let resolved_log_filter = log_filter
        .filter(|candidate_filter| !candidate_filter.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_LOG_FILTER.to_string());
    let environment_filter = EnvFilter::try_new(&resolved_log_filter)
        .map_err(|error| PyValueError::new_err(format!("Invalid g log filter: {error}")))?;

    let mut worker_guards = Vec::new();
    let stderr_layer = if log_stderr {
        let (stderr_writer, stderr_guard) =
            build_non_blocking_writer(std::io::stderr(), "g-tracing-stderr", log_queue_size, log_lossy);
        worker_guards.push(stderr_guard);
        let layer = tracing_subscriber::fmt::layer()
            .compact()
            .with_writer(stderr_writer)
            .with_ansi(true)
            .with_file(include_source_location)
            .with_line_number(include_source_location)
            .with_span_events(resolve_span_events(include_span_events));
        Some(layer.boxed())
    } else {
        None
    };
    let file_layer = if let Some(log_file_path) = log_file {
        let (file_writer, maybe_file_guard) =
            build_shared_or_log_file_writer(Path::new(&log_file_path), log_queue_size, log_lossy, None)?;
        if let Some(file_guard) = maybe_file_guard {
            worker_guards.push(file_guard);
        }
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_ansi(false)
            .with_writer(file_writer)
            .with_file(include_source_location)
            .with_line_number(include_source_location)
            .with_span_events(resolve_span_events(include_span_events));
        Some(layer.boxed())
    } else {
        None
    };
    let trace_layer = if let Some(trace_file_path) = trace_file {
        let (trace_writer, maybe_trace_guard) = build_shared_or_log_file_writer(
            Path::new(&trace_file_path),
            log_queue_size,
            log_lossy,
            normalize_event_cap(trace_event_cap),
        )?;
        if let Some(trace_guard) = maybe_trace_guard {
            worker_guards.push(trace_guard);
        }
        let resolved_trace_filter = trace_filter
            .filter(|candidate_filter| !candidate_filter.trim().is_empty())
            .unwrap_or_else(|| resolved_log_filter.clone());
        let trace_environment_filter = EnvFilter::try_new(&resolved_trace_filter)
            .map_err(|error| PyValueError::new_err(format!("Invalid g trace filter: {error}")))?;
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_ansi(false)
            .with_writer(trace_writer)
            .with_file(true)
            .with_line_number(true)
            .with_span_events(FmtSpan::FULL)
            .with_filter(trace_environment_filter);
        Some(layer.boxed())
    } else {
        None
    };

    let subscriber =
        tracing_subscriber::registry().with(environment_filter).with(stderr_layer).with(file_layer).with(trace_layer);
    if subscriber.try_init().is_err() {
        setup_python_logging(py)?;
        return Ok(false);
    }

    setup_python_logging(py)?;
    *logging_guards = Some(worker_guards);
    tracing::info!(target: "g.logging", "logging initialized");
    Ok(true)
}

#[pyfunction]
pub fn shutdown_logging() -> PyResult<()> {
    let mut logging_guards = lock_logging_guards()?;
    let _dropped_guards = logging_guards.take();
    Ok(())
}

fn lock_logging_guards() -> PyResult<std::sync::MutexGuard<'static, Option<Vec<WorkerGuard>>>> {
    LOGGING_GUARDS.lock().map_err(|_| PyRuntimeError::new_err("Logging guard mutex was poisoned."))
}

fn lock_telemetry_writer() -> PyResult<std::sync::MutexGuard<'static, Option<SharedTelemetryWriter>>> {
    TELEMETRY_WRITER.lock().map_err(|_| PyRuntimeError::new_err("Telemetry writer mutex was poisoned."))
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
    let shutdown_logging_function = core_module.getattr("shutdown_logging")?;
    let atexit = py.import("atexit")?;
    atexit.call_method1("register", (shutdown_logging_function,))?;
    Ok(())
}

fn resolve_span_events(include_span_events: bool) -> FmtSpan {
    if include_span_events { FmtSpan::FULL } else { FmtSpan::NONE }
}

fn normalize_event_cap(event_cap: Option<usize>) -> Option<usize> {
    event_cap.filter(|cap| *cap > 0)
}

fn telemetry_event_cap_to_usize(event_cap: Option<i64>) -> PyResult<Option<usize>> {
    event_cap
        .map(|value| {
            usize::try_from(value).map_err(|_| PyValueError::new_err("Telemetry event cap must be non-negative."))
        })
        .transpose()
}

fn build_log_file_writer(path: &Path, log_queue_size: usize, log_lossy: bool) -> PyResult<(NonBlocking, WorkerGuard)> {
    if let Some(parent_directory) = path.parent().filter(|parent_directory| !parent_directory.as_os_str().is_empty()) {
        fs::create_dir_all(parent_directory).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    }
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(build_non_blocking_writer(log_file, "g-tracing-file", log_queue_size, log_lossy))
}

fn build_telemetry_file_writer(
    path: &Path,
    log_queue_size: usize,
    log_lossy: bool,
    event_cap: Option<usize>,
) -> PyResult<(TelemetryWriterFactory, WorkerGuard)> {
    let (writer, guard) = build_log_file_writer(path, log_queue_size, log_lossy)?;
    let event_cap_state = native_telemetry_session::TelemetryEventCapState::new(path, event_cap, log_lossy);
    Ok((TelemetryWriterFactory::new(writer, event_cap_state), guard))
}

fn build_shared_or_log_file_writer(
    path: &Path,
    log_queue_size: usize,
    log_lossy: bool,
    event_cap: Option<usize>,
) -> PyResult<(TelemetryWriterFactory, Option<WorkerGuard>)> {
    if let Some(shared_writer) = shared_telemetry_writer_for_path(path)? {
        return Ok((shared_writer, None));
    }
    let (writer, guard) = build_telemetry_file_writer(path, log_queue_size, log_lossy, event_cap)?;
    Ok((writer, Some(guard)))
}

fn shared_telemetry_writer_for_path(path: &Path) -> PyResult<Option<TelemetryWriterFactory>> {
    let normalized_path = normalize_path_for_comparison(path);
    let telemetry_writer = lock_telemetry_writer()?;
    Ok(telemetry_writer
        .as_ref()
        .filter(|shared_writer| normalize_path_for_comparison(&shared_writer.path) == normalized_path)
        .map(|shared_writer| shared_writer.writer.clone()))
}

fn replace_shared_telemetry_writer(path: PathBuf, writer: TelemetryWriterFactory) -> PyResult<()> {
    let mut telemetry_writer = lock_telemetry_writer()?;
    *telemetry_writer = Some(SharedTelemetryWriter { path, writer });
    Ok(())
}

fn clear_shared_telemetry_writer(path: &Path) -> PyResult<()> {
    let normalized_path = normalize_path_for_comparison(path);
    let mut telemetry_writer = lock_telemetry_writer()?;
    if telemetry_writer
        .as_ref()
        .is_some_and(|shared_writer| normalize_path_for_comparison(&shared_writer.path) == normalized_path)
    {
        let _dropped_writer = telemetry_writer.take();
    }
    Ok(())
}

fn normalize_path_for_comparison(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn build_non_blocking_writer<Writer>(
    writer: Writer,
    thread_name: &str,
    log_queue_size: usize,
    log_lossy: bool,
) -> (NonBlocking, WorkerGuard)
where
    Writer: std::io::Write + Send + 'static,
{
    NonBlockingBuilder::default()
        .lossy(log_lossy)
        .buffered_lines_limit(log_queue_size)
        .thread_name(thread_name)
        .finish(writer)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static NEXT_TEST_FILE_ID: AtomicUsize = AtomicUsize::new(0);

    fn telemetry_test_path(test_name: &str) -> PathBuf {
        let file_id = NEXT_TEST_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("g-{test_name}-{}-{file_id}.jsonl", std::process::id()))
    }

    #[test]
    fn telemetry_event_cap_fails_without_lossy_mode() {
        let path = telemetry_test_path("telemetry-cap-fails");
        let (telemetry_writer, guard) =
            build_telemetry_file_writer(&path, 32, false, Some(2)).expect("writer should build");

        telemetry_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should write");
        let error =
            telemetry_writer.write_json_line(r#"{"event":"third"}"#).expect_err("third event should exceed cap");

        assert!(error.to_string().contains("Trace telemetry event cap exceeded at 2 events"));
        assert!(telemetry_writer.fail_if_lossless_cap_exceeded().is_err());
        drop(telemetry_writer);
        drop(guard);

        let line_count = fs::read_to_string(&path).expect("telemetry file should be readable").lines().count();
        assert_eq!(line_count, 2);
        fs::remove_file(path).expect("telemetry test file should be removed");
    }

    #[test]
    fn telemetry_event_cap_drops_with_lossy_mode() {
        let path = telemetry_test_path("telemetry-cap-drops");
        let (telemetry_writer, guard) =
            build_telemetry_file_writer(&path, 32, true, Some(1)).expect("writer should build");

        telemetry_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should drop");
        telemetry_writer.write_json_line(r#"{"event":"third"}"#).expect("third event should drop");
        assert!(telemetry_writer.fail_if_lossless_cap_exceeded().is_ok());
        drop(telemetry_writer);
        drop(guard);

        let telemetry_text = fs::read_to_string(&path).expect("telemetry file should be readable");
        assert_eq!(telemetry_text.lines().count(), 1);
        assert!(telemetry_text.contains(r#""event":"first""#));
        fs::remove_file(path).expect("telemetry test file should be removed");
    }

    #[test]
    fn telemetry_event_cap_zero_disables_cap() {
        let path = telemetry_test_path("telemetry-cap-disabled");
        let (telemetry_writer, guard) =
            build_telemetry_file_writer(&path, 32, false, normalize_event_cap(Some(0))).expect("writer should build");

        telemetry_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should write");
        telemetry_writer.write_json_line(r#"{"event":"third"}"#).expect("third event should write");
        assert!(telemetry_writer.fail_if_lossless_cap_exceeded().is_ok());
        drop(telemetry_writer);
        drop(guard);

        let line_count = fs::read_to_string(&path).expect("telemetry file should be readable").lines().count();
        assert_eq!(line_count, 3);
        fs::remove_file(path).expect("telemetry test file should be removed");
    }
}
