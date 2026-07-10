//! Native telemetry session at the Python thread-name and error boundary.

#![allow(clippy::missing_errors_doc)]

use std::path::Path;
use std::sync::Mutex;

use super::errors;
use super::telemetry_policy;
use g_runtime as native_telemetry_session;
use g_runtime as native_telemetry_writer;
use g_runtime as native_run_events;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde::Serialize;

pub(crate) struct NativeTelemetryRunSession {
    state: Mutex<native_telemetry_session::TelemetryRunSessionState>,
    native_telemetry_session: Option<NativeTelemetrySession>,
}

impl NativeTelemetryRunSession {
    pub(crate) fn new(
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

        Ok(Self { state: Mutex::new(state), native_telemetry_session })
    }

    pub(crate) fn emit_current_event<Fields>(&self, event: &str, level: &str, fields: &Fields) -> PyResult<()>
    where
        Fields: Serialize,
    {
        self.emit_current_event_fields(event, level, fields)
    }

    pub(crate) fn emit_run_failed_event(&self, event: &native_run_events::RunFailedEventPayload) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_run_failed_telemetry_fields(event);
        self.emit_telemetry_event_fields(native_run_events::RunTelemetryEventKind::RunFailed, &telemetry_fields)
    }

    pub(crate) fn emit_execution_plan_prepared_event(
        &self,
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
        self.emit_telemetry_event_fields(
            native_run_events::RunTelemetryEventKind::ExecutionPlanPrepared,
            &telemetry_fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn emit_phenotype_writer_finished_event(
        &self,
        association_mode: &str,
        phenotype: &str,
        final_output_path: Option<String>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_phenotype_writer_finished_telemetry_fields(
            association_mode,
            phenotype,
            final_output_path.as_deref(),
        );
        self.emit_telemetry_event_fields(native_run_events::RunTelemetryEventKind::WriterFinished, &telemetry_fields)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn emit_multi_phenotype_writer_finished_event(
        &self,
        association_mode: &str,
        phenotype_count: i64,
        final_output_paths: Vec<Option<String>>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_multi_phenotype_writer_finished_telemetry_fields(
            association_mode,
            phenotype_count,
            &final_output_paths,
        );
        self.emit_telemetry_event_fields(native_run_events::RunTelemetryEventKind::WriterFinished, &telemetry_fields)
    }

    pub(crate) fn emit_single_trait_preflight_completed_event(
        &self,
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
        self.emit_telemetry_event_fields(
            native_run_events::RunTelemetryEventKind::PreflightCompleted,
            &telemetry_fields,
        )
    }

    pub(crate) fn emit_multi_phenotype_preflight_completed_event(
        &self,
        association_mode: &str,
        phenotype_count: i64,
        sample_count: i64,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_multi_phenotype_preflight_completed_telemetry_fields(
            association_mode,
            phenotype_count,
            sample_count,
        );
        self.emit_telemetry_event_fields(
            native_run_events::RunTelemetryEventKind::PreflightCompleted,
            &telemetry_fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn emit_sample_alignment_completed_event(
        &self,
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
        self.emit_telemetry_event_fields(
            native_run_events::RunTelemetryEventKind::SampleAlignmentCompleted,
            &telemetry_fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn emit_prediction_source_loaded_event(
        &self,
        association_mode: &str,
        phenotype: Option<String>,
        phenotype_count: Option<i64>,
    ) -> PyResult<()> {
        let telemetry_fields = native_run_events::build_prediction_source_loaded_telemetry_fields(
            association_mode,
            phenotype.as_deref(),
            phenotype_count,
        );
        self.emit_telemetry_event_fields(
            native_run_events::RunTelemetryEventKind::PredictionSourceLoaded,
            &telemetry_fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn emit_multi_phenotype_sample_summary_event(
        &self,
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
        self.emit_telemetry_event_fields(
            native_run_events::RunTelemetryEventKind::MultiPhenotypeSampleSummary,
            &telemetry_fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn emit_gpu_genotype_format_resolved_event(
        &self,
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
        self.emit_telemetry_event_fields(
            native_run_events::RunTelemetryEventKind::GpuGenotypeFormatResolved,
            &telemetry_fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn emit_association_backend_selected_event(
        &self,
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
        self.emit_telemetry_event_fields(
            native_run_events::RunTelemetryEventKind::AssociationBackendSelected,
            &telemetry_fields,
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn emit_bgen_engine_opened_event(
        &self,
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
        self.emit_telemetry_event_fields(native_run_events::RunTelemetryEventKind::BgenEngineOpened, &telemetry_fields)
    }

    pub(crate) fn finish_with_current_close_event(&self) -> PyResult<()> {
        let Some(native_telemetry_session) = self.native_telemetry_session.as_ref() else {
            return Ok(());
        };
        native_telemetry_session.finish_with_current_close_event(&self.run_id_value()?)
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

    fn emit_current_event<Fields>(&self, run_id: &str, event: &str, level: &str, fields: &Fields) -> PyResult<()>
    where
        Fields: Serialize,
    {
        let thread_name = current_python_thread_name()?;
        let envelope =
            native_telemetry_session::build_current_telemetry_event_envelope(run_id, event, level, &thread_name);
        let json_line = native_telemetry_session::serialize_telemetry_event_json_line(&envelope, fields)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        self.emit_json_line(&json_line)
    }

    fn finish_with_current_close_event(&self, run_id: &str) -> PyResult<()> {
        let _ = self.emit_current_close_event(run_id);
        self.lock_writer()?
            .finish_counter_snapshot()
            .map_err(|error| errors::convert_telemetry_writer_error(&error))?;
        Ok(())
    }

    fn counter_snapshot(&self) -> PyResult<native_telemetry_session::TelemetryWriterCounterSnapshot> {
        Ok(self.lock_writer()?.counter_snapshot())
    }

    fn emit_current_close_event(&self, run_id: &str) -> PyResult<()> {
        let close_event_payload =
            native_telemetry_session::build_telemetry_close_event_payload(self.counter_snapshot()?);
        self.emit_current_event(
            run_id,
            &close_event_payload.event_name,
            &close_event_payload.level,
            &close_event_payload,
        )
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

    fn emit_current_event_fields<Fields>(&self, event: &str, level: &str, fields: &Fields) -> PyResult<()>
    where
        Fields: Serialize,
    {
        let emission_plan = self.state_guard()?.plan_event_emission(self.native_telemetry_session.is_some());
        if !emission_plan.should_emit {
            return Ok(());
        }
        let Some(native_telemetry_session) = self.native_telemetry_session.as_ref() else {
            return Ok(());
        };
        native_telemetry_session.emit_current_event(&self.run_id_value()?, event, level, fields)
    }

    fn emit_telemetry_event_fields<Fields>(
        &self,
        event_kind: native_run_events::RunTelemetryEventKind,
        fields: &Fields,
    ) -> PyResult<()>
    where
        Fields: Serialize,
    {
        self.emit_current_event_fields(event_kind.event_name(), event_kind.level(), fields)
    }
}

fn current_python_thread_name() -> PyResult<String> {
    Python::attach(|py| {
        let threading_module = PyModule::import(py, "threading")?;
        threading_module.call_method0("current_thread")?.getattr("name")?.extract::<String>()
    })
}

fn telemetry_event_cap_to_usize(event_cap: Option<i64>) -> PyResult<Option<usize>> {
    event_cap
        .map(|value| {
            usize::try_from(value).map_err(|_| PyValueError::new_err("Telemetry event cap must be non-negative."))
        })
        .transpose()
}
