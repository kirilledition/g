use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use serde::Serialize;

use crate::run_events::{
    RunFailedEventPayload, RunTelemetryEventKind, build_association_backend_selected_telemetry_fields,
    build_execution_plan_prepared_telemetry_fields, build_multi_phenotype_writer_finished_telemetry_fields,
    build_phenotype_writer_finished_telemetry_fields, build_run_failed_telemetry_fields,
};
use crate::telemetry_writer::TelemetrySessionWriter;

use super::{build_current_telemetry_event_envelope, generate_run_id, serialize_telemetry_event_json_line};

const TELEMETRY_SESSION_CLOSED_EVENT_NAME: &str = "telemetry_session_closed";
const TELEMETRY_SESSION_CLOSED_EVENT_LEVEL: &str = "debug";

#[derive(Debug)]
pub enum TelemetryRunError {
    MissingStreamFile,
    EventCapOutOfRange,
    WriterLockPoisoned,
    Io(io::Error),
    Serialization(serde_json::Error),
}

impl fmt::Display for TelemetryRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingStreamFile => {
                formatter.write_str("Telemetry stream file is required when telemetry is enabled.")
            }
            Self::EventCapOutOfRange => formatter.write_str("Telemetry event cap does not fit native usize."),
            Self::WriterLockPoisoned => formatter.write_str("Telemetry writer lock was poisoned."),
            Self::Io(error) => error.fmt(formatter),
            Self::Serialization(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TelemetryRunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Serialization(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for TelemetryRunError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for TelemetryRunError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

pub struct TelemetryRunSession {
    run_id: String,
    writer: Option<Mutex<TelemetrySessionWriter>>,
}

impl TelemetryRunSession {
    /// Open the telemetry writer selected by the canonical runtime policy.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured stream is missing, its cap is out
    /// of range, or the writer cannot be opened.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        telemetry_mode: g_plan::TelemetryMode,
        stream_file: Option<PathBuf>,
        queue_size: usize,
        lossy: bool,
        trace_event_cap: i64,
        run_id: Option<String>,
    ) -> Result<Self, TelemetryRunError> {
        let writer = if telemetry_mode == g_plan::TelemetryMode::Off {
            None
        } else {
            let stream_file = stream_file.ok_or(TelemetryRunError::MissingStreamFile)?;
            let event_cap = (telemetry_mode == g_plan::TelemetryMode::Trace && trace_event_cap > 0)
                .then_some(trace_event_cap)
                .map(usize::try_from)
                .transpose()
                .map_err(|_| TelemetryRunError::EventCapOutOfRange)?;
            Some(Mutex::new(TelemetrySessionWriter::new(stream_file, queue_size, lossy, event_cap)?))
        };
        Ok(Self { run_id: run_id.unwrap_or_else(generate_run_id), writer })
    }

    /// Serialize and emit one typed telemetry event.
    ///
    /// # Errors
    ///
    /// Returns an error when session state is unavailable, serialization
    /// fails, or the writer rejects the event.
    pub fn emit_current_event<Fields>(
        &self,
        thread_name: &str,
        event: &str,
        level: &str,
        fields: &Fields,
    ) -> Result<(), TelemetryRunError>
    where
        Fields: Serialize,
    {
        let Some(_) = self.writer.as_ref() else {
            return Ok(());
        };
        let envelope = build_current_telemetry_event_envelope(&self.run_id, event, level, thread_name);
        let json_line = serialize_telemetry_event_json_line(&envelope, fields)?;
        self.writer_guard()?.write_json_line(&json_line)?;
        Ok(())
    }

    /// Emit the canonical run-failed event.
    ///
    /// # Errors
    ///
    /// Returns a telemetry state, serialization, or writer error.
    pub fn emit_run_failed_event(
        &self,
        thread_name: &str,
        event: &RunFailedEventPayload,
    ) -> Result<(), TelemetryRunError> {
        self.emit_kind(thread_name, RunTelemetryEventKind::RunFailed, &build_run_failed_telemetry_fields(event))
    }

    /// Emit the canonical execution-plan event.
    ///
    /// # Errors
    ///
    /// Returns a telemetry state, serialization, or writer error.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_execution_plan_prepared_event(
        &self,
        thread_name: &str,
        association_mode: &str,
        trait_type: &str,
        phenotype_count: i64,
        chunk_size: i64,
        variant_limit: Option<i64>,
        device: &str,
    ) -> Result<(), TelemetryRunError> {
        let fields = build_execution_plan_prepared_telemetry_fields(
            association_mode,
            trait_type,
            phenotype_count,
            chunk_size,
            variant_limit,
            device,
        );
        self.emit_kind(thread_name, RunTelemetryEventKind::ExecutionPlanPrepared, &fields)
    }

    /// Emit completion for one phenotype writer.
    ///
    /// # Errors
    ///
    /// Returns a telemetry state, serialization, or writer error.
    pub fn emit_phenotype_writer_finished_event(
        &self,
        thread_name: &str,
        association_mode: &str,
        phenotype: &str,
        parquet_dataset_path: &str,
    ) -> Result<(), TelemetryRunError> {
        let fields =
            build_phenotype_writer_finished_telemetry_fields(association_mode, phenotype, parquet_dataset_path);
        self.emit_kind(thread_name, RunTelemetryEventKind::WriterFinished, &fields)
    }

    /// Emit completion for a multi-phenotype writer set.
    ///
    /// # Errors
    ///
    /// Returns a telemetry state, serialization, or writer error.
    pub fn emit_multi_phenotype_writer_finished_event(
        &self,
        thread_name: &str,
        association_mode: &str,
        phenotype_count: i64,
        parquet_dataset_paths: &[&str],
    ) -> Result<(), TelemetryRunError> {
        let fields = build_multi_phenotype_writer_finished_telemetry_fields(
            association_mode,
            phenotype_count,
            parquet_dataset_paths,
        );
        self.emit_kind(thread_name, RunTelemetryEventKind::WriterFinished, &fields)
    }

    /// Emit the resolved association-backend event.
    ///
    /// # Errors
    ///
    /// Returns a telemetry state, serialization, or writer error.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_association_backend_selected_event(
        &self,
        thread_name: &str,
        association_mode: &str,
        association_backend_kind: &str,
        device: &str,
        genotype_format: &str,
        phenotype: Option<&str>,
        phenotype_count: Option<i64>,
    ) -> Result<(), TelemetryRunError> {
        let fields = build_association_backend_selected_telemetry_fields(
            association_mode,
            association_backend_kind,
            device,
            genotype_format,
            phenotype,
            phenotype_count,
        );
        self.emit_kind(thread_name, RunTelemetryEventKind::AssociationBackendSelected, &fields)
    }

    /// Emit close counters and flush the owned writer.
    ///
    /// # Errors
    ///
    /// Returns a telemetry state, serialization, writer, or flush error.
    pub fn finish(&self, thread_name: &str) -> Result<(), TelemetryRunError> {
        if self.writer.is_none() {
            return Ok(());
        }
        let writer_counters = self.writer_guard()?.counter_snapshot();
        let close_result = self.emit_current_event(
            thread_name,
            TELEMETRY_SESSION_CLOSED_EVENT_NAME,
            TELEMETRY_SESSION_CLOSED_EVENT_LEVEL,
            &writer_counters,
        );
        let finish_result = self.writer_guard()?.finish().map_err(Into::into);
        close_result.and(finish_result)
    }

    fn emit_kind<Fields>(
        &self,
        thread_name: &str,
        event_kind: RunTelemetryEventKind,
        fields: &Fields,
    ) -> Result<(), TelemetryRunError>
    where
        Fields: Serialize,
    {
        self.emit_current_event(thread_name, event_kind.event_name(), event_kind.level(), fields)
    }

    fn writer_guard(&self) -> Result<MutexGuard<'_, TelemetrySessionWriter>, TelemetryRunError> {
        self.writer
            .as_ref()
            .ok_or(TelemetryRunError::MissingStreamFile)?
            .lock()
            .map_err(|_| TelemetryRunError::WriterLockPoisoned)
    }
}
