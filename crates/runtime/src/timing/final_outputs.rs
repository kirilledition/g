use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Serialize;

const FINAL_TIMING_OUTPUTS_WRITE_STARTED_EVENT_LEVEL: &str = "debug";
const FINAL_TIMING_OUTPUTS_WRITE_STARTED_EVENT_NAME: &str = "runner_final_timing_outputs_write_started";
pub const FINAL_TIMING_OUTPUTS_WRITE_STARTED_MESSAGE: &str = "Writing final timing outputs.";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageTimingRecorderPlan {
    pub should_create: bool,
    pub exact_stage_timings: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimingFileWritePlan {
    pub should_write: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalTimingOutputsWriteStartedDiagnosticPayload {
    pub level: &'static str,
    pub event_name: &'static str,
    pub message: &'static str,
    pub stage_timing_path: Option<String>,
    pub profile_summary_path: Option<String>,
    pub run_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalTimingOutputsWriteResultPayload {
    pub wrote_stage_timing_snapshot: bool,
    pub wrote_profile_summary: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalTimingOutputContext {
    pub stage_timing_path: Option<String>,
    pub profile_summary_path: Option<String>,
    pub run_id: Option<String>,
    pub force_stage_timing_recorder: bool,
}

#[derive(Debug)]
pub enum TimingFileError {
    CreateParentDirectory { path: PathBuf, source: std::io::Error },
    Serialize { source: serde_json::Error },
    WriteFile { path: PathBuf, source: std::io::Error },
}

impl fmt::Display for TimingFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateParentDirectory { path, source } => {
                write!(formatter, "failed to create timing file parent directory for {}: {source}", path.display())
            }
            Self::Serialize { source } => write!(formatter, "failed to serialize timing payload: {source}"),
            Self::WriteFile { path, source } => {
                write!(formatter, "failed to write timing file {}: {source}", path.display())
            }
        }
    }
}

impl Error for TimingFileError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CreateParentDirectory { source, .. } | Self::WriteFile { source, .. } => Some(source),
            Self::Serialize { source } => Some(source),
        }
    }
}

#[must_use]
pub const fn plan_stage_timing_recorder(stage_timing_path_configured: bool, force: bool) -> StageTimingRecorderPlan {
    StageTimingRecorderPlan {
        should_create: stage_timing_path_configured || force,
        exact_stage_timings: stage_timing_path_configured,
    }
}

#[must_use]
pub const fn plan_timing_file_write(has_stage_timing_recorder: bool, path_configured: bool) -> TimingFileWritePlan {
    TimingFileWritePlan { should_write: has_stage_timing_recorder && path_configured }
}

#[must_use]
pub fn build_final_timing_outputs_write_started_diagnostic_payload(
    stage_timing_path: Option<&str>,
    profile_summary_path: Option<&str>,
    run_id: Option<&str>,
) -> FinalTimingOutputsWriteStartedDiagnosticPayload {
    FinalTimingOutputsWriteStartedDiagnosticPayload {
        level: FINAL_TIMING_OUTPUTS_WRITE_STARTED_EVENT_LEVEL,
        event_name: FINAL_TIMING_OUTPUTS_WRITE_STARTED_EVENT_NAME,
        message: FINAL_TIMING_OUTPUTS_WRITE_STARTED_MESSAGE,
        stage_timing_path: stage_timing_path.map(str::to_string),
        profile_summary_path: profile_summary_path.map(str::to_string),
        run_id: run_id.map(str::to_string),
    }
}

/// Serialize final timing output diagnostic fields for native diagnostic emission.
///
/// This keeps the event payload's JSON field shape in `g-runtime`; PyO3 callers
/// only pass the serialized fields through to the logging boundary.
///
/// # Errors
///
/// Returns a serialization error if the diagnostic field payload cannot be
/// encoded as JSON.
pub fn serialize_final_timing_outputs_write_started_diagnostic_fields_json(
    payload: &FinalTimingOutputsWriteStartedDiagnosticPayload,
) -> Result<String, serde_json::Error> {
    serde_json::to_string(&serde_json::json!({
        "stage_timing_path": payload.stage_timing_path.as_deref(),
        "profile_summary_path": payload.profile_summary_path.as_deref(),
        "run_id": payload.run_id.as_deref(),
    }))
}

#[must_use]
pub fn resolve_final_timing_output_context(
    diagnostics_stage_timing_path: Option<&str>,
    telemetry_stage_timing_path: Option<&str>,
    telemetry_profile_summary_path: Option<&str>,
    telemetry_run_id: Option<&str>,
    telemetry_profile_enabled: bool,
    has_telemetry_session: bool,
) -> FinalTimingOutputContext {
    if has_telemetry_session {
        return FinalTimingOutputContext {
            stage_timing_path: telemetry_stage_timing_path.map(str::to_string),
            profile_summary_path: telemetry_profile_summary_path.map(str::to_string),
            run_id: telemetry_run_id.map(str::to_string),
            force_stage_timing_recorder: telemetry_profile_enabled,
        };
    }
    FinalTimingOutputContext {
        stage_timing_path: diagnostics_stage_timing_path.map(str::to_string),
        profile_summary_path: None,
        run_id: None,
        force_stage_timing_recorder: false,
    }
}

pub(super) fn write_pretty_json_payload<T>(path: &Path, payload: &T) -> Result<(), TimingFileError>
where
    T: Serialize,
{
    if let Some(parent_directory) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent_directory).map_err(|source| TimingFileError::CreateParentDirectory {
            path: parent_directory.to_path_buf(),
            source,
        })?;
    }
    let payload_text = serde_json::to_string_pretty(payload).map_err(|source| TimingFileError::Serialize { source })?;
    std::fs::write(path, format!("{payload_text}\n"))
        .map_err(|source| TimingFileError::WriteFile { path: path.to_path_buf(), source })
}
