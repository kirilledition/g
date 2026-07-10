//! Native stage timing aggregation and final output.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Serialize;

const FINAL_TIMING_OUTPUTS_WRITE_STARTED_EVENT_LEVEL: &str = "debug";
const FINAL_TIMING_OUTPUTS_WRITE_STARTED_EVENT_NAME: &str = "runner_final_timing_outputs_write_started";
pub const FINAL_TIMING_OUTPUTS_WRITE_STARTED_MESSAGE: &str = "Writing final timing outputs.";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalTimingOutputsWriteStartedDiagnosticPayload {
    pub level: &'static str,
    pub event_name: &'static str,
    pub message: &'static str,
    pub stage_timing_path: Option<String>,
    pub profile_summary_path: Option<String>,
    pub run_id: Option<String>,
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageTimingRecorder {
    stage_totals_seconds: BTreeMap<String, f64>,
    stage_counts: BTreeMap<String, u64>,
}

#[derive(Serialize)]
struct ProfileSummaryPayload<'recorder> {
    schema_version: i64,
    run_id: Option<String>,
    stage_totals_seconds: &'recorder BTreeMap<String, f64>,
    stage_counts: &'recorder BTreeMap<String, u64>,
}

#[derive(Serialize)]
struct StageTimingSnapshotPayload<'recorder> {
    stage_totals_seconds: &'recorder BTreeMap<String, f64>,
    stage_counts: &'recorder BTreeMap<String, u64>,
}

impl StageTimingRecorder {
    #[must_use]
    pub fn from_config(stage_timing_path_configured: bool, force: bool) -> Option<Self> {
        (stage_timing_path_configured || force).then(Self::default)
    }

    pub fn add_stage_duration(&mut self, stage_name: String, duration_seconds: f64) {
        *self.stage_totals_seconds.entry(stage_name.clone()).or_insert(0.0) += duration_seconds;
        let stage_count = self.stage_counts.entry(stage_name).or_insert(0);
        *stage_count = stage_count.saturating_add(1);
    }

    /// Write every configured final timing output.
    ///
    /// # Errors
    ///
    /// Returns an error when a timing payload cannot be written.
    pub fn write_final_timing_outputs(
        &self,
        stage_timing_path: Option<&Path>,
        profile_summary_path: Option<&Path>,
        run_id: Option<String>,
    ) -> Result<(), TimingFileError> {
        if let Some(path) = stage_timing_path {
            write_pretty_json_payload(
                path,
                &StageTimingSnapshotPayload {
                    stage_totals_seconds: &self.stage_totals_seconds,
                    stage_counts: &self.stage_counts,
                },
            )?;
        }
        if let Some(path) = profile_summary_path {
            write_pretty_json_payload(
                path,
                &ProfileSummaryPayload {
                    schema_version: 1,
                    run_id,
                    stage_totals_seconds: &self.stage_totals_seconds,
                    stage_counts: &self.stage_counts,
                },
            )?;
        }
        Ok(())
    }
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

fn write_pretty_json_payload<T>(path: &Path, payload: &T) -> Result<(), TimingFileError>
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
