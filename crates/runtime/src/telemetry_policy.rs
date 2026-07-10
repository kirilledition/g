//! Deterministic telemetry path and counter policy helpers.

use std::fmt;
use std::path::{Component, Path, PathBuf};

use chrono::{DateTime, SecondsFormat};

const EVENTS_JSONL_FILE_NAME: &str = "events.jsonl";
const PROFILE_SUMMARY_JSON_FILE_NAME: &str = "profile.summary.json";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryPathError {
    message: String,
}

impl TelemetryPathError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for TelemetryPathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for TelemetryPathError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TelemetryPathsPayload {
    pub(crate) stream_file: Option<String>,
    pub(crate) profile_summary_json: Option<String>,
    pub(crate) stage_timings_json: Option<String>,
}

#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_sign_loss)]
#[must_use]
pub(crate) fn format_timestamp(timestamp_seconds: f64) -> String {
    let whole_seconds = timestamp_seconds.floor() as i64;
    let nanoseconds = (((timestamp_seconds - whole_seconds as f64) * 1_000_000_000.0) as u32).min(999_999_999);
    DateTime::from_timestamp(whole_seconds, nanoseconds).map_or_else(
        || "1970-01-01T00:00:00.000000Z".to_string(),
        |timestamp| timestamp.to_rfc3339_opts(SecondsFormat::Micros, true),
    )
}

/// Resolve all telemetry file-system paths for one run.
///
/// # Errors
///
/// Returns an error when telemetry stream path options conflict.
pub(crate) fn resolve_telemetry_paths(run_plan: &g_plan::RunPlan) -> Result<TelemetryPathsPayload, TelemetryPathError> {
    let diagnostics = &run_plan.diagnostics;
    let telemetry_mode = diagnostics.telemetry;
    let resolved_log_dir = match (diagnostics.log_directory.as_deref().map(Path::new), telemetry_mode) {
        (Some(path), _) => Some(path.to_path_buf()),
        (None, g_plan::TelemetryMode::Off) => None,
        (None, _) => Some(Path::new(&run_plan.output.output_run_root).join("logs")),
    };
    let stream_file = resolve_telemetry_stream_file(
        telemetry_mode,
        resolved_log_dir.as_deref(),
        diagnostics.log_file.as_deref().map(Path::new),
        diagnostics.trace_file.as_deref().map(Path::new),
    )?;
    let resolved_profile_summary_json =
        match (diagnostics.profile_summary_path.as_deref().map(Path::new), resolved_log_dir.as_deref(), telemetry_mode)
        {
            (Some(path), _, _) => Some(path.to_path_buf()),
            (None, Some(directory), g_plan::TelemetryMode::Profile) => {
                Some(directory.join(PROFILE_SUMMARY_JSON_FILE_NAME))
            }
            _ => None,
        };
    Ok(TelemetryPathsPayload {
        stream_file: optional_path_string(stream_file.as_deref()),
        profile_summary_json: optional_path_string(resolved_profile_summary_json.as_deref()),
        stage_timings_json: diagnostics.stage_timings_path.clone(),
    })
}

/// Resolve the unified telemetry event stream path.
///
/// # Errors
///
/// Returns an error when `log_file` and `trace_file` point to different files.
fn resolve_telemetry_stream_file(
    telemetry_mode: g_plan::TelemetryMode,
    log_dir: Option<&Path>,
    log_file: Option<&Path>,
    trace_file: Option<&Path>,
) -> Result<Option<PathBuf>, TelemetryPathError> {
    if telemetry_mode == g_plan::TelemetryMode::Off {
        return Ok(None);
    }
    if let (Some(log_file_path), Some(trace_file_path)) = (log_file, trace_file)
        && !paths_refer_to_same_file(log_file_path, trace_file_path)
    {
        return Err(TelemetryPathError::new(
            "log_file and trace_file both configure the unified telemetry stream; use one path.",
        ));
    }
    if let Some(path) = log_file {
        return Ok(Some(path.to_path_buf()));
    }
    if let Some(path) = trace_file {
        return Ok(Some(path.to_path_buf()));
    }
    Ok(log_dir.map(|directory| directory.join(EVENTS_JSONL_FILE_NAME)))
}

#[must_use]
fn paths_refer_to_same_file(first_path: &Path, second_path: &Path) -> bool {
    normalize_path_for_comparison(first_path) == normalize_path_for_comparison(second_path)
}

fn optional_path_string(path: Option<&Path>) -> Option<String> {
    path.map(|value| value.display().to_string())
}

fn normalize_path_for_comparison(path: &Path) -> PathBuf {
    if let Ok(canonical_path) = std::fs::canonicalize(path) {
        return canonical_path;
    }
    let absolute_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(path)
    };
    let mut normalized_path = PathBuf::new();
    for component in absolute_path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized_path.pop();
            }
            Component::RootDir | Component::Prefix(_) => normalized_path.push(component.as_os_str()),
            Component::Normal(value) => {
                let candidate_path = normalized_path.join(value);
                normalized_path = std::fs::canonicalize(&candidate_path).unwrap_or(candidate_path);
            }
        }
    }
    normalized_path
}
