//! Deterministic telemetry path and counter policy helpers.

use std::path::{Component, Path, PathBuf};

use chrono::{DateTime, SecondsFormat};

const EVENTS_JSONL_FILE_NAME: &str = "events.jsonl";
const PROFILE_SUMMARY_JSON_FILE_NAME: &str = "profile.summary.json";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TelemetryPathsPayload {
    pub(crate) log_dir: Option<String>,
    pub(crate) stream_file: Option<String>,
    pub(crate) profile_summary_json: Option<String>,
    pub(crate) stage_timings_json: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TelemetryWriterCountersPayload {
    pub(crate) accepted_event_count: i64,
    pub(crate) written_event_count: i64,
    pub(crate) dropped_event_count: i64,
    pub(crate) cap_dropped_event_count: i64,
    pub(crate) queue_dropped_event_count: i64,
    pub(crate) event_cap_exceeded: bool,
    pub(crate) lossy: bool,
    pub(crate) event_cap: Option<i64>,
    pub(crate) finish_flush_duration_seconds: Option<f64>,
}

#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_sign_loss)]
pub(crate) fn format_timestamp(timestamp_seconds: f64) -> String {
    let whole_seconds = timestamp_seconds.floor() as i64;
    let nanoseconds = (((timestamp_seconds - whole_seconds as f64) * 1_000_000_000.0) as u32).min(999_999_999);
    DateTime::from_timestamp(whole_seconds, nanoseconds).map_or_else(
        || "1970-01-01T00:00:00.000000Z".to_string(),
        |timestamp| timestamp.to_rfc3339_opts(SecondsFormat::Micros, true),
    )
}

pub(crate) fn resolve_output_run_root(output_path: &Path, output_run_directory: Option<&Path>) -> PathBuf {
    if let Some(run_directory) = output_run_directory {
        return run_directory.to_path_buf();
    }
    output_path.with_file_name(format!(
        "{}.g",
        output_path.file_name().map_or_else(|| output_path.display().to_string(), |name| name.to_string_lossy().into())
    ))
}

pub(crate) fn resolve_telemetry_paths(
    output_path: &Path,
    output_run_directory: Option<&Path>,
    telemetry_mode: &str,
    log_dir: Option<&Path>,
    log_file: Option<&Path>,
    trace_file: Option<&Path>,
    profile_summary_json: Option<&Path>,
    stage_timings_json: Option<&Path>,
) -> Result<TelemetryPathsPayload, String> {
    let resolved_log_dir = match (log_dir, telemetry_mode) {
        (Some(path), _) => Some(path.to_path_buf()),
        (None, "off") => None,
        (None, _) => Some(resolve_output_run_root(output_path, output_run_directory).join("logs")),
    };
    let stream_file = resolve_telemetry_stream_file(telemetry_mode, resolved_log_dir.as_deref(), log_file, trace_file)?;
    let resolved_profile_summary_json = match (profile_summary_json, resolved_log_dir.as_deref(), telemetry_mode) {
        (Some(path), _, _) => Some(path.to_path_buf()),
        (None, Some(directory), "profile" | "trace") => Some(directory.join(PROFILE_SUMMARY_JSON_FILE_NAME)),
        _ => None,
    };
    Ok(TelemetryPathsPayload {
        log_dir: optional_path_string(resolved_log_dir.as_deref()),
        stream_file: optional_path_string(stream_file.as_deref()),
        profile_summary_json: optional_path_string(resolved_profile_summary_json.as_deref()),
        stage_timings_json: optional_path_string(stage_timings_json),
    })
}

pub(crate) fn resolve_telemetry_stream_file(
    telemetry_mode: &str,
    log_dir: Option<&Path>,
    log_file: Option<&Path>,
    trace_file: Option<&Path>,
) -> Result<Option<PathBuf>, String> {
    if telemetry_mode == "off" {
        return Ok(None);
    }
    if let (Some(log_file_path), Some(trace_file_path)) = (log_file, trace_file)
        && !paths_refer_to_same_file(log_file_path, trace_file_path)
    {
        return Err("log_file and trace_file both configure the unified telemetry stream; use one path.".to_string());
    }
    if let Some(path) = log_file {
        return Ok(Some(path.to_path_buf()));
    }
    if let Some(path) = trace_file {
        return Ok(Some(path.to_path_buf()));
    }
    Ok(log_dir.map(|directory| directory.join(EVENTS_JSONL_FILE_NAME)))
}

pub(crate) fn paths_refer_to_same_file(first_path: &Path, second_path: &Path) -> bool {
    normalize_path_for_comparison(first_path) == normalize_path_for_comparison(second_path)
}

pub(crate) fn build_empty_writer_counters() -> TelemetryWriterCountersPayload {
    TelemetryWriterCountersPayload {
        accepted_event_count: 0,
        written_event_count: 0,
        dropped_event_count: 0,
        cap_dropped_event_count: 0,
        queue_dropped_event_count: 0,
        event_cap_exceeded: false,
        lossy: true,
        event_cap: None,
        finish_flush_duration_seconds: None,
    }
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
