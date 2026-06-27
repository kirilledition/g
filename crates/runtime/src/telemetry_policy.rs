//! Deterministic telemetry path and counter policy helpers.

use std::path::{Component, Path, PathBuf};

use chrono::{DateTime, SecondsFormat};

const EVENTS_JSONL_FILE_NAME: &str = "events.jsonl";
const PROFILE_SUMMARY_JSON_FILE_NAME: &str = "profile.summary.json";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryPathsPayload {
    pub log_dir: Option<String>,
    pub stream_file: Option<String>,
    pub profile_summary_json: Option<String>,
    pub stage_timings_json: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryWriterCountersPayload {
    pub accepted_event_count: i64,
    pub written_event_count: i64,
    pub dropped_event_count: i64,
    pub cap_dropped_event_count: i64,
    pub queue_dropped_event_count: i64,
    pub event_cap_exceeded: bool,
    pub lossy: bool,
    pub event_cap: Option<i64>,
    pub finish_flush_duration_seconds: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetrySessionPolicyPayload {
    pub enabled: bool,
    pub profile_enabled: bool,
    pub event_cap: Option<i64>,
}

#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_sign_loss)]
#[must_use]
pub fn format_timestamp(timestamp_seconds: f64) -> String {
    let whole_seconds = timestamp_seconds.floor() as i64;
    let nanoseconds = (((timestamp_seconds - whole_seconds as f64) * 1_000_000_000.0) as u32).min(999_999_999);
    DateTime::from_timestamp(whole_seconds, nanoseconds).map_or_else(
        || "1970-01-01T00:00:00.000000Z".to_string(),
        |timestamp| timestamp.to_rfc3339_opts(SecondsFormat::Micros, true),
    )
}

#[must_use]
pub fn resolve_output_run_root(output_path: &Path, output_run_directory: Option<&Path>) -> PathBuf {
    if let Some(run_directory) = output_run_directory {
        return run_directory.to_path_buf();
    }
    output_path.with_file_name(format!(
        "{}.g",
        output_path.file_name().map_or_else(|| output_path.display().to_string(), |name| name.to_string_lossy().into())
    ))
}

/// Resolve all telemetry file-system paths for one run.
///
/// # Errors
///
/// Returns an error when telemetry stream path options conflict.
pub fn resolve_telemetry_paths(
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

/// Resolve the unified telemetry event stream path.
///
/// # Errors
///
/// Returns an error when `log_file` and `trace_file` point to different files.
pub fn resolve_telemetry_stream_file(
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

#[must_use]
pub fn paths_refer_to_same_file(first_path: &Path, second_path: &Path) -> bool {
    normalize_path_for_comparison(first_path) == normalize_path_for_comparison(second_path)
}

#[must_use]
pub fn build_empty_writer_counters() -> TelemetryWriterCountersPayload {
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

#[must_use]
pub fn resolve_telemetry_session_policy(telemetry_mode: &str, trace_event_cap: i64) -> TelemetrySessionPolicyPayload {
    TelemetrySessionPolicyPayload {
        enabled: telemetry_mode != "off",
        profile_enabled: matches!(telemetry_mode, "profile" | "trace"),
        event_cap: if telemetry_mode == "trace" && trace_event_cap > 0 { Some(trace_event_cap) } else { None },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_profile_paths_under_default_log_directory() {
        let paths = resolve_telemetry_paths(
            Path::new("results/output"),
            None,
            "profile",
            None,
            None,
            None,
            None,
            Some(Path::new("stage-timings.json")),
        )
        .unwrap();

        assert_eq!(paths.log_dir, Some("results/output.g/logs".to_string()));
        assert_eq!(paths.stream_file, Some("results/output.g/logs/events.jsonl".to_string()));
        assert_eq!(paths.profile_summary_json, Some("results/output.g/logs/profile.summary.json".to_string()));
        assert_eq!(paths.stage_timings_json, Some("stage-timings.json".to_string()));
    }

    #[test]
    fn rejects_conflicting_telemetry_stream_files() {
        let stream_file = resolve_telemetry_stream_file(
            "trace",
            Some(Path::new("logs")),
            Some(Path::new("events.jsonl")),
            Some(Path::new("trace.jsonl")),
        );

        assert_eq!(
            stream_file,
            Err("log_file and trace_file both configure the unified telemetry stream; use one path.".to_string()),
        );
    }

    #[test]
    fn resolves_telemetry_session_policy() {
        assert_eq!(
            resolve_telemetry_session_policy("off", 10),
            TelemetrySessionPolicyPayload { enabled: false, profile_enabled: false, event_cap: None },
        );
        assert_eq!(
            resolve_telemetry_session_policy("progress", 10),
            TelemetrySessionPolicyPayload { enabled: true, profile_enabled: false, event_cap: None },
        );
        assert_eq!(
            resolve_telemetry_session_policy("profile", 10),
            TelemetrySessionPolicyPayload { enabled: true, profile_enabled: true, event_cap: None },
        );
        assert_eq!(
            resolve_telemetry_session_policy("trace", 10),
            TelemetrySessionPolicyPayload { enabled: true, profile_enabled: true, event_cap: Some(10) },
        );
        assert_eq!(
            resolve_telemetry_session_policy("trace", 0),
            TelemetrySessionPolicyPayload { enabled: true, profile_enabled: true, event_cap: None },
        );
    }
}
