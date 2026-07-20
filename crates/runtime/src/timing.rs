//! Native stage timing aggregation and final output.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

const PROFILE_SUMMARY_SCHEMA_VERSION: i64 = 0;

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

#[derive(Debug, Default, PartialEq)]
pub struct StageTimingRecorder {
    stages: BTreeMap<String, StageTimingAggregate>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct StageTimingAggregate {
    total_seconds: f64,
    count: u64,
}

#[derive(Clone, Copy)]
enum StageTimingMetric {
    TotalSeconds,
    Count,
}

#[derive(Clone, Copy)]
struct StageTimingMap<'recorder> {
    stages: &'recorder BTreeMap<String, StageTimingAggregate>,
    metric: StageTimingMetric,
}

#[derive(Serialize)]
struct ProfileSummaryPayload<'recorder> {
    schema_version: i64,
    run_id: Option<&'recorder str>,
    stage_totals_seconds: StageTimingMap<'recorder>,
    stage_counts: StageTimingMap<'recorder>,
}

#[derive(Serialize)]
struct StageTimingSnapshotPayload<'recorder> {
    stage_totals_seconds: StageTimingMap<'recorder>,
    stage_counts: StageTimingMap<'recorder>,
}

impl Serialize for StageTimingMap<'_> {
    fn serialize<Output>(&self, serializer: Output) -> Result<Output::Ok, Output::Error>
    where
        Output: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.stages.len()))?;
        for (stage_name, aggregate) in self.stages {
            match self.metric {
                StageTimingMetric::TotalSeconds => map.serialize_entry(stage_name, &aggregate.total_seconds)?,
                StageTimingMetric::Count => map.serialize_entry(stage_name, &aggregate.count)?,
            }
        }
        map.end()
    }
}

impl StageTimingRecorder {
    pub fn add_stage_duration(&mut self, stage_name: &str, duration_seconds: f64) {
        if let Some(aggregate) = self.stages.get_mut(stage_name) {
            aggregate.total_seconds += duration_seconds;
            aggregate.count = aggregate.count.saturating_add(1);
            return;
        }
        self.stages.insert(stage_name.to_owned(), StageTimingAggregate { total_seconds: duration_seconds, count: 1 });
    }

    /// Write every configured final timing output.
    ///
    /// # Errors
    ///
    /// Returns an error when a timing payload cannot be written.
    pub(crate) fn write_final_timing_outputs(
        &self,
        stage_timing_path: Option<&Path>,
        profile_summary_path: Option<&Path>,
        run_id: Option<&str>,
    ) -> Result<(), TimingFileError> {
        if let Some(path) = stage_timing_path {
            write_pretty_json_payload(
                path,
                &StageTimingSnapshotPayload {
                    stage_totals_seconds: StageTimingMap {
                        stages: &self.stages,
                        metric: StageTimingMetric::TotalSeconds,
                    },
                    stage_counts: StageTimingMap { stages: &self.stages, metric: StageTimingMetric::Count },
                },
            )?;
        }
        if let Some(path) = profile_summary_path {
            write_pretty_json_payload(
                path,
                &ProfileSummaryPayload {
                    schema_version: PROFILE_SUMMARY_SCHEMA_VERSION,
                    run_id,
                    stage_totals_seconds: StageTimingMap {
                        stages: &self.stages,
                        metric: StageTimingMetric::TotalSeconds,
                    },
                    stage_counts: StageTimingMap { stages: &self.stages, metric: StageTimingMetric::Count },
                },
            )?;
        }
        Ok(())
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
    let mut payload_text =
        serde_json::to_string_pretty(payload).map_err(|source| TimingFileError::Serialize { source })?;
    payload_text.push('\n');
    std::fs::write(path, payload_text).map_err(|source| TimingFileError::WriteFile { path: path.to_path_buf(), source })
}

#[cfg(test)]
mod tests {
    use serde::ser::Error as _;

    use super::*;
    use crate::test_support::TemporaryDirectory;

    const TOLERANCE: f64 = 1.0e-12;

    struct SerializationFailure;

    impl Serialize for SerializationFailure {
        fn serialize<SerializerType>(
            &self,
            _serializer: SerializerType,
        ) -> Result<SerializerType::Ok, SerializerType::Error>
        where
            SerializerType: Serializer,
        {
            Err(SerializerType::Error::custom("intentional serialization failure"))
        }
    }

    #[test]
    fn recorder_aggregates_durations_counts_and_saturates() {
        let mut recorder = StageTimingRecorder::default();
        recorder.add_stage_duration("compute", 0.25);
        recorder.add_stage_duration("compute", 0.5);
        recorder.add_stage_duration("output", 1.25);

        let compute = recorder.stages.get("compute").expect("compute aggregate should exist");
        assert!((compute.total_seconds - 0.75).abs() < TOLERANCE);
        assert_eq!(compute.count, 2);
        let output = recorder.stages.get("output").expect("output aggregate should exist");
        assert!((output.total_seconds - 1.25).abs() < TOLERANCE);
        assert_eq!(output.count, 1);

        recorder.stages.get_mut("compute").expect("compute aggregate should exist").count = u64::MAX;
        recorder.add_stage_duration("compute", 0.25);
        let saturated = recorder.stages.get("compute").expect("compute aggregate should exist");
        assert_eq!(saturated.count, u64::MAX);
        assert!((saturated.total_seconds - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn final_outputs_preserve_sorted_metrics_and_prerelease_version() {
        let temporary_directory = TemporaryDirectory::new("timing-output");
        let stage_path = temporary_directory.path().join("nested/stages.json");
        let profile_path = temporary_directory.path().join("nested/profile.json");
        let mut recorder = StageTimingRecorder::default();
        recorder.add_stage_duration("output", 1.0);
        recorder.add_stage_duration("compute", 0.125);
        recorder.add_stage_duration("compute", 0.375);

        recorder
            .write_final_timing_outputs(Some(&stage_path), Some(&profile_path), Some("run-123"))
            .expect("timing outputs should be written");

        let stage_text = std::fs::read_to_string(&stage_path).expect("stage timing output should be readable");
        let profile_text = std::fs::read_to_string(&profile_path).expect("profile summary should be readable");
        assert!(stage_text.ends_with('\n'));
        assert!(profile_text.ends_with('\n'));
        assert!(
            stage_text.find("compute").expect("compute key should exist")
                < stage_text.find("output").expect("output key should exist")
        );

        let stage_payload: serde_json::Value =
            serde_json::from_str(&stage_text).expect("stage timing output should parse");
        let profile_payload: serde_json::Value =
            serde_json::from_str(&profile_text).expect("profile summary should parse");
        assert!(stage_payload.get("schema_version").is_none());
        assert_eq!(profile_payload["schema_version"], PROFILE_SUMMARY_SCHEMA_VERSION);
        assert_eq!(profile_payload["run_id"], "run-123");
        assert_eq!(stage_payload["stage_counts"]["compute"], 2);
        assert_eq!(stage_payload["stage_counts"]["output"], 1);
        let compute_total =
            stage_payload["stage_totals_seconds"]["compute"].as_f64().expect("compute total should be numeric");
        assert!((compute_total - 0.5).abs() < TOLERANCE);
        assert_eq!(stage_payload["stage_totals_seconds"], profile_payload["stage_totals_seconds"]);
        assert_eq!(stage_payload["stage_counts"], profile_payload["stage_counts"]);
    }

    #[test]
    fn final_output_handles_absent_paths_and_null_run_identifier() {
        let temporary_directory = TemporaryDirectory::new("timing-optional");
        let profile_path = temporary_directory.path().join("profile.json");
        let recorder = StageTimingRecorder::default();

        recorder.write_final_timing_outputs(None, None, None).expect("absent outputs should be a no-op");
        recorder
            .write_final_timing_outputs(None, Some(&profile_path), None)
            .expect("profile-only output should be written");

        let profile_payload: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&profile_path).expect("profile-only output should be readable"))
                .expect("profile-only output should parse");
        assert_eq!(profile_payload["schema_version"], 0);
        assert!(profile_payload["run_id"].is_null());
        assert_eq!(profile_payload["stage_counts"], serde_json::json!({}));
    }

    #[test]
    fn timing_file_errors_identify_create_serialize_and_write_failures() {
        let temporary_directory = TemporaryDirectory::new("timing-errors");
        let blocking_file = temporary_directory.path().join("blocking-file");
        std::fs::write(&blocking_file, b"not a directory").expect("blocking fixture should be written");

        let recorder = StageTimingRecorder::default();
        let create_error = recorder
            .write_final_timing_outputs(Some(&blocking_file.join("stages.json")), None, None)
            .expect_err("file parent should reject directory creation");
        assert!(matches!(&create_error, TimingFileError::CreateParentDirectory { .. }));
        assert!(create_error.source().is_some());
        assert!(create_error.to_string().contains("failed to create timing file parent directory"));

        let serialize_error =
            write_pretty_json_payload(&temporary_directory.path().join("unwritten.json"), &SerializationFailure)
                .expect_err("failing serializer should be reported");
        assert!(matches!(&serialize_error, TimingFileError::Serialize { .. }));
        assert!(serialize_error.source().is_some());
        assert!(serialize_error.to_string().contains("failed to serialize timing payload"));

        let write_error = write_pretty_json_payload(temporary_directory.path(), &serde_json::json!({"ok": true}))
            .expect_err("directory path should reject file write");
        assert!(matches!(&write_error, TimingFileError::WriteFile { .. }));
        assert!(write_error.source().is_some());
        assert!(write_error.to_string().contains("failed to write timing file"));
    }
}
