//! Native stage timing aggregation and final output.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

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
                    schema_version: 1,
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
