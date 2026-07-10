//! Stage timing state used by the native run engine.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use g_runtime as native_timing;

pub(crate) struct NativeStageTimingRecorder {
    recorder: Mutex<native_timing::StageTimingRecorder>,
}

impl NativeStageTimingRecorder {
    pub(crate) fn from_config(stage_timing_path_configured: bool, force: bool) -> Option<Self> {
        native_timing::StageTimingRecorder::from_config(stage_timing_path_configured, force)
            .map(|recorder| Self { recorder: Mutex::new(recorder) })
    }

    pub(crate) fn should_collect_exact_stage_timings(&self) -> PyResult<bool> {
        Ok(self.lock_recorder()?.should_collect_exact_stage_timings())
    }

    pub(crate) fn record_stage_duration(&self, stage_name: &str, duration_seconds: f64) -> PyResult<()> {
        self.lock_recorder()?.add_stage_duration(stage_name.to_string(), duration_seconds);
        Ok(())
    }

    pub(crate) fn write_final_timing_outputs(
        &self,
        stage_timing_path: Option<&str>,
        profile_summary_path: Option<&str>,
        run_id: Option<String>,
    ) -> PyResult<()> {
        self.lock_recorder()?
            .write_final_timing_outputs(stage_timing_path.map(Path::new), profile_summary_path.map(Path::new), run_id)
            .map(|_| ())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn lock_recorder(&self) -> PyResult<MutexGuard<'_, native_timing::StageTimingRecorder>> {
        self.recorder.lock().map_err(|_| PyRuntimeError::new_err("Stage timing recorder lock was poisoned."))
    }
}

pub(crate) fn record_final_timing_outputs_write_started(
    stage_timing_path: Option<&str>,
    profile_summary_path: Option<&str>,
    run_id: Option<&str>,
) -> PyResult<()> {
    let payload = native_timing::build_final_timing_outputs_write_started_diagnostic_payload(
        stage_timing_path,
        profile_summary_path,
        run_id,
    );
    let fields_json = native_timing::serialize_final_timing_outputs_write_started_diagnostic_fields_json(&payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    super::logging::emit_diagnostic_event(payload.level, payload.event_name, payload.message, Some(fields_json))
}
