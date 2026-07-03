use pyo3::exceptions::{PyAttributeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use g_runtime::{
    CliRunLifecycleState, CliTelemetryCloseFailurePlan,
    plan_cli_run_failed_telemetry_emission as native_plan_cli_run_failed_telemetry_emission,
    plan_cli_telemetry_close_failure as native_plan_cli_telemetry_close_failure,
};

#[pyclass]
pub(super) struct NativeCliRunLifecycleState {
    state: CliRunLifecycleState,
}

#[pyclass]
pub(super) struct NativeCliTelemetryCloseFailurePlan {
    plan: CliTelemetryCloseFailurePlan,
}

#[pymethods]
impl NativeCliTelemetryCloseFailurePlan {
    #[getter]
    fn should_report_failure(&self) -> bool {
        self.plan.should_report_failure
    }

    #[getter]
    fn exit_code(&self) -> i32 {
        self.plan.exit_code
    }
}

#[pymethods]
impl NativeCliRunLifecycleState {
    #[new]
    fn new() -> Self {
        Self { state: CliRunLifecycleState::default() }
    }

    #[getter]
    fn runner_started(&self) -> bool {
        self.state.runner_started()
    }

    fn mark_runner_started(&mut self) {
        self.state.mark_runner_started();
    }

    #[allow(clippy::unused_self)]
    fn plan_telemetry_close_failure(
        &self,
        current_exit_code: i32,
        runtime_failure_exit_code: i32,
    ) -> NativeCliTelemetryCloseFailurePlan {
        NativeCliTelemetryCloseFailurePlan {
            plan: native_plan_cli_telemetry_close_failure(current_exit_code, runtime_failure_exit_code),
        }
    }

    #[allow(clippy::missing_errors_doc)]
    fn emit_run_failed_telemetry_event(
        &self,
        telemetry_session: &Bound<'_, PyAny>,
        failed_event: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let telemetry_plan = self.state.plan_run_failed_telemetry();
        if !telemetry_plan.should_log_run_failed_to_telemetry || telemetry_session.is_none() {
            return Ok(());
        }
        let native_telemetry_session =
            match optional_native_telemetry_session(telemetry_session.py(), telemetry_session) {
                Ok(native_telemetry_session) => native_telemetry_session,
                Err(error) => {
                    let emission_plan = native_plan_cli_run_failed_telemetry_emission(
                        telemetry_plan.should_log_run_failed_to_telemetry,
                        true,
                    );
                    if emission_plan.should_suppress_errors {
                        return Ok(());
                    }
                    return Err(error);
                }
            };
        let emission_plan = native_plan_cli_run_failed_telemetry_emission(
            telemetry_plan.should_log_run_failed_to_telemetry,
            native_telemetry_session.is_some(),
        );
        if !emission_plan.should_emit {
            return Ok(());
        }

        let Some(active_native_telemetry_session) = native_telemetry_session else {
            return Ok(());
        };
        let emission_result = active_native_telemetry_session.call_method1("emit_run_failed_event", (failed_event,));
        if emission_plan.should_suppress_errors {
            let _ = emission_result;
            return Ok(());
        }
        emission_result.map(|_| ())
    }
}

fn optional_native_telemetry_session<'py>(
    py: Python<'py>,
    telemetry_session: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match telemetry_session.getattr("native_telemetry_session") {
        Ok(native_telemetry_session) if native_telemetry_session.is_none() => Ok(None),
        Ok(native_telemetry_session) => Ok(Some(native_telemetry_session)),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Err(PyTypeError::new_err(
            "CLI run-failed telemetry requires a TelemetrySession with a native telemetry session handle.",
        )),
        Err(error) => Err(error),
    }
}

pub(super) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunLifecycleState>()?;
    module.add_class::<NativeCliTelemetryCloseFailurePlan>()?;
    Ok(())
}
