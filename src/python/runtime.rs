use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use g_genotype::bgen::set_bgen_decode_tile_variant_count;
use g_runtime::{
    CliRunFailureTelemetryPlan, CliRunLifecycleState, CliTelemetryCloseFailurePlan, RayonRuntimeError,
    configure_global_rayon_thread_pool, format_global_rayon_thread_pool_configuration_error,
    plan_cli_run_failed_telemetry_emission as native_plan_cli_run_failed_telemetry_emission,
    plan_cli_telemetry_close_failure as native_plan_cli_telemetry_close_failure,
};

use super::errors;

#[pyclass]
pub(super) struct NativeCliRunLifecycleState {
    state: CliRunLifecycleState,
}

#[pyclass]
pub(super) struct NativeCliRunFailureTelemetryPlan {
    plan: CliRunFailureTelemetryPlan,
}

#[pyclass]
pub(super) struct NativeCliTelemetryCloseFailurePlan {
    plan: CliTelemetryCloseFailurePlan,
}

#[pymethods]
impl NativeCliRunFailureTelemetryPlan {
    #[getter]
    fn should_log_run_failed_to_telemetry(&self) -> bool {
        self.plan.should_log_run_failed_to_telemetry
    }
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

    fn plan_run_failed_telemetry(&self) -> NativeCliRunFailureTelemetryPlan {
        NativeCliRunFailureTelemetryPlan { plan: self.state.plan_run_failed_telemetry() }
    }
}

#[pyfunction]
fn plan_cli_telemetry_close_failure(
    current_exit_code: i32,
    runtime_failure_exit_code: i32,
) -> NativeCliTelemetryCloseFailurePlan {
    NativeCliTelemetryCloseFailurePlan {
        plan: native_plan_cli_telemetry_close_failure(current_exit_code, runtime_failure_exit_code),
    }
}

#[pyfunction]
#[allow(clippy::missing_errors_doc)]
fn emit_cli_run_failed_telemetry_event(
    telemetry_session: &Bound<'_, PyAny>,
    failed_event: &Bound<'_, PyAny>,
    should_log_run_failed_to_telemetry: bool,
) -> PyResult<()> {
    let emission_plan =
        native_plan_cli_run_failed_telemetry_emission(should_log_run_failed_to_telemetry, !telemetry_session.is_none());
    if !emission_plan.should_emit {
        return Ok(());
    }

    let emission_result = telemetry_session.call_method1("log_run_failed", (failed_event,));
    if emission_plan.should_suppress_errors {
        let _ = emission_result;
        return Ok(());
    }
    emission_result.map(|_| ())
}

#[pyfunction]
#[allow(clippy::missing_errors_doc)]
pub(super) fn configure_bgen_decode_tile_variant_count(tile_variant_count: usize) -> PyResult<()> {
    set_bgen_decode_tile_variant_count(tile_variant_count)
        .map_err(|error| errors::convert_bgen_error("configure_bgen_decode_tile_variant_count", error))
}

#[pyfunction]
#[allow(clippy::missing_errors_doc)]
pub(super) fn configure_rayon_global_thread_pool(thread_count: usize) -> PyResult<()> {
    configure_global_rayon_thread_pool(thread_count).map_err(|error| rayon_runtime_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(super) fn format_rayon_thread_pool_configuration_error_value(thread_count: i64, source_error: String) -> String {
    format_global_rayon_thread_pool_configuration_error(thread_count, &source_error)
}

pub(super) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunFailureTelemetryPlan>()?;
    module.add_class::<NativeCliRunLifecycleState>()?;
    module.add_class::<NativeCliTelemetryCloseFailurePlan>()?;
    module.add_function(wrap_pyfunction!(emit_cli_run_failed_telemetry_event, module)?)?;
    module.add_function(wrap_pyfunction!(plan_cli_telemetry_close_failure, module)?)?;
    module.add_function(wrap_pyfunction!(configure_bgen_decode_tile_variant_count, module)?)?;
    module.add_function(wrap_pyfunction!(configure_rayon_global_thread_pool, module)?)?;
    module.add_function(wrap_pyfunction!(format_rayon_thread_pool_configuration_error_value, module)?)?;
    Ok(())
}

fn rayon_runtime_error_to_py(error: &RayonRuntimeError) -> PyErr {
    match error {
        RayonRuntimeError::InvalidThreadCount => PyValueError::new_err(error.to_string()),
        RayonRuntimeError::GlobalThreadPool { .. } => PyRuntimeError::new_err(error.to_string()),
    }
}
