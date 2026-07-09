//! PyO3 adapters for callback diagnostics policy.

use numpy::ndarray::IxDyn;
use numpy::{PyArray, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods, dtype};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_engine as native_callback_diagnostics;
use g_runtime as native_run_events;

use crate::binding::errors::convert_callback_diagnostics_error;
use crate::binding::telemetry::logging;

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn enforce_null_logistic_nonconvergence_from_array(
    py: Python<'_>,
    chromosome: String,
    convergence_values: &Bound<'_, PyUntypedArray>,
    phenotype_names: Option<Vec<String>>,
    policy: String,
) -> PyResult<usize> {
    let (convergence_flags, scalar_convergence) = parse_convergence_flags(py, convergence_values)?;
    let native_plan = native_callback_diagnostics::plan_null_logistic_nonconvergence(
        &chromosome,
        &convergence_flags,
        scalar_convergence,
        phenotype_names.as_deref(),
        &policy,
    )
    .map_err(|error| convert_callback_diagnostics_error(&error))?;
    match native_plan.action {
        native_callback_diagnostics::NullLogisticNonconvergenceAction::Continue => {}
        native_callback_diagnostics::NullLogisticNonconvergenceAction::Fail => {
            let message = native_plan.message.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("Native null-logistic nonconvergence fail plan did not include a message.")
            })?;
            return Err(PyRuntimeError::new_err(message.clone()));
        }
        native_callback_diagnostics::NullLogisticNonconvergenceAction::Warn => {
            let warning_message = native_plan.warning_message.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err(
                    "Native null-logistic nonconvergence warning plan did not include a warning message.",
                )
            })?;
            emit_null_logistic_nonconvergence_warning(
                warning_message,
                &chromosome,
                &native_plan,
                phenotype_names.as_ref().map_or(0, Vec::len),
                &policy,
            )?;
        }
    }
    Ok(native_plan.nonconverged_count)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(enforce_null_logistic_nonconvergence_from_array, module)?)?;
    Ok(())
}

fn parse_convergence_flags(
    py: Python<'_>,
    convergence_values: &Bound<'_, PyUntypedArray>,
) -> PyResult<(Vec<bool>, bool)> {
    let element_type = convergence_values.dtype();
    if !element_type.is_equiv_to(&dtype::<bool>(py)) {
        return Err(PyValueError::new_err("Null logistic convergence values must have bool dtype."));
    }
    let typed_values = convergence_values.cast::<PyArray<bool, IxDyn>>()?;
    let readonly_values = typed_values.readonly();
    let convergence_array = readonly_values.as_array();
    let scalar_convergence = convergence_array.shape().is_empty();
    Ok((convergence_array.iter().copied().collect::<Vec<_>>(), scalar_convergence))
}

fn emit_null_logistic_nonconvergence_warning(
    warning_message: &str,
    chromosome: &str,
    native_plan: &native_callback_diagnostics::NullLogisticNonconvergencePlan,
    phenotype_count: usize,
    policy: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_callback_null_logistic_nonconvergence_warning_diagnostic_payload(
        warning_message,
        chromosome,
        usize_to_i64(native_plan.nonconverged_count, "nonconverged_count")?,
        usize_to_i64(phenotype_count, "phenotype_count")?,
        policy,
        native_plan.scalar_convergence,
        usize_to_i64(native_plan.total_fit_count, "total_fit_count")?,
    );
    let fields_json = native_run_events::serialize_run_diagnostic_fields_json(&payload.fields)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    logging::emit_diagnostic_event(payload.level, payload.event_name, &payload.message, Some(fields_json))
}

fn usize_to_i64(value: usize, value_name: &str) -> PyResult<i64> {
    i64::try_from(value).map_err(|_| PyValueError::new_err(format!("{value_name} exceeds i64 range.")))
}
