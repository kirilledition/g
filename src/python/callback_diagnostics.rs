//! PyO3 adapters for callback diagnostics policy.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_engine::callback_diagnostics as native_callback_diagnostics;

#[pyclass]
pub(crate) struct NativeNullLogisticNonconvergencePlan {
    inner: native_callback_diagnostics::NullLogisticNonconvergencePlan,
}

#[pymethods]
impl NativeNullLogisticNonconvergencePlan {
    #[getter]
    fn action(&self) -> &'static str {
        self.inner.action.as_value()
    }

    #[getter]
    fn failed_trait_indices(&self) -> Vec<usize> {
        self.inner.failed_trait_indices.clone()
    }

    #[getter]
    fn message(&self) -> Option<&str> {
        self.inner.message.as_deref()
    }

    #[getter]
    fn warning_message(&self) -> Option<&str> {
        self.inner.warning_message.as_deref()
    }
}

impl From<native_callback_diagnostics::NullLogisticNonconvergencePlan> for NativeNullLogisticNonconvergencePlan {
    fn from(plan: native_callback_diagnostics::NullLogisticNonconvergencePlan) -> Self {
        Self { inner: plan }
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_null_logistic_nonconvergence(
    chromosome: String,
    convergence_flags: Vec<bool>,
    scalar_convergence: bool,
    phenotype_names: Option<Vec<String>>,
    policy: String,
) -> PyResult<NativeNullLogisticNonconvergencePlan> {
    native_callback_diagnostics::plan_null_logistic_nonconvergence(
        &chromosome,
        &convergence_flags,
        scalar_convergence,
        phenotype_names.as_deref(),
        &policy,
    )
    .map(Into::into)
    .map_err(|error| callback_diagnostics_error_to_py(&error))
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeNullLogisticNonconvergencePlan>()?;
    module.add_function(wrap_pyfunction!(plan_null_logistic_nonconvergence, module)?)?;
    Ok(())
}

fn callback_diagnostics_error_to_py(error: &native_callback_diagnostics::CallbackDiagnosticsError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
