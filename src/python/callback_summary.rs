//! PyO3 adapters for callback summary counters.

use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use g_engine::callback_summary as native_callback_summary;

#[pyclass]
pub(crate) struct NativeBinaryCorrectionSummary {
    state: Mutex<native_callback_summary::BinaryCorrectionSummaryState>,
}

#[pyclass]
pub(crate) struct NativeBinaryCorrectionDiagnosticsRecordPlan {
    inner: native_callback_summary::BinaryCorrectionDiagnosticsRecordPlan,
}

#[pyclass]
pub(crate) struct NativeBinaryCorrectionSummaryEmitPlan {
    inner: native_callback_summary::BinaryCorrectionSummaryEmitPlan,
}

#[pymethods]
impl NativeBinaryCorrectionSummary {
    #[new]
    fn new() -> Self {
        Self::new_summary()
    }

    #[getter]
    fn chunk_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.chunk_count)
    }

    #[getter]
    fn score_only_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.score_only_count)
    }

    #[getter]
    fn score_test_candidate_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.score_test_candidate_count)
    }

    #[getter]
    fn firth_attempted_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.firth_attempted_count)
    }

    #[getter]
    fn firth_success_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.firth_success_count)
    }

    #[getter]
    fn firth_failed_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.firth_failed_count)
    }

    #[getter]
    fn firth_numerical_failure_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.firth_numerical_failure_count)
    }

    #[getter]
    fn firth_max_iteration_failure_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.firth_max_iteration_failure_count)
    }

    #[getter]
    fn firth_invalid_statistic_failure_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.firth_invalid_statistic_failure_count)
    }

    #[getter]
    fn firth_step_halving_failure_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.firth_step_halving_failure_count)
    }

    #[getter]
    fn pseudo_firth_attempt_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.pseudo_firth_attempt_count)
    }

    #[getter]
    fn pseudo_firth_success_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.pseudo_firth_success_count)
    }

    #[getter]
    fn nr_zero_start_attempt_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.nr_zero_start_attempt_count)
    }

    #[getter]
    fn nr_zero_start_success_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.nr_zero_start_success_count)
    }

    #[getter]
    fn nr_warm_start_attempt_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.nr_warm_start_attempt_count)
    }

    #[getter]
    fn nr_warm_start_success_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.nr_warm_start_success_count)
    }

    #[getter]
    fn sparse_correction_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.sparse_correction_count)
    }

    #[getter]
    fn dense_correction_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.dense_correction_count)
    }

    #[getter]
    fn null_model_failure_count(&self) -> PyResult<i64> {
        Ok(self.lock_state()?.null_model_failure_count)
    }

    fn add_null_model_failure_count(&self, failure_count: i64) -> PyResult<()> {
        self.add_null_model_failure_count_value(failure_count)
    }

    fn add_diagnostics_mapping(&self, diagnostics: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed_diagnostics = parse_binary_chunk_diagnostics(diagnostics)?;
        self.lock_state()?.add_diagnostics(&parsed_diagnostics);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn add_diagnostics_counts(
        &self,
        score_only_count: i64,
        score_test_candidate_count: i64,
        firth_candidate_count: i64,
        firth_converged_count: i64,
        firth_failed_count: i64,
        firth_numerical_failure_count: i64,
        firth_max_iteration_failure_count: i64,
        firth_invalid_statistic_failure_count: i64,
        firth_step_halving_failure_count: i64,
        pseudo_firth_attempt_count: i64,
        pseudo_firth_success_count: i64,
        nr_zero_start_attempt_count: i64,
        nr_zero_start_success_count: i64,
        nr_warm_start_attempt_count: i64,
        nr_warm_start_success_count: i64,
        sparse_correction_count: i64,
        dense_correction_count: i64,
    ) -> PyResult<()> {
        self.lock_state()?.add_diagnostics(&native_callback_summary::BinaryChunkDiagnosticsInput {
            score_only_count,
            score_test_candidate_count,
            firth_candidate_count,
            firth_converged_count,
            firth_failed_count,
            firth_numerical_failure_count,
            firth_max_iteration_failure_count,
            firth_invalid_statistic_failure_count,
            firth_step_halving_failure_count,
            pseudo_firth_attempt_count,
            pseudo_firth_success_count,
            nr_zero_start_attempt_count,
            nr_zero_start_success_count,
            nr_warm_start_attempt_count,
            nr_warm_start_success_count,
            sparse_correction_count,
            dense_correction_count,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn add_diagnostics_totals(
        &self,
        chunk_count: i64,
        score_only_count: i64,
        score_test_candidate_count: i64,
        firth_candidate_count: i64,
        firth_converged_count: i64,
        firth_failed_count: i64,
        firth_numerical_failure_count: i64,
        firth_max_iteration_failure_count: i64,
        firth_invalid_statistic_failure_count: i64,
        firth_step_halving_failure_count: i64,
        pseudo_firth_attempt_count: i64,
        pseudo_firth_success_count: i64,
        nr_zero_start_attempt_count: i64,
        nr_zero_start_success_count: i64,
        nr_warm_start_attempt_count: i64,
        nr_warm_start_success_count: i64,
        sparse_correction_count: i64,
        dense_correction_count: i64,
    ) -> PyResult<()> {
        self.add_diagnostics_totals_value(
            chunk_count,
            score_only_count,
            score_test_candidate_count,
            firth_candidate_count,
            firth_converged_count,
            firth_failed_count,
            firth_numerical_failure_count,
            firth_max_iteration_failure_count,
            firth_invalid_statistic_failure_count,
            firth_step_halving_failure_count,
            pseudo_firth_attempt_count,
            pseudo_firth_success_count,
            nr_zero_start_attempt_count,
            nr_zero_start_success_count,
            nr_warm_start_attempt_count,
            nr_warm_start_success_count,
            sparse_correction_count,
            dense_correction_count,
        )
    }

    fn should_emit(&self) -> PyResult<bool> {
        Ok(self.lock_state()?.should_emit())
    }

    fn chunk_count_with_pending(&self, pending_diagnostics_count: i64) -> PyResult<i64> {
        self.chunk_count_with_pending_value(pending_diagnostics_count)
    }

    fn plan_diagnostics_record(
        &self,
        has_telemetry_session: bool,
        has_diagnostics: bool,
    ) -> PyResult<NativeBinaryCorrectionDiagnosticsRecordPlan> {
        self.plan_diagnostics_record_value(has_telemetry_session, has_diagnostics)
    }

    fn plan_summary_emit(
        &self,
        has_telemetry_session: bool,
        pending_diagnostics_count: i64,
    ) -> PyResult<NativeBinaryCorrectionSummaryEmitPlan> {
        self.plan_summary_emit_value(has_telemetry_session, pending_diagnostics_count)
    }

    fn summary_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.summary_payload_value(py)
    }
}

#[pymethods]
impl NativeBinaryCorrectionDiagnosticsRecordPlan {
    #[getter]
    fn should_record(&self) -> bool {
        self.inner.should_record
    }
}

impl NativeBinaryCorrectionSummaryEmitPlan {
    pub(crate) const fn should_flush_pending_diagnostics_value(&self) -> bool {
        self.inner.should_flush_pending_diagnostics
    }

    pub(crate) const fn should_emit_summary_value(&self) -> bool {
        self.inner.should_emit_summary
    }
}

#[pymethods]
impl NativeBinaryCorrectionSummaryEmitPlan {
    #[getter]
    fn should_flush_pending_diagnostics(&self) -> bool {
        self.should_flush_pending_diagnostics_value()
    }

    #[getter]
    fn should_emit_summary(&self) -> bool {
        self.should_emit_summary_value()
    }
}

impl NativeBinaryCorrectionSummary {
    pub(crate) fn new_summary() -> Self {
        Self { state: Mutex::new(native_callback_summary::BinaryCorrectionSummaryState::default()) }
    }

    pub(crate) fn add_null_model_failure_count_value(&self, failure_count: i64) -> PyResult<()> {
        self.lock_state()?.add_null_model_failure_count(failure_count);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_diagnostics_totals_value(
        &self,
        chunk_count: i64,
        score_only_count: i64,
        score_test_candidate_count: i64,
        firth_candidate_count: i64,
        firth_converged_count: i64,
        firth_failed_count: i64,
        firth_numerical_failure_count: i64,
        firth_max_iteration_failure_count: i64,
        firth_invalid_statistic_failure_count: i64,
        firth_step_halving_failure_count: i64,
        pseudo_firth_attempt_count: i64,
        pseudo_firth_success_count: i64,
        nr_zero_start_attempt_count: i64,
        nr_zero_start_success_count: i64,
        nr_warm_start_attempt_count: i64,
        nr_warm_start_success_count: i64,
        sparse_correction_count: i64,
        dense_correction_count: i64,
    ) -> PyResult<()> {
        if chunk_count < 0 {
            return Err(PyValueError::new_err("Binary correction diagnostics chunk_count must be non-negative."));
        }
        self.lock_state()?.add_diagnostics_totals(
            chunk_count,
            &native_callback_summary::BinaryChunkDiagnosticsInput {
                score_only_count,
                score_test_candidate_count,
                firth_candidate_count,
                firth_converged_count,
                firth_failed_count,
                firth_numerical_failure_count,
                firth_max_iteration_failure_count,
                firth_invalid_statistic_failure_count,
                firth_step_halving_failure_count,
                pseudo_firth_attempt_count,
                pseudo_firth_success_count,
                nr_zero_start_attempt_count,
                nr_zero_start_success_count,
                nr_warm_start_attempt_count,
                nr_warm_start_success_count,
                sparse_correction_count,
                dense_correction_count,
            },
        );
        Ok(())
    }

    pub(crate) fn chunk_count_with_pending_value(&self, pending_diagnostics_count: i64) -> PyResult<i64> {
        Ok(self.lock_state()?.chunk_count_with_pending(pending_diagnostics_count))
    }

    pub(crate) fn plan_diagnostics_record_value(
        &self,
        has_telemetry_session: bool,
        has_diagnostics: bool,
    ) -> PyResult<NativeBinaryCorrectionDiagnosticsRecordPlan> {
        drop(self.lock_state()?);
        Ok(native_callback_summary::BinaryCorrectionSummaryState::plan_diagnostics_record(
            has_telemetry_session,
            has_diagnostics,
        )
        .into())
    }

    pub(crate) fn plan_summary_emit_value(
        &self,
        has_telemetry_session: bool,
        pending_diagnostics_count: i64,
    ) -> PyResult<NativeBinaryCorrectionSummaryEmitPlan> {
        Ok(self.lock_state()?.plan_summary_emit(has_telemetry_session, pending_diagnostics_count).into())
    }

    pub(crate) fn summary_payload_value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.lock_state()?;
        let payload = PyDict::new(py);
        payload.set_item("chunk_count", state.chunk_count)?;
        payload.set_item("score_only_count", state.score_only_count)?;
        payload.set_item("score_test_candidate_count", state.score_test_candidate_count)?;
        payload.set_item("firth_attempted_count", state.firth_attempted_count)?;
        payload.set_item("firth_success_count", state.firth_success_count)?;
        payload.set_item("firth_failed_count", state.firth_failed_count)?;
        payload.set_item("firth_numerical_failure_count", state.firth_numerical_failure_count)?;
        payload.set_item("firth_max_iteration_failure_count", state.firth_max_iteration_failure_count)?;
        payload.set_item("firth_invalid_statistic_failure_count", state.firth_invalid_statistic_failure_count)?;
        payload.set_item("firth_step_halving_failure_count", state.firth_step_halving_failure_count)?;
        payload.set_item("pseudo_firth_attempt_count", state.pseudo_firth_attempt_count)?;
        payload.set_item("pseudo_firth_success_count", state.pseudo_firth_success_count)?;
        payload.set_item("nr_zero_start_attempt_count", state.nr_zero_start_attempt_count)?;
        payload.set_item("nr_zero_start_success_count", state.nr_zero_start_success_count)?;
        payload.set_item("nr_warm_start_attempt_count", state.nr_warm_start_attempt_count)?;
        payload.set_item("nr_warm_start_success_count", state.nr_warm_start_success_count)?;
        payload.set_item("sparse_correction_count", state.sparse_correction_count)?;
        payload.set_item("dense_correction_count", state.dense_correction_count)?;
        payload.set_item("null_model_failure_count", state.null_model_failure_count)?;
        Ok(payload)
    }

    fn lock_state(&self) -> PyResult<MutexGuard<'_, native_callback_summary::BinaryCorrectionSummaryState>> {
        self.state.lock().map_err(|_| PyRuntimeError::new_err("Binary correction summary lock was poisoned."))
    }
}

impl From<native_callback_summary::BinaryCorrectionDiagnosticsRecordPlan>
    for NativeBinaryCorrectionDiagnosticsRecordPlan
{
    fn from(record_plan: native_callback_summary::BinaryCorrectionDiagnosticsRecordPlan) -> Self {
        Self { inner: record_plan }
    }
}

impl From<native_callback_summary::BinaryCorrectionSummaryEmitPlan> for NativeBinaryCorrectionSummaryEmitPlan {
    fn from(emit_plan: native_callback_summary::BinaryCorrectionSummaryEmitPlan) -> Self {
        Self { inner: emit_plan }
    }
}

fn parse_binary_chunk_diagnostics(
    diagnostics: &Bound<'_, PyAny>,
) -> PyResult<native_callback_summary::BinaryChunkDiagnosticsInput> {
    Ok(native_callback_summary::BinaryChunkDiagnosticsInput {
        score_only_count: extract_required_integer(diagnostics, "score_only_count")?,
        score_test_candidate_count: extract_required_integer(diagnostics, "score_test_candidate_count")?,
        firth_candidate_count: extract_required_integer(diagnostics, "firth_candidate_count")?,
        firth_converged_count: extract_required_integer(diagnostics, "firth_converged_count")?,
        firth_failed_count: extract_required_integer(diagnostics, "firth_failed_count")?,
        firth_numerical_failure_count: extract_required_integer(diagnostics, "firth_numerical_failure_count")?,
        firth_max_iteration_failure_count: extract_required_integer(diagnostics, "firth_max_iteration_failure_count")?,
        firth_invalid_statistic_failure_count: extract_required_integer(
            diagnostics,
            "firth_invalid_statistic_failure_count",
        )?,
        firth_step_halving_failure_count: extract_required_integer(diagnostics, "firth_step_halving_failure_count")?,
        pseudo_firth_attempt_count: extract_required_integer(diagnostics, "pseudo_firth_attempt_count")?,
        pseudo_firth_success_count: extract_required_integer(diagnostics, "pseudo_firth_success_count")?,
        nr_zero_start_attempt_count: extract_required_integer(diagnostics, "nr_zero_start_attempt_count")?,
        nr_zero_start_success_count: extract_required_integer(diagnostics, "nr_zero_start_success_count")?,
        nr_warm_start_attempt_count: extract_required_integer(diagnostics, "nr_warm_start_attempt_count")?,
        nr_warm_start_success_count: extract_required_integer(diagnostics, "nr_warm_start_success_count")?,
        sparse_correction_count: extract_required_integer(diagnostics, "sparse_correction_count")?,
        dense_correction_count: extract_required_integer(diagnostics, "dense_correction_count")?,
    })
}

fn extract_required_integer(diagnostics: &Bound<'_, PyAny>, key: &str) -> PyResult<i64> {
    diagnostics.get_item(key)?.extract::<i64>().map_err(|error| {
        PyValueError::new_err(format!("Binary chunk diagnostics field {key:?} must be an integer: {error}"))
    })
}
