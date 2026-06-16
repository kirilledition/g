//! PyO3 adapters for native stage timing recorder state.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::timing as native_timing;

#[pyclass]
pub(crate) struct NativeStageTimingRecorder {
    exact_stage_timings: bool,
    state: Mutex<native_timing::StageTimingState>,
}

#[pymethods]
impl NativeStageTimingRecorder {
    #[new]
    fn new(exact_stage_timings: bool) -> Self {
        Self { exact_stage_timings, state: Mutex::new(native_timing::StageTimingState::default()) }
    }

    #[getter]
    fn exact_stage_timings(&self) -> bool {
        self.exact_stage_timings
    }

    #[allow(clippy::needless_pass_by_value)]
    fn add_stage_duration(&self, stage_name: String, duration_seconds: f64) -> PyResult<()> {
        self.lock_state()?.add_stage_duration(stage_name, duration_seconds);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn add_chunk_stage_duration(
        &self,
        chunk_identifier: i64,
        chromosome: String,
        variant_start_index: i64,
        variant_stop_index: i64,
        variant_count: i64,
        stage_name: String,
        duration_seconds: f64,
    ) -> PyResult<()> {
        self.lock_state()?.add_chunk_stage_duration(native_timing::ChunkStageTiming {
            chunk_identifier,
            chromosome,
            variant_start_index,
            variant_stop_index,
            variant_count,
            stage_name,
            duration_seconds,
        });
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn set_native_bgen_profile(&self, profile_snapshot: BTreeMap<String, i64>) -> PyResult<()> {
        self.lock_state()?.set_native_bgen_profile(profile_snapshot);
        Ok(())
    }

    fn add_binary_chunk_diagnostics(&self, diagnostics: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed_diagnostics = parse_numeric_diagnostics_mapping(diagnostics)?;
        self.lock_state()?.add_binary_chunk_diagnostics(parsed_diagnostics);
        Ok(())
    }

    fn add_null_logistic_diagnostics(&self, diagnostics: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed_diagnostics = parse_null_logistic_diagnostics_mapping(diagnostics)?;
        self.lock_state()?.add_null_logistic_diagnostics(parsed_diagnostics);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn add_queue_backpressure_observation(
        &self,
        queue_name: String,
        operation_name: String,
        queue_depth: i64,
        queue_capacity: i64,
        elapsed_seconds: f64,
        blocked_seconds: f64,
    ) -> PyResult<()> {
        self.lock_state()?.add_queue_backpressure_observation(
            native_timing::QueueBackpressureKey { queue_name, operation_name },
            queue_depth,
            queue_capacity,
            elapsed_seconds,
            blocked_seconds,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn add_transfer_metadata(
        &self,
        transfer_name: String,
        array_role: String,
        dtype_name: String,
        ndim: i64,
        byte_count: i64,
        element_count: i64,
    ) -> PyResult<()> {
        self.lock_state()?.add_transfer_metadata(
            native_timing::TransferMetadataKey { transfer_name, array_role, dtype_name, dimension_count: ndim },
            byte_count,
            element_count,
        );
        Ok(())
    }

    fn snapshot_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.lock_state()?;
        let payload = PyDict::new(py);
        payload.set_item("stage_totals_seconds", build_float_mapping(py, &state.stage_totals_seconds)?)?;
        payload.set_item("stage_counts", build_integer_mapping(py, &state.stage_counts)?)?;
        payload.set_item("chunk_stage_timings", build_chunk_stage_timing_payloads(py, &state.chunk_stage_timings)?)?;
        payload.set_item("native_bgen_profile", build_integer_mapping(py, &state.native_bgen_profile)?)?;
        payload.set_item(
            "binary_chunk_diagnostics",
            build_binary_chunk_diagnostics_payloads(py, &state.binary_chunk_diagnostics)?,
        )?;
        payload.set_item(
            "null_logistic_diagnostics",
            build_null_logistic_diagnostics_payloads(py, &state.null_logistic_diagnostics)?,
        )?;
        payload.set_item("queue_backpressure", build_queue_backpressure_payloads(py, &state.queue_backpressure)?)?;
        payload.set_item("transfer_metadata", build_transfer_metadata_payloads(py, &state.transfer_metadata)?)?;
        Ok(payload)
    }
}

impl NativeStageTimingRecorder {
    fn lock_state(&self) -> PyResult<MutexGuard<'_, native_timing::StageTimingState>> {
        self.state.lock().map_err(|_| PyRuntimeError::new_err("Stage timing recorder lock was poisoned."))
    }
}

fn parse_numeric_diagnostics_mapping(
    diagnostics: &Bound<'_, PyAny>,
) -> PyResult<BTreeMap<String, native_timing::NumericDiagnosticValue>> {
    let mut parsed_diagnostics = BTreeMap::new();
    for item in diagnostics.call_method0("items")?.try_iter()? {
        let pair = item?;
        let key = pair.get_item(0)?.extract::<String>()?;
        let value = pair.get_item(1)?;
        parsed_diagnostics.insert(key, parse_numeric_diagnostic_value(&value)?);
    }
    Ok(parsed_diagnostics)
}

fn parse_null_logistic_diagnostics_mapping(
    diagnostics: &Bound<'_, PyAny>,
) -> PyResult<BTreeMap<String, native_timing::NullLogisticDiagnosticValue>> {
    let mut parsed_diagnostics = BTreeMap::new();
    for item in diagnostics.call_method0("items")?.try_iter()? {
        let pair = item?;
        let key = pair.get_item(0)?.extract::<String>()?;
        let value = pair.get_item(1)?;
        let parsed_value = if null_logistic_integer_key(&key) {
            native_timing::NullLogisticDiagnosticValue::Integer(
                value
                    .extract::<i64>()
                    .or_else(|_| value.str()?.extract::<String>()?.parse::<i64>().map_err(PyValueError::new_err))?,
            )
        } else {
            native_timing::NullLogisticDiagnosticValue::Text(value.str()?.to_string_lossy().into_owned())
        };
        parsed_diagnostics.insert(key, parsed_value);
    }
    Ok(parsed_diagnostics)
}

fn parse_numeric_diagnostic_value(value: &Bound<'_, PyAny>) -> PyResult<native_timing::NumericDiagnosticValue> {
    if let Ok(integer_value) = value.extract::<i64>() {
        return Ok(native_timing::NumericDiagnosticValue::Integer(integer_value));
    }
    if let Ok(float_value) = value.extract::<f64>() {
        return Ok(native_timing::NumericDiagnosticValue::Float(float_value));
    }
    Err(PyValueError::new_err("Binary chunk diagnostics must contain numeric values."))
}

fn null_logistic_integer_key(key: &str) -> bool {
    matches!(key, "iteration_count" | "converged" | "firth_iteration_count" | "firth_convergence_reason_code")
}

fn build_float_mapping<'py>(py: Python<'py>, values: &BTreeMap<String, f64>) -> PyResult<Bound<'py, PyDict>> {
    let mapping = PyDict::new(py);
    for (key, value) in values {
        mapping.set_item(key, value)?;
    }
    Ok(mapping)
}

fn build_integer_mapping<'py>(py: Python<'py>, values: &BTreeMap<String, i64>) -> PyResult<Bound<'py, PyDict>> {
    let mapping = PyDict::new(py);
    for (key, value) in values {
        mapping.set_item(key, value)?;
    }
    Ok(mapping)
}

fn build_chunk_stage_timing_payloads<'py>(
    py: Python<'py>,
    timings: &[native_timing::ChunkStageTiming],
) -> PyResult<Bound<'py, PyTuple>> {
    let payloads = timings
        .iter()
        .map(|timing| {
            let payload = PyDict::new(py);
            payload.set_item("chunk_identifier", timing.chunk_identifier)?;
            payload.set_item("chromosome", &timing.chromosome)?;
            payload.set_item("variant_start_index", timing.variant_start_index)?;
            payload.set_item("variant_stop_index", timing.variant_stop_index)?;
            payload.set_item("variant_count", timing.variant_count)?;
            payload.set_item("stage_name", &timing.stage_name)?;
            payload.set_item("duration_seconds", timing.duration_seconds)?;
            Ok(payload)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &payloads)
}

fn build_binary_chunk_diagnostics_payloads<'py>(
    py: Python<'py>,
    diagnostics: &[BTreeMap<String, native_timing::NumericDiagnosticValue>],
) -> PyResult<Bound<'py, PyTuple>> {
    let payloads = diagnostics
        .iter()
        .map(|diagnostic_mapping| {
            let payload = PyDict::new(py);
            for (key, value) in diagnostic_mapping {
                match value {
                    native_timing::NumericDiagnosticValue::Integer(integer_value) => {
                        payload.set_item(key, integer_value)?;
                    }
                    native_timing::NumericDiagnosticValue::Float(float_value) => {
                        payload.set_item(key, float_value)?;
                    }
                }
            }
            Ok(payload)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &payloads)
}

fn build_null_logistic_diagnostics_payloads<'py>(
    py: Python<'py>,
    diagnostics: &[BTreeMap<String, native_timing::NullLogisticDiagnosticValue>],
) -> PyResult<Bound<'py, PyTuple>> {
    let payloads = diagnostics
        .iter()
        .map(|diagnostic_mapping| {
            let payload = PyDict::new(py);
            for (key, value) in diagnostic_mapping {
                match value {
                    native_timing::NullLogisticDiagnosticValue::Integer(integer_value) => {
                        payload.set_item(key, integer_value)?;
                    }
                    native_timing::NullLogisticDiagnosticValue::Text(text_value) => {
                        payload.set_item(key, text_value)?;
                    }
                }
            }
            Ok(payload)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &payloads)
}

fn build_queue_backpressure_payloads<'py>(
    py: Python<'py>,
    queue_backpressure: &BTreeMap<native_timing::QueueBackpressureKey, native_timing::QueueBackpressureAccumulator>,
) -> PyResult<Bound<'py, PyTuple>> {
    let payloads = queue_backpressure
        .iter()
        .map(|(key, accumulator)| {
            let payload = PyDict::new(py);
            payload.set_item("queue_name", &key.queue_name)?;
            payload.set_item("operation_name", &key.operation_name)?;
            payload.set_item("observation_count", accumulator.observation_count)?;
            payload.set_item("max_depth", accumulator.max_depth)?;
            payload.set_item("max_capacity", accumulator.max_capacity)?;
            payload.set_item("total_elapsed_seconds", accumulator.total_elapsed_seconds)?;
            payload.set_item("total_blocked_seconds", accumulator.total_blocked_seconds)?;
            Ok(payload)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &payloads)
}

fn build_transfer_metadata_payloads<'py>(
    py: Python<'py>,
    transfer_metadata: &BTreeMap<native_timing::TransferMetadataKey, native_timing::TransferMetadataAccumulator>,
) -> PyResult<Bound<'py, PyTuple>> {
    let payloads = transfer_metadata
        .iter()
        .map(|(key, accumulator)| {
            let payload = PyDict::new(py);
            payload.set_item("transfer_name", &key.transfer_name)?;
            payload.set_item("array_role", &key.array_role)?;
            payload.set_item("dtype_name", &key.dtype_name)?;
            payload.set_item("ndim", key.dimension_count)?;
            payload.set_item("observation_count", accumulator.observation_count)?;
            payload.set_item("total_bytes", accumulator.total_bytes)?;
            payload.set_item("max_bytes", accumulator.max_bytes)?;
            payload.set_item("total_elements", accumulator.total_elements)?;
            Ok(payload)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &payloads)
}
