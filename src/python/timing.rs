//! PyO3 adapters for native stage timing recorder state.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use numpy::ndarray::IxDyn;
use numpy::{PyArray, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods, dtype};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use g_runtime as native_timing;

use super::logging;

#[pyclass]
pub(crate) struct NativeStageTimingRecorder {
    recorder: Mutex<native_timing::StageTimingRecorder>,
}

#[pyclass]
pub(crate) struct NativeFinalTimingOutputContext {
    inner: native_timing::FinalTimingOutputContext,
}

#[pymethods]
impl NativeFinalTimingOutputContext {
    #[getter]
    fn stage_timing_path(&self) -> Option<String> {
        self.inner.stage_timing_path.clone()
    }

    #[getter]
    fn profile_summary_path(&self) -> Option<String> {
        self.inner.profile_summary_path.clone()
    }

    #[getter]
    fn run_id(&self) -> Option<String> {
        self.inner.run_id.clone()
    }

    #[getter]
    fn force_stage_timing_recorder(&self) -> bool {
        self.inner.force_stage_timing_recorder
    }
}

#[pymethods]
impl NativeStageTimingRecorder {
    #[new]
    fn new(exact_stage_timings: bool) -> Self {
        Self::from_recorder(native_timing::StageTimingRecorder::new(exact_stage_timings))
    }

    #[staticmethod]
    fn from_config(stage_timing_path_configured: bool, force: bool) -> Option<Self> {
        native_timing::StageTimingRecorder::from_config(stage_timing_path_configured, force).map(Self::from_recorder)
    }

    #[getter]
    fn exact_stage_timings(&self) -> PyResult<bool> {
        Ok(self.lock_recorder()?.exact_stage_timings())
    }

    fn should_collect_exact_stage_timings(&self) -> PyResult<bool> {
        Ok(self.lock_recorder()?.should_collect_exact_stage_timings())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn add_stage_duration(&self, stage_name: String, duration_seconds: f64) -> PyResult<()> {
        self.lock_recorder()?.add_stage_duration(stage_name, duration_seconds);
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
        self.lock_recorder()?.add_chunk_stage_duration(native_timing::ChunkStageTiming {
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
        self.lock_recorder()?.set_native_bgen_profile(profile_snapshot);
        Ok(())
    }

    fn add_binary_chunk_diagnostics(&self, diagnostics: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed_diagnostics = parse_numeric_diagnostics_mapping(diagnostics)?;
        self.lock_recorder()?.add_binary_chunk_diagnostics(parsed_diagnostics);
        Ok(())
    }

    fn add_null_logistic_diagnostics(&self, diagnostics: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed_diagnostics = parse_null_logistic_diagnostics_mapping(diagnostics)?;
        self.lock_recorder()?.add_null_logistic_diagnostics(parsed_diagnostics);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn add_scalar_null_logistic_diagnostics_from_arrays(
        &self,
        py: Python<'_>,
        chromosome: String,
        convergence_values: &Bound<'_, PyUntypedArray>,
        iteration_count_values: &Bound<'_, PyUntypedArray>,
        firth_iteration_count_values: &Bound<'_, PyUntypedArray>,
        firth_convergence_reason_code_values: &Bound<'_, PyUntypedArray>,
        correction_method: String,
    ) -> PyResult<()> {
        let convergence_flags = parse_bool_array(py, convergence_values, "Null logistic convergence values")?;
        let iteration_counts = parse_i64_array(py, iteration_count_values, "Null logistic iteration counts")?;
        let firth_iteration_counts = parse_i64_array(py, firth_iteration_count_values, "Null Firth iteration counts")?;
        let firth_convergence_reason_codes =
            parse_i64_array(py, firth_convergence_reason_code_values, "Null Firth convergence reason codes")?;
        let diagnostics = build_scalar_null_logistic_diagnostics(
            chromosome,
            convergence_flags,
            iteration_counts,
            firth_iteration_counts,
            firth_convergence_reason_codes,
            correction_method,
        )?;
        self.lock_recorder()?.add_null_logistic_diagnostics(diagnostics);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn add_multi_null_logistic_diagnostics_from_arrays(
        &self,
        py: Python<'_>,
        chromosome: String,
        convergence_values: &Bound<'_, PyUntypedArray>,
        iteration_count_values: &Bound<'_, PyUntypedArray>,
        phenotype_names: Vec<String>,
        correction_method: String,
    ) -> PyResult<()> {
        let convergence_flags = parse_bool_array(py, convergence_values, "Null logistic convergence values")?;
        let iteration_counts = parse_i64_array(py, iteration_count_values, "Null logistic iteration counts")?;
        let diagnostics = build_multi_null_logistic_diagnostics(
            &chromosome,
            convergence_flags,
            iteration_counts,
            phenotype_names,
            &correction_method,
        )?;
        let mut recorder = self.lock_recorder()?;
        for diagnostic_payload in diagnostics {
            recorder.add_null_logistic_diagnostics(diagnostic_payload);
        }
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
        self.lock_recorder()?.add_queue_backpressure_observation(
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
        self.lock_recorder()?.add_transfer_metadata(
            native_timing::TransferMetadataKey { transfer_name, array_role, dtype_name, dimension_count: ndim },
            byte_count,
            element_count,
        );
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn add_transfer_metadata_for_shape(
        &self,
        transfer_name: String,
        array_role: String,
        dtype_name: String,
        shape_dimensions: Vec<i64>,
        item_size: i64,
    ) -> PyResult<()> {
        self.lock_recorder()?
            .add_transfer_metadata_for_shape(&transfer_name, &array_role, &dtype_name, &shape_dimensions, item_size)
            .map_err(|error| transfer_metadata_error_to_py(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn write_final_timing_outputs<'py>(
        &self,
        py: Python<'py>,
        stage_timing_path: Option<String>,
        profile_summary_path: Option<String>,
        run_id: Option<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .lock_recorder()?
            .write_final_timing_outputs(
                stage_timing_path.as_deref().map(Path::new),
                profile_summary_path.as_deref().map(Path::new),
                run_id,
            )
            .map_err(|error| timing_file_error_to_py(&error))?;
        final_timing_outputs_write_result_payload_to_dict(py, &result)
    }
}

impl NativeStageTimingRecorder {
    fn from_recorder(recorder: native_timing::StageTimingRecorder) -> Self {
        Self { recorder: Mutex::new(recorder) }
    }

    fn lock_recorder(&self) -> PyResult<MutexGuard<'_, native_timing::StageTimingRecorder>> {
        self.recorder.lock().map_err(|_| PyRuntimeError::new_err("Stage timing recorder lock was poisoned."))
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn resolve_final_timing_output_context(
    diagnostics_stage_timing_path: Option<String>,
    telemetry_session: &Bound<'_, PyAny>,
) -> PyResult<NativeFinalTimingOutputContext> {
    let telemetry_fields = optional_final_timing_telemetry_fields(telemetry_session)?;
    let context = match telemetry_fields {
        Some(fields) => native_timing::resolve_final_timing_output_context(
            diagnostics_stage_timing_path.as_deref(),
            fields.stage_timing_path.as_deref(),
            fields.profile_summary_path.as_deref(),
            fields.run_id.as_deref(),
            fields.profile_enabled,
            true,
        ),
        None => native_timing::resolve_final_timing_output_context(
            diagnostics_stage_timing_path.as_deref(),
            None,
            None,
            None,
            false,
            false,
        ),
    };
    Ok(NativeFinalTimingOutputContext { inner: context })
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn record_final_timing_outputs_write_started_diagnostic_event(
    stage_timing_path: Option<String>,
    profile_summary_path: Option<String>,
    run_id: Option<String>,
) -> PyResult<()> {
    let payload = native_timing::build_final_timing_outputs_write_started_diagnostic_payload(
        stage_timing_path.as_deref(),
        profile_summary_path.as_deref(),
        run_id.as_deref(),
    );
    let fields_json = native_timing::serialize_final_timing_outputs_write_started_diagnostic_fields_json(&payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    logging::emit_diagnostic_event(payload.level, payload.event_name, payload.message, Some(fields_json))
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeFinalTimingOutputContext>()?;
    module.add_class::<NativeStageTimingRecorder>()?;
    module.add_function(wrap_pyfunction!(resolve_final_timing_output_context, module)?)?;
    module.add_function(wrap_pyfunction!(record_final_timing_outputs_write_started_diagnostic_event, module)?)?;
    Ok(())
}

struct FinalTimingTelemetryFields {
    stage_timing_path: Option<String>,
    profile_summary_path: Option<String>,
    run_id: Option<String>,
    profile_enabled: bool,
}

fn optional_final_timing_telemetry_fields(
    telemetry_session: &Bound<'_, PyAny>,
) -> PyResult<Option<FinalTimingTelemetryFields>> {
    if telemetry_session.is_none() {
        return Ok(None);
    }
    let telemetry_paths = telemetry_session.getattr("paths")?;
    Ok(Some(FinalTimingTelemetryFields {
        stage_timing_path: optional_path_string(&telemetry_paths, "stage_timings_json")?,
        profile_summary_path: optional_path_string(&telemetry_paths, "profile_summary_json")?,
        run_id: Some(telemetry_session.getattr("run_id")?.extract::<String>()?),
        profile_enabled: telemetry_session.getattr("profile_enabled")?.extract::<bool>()?,
    }))
}

fn optional_path_string(source: &Bound<'_, PyAny>, attribute_name: &str) -> PyResult<Option<String>> {
    let value = source.getattr(attribute_name)?;
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.str()?.to_string_lossy().into_owned()))
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

fn parse_bool_array(py: Python<'_>, values: &Bound<'_, PyUntypedArray>, value_label: &str) -> PyResult<Vec<bool>> {
    let element_type = values.dtype();
    if !element_type.is_equiv_to(&dtype::<bool>(py)) {
        return Err(PyValueError::new_err(format!("{value_label} must have bool dtype.")));
    }
    let typed_values = values.cast::<PyArray<bool, IxDyn>>()?;
    let readonly_values = typed_values.readonly();
    Ok(readonly_values.as_array().iter().copied().collect())
}

fn parse_i64_array(py: Python<'_>, values: &Bound<'_, PyUntypedArray>, value_label: &str) -> PyResult<Vec<i64>> {
    let element_type = values.dtype();
    if !element_type.is_equiv_to(&dtype::<i64>(py)) {
        return Err(PyValueError::new_err(format!("{value_label} must have int64 dtype.")));
    }
    let typed_values = values.cast::<PyArray<i64, IxDyn>>()?;
    let readonly_values = typed_values.readonly();
    Ok(readonly_values.as_array().iter().copied().collect())
}

fn require_single_value<T>(values: Vec<T>, value_label: &str) -> PyResult<T> {
    if values.len() != 1 {
        return Err(PyValueError::new_err(format!("{value_label} must contain exactly one value.")));
    }
    values
        .into_iter()
        .next()
        .ok_or_else(|| PyValueError::new_err(format!("{value_label} must contain exactly one value.")))
}

fn build_scalar_null_logistic_diagnostics(
    chromosome: String,
    convergence_flags: Vec<bool>,
    iteration_counts: Vec<i64>,
    firth_iteration_counts: Vec<i64>,
    firth_convergence_reason_codes: Vec<i64>,
    correction_method: String,
) -> PyResult<BTreeMap<String, native_timing::NullLogisticDiagnosticValue>> {
    let converged = require_single_value(convergence_flags, "Scalar null logistic convergence values")?;
    let iteration_count = require_single_value(iteration_counts, "Scalar null logistic iteration counts")?;
    let firth_iteration_count = require_single_value(firth_iteration_counts, "Scalar null Firth iteration counts")?;
    let firth_convergence_reason_code =
        require_single_value(firth_convergence_reason_codes, "Scalar null Firth convergence reason codes")?;
    let mut diagnostics = BTreeMap::new();
    diagnostics.insert("chromosome".to_string(), native_timing::NullLogisticDiagnosticValue::Text(chromosome));
    diagnostics
        .insert("iteration_count".to_string(), native_timing::NullLogisticDiagnosticValue::Integer(iteration_count));
    diagnostics
        .insert("converged".to_string(), native_timing::NullLogisticDiagnosticValue::Integer(i64::from(converged)));
    diagnostics.insert(
        "firth_iteration_count".to_string(),
        native_timing::NullLogisticDiagnosticValue::Integer(firth_iteration_count),
    );
    diagnostics.insert(
        "firth_convergence_reason_code".to_string(),
        native_timing::NullLogisticDiagnosticValue::Integer(firth_convergence_reason_code),
    );
    diagnostics
        .insert("correction_method".to_string(), native_timing::NullLogisticDiagnosticValue::Text(correction_method));
    Ok(diagnostics)
}

fn build_multi_null_logistic_diagnostics(
    chromosome: &str,
    convergence_flags: Vec<bool>,
    iteration_counts: Vec<i64>,
    phenotype_names: Vec<String>,
    correction_method: &str,
) -> PyResult<Vec<BTreeMap<String, native_timing::NullLogisticDiagnosticValue>>> {
    if convergence_flags.len() != iteration_counts.len() {
        return Err(PyValueError::new_err(format!(
            "Null logistic convergence value count ({}) must match iteration count value count ({}).",
            convergence_flags.len(),
            iteration_counts.len()
        )));
    }
    if phenotype_names.len() != convergence_flags.len() {
        return Err(PyValueError::new_err(format!(
            "Null logistic phenotype name count ({}) must match convergence value count ({}).",
            phenotype_names.len(),
            convergence_flags.len()
        )));
    }
    convergence_flags
        .into_iter()
        .zip(iteration_counts)
        .zip(phenotype_names)
        .map(|((converged, iteration_count), phenotype_name)| {
            let mut diagnostics = BTreeMap::new();
            diagnostics.insert(
                "chromosome".to_string(),
                native_timing::NullLogisticDiagnosticValue::Text(chromosome.to_string()),
            );
            diagnostics
                .insert("phenotype".to_string(), native_timing::NullLogisticDiagnosticValue::Text(phenotype_name));
            diagnostics.insert(
                "iteration_count".to_string(),
                native_timing::NullLogisticDiagnosticValue::Integer(iteration_count),
            );
            diagnostics.insert(
                "converged".to_string(),
                native_timing::NullLogisticDiagnosticValue::Integer(i64::from(converged)),
            );
            diagnostics.insert(
                "correction_method".to_string(),
                native_timing::NullLogisticDiagnosticValue::Text(correction_method.to_string()),
            );
            Ok(diagnostics)
        })
        .collect()
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

fn timing_file_error_to_py(error: &native_timing::TimingFileError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn transfer_metadata_error_to_py(error: &native_timing::TransferMetadataError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn final_timing_outputs_write_result_payload_to_dict<'py>(
    py: Python<'py>,
    payload: &native_timing::FinalTimingOutputsWriteResultPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let result_payload = PyDict::new(py);
    result_payload.set_item("wrote_stage_timing_snapshot", payload.wrote_stage_timing_snapshot)?;
    result_payload.set_item("wrote_profile_summary", payload.wrote_profile_summary)?;
    Ok(result_payload)
}
