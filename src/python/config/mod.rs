use std::path::Path;

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use g_interface as interface;
use g_interface::{
    BinaryConfigData, CliOutcomeData, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData, InputConfigData,
    RegenieConfigData, TraitConfigData,
};

mod conversion;

use super::json_bridge;
use conversion::{
    enum_value, normalized_toml_table_from_py_options, optional_enum_value, optional_path, path_to_string, string_tuple,
};

#[pyclass(name = "InputConfig", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct InputConfig {
    data: InputConfigData,
}

#[pyclass(name = "TraitConfig", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct TraitConfig {
    data: TraitConfigData,
}

#[pyclass(name = "BinaryConfig", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct BinaryConfig {
    data: BinaryConfigData,
}

#[pyclass(name = "GComputeConfig", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct GComputeConfig {
    data: GComputeConfigData,
}

#[pyclass(name = "GOutputConfig", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct GOutputConfig {
    data: GOutputConfigData,
}

#[pyclass(name = "GDiagnosticsConfig", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct GDiagnosticsConfig {
    data: GDiagnosticsConfigData,
}

#[pyclass(name = "RegenieConfig", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct RegenieConfig {
    data: RegenieConfigData,
}

#[pyclass(name = "CliOutcome", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct CliOutcome {
    data: CliOutcomeData,
}

impl InputConfig {
    fn new(data: InputConfigData) -> Self {
        Self { data }
    }
}

impl TraitConfig {
    fn new(data: TraitConfigData) -> Self {
        Self { data }
    }
}

impl BinaryConfig {
    fn new(data: BinaryConfigData) -> Self {
        Self { data }
    }
}

impl GComputeConfig {
    fn new(data: GComputeConfigData) -> Self {
        Self { data }
    }
}

impl GOutputConfig {
    fn new(data: GOutputConfigData) -> Self {
        Self { data }
    }
}

impl GDiagnosticsConfig {
    fn new(data: GDiagnosticsConfigData) -> Self {
        Self { data }
    }
}

impl RegenieConfig {
    fn new(data: RegenieConfigData) -> Self {
        Self { data }
    }

    pub(crate) fn data(&self) -> &RegenieConfigData {
        &self.data
    }
}

impl CliOutcome {
    fn new(data: CliOutcomeData) -> Self {
        Self { data }
    }
}

#[pymethods]
impl InputConfig {
    #[getter]
    fn bgen(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.bgen.as_ref())
    }

    #[getter]
    fn sample(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.sample.as_ref())
    }

    #[getter]
    fn pheno_file(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.pheno_file.as_ref())
    }

    #[getter]
    fn pheno_columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        string_tuple(py, &self.data.pheno_columns)
    }

    #[getter]
    fn covar_file(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.covar_file.as_ref())
    }

    #[getter]
    fn covar_columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        string_tuple(py, &self.data.covar_columns)
    }

    #[getter]
    fn pred(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.pred.as_ref())
    }

    #[expect(clippy::needless_pass_by_value, reason = "PyO3 __richcmp__ requires owned PyRef extraction.")]
    fn __richcmp__(&self, other: PyRef<'_, Self>, operation: CompareOp) -> bool {
        compare_bool(self.data == other.data, operation)
    }
}

#[pymethods]
impl TraitConfig {
    #[getter]
    fn step(&self) -> i64 {
        i64::from(self.data.step)
    }

    #[getter]
    fn trait_type(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "RegenieTraitType", self.data.trait_type.as_str())
    }

    #[getter]
    fn bsize(&self) -> i64 {
        i64::from(self.data.bsize.get())
    }

    #[getter]
    fn threads(&self) -> Option<i64> {
        self.data.threads.map(|value| i64::from(value.get()))
    }

    #[expect(clippy::needless_pass_by_value, reason = "PyO3 __richcmp__ requires owned PyRef extraction.")]
    fn __richcmp__(&self, other: PyRef<'_, Self>, operation: CompareOp) -> bool {
        compare_bool(self.data == other.data, operation)
    }
}

#[pymethods]
impl BinaryConfig {
    #[getter]
    fn firth(&self) -> bool {
        self.data.firth
    }

    #[getter]
    fn approx(&self) -> bool {
        self.data.approx
    }

    #[getter]
    fn spa(&self) -> bool {
        self.data.spa
    }

    #[getter]
    fn p_threshold(&self) -> f64 {
        f64::from(self.data.p_threshold)
    }

    #[getter]
    fn firth_se(&self) -> bool {
        self.data.firth_se
    }

    #[expect(clippy::needless_pass_by_value, reason = "PyO3 __richcmp__ requires owned PyRef extraction.")]
    fn __richcmp__(&self, other: PyRef<'_, Self>, operation: CompareOp) -> bool {
        compare_bool(self.data == other.data, operation)
    }
}

#[pymethods]
impl GComputeConfig {
    #[getter]
    fn device(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "Device", self.data.device.as_str())
    }

    #[getter]
    fn staging_depth(&self) -> i64 {
        i64::from(self.data.staging_depth.get())
    }

    #[getter]
    fn native_callback_batch_size(&self) -> i64 {
        i64::from(self.data.native_callback_batch_size.get())
    }

    #[getter]
    fn result_in_flight_limit(&self) -> Option<i64> {
        self.data.result_in_flight_limit.map(|value| i64::from(value.get()))
    }

    #[getter]
    fn dosage_buffer_limit(&self) -> Option<i64> {
        self.data.dosage_buffer_limit.map(|value| i64::from(value.get()))
    }

    #[getter]
    fn variant_limit(&self) -> Option<i64> {
        self.data.variant_limit.map(|value| i64::from(value.get()))
    }

    #[getter]
    fn trusted_no_missing_diploid(&self) -> bool {
        self.data.trusted_no_missing_diploid
    }

    #[getter]
    fn trusted_bgen_validation_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "TrustedBgenValidationMode", self.data.trusted_bgen_validation_mode.as_str())
    }

    #[getter]
    fn sample_key_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "SampleKeyMode", self.data.sample_key_mode.as_str())
    }

    #[getter]
    fn multi_phenotype_sample_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "MultiPhenotypeSampleMode", self.data.multi_phenotype_sample_mode.as_str())
    }

    #[getter]
    fn firth_batch_size(&self) -> i64 {
        i64::from(self.data.firth_batch_size.get())
    }

    #[getter]
    fn firth_candidate_capacity(&self) -> i64 {
        i64::from(self.data.firth_candidate_capacity.get())
    }

    #[getter]
    fn binary_null_maximum_iterations(&self) -> i64 {
        i64::from(self.data.binary_null_maximum_iterations.get())
    }

    #[getter]
    fn binary_null_coefficient_tolerance(&self) -> f64 {
        f64::from(self.data.binary_null_coefficient_tolerance)
    }

    #[getter]
    fn null_logistic_nonconvergence_policy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "NullLogisticNonconvergencePolicy", self.data.null_logistic_nonconvergence_policy.as_str())
    }

    #[getter]
    fn binary_minimum_probability(&self) -> f64 {
        f64::from(self.data.binary_minimum_probability)
    }

    #[getter]
    fn binary_minimum_variance(&self) -> f64 {
        f64::from(self.data.binary_minimum_variance)
    }

    #[getter]
    fn binary_relative_variance_tolerance(&self) -> f64 {
        f64::from(self.data.binary_relative_variance_tolerance)
    }

    #[getter]
    fn linear_minimum_variance(&self) -> f64 {
        f64::from(self.data.linear_minimum_variance)
    }

    #[getter]
    fn linear_relative_variance_tolerance(&self) -> f64 {
        f64::from(self.data.linear_relative_variance_tolerance)
    }

    #[getter]
    fn firth_maximum_iterations(&self) -> i64 {
        i64::from(self.data.firth_maximum_iterations.get())
    }

    #[getter]
    fn firth_gradient_tolerance(&self) -> f64 {
        f64::from(self.data.firth_gradient_tolerance)
    }

    #[getter]
    fn firth_coefficient_tolerance(&self) -> f64 {
        f64::from(self.data.firth_coefficient_tolerance)
    }

    #[getter]
    fn firth_likelihood_tolerance(&self) -> f64 {
        f64::from(self.data.firth_likelihood_tolerance)
    }

    #[getter]
    fn firth_maximum_step_size(&self) -> f64 {
        f64::from(self.data.firth_maximum_step_size)
    }

    #[getter]
    fn firth_pseudo_maximum_iterations(&self) -> i64 {
        i64::from(self.data.firth_pseudo_maximum_iterations.get())
    }

    #[getter]
    fn firth_pseudo_inner_maximum_iterations(&self) -> i64 {
        i64::from(self.data.firth_pseudo_inner_maximum_iterations.get())
    }

    #[getter]
    fn firth_newton_raphson_zero_start_iterations(&self) -> i64 {
        i64::from(self.data.firth_newton_raphson_zero_start_iterations.get())
    }

    #[getter]
    fn firth_line_search_maximum_attempts(&self) -> i64 {
        i64::from(self.data.firth_line_search_maximum_attempts.get())
    }

    #[getter]
    fn firth_step_halving_maximum_attempts(&self) -> i64 {
        i64::from(self.data.firth_step_halving_maximum_attempts.get())
    }

    #[getter]
    fn firth_initial_response_scale(&self) -> f64 {
        f64::from(self.data.firth_initial_response_scale)
    }

    #[getter]
    fn firth_sparse_carrier_dosage_threshold(&self) -> f64 {
        f64::from(self.data.firth_sparse_carrier_dosage_threshold)
    }

    #[getter]
    fn firth_step_halving_scale(&self) -> f64 {
        f64::from(self.data.firth_step_halving_scale)
    }

    #[getter]
    fn null_firth_maximum_iterations(&self) -> i64 {
        i64::from(self.data.null_firth_maximum_iterations.get())
    }

    #[getter]
    fn null_firth_gradient_tolerance(&self) -> f64 {
        f64::from(self.data.null_firth_gradient_tolerance)
    }

    #[getter]
    fn null_firth_maximum_step_size(&self) -> f64 {
        f64::from(self.data.null_firth_maximum_step_size)
    }

    #[getter]
    fn null_firth_fallback_iteration_multiplier(&self) -> i64 {
        i64::from(self.data.null_firth_fallback_iteration_multiplier.get())
    }

    #[getter]
    fn null_firth_fallback_step_divisor(&self) -> f64 {
        f64::from(self.data.null_firth_fallback_step_divisor)
    }

    #[getter]
    fn null_firth_line_search_maximum_attempts(&self) -> i64 {
        i64::from(self.data.null_firth_line_search_maximum_attempts.get())
    }

    #[getter]
    fn null_firth_step_halving_scale(&self) -> f64 {
        f64::from(self.data.null_firth_step_halving_scale)
    }

    #[getter]
    fn use_block_firth_math(&self) -> bool {
        self.data.use_block_firth_math
    }

    #[getter]
    fn bgen_decode_tile_variant_count(&self) -> i64 {
        i64::from(self.data.bgen_decode_tile_variant_count.get())
    }

    #[getter]
    fn gpu_genotype_format(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "GpuGenotypeFormat", self.data.gpu_genotype_format.as_str())
    }

    #[getter]
    fn score_dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "FloatingPointDtype", self.data.score_dtype.as_str())
    }

    #[getter]
    fn firth_dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "FloatingPointDtype", self.data.firth_dtype.as_str())
    }

    #[getter]
    fn jax_cache_dir(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.jax_cache_dir.as_ref())
    }

    #[getter]
    fn jax_matmul_precision(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_enum_value(
            py,
            "JaxMatmulPrecision",
            self.data.jax_matmul_precision.as_ref().map(|value| value.as_str()),
        )
    }

    #[getter]
    fn jax_persistent_cache(&self) -> bool {
        self.data.jax_persistent_cache
    }

    #[getter]
    fn jax_persistent_cache_min_entry_size_bytes(&self) -> i64 {
        self.data.jax_persistent_cache_min_entry_size_bytes
    }

    #[getter]
    fn jax_persistent_cache_min_compile_time_seconds(&self) -> i64 {
        i64::from(self.data.jax_persistent_cache_min_compile_time_seconds)
    }

    #[getter]
    fn jax_xla_autotune_cache(&self) -> bool {
        self.data.jax_xla_autotune_cache
    }

    #[getter]
    fn jax_transfer_guard(&self) -> bool {
        self.data.jax_transfer_guard
    }

    #[expect(clippy::needless_pass_by_value, reason = "PyO3 __richcmp__ requires owned PyRef extraction.")]
    fn __richcmp__(&self, other: PyRef<'_, Self>, operation: CompareOp) -> bool {
        compare_bool(self.data == other.data, operation)
    }
}

#[pymethods]
impl GOutputConfig {
    #[getter]
    fn out(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.out.as_ref())
    }

    #[getter]
    fn format(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "OutputFormat", self.data.format.as_str())
    }

    #[getter]
    fn output_run_directory(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.output_run_directory.as_ref())
    }

    #[getter]
    fn writer_threads(&self) -> i64 {
        i64::from(self.data.writer_threads.get())
    }

    #[getter]
    fn writer_queue_depth(&self) -> i64 {
        i64::from(self.data.writer_queue_depth.get())
    }

    #[getter]
    fn chunks_per_arrow_file(&self) -> i64 {
        i64::from(self.data.chunks_per_arrow_file.get())
    }

    #[getter]
    fn arrow_compression(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "ArrowCompression", self.data.arrow_compression.as_str())
    }

    #[getter]
    fn parquet_compression(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "ParquetCompression", self.data.parquet_compression.as_str())
    }

    #[getter]
    fn output_statistic_dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "FloatingPointDtype", self.data.output_statistic_dtype.as_str())
    }

    #[getter]
    fn resume(&self) -> bool {
        self.data.resume
    }

    #[getter]
    fn resume_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "ResumeMode", self.data.resume_mode.as_str())
    }

    #[getter]
    fn finalize_parquet(&self) -> bool {
        self.data.finalize_parquet
    }

    #[expect(clippy::needless_pass_by_value, reason = "PyO3 __richcmp__ requires owned PyRef extraction.")]
    fn __richcmp__(&self, other: PyRef<'_, Self>, operation: CompareOp) -> bool {
        compare_bool(self.data == other.data, operation)
    }
}

#[pymethods]
impl GDiagnosticsConfig {
    #[getter]
    fn telemetry(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "TelemetryMode", self.data.telemetry.as_str())
    }

    #[getter]
    fn log_dir(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.log_dir.as_ref())
    }

    #[getter]
    fn stage_timings_json(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.stage_timings_json.as_ref())
    }

    #[getter]
    fn log_filter(&self) -> String {
        self.data.log_filter.clone()
    }

    #[getter]
    fn log_file(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.log_file.as_ref())
    }

    #[getter]
    fn log_stderr(&self) -> bool {
        self.data.log_stderr
    }

    #[getter]
    fn progress_interval_seconds(&self) -> f64 {
        f64::from(self.data.progress_interval_seconds)
    }

    #[getter]
    fn progress_interval_chunks(&self) -> i64 {
        i64::from(self.data.progress_interval_chunks.get())
    }

    #[getter]
    fn profile_summary_json(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.profile_summary_json.as_ref())
    }

    #[getter]
    fn trace_file(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.trace_file.as_ref())
    }

    #[getter]
    fn trace_filter(&self) -> String {
        self.data.trace_filter.clone()
    }

    #[getter]
    fn trace_event_cap(&self) -> i64 {
        i64::from(self.data.trace_event_cap)
    }

    #[getter]
    fn log_queue_size(&self) -> i64 {
        i64::from(self.data.log_queue_size.get())
    }

    #[getter]
    fn log_lossy(&self) -> bool {
        self.data.log_lossy
    }

    #[getter]
    fn include_source_location(&self) -> bool {
        self.data.include_source_location
    }

    #[getter]
    fn include_span_events(&self) -> bool {
        self.data.include_span_events
    }

    #[expect(clippy::needless_pass_by_value, reason = "PyO3 __richcmp__ requires owned PyRef extraction.")]
    fn __richcmp__(&self, other: PyRef<'_, Self>, operation: CompareOp) -> bool {
        compare_bool(self.data == other.data, operation)
    }
}

#[pymethods]
impl RegenieConfig {
    #[staticmethod]
    fn from_options(raw_options: &Bound<'_, PyAny>) -> PyResult<Self> {
        config_from_options(raw_options)
    }

    #[staticmethod]
    fn from_toml(path: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path_text = path_to_string(path)?;
        interface::from_toml_path(Path::new(&path_text))
            .map(Self::new)
            .map_err(|error| config_error_to_py("from_toml", error))
    }

    #[getter]
    fn input(&self) -> InputConfig {
        InputConfig::new(self.data.input.clone())
    }

    #[getter]
    fn trait_(&self) -> TraitConfig {
        TraitConfig::new(self.data.trait_config.clone())
    }

    #[getter]
    fn r#trait(&self) -> TraitConfig {
        TraitConfig::new(self.data.trait_config.clone())
    }

    #[getter]
    fn binary(&self) -> BinaryConfig {
        BinaryConfig::new(self.data.binary.clone())
    }

    #[getter]
    fn g_compute(&self) -> GComputeConfig {
        GComputeConfig::new(self.data.g_compute.clone())
    }

    #[getter]
    fn g_output(&self) -> GOutputConfig {
        GOutputConfig::new(self.data.g_output.clone())
    }

    #[getter]
    fn g_diagnostics(&self) -> GDiagnosticsConfig {
        GDiagnosticsConfig::new(self.data.g_diagnostics.clone())
    }

    #[getter]
    fn is_validated(&self) -> bool {
        self.data.is_validated
    }

    fn to_toml(&self) -> PyResult<String> {
        interface::dumps_toml(&self.data).map_err(|error| config_error_to_py("to_toml", error))
    }

    #[expect(clippy::needless_pass_by_value, reason = "PyO3 __richcmp__ requires owned PyRef extraction.")]
    fn __richcmp__(&self, other: PyRef<'_, Self>, operation: CompareOp) -> bool {
        compare_bool(regenie_config_data_equal(&self.data, &other.data), operation)
    }
}

#[pymethods]
impl CliOutcome {
    #[getter]
    fn exit_code(&self) -> i32 {
        self.data.exit_code
    }

    #[getter]
    fn stdout(&self) -> String {
        self.data.stdout.clone()
    }

    #[getter]
    fn stderr(&self) -> String {
        self.data.stderr.clone()
    }

    #[getter]
    fn config(&self, py: Python<'_>) -> Py<PyAny> {
        self.data.config.clone().map_or_else(
            || py.None(),
            |config| Py::new(py, RegenieConfig::new(config)).expect("config allocation").into_any(),
        )
    }
}

#[pyfunction]
fn config_from_options(raw_options: &Bound<'_, PyAny>) -> PyResult<RegenieConfig> {
    let option_table = normalized_toml_table_from_py_options(raw_options)?;
    interface::from_options(&option_table)
        .map(RegenieConfig::new)
        .map_err(|error| config_error_to_py("from_options", error))
}

#[pyfunction]
fn config_from_toml(path: &Bound<'_, PyAny>) -> PyResult<RegenieConfig> {
    RegenieConfig::from_toml(path)
}

#[pyfunction]
fn load_packaged_config() -> PyResult<RegenieConfig> {
    interface::load_packaged_config_data()
        .map(RegenieConfig::new)
        .map_err(|error| config_error_to_py("load_packaged_config_data", error))
}

#[pyfunction]
fn config_option_schema(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let entries = PyList::empty(py);
    for metadata in interface::config_option_metadata() {
        let entry = PyDict::new(py);
        entry.set_item("section", metadata.section)?;
        entry.set_item("toml_name", metadata.toml_name)?;
        entry.set_item("accepted_toml_names", metadata.accepted_toml_names)?;
        entry.set_item("cli_long_name", metadata.cli_long_name)?;
        entry.set_item("negative_cli_long_name", metadata.negative_cli_long_name)?;
        entry.set_item("flat_python_names", metadata.flat_python_names)?;
        entry.set_item("value_kind", metadata.value_kind.as_str())?;
        entries.append(entry)?;
    }
    Ok(entries.into_any().unbind())
}

#[pyfunction]
fn dumps_config_toml(config: &RegenieConfig) -> PyResult<String> {
    interface::dumps_toml(config.data()).map_err(|error| config_error_to_py("dumps_toml", error))
}

#[pyfunction]
fn write_config_toml(config: &RegenieConfig, path: &Bound<'_, PyAny>) -> PyResult<()> {
    let path_text = path_to_string(path)?;
    interface::write_toml(config.data(), Path::new(&path_text)).map_err(|error| config_error_to_py("write_toml", error))
}

#[pyfunction]
fn validate_regenie_config(config: &RegenieConfig) -> PyResult<()> {
    if config.data().is_validated {
        return Ok(());
    }
    interface::validate_config(config.data()).map_err(|error| config_error_to_py("validate_config", error))
}

#[pyfunction]
fn validate_regenie_config_for_run(config: &RegenieConfig) -> PyResult<()> {
    interface::validate_config_for_run(config.data())
        .map_err(|error| config_error_to_py("validate_config_for_run", error))
}

#[pyfunction]
fn compile_run_request_json(config: &RegenieConfig) -> PyResult<String> {
    let run_request = interface::compile_run_request(config.data())
        .map_err(|error| config_error_to_py("compile_run_request", error))?;
    serde_json::to_string(&run_request)
        .map_err(|error| PyValueError::new_err(format!("Failed to serialize run request: {error}.")))
}

#[pyfunction]
fn compile_run_request_payload(py: Python<'_>, config: &RegenieConfig) -> PyResult<Py<PyAny>> {
    let run_request = interface::compile_run_request(config.data())
        .map_err(|error| config_error_to_py("compile_run_request", error))?;
    let run_request_value = serde_json::to_value(&run_request)
        .map_err(|error| PyValueError::new_err(format!("Failed to serialize run request: {error}.")))?;
    json_bridge::json_value_to_py_object(py, &run_request_value)
}

#[pyfunction]
#[expect(clippy::needless_pass_by_value, reason = "PyO3 extracts Python list arguments into owned Vec values.")]
fn dispatch_cli(args: Vec<String>) -> CliOutcome {
    CliOutcome::new(interface::dispatch_cli(&args))
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<InputConfig>()?;
    module.add_class::<TraitConfig>()?;
    module.add_class::<BinaryConfig>()?;
    module.add_class::<GComputeConfig>()?;
    module.add_class::<GOutputConfig>()?;
    module.add_class::<GDiagnosticsConfig>()?;
    module.add_class::<RegenieConfig>()?;
    module.add_class::<CliOutcome>()?;
    module.add_function(wrap_pyfunction!(config_from_options, module)?)?;
    module.add_function(wrap_pyfunction!(config_from_toml, module)?)?;
    module.add_function(wrap_pyfunction!(load_packaged_config, module)?)?;
    module.add_function(wrap_pyfunction!(config_option_schema, module)?)?;
    module.add_function(wrap_pyfunction!(dumps_config_toml, module)?)?;
    module.add_function(wrap_pyfunction!(write_config_toml, module)?)?;
    module.add_function(wrap_pyfunction!(validate_regenie_config, module)?)?;
    module.add_function(wrap_pyfunction!(validate_regenie_config_for_run, module)?)?;
    module.add_function(wrap_pyfunction!(compile_run_request_json, module)?)?;
    module.add_function(wrap_pyfunction!(compile_run_request_payload, module)?)?;
    module.add_function(wrap_pyfunction!(dispatch_cli, module)?)?;
    Ok(())
}

fn compare_bool(equal: bool, operation: CompareOp) -> bool {
    match operation {
        CompareOp::Eq => equal,
        CompareOp::Ne => !equal,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => false,
    }
}

fn regenie_config_data_equal(left: &RegenieConfigData, right: &RegenieConfigData) -> bool {
    left.input == right.input
        && left.trait_config == right.trait_config
        && left.binary == right.binary
        && left.g_compute == right.g_compute
        && left.g_output == right.g_output
        && left.g_diagnostics == right.g_diagnostics
}

#[expect(clippy::needless_pass_by_value, reason = "Result::map_err passes owned errors to the adapter.")]
fn config_error_to_py(operation: &str, error: interface::ConfigError) -> PyErr {
    let error_message = error.to_string();
    tracing::warn!(
        target: "g.python.config",
        g_event = "native_config_error",
        operation = operation,
        error_message = %error_message,
        "Converting Rust config error to Python."
    );
    PyValueError::new_err(error_message)
}
