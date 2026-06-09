use std::path::Path;

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::config_frontend::{
    self, BinaryConfigData, CliOutcomeData, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData,
    InputConfigData, RegenieConfigData, TraitConfigData,
};

mod conversion;

use conversion::{
    enum_value, option_table_from_py_mapping, option_table_to_py_dict, optional_enum_value, optional_path,
    path_to_string, string_tuple, text_from_py_bytes_or_string,
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
        self.data.step
    }

    #[getter]
    fn trait_type(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "RegenieTraitType", &self.data.trait_type)
    }

    #[getter]
    fn bsize(&self) -> i64 {
        self.data.bsize
    }

    #[getter]
    fn threads(&self) -> Option<i64> {
        self.data.threads
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
        self.data.p_threshold
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
        enum_value(py, "Device", &self.data.device)
    }

    #[getter]
    fn staging_depth(&self) -> i64 {
        self.data.staging_depth
    }

    #[getter]
    fn variant_limit(&self) -> Option<i64> {
        self.data.variant_limit
    }

    #[getter]
    fn trusted_no_missing_diploid(&self) -> bool {
        self.data.trusted_no_missing_diploid
    }

    #[getter]
    fn trusted_bgen_validation_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "TrustedBgenValidationMode", &self.data.trusted_bgen_validation_mode)
    }

    #[getter]
    fn sample_key_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "SampleKeyMode", &self.data.sample_key_mode)
    }

    #[getter]
    fn multi_phenotype_sample_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "MultiPhenotypeSampleMode", &self.data.multi_phenotype_sample_mode)
    }

    #[getter]
    fn firth_batch_size(&self) -> i64 {
        self.data.firth_batch_size
    }

    #[getter]
    fn firth_candidate_capacity(&self) -> i64 {
        self.data.firth_candidate_capacity
    }

    #[getter]
    fn binary_null_maximum_iterations(&self) -> i64 {
        self.data.binary_null_maximum_iterations
    }

    #[getter]
    fn binary_null_coefficient_tolerance(&self) -> f64 {
        self.data.binary_null_coefficient_tolerance
    }

    #[getter]
    fn null_logistic_nonconvergence_policy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "NullLogisticNonconvergencePolicy", &self.data.null_logistic_nonconvergence_policy)
    }

    #[getter]
    fn binary_minimum_probability(&self) -> f64 {
        self.data.binary_minimum_probability
    }

    #[getter]
    fn binary_minimum_variance(&self) -> f64 {
        self.data.binary_minimum_variance
    }

    #[getter]
    fn binary_relative_variance_tolerance(&self) -> f64 {
        self.data.binary_relative_variance_tolerance
    }

    #[getter]
    fn linear_minimum_variance(&self) -> f64 {
        self.data.linear_minimum_variance
    }

    #[getter]
    fn linear_relative_variance_tolerance(&self) -> f64 {
        self.data.linear_relative_variance_tolerance
    }

    #[getter]
    fn firth_maximum_iterations(&self) -> i64 {
        self.data.firth_maximum_iterations
    }

    #[getter]
    fn firth_gradient_tolerance(&self) -> f64 {
        self.data.firth_gradient_tolerance
    }

    #[getter]
    fn firth_coefficient_tolerance(&self) -> f64 {
        self.data.firth_coefficient_tolerance
    }

    #[getter]
    fn firth_likelihood_tolerance(&self) -> f64 {
        self.data.firth_likelihood_tolerance
    }

    #[getter]
    fn firth_maximum_step_size(&self) -> f64 {
        self.data.firth_maximum_step_size
    }

    #[getter]
    fn firth_pseudo_maximum_iterations(&self) -> i64 {
        self.data.firth_pseudo_maximum_iterations
    }

    #[getter]
    fn firth_pseudo_inner_maximum_iterations(&self) -> i64 {
        self.data.firth_pseudo_inner_maximum_iterations
    }

    #[getter]
    fn firth_newton_raphson_zero_start_iterations(&self) -> i64 {
        self.data.firth_newton_raphson_zero_start_iterations
    }

    #[getter]
    fn firth_line_search_maximum_attempts(&self) -> i64 {
        self.data.firth_line_search_maximum_attempts
    }

    #[getter]
    fn firth_step_halving_maximum_attempts(&self) -> i64 {
        self.data.firth_step_halving_maximum_attempts
    }

    #[getter]
    fn firth_initial_response_scale(&self) -> f64 {
        self.data.firth_initial_response_scale
    }

    #[getter]
    fn firth_sparse_carrier_dosage_threshold(&self) -> f64 {
        self.data.firth_sparse_carrier_dosage_threshold
    }

    #[getter]
    fn firth_step_halving_scale(&self) -> f64 {
        self.data.firth_step_halving_scale
    }

    #[getter]
    fn null_firth_maximum_iterations(&self) -> i64 {
        self.data.null_firth_maximum_iterations
    }

    #[getter]
    fn null_firth_gradient_tolerance(&self) -> f64 {
        self.data.null_firth_gradient_tolerance
    }

    #[getter]
    fn null_firth_maximum_step_size(&self) -> f64 {
        self.data.null_firth_maximum_step_size
    }

    #[getter]
    fn null_firth_fallback_iteration_multiplier(&self) -> i64 {
        self.data.null_firth_fallback_iteration_multiplier
    }

    #[getter]
    fn null_firth_fallback_step_divisor(&self) -> f64 {
        self.data.null_firth_fallback_step_divisor
    }

    #[getter]
    fn null_firth_line_search_maximum_attempts(&self) -> i64 {
        self.data.null_firth_line_search_maximum_attempts
    }

    #[getter]
    fn null_firth_step_halving_scale(&self) -> f64 {
        self.data.null_firth_step_halving_scale
    }

    #[getter]
    fn use_block_firth_math(&self) -> bool {
        self.data.use_block_firth_math
    }

    #[getter]
    fn bgen_decode_tile_variant_count(&self) -> i64 {
        self.data.bgen_decode_tile_variant_count
    }

    #[getter]
    fn gpu_genotype_format(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "GpuGenotypeFormat", &self.data.gpu_genotype_format)
    }

    #[getter]
    fn score_dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "FloatingPointDtype", &self.data.score_dtype)
    }

    #[getter]
    fn firth_dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "FloatingPointDtype", &self.data.firth_dtype)
    }

    #[getter]
    fn jax_cache_dir(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.jax_cache_dir.as_ref())
    }

    #[getter]
    fn jax_matmul_precision(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_enum_value(py, "JaxMatmulPrecision", self.data.jax_matmul_precision.as_ref())
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
        self.data.jax_persistent_cache_min_compile_time_seconds
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
        enum_value(py, "OutputFormat", &self.data.format)
    }

    #[getter]
    fn output_run_directory(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_path(py, self.data.output_run_directory.as_ref())
    }

    #[getter]
    fn writer_threads(&self) -> i64 {
        self.data.writer_threads
    }

    #[getter]
    fn writer_queue_depth(&self) -> i64 {
        self.data.writer_queue_depth
    }

    #[getter]
    fn chunks_per_arrow_file(&self) -> i64 {
        self.data.chunks_per_arrow_file
    }

    #[getter]
    fn arrow_compression(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "ArrowCompression", &self.data.arrow_compression)
    }

    #[getter]
    fn parquet_compression(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "ParquetCompression", &self.data.parquet_compression)
    }

    #[getter]
    fn resume(&self) -> bool {
        self.data.resume
    }

    #[getter]
    fn resume_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        enum_value(py, "ResumeMode", &self.data.resume_mode)
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
        enum_value(py, "TelemetryMode", &self.data.telemetry)
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
        self.data.progress_interval_seconds
    }

    #[getter]
    fn progress_interval_chunks(&self) -> i64 {
        self.data.progress_interval_chunks
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
        self.data.trace_event_cap
    }

    #[getter]
    fn log_queue_size(&self) -> i64 {
        self.data.log_queue_size
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
        config_frontend::from_toml_path(Path::new(&path_text)).map(Self::new).map_err(config_error_to_py)
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
    fn explicit_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = self.data.explicit_options.iter().cloned().collect::<Vec<_>>();
        let builtins = PyModule::import(py, "builtins")?;
        builtins.getattr("frozenset")?.call1((values,)).map(Bound::unbind)
    }

    #[getter]
    fn is_validated(&self) -> bool {
        self.data.is_validated
    }

    fn to_toml(&self) -> PyResult<String> {
        config_frontend::dumps_toml(&self.data).map_err(config_error_to_py)
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
    let option_table = option_table_from_py_mapping(raw_options)?;
    config_frontend::from_options(&option_table).map(RegenieConfig::new).map_err(config_error_to_py)
}

#[pyfunction]
fn config_from_toml(path: &Bound<'_, PyAny>) -> PyResult<RegenieConfig> {
    RegenieConfig::from_toml(path)
}

#[pyfunction]
fn load_packaged_config() -> PyResult<RegenieConfig> {
    config_frontend::load_packaged_config_data().map(RegenieConfig::new).map_err(config_error_to_py)
}

#[pyfunction]
fn dumps_config_toml(config: &RegenieConfig) -> PyResult<String> {
    config_frontend::dumps_toml(config.data()).map_err(config_error_to_py)
}

#[pyfunction]
fn write_config_toml(config: &RegenieConfig, path: &Bound<'_, PyAny>) -> PyResult<()> {
    let path_text = path_to_string(path)?;
    config_frontend::write_toml(config.data(), Path::new(&path_text)).map_err(config_error_to_py)
}

#[pyfunction]
fn validate_regenie_config(config: &RegenieConfig) -> PyResult<()> {
    if config.data().is_validated {
        return Ok(());
    }
    config_frontend::validate_config(config.data()).map_err(config_error_to_py)
}

#[pyfunction]
fn build_config_template() -> PyResult<String> {
    config_frontend::build_template().map_err(config_error_to_py)
}

#[pyfunction]
fn explain_config_option(name: &str) -> PyResult<String> {
    config_frontend::explain_option(name).map_err(config_error_to_py)
}

#[pyfunction]
fn iter_config_explanations() -> Vec<String> {
    config_frontend::iter_explanations()
}

#[pyfunction]
fn decode_config_toml_mapping(py: Python<'_>, toml_data: &Bound<'_, PyAny>, source: &str) -> PyResult<Py<PyAny>> {
    let toml_text = text_from_py_bytes_or_string(toml_data)?;
    let option_table = config_frontend::decode_toml_text(&toml_text, source).map_err(config_error_to_py)?;
    option_table_to_py_dict(py, &option_table)
}

#[pyfunction]
fn flatten_config_toml_mapping(py: Python<'_>, raw_options: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let option_table = option_table_from_py_mapping(raw_options)?;
    let flattened_options = config_frontend::flatten_toml_mapping(&option_table);
    option_table_to_py_dict(py, &flattened_options)
}

#[pyfunction]
fn normalize_config_option_name(option_name: &str) -> String {
    config_frontend::normalize_option_name(option_name)
}

#[pyfunction]
fn normalize_config_option_dictionary(py: Python<'_>, raw_options: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let option_table = option_table_from_py_mapping(raw_options)?;
    let normalized_options = config_frontend::normalize_option_dictionary(&option_table);
    option_table_to_py_dict(py, &normalized_options)
}

#[pyfunction]
fn option_dictionary_to_config_toml_layer(
    py: Python<'_>,
    raw_options: &Bound<'_, PyAny>,
    source: &str,
) -> PyResult<Py<PyAny>> {
    let option_table = option_table_from_py_mapping(raw_options)?;
    let config_layer =
        config_frontend::option_dictionary_to_toml_config_layer(&option_table, source).map_err(config_error_to_py)?;
    let layer_dictionary = PyDict::new(py);
    layer_dictionary.set_item("toml_config", option_table_to_py_dict(py, config_layer.toml_config())?)?;
    layer_dictionary
        .set_item("explicit_options", config_layer.explicit_options().iter().cloned().collect::<Vec<_>>())?;
    Ok(layer_dictionary.into_any().unbind())
}

#[pyfunction]
#[expect(clippy::needless_pass_by_value, reason = "PyO3 extracts Python list arguments into owned Vec values.")]
fn dispatch_cli(args: Vec<String>, direct_regenie: bool) -> CliOutcome {
    CliOutcome::new(config_frontend::dispatch_cli(&args, direct_regenie))
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
    module.add_function(wrap_pyfunction!(dumps_config_toml, module)?)?;
    module.add_function(wrap_pyfunction!(write_config_toml, module)?)?;
    module.add_function(wrap_pyfunction!(validate_regenie_config, module)?)?;
    module.add_function(wrap_pyfunction!(build_config_template, module)?)?;
    module.add_function(wrap_pyfunction!(explain_config_option, module)?)?;
    module.add_function(wrap_pyfunction!(iter_config_explanations, module)?)?;
    module.add_function(wrap_pyfunction!(decode_config_toml_mapping, module)?)?;
    module.add_function(wrap_pyfunction!(flatten_config_toml_mapping, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_config_option_name, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_config_option_dictionary, module)?)?;
    module.add_function(wrap_pyfunction!(option_dictionary_to_config_toml_layer, module)?)?;
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
fn config_error_to_py(error: config_frontend::ConfigError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
