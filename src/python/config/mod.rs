use std::path::{Path, PathBuf};

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

use g_interface as interface;
use g_interface::{
    BinaryConfigData, CliOutcomeData, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData, InputConfigData,
    RegenieConfigData, TraitConfigData,
};
use g_plan as native_plan;

mod conversion;

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

#[pyclass(name = "NativeRunRequest", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeRunRequest {
    data: native_plan::RunRequest,
}

#[pyclass(name = "NativePhenotypeRunPlan", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativePhenotypeRunPlan {
    phenotype_index: i64,
    phenotype_name: String,
    output_directory_name: String,
}

#[pyclass(name = "NativePhenotypeComputeGroup", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativePhenotypeComputeGroup {
    group_mode: String,
    phenotype_indices: Vec<i64>,
    phenotype_names: Vec<String>,
    sample_mode: String,
    sample_set_fingerprint: Option<String>,
    covariate_design_fingerprint: Option<String>,
    prediction_alignment_fingerprint: Option<String>,
}

#[pyclass(name = "NativeBinaryCorrectionPlan", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeBinaryCorrectionPlan {
    method: String,
    p_threshold: f64,
    firth_se: bool,
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

impl NativeRunRequest {
    pub(crate) fn new(data: native_plan::RunRequest) -> Self {
        Self { data }
    }
}

impl NativePhenotypeRunPlan {
    fn from_native_plan(data: &native_plan::PhenotypeRunPlan) -> Self {
        Self {
            phenotype_index: i64::from(data.phenotype_index),
            phenotype_name: data.phenotype_name.clone(),
            output_directory_name: data.output_directory_name.clone(),
        }
    }
}

impl NativePhenotypeComputeGroup {
    fn from_native_group(data: &native_plan::PhenotypeComputeGroup) -> Self {
        Self {
            group_mode: data.group_mode.as_str().to_string(),
            phenotype_indices: data.phenotype_indices.iter().map(|value| i64::from(*value)).collect(),
            phenotype_names: data.phenotype_names.clone(),
            sample_mode: data.sample_mode.as_str().to_string(),
            sample_set_fingerprint: data.sample_set_fingerprint.clone(),
            covariate_design_fingerprint: data.covariate_design_fingerprint.clone(),
            prediction_alignment_fingerprint: data.prediction_alignment_fingerprint.clone(),
        }
    }

    pub(crate) fn from_host_policy_payload(data: native_plan::PhenotypeComputeGroupPayload) -> Self {
        Self {
            group_mode: data.group_mode.to_string(),
            phenotype_indices: data.phenotype_indices,
            phenotype_names: data.phenotype_names,
            sample_mode: data.sample_mode.to_string(),
            sample_set_fingerprint: data.sample_set_fingerprint,
            covariate_design_fingerprint: data.covariate_design_fingerprint,
            prediction_alignment_fingerprint: data.prediction_alignment_fingerprint,
        }
    }
}

impl NativeBinaryCorrectionPlan {
    fn from_native_plan(data: &native_plan::CorrectionPlan) -> Self {
        Self { method: data.method.as_str().to_string(), p_threshold: data.p_threshold, firth_se: data.firth_se }
    }

    pub(crate) fn from_host_policy_payload(data: native_plan::BinaryCorrectionPlanPayload) -> Self {
        Self { method: data.method.to_string(), p_threshold: data.p_threshold, firth_se: data.firth_se }
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
impl NativeRunRequest {
    #[getter]
    fn association_mode(&self) -> &str {
        self.data.association_mode.as_str()
    }

    #[getter]
    fn input_bgen_path(&self) -> &str {
        &self.data.input.bgen_path
    }

    #[getter]
    fn input_sample_path(&self) -> Option<&str> {
        self.data.input.sample_path.as_deref()
    }

    #[getter]
    fn input_phenotype_path(&self) -> &str {
        &self.data.input.phenotype_path
    }

    #[getter]
    fn input_prediction_list_path(&self) -> &str {
        &self.data.input.prediction_list_path
    }

    #[getter]
    fn input_covariate_path(&self) -> Option<&str> {
        self.data.input.covariate_path.as_deref()
    }

    #[getter]
    fn input_covariate_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.data.input.covariate_names)
    }

    #[getter]
    fn input_sample_key_mode(&self) -> &str {
        self.data.input.sample_key_mode.as_str()
    }

    #[getter]
    fn trait_type(&self) -> &str {
        self.data.trait_request.trait_type.as_str()
    }

    #[getter]
    fn trait_chunk_size(&self) -> u32 {
        self.data.trait_request.chunk_size
    }

    #[getter]
    fn trait_thread_count(&self) -> Option<u32> {
        self.data.trait_request.thread_count
    }

    #[getter]
    fn compute_device(&self) -> &str {
        self.data.compute.device.as_str()
    }

    #[getter]
    fn compute_staging_depth(&self) -> u32 {
        self.data.compute.staging_depth
    }

    #[getter]
    fn compute_native_callback_batch_size(&self) -> u32 {
        self.data.compute.native_callback_batch_size
    }

    #[getter]
    fn compute_result_in_flight_limit(&self) -> Option<u32> {
        self.data.compute.result_in_flight_limit
    }

    #[getter]
    fn compute_dosage_buffer_limit(&self) -> Option<u32> {
        self.data.compute.dosage_buffer_limit
    }

    #[getter]
    fn compute_variant_limit(&self) -> Option<u32> {
        self.data.compute.variant_limit
    }

    #[getter]
    fn compute_bgen_decode_tile_variant_count(&self) -> u32 {
        self.data.compute.bgen_decode_tile_variant_count
    }

    #[getter]
    fn compute_requested_gpu_genotype_format(&self) -> &str {
        self.data.compute.requested_gpu_genotype_format.as_str()
    }

    #[getter]
    fn compute_trusted_no_missing_diploid(&self) -> bool {
        self.data.compute.trusted_no_missing_diploid
    }

    #[getter]
    fn compute_trusted_bgen_validation_mode(&self) -> &str {
        self.data.compute.trusted_bgen_validation_mode.as_str()
    }

    #[getter]
    fn compute_multi_phenotype_sample_mode(&self) -> &str {
        self.data.compute.multi_phenotype_sample_mode.as_str()
    }

    #[getter]
    fn compute_score_dtype(&self) -> &str {
        self.data.compute.score_dtype.as_str()
    }

    #[getter]
    fn compute_firth_dtype(&self) -> &str {
        self.data.compute.firth_dtype.as_str()
    }

    #[getter]
    fn correction(&self) -> NativeBinaryCorrectionPlan {
        NativeBinaryCorrectionPlan::from_native_plan(&self.data.correction)
    }

    #[getter]
    fn output_prefix(&self) -> &str {
        &self.data.output.output_prefix
    }

    #[getter]
    fn output_run_root(&self) -> &str {
        &self.data.output.output_run_root
    }

    #[getter]
    fn output_resume(&self) -> bool {
        self.data.output.resume
    }

    #[getter]
    fn output_resume_mode(&self) -> &str {
        self.data.output.resume_mode.as_str()
    }

    #[getter]
    fn output_finalize_parquet(&self) -> bool {
        self.data.output.finalize_parquet
    }

    #[getter]
    fn output_writer_thread_count(&self) -> u32 {
        self.data.output.writer_thread_count
    }

    #[getter]
    fn output_writer_queue_depth(&self) -> u32 {
        self.data.output.writer_queue_depth
    }

    #[getter]
    fn output_chunks_per_arrow_file(&self) -> u32 {
        self.data.output.chunks_per_arrow_file
    }

    #[getter]
    fn output_arrow_compression(&self) -> &str {
        self.data.output.arrow_compression.as_str()
    }

    #[getter]
    fn output_parquet_compression(&self) -> &str {
        self.data.output.parquet_compression.as_str()
    }

    #[getter]
    fn output_format(&self) -> &str {
        self.data.output.output_format.as_str()
    }

    #[getter]
    fn output_statistic_dtype(&self) -> &str {
        self.data.output.output_statistic_dtype.as_str()
    }

    #[getter]
    fn runtime_jax_cache_directory(&self) -> Option<&str> {
        self.data.runtime.jax_cache_directory.as_deref()
    }

    #[getter]
    fn runtime_jax_matmul_precision(&self) -> Option<&str> {
        self.data.runtime.jax_matmul_precision.map(native_plan::JaxMatmulPrecision::as_str)
    }

    #[getter]
    fn runtime_persistent_cache_enabled(&self) -> bool {
        self.data.runtime.persistent_cache_enabled
    }

    #[getter]
    fn runtime_persistent_cache_min_entry_size_bytes(&self) -> i64 {
        self.data.runtime.persistent_cache_min_entry_size_bytes
    }

    #[getter]
    fn runtime_persistent_cache_min_compile_time_seconds(&self) -> u32 {
        self.data.runtime.persistent_cache_min_compile_time_seconds
    }

    #[getter]
    fn runtime_xla_autotune_cache_enabled(&self) -> bool {
        self.data.runtime.xla_autotune_cache_enabled
    }

    #[getter]
    fn runtime_transfer_guard_enabled(&self) -> bool {
        self.data.runtime.transfer_guard_enabled
    }

    #[getter]
    fn phenotype_runs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let plans = self
            .data
            .phenotype_runs
            .iter()
            .map(|plan| Py::new(py, NativePhenotypeRunPlan::from_native_plan(plan)))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, &plans)
    }

    #[getter]
    fn phenotype_compute_groups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let groups = self
            .data
            .phenotype_compute_groups
            .iter()
            .map(|group| Py::new(py, NativePhenotypeComputeGroup::from_native_group(group)))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, &groups)
    }

    #[getter]
    fn stage_timings_json(&self) -> Option<&str> {
        self.data.stage_timings_json.as_deref()
    }
}

#[pymethods]
impl NativePhenotypeRunPlan {
    #[getter]
    fn phenotype_index(&self) -> i64 {
        self.phenotype_index
    }

    #[getter]
    fn phenotype_name(&self) -> &str {
        &self.phenotype_name
    }

    #[getter]
    fn output_directory_name(&self) -> &str {
        &self.output_directory_name
    }
}

#[pymethods]
impl NativePhenotypeComputeGroup {
    #[getter]
    fn group_mode(&self) -> &str {
        &self.group_mode
    }

    #[getter]
    fn phenotype_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.phenotype_indices)
    }

    #[getter]
    fn phenotype_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.phenotype_names)
    }

    #[getter]
    fn sample_mode(&self) -> &str {
        &self.sample_mode
    }

    #[getter]
    fn sample_set_fingerprint(&self) -> Option<&str> {
        self.sample_set_fingerprint.as_deref()
    }

    #[getter]
    fn covariate_design_fingerprint(&self) -> Option<&str> {
        self.covariate_design_fingerprint.as_deref()
    }

    #[getter]
    fn prediction_alignment_fingerprint(&self) -> Option<&str> {
        self.prediction_alignment_fingerprint.as_deref()
    }
}

#[pymethods]
impl NativeBinaryCorrectionPlan {
    #[getter]
    fn method(&self) -> &str {
        &self.method
    }

    #[getter]
    fn p_threshold(&self) -> f64 {
        self.p_threshold
    }

    #[getter]
    fn firth_se(&self) -> bool {
        self.firth_se
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
fn compile_run_request(config: &RegenieConfig) -> PyResult<NativeRunRequest> {
    let run_request = interface::compile_run_request(config.data())
        .map_err(|error| config_error_to_py("compile_run_request", error))?;
    Ok(NativeRunRequest::new(run_request))
}

#[pyfunction]
#[expect(clippy::needless_pass_by_value, reason = "PyO3 extracts Python list arguments into owned Vec values.")]
fn dispatch_cli(args: Vec<String>) -> CliOutcome {
    CliOutcome::new(interface::dispatch_cli(&args))
}

#[pyfunction]
#[expect(clippy::needless_pass_by_value, reason = "PyO3 extracts Python list arguments into owned Vec values.")]
fn run_native_cli_python_bridge(
    args: Vec<String>,
    python_executable_path: &Bound<'_, PyAny>,
    sentinel_environment_variable: String,
) -> PyResult<CliOutcome> {
    let python_executable_path_text = path_to_string(python_executable_path)?;
    let execution_adapter = g_cli::PythonBridgeExecutionAdapter::new_with_environment_overrides(
        PathBuf::from(python_executable_path_text),
        vec![(sentinel_environment_variable, "1".to_string())],
    );
    let native_outcome = g_cli::dispatch_native_cli_with_adapter(&args, &execution_adapter);
    Ok(CliOutcome::new(native_cli_outcome_to_cli_outcome_data(native_outcome)))
}

fn native_cli_outcome_to_cli_outcome_data(native_outcome: g_cli::NativeCliOutcome) -> CliOutcomeData {
    CliOutcomeData {
        exit_code: native_outcome.exit_code,
        stdout: native_outcome.stdout,
        stderr: native_outcome.stderr,
        config: None,
    }
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
    module.add_class::<NativeRunRequest>()?;
    module.add_class::<NativePhenotypeRunPlan>()?;
    module.add_class::<NativePhenotypeComputeGroup>()?;
    module.add_class::<NativeBinaryCorrectionPlan>()?;
    module.add_function(wrap_pyfunction!(config_from_options, module)?)?;
    module.add_function(wrap_pyfunction!(config_from_toml, module)?)?;
    module.add_function(wrap_pyfunction!(load_packaged_config, module)?)?;
    module.add_function(wrap_pyfunction!(config_option_schema, module)?)?;
    module.add_function(wrap_pyfunction!(dumps_config_toml, module)?)?;
    module.add_function(wrap_pyfunction!(write_config_toml, module)?)?;
    module.add_function(wrap_pyfunction!(validate_regenie_config, module)?)?;
    module.add_function(wrap_pyfunction!(validate_regenie_config_for_run, module)?)?;
    module.add_function(wrap_pyfunction!(compile_run_request, module)?)?;
    module.add_function(wrap_pyfunction!(dispatch_cli, module)?)?;
    module.add_function(wrap_pyfunction!(run_native_cli_python_bridge, module)?)?;
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
