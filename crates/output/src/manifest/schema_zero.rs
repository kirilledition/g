//! Strict schema-zero execution-plan wire contract.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(test)]
use serde_json::json;

use crate::error::{OutputError, OutputResult};

use super::RESUME_POLICY;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExecutionPlanSchemaZero {
    pub(super) association_mode: g_plan::AssociationMode,
    pub(super) association_backend: AssociationBackendSchemaZero,
    pub(super) bgen: BgenFingerprintSchemaZero,
    pub(super) sample: FileFingerprintSchemaZero,
    pub(super) phenotype_file: FileFingerprintSchemaZero,
    pub(super) phenotype_name: String,
    pub(super) covariate_file: RequiredNullableSchemaZero<FileFingerprintSchemaZero>,
    pub(super) covariate_names: Vec<String>,
    pub(super) prediction_inputs: PredictionInputsSchemaZero,
    pub(super) sample_count: i64,
    pub(super) variant_count: i64,
    pub(super) chunk_size: u32,
    pub(super) binary_correction_plan: BinaryCorrectionPlanSchemaZero,
    pub(super) binary_kernel_config: RequiredNullableSchemaZero<KernelPlanSchemaZero>,
    pub(super) jax_policy: JaxPolicySchemaZero,
    pub(super) score_dtype: FloatingPointDtypeSchemaZero,
    pub(super) multi_phenotype_sample_mode: g_plan::MultiPhenotypeSampleMode,
    pub(super) phenotype_compute_group_id: String,
    pub(super) sample_set_fingerprint: String,
    pub(super) covariate_design_fingerprint: String,
    pub(super) phenotype_design_fingerprint: String,
    pub(super) prediction_alignment_fingerprint: String,
    pub(super) output_writer: OutputWriterSchemaZero,
    pub(super) resume_policy: ResumePolicySchemaZero,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) enum AssociationBackendKindSchemaZero {
    #[serde(rename = "jax_dosage")]
    JaxDosage,
    #[serde(rename = "jax_packed8")]
    JaxPacked8,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AssociationBackendSchemaZero {
    pub(super) kind: AssociationBackendKindSchemaZero,
    pub(super) genotype_format: g_plan::GpuGenotypeFormat,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BgenFingerprintSchemaZero {
    pub(super) content_sha256: RequiredNullableSchemaZero<g_genotype_contracts::BgenContentSha256>,
    pub(super) byte_count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) enum FileContentHashAlgorithmSchemaZero {
    #[serde(rename = "sha256")]
    Sha256,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FileFingerprintSchemaZero {
    pub(super) path: String,
    pub(super) size: u64,
    pub(super) mtime_ns: i64,
    pub(super) content_hash_algorithm: FileContentHashAlgorithmSchemaZero,
    pub(super) content_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PredictionInputsSchemaZero {
    pub(super) prediction_list: FileFingerprintSchemaZero,
    pub(super) loco_files: Vec<PredictionLocoFileFingerprintSchemaZero>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PredictionLocoFileFingerprintSchemaZero {
    pub(super) phenotype: String,
    pub(super) path: String,
    pub(super) size: u64,
    pub(super) mtime_ns: i64,
    pub(super) content_hash_algorithm: FileContentHashAlgorithmSchemaZero,
    pub(super) content_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BinaryCorrectionPlanSchemaZero {
    pub(super) method: g_plan::BinaryFallbackMethod,
    pub(super) p_threshold: g_plan::Probability,
    pub(super) firth_se: bool,
    pub(super) approximate_firth_sparse_pseudo_budget_policy:
        RequiredNullableSchemaZero<ApproximateFirthSparsePseudoBudgetPolicySchemaZero>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) enum ApproximateFirthSparsePseudoBudgetPolicySchemaZero {
    #[serde(rename = "half_total_uncapped_by_dense_cap")]
    HalfTotalUncappedByDenseCap,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct KernelPlanSchemaZero {
    linear: LinearKernelPlanSchemaZero,
    binary_null: BinaryNullKernelPlanSchemaZero,
    firth: FirthKernelPlanSchemaZero,
    null_firth: NullFirthKernelPlanSchemaZero,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LinearKernelPlanSchemaZero {
    minimum_variance: g_plan::PositiveF32,
    relative_variance_tolerance: g_plan::PositiveF32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BinaryNullKernelPlanSchemaZero {
    maximum_iterations: u32,
    coefficient_tolerance: g_plan::PositiveF32,
    nonconvergence_policy: g_plan::NullLogisticNonconvergencePolicy,
    minimum_probability: g_plan::ProbabilityFloor,
    minimum_variance: g_plan::PositiveF32,
    relative_variance_tolerance: g_plan::PositiveF32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FirthKernelPlanSchemaZero {
    batch_size: u32,
    candidate_capacity: u32,
    maximum_iterations: u32,
    gradient_tolerance: g_plan::PositiveF64,
    maximum_step_size: g_plan::PositiveF64,
    pseudo_maximum_iterations: u32,
    pseudo_inner_maximum_iterations: u32,
    line_search_maximum_attempts: u32,
    sparse_carrier_dosage_threshold: g_plan::DosageThreshold,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NullFirthKernelPlanSchemaZero {
    maximum_iterations: u32,
    gradient_tolerance: g_plan::PositiveF64,
    maximum_step_size: g_plan::PositiveF64,
    fallback_iteration_multiplier: u32,
    fallback_step_divisor: g_plan::PositiveF64,
    line_search_maximum_attempts: u32,
    step_halving_scale: g_plan::StepScale,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) enum MatmulPrecisionSchemaZero {
    #[serde(rename = "float32")]
    Float32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) enum ApproximateFirthPseudoInnerPolicySchemaZero {
    #[serde(rename = "float32_elementwise_float64_reduction")]
    Float32ElementwiseFloat64Reduction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct JaxPolicySchemaZero {
    pub(super) device: g_plan::Device,
    pub(super) enable_x64: bool,
    pub(super) matmul_precision: MatmulPrecisionSchemaZero,
    pub(super) approximate_firth_pseudo_inner_policy:
        RequiredNullableSchemaZero<ApproximateFirthPseudoInnerPolicySchemaZero>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) enum ParquetCompressionSchemaZero {
    #[serde(rename = "zstd")]
    Zstd,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) enum ParquetFloatColumnEncodingSchemaZero {
    #[serde(rename = "BYTE_STREAM_SPLIT")]
    ByteStreamSplit,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) enum FloatingPointDtypeSchemaZero {
    #[serde(rename = "float32")]
    Float32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OutputWriterSchemaZero {
    pub(super) writer_thread_count: u32,
    pub(super) writer_queue_depth: u64,
    pub(super) chunks_per_parquet_file: u64,
    pub(super) parquet_compression: ParquetCompressionSchemaZero,
    pub(super) parquet_writer_version: i32,
    pub(super) parquet_write_batch_size: u64,
    pub(super) parquet_max_row_group_size: u64,
    pub(super) parquet_float_column_encoding: ParquetFloatColumnEncodingSchemaZero,
    pub(super) result_statistic_dtype: FloatingPointDtypeSchemaZero,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) enum ResumePolicySchemaZero {
    #[serde(rename = "lineage_receipts_exact_coverage")]
    LineageReceiptsExactCoverage,
}

// Unlike `Option`, this untagged enum rejects Serde's missing-field
// deserializer while retaining the exact value-or-null JSON representation.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub(super) enum RequiredNullableSchemaZero<ValueType> {
    Value(ValueType),
    Null(()),
}

impl<ValueType> RequiredNullableSchemaZero<ValueType> {
    pub(super) fn new(value: Option<ValueType>) -> Self {
        match value {
            Some(value) => Self::Value(value),
            None => Self::Null(()),
        }
    }

    fn as_ref(&self) -> Option<&ValueType> {
        match self {
            Self::Value(value) => Some(value),
            Self::Null(()) => None,
        }
    }
}

impl From<&g_plan::KernelPlan> for KernelPlanSchemaZero {
    fn from(kernel_plan: &g_plan::KernelPlan) -> Self {
        let g_plan::KernelPlan { linear, binary_null, firth, null_firth } = kernel_plan;
        let linear = {
            let g_plan::LinearKernelPlan { minimum_variance, relative_variance_tolerance } = *linear;
            LinearKernelPlanSchemaZero { minimum_variance, relative_variance_tolerance }
        };
        let binary_null = {
            let g_plan::BinaryNullKernelPlan {
                maximum_iterations,
                coefficient_tolerance,
                nonconvergence_policy,
                minimum_probability,
                minimum_variance,
                relative_variance_tolerance,
            } = *binary_null;
            BinaryNullKernelPlanSchemaZero {
                maximum_iterations,
                coefficient_tolerance,
                nonconvergence_policy,
                minimum_probability,
                minimum_variance,
                relative_variance_tolerance,
            }
        };
        let firth = {
            let g_plan::FirthKernelPlan {
                batch_size,
                candidate_capacity,
                maximum_iterations,
                gradient_tolerance,
                maximum_step_size,
                pseudo_maximum_iterations,
                pseudo_inner_maximum_iterations,
                line_search_maximum_attempts,
                sparse_carrier_dosage_threshold,
            } = *firth;
            FirthKernelPlanSchemaZero {
                batch_size,
                candidate_capacity,
                maximum_iterations,
                gradient_tolerance,
                maximum_step_size,
                pseudo_maximum_iterations,
                pseudo_inner_maximum_iterations,
                line_search_maximum_attempts,
                sparse_carrier_dosage_threshold,
            }
        };
        let null_firth = {
            let g_plan::NullFirthKernelPlan {
                maximum_iterations,
                gradient_tolerance,
                maximum_step_size,
                fallback_iteration_multiplier,
                fallback_step_divisor,
                line_search_maximum_attempts,
                step_halving_scale,
            } = *null_firth;
            NullFirthKernelPlanSchemaZero {
                maximum_iterations,
                gradient_tolerance,
                maximum_step_size,
                fallback_iteration_multiplier,
                fallback_step_divisor,
                line_search_maximum_attempts,
                step_halving_scale,
            }
        };

        Self { linear, binary_null, firth, null_firth }
    }
}

impl ExecutionPlanSchemaZero {
    #[cfg(test)]
    pub(crate) fn from_value(execution_plan: Value) -> OutputResult<Self> {
        let schema = serde_json::from_value::<Self>(execution_plan).map_err(|error| {
            OutputError::InvalidInput(format!("Run manifest execution plan schema zero is invalid: {error}"))
        })?;
        schema.validate()?;
        Ok(schema)
    }

    pub(crate) fn to_value(&self) -> OutputResult<Value> {
        serde_json::to_value(self).map_err(OutputError::runtime)
    }

    pub(crate) fn gpu_genotype_format(&self) -> g_plan::GpuGenotypeFormat {
        self.association_backend.genotype_format
    }

    pub(crate) fn bgen_content_fingerprint(&self) -> Option<g_genotype_contracts::BgenContentFingerprint> {
        self.bgen.content_fingerprint()
    }

    pub(crate) fn phenotype_name(&self) -> &str {
        &self.phenotype_name
    }

    pub(crate) fn runtime_device(&self) -> g_plan::Device {
        self.jax_policy.device
    }

    pub(crate) fn writer_thread_count(&self) -> u32 {
        self.output_writer.writer_thread_count
    }

    pub(crate) fn uses_approximate_firth(&self) -> bool {
        self.association_mode == g_plan::AssociationMode::Regenie2Binary
            && self.binary_correction_plan.method == g_plan::BinaryFallbackMethod::FirthApproximate
    }

    pub(crate) fn phenotype_compute_group_id(&self) -> &str {
        &self.phenotype_compute_group_id
    }

    pub(crate) fn validate(&self) -> OutputResult<()> {
        self.bgen.validate();
        self.sample.validate("sample")?;
        self.phenotype_file.validate("phenotype_file")?;
        validate_non_empty_execution_plan_string(&self.phenotype_name, "phenotype_name")?;
        if let Some(covariate_file) = self.covariate_file.as_ref() {
            covariate_file.validate("covariate_file")?;
        }
        for covariate_name in &self.covariate_names {
            validate_non_empty_execution_plan_string(covariate_name, "covariate_names entry")?;
        }
        self.prediction_inputs.validate()?;
        validate_positive_i64(self.sample_count, "sample_count")?;
        validate_positive_i64(self.variant_count, "variant_count")?;
        validate_positive_u32(self.chunk_size, "chunk_size")?;
        if let Some(binary_kernel_config) = self.binary_kernel_config.as_ref() {
            binary_kernel_config.validate()?;
        }
        self.jax_policy.validate()?;
        self.output_writer.validate()?;
        validate_non_empty_execution_plan_string(&self.phenotype_compute_group_id, "phenotype_compute_group_id")?;
        validate_non_empty_execution_plan_string(&self.sample_set_fingerprint, "sample_set_fingerprint")?;
        validate_non_empty_execution_plan_string(&self.covariate_design_fingerprint, "covariate_design_fingerprint")?;
        validate_non_empty_execution_plan_string(&self.phenotype_design_fingerprint, "phenotype_design_fingerprint")?;
        validate_non_empty_execution_plan_string(
            &self.prediction_alignment_fingerprint,
            "prediction_alignment_fingerprint",
        )?;
        if self.score_dtype != FloatingPointDtypeSchemaZero::Float32 {
            return Err(invalid_execution_plan("field 'score_dtype' must equal 'float32'"));
        }
        if self.resume_policy != ResumePolicySchemaZero::LineageReceiptsExactCoverage
            || RESUME_POLICY != "lineage_receipts_exact_coverage"
        {
            return Err(invalid_execution_plan("field 'resume_policy' must equal 'lineage_receipts_exact_coverage'"));
        }
        let expected_backend_kind = match self.association_backend.genotype_format {
            g_plan::GpuGenotypeFormat::Dosage => AssociationBackendKindSchemaZero::JaxDosage,
            g_plan::GpuGenotypeFormat::Packed8 => AssociationBackendKindSchemaZero::JaxPacked8,
        };
        if self.association_backend.kind != expected_backend_kind {
            return Err(invalid_execution_plan("association backend kind does not match its genotype format"));
        }
        if self.jax_policy.device == g_plan::Device::Cpu
            && self.association_backend.genotype_format != g_plan::GpuGenotypeFormat::Dosage
        {
            return Err(invalid_execution_plan("CPU JAX policy requires dosage genotype format"));
        }
        self.validate_mode_dependent_policy()
    }

    fn validate_mode_dependent_policy(&self) -> OutputResult<()> {
        let binary_kernel_config = self.binary_kernel_config.as_ref();
        let approximate_firth_policy = self.jax_policy.approximate_firth_pseudo_inner_policy.as_ref();
        let sparse_pseudo_budget_policy =
            self.binary_correction_plan.approximate_firth_sparse_pseudo_budget_policy.as_ref();
        match self.association_mode {
            g_plan::AssociationMode::Regenie2Linear => {
                if binary_kernel_config.is_some() {
                    return Err(invalid_execution_plan(
                        "field 'binary_kernel_config' must be null for linear association",
                    ));
                }
                if self.binary_correction_plan.method != g_plan::BinaryFallbackMethod::ScoreOnly
                    || self.binary_correction_plan.firth_se
                {
                    return Err(invalid_execution_plan(
                        "linear association requires score-only binary correction with Firth standard errors disabled",
                    ));
                }
                if approximate_firth_policy.is_some() {
                    return Err(invalid_execution_plan(
                        "approximate-Firth JAX policy must be null for linear association",
                    ));
                }
                if sparse_pseudo_budget_policy.is_some() {
                    return Err(invalid_execution_plan(
                        "field 'binary_correction_plan.approximate_firth_sparse_pseudo_budget_policy' must be null for linear association",
                    ));
                }
            }
            g_plan::AssociationMode::Regenie2Binary => {
                if binary_kernel_config.is_none() {
                    return Err(invalid_execution_plan(
                        "field 'binary_kernel_config' must contain an object for binary association",
                    ));
                }
                match self.binary_correction_plan.method {
                    g_plan::BinaryFallbackMethod::ScoreOnly => {
                        if self.binary_correction_plan.firth_se {
                            return Err(invalid_execution_plan(
                                "score-only binary correction requires Firth standard errors to be disabled",
                            ));
                        }
                        if approximate_firth_policy.is_some() {
                            return Err(invalid_execution_plan(
                                "approximate-Firth JAX policy must be null for score-only binary correction",
                            ));
                        }
                        if sparse_pseudo_budget_policy.is_some() {
                            return Err(invalid_execution_plan(
                                "field 'binary_correction_plan.approximate_firth_sparse_pseudo_budget_policy' must be null for score-only binary correction",
                            ));
                        }
                    }
                    g_plan::BinaryFallbackMethod::FirthApproximate => {
                        if approximate_firth_policy
                            != Some(&ApproximateFirthPseudoInnerPolicySchemaZero::Float32ElementwiseFloat64Reduction)
                        {
                            return Err(invalid_execution_plan(
                                "approximate-Firth binary correction requires its fixed JAX reduction policy",
                            ));
                        }
                        if sparse_pseudo_budget_policy
                            != Some(&ApproximateFirthSparsePseudoBudgetPolicySchemaZero::HalfTotalUncappedByDenseCap)
                        {
                            return Err(invalid_execution_plan(
                                "field 'binary_correction_plan.approximate_firth_sparse_pseudo_budget_policy' must equal 'half_total_uncapped_by_dense_cap' for approximate-Firth binary correction",
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl BgenFingerprintSchemaZero {
    fn validate(&self) {
        let _ = (self.content_sha256.as_ref(), self.byte_count);
    }

    fn content_fingerprint(&self) -> Option<g_genotype_contracts::BgenContentFingerprint> {
        self.content_sha256.as_ref().map(|content_sha256| g_genotype_contracts::BgenContentFingerprint {
            content_sha256: *content_sha256,
            byte_count: self.byte_count,
        })
    }
}

impl FileFingerprintSchemaZero {
    fn validate(&self, field_name: &str) -> OutputResult<()> {
        validate_non_empty_execution_plan_string(&self.path, &format!("{field_name}.path"))?;
        if self.content_hash_algorithm != FileContentHashAlgorithmSchemaZero::Sha256 {
            return Err(invalid_execution_plan(&format!(
                "field '{field_name}.content_hash_algorithm' must equal 'sha256'"
            )));
        }
        validate_execution_plan_sha256(&self.content_sha256, &format!("{field_name}.content_sha256"))?;
        let _ = (self.size, self.mtime_ns);
        Ok(())
    }
}

impl PredictionInputsSchemaZero {
    fn validate(&self) -> OutputResult<()> {
        self.prediction_list.validate("prediction_inputs.prediction_list")?;
        for loco_file in &self.loco_files {
            loco_file.validate()?;
        }
        Ok(())
    }
}

impl PredictionLocoFileFingerprintSchemaZero {
    fn validate(&self) -> OutputResult<()> {
        validate_non_empty_execution_plan_string(&self.phenotype, "prediction_inputs.loco_files.phenotype")?;
        validate_non_empty_execution_plan_string(&self.path, "prediction_inputs.loco_files.path")?;
        if self.content_hash_algorithm != FileContentHashAlgorithmSchemaZero::Sha256 {
            return Err(invalid_execution_plan(
                "field 'prediction_inputs.loco_files.content_hash_algorithm' must equal 'sha256'",
            ));
        }
        validate_execution_plan_sha256(&self.content_sha256, "prediction_inputs.loco_files.content_sha256")?;
        let _ = (self.size, self.mtime_ns);
        Ok(())
    }
}

impl KernelPlanSchemaZero {
    fn validate(&self) -> OutputResult<()> {
        self.linear.validate();
        self.binary_null.validate()?;
        self.firth.validate()?;
        self.null_firth.validate()
    }
}

impl LinearKernelPlanSchemaZero {
    fn validate(self) {
        let _ = (self.minimum_variance.get(), self.relative_variance_tolerance.get());
    }
}

impl BinaryNullKernelPlanSchemaZero {
    fn validate(&self) -> OutputResult<()> {
        validate_positive_u32(self.maximum_iterations, "binary_kernel_config.binary_null.maximum_iterations")?;
        let _ = (
            self.coefficient_tolerance.get(),
            self.nonconvergence_policy,
            self.minimum_probability.get(),
            self.minimum_variance.get(),
            self.relative_variance_tolerance.get(),
        );
        Ok(())
    }
}

impl FirthKernelPlanSchemaZero {
    fn validate(&self) -> OutputResult<()> {
        for (value, field_name) in [
            (self.batch_size, "binary_kernel_config.firth.batch_size"),
            (self.candidate_capacity, "binary_kernel_config.firth.candidate_capacity"),
            (self.maximum_iterations, "binary_kernel_config.firth.maximum_iterations"),
            (self.pseudo_maximum_iterations, "binary_kernel_config.firth.pseudo_maximum_iterations"),
            (self.pseudo_inner_maximum_iterations, "binary_kernel_config.firth.pseudo_inner_maximum_iterations"),
            (self.line_search_maximum_attempts, "binary_kernel_config.firth.line_search_maximum_attempts"),
        ] {
            validate_positive_u32(value, field_name)?;
        }
        let _ =
            (self.gradient_tolerance.get(), self.maximum_step_size.get(), self.sparse_carrier_dosage_threshold.get());
        Ok(())
    }
}

impl NullFirthKernelPlanSchemaZero {
    fn validate(&self) -> OutputResult<()> {
        for (value, field_name) in [
            (self.maximum_iterations, "binary_kernel_config.null_firth.maximum_iterations"),
            (self.fallback_iteration_multiplier, "binary_kernel_config.null_firth.fallback_iteration_multiplier"),
            (self.line_search_maximum_attempts, "binary_kernel_config.null_firth.line_search_maximum_attempts"),
        ] {
            validate_positive_u32(value, field_name)?;
        }
        let _ = (
            self.gradient_tolerance.get(),
            self.maximum_step_size.get(),
            self.fallback_step_divisor.get(),
            self.step_halving_scale.get(),
        );
        Ok(())
    }
}

impl JaxPolicySchemaZero {
    fn validate(&self) -> OutputResult<()> {
        if !self.enable_x64 {
            return Err(invalid_execution_plan("field 'jax_policy.enable_x64' must equal true"));
        }
        if self.matmul_precision != MatmulPrecisionSchemaZero::Float32 {
            return Err(invalid_execution_plan("field 'jax_policy.matmul_precision' must equal 'float32'"));
        }
        let _ = (self.device, self.approximate_firth_pseudo_inner_policy.as_ref());
        Ok(())
    }
}

impl OutputWriterSchemaZero {
    fn validate(&self) -> OutputResult<()> {
        validate_positive_u32(self.writer_thread_count, "output_writer.writer_thread_count")?;
        let writer_queue_depth = u64::try_from(crate::WRITER_QUEUE_DEPTH).map_err(|error| {
            OutputError::Runtime(format!("Writer queue depth does not fit manifest uint64: {error}"))
        })?;
        if self.writer_queue_depth != writer_queue_depth {
            return Err(invalid_execution_plan(&format!(
                "field 'output_writer.writer_queue_depth' must equal {}",
                crate::WRITER_QUEUE_DEPTH
            )));
        }
        let chunks_per_parquet_file = u64::try_from(crate::CHUNKS_PER_PARQUET_FILE).map_err(|error| {
            OutputError::Runtime(format!("Chunks per Parquet file does not fit manifest uint64: {error}"))
        })?;
        if self.chunks_per_parquet_file != chunks_per_parquet_file {
            return Err(invalid_execution_plan(&format!(
                "field 'output_writer.chunks_per_parquet_file' must equal {}",
                crate::CHUNKS_PER_PARQUET_FILE
            )));
        }
        if self.parquet_compression != ParquetCompressionSchemaZero::Zstd {
            return Err(invalid_execution_plan("field 'output_writer.parquet_compression' must equal 'zstd'"));
        }
        let expected_writer_version = crate::writer::REGENIE_STEP2_PARQUET_WRITER_VERSION.as_num();
        if self.parquet_writer_version != expected_writer_version {
            return Err(invalid_execution_plan(&format!(
                "field 'output_writer.parquet_writer_version' must equal {expected_writer_version}"
            )));
        }
        let parquet_write_batch_size =
            u64::try_from(crate::writer::REGENIE_STEP2_PARQUET_WRITE_BATCH_SIZE).map_err(|error| {
                OutputError::Runtime(format!("Parquet write batch size does not fit manifest uint64: {error}"))
            })?;
        if self.parquet_write_batch_size != parquet_write_batch_size {
            return Err(invalid_execution_plan(&format!(
                "field 'output_writer.parquet_write_batch_size' must equal {}",
                crate::writer::REGENIE_STEP2_PARQUET_WRITE_BATCH_SIZE
            )));
        }
        let parquet_max_row_group_size = u64::try_from(crate::writer::REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE)
            .map_err(|error| {
                OutputError::Runtime(format!("Parquet maximum row-group size does not fit manifest uint64: {error}"))
            })?;
        if self.parquet_max_row_group_size != parquet_max_row_group_size {
            return Err(invalid_execution_plan(&format!(
                "field 'output_writer.parquet_max_row_group_size' must equal {}",
                crate::writer::REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE
            )));
        }
        if self.parquet_float_column_encoding != ParquetFloatColumnEncodingSchemaZero::ByteStreamSplit {
            return Err(invalid_execution_plan(
                "field 'output_writer.parquet_float_column_encoding' must equal 'BYTE_STREAM_SPLIT'",
            ));
        }
        if self.result_statistic_dtype != FloatingPointDtypeSchemaZero::Float32 {
            return Err(invalid_execution_plan("field 'output_writer.result_statistic_dtype' must equal 'float32'"));
        }
        Ok(())
    }
}

fn invalid_execution_plan(message: &str) -> OutputError {
    OutputError::InvalidInput(format!("Run manifest execution plan {message}."))
}

fn validate_non_empty_execution_plan_string(value: &str, field_name: &str) -> OutputResult<()> {
    if value.is_empty() {
        return Err(invalid_execution_plan(&format!("field '{field_name}' must contain a non-empty string")));
    }
    Ok(())
}

fn validate_execution_plan_sha256(digest: &str, field_name: &str) -> OutputResult<()> {
    if !crate::digest::is_canonical_sha256(digest) {
        return Err(invalid_execution_plan(&format!(
            "field '{field_name}' must contain exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_positive_i64(value: i64, field_name: &str) -> OutputResult<()> {
    if value <= 0 {
        return Err(invalid_execution_plan(&format!("field '{field_name}' must contain a positive integer")));
    }
    Ok(())
}

fn validate_positive_u32(value: u32, field_name: &str) -> OutputResult<()> {
    if value == 0 {
        return Err(invalid_execution_plan(&format!("field '{field_name}' must contain a positive integer")));
    }
    Ok(())
}

#[cfg(test)]
fn canonical_binary_kernel_config_schema_zero_test_value() -> Value {
    json!({
        "linear": {
            "minimum_variance": 1.0e-8,
            "relative_variance_tolerance": 1.0e-6,
        },
        "binary_null": {
            "maximum_iterations": 50,
            "coefficient_tolerance": 1.0e-6,
            "nonconvergence_policy": "fail",
            "minimum_probability": 1.0e-6,
            "minimum_variance": 1.0e-8,
            "relative_variance_tolerance": 1.0e-6,
        },
        "firth": {
            "batch_size": 2,
            "candidate_capacity": 4,
            "maximum_iterations": 25,
            "gradient_tolerance": 1.0e-5,
            "maximum_step_size": 5.0,
            "pseudo_maximum_iterations": 10,
            "pseudo_inner_maximum_iterations": 5,
            "line_search_maximum_attempts": 5,
            "sparse_carrier_dosage_threshold": 0.1,
        },
        "null_firth": {
            "maximum_iterations": 50,
            "gradient_tolerance": 1.0e-5,
            "maximum_step_size": 10.0,
            "fallback_iteration_multiplier": 2,
            "fallback_step_divisor": 2.0,
            "line_search_maximum_attempts": 5,
            "step_halving_scale": 0.5,
        },
    })
}

#[cfg(test)]
pub(crate) fn canonical_execution_plan_schema_zero_test_value() -> Value {
    let fixture = json!({
        "association_mode": "regenie2_binary",
        "association_backend": {
            "kind": "jax_packed8",
            "genotype_format": "packed8",
        },
        "bgen": {
            "content_sha256": "e".repeat(64),
            "byte_count": 1024,
        },
        "sample": {
            "path": "/input/genotypes.sample",
            "size": 128,
            "mtime_ns": 20,
            "content_hash_algorithm": "sha256",
            "content_sha256": "a".repeat(64),
        },
        "phenotype_file": {
            "path": "/input/phenotypes.tsv",
            "size": 256,
            "mtime_ns": 21,
            "content_hash_algorithm": "sha256",
            "content_sha256": "b".repeat(64),
        },
        "phenotype_name": "phenotype",
        "covariate_file": null,
        "covariate_names": [],
        "prediction_inputs": {
            "prediction_list": {
                "path": "/input/predictions.list",
                "size": 64,
                "mtime_ns": 22,
                "content_hash_algorithm": "sha256",
                "content_sha256": "c".repeat(64),
            },
            "loco_files": [{
                "phenotype": "phenotype",
                "path": "/input/phenotype.loco",
                "size": 512,
                "mtime_ns": 23,
                "content_hash_algorithm": "sha256",
                "content_sha256": "d".repeat(64),
            }],
        },
        "sample_count": 100,
        "variant_count": 200,
        "chunk_size": 32,
        "binary_correction_plan": {
            "method": "firth_approximate",
            "p_threshold": 0.05,
            "firth_se": false,
            "approximate_firth_sparse_pseudo_budget_policy": "half_total_uncapped_by_dense_cap",
        },
        "binary_kernel_config": canonical_binary_kernel_config_schema_zero_test_value(),
        "jax_policy": {
            "device": "gpu",
            "enable_x64": true,
            "matmul_precision": "float32",
            "approximate_firth_pseudo_inner_policy": "float32_elementwise_float64_reduction",
        },
        "score_dtype": "float32",
        "multi_phenotype_sample_mode": "complete-case",
        "phenotype_compute_group_id": "group-id",
        "sample_set_fingerprint": "sample-fingerprint",
        "covariate_design_fingerprint": "covariate-fingerprint",
        "phenotype_design_fingerprint": "phenotype-fingerprint",
        "prediction_alignment_fingerprint": "prediction-fingerprint",
        "output_writer": {
            "writer_thread_count": 1,
            "writer_queue_depth": crate::WRITER_QUEUE_DEPTH,
            "chunks_per_parquet_file": crate::CHUNKS_PER_PARQUET_FILE,
            "parquet_compression": "zstd",
            "parquet_writer_version": crate::writer::REGENIE_STEP2_PARQUET_WRITER_VERSION.as_num(),
            "parquet_write_batch_size": crate::writer::REGENIE_STEP2_PARQUET_WRITE_BATCH_SIZE,
            "parquet_max_row_group_size": crate::writer::REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE,
            "parquet_float_column_encoding": "BYTE_STREAM_SPLIT",
            "result_statistic_dtype": "float32",
        },
        "resume_policy": "lineage_receipts_exact_coverage",
    });
    ExecutionPlanSchemaZero::from_value(fixture)
        .expect("canonical schema-zero execution-plan fixture is valid")
        .to_value()
        .expect("canonical schema-zero execution-plan fixture serializes")
}

#[cfg(test)]
mod tests {
    use super::{ExecutionPlanSchemaZero, canonical_execution_plan_schema_zero_test_value};

    #[test]
    fn schema_zero_required_null_and_nullable_fields_must_be_present() {
        for field_path in [
            &["covariate_file"][..],
            &["binary_kernel_config"][..],
            &["jax_policy", "approximate_firth_pseudo_inner_policy"][..],
            &["binary_correction_plan", "approximate_firth_sparse_pseudo_budget_policy"][..],
            &["bgen", "content_sha256"][..],
            &["bgen", "byte_count"][..],
        ] {
            let mut fixture = canonical_execution_plan_schema_zero_test_value();
            let (field_name, parent_path) = field_path.split_last().expect("field path is non-empty");
            let mut parent = &mut fixture;
            for component in parent_path {
                parent = parent.get_mut(*component).expect("fixture parent field exists");
            }
            parent.as_object_mut().expect("fixture parent is an object").remove(*field_name);
            assert!(
                ExecutionPlanSchemaZero::from_value(fixture).is_err(),
                "missing required field {field_path:?} is rejected"
            );
        }
    }

    #[test]
    fn schema_zero_bgen_fingerprint_is_an_exact_strict_two_key_object() {
        let canonical = canonical_execution_plan_schema_zero_test_value();
        let bgen = canonical["bgen"].as_object().expect("canonical BGEN fingerprint is an object");
        assert_eq!(bgen.len(), 2);
        assert!(bgen.contains_key("content_sha256"));
        assert!(bgen.contains_key("byte_count"));

        let mut unknown_field = canonical.clone();
        unknown_field["bgen"]["locator"] = serde_json::Value::String("input.bgen".to_string());
        assert!(ExecutionPlanSchemaZero::from_value(unknown_field).is_err());

        for invalid_digest in [
            serde_json::Value::Bool(false),
            serde_json::Value::Number(1_u64.into()),
            serde_json::Value::String("E".repeat(64)),
            serde_json::Value::String("e".repeat(63)),
        ] {
            let mut fixture = canonical.clone();
            fixture["bgen"]["content_sha256"] = invalid_digest;
            assert!(ExecutionPlanSchemaZero::from_value(fixture).is_err());
        }

        for invalid_byte_count in [
            serde_json::Value::Bool(false),
            serde_json::Value::String("1024".to_string()),
            serde_json::json!(1024.0),
            serde_json::json!(-1),
        ] {
            let mut fixture = canonical.clone();
            fixture["bgen"]["byte_count"] = invalid_byte_count;
            assert!(ExecutionPlanSchemaZero::from_value(fixture).is_err());
        }

        let mut unattested = canonical;
        unattested["bgen"]["content_sha256"] = serde_json::Value::Null;
        ExecutionPlanSchemaZero::from_value(unattested).expect("explicit null content evidence is schema-valid");
    }

    #[test]
    fn schema_zero_rejects_cpu_packed_genotypes_and_unknown_kernel_fields() {
        let mut cpu_packed = canonical_execution_plan_schema_zero_test_value();
        cpu_packed["jax_policy"]["device"] = serde_json::Value::String("cpu".to_string());
        assert!(ExecutionPlanSchemaZero::from_value(cpu_packed).is_err());

        let mut unknown_kernel_field = canonical_execution_plan_schema_zero_test_value();
        unknown_kernel_field["binary_kernel_config"]["firth"]["unknown"] = serde_json::Value::Null;
        assert!(ExecutionPlanSchemaZero::from_value(unknown_kernel_field).is_err());
    }

    #[test]
    fn schema_zero_sparse_pseudo_budget_policy_is_exact_and_mode_dependent() {
        const POLICY_FIELD: &str = "approximate_firth_sparse_pseudo_budget_policy";

        let canonical = canonical_execution_plan_schema_zero_test_value();
        assert_eq!(
            canonical["binary_correction_plan"][POLICY_FIELD],
            serde_json::Value::String("half_total_uncapped_by_dense_cap".to_string())
        );

        for invalid_policy in [
            serde_json::Value::String("dense_cap_applies_to_all_lanes".to_string()),
            serde_json::Value::String("unknown_policy".to_string()),
            serde_json::Value::Bool(false),
            serde_json::json!({"policy": "half_total_uncapped_by_dense_cap"}),
            serde_json::Value::Null,
        ] {
            let mut fixture = canonical.clone();
            fixture["binary_correction_plan"][POLICY_FIELD] = invalid_policy;
            assert!(
                ExecutionPlanSchemaZero::from_value(fixture).is_err(),
                "approximate Firth rejects a null, unknown, or wrongly typed sparse-budget policy"
            );
        }
        let mut unknown_field = canonical.clone();
        unknown_field["binary_correction_plan"]["sparse_pseudo_budget_policy"] = serde_json::Value::Null;
        assert!(ExecutionPlanSchemaZero::from_value(unknown_field).is_err());

        let mut score_only = canonical.clone();
        score_only["binary_correction_plan"]["method"] = serde_json::Value::String("score_only".to_string());
        score_only["binary_correction_plan"][POLICY_FIELD] = serde_json::Value::Null;
        score_only["jax_policy"]["approximate_firth_pseudo_inner_policy"] = serde_json::Value::Null;
        ExecutionPlanSchemaZero::from_value(score_only.clone())
            .expect("score-only binary correction requires an explicit null sparse-budget policy");
        score_only["binary_correction_plan"][POLICY_FIELD] =
            serde_json::Value::String("half_total_uncapped_by_dense_cap".to_string());
        assert!(ExecutionPlanSchemaZero::from_value(score_only).is_err());

        let mut linear = canonical;
        linear["association_mode"] = serde_json::Value::String("regenie2_linear".to_string());
        linear["association_backend"]["kind"] = serde_json::Value::String("jax_dosage".to_string());
        linear["association_backend"]["genotype_format"] = serde_json::Value::String("dosage".to_string());
        linear["binary_correction_plan"]["method"] = serde_json::Value::String("score_only".to_string());
        linear["binary_correction_plan"][POLICY_FIELD] = serde_json::Value::Null;
        linear["binary_kernel_config"] = serde_json::Value::Null;
        linear["jax_policy"]["approximate_firth_pseudo_inner_policy"] = serde_json::Value::Null;
        ExecutionPlanSchemaZero::from_value(linear)
            .expect("linear association requires an explicit null sparse-budget policy");
    }
}
