use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::error::{OutputError, OutputResult};

use super::schema_zero::{
    ApproximateFirthPseudoInnerPolicySchemaZero, AssociationBackendKindSchemaZero, AssociationBackendSchemaZero,
    BgenFingerprintSchemaZero, BinaryCorrectionPlanSchemaZero, ExecutionPlanSchemaZero,
    FileContentHashAlgorithmSchemaZero, FileFingerprintSchemaZero, FloatingPointDtypeSchemaZero, JaxPolicySchemaZero,
    KernelPlanSchemaZero, MatmulPrecisionSchemaZero, OutputWriterSchemaZero, ParquetCompressionSchemaZero,
    ParquetFloatColumnEncodingSchemaZero, PredictionInputsSchemaZero, PredictionLocoFileFingerprintSchemaZero,
    RequiredNullableSchemaZero, ResumePolicySchemaZero,
};
use super::{
    ManifestFileFingerprintCache, OUTPUT_SCHEMA_VERSION, RUN_MANIFEST_SCHEMA_VERSION, build_manifest_value_sha256,
};

pub struct CurrentRunManifestHeaderInput {
    pub phenotype_name: String,
    pub bgen_content_evidence: Arc<g_genotype_contracts::BgenContentEvidence>,
    pub covariate_names: Arc<[String]>,
    pub prediction_loco_files: Arc<[PredictionLocoFileFingerprint]>,
    pub sample_count: usize,
    pub variant_count: usize,
    pub resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat,
    pub sample_mode: g_plan::MultiPhenotypeSampleMode,
    pub phenotype_compute_group_id: Arc<str>,
    pub sample_set_fingerprint: Arc<str>,
    pub covariate_design_fingerprint: Arc<str>,
    pub phenotype_design_fingerprint: Arc<str>,
    pub prediction_alignment_fingerprint: Arc<str>,
}

#[derive(Clone)]
pub struct PredictionLocoFileFingerprint {
    pub(crate) phenotype_name: Arc<str>,
    pub(crate) file_fingerprint: Arc<super::ManifestFileFingerprint>,
}

struct ExecutionPlanFileFingerprints {
    bgen: BgenFingerprintSchemaZero,
    sample: FileFingerprintSchemaZero,
    phenotype_file: FileFingerprintSchemaZero,
    covariate_file: Option<FileFingerprintSchemaZero>,
    prediction_list: FileFingerprintSchemaZero,
    prediction_loco_files: Vec<PredictionLocoFileFingerprintSchemaZero>,
}

pub(crate) fn build_current_run_manifest_header_value_with_cache(
    run_plan: &g_plan::RunPlan,
    input: &CurrentRunManifestHeaderInput,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<Value, OutputError> {
    let execution_plan = build_execution_plan_schema_zero_with_cache(run_plan, input, fingerprint_cache)?;
    let execution_plan_value = execution_plan.to_value()?;
    let execution_plan_hash = build_manifest_value_sha256(&execution_plan_value)?;
    Ok(json!({
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "execution_plan": execution_plan_value,
        "execution_plan_hash": execution_plan_hash,
    }))
}

pub(crate) fn build_execution_plan_schema_zero_with_cache(
    run_plan: &g_plan::RunPlan,
    input: &CurrentRunManifestHeaderInput,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> OutputResult<ExecutionPlanSchemaZero> {
    let fingerprints = build_execution_plan_file_fingerprints_with_cache(run_plan, input, fingerprint_cache)?;
    let sample_count = i64::try_from(input.sample_count)
        .map_err(|_| OutputError::InvalidInput("Sample count does not fit manifest int64.".to_string()))?;
    let variant_count = i64::try_from(input.variant_count)
        .map_err(|_| OutputError::InvalidInput("Variant count does not fit manifest int64.".to_string()))?;
    let execution_plan = ExecutionPlanSchemaZero {
        association_mode: run_plan.association_mode,
        association_backend: build_association_backend(input.resolved_gpu_genotype_format),
        bgen: fingerprints.bgen,
        sample: fingerprints.sample,
        phenotype_file: fingerprints.phenotype_file,
        phenotype_name: input.phenotype_name.clone(),
        covariate_file: RequiredNullableSchemaZero::new(fingerprints.covariate_file),
        covariate_names: input.covariate_names.to_vec(),
        prediction_inputs: PredictionInputsSchemaZero {
            prediction_list: fingerprints.prediction_list,
            loco_files: fingerprints.prediction_loco_files,
        },
        sample_count,
        variant_count,
        chunk_size: run_plan.chunk_size,
        binary_correction_plan: BinaryCorrectionPlanSchemaZero {
            method: run_plan.correction.method,
            p_threshold: run_plan.correction.p_threshold,
            firth_se: run_plan.correction.firth_se,
        },
        binary_kernel_config: build_binary_kernel_config(run_plan),
        jax_policy: build_jax_policy(run_plan),
        score_dtype: FloatingPointDtypeSchemaZero::Float32,
        multi_phenotype_sample_mode: input.sample_mode,
        phenotype_compute_group_id: input.phenotype_compute_group_id.to_string(),
        sample_set_fingerprint: input.sample_set_fingerprint.to_string(),
        covariate_design_fingerprint: input.covariate_design_fingerprint.to_string(),
        phenotype_design_fingerprint: input.phenotype_design_fingerprint.to_string(),
        prediction_alignment_fingerprint: input.prediction_alignment_fingerprint.to_string(),
        output_writer: build_output_writer(run_plan)?,
        resume_policy: ResumePolicySchemaZero::LineageReceiptsExactCoverage,
    };
    execution_plan.validate()?;
    Ok(execution_plan)
}

fn build_execution_plan_file_fingerprints_with_cache(
    run_plan: &g_plan::RunPlan,
    input: &CurrentRunManifestHeaderInput,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> OutputResult<ExecutionPlanFileFingerprints> {
    Ok(ExecutionPlanFileFingerprints {
        bgen: bgen_content_evidence_to_schema_zero(&input.bgen_content_evidence),
        sample: build_required_file_fingerprint_with_cache(
            fingerprint_cache,
            Path::new(&run_plan.input.sample_path),
            "sample file",
        )?,
        phenotype_file: build_required_file_fingerprint_with_cache(
            fingerprint_cache,
            Path::new(&run_plan.input.phenotype_path),
            "phenotype file",
        )?,
        covariate_file: build_optional_file_fingerprint_with_cache(
            fingerprint_cache,
            run_plan.input.covariate_path.as_deref().map(Path::new),
        )?,
        prediction_list: build_required_file_fingerprint_with_cache(
            fingerprint_cache,
            Path::new(&run_plan.input.prediction_list_path),
            "prediction list",
        )?,
        prediction_loco_files: prediction_loco_file_fingerprints_to_schema_zero(&input.prediction_loco_files)?,
    })
}

fn build_association_backend(genotype_format: g_plan::GpuGenotypeFormat) -> AssociationBackendSchemaZero {
    AssociationBackendSchemaZero {
        kind: match genotype_format {
            g_plan::GpuGenotypeFormat::Dosage => AssociationBackendKindSchemaZero::JaxDosage,
            g_plan::GpuGenotypeFormat::Packed8 => AssociationBackendKindSchemaZero::JaxPacked8,
        },
        genotype_format,
    }
}

fn build_binary_kernel_config(run_plan: &g_plan::RunPlan) -> RequiredNullableSchemaZero<KernelPlanSchemaZero> {
    RequiredNullableSchemaZero::new(
        (run_plan.association_mode == g_plan::AssociationMode::Regenie2Binary)
            .then(|| KernelPlanSchemaZero::from(&run_plan.compute.kernels)),
    )
}

fn build_jax_policy(run_plan: &g_plan::RunPlan) -> JaxPolicySchemaZero {
    let approximate_firth_policy = (run_plan.association_mode == g_plan::AssociationMode::Regenie2Binary
        && run_plan.correction.method == g_plan::BinaryFallbackMethod::FirthApproximate)
        .then_some(ApproximateFirthPseudoInnerPolicySchemaZero::Float32ElementwiseFloat64Reduction);
    JaxPolicySchemaZero {
        device: run_plan.compute.device,
        enable_x64: true,
        matmul_precision: MatmulPrecisionSchemaZero::Float32,
        approximate_firth_pseudo_inner_policy: RequiredNullableSchemaZero::new(approximate_firth_policy),
    }
}

fn build_output_writer(run_plan: &g_plan::RunPlan) -> OutputResult<OutputWriterSchemaZero> {
    Ok(OutputWriterSchemaZero {
        writer_thread_count: run_plan.output.writer_thread_count,
        writer_queue_depth: usize_to_manifest_u64(crate::WRITER_QUEUE_DEPTH, "Writer queue depth")?,
        chunks_per_parquet_file: usize_to_manifest_u64(crate::CHUNKS_PER_PARQUET_FILE, "Chunks per Parquet file")?,
        parquet_compression: ParquetCompressionSchemaZero::Zstd,
        parquet_writer_version: crate::writer::REGENIE_STEP2_PARQUET_WRITER_VERSION.as_num(),
        parquet_write_batch_size: usize_to_manifest_u64(
            crate::writer::REGENIE_STEP2_PARQUET_WRITE_BATCH_SIZE,
            "Parquet write batch size",
        )?,
        parquet_max_row_group_size: usize_to_manifest_u64(
            crate::writer::REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE,
            "Parquet maximum row-group size",
        )?,
        parquet_float_column_encoding: ParquetFloatColumnEncodingSchemaZero::ByteStreamSplit,
        result_statistic_dtype: FloatingPointDtypeSchemaZero::Float32,
    })
}

fn usize_to_manifest_u64(value: usize, field_name: &str) -> OutputResult<u64> {
    u64::try_from(value)
        .map_err(|error| OutputError::Runtime(format!("{field_name} does not fit manifest uint64: {error}")))
}

fn bgen_content_evidence_to_schema_zero(
    evidence: &g_genotype_contracts::BgenContentEvidence,
) -> BgenFingerprintSchemaZero {
    match evidence {
        g_genotype_contracts::BgenContentEvidence::OwnedSnapshot(fingerprint) => BgenFingerprintSchemaZero {
            content_sha256: RequiredNullableSchemaZero::new(Some(fingerprint.content_sha256)),
            byte_count: fingerprint.byte_count,
        },
        g_genotype_contracts::BgenContentEvidence::PositionedUnattested(identity) => BgenFingerprintSchemaZero {
            content_sha256: RequiredNullableSchemaZero::new(None),
            byte_count: identity.file_size,
        },
    }
}

fn prediction_loco_file_fingerprints_to_schema_zero(
    fingerprints: &[PredictionLocoFileFingerprint],
) -> OutputResult<Vec<PredictionLocoFileFingerprintSchemaZero>> {
    fingerprints
        .iter()
        .map(|fingerprint| {
            let content_sha256 = fingerprint.file_fingerprint.content_sha256.clone().ok_or_else(|| {
                OutputError::Runtime("LOCO prediction file fingerprint must include a content hash.".to_string())
            })?;
            validate_constructed_file_hash_algorithm(
                &fingerprint.file_fingerprint.content_hash_algorithm,
                "LOCO prediction file",
            )?;
            Ok(PredictionLocoFileFingerprintSchemaZero {
                phenotype: fingerprint.phenotype_name.to_string(),
                path: fingerprint.file_fingerprint.path.clone(),
                size: fingerprint.file_fingerprint.size,
                mtime_ns: fingerprint.file_fingerprint.mtime_ns,
                content_hash_algorithm: FileContentHashAlgorithmSchemaZero::Sha256,
                content_sha256,
            })
        })
        .collect()
}

fn build_required_file_fingerprint_with_cache(
    fingerprint_cache: &mut ManifestFileFingerprintCache,
    path: &Path,
    role_name: &str,
) -> OutputResult<FileFingerprintSchemaZero> {
    build_optional_file_fingerprint_with_cache(fingerprint_cache, Some(path))?
        .ok_or_else(|| OutputError::InvalidInput(format!("{role_name} fingerprint is required.")))
}

fn build_optional_file_fingerprint_with_cache(
    fingerprint_cache: &mut ManifestFileFingerprintCache,
    path: Option<&Path>,
) -> OutputResult<Option<FileFingerprintSchemaZero>> {
    let Some(file_path) = path else {
        return Ok(None);
    };
    let fingerprint = fingerprint_cache.build_file_fingerprint(file_path, true)?;
    validate_constructed_file_hash_algorithm(&fingerprint.content_hash_algorithm, "input file")?;
    let content_sha256 = fingerprint.content_sha256.clone().ok_or_else(|| {
        OutputError::Runtime("Hashed input file fingerprint must include a content hash.".to_string())
    })?;
    Ok(Some(FileFingerprintSchemaZero {
        path: fingerprint.path.clone(),
        size: fingerprint.size,
        mtime_ns: fingerprint.mtime_ns,
        content_hash_algorithm: FileContentHashAlgorithmSchemaZero::Sha256,
        content_sha256,
    }))
}

fn validate_constructed_file_hash_algorithm(content_hash_algorithm: &str, role_name: &str) -> OutputResult<()> {
    if content_hash_algorithm != "sha256" {
        return Err(OutputError::Runtime(format!(
            "{role_name} fingerprint must use the SHA-256 content hash algorithm."
        )));
    }
    Ok(())
}
