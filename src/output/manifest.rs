use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::output::resume;
use crate::output::writer::{OutputFileFormat, OutputWriterError};

const RUN_MANIFEST_FILE_NAME: &str = "run_manifest.json";
const RUN_MANIFEST_SCHEMA_VERSION: i64 = 8;
const OUTPUT_SCHEMA_VERSION: i64 = 2;
const JAX_MATMUL_PRECISION_WHEN_UNSET: &str = "float32";
const RESUME_POLICY: &str = "manifest_committed_chunks";
const FILE_FINGERPRINT_CONTENT_HASH_ALGORITHM: &str = "sha256";
const FILE_FINGERPRINT_METADATA_ONLY: &str = "metadata-only";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OutputRunPaths {
    pub(crate) run_directory: PathBuf,
    pub(crate) chunks_directory: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedOutputRun {
    pub(crate) output_run_paths: OutputRunPaths,
    pub(crate) existing_manifest_json: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InitializedOutputRun {
    pub(crate) committed_chunk_identifiers: Vec<i64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OutputResumeMode {
    Fast,
    Strict,
}

impl OutputResumeMode {
    pub(crate) fn parse(resume_mode: &str) -> Result<Self, OutputWriterError> {
        match resume_mode {
            "fast" => Ok(Self::Fast),
            "strict" => Ok(Self::Strict),
            unsupported_resume_mode => Err(OutputWriterError::InvalidInput(format!(
                "Resume mode must be 'fast' or 'strict', observed '{unsupported_resume_mode}'."
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RunManifestChunkCommit {
    pub(crate) chunk_identifier: i64,
    pub(crate) output_format: String,
    pub(crate) compression: String,
    pub(crate) variant_start_index: i64,
    pub(crate) variant_stop_index: i64,
    pub(crate) row_count: usize,
    pub(crate) chunk_file_name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ManifestFileFingerprint {
    pub(crate) path: String,
    pub(crate) size: u64,
    pub(crate) mtime_ns: i64,
    pub(crate) content_hash_algorithm: String,
    pub(crate) content_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CurrentRunManifestHeaderInput {
    pub(crate) association_mode: String,
    pub(crate) association_backend_kind: String,
    pub(crate) bgen_path: PathBuf,
    pub(crate) sample_path: Option<PathBuf>,
    pub(crate) phenotype_path: PathBuf,
    pub(crate) phenotype_name: String,
    pub(crate) covariate_path: Option<PathBuf>,
    pub(crate) covariate_names: Vec<String>,
    pub(crate) prediction_list_path: PathBuf,
    pub(crate) sample_count: i64,
    pub(crate) variant_count: i64,
    pub(crate) chunk_size: i64,
    pub(crate) variant_limit: Option<i64>,
    pub(crate) binary_correction_plan_method: String,
    pub(crate) binary_correction_plan_p_threshold: f64,
    pub(crate) binary_correction_plan_firth_se: bool,
    pub(crate) trusted_no_missing_diploid: bool,
    pub(crate) sample_key_mode: String,
    pub(crate) binary_kernel_config_json: Option<String>,
    pub(crate) bgen_decode_tile_variant_count: i64,
    pub(crate) trusted_bgen_validation_mode: String,
    pub(crate) jax_device: String,
    pub(crate) jax_enable_x64: bool,
    pub(crate) jax_matmul_precision: Option<String>,
    pub(crate) gpu_genotype_format: String,
    pub(crate) score_dtype: String,
    pub(crate) firth_dtype: String,
    pub(crate) multi_phenotype_sample_mode: String,
    pub(crate) phenotype_compute_group_id: Option<String>,
    pub(crate) sample_set_fingerprint: Option<String>,
    pub(crate) covariate_design_fingerprint: Option<String>,
    pub(crate) prediction_alignment_fingerprint: Option<String>,
    pub(crate) output_format: String,
    pub(crate) finalize_parquet: bool,
    pub(crate) writer_thread_count: i64,
    pub(crate) writer_queue_depth: i64,
    pub(crate) chunks_per_arrow_file: i64,
    pub(crate) arrow_compression: String,
    pub(crate) parquet_compression: String,
    pub(crate) output_statistic_dtype: String,
}

pub(crate) fn build_current_run_manifest_header_json(
    input: CurrentRunManifestHeaderInput,
) -> Result<String, OutputWriterError> {
    let bgen_fingerprint = build_required_file_fingerprint(&input.bgen_path, false, "BGEN")?;
    let sample_fingerprint = build_optional_file_fingerprint(input.sample_path.as_deref(), true)?;
    let phenotype_file_fingerprint = build_required_file_fingerprint(&input.phenotype_path, true, "phenotype file")?;
    let covariate_file_fingerprint = build_optional_file_fingerprint(input.covariate_path.as_deref(), true)?;
    let prediction_list_fingerprint =
        build_required_file_fingerprint(&input.prediction_list_path, true, "prediction list")?;
    let binary_correction_plan = json!({
        "method": input.binary_correction_plan_method,
        "p_threshold": input.binary_correction_plan_p_threshold,
        "firth_se": input.binary_correction_plan_firth_se,
    });
    let binary_kernel_config = match input.binary_kernel_config_json {
        Some(binary_kernel_config_json) => serde_json::from_str::<Value>(&binary_kernel_config_json)
            .map_err(|error| OutputWriterError::InvalidInput(error.to_string()))?,
        None => Value::Null,
    };
    let association_backend = json!({
        "kind": input.association_backend_kind,
        "association_mode": input.association_mode,
        "device": input.jax_device,
        "genotype_format": input.gpu_genotype_format,
    });
    let jax_policy = json!({
        "device": input.jax_device,
        "enable_x64": input.jax_enable_x64,
        "matmul_precision": input.jax_matmul_precision.unwrap_or_else(|| JAX_MATMUL_PRECISION_WHEN_UNSET.to_string()),
    });
    let output_writer = json!({
        "output_format": input.output_format,
        "finalize_parquet": input.finalize_parquet,
        "writer_thread_count": input.writer_thread_count,
        "writer_queue_depth": input.writer_queue_depth,
        "chunks_per_arrow_file": input.chunks_per_arrow_file,
        "arrow_compression": input.arrow_compression,
        "parquet_compression": input.parquet_compression,
        "result_statistic_dtype": input.output_statistic_dtype,
    });
    let execution_plan = json!({
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "association_mode": input.association_mode,
        "association_backend": association_backend,
        "bgen": bgen_fingerprint,
        "sample": sample_fingerprint,
        "phenotype_file": phenotype_file_fingerprint,
        "phenotype_name": input.phenotype_name,
        "covariate_file": covariate_file_fingerprint,
        "covariate_names": input.covariate_names,
        "prediction_list": prediction_list_fingerprint,
        "sample_count": input.sample_count,
        "variant_count": input.variant_count,
        "chunk_size": input.chunk_size,
        "variant_limit": input.variant_limit,
        "binary_correction_plan": binary_correction_plan,
        "binary_kernel_config": binary_kernel_config,
        "trusted_no_missing_diploid": input.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": input.trusted_bgen_validation_mode,
        "sample_key_mode": input.sample_key_mode,
        "bgen_decode_tile_variant_count": input.bgen_decode_tile_variant_count,
        "jax_policy": jax_policy,
        "gpu_genotype_format": input.gpu_genotype_format,
        "score_dtype": input.score_dtype,
        "firth_dtype": input.firth_dtype,
        "multi_phenotype_sample_mode": input.multi_phenotype_sample_mode,
        "phenotype_compute_group_id": input.phenotype_compute_group_id,
        "sample_set_fingerprint": input.sample_set_fingerprint,
        "covariate_design_fingerprint": input.covariate_design_fingerprint,
        "prediction_alignment_fingerprint": input.prediction_alignment_fingerprint,
        "output_writer": output_writer,
        "resume_policy": RESUME_POLICY,
    });
    let execution_plan_hash = build_manifest_value_sha256(&execution_plan)?;
    let current_header = json!({
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "association_mode": input.association_mode,
        "association_backend": execution_plan["association_backend"].clone(),
        "bgen": execution_plan["bgen"].clone(),
        "sample": execution_plan["sample"].clone(),
        "phenotype_file": execution_plan["phenotype_file"].clone(),
        "phenotype_name": input.phenotype_name,
        "covariate_file": execution_plan["covariate_file"].clone(),
        "covariate_names": execution_plan["covariate_names"].clone(),
        "prediction_list": execution_plan["prediction_list"].clone(),
        "sample_count": input.sample_count,
        "variant_count": input.variant_count,
        "chunk_size": input.chunk_size,
        "variant_limit": input.variant_limit,
        "binary_correction_plan": execution_plan["binary_correction_plan"].clone(),
        "binary_kernel_config": execution_plan["binary_kernel_config"].clone(),
        "trusted_no_missing_diploid": input.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": input.trusted_bgen_validation_mode,
        "sample_key_mode": input.sample_key_mode,
        "bgen_decode_tile_variant_count": input.bgen_decode_tile_variant_count,
        "jax_policy": execution_plan["jax_policy"].clone(),
        "gpu_genotype_format": input.gpu_genotype_format,
        "score_dtype": input.score_dtype,
        "firth_dtype": input.firth_dtype,
        "multi_phenotype_sample_mode": input.multi_phenotype_sample_mode,
        "phenotype_compute_group_id": input.phenotype_compute_group_id,
        "sample_set_fingerprint": input.sample_set_fingerprint,
        "covariate_design_fingerprint": input.covariate_design_fingerprint,
        "prediction_alignment_fingerprint": input.prediction_alignment_fingerprint,
        "output_writer": execution_plan["output_writer"].clone(),
        "resume_policy": RESUME_POLICY,
        "execution_plan": execution_plan,
        "execution_plan_hash": execution_plan_hash,
    });
    serde_json::to_string(&current_header).map_err(OutputWriterError::runtime)
}

pub(crate) fn resolve_output_run_paths(
    output_root: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> OutputRunPaths {
    let run_directory = if output_root.extension().is_some_and(|extension| extension == "run") {
        output_root.to_path_buf()
    } else {
        PathBuf::from(format!("{}.{}.run", output_root.display(), association_mode))
    };
    let output_directory_name = match output_format {
        OutputFileFormat::Arrow => "chunks",
        OutputFileFormat::Parquet => "parts",
        OutputFileFormat::Regenie => "regenie",
    };
    OutputRunPaths { chunks_directory: run_directory.join(output_directory_name), run_directory }
}

pub(crate) fn prepare_output_run(
    output_root: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
    resume: bool,
) -> Result<PreparedOutputRun, OutputWriterError> {
    let output_run_paths = resolve_output_run_paths(output_root, association_mode, output_format);
    if !resume && directory_exists_and_is_non_empty(&output_run_paths.run_directory)? {
        return Err(OutputWriterError::InvalidInput(format!(
            "Output run directory '{}' already exists and is not empty. Use --resume or choose a new output path.",
            output_run_paths.run_directory.display()
        )));
    }
    std::fs::create_dir_all(&output_run_paths.chunks_directory).map_err(OutputWriterError::runtime)?;
    let existing_manifest_json = load_run_manifest_json(&output_run_paths.run_directory)?;
    if resume && existing_manifest_json.is_none() {
        return Err(OutputWriterError::InvalidInput("Resume requires run_manifest.json.".to_string()));
    }
    Ok(PreparedOutputRun { output_run_paths, existing_manifest_json })
}

pub(crate) fn load_run_manifest_json(run_directory: &Path) -> Result<Option<String>, OutputWriterError> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest_json = std::fs::read_to_string(&manifest_path).map_err(OutputWriterError::runtime)?;
    parse_run_manifest_text(&manifest_json, Some(&manifest_path))?;
    Ok(Some(manifest_json))
}

pub(crate) fn write_run_manifest_json(run_directory: &Path, manifest_json: &str) -> Result<(), OutputWriterError> {
    let manifest = parse_run_manifest_text(manifest_json, None)?;
    write_run_manifest_value(run_directory, &manifest).map_err(OutputWriterError::runtime)
}

pub(crate) fn validate_run_manifest_compatibility(
    manifest_json: &str,
    current_header_json: &str,
) -> Result<(), OutputWriterError> {
    let manifest = parse_run_manifest_text(manifest_json, None)?;
    let current_header = parse_current_header_text(current_header_json)?;
    validate_manifest_compatibility_values(&manifest, &current_header)
}

pub(crate) fn read_run_manifest_committed_chunk_identifiers_from_text(
    manifest_json: &str,
) -> Result<Vec<i64>, OutputWriterError> {
    let manifest = parse_run_manifest_text(manifest_json, None)?;
    read_run_manifest_committed_chunk_identifiers(&manifest)
}

pub(crate) fn initialize_output_run(
    run_directory: &Path,
    chunks_directory: &Path,
    existing_manifest_json: Option<&str>,
    current_header_json: &str,
    resume: bool,
    resume_mode: OutputResumeMode,
) -> Result<InitializedOutputRun, OutputWriterError> {
    let current_header = parse_current_header_text(current_header_json)?;
    let (mut manifest, committed_chunks, committed_chunk_identifiers) = if let Some(existing_manifest_text) =
        existing_manifest_json
    {
        let existing_manifest = parse_run_manifest_text(existing_manifest_text, None)?;
        validate_manifest_compatibility_values(&existing_manifest, &current_header)?;
        let manifest_committed_chunks = read_run_manifest_committed_chunks(&existing_manifest)?;
        let _validated_committed_chunk_identifiers = read_run_manifest_committed_chunk_identifiers(&existing_manifest)?;
        if resume {
            match resume_mode {
                OutputResumeMode::Fast => {
                    let committed_chunk_identifiers =
                        read_run_manifest_committed_chunk_identifiers(&existing_manifest)?;
                    (existing_manifest, manifest_committed_chunks, committed_chunk_identifiers)
                }
                OutputResumeMode::Strict => {
                    let repaired_commits =
                        resume::repair_strict_manifest_chunk_commits(chunks_directory, existing_manifest_text)?;
                    let committed_chunk_identifiers =
                        repaired_commits.iter().map(|chunk_commit| chunk_commit.chunk_identifier).collect();
                    let committed_chunks = repaired_commits.iter().map(chunk_commit_to_value).collect::<Vec<_>>();
                    (existing_manifest, committed_chunks, committed_chunk_identifiers)
                }
            }
        } else {
            (existing_manifest, manifest_committed_chunks, Vec::new())
        }
    } else {
        if resume {
            return Err(OutputWriterError::InvalidInput("Resume requires run_manifest.json.".to_string()));
        }
        let manifest = load_run_manifest_value(run_directory)?.unwrap_or_else(|| Value::Object(Map::new()));
        (manifest, Vec::new(), Vec::new())
    };
    merge_manifest_header(&mut manifest, &current_header)?;
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| OutputWriterError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
    manifest_object.insert("committed_chunks".to_string(), Value::Array(committed_chunks));
    manifest_object.entry("finalized".to_string()).or_insert(Value::Bool(false));
    write_run_manifest_value(run_directory, &manifest).map_err(OutputWriterError::runtime)?;
    Ok(InitializedOutputRun { committed_chunk_identifiers })
}

pub(crate) fn read_run_manifest_chunk_commits(run_directory: &Path) -> Result<Vec<RunManifestChunkCommit>, String> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    let manifest_text = std::fs::read_to_string(&manifest_path).map_err(|error| error.to_string())?;
    read_run_manifest_chunk_commits_from_text(&manifest_text)
}

pub(crate) fn read_run_manifest_chunk_commits_from_text(
    manifest_json: &str,
) -> Result<Vec<RunManifestChunkCommit>, String> {
    let manifest = serde_json::from_str::<Value>(manifest_json).map_err(|error| error.to_string())?;
    let committed_chunks = manifest
        .get("committed_chunks")
        .and_then(Value::as_array)
        .ok_or_else(|| "Run manifest committed_chunks field must be a list.".to_string())?;
    let mut committed_chunks_by_identifier = BTreeMap::new();
    for committed_chunk in committed_chunks {
        insert_or_validate_chunk_commit(
            &mut committed_chunks_by_identifier,
            read_run_manifest_chunk_commit(committed_chunk)?,
        )?;
    }
    Ok(committed_chunks_by_identifier.into_values().collect())
}

fn build_required_file_fingerprint(
    path: &Path,
    include_content_hash: bool,
    role_name: &str,
) -> Result<Value, OutputWriterError> {
    build_optional_file_fingerprint(Some(path), include_content_hash)?
        .ok_or_else(|| OutputWriterError::InvalidInput(format!("{role_name} fingerprint is required.")))
}

fn build_optional_file_fingerprint(
    path: Option<&Path>,
    include_content_hash: bool,
) -> Result<Option<Value>, OutputWriterError> {
    let Some(file_path) = path else {
        return Ok(None);
    };
    build_manifest_file_fingerprint(file_path, include_content_hash)
        .map(|fingerprint| Some(manifest_file_fingerprint_to_value(&fingerprint)))
}

pub(crate) fn build_manifest_file_fingerprint(
    file_path: &Path,
    include_content_hash: bool,
) -> Result<ManifestFileFingerprint, OutputWriterError> {
    let metadata = file_path.metadata().map_err(OutputWriterError::runtime)?;
    let content_hash_algorithm =
        if include_content_hash { FILE_FINGERPRINT_CONTENT_HASH_ALGORITHM } else { FILE_FINGERPRINT_METADATA_ONLY };
    let content_sha256 = if include_content_hash { Some(build_file_content_sha256(file_path)?) } else { None };
    let mtime_ns = metadata
        .mtime()
        .checked_mul(1_000_000_000)
        .and_then(|mtime_seconds_ns| mtime_seconds_ns.checked_add(metadata.mtime_nsec()))
        .ok_or_else(|| OutputWriterError::Runtime("File modification timestamp overflowed nanoseconds.".to_string()))?;
    let resolved_path = file_path.canonicalize().map_err(OutputWriterError::runtime)?;
    Ok(ManifestFileFingerprint {
        path: resolved_path.display().to_string(),
        size: metadata.len(),
        mtime_ns,
        content_hash_algorithm: content_hash_algorithm.to_string(),
        content_sha256,
    })
}

pub(crate) fn manifest_file_fingerprint_to_value(file_fingerprint: &ManifestFileFingerprint) -> Value {
    json!({
        "path": &file_fingerprint.path,
        "size": file_fingerprint.size,
        "mtime_ns": file_fingerprint.mtime_ns,
        "content_hash_algorithm": &file_fingerprint.content_hash_algorithm,
        "content_sha256": &file_fingerprint.content_sha256,
    })
}

pub(crate) fn build_file_content_sha256(path: &Path) -> Result<String, OutputWriterError> {
    let mut file = File::open(path).map_err(OutputWriterError::runtime)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer).map_err(OutputWriterError::runtime)?;
        if bytes_read == 0 {
            break;
        }
        digest.update(&buffer[..bytes_read]);
    }
    Ok(encode_sha256_hex(digest))
}

pub(crate) fn build_manifest_json_sha256(manifest_json: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(manifest_json.as_bytes());
    encode_sha256_hex(digest)
}

fn build_manifest_value_sha256(value: &Value) -> Result<String, OutputWriterError> {
    let manifest_bytes = serde_json::to_vec(value).map_err(OutputWriterError::runtime)?;
    let mut digest = Sha256::new();
    digest.update(manifest_bytes);
    Ok(encode_sha256_hex(digest))
}

fn encode_sha256_hex(digest: Sha256) -> String {
    digest.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn directory_exists_and_is_non_empty(directory_path: &Path) -> Result<bool, OutputWriterError> {
    if !directory_path.exists() {
        return Ok(false);
    }
    let mut directory_entries = std::fs::read_dir(directory_path).map_err(OutputWriterError::runtime)?;
    match directory_entries.next() {
        Some(Ok(_directory_entry)) => Ok(true),
        Some(Err(error)) => Err(OutputWriterError::runtime(error)),
        None => Ok(false),
    }
}

fn load_run_manifest_value(run_directory: &Path) -> Result<Option<Value>, OutputWriterError> {
    let Some(manifest_json) = load_run_manifest_json(run_directory)? else {
        return Ok(None);
    };
    parse_run_manifest_text(&manifest_json, Some(&run_directory.join(RUN_MANIFEST_FILE_NAME))).map(Some)
}

fn parse_run_manifest_text(manifest_json: &str, manifest_path: Option<&Path>) -> Result<Value, OutputWriterError> {
    let manifest = serde_json::from_str::<Value>(manifest_json)
        .map_err(|error| OutputWriterError::InvalidInput(error.to_string()))?;
    if manifest.is_object() {
        return Ok(manifest);
    }
    let message = match manifest_path {
        Some(path) => format!("Run manifest '{}' must contain a JSON object.", path.display()),
        None => "Run manifest must contain a JSON object.".to_string(),
    };
    Err(OutputWriterError::InvalidInput(message))
}

fn parse_current_header_text(current_header_json: &str) -> Result<Value, OutputWriterError> {
    let current_header = serde_json::from_str::<Value>(current_header_json)
        .map_err(|error| OutputWriterError::InvalidInput(error.to_string()))?;
    if current_header.is_object() {
        return Ok(current_header);
    }
    Err(OutputWriterError::InvalidInput("Current run manifest header must contain a JSON object.".to_string()))
}

fn validate_manifest_compatibility_values(manifest: &Value, current_header: &Value) -> Result<(), OutputWriterError> {
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| OutputWriterError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
    let current_header_object = current_header.as_object().ok_or_else(|| {
        OutputWriterError::InvalidInput("Current run manifest header must contain a JSON object.".to_string())
    })?;
    for (field_name, current_value) in current_header_object {
        let Some(manifest_value) = manifest_object.get(field_name) else {
            return Err(OutputWriterError::InvalidInput(format!("Run manifest field '{field_name}' is missing.")));
        };
        if let Some(mismatch_path) = find_first_manifest_mismatch_path(manifest_value, current_value, field_name) {
            return Err(OutputWriterError::InvalidInput(format!(
                "Run manifest field '{mismatch_path}' is incompatible with the requested run."
            )));
        }
    }
    Ok(())
}

fn find_first_manifest_mismatch_path(
    manifest_value: &Value,
    current_value: &Value,
    field_path: &str,
) -> Option<String> {
    match (manifest_value, current_value) {
        (Value::Object(manifest_object), Value::Object(current_object)) => {
            let field_names = manifest_object.keys().chain(current_object.keys()).collect::<BTreeSet<_>>();
            for field_name in field_names {
                let nested_path = format!("{field_path}.{field_name}");
                match (manifest_object.get(field_name), current_object.get(field_name)) {
                    (Some(nested_manifest_value), Some(nested_current_value)) => {
                        if let Some(mismatch_path) =
                            find_first_manifest_mismatch_path(nested_manifest_value, nested_current_value, &nested_path)
                        {
                            return Some(mismatch_path);
                        }
                    }
                    _ => return Some(nested_path),
                }
            }
            None
        }
        (Value::Array(manifest_array), Value::Array(current_array)) => {
            for (index, (manifest_item, current_item)) in manifest_array.iter().zip(current_array).enumerate() {
                let nested_path = format!("{field_path}[{index}]");
                if let Some(mismatch_path) =
                    find_first_manifest_mismatch_path(manifest_item, current_item, &nested_path)
                {
                    return Some(mismatch_path);
                }
            }
            if manifest_array.len() != current_array.len() {
                return Some(field_path.to_string());
            }
            None
        }
        _ if manifest_scalar_values_match(manifest_value, current_value) => None,
        _ => Some(field_path.to_string()),
    }
}

fn manifest_scalar_values_match(manifest_value: &Value, current_value: &Value) -> bool {
    match (manifest_value, current_value) {
        (Value::Number(manifest_number), Value::Number(current_number)) => {
            if let (Some(manifest_integer), Some(current_integer)) = (manifest_number.as_i64(), current_number.as_i64())
            {
                return manifest_integer == current_integer;
            }
            if let (Some(manifest_integer), Some(current_integer)) = (manifest_number.as_u64(), current_number.as_u64())
            {
                return manifest_integer == current_integer;
            }
            manifest_number.as_f64() == current_number.as_f64()
        }
        _ => manifest_value == current_value,
    }
}

fn read_run_manifest_committed_chunks(manifest: &Value) -> Result<Vec<Value>, OutputWriterError> {
    let Some(committed_chunks) = manifest.get("committed_chunks") else {
        return Ok(Vec::new());
    };
    let committed_chunks_array = committed_chunks.as_array().ok_or_else(|| {
        OutputWriterError::InvalidInput("Run manifest committed_chunks field must be a list.".to_string())
    })?;
    for committed_chunk in committed_chunks_array {
        if !committed_chunk.is_object() {
            return Err(OutputWriterError::InvalidInput(
                "Run manifest committed chunk entries must be objects.".to_string(),
            ));
        }
    }
    Ok(committed_chunks_array.clone())
}

fn read_run_manifest_committed_chunk_identifiers(manifest: &Value) -> Result<Vec<i64>, OutputWriterError> {
    let committed_chunks = read_run_manifest_committed_chunks(manifest)?;
    let mut committed_chunk_identifiers = BTreeSet::new();
    for committed_chunk in committed_chunks {
        let Some(chunk_identifier) = committed_chunk.get("chunk_identifier").and_then(Value::as_i64) else {
            return Err(OutputWriterError::InvalidInput(
                "Run manifest committed chunk entry is missing chunk_identifier.".to_string(),
            ));
        };
        committed_chunk_identifiers.insert(chunk_identifier);
    }
    Ok(committed_chunk_identifiers.into_iter().collect())
}

fn merge_manifest_header(manifest: &mut Value, current_header: &Value) -> Result<(), OutputWriterError> {
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| OutputWriterError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
    let current_header_object = current_header.as_object().ok_or_else(|| {
        OutputWriterError::InvalidInput("Current run manifest header must contain a JSON object.".to_string())
    })?;
    for (field_name, field_value) in current_header_object {
        manifest_object.insert(field_name.clone(), field_value.clone());
    }
    Ok(())
}

fn write_run_manifest_value(run_directory: &Path, manifest: &Value) -> Result<(), String> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    let manifest_lock = get_run_manifest_update_lock();
    let _manifest_guard = manifest_lock.lock().map_err(|_| "Run manifest update lock was poisoned.".to_string())?;
    write_run_manifest_value_atomic(&manifest_path, manifest)
}

pub(crate) fn record_run_manifest_chunk_commits(
    run_directory: &Path,
    chunk_commits: Vec<RunManifestChunkCommit>,
) -> Result<(), String> {
    if chunk_commits.is_empty() {
        return Ok(());
    }
    update_run_manifest(run_directory, |manifest| {
        let manifest_object =
            manifest.as_object_mut().ok_or_else(|| "Run manifest must contain a JSON object.".to_string())?;
        let committed_chunks = manifest_object
            .entry("committed_chunks")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| "Run manifest committed_chunks field must be a list.".to_string())?;
        let mut committed_chunks_by_identifier = BTreeMap::new();
        for committed_chunk in committed_chunks.iter() {
            let existing_commit = read_run_manifest_chunk_commit(committed_chunk)?;
            insert_or_validate_chunk_commit(&mut committed_chunks_by_identifier, existing_commit)?;
        }
        for chunk_commit in chunk_commits {
            insert_or_validate_chunk_commit(&mut committed_chunks_by_identifier, chunk_commit)?;
        }
        *committed_chunks = committed_chunks_by_identifier.values().map(chunk_commit_to_value).collect();
        Ok(())
    })
}

fn insert_or_validate_chunk_commit(
    committed_chunks_by_identifier: &mut BTreeMap<i64, RunManifestChunkCommit>,
    chunk_commit: RunManifestChunkCommit,
) -> Result<(), String> {
    match committed_chunks_by_identifier.get(&chunk_commit.chunk_identifier) {
        Some(existing_commit) if existing_commit != &chunk_commit => {
            Err(format!("Run manifest has conflicting commit metadata for chunk {}.", chunk_commit.chunk_identifier))
        }
        Some(_) => Ok(()),
        None => {
            committed_chunks_by_identifier.insert(chunk_commit.chunk_identifier, chunk_commit);
            Ok(())
        }
    }
}

fn chunk_commit_to_value(chunk_commit: &RunManifestChunkCommit) -> Value {
    json!({
        "chunk_identifier": chunk_commit.chunk_identifier,
        "output_format": chunk_commit.output_format,
        "compression": chunk_commit.compression,
        "variant_start_index": chunk_commit.variant_start_index,
        "variant_stop_index": chunk_commit.variant_stop_index,
        "row_count": chunk_commit.row_count,
        "chunk_file_name": chunk_commit.chunk_file_name,
    })
}

fn read_run_manifest_chunk_commit(committed_chunk: &Value) -> Result<RunManifestChunkCommit, String> {
    let chunk_file_name = committed_chunk
        .get("chunk_file_name")
        .and_then(Value::as_str)
        .ok_or_else(|| "Run manifest committed chunk entry is missing chunk_file_name.".to_string())?;
    Ok(RunManifestChunkCommit {
        chunk_identifier: read_manifest_integer(committed_chunk, "chunk_identifier")?,
        output_format: read_optional_manifest_string(committed_chunk, "output_format")
            .unwrap_or_else(|| infer_output_format_from_file_name(chunk_file_name).to_string()),
        compression: read_optional_manifest_string(committed_chunk, "compression")
            .unwrap_or_else(|| "none".to_string()),
        variant_start_index: read_manifest_integer(committed_chunk, "variant_start_index")?,
        variant_stop_index: read_manifest_integer(committed_chunk, "variant_stop_index")?,
        row_count: read_manifest_usize(committed_chunk, "row_count")?,
        chunk_file_name: chunk_file_name.to_string(),
    })
}

fn read_optional_manifest_string(committed_chunk: &Value, field_name: &str) -> Option<String> {
    committed_chunk.get(field_name).and_then(Value::as_str).map(str::to_string)
}

fn infer_output_format_from_file_name(chunk_file_name: &str) -> &'static str {
    if chunk_file_name.ends_with(".regenie") {
        return "regenie";
    }
    if chunk_file_name.ends_with(".parquet") {
        return "parquet";
    }
    "arrow"
}

fn read_manifest_integer(committed_chunk: &Value, field_name: &str) -> Result<i64, String> {
    committed_chunk
        .get(field_name)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("Run manifest committed chunk entry is missing {field_name}."))
}

fn read_manifest_usize(committed_chunk: &Value, field_name: &str) -> Result<usize, String> {
    let value = read_manifest_integer(committed_chunk, field_name)?;
    usize::try_from(value).map_err(|_| format!("Run manifest committed chunk entry {field_name} must be non-negative."))
}

pub(crate) fn mark_run_manifest_finalized(
    final_parquet_path: &Path,
    row_count: usize,
    chunk_file_count: usize,
) -> Result<(), String> {
    mark_run_manifest_finalized_output(final_parquet_path, row_count, chunk_file_count, "parquet")
}

pub(crate) fn mark_run_manifest_finalized_output(
    final_output_path: &Path,
    row_count: usize,
    chunk_file_count: usize,
    output_format: &str,
) -> Result<(), String> {
    let Some(run_directory) = final_output_path.parent() else {
        return Ok(());
    };
    update_run_manifest(run_directory, |manifest| {
        let manifest_object =
            manifest.as_object_mut().ok_or_else(|| "Run manifest must contain a JSON object.".to_string())?;
        manifest_object.insert("finalized".to_string(), Value::Bool(true));
        manifest_object.insert("final_output".to_string(), Value::String(final_output_path.display().to_string()));
        manifest_object.insert("final_output_format".to_string(), Value::String(output_format.to_string()));
        match output_format {
            "parquet" => {
                manifest_object
                    .insert("final_parquet".to_string(), Value::String(final_output_path.display().to_string()));
                manifest_object.remove("final_regenie");
            }
            "regenie" => {
                manifest_object
                    .insert("final_regenie".to_string(), Value::String(final_output_path.display().to_string()));
                manifest_object.remove("final_parquet");
            }
            _ => {
                manifest_object.remove("final_parquet");
                manifest_object.remove("final_regenie");
            }
        }
        manifest_object.insert("final_row_count".to_string(), json!(row_count));
        manifest_object.insert("final_chunk_file_count".to_string(), json!(chunk_file_count));
        manifest_object.remove("interrupted");
        manifest_object.remove("interrupted_signal");
        Ok(())
    })
}

pub(crate) fn mark_run_manifest_interrupted(run_directory: &Path, signal_name: &str) -> Result<(), String> {
    update_run_manifest(run_directory, |manifest| {
        let manifest_object =
            manifest.as_object_mut().ok_or_else(|| "Run manifest must contain a JSON object.".to_string())?;
        manifest_object.insert("finalized".to_string(), Value::Bool(false));
        manifest_object.insert("interrupted".to_string(), Value::Bool(true));
        manifest_object.insert("interrupted_signal".to_string(), Value::String(signal_name.to_string()));
        manifest_object.remove("final_parquet");
        manifest_object.remove("final_regenie");
        manifest_object.remove("final_output");
        manifest_object.remove("final_output_format");
        manifest_object.remove("final_row_count");
        manifest_object.remove("final_chunk_file_count");
        Ok(())
    })
}

fn update_run_manifest(
    run_directory: &Path,
    update_manifest: impl FnOnce(&mut Value) -> Result<(), String>,
) -> Result<(), String> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    if !manifest_path.exists() {
        return Ok(());
    }
    let manifest_lock = get_run_manifest_update_lock();
    let _manifest_guard = manifest_lock.lock().map_err(|_| "Run manifest update lock was poisoned.".to_string())?;
    let manifest_text = std::fs::read_to_string(&manifest_path).map_err(|error| error.to_string())?;
    let mut manifest = serde_json::from_str::<Value>(&manifest_text).map_err(|error| error.to_string())?;
    update_manifest(&mut manifest)?;
    write_run_manifest_value_atomic(&manifest_path, &manifest)
}

fn write_run_manifest_value_atomic(manifest_path: &Path, manifest: &Value) -> Result<(), String> {
    let temporary_manifest_path = manifest_path.with_extension("json.tmp");
    let mut temporary_manifest_file = File::create(&temporary_manifest_path).map_err(|error| error.to_string())?;
    let manifest_bytes = serde_json::to_vec_pretty(manifest).map_err(|error| error.to_string())?;
    temporary_manifest_file.write_all(&manifest_bytes).map_err(|error| error.to_string())?;
    temporary_manifest_file.write_all(b"\n").map_err(|error| error.to_string())?;
    temporary_manifest_file.sync_all().map_err(|error| error.to_string())?;
    std::fs::rename(&temporary_manifest_path, manifest_path).map_err(|error| error.to_string())
}

fn get_run_manifest_update_lock() -> &'static Mutex<()> {
    static RUN_MANIFEST_UPDATE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    RUN_MANIFEST_UPDATE_LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn create_test_directory() -> PathBuf {
        let unique_suffix =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
        let directory_path = std::env::temp_dir().join(format!("g-output-manifest-test-{unique_suffix}"));
        std::fs::create_dir_all(&directory_path).expect("test directory should be created");
        directory_path
    }

    fn build_chunk_commit(chunk_identifier: i64) -> RunManifestChunkCommit {
        RunManifestChunkCommit {
            chunk_identifier,
            output_format: "arrow".to_string(),
            compression: "none".to_string(),
            variant_start_index: chunk_identifier,
            variant_stop_index: chunk_identifier + 2,
            row_count: 2,
            chunk_file_name: format!("chunk_{chunk_identifier:09}.arrow"),
        }
    }

    #[test]
    fn prepares_output_run_paths_and_rejects_unsafe_directory_state() {
        let root_directory = create_test_directory();
        let output_root = root_directory.join("result");

        let prepared_output_run = prepare_output_run(&output_root, "regenie2_linear", OutputFileFormat::Parquet, false)
            .expect("output run should prepare");

        assert_eq!(
            prepared_output_run.output_run_paths.run_directory,
            root_directory.join("result.regenie2_linear.run")
        );
        assert_eq!(
            prepared_output_run.output_run_paths.chunks_directory,
            root_directory.join("result.regenie2_linear.run").join("parts")
        );
        assert!(prepared_output_run.output_run_paths.chunks_directory.exists());
        assert_eq!(prepared_output_run.existing_manifest_json, None);

        let stale_output_root = root_directory.join("stale");
        let stale_run_paths = resolve_output_run_paths(&stale_output_root, "regenie2_linear", OutputFileFormat::Arrow);
        std::fs::create_dir_all(&stale_run_paths.run_directory).expect("stale run directory should be created");
        std::fs::write(stale_run_paths.run_directory.join("stale.txt"), "stale").expect("stale file should be written");
        let stale_error = prepare_output_run(&stale_output_root, "regenie2_linear", OutputFileFormat::Arrow, false)
            .expect_err("non-empty output directory should be rejected");
        assert!(stale_error.to_string().contains("already exists and is not empty"));

        let missing_resume_error =
            prepare_output_run(&root_directory.join("missing"), "regenie2_linear", OutputFileFormat::Arrow, true)
                .expect_err("resume without manifest should be rejected");
        assert_eq!(missing_resume_error.to_string(), "Resume requires run_manifest.json.");

        std::fs::remove_dir_all(root_directory).expect("test directory should be removed");
    }

    #[test]
    fn initializes_manifest_lifecycle_and_preserves_preinitialized_metadata() {
        let run_directory = create_test_directory();
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
        write_run_manifest_json(&run_directory, r#"{"command":{"interface":"g regenie"},"runtime":{"device":"cpu"}}"#)
            .expect("preinitialized manifest should be written");

        let initialized_output_run = initialize_output_run(
            &run_directory,
            &chunks_directory,
            None,
            r#"{"schema_version":7,"execution_plan":{"chunk_size":2},"execution_plan_hash":"hash"}"#,
            false,
            OutputResumeMode::Fast,
        )
        .expect("output run should initialize");

        assert_eq!(initialized_output_run.committed_chunk_identifiers, Vec::<i64>::new());
        let manifest_json =
            load_run_manifest_json(&run_directory).expect("manifest should load").expect("manifest should exist");
        let manifest = serde_json::from_str::<Value>(&manifest_json).expect("manifest should parse");
        assert_eq!(manifest.pointer("/command/interface").and_then(Value::as_str), Some("g regenie"));
        assert_eq!(manifest.pointer("/runtime/device").and_then(Value::as_str), Some("cpu"));
        assert_eq!(manifest.get("schema_version").and_then(Value::as_i64), Some(7));
        assert_eq!(manifest.get("finalized").and_then(Value::as_bool), Some(false));
        assert_eq!(manifest.get("committed_chunks").and_then(Value::as_array).map(Vec::len), Some(0));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn initialize_rejects_incompatible_manifest_without_rewrite() {
        let run_directory = create_test_directory();
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
        let manifest_json = r#"{"schema_version":7,"execution_plan":{"chunk_size":4},"execution_plan_hash":"old","committed_chunks":[]}"#;
        write_run_manifest_json(&run_directory, manifest_json).expect("manifest should be written");
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        let original_manifest_bytes = std::fs::read(&manifest_path).expect("manifest should be readable");

        let error = initialize_output_run(
            &run_directory,
            &chunks_directory,
            Some(manifest_json),
            r#"{"schema_version":7,"execution_plan":{"chunk_size":2},"execution_plan_hash":"new"}"#,
            true,
            OutputResumeMode::Fast,
        )
        .expect_err("incompatible manifest should be rejected");

        assert!(error.to_string().contains("execution_plan.chunk_size"));
        assert_eq!(std::fs::read(&manifest_path).expect("manifest should be readable"), original_manifest_bytes);

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn records_committed_chunks_once_in_identifier_order() {
        let run_directory = create_test_directory();
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        std::fs::write(&manifest_path, "{\n  \"committed_chunks\": []\n}\n").expect("manifest should be written");

        record_run_manifest_chunk_commits(
            &run_directory,
            vec![build_chunk_commit(2), build_chunk_commit(0), build_chunk_commit(2)],
        )
        .expect("manifest commits should be recorded");

        let manifest_text = std::fs::read_to_string(&manifest_path).expect("manifest should be readable");
        let manifest = serde_json::from_str::<Value>(&manifest_text).expect("manifest should be JSON");
        let committed_chunks =
            manifest.get("committed_chunks").and_then(Value::as_array).expect("committed chunks should be an array");
        let committed_chunk_identifiers = committed_chunks
            .iter()
            .map(|committed_chunk| {
                committed_chunk
                    .get("chunk_identifier")
                    .and_then(Value::as_i64)
                    .expect("chunk identifier should be present")
            })
            .collect::<Vec<_>>();

        assert_eq!(committed_chunk_identifiers, vec![0, 2]);

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn rejects_conflicting_duplicate_chunk_commit() {
        let run_directory = create_test_directory();
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        std::fs::write(&manifest_path, "{\n  \"committed_chunks\": []\n}\n").expect("manifest should be written");
        let mut conflicting_commit = build_chunk_commit(2);
        conflicting_commit.row_count = 3;

        let error = record_run_manifest_chunk_commits(&run_directory, vec![build_chunk_commit(2), conflicting_commit])
            .expect_err("conflicting duplicate chunk should be rejected");

        assert!(error.contains("conflicting commit metadata for chunk 2"));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn reads_manifest_chunk_commits_from_text() {
        let manifest = r#"{
          "committed_chunks": [
            {
              "chunk_identifier": 4,
              "variant_start_index": 4,
              "variant_stop_index": 6,
              "row_count": 2,
              "chunk_file_name": "part_000000000.parquet"
            }
          ]
        }"#;

        let chunk_commits = read_run_manifest_chunk_commits_from_text(manifest).expect("manifest commits should parse");

        assert_eq!(chunk_commits.len(), 1);
        assert_eq!(chunk_commits[0].chunk_identifier, 4);
        assert_eq!(chunk_commits[0].output_format, "parquet");
        assert_eq!(chunk_commits[0].compression, "none");
    }

    #[test]
    fn marks_manifest_interrupted_without_final_outputs() {
        let run_directory = create_test_directory();
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        std::fs::write(
            &manifest_path,
            "{\n  \"committed_chunks\": [],\n  \"finalized\": true,\n  \"final_parquet\": \"old.parquet\",\n  \"final_row_count\": 1,\n  \"final_chunk_file_count\": 1\n}\n",
        )
        .expect("manifest should be written");

        mark_run_manifest_interrupted(&run_directory, "SIGTERM").expect("manifest should be marked interrupted");

        let manifest_text = std::fs::read_to_string(&manifest_path).expect("manifest should be readable");
        let manifest = serde_json::from_str::<Value>(&manifest_text).expect("manifest should be JSON");

        assert_eq!(manifest.get("finalized").and_then(Value::as_bool), Some(false));
        assert_eq!(manifest.get("interrupted").and_then(Value::as_bool), Some(true));
        assert_eq!(manifest.get("interrupted_signal").and_then(Value::as_str), Some("SIGTERM"));
        assert!(manifest.get("final_parquet").is_none());
        assert!(manifest.get("final_row_count").is_none());
        assert!(manifest.get("final_chunk_file_count").is_none());

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }
}
