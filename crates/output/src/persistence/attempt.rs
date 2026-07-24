use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{ErrorKind, Read};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::error::{OutputError, OutputResult};
use crate::manifest::{
    ExecutionPlanSchemaZero, OUTPUT_SCHEMA_VERSION, RUN_MANIFEST_SCHEMA_VERSION, build_manifest_value_sha256,
};
use crate::persistence::identifier::{AttemptIdentifier, validate_run_set_identifier, validate_safe_path_component};
use crate::persistence::io::{
    FileIntegrity, clone_file_no_replace_verified, create_directories_durable, sync_directory, write_bytes_atomic,
    write_json_atomic,
};
use crate::persistence::model::{CanonicalChunkPlan, OutputChunkCommit, OutputPartBinding};
use crate::persistence::receipt::{OutputPartReceipt, publish_part_receipt, read_part_receipt, verify_part_receipt};

pub(crate) const ATTEMPT_MANIFEST_SCHEMA_VERSION: u32 = 0;
/// Maximum encoded attempt-manifest size accepted by persistence recovery.
///
/// The 1 `GiB` schema-zero ceiling covers the known chromosome-22 scale at a
/// supported chunk size of one: 418,943 chunk commits appear both in receipts
/// and in the flattened commit list. Repeated interrupted flushes can validly
/// produce one receipt per chunk, so the measured regression considers every
/// receipt grouping from one through the configured maximum. It uses the
/// longest accepted lineage identifiers, a 255-byte phenotype name, and 32
/// `MiB` of variable-header reserve while retaining at least 20% capacity. The
/// ceiling still prevents an untrusted file from causing an unbounded recovery
/// allocation. Larger datasets must use larger chunks or a future output
/// schema with a different control-plane representation.
pub(crate) const ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES: u64 = 1024 * 1024 * 1024;

const ATTEMPT_MANIFEST_BASE_FIELD_NAMES: [&str; 15] = [
    "schema_version",
    "output_schema_version",
    "execution_plan",
    "execution_plan_hash",
    "attempt_manifest_schema_version",
    "run_set_id",
    "attempt_id",
    "phenotype_name",
    "output_directory_name",
    "chunk_plan_hash",
    "status",
    "committed_parts",
    "committed_chunks",
    "command",
    "runtime",
];
const ATTEMPT_MANIFEST_COMMAND_FIELD_NAMES: [&str; 3] = ["interface", "phenotype", "effective_config"];
const ATTEMPT_MANIFEST_RUNTIME_FIELD_NAMES: [&str; 6] =
    ["device", "cpu_threads", "writer_threads", "writer_queue_depth", "chunks_per_parquet_file", "parquet_compression"];
const ATTEMPT_MANIFEST_BOUNDED_READ_LIMIT_BYTES: u64 = ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES + 1;
const LINUX_OPEN_NO_FOLLOW_FLAG: i32 = 0o400_000;
const LINUX_OPEN_NONBLOCKING_FLAG: i32 = 0o4_000;

#[cfg(test)]
#[derive(Clone, Copy)]
enum MaterializedManifestTestReplacement {
    SymbolicLink,
    Socket,
}

#[cfg(test)]
thread_local! {
    static MATERIALIZED_MANIFEST_TEST_REPLACEMENT:
        std::cell::Cell<Option<MaterializedManifestTestReplacement>> = const { std::cell::Cell::new(None) };
}

#[derive(Clone, Debug)]
pub(crate) struct AttemptRunPaths {
    pub(crate) run_directory: PathBuf,
    pub(crate) parts_directory: PathBuf,
    pub(crate) commits_directory: PathBuf,
    pub(crate) manifest_path: PathBuf,
    pub(crate) effective_config_path: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct AttemptManifestBinding {
    pub(crate) run_set_id: String,
    pub(crate) attempt_id: AttemptIdentifier,
    pub(crate) phenotype_name: String,
    pub(crate) output_directory_name: String,
    pub(crate) execution_plan_sha256: String,
    pub(crate) chunk_plan_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttemptManifestStatus {
    Running,
    Completed,
    Interrupted,
    Failed,
}

// Schema zero deliberately keeps one flat, status-dependent wire object. The
// explicit field sets distinguish a missing required field from an explicit
// null and provide precise duplicate, unknown, and inapplicable-detail
// diagnostics before this single typed DTO parses values. Construction retains
// the matching explicit map: its deterministic field order is part of the
// canonical JSON bytes bound by terminal hashes. Replacing this with tagged or
// nested status DTOs belongs in a future schema revision.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttemptManifestSchemaZero {
    schema_version: i64,
    output_schema_version: i64,
    execution_plan: ExecutionPlanSchemaZero,
    execution_plan_hash: String,
    attempt_manifest_schema_version: u32,
    run_set_id: String,
    attempt_id: String,
    phenotype_name: String,
    output_directory_name: String,
    chunk_plan_hash: String,
    status: AttemptManifestStatus,
    committed_parts: Vec<OutputPartReceipt>,
    committed_chunks: Vec<OutputChunkCommit>,
    command: AttemptManifestCommandSchemaZero,
    runtime: AttemptManifestRuntimeSchemaZero,
    #[serde(default)]
    interrupted_signal: Option<String>,
    #[serde(default)]
    failure_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttemptManifestCommandSchemaZero {
    interface: String,
    phenotype: String,
    effective_config: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttemptManifestRuntimeSchemaZero {
    device: String,
    cpu_threads: RequiredNullableCpuThreads,
    writer_threads: u32,
    writer_queue_depth: u64,
    chunks_per_parquet_file: u64,
    parquet_compression: String,
}

#[derive(Debug, Deserialize)]
#[serde(transparent)]
struct RequiredNullableCpuThreads(Option<u32>);

#[derive(Debug)]
pub(crate) struct ValidatedAttemptManifestSchemaZero {
    pub(crate) status: AttemptManifestStatus,
    pub(crate) run_set_id: String,
    pub(crate) attempt_id: AttemptIdentifier,
    pub(crate) phenotype_name: String,
    pub(crate) output_directory_name: String,
    pub(crate) execution_plan_hash: String,
    pub(crate) chunk_plan_hash: String,
    pub(crate) committed_parts: Vec<OutputPartReceipt>,
    pub(crate) committed_chunks: Vec<OutputChunkCommit>,
    execution_plan: ExecutionPlanSchemaZero,
    command: AttemptManifestCommandSchemaZero,
}

impl ValidatedAttemptManifestSchemaZero {
    pub(crate) fn gpu_genotype_format(&self) -> g_plan::GpuGenotypeFormat {
        self.execution_plan.gpu_genotype_format()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct VerifiedAttemptRun {
    pub(crate) status: AttemptManifestStatus,
    pub(crate) receipts: Vec<OutputPartReceipt>,
    pub(crate) committed_chunk_identifiers: BTreeSet<usize>,
    pub(crate) manifest_sha256: String,
}

#[derive(Clone, Copy)]
pub(crate) enum OrphanPartPolicy {
    /// Reject receipt-less parts while verifying terminal authority.
    Reject,
    /// Ignore receipt-less parts so a nonterminal writer can regenerate them.
    Observe,
    /// Reject receipt-less parts at the terminal reconciliation barrier.
    Reconcile,
}

pub(crate) struct AttemptManifestWrite<'value> {
    pub(crate) paths: &'value AttemptRunPaths,
    pub(crate) binding: &'value AttemptManifestBinding,
    pub(crate) header: &'value Value,
    pub(crate) status: AttemptManifestStatus,
    pub(crate) interrupted_signal: Option<&'value str>,
    pub(crate) failure_reason: Option<&'value str>,
    pub(crate) receipts: &'value [OutputPartReceipt],
    pub(crate) run_plan: &'value g_plan::RunPlan,
}

impl AttemptRunPaths {
    pub(crate) fn new(
        attempts_directory: &Path,
        attempt_id: &AttemptIdentifier,
        output_directory_name: &str,
    ) -> OutputResult<Self> {
        validate_safe_path_component(output_directory_name, "phenotype directory name")?;
        let run_directory = attempts_directory.join(attempt_id.as_str()).join(output_directory_name);
        Ok(Self {
            parts_directory: run_directory.join("parts"),
            commits_directory: run_directory.join("commits"),
            manifest_path: run_directory.join("run_manifest.json"),
            effective_config_path: run_directory.join("effective_config.toml"),
            run_directory,
        })
    }

    pub(crate) fn initialize_directories(&self) -> OutputResult<()> {
        create_directories_durable(&self.run_directory)?;
        create_directories_durable(&self.parts_directory)?;
        create_directories_durable(&self.commits_directory)
    }

    pub(crate) fn reestablish_directory_durability(&self) -> OutputResult<()> {
        sync_existing_directory(&self.parts_directory)?;
        sync_existing_directory(&self.commits_directory)?;
        sync_existing_directory(&self.run_directory)
    }
}

impl AttemptManifestBinding {
    pub(crate) fn part_binding(&self) -> OutputPartBinding {
        OutputPartBinding {
            run_set_id: self.run_set_id.clone(),
            attempt_id: self.attempt_id.clone(),
            phenotype_name: self.phenotype_name.clone(),
            execution_plan_sha256: self.execution_plan_sha256.clone(),
            chunk_plan_sha256: self.chunk_plan_sha256.clone(),
        }
    }
}

pub(crate) fn write_effective_config(paths: &AttemptRunPaths, effective_config_toml: &str) -> OutputResult<()> {
    write_bytes_atomic(&paths.effective_config_path, effective_config_toml.as_bytes())
}

pub(crate) fn write_attempt_manifest(input: &AttemptManifestWrite<'_>) -> OutputResult<String> {
    let manifest = build_attempt_manifest_value(input)?;
    let manifest_sha256 = attempt_manifest_value_sha256(&manifest)?;
    write_json_atomic(&input.paths.manifest_path, &manifest)?;
    Ok(manifest_sha256)
}

pub(crate) fn build_attempt_manifest_value(input: &AttemptManifestWrite<'_>) -> OutputResult<Value> {
    validate_terminal_details(&input.status, input.interrupted_signal, input.failure_reason)?;
    let mut receipts = input.receipts.to_vec();
    sort_and_validate_receipts(&mut receipts)?;
    let committed_chunks = flatten_receipt_chunks(&receipts)?;
    let header_object = input.header.as_object().ok_or_else(|| {
        OutputError::InvalidInput("Current run manifest header must contain a JSON object.".to_string())
    })?;
    let mut manifest_object = header_object.clone();
    manifest_object.insert("attempt_manifest_schema_version".to_string(), Value::from(ATTEMPT_MANIFEST_SCHEMA_VERSION));
    manifest_object.insert("run_set_id".to_string(), Value::String(input.binding.run_set_id.clone()));
    manifest_object.insert("attempt_id".to_string(), Value::String(input.binding.attempt_id.as_str().to_string()));
    manifest_object.insert("phenotype_name".to_string(), Value::String(input.binding.phenotype_name.clone()));
    manifest_object
        .insert("output_directory_name".to_string(), Value::String(input.binding.output_directory_name.clone()));
    manifest_object.insert("chunk_plan_hash".to_string(), Value::String(input.binding.chunk_plan_sha256.clone()));
    manifest_object.insert("status".to_string(), serde_json::to_value(&input.status).map_err(OutputError::runtime)?);
    manifest_object
        .insert("committed_parts".to_string(), serde_json::to_value(&receipts).map_err(OutputError::runtime)?);
    manifest_object
        .insert("committed_chunks".to_string(), serde_json::to_value(&committed_chunks).map_err(OutputError::runtime)?);
    manifest_object.insert(
        "command".to_string(),
        json!({
            "interface": "g regenie",
            "phenotype": input.binding.phenotype_name,
            "effective_config": input.paths.effective_config_path.display().to_string(),
        }),
    );
    manifest_object.insert(
        "runtime".to_string(),
        json!({
            "device": input.run_plan.compute.device.as_str(),
            "cpu_threads": input.run_plan.compute.cpu_thread_count,
            "writer_threads": input.run_plan.output.writer_thread_count,
            "writer_queue_depth": crate::WRITER_QUEUE_DEPTH,
            "chunks_per_parquet_file": crate::CHUNKS_PER_PARQUET_FILE,
            "parquet_compression": "zstd",
        }),
    );
    match input.interrupted_signal {
        Some(signal_name) => {
            manifest_object.insert("interrupted_signal".to_string(), Value::String(signal_name.to_string()));
        }
        None => {
            manifest_object.remove("interrupted_signal");
        }
    }
    match input.failure_reason {
        Some(failure_reason) => {
            manifest_object.insert("failure_reason".to_string(), Value::String(failure_reason.to_string()));
        }
        None => {
            manifest_object.remove("failure_reason");
        }
    }
    Ok(Value::Object(manifest_object))
}

pub(crate) fn attempt_manifest_value_sha256(manifest: &Value) -> OutputResult<String> {
    let mut bytes = serde_json::to_vec_pretty(manifest).map_err(OutputError::runtime)?;
    bytes.push(b'\n');
    let byte_count = u64::try_from(bytes.len())
        .map_err(|error| OutputError::Runtime(format!("Attempt manifest byte count does not fit uint64: {error}")))?;
    validate_attempt_manifest_encoded_size(byte_count)?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn validate_attempt_manifest_encoded_size(byte_count: u64) -> OutputResult<()> {
    if byte_count > ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt manifest exceeds the maximum encoded size of {ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES} bytes."
        )));
    }
    Ok(())
}

/// Reads an optional attempt manifest through a bounded regular-file descriptor.
///
/// A missing path returns `None`. Existing symbolic links and non-regular files
/// are rejected. The read is capped at one byte beyond
/// [`ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES`] so growth after descriptor metadata
/// inspection cannot cause an unbounded allocation.
pub(crate) fn read_optional_attempt_manifest_bytes(manifest_path: &Path) -> OutputResult<Option<Vec<u8>>> {
    let path_metadata = match std::fs::symlink_metadata(manifest_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(OutputError::Runtime(format!(
                "Failed to inspect output attempt manifest '{}': {error}",
                manifest_path.display()
            )));
        }
    };
    if path_metadata.file_type().is_symlink() || !path_metadata.file_type().is_file() {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt manifest '{}' must be a regular file and must not be a symbolic link.",
            manifest_path.display()
        )));
    }

    let manifest_file = match OpenOptions::new()
        .read(true)
        .custom_flags(LINUX_OPEN_NO_FOLLOW_FLAG | LINUX_OPEN_NONBLOCKING_FLAG)
        .open(manifest_path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(OutputError::Runtime(format!(
                "Failed to open output attempt manifest '{}': {error}",
                manifest_path.display()
            )));
        }
    };
    let descriptor_metadata = manifest_file.metadata().map_err(|error| {
        OutputError::Runtime(format!(
            "Failed to inspect opened output attempt manifest '{}': {error}",
            manifest_path.display()
        ))
    })?;
    if !descriptor_metadata.file_type().is_file() {
        return Err(OutputError::InvalidInput(format!(
            "Opened output attempt manifest '{}' is not a regular file.",
            manifest_path.display()
        )));
    }
    if descriptor_metadata.len() > ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt manifest '{}' exceeds the maximum size of {ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES} bytes.",
            manifest_path.display()
        )));
    }

    let initial_capacity = usize::try_from(descriptor_metadata.len()).map_err(|error| {
        OutputError::Runtime(format!(
            "Output attempt manifest '{}' size does not fit the platform index width: {error}",
            manifest_path.display()
        ))
    })?;
    let mut manifest_bytes = Vec::with_capacity(initial_capacity);
    manifest_file.take(ATTEMPT_MANIFEST_BOUNDED_READ_LIMIT_BYTES).read_to_end(&mut manifest_bytes).map_err(
        |error| {
            OutputError::Runtime(format!(
                "Failed to read opened output attempt manifest '{}': {error}",
                manifest_path.display()
            ))
        },
    )?;
    let observed_size = u64::try_from(manifest_bytes.len()).map_err(|error| {
        OutputError::Runtime(format!(
            "Read output attempt manifest '{}' size does not fit uint64: {error}",
            manifest_path.display()
        ))
    })?;
    if observed_size > ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt manifest '{}' grew beyond the maximum size of {ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES} bytes while it was read.",
            manifest_path.display()
        )));
    }
    Ok(Some(manifest_bytes))
}

pub(crate) fn materialize_attempt_manifest(
    paths: &AttemptRunPaths,
    manifest: &Value,
    expected_sha256: &str,
) -> OutputResult<()> {
    if attempt_manifest_value_sha256(manifest)? != expected_sha256 {
        return Err(OutputError::InvalidInput(
            "Staged output attempt manifest does not match its terminal binding.".to_string(),
        ));
    }
    write_json_atomic(&paths.manifest_path, manifest)?;
    #[cfg(test)]
    replace_materialized_manifest_at_test_point(&paths.manifest_path)?;
    let observed_bytes = read_optional_attempt_manifest_bytes(&paths.manifest_path)?.ok_or_else(|| {
        OutputError::Runtime(format!(
            "Materialized output attempt manifest '{}' disappeared before verification.",
            paths.manifest_path.display()
        ))
    })?;
    let observed_sha256 = hex::encode(Sha256::digest(observed_bytes));
    if observed_sha256 != expected_sha256 {
        return Err(OutputError::Runtime(format!(
            "Materialized output attempt manifest '{}' does not match its staged SHA-256.",
            paths.manifest_path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
fn replace_materialized_manifest_at_test_point(manifest_path: &Path) -> OutputResult<()> {
    let replacement = MATERIALIZED_MANIFEST_TEST_REPLACEMENT.with(std::cell::Cell::take);
    let Some(replacement) = replacement else {
        return Ok(());
    };
    std::fs::remove_file(manifest_path).map_err(|error| {
        OutputError::Runtime(format!(
            "Failed to remove materialized output attempt manifest '{}' at the test replacement point: {error}",
            manifest_path.display()
        ))
    })?;
    match replacement {
        MaterializedManifestTestReplacement::SymbolicLink => {
            std::os::unix::fs::symlink("replacement-target", manifest_path).map_err(|error| {
                OutputError::Runtime(format!(
                    "Failed to install output attempt manifest test symlink '{}': {error}",
                    manifest_path.display()
                ))
            })?;
        }
        MaterializedManifestTestReplacement::Socket => {
            let listener = std::os::unix::net::UnixListener::bind(manifest_path).map_err(|error| {
                OutputError::Runtime(format!(
                    "Failed to install output attempt manifest test socket '{}': {error}",
                    manifest_path.display()
                ))
            })?;
            drop(listener);
        }
    }
    Ok(())
}

pub(crate) fn verify_attempt_run(
    paths: &AttemptRunPaths,
    manifest_bytes: &[u8],
    binding: &AttemptManifestBinding,
    expected_header: &Value,
    canonical_chunk_plan: &CanonicalChunkPlan,
    allowed_producer_attempts: &BTreeSet<AttemptIdentifier>,
    require_terminal_manifest: bool,
    orphan_part_policy: OrphanPartPolicy,
) -> OutputResult<VerifiedAttemptRun> {
    let manifest_sha256 = hex::encode(Sha256::digest(manifest_bytes));
    let manifest = parse_attempt_manifest_json(manifest_bytes, &paths.manifest_path)?;
    validate_current_manifest_header(&manifest, expected_header)?;
    let ValidatedAttemptManifestSchemaZero {
        status,
        committed_parts: sorted_manifest_receipts,
        committed_chunks: _manifest_chunks,
        ..
    } = validate_attempt_manifest_schema_zero(manifest, paths, binding)?;
    if require_terminal_manifest && status == AttemptManifestStatus::Running {
        return Err(OutputError::InvalidInput(
            "Durable output terminal references a running attempt manifest.".to_string(),
        ));
    }

    let receipts =
        scan_verified_receipts(paths, binding, canonical_chunk_plan, allowed_producer_attempts, orphan_part_policy)?;
    let receipt_map =
        receipts.iter().map(|receipt| (receipt.footer.receipt_id.as_str(), receipt)).collect::<BTreeMap<_, _>>();
    for manifest_receipt in &sorted_manifest_receipts {
        if receipt_map.get(manifest_receipt.footer.receipt_id.as_str()).copied() != Some(manifest_receipt) {
            return Err(OutputError::InvalidInput(format!(
                "Output attempt manifest references missing or mismatched receipt '{}'.",
                manifest_receipt.footer.receipt_id
            )));
        }
    }
    if status != AttemptManifestStatus::Running && sorted_manifest_receipts != receipts {
        return Err(OutputError::InvalidInput(
            "Terminal output attempt manifest does not bind every durable receipt.".to_string(),
        ));
    }
    let committed_chunk_identifiers = receipt_chunk_identifiers(&receipts)?;
    canonical_chunk_plan
        .validate_exact_coverage(receipts.iter().flat_map(|receipt| receipt.footer.chunks.iter()))
        .or_else(|error| if status == AttemptManifestStatus::Completed { Err(error) } else { Ok(()) })?;
    Ok(VerifiedAttemptRun { status, receipts, committed_chunk_identifiers, manifest_sha256 })
}

pub(crate) fn inspect_unmaterialized_attempt_run(
    paths: &AttemptRunPaths,
    binding: &AttemptManifestBinding,
    canonical_chunk_plan: &CanonicalChunkPlan,
    allowed_producer_attempts: &BTreeSet<AttemptIdentifier>,
) -> OutputResult<VerifiedAttemptRun> {
    let receipts = scan_verified_receipts(
        paths,
        binding,
        canonical_chunk_plan,
        allowed_producer_attempts,
        OrphanPartPolicy::Observe,
    )?;
    Ok(VerifiedAttemptRun {
        status: AttemptManifestStatus::Running,
        committed_chunk_identifiers: receipt_chunk_identifiers(&receipts)?,
        receipts,
        manifest_sha256: String::new(),
    })
}

pub(crate) fn reuse_verified_receipts(
    source_paths: &AttemptRunPaths,
    destination_paths: &AttemptRunPaths,
    receipts: &[OutputPartReceipt],
) -> OutputResult<()> {
    destination_paths.initialize_directories()?;
    for receipt in receipts {
        verify_part_receipt(&source_paths.parts_directory, receipt)?;
        let source_part_path = source_paths.parts_directory.join(&receipt.footer.part_file_name);
        let destination_part_path = destination_paths.parts_directory.join(&receipt.footer.part_file_name);
        let expected_integrity =
            FileIntegrity { size_bytes: receipt.part_size_bytes, sha256: receipt.part_sha256.clone() };
        clone_file_no_replace_verified(&source_part_path, &destination_part_path, &expected_integrity)?;
        publish_part_receipt(&destination_paths.commits_directory, receipt)?;
        verify_part_receipt(&destination_paths.parts_directory, receipt)?;
    }
    destination_paths.reestablish_directory_durability()?;
    Ok(())
}

struct DuplicateRejectingJsonValue(Value);

struct DuplicateRejectingJsonValueVisitor;

impl<'de> Deserialize<'de> for DuplicateRejectingJsonValue {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(DuplicateRejectingJsonValueVisitor)
    }
}

impl<'de> serde::de::Visitor<'de> for DuplicateRejectingJsonValueVisitor {
    type Value = DuplicateRejectingJsonValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<ErrorType>(self, value: bool) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        Ok(DuplicateRejectingJsonValue(Value::Bool(value)))
    }

    fn visit_i64<ErrorType>(self, value: i64) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        Ok(DuplicateRejectingJsonValue(Value::from(value)))
    }

    fn visit_u64<ErrorType>(self, value: u64) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        Ok(DuplicateRejectingJsonValue(Value::from(value)))
    }

    fn visit_f64<ErrorType>(self, value: f64) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        let number =
            serde_json::Number::from_f64(value).ok_or_else(|| ErrorType::custom("JSON numbers must be finite"))?;
        Ok(DuplicateRejectingJsonValue(Value::Number(number)))
    }

    fn visit_str<ErrorType>(self, value: &str) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        Ok(DuplicateRejectingJsonValue(Value::String(value.to_string())))
    }

    fn visit_borrowed_str<ErrorType>(self, value: &'de str) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        Ok(DuplicateRejectingJsonValue(Value::String(value.to_string())))
    }

    fn visit_string<ErrorType>(self, value: String) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        Ok(DuplicateRejectingJsonValue(Value::String(value)))
    }

    fn visit_none<ErrorType>(self) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        Ok(DuplicateRejectingJsonValue(Value::Null))
    }

    fn visit_unit<ErrorType>(self) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        Ok(DuplicateRejectingJsonValue(Value::Null))
    }

    fn visit_seq<SequenceType>(self, mut sequence: SequenceType) -> Result<Self::Value, SequenceType::Error>
    where
        SequenceType: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
        while let Some(value) = sequence.next_element::<DuplicateRejectingJsonValue>()? {
            values.push(value.0);
        }
        Ok(DuplicateRejectingJsonValue(Value::Array(values)))
    }

    fn visit_map<MapType>(self, mut map: MapType) -> Result<Self::Value, MapType::Error>
    where
        MapType: serde::de::MapAccess<'de>,
    {
        let mut object = serde_json::Map::new();
        while let Some(field_name) = map.next_key::<String>()? {
            if object.contains_key(&field_name) {
                return Err(serde::de::Error::custom(format!("duplicate object key '{field_name}'")));
            }
            let value = map.next_value::<DuplicateRejectingJsonValue>()?;
            object.insert(field_name, value.0);
        }
        Ok(DuplicateRejectingJsonValue(Value::Object(object)))
    }
}

pub(crate) fn parse_attempt_manifest_json(manifest_bytes: &[u8], manifest_path: &Path) -> OutputResult<Value> {
    serde_json::from_slice::<DuplicateRejectingJsonValue>(manifest_bytes).map(|manifest| manifest.0).map_err(|error| {
        OutputError::InvalidInput(format!(
            "Output attempt manifest '{}' is invalid JSON: {error}",
            manifest_path.display()
        ))
    })
}

pub(crate) fn validate_attempt_manifest_schema_zero(
    manifest: Value,
    paths: &AttemptRunPaths,
    binding: &AttemptManifestBinding,
) -> OutputResult<ValidatedAttemptManifestSchemaZero> {
    let validated = validate_attempt_manifest_schema_zero_shape(manifest)?;
    validate_attempt_manifest_authority_binding(&validated, binding)?;
    validate_attempt_manifest_command_binding(&validated.command, paths, binding)?;
    Ok(validated)
}

pub(crate) fn validate_attempt_manifest_schema_zero_shape(
    manifest: Value,
) -> OutputResult<ValidatedAttemptManifestSchemaZero> {
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| OutputError::InvalidInput("Output attempt manifest must contain an object.".to_string()))?;
    if manifest_object.get("attempt_manifest_schema_version") != Some(&Value::from(ATTEMPT_MANIFEST_SCHEMA_VERSION)) {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest has an unsupported schema version.".to_string(),
        ));
    }
    let status = serde_json::from_value::<AttemptManifestStatus>(
        manifest_object
            .get("status")
            .cloned()
            .ok_or_else(|| OutputError::InvalidInput("Output attempt manifest status is missing.".to_string()))?,
    )
    .map_err(|error| OutputError::InvalidInput(format!("Output attempt manifest status is invalid: {error}")))?;

    let mut expected_field_names = ATTEMPT_MANIFEST_BASE_FIELD_NAMES.to_vec();
    match &status {
        AttemptManifestStatus::Running | AttemptManifestStatus::Completed => {}
        AttemptManifestStatus::Interrupted => expected_field_names.push("interrupted_signal"),
        AttemptManifestStatus::Failed => expected_field_names.push("failure_reason"),
    }
    validate_exact_object_fields(manifest_object, &expected_field_names, "Output attempt manifest")?;
    validate_attempt_manifest_nested_field_sets(manifest_object)?;
    let execution_plan_sha256 =
        build_manifest_value_sha256(manifest_object.get("execution_plan").ok_or_else(|| {
            OutputError::InvalidInput("Output attempt manifest field 'execution_plan' is missing.".to_string())
        })?)?;
    let mut schema = serde_json::from_value::<AttemptManifestSchemaZero>(manifest).map_err(|error| {
        OutputError::InvalidInput(format!("Output attempt manifest schema zero is invalid: {error}"))
    })?;
    validate_attempt_manifest_schema_zero_semantics(&mut schema, &execution_plan_sha256)?;
    let attempt_id = AttemptIdentifier::parse(&schema.attempt_id)?;
    Ok(ValidatedAttemptManifestSchemaZero {
        status: schema.status,
        run_set_id: schema.run_set_id,
        attempt_id,
        phenotype_name: schema.phenotype_name,
        output_directory_name: schema.output_directory_name,
        execution_plan_hash: schema.execution_plan_hash,
        chunk_plan_hash: schema.chunk_plan_hash,
        committed_parts: schema.committed_parts,
        committed_chunks: schema.committed_chunks,
        execution_plan: schema.execution_plan,
        command: schema.command,
    })
}

fn validate_attempt_manifest_nested_field_sets(manifest_object: &serde_json::Map<String, Value>) -> OutputResult<()> {
    let command_object = manifest_object.get("command").and_then(Value::as_object).ok_or_else(|| {
        OutputError::InvalidInput("Output attempt manifest command must contain an object.".to_string())
    })?;
    validate_exact_object_fields(
        command_object,
        &ATTEMPT_MANIFEST_COMMAND_FIELD_NAMES,
        "Output attempt manifest command",
    )?;
    let runtime_object = manifest_object.get("runtime").and_then(Value::as_object).ok_or_else(|| {
        OutputError::InvalidInput("Output attempt manifest runtime must contain an object.".to_string())
    })?;
    validate_exact_object_fields(
        runtime_object,
        &ATTEMPT_MANIFEST_RUNTIME_FIELD_NAMES,
        "Output attempt manifest runtime",
    )
}

fn validate_exact_object_fields(
    object: &serde_json::Map<String, Value>,
    expected_field_names: &[&str],
    object_description: &str,
) -> OutputResult<()> {
    for field_name in expected_field_names {
        if !object.contains_key(*field_name) {
            return Err(OutputError::InvalidInput(format!("{object_description} field '{field_name}' is missing.")));
        }
    }
    for field_name in object.keys() {
        if !expected_field_names.contains(&field_name.as_str()) {
            return Err(OutputError::InvalidInput(format!(
                "{object_description} contains unknown field '{field_name}'."
            )));
        }
    }
    Ok(())
}

fn validate_attempt_manifest_schema_zero_semantics(
    schema: &mut AttemptManifestSchemaZero,
    execution_plan_sha256: &str,
) -> OutputResult<()> {
    if schema.schema_version != RUN_MANIFEST_SCHEMA_VERSION {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt manifest field 'schema_version' must equal {RUN_MANIFEST_SCHEMA_VERSION}."
        )));
    }
    if schema.output_schema_version != OUTPUT_SCHEMA_VERSION {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt manifest field 'output_schema_version' must equal {OUTPUT_SCHEMA_VERSION}."
        )));
    }
    if schema.attempt_manifest_schema_version != ATTEMPT_MANIFEST_SCHEMA_VERSION {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest has an unsupported schema version.".to_string(),
        ));
    }
    schema.execution_plan.validate()?;
    validate_sha256(&schema.execution_plan_hash, "execution plan")?;
    if execution_plan_sha256 != schema.execution_plan_hash {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest execution plan contents do not match its hash.".to_string(),
        ));
    }
    validate_run_set_identifier(&schema.run_set_id)?;
    AttemptIdentifier::parse(&schema.attempt_id)?;
    if schema.phenotype_name.is_empty() {
        return Err(OutputError::InvalidInput("Output attempt manifest phenotype name must not be empty.".to_string()));
    }
    if schema.execution_plan.phenotype_name() != schema.phenotype_name {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest execution plan phenotype does not match its manifest phenotype.".to_string(),
        ));
    }
    validate_safe_path_component(&schema.output_directory_name, "phenotype directory name")?;
    validate_sha256(&schema.chunk_plan_hash, "chunk plan")?;
    validate_attempt_manifest_command(&schema.command, &schema.phenotype_name)?;
    validate_attempt_manifest_runtime(&schema.runtime)?;
    validate_attempt_manifest_runtime_execution_plan_binding(&schema.runtime, &schema.execution_plan)?;
    validate_terminal_details(&schema.status, schema.interrupted_signal.as_deref(), schema.failure_reason.as_deref())?;

    sort_and_validate_receipts(&mut schema.committed_parts)?;
    if flatten_receipt_chunks(&schema.committed_parts)? != schema.committed_chunks {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest committed chunks do not match its committed part receipts.".to_string(),
        ));
    }
    Ok(())
}

fn validate_attempt_manifest_command(
    command: &AttemptManifestCommandSchemaZero,
    phenotype_name: &str,
) -> OutputResult<()> {
    if command.interface != "g regenie" {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest command field 'interface' is invalid.".to_string(),
        ));
    }
    if command.phenotype.is_empty() {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest command field 'phenotype' must contain a non-empty string.".to_string(),
        ));
    }
    if command.phenotype != phenotype_name {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest command phenotype does not match its manifest phenotype.".to_string(),
        ));
    }
    if command.effective_config.is_empty() {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest command field 'effective_config' must contain a non-empty string.".to_string(),
        ));
    }
    Ok(())
}

fn validate_attempt_manifest_command_binding(
    command: &AttemptManifestCommandSchemaZero,
    paths: &AttemptRunPaths,
    binding: &AttemptManifestBinding,
) -> OutputResult<()> {
    if command.phenotype != binding.phenotype_name {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest command field 'phenotype' does not match its lineage binding.".to_string(),
        ));
    }
    let expected_effective_config_path = paths.effective_config_path.display().to_string();
    if command.effective_config != expected_effective_config_path {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest command field 'effective_config' does not match its attempt path.".to_string(),
        ));
    }
    Ok(())
}

fn validate_attempt_manifest_authority_binding(
    manifest: &ValidatedAttemptManifestSchemaZero,
    binding: &AttemptManifestBinding,
) -> OutputResult<()> {
    if manifest.run_set_id != binding.run_set_id {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest run set does not match its immutable lineage binding.".to_string(),
        ));
    }
    if manifest.attempt_id != binding.attempt_id {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest attempt does not match its immutable lineage attempt.".to_string(),
        ));
    }
    if manifest.phenotype_name != binding.phenotype_name
        || manifest.output_directory_name != binding.output_directory_name
    {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest phenotype does not match its immutable lineage binding.".to_string(),
        ));
    }
    if manifest.execution_plan_hash != binding.execution_plan_sha256 {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest does not match its immutable lineage execution plan.".to_string(),
        ));
    }
    if manifest.chunk_plan_hash != binding.chunk_plan_sha256 {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest does not match its immutable lineage chunk plan.".to_string(),
        ));
    }
    Ok(())
}

fn validate_attempt_manifest_runtime(runtime: &AttemptManifestRuntimeSchemaZero) -> OutputResult<()> {
    if !matches!(runtime.device.as_str(), "cpu" | "gpu") {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest runtime field 'device' must be 'cpu' or 'gpu'.".to_string(),
        ));
    }
    if runtime.cpu_threads.0 == Some(0) {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest runtime field 'cpu_threads' must be null or a positive integer.".to_string(),
        ));
    }
    if runtime.writer_threads == 0 {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest runtime field 'writer_threads' must be a positive integer.".to_string(),
        ));
    }
    let writer_queue_depth = u64::try_from(crate::WRITER_QUEUE_DEPTH)
        .map_err(|error| OutputError::Runtime(format!("Writer queue depth does not fit manifest uint64: {error}")))?;
    if runtime.writer_queue_depth != writer_queue_depth {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt manifest runtime field 'writer_queue_depth' must equal {}.",
            crate::WRITER_QUEUE_DEPTH
        )));
    }
    let chunks_per_parquet_file = u64::try_from(crate::CHUNKS_PER_PARQUET_FILE).map_err(|error| {
        OutputError::Runtime(format!("Chunks per Parquet file does not fit manifest uint64: {error}"))
    })?;
    if runtime.chunks_per_parquet_file != chunks_per_parquet_file {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt manifest runtime field 'chunks_per_parquet_file' must equal {}.",
            crate::CHUNKS_PER_PARQUET_FILE
        )));
    }
    if runtime.parquet_compression != "zstd" {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest runtime field 'parquet_compression' must be 'zstd'.".to_string(),
        ));
    }
    Ok(())
}

fn validate_attempt_manifest_runtime_execution_plan_binding(
    runtime: &AttemptManifestRuntimeSchemaZero,
    execution_plan: &ExecutionPlanSchemaZero,
) -> OutputResult<()> {
    if runtime.device != execution_plan.runtime_device().as_str() {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest runtime field 'device' does not match its execution plan.".to_string(),
        ));
    }
    if runtime.writer_threads != execution_plan.writer_thread_count() {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest runtime field 'writer_threads' does not match its execution plan.".to_string(),
        ));
    }
    Ok(())
}

fn validate_sha256(digest: &str, role: &str) -> OutputResult<()> {
    if !crate::digest::is_canonical_sha256(digest) {
        return Err(OutputError::InvalidInput(format!(
            "Output {role} SHA-256 must contain exactly 64 hexadecimal characters."
        )));
    }
    Ok(())
}

fn sync_existing_directory(path: &Path) -> OutputResult<()> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => sync_directory(path),
        Ok(_) => Err(OutputError::InvalidInput(format!(
            "Expected output directory '{}' is not a directory.",
            path.display()
        ))),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(OutputError::Runtime(format!("Failed to inspect output directory '{}': {error}", path.display())))
        }
    }
}

fn scan_verified_receipts(
    paths: &AttemptRunPaths,
    binding: &AttemptManifestBinding,
    canonical_chunk_plan: &CanonicalChunkPlan,
    allowed_producer_attempts: &BTreeSet<AttemptIdentifier>,
    orphan_part_policy: OrphanPartPolicy,
) -> OutputResult<Vec<OutputPartReceipt>> {
    let directory_entries = match std::fs::read_dir(&paths.commits_directory) {
        Ok(entries) => Some(entries),
        Err(error) if error.kind() == ErrorKind::NotFound => None,
        Err(error) => {
            return Err(OutputError::Runtime(format!(
                "Failed to read output receipt directory '{}': {error}",
                paths.commits_directory.display()
            )));
        }
    };
    let mut receipts = Vec::new();
    if let Some(directory_entries) = directory_entries {
        for directory_entry in directory_entries {
            let directory_entry = directory_entry.map_err(OutputError::runtime)?;
            let file_name = directory_entry.file_name();
            let file_name_text = file_name.to_string_lossy();
            if file_name_text.starts_with('.') && file_name_text.strip_suffix(".tmp").is_some() {
                continue;
            }
            let receipt_path = directory_entry.path();
            if !directory_entry.file_type().map_err(OutputError::runtime)?.is_file()
                || receipt_path.extension().is_none_or(|extension| extension != "json")
            {
                return Err(OutputError::InvalidInput(format!(
                    "Output receipt directory contains unexpected entry '{}'.",
                    receipt_path.display()
                )));
            }
            let receipt = read_part_receipt(&receipt_path)?;
            validate_receipt_binding(&receipt, binding, allowed_producer_attempts)?;
            for chunk in &receipt.footer.chunks {
                canonical_chunk_plan.validate_commit(chunk)?;
            }
            verify_part_receipt(&paths.parts_directory, &receipt)?;
            receipts.push(receipt);
        }
    }
    sort_and_validate_receipts(&mut receipts)?;
    reject_or_ignore_uncommitted_parts(paths, orphan_part_policy, &receipts)?;
    sort_and_validate_receipts(&mut receipts)?;
    let _ = receipt_chunk_identifiers(&receipts)?;
    Ok(receipts)
}

fn reject_or_ignore_uncommitted_parts(
    paths: &AttemptRunPaths,
    orphan_part_policy: OrphanPartPolicy,
    receipts: &[OutputPartReceipt],
) -> OutputResult<()> {
    let directory_entries = match std::fs::read_dir(&paths.parts_directory) {
        Ok(entries) => entries,
        Err(error)
            if error.kind() == ErrorKind::NotFound && !matches!(orphan_part_policy, OrphanPartPolicy::Reject) =>
        {
            return Ok(());
        }
        Err(error) => {
            return Err(OutputError::Runtime(format!(
                "Failed to read output parts directory '{}': {error}",
                paths.parts_directory.display()
            )));
        }
    };
    let receipt_part_file_names =
        receipts.iter().map(|receipt| receipt.footer.part_file_name.as_str()).collect::<BTreeSet<_>>();
    for directory_entry in directory_entries {
        let directory_entry = directory_entry.map_err(OutputError::runtime)?;
        let file_name = directory_entry.file_name();
        let file_name_text = file_name.to_str().ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Output parts directory contains a non-UTF-8 entry under '{}'.",
                paths.parts_directory.display()
            ))
        })?;
        if file_name_text.starts_with('.') && file_name_text.strip_suffix(".tmp").is_some() {
            continue;
        }
        let part_path = directory_entry.path();
        if !directory_entry.file_type().map_err(OutputError::runtime)?.is_file()
            || part_path.extension().is_none_or(|extension| extension != "parquet")
        {
            return Err(OutputError::InvalidInput(format!(
                "Output parts directory contains unexpected entry '{}'.",
                part_path.display()
            )));
        }
        if receipt_part_file_names.contains(file_name_text) {
            continue;
        }
        // Receipt-less bytes carry no authority. Nonterminal recovery must leave
        // them uncommitted so the writer regenerates the part and performs its
        // existing footer and raw-byte comparison before publishing a receipt.
        if !matches!(orphan_part_policy, OrphanPartPolicy::Observe) {
            return Err(OutputError::InvalidInput(format!(
                "Output part '{}' has no immutable receipt.",
                part_path.display()
            )));
        }
    }
    Ok(())
}

fn validate_current_manifest_header(manifest: &Value, expected_header: &Value) -> OutputResult<()> {
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| OutputError::InvalidInput("Output attempt manifest must contain an object.".to_string()))?;
    let expected_header_object = expected_header
        .as_object()
        .ok_or_else(|| OutputError::InvalidInput("Current run manifest header must contain an object.".to_string()))?;
    for (field_name, expected_value) in expected_header_object {
        if manifest_object.get(field_name) != Some(expected_value) {
            return Err(OutputError::InvalidInput(format!(
                "Output attempt manifest field '{field_name}' does not match the current execution plan."
            )));
        }
    }
    Ok(())
}

fn validate_receipt_binding(
    receipt: &OutputPartReceipt,
    binding: &AttemptManifestBinding,
    allowed_producer_attempts: &BTreeSet<AttemptIdentifier>,
) -> OutputResult<()> {
    let observed = receipt.footer.binding();
    if observed.run_set_id != binding.run_set_id
        || observed.phenotype_name != binding.phenotype_name
        || observed.execution_plan_sha256 != binding.execution_plan_sha256
        || observed.chunk_plan_sha256 != binding.chunk_plan_sha256
        || !allowed_producer_attempts.contains(&observed.attempt_id)
    {
        return Err(OutputError::InvalidInput(format!(
            "Output receipt '{}' does not match its lineage, phenotype, execution plan, chunk plan, or producer ancestry.",
            receipt.footer.receipt_id
        )));
    }
    Ok(())
}

fn validate_terminal_details(
    status: &AttemptManifestStatus,
    interrupted_signal: Option<&str>,
    failure_reason: Option<&str>,
) -> OutputResult<()> {
    match (status, interrupted_signal, failure_reason) {
        (AttemptManifestStatus::Running | AttemptManifestStatus::Completed, None, None) => Ok(()),
        (AttemptManifestStatus::Interrupted, Some(signal), None) if !signal.trim().is_empty() => Ok(()),
        (AttemptManifestStatus::Failed, None, Some(reason)) if !reason.trim().is_empty() => Ok(()),
        _ => Err(OutputError::InvalidInput(
            "Output attempt manifest terminal details do not match its status.".to_string(),
        )),
    }
}

fn sort_and_validate_receipts(receipts: &mut [OutputPartReceipt]) -> OutputResult<()> {
    receipts.sort_by(|left, right| left.footer.receipt_id.cmp(&right.footer.receipt_id));
    let mut part_identifiers = BTreeSet::new();
    let mut receipt_identifiers = BTreeSet::new();
    for receipt in receipts {
        receipt.validate()?;
        if !part_identifiers.insert(receipt.footer.part_id.clone())
            || !receipt_identifiers.insert(receipt.footer.receipt_id.clone())
        {
            return Err(OutputError::InvalidInput(
                "Output attempt contains duplicate part or receipt identifiers.".to_string(),
            ));
        }
    }
    Ok(())
}

fn flatten_receipt_chunks(receipts: &[OutputPartReceipt]) -> OutputResult<Vec<OutputChunkCommit>> {
    let mut chunks_by_identifier = BTreeMap::new();
    for receipt in receipts {
        for chunk in &receipt.footer.chunks {
            if chunks_by_identifier.insert(chunk.chunk_identifier, chunk.clone()).is_some() {
                return Err(OutputError::InvalidInput(format!(
                    "Output attempt contains duplicate chunk identifier {}.",
                    chunk.chunk_identifier
                )));
            }
        }
    }
    Ok(chunks_by_identifier.into_values().collect())
}

fn receipt_chunk_identifiers(receipts: &[OutputPartReceipt]) -> OutputResult<BTreeSet<usize>> {
    flatten_receipt_chunks(receipts)?
        .into_iter()
        .map(|chunk| {
            usize::try_from(chunk.chunk_identifier).map_err(|_| {
                OutputError::InvalidInput(format!(
                    "Output chunk identifier {} does not fit the platform index width.",
                    chunk.chunk_identifier
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::{
        ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES, ATTEMPT_MANIFEST_SCHEMA_VERSION, AttemptManifestBinding,
        AttemptManifestStatus, AttemptManifestWrite, AttemptRunPaths, MATERIALIZED_MANIFEST_TEST_REPLACEMENT,
        MaterializedManifestTestReplacement, OrphanPartPolicy, OutputPartReceipt, attempt_manifest_value_sha256,
        build_attempt_manifest_value, materialize_attempt_manifest, parse_attempt_manifest_json,
        read_optional_attempt_manifest_bytes, reject_or_ignore_uncommitted_parts,
        validate_attempt_manifest_encoded_size, validate_attempt_manifest_schema_zero,
    };
    use crate::manifest::build_manifest_value_sha256;
    use crate::persistence::identifier::AttemptIdentifier;
    use crate::persistence::io::FileIntegrity;
    use crate::persistence::model::OutputChunkCommit;
    use crate::persistence::receipt::OutputPartFooter;

    fn schema_test_paths() -> AttemptRunPaths {
        let run_directory = PathBuf::from("/output/attempts/attempt-test/trait_0001_phenotype");
        AttemptRunPaths {
            parts_directory: run_directory.join("parts"),
            commits_directory: run_directory.join("commits"),
            manifest_path: run_directory.join("run_manifest.json"),
            effective_config_path: run_directory.join("effective_config.toml"),
            run_directory,
        }
    }

    fn schema_test_binding() -> AttemptManifestBinding {
        AttemptManifestBinding {
            run_set_id: "run-set-test".to_string(),
            attempt_id: AttemptIdentifier::for_test("attempt-test"),
            phenotype_name: "phenotype".to_string(),
            output_directory_name: "trait_0001_phenotype".to_string(),
            execution_plan_sha256: build_manifest_value_sha256(&schema_test_execution_plan())
                .expect("execution plan hashes"),
            chunk_plan_sha256: std::iter::repeat_n('b', 64).collect(),
        }
    }

    fn schema_test_execution_plan() -> Value {
        crate::manifest::canonical_execution_plan_schema_zero_test_value()
    }

    fn schema_test_header() -> Value {
        let execution_plan = schema_test_execution_plan();
        json!({
            "schema_version": 0,
            "output_schema_version": 0,
            "execution_plan_hash": build_manifest_value_sha256(&execution_plan).expect("execution plan hashes"),
            "execution_plan": execution_plan,
        })
    }

    fn schema_test_run_plan() -> g_plan::RunPlan {
        let kernel_plan = serde_json::from_value(schema_test_execution_plan()["binary_kernel_config"].clone())
            .expect("schema test kernel plan parses");
        g_plan::RunPlan {
            association_mode: g_plan::AssociationMode::Regenie2Binary,
            chunk_size: 32,
            input: g_plan::InputPlan {
                bgen_path: "genotypes.bgen".to_string(),
                sample_path: "genotypes.sample".to_string(),
                phenotype_path: "phenotypes.tsv".to_string(),
                prediction_list_path: "predictions.list".to_string(),
                covariate_path: None,
                covariate_names: Vec::new(),
            },
            compute: g_plan::ComputePlan {
                device: g_plan::Device::Gpu,
                cpu_thread_count: None,
                jax_cache_directory: None,
                multi_phenotype_sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
                kernels: kernel_plan,
            },
            correction: g_plan::CorrectionPlan {
                method: g_plan::BinaryFallbackMethod::FirthApproximate,
                p_threshold: g_plan::Probability::try_from(0.05).expect("valid correction threshold"),
                firth_se: false,
            },
            output: g_plan::OutputPlan {
                output_run_root: "/output".to_string(),
                resume: false,
                recover_attempt: None,
                fenced_owner_claim_id: None,
                writer_thread_count: 1,
            },
            telemetry: g_plan::TelemetryMode::Off,
            phenotype_runs: vec![g_plan::PhenotypeRunPlan {
                phenotype_name: "phenotype".to_string(),
                output_directory_name: "trait_0001_phenotype".to_string(),
            }],
        }
    }

    fn valid_schema_test_manifest(status: &AttemptManifestStatus) -> Value {
        let paths = schema_test_paths();
        let binding = schema_test_binding();
        let execution_plan = schema_test_execution_plan();
        let mut manifest = json!({
            "schema_version": 0,
            "output_schema_version": 0,
            "execution_plan": execution_plan,
            "execution_plan_hash": binding.execution_plan_sha256,
            "attempt_manifest_schema_version": ATTEMPT_MANIFEST_SCHEMA_VERSION,
            "run_set_id": binding.run_set_id,
            "attempt_id": binding.attempt_id.as_str(),
            "phenotype_name": binding.phenotype_name,
            "output_directory_name": binding.output_directory_name,
            "chunk_plan_hash": binding.chunk_plan_sha256,
            "status": status,
            "committed_parts": [],
            "committed_chunks": [],
            "command": {
                "interface": "g regenie",
                "phenotype": "phenotype",
                "effective_config": paths.effective_config_path.display().to_string(),
            },
            "runtime": {
                "device": "gpu",
                "cpu_threads": null,
                "writer_threads": 1,
                "writer_queue_depth": crate::WRITER_QUEUE_DEPTH,
                "chunks_per_parquet_file": crate::CHUNKS_PER_PARQUET_FILE,
                "parquet_compression": "zstd",
            },
        });
        match status {
            AttemptManifestStatus::Running | AttemptManifestStatus::Completed => {}
            AttemptManifestStatus::Interrupted => {
                manifest
                    .as_object_mut()
                    .expect("manifest is an object")
                    .insert("interrupted_signal".to_string(), Value::String("SIGTERM".to_string()));
            }
            AttemptManifestStatus::Failed => {
                manifest
                    .as_object_mut()
                    .expect("manifest is an object")
                    .insert("failure_reason".to_string(), Value::String("writer failed".to_string()));
            }
        }
        manifest
    }

    fn validate_schema_test_manifest(manifest: &Value) -> Result<AttemptManifestStatus, String> {
        validate_attempt_manifest_schema_zero(manifest.clone(), &schema_test_paths(), &schema_test_binding())
            .map(|validated| validated.status)
            .map_err(|error| error.to_string())
    }

    fn maximum_scale_binding() -> AttemptManifestBinding {
        AttemptManifestBinding {
            run_set_id: "r".repeat(128),
            attempt_id: AttemptIdentifier::for_test(&"a".repeat(128)),
            phenotype_name: "p".repeat(255),
            output_directory_name: "d".repeat(255),
            execution_plan_sha256: build_manifest_value_sha256(&schema_test_execution_plan())
                .expect("execution plan hashes"),
            chunk_plan_sha256: "b".repeat(64),
        }
    }

    fn representative_receipt(
        binding: &AttemptManifestBinding,
        first_chunk_identifier: i64,
        chunk_count: i64,
    ) -> OutputPartReceipt {
        let last_chunk_identifier = first_chunk_identifier + chunk_count - 1;
        let part_file_name = crate::writer::build_part_file_name(first_chunk_identifier, last_chunk_identifier);
        let chunks = (first_chunk_identifier..=last_chunk_identifier)
            .map(|chunk_identifier| OutputChunkCommit {
                chunk_identifier,
                variant_start_index: chunk_identifier,
                variant_stop_index: chunk_identifier + 1,
                row_count: 1,
                chunk_file_name: part_file_name.clone(),
            })
            .collect();
        let footer = OutputPartFooter::new(&binding.part_binding(), part_file_name, chunks)
            .expect("representative footer builds");
        OutputPartReceipt::new(footer, FileIntegrity { size_bytes: u64::MAX, sha256: "c".repeat(64) })
            .expect("representative receipt builds")
    }

    fn emitted_manifest_with_receipts(binding: &AttemptManifestBinding, receipts: &[OutputPartReceipt]) -> Value {
        build_attempt_manifest_value(&AttemptManifestWrite {
            paths: &schema_test_paths(),
            binding,
            header: &schema_test_header(),
            status: AttemptManifestStatus::Running,
            interrupted_signal: None,
            failure_reason: None,
            receipts,
            run_plan: &schema_test_run_plan(),
        })
        .expect("representative attempt manifest emits")
    }

    fn encoded_manifest_size(manifest: &Value) -> u64 {
        let encoded_size = serde_json::to_vec_pretty(manifest).expect("manifest serializes").len() + 1;
        u64::try_from(encoded_size).expect("manifest size fits uint64")
    }

    fn temporary_attempt_paths(test_name: &str) -> AttemptRunPaths {
        let test_directory = std::env::temp_dir().join(format!(
            "g-output-attempt-{test_name}-{}-{}",
            std::process::id(),
            AttemptIdentifier::generate().as_str()
        ));
        let paths = AttemptRunPaths {
            run_directory: test_directory.clone(),
            parts_directory: test_directory.join("parts"),
            commits_directory: test_directory.join("commits"),
            manifest_path: test_directory.join("run_manifest.json"),
            effective_config_path: test_directory.join("effective_config.toml"),
        };
        paths.initialize_directories().expect("attempt directories initialize");
        paths
    }

    fn remove_attempt(paths: &AttemptRunPaths) {
        let _ = std::fs::remove_dir_all(&paths.run_directory);
    }

    fn write_invalid_orphan(parts_directory: &Path) -> PathBuf {
        let orphan_path = parts_directory.join("part_000000000.parquet");
        std::fs::write(&orphan_path, b"not a parquet file").expect("orphan writes");
        orphan_path
    }

    #[test]
    fn attempt_manifest_json_rejects_duplicate_object_keys_recursively() {
        let duplicate_manifests: [&[u8]; 3] = [
            br#"{"field":0,"field":1}"#,
            br#"{"outer":{"field":0,"field":1}}"#,
            br#"{"outer":[{"field":0,"field":1}]}"#,
        ];
        for manifest_bytes in duplicate_manifests {
            let error = parse_attempt_manifest_json(manifest_bytes, Path::new("run_manifest.json"))
                .expect_err("duplicate object key is rejected");
            assert!(error.to_string().contains("duplicate object key 'field'"));
        }
    }

    #[test]
    fn attempt_manifest_reader_is_optional_and_bounded() {
        let paths = temporary_attempt_paths("bounded-manifest");
        assert!(
            read_optional_attempt_manifest_bytes(&paths.manifest_path).expect("missing manifest is optional").is_none()
        );

        let oversized_manifest = std::fs::File::create(&paths.manifest_path).expect("manifest creates");
        oversized_manifest
            .set_len(ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES + 1)
            .expect("sparse oversized manifest sets length");
        let error =
            read_optional_attempt_manifest_bytes(&paths.manifest_path).expect_err("oversized manifest is rejected");
        assert!(error.to_string().contains("exceeds the maximum size"));

        remove_attempt(&paths);
    }

    #[test]
    fn attempt_manifest_write_guard_rejects_the_first_oversized_byte() {
        validate_attempt_manifest_encoded_size(ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES)
            .expect("maximum-size encoded manifest is accepted");
        let error = validate_attempt_manifest_encoded_size(ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES + 1)
            .expect_err("first oversized encoded byte is rejected");
        assert!(error.to_string().contains("exceeds the maximum encoded size"));
    }

    #[test]
    fn attempt_manifest_limit_covers_measured_single_variant_chunk_scale() {
        const KNOWN_CHROMOSOME_22_CHUNK_COUNT: u64 = 418_943;
        const VARIABLE_HEADER_RESERVE_BYTES: u64 = 32 * 1024 * 1024;

        let maximum_chunks_per_receipt =
            u64::try_from(crate::CHUNKS_PER_PARQUET_FILE).expect("chunks per output part fit uint64");
        let known_chunk_count_i64 =
            i64::try_from(KNOWN_CHROMOSOME_22_CHUNK_COUNT).expect("known chromosome chunk count fits int64");
        let binding = maximum_scale_binding();
        let base_size = encoded_manifest_size(&emitted_manifest_with_receipts(&binding, &[]));
        let mut maximum_increment_per_chunk = 0;
        for chunks_per_receipt in 1..=maximum_chunks_per_receipt {
            let chunks_per_receipt_i64 = i64::try_from(chunks_per_receipt).expect("receipt chunk count fits int64");
            let first_chunk_identifier = known_chunk_count_i64 - 2 * chunks_per_receipt_i64;
            let first_receipt = representative_receipt(&binding, first_chunk_identifier, chunks_per_receipt_i64);
            let second_receipt = representative_receipt(
                &binding,
                first_chunk_identifier + chunks_per_receipt_i64,
                chunks_per_receipt_i64,
            );
            let first_size =
                encoded_manifest_size(&emitted_manifest_with_receipts(&binding, std::slice::from_ref(&first_receipt)));
            let second_size =
                encoded_manifest_size(&emitted_manifest_with_receipts(&binding, &[first_receipt, second_receipt]));
            let first_increment = first_size - base_size;
            let subsequent_increment = second_size - first_size;
            let increment_per_chunk = first_increment.max(subsequent_increment).div_ceil(chunks_per_receipt);
            maximum_increment_per_chunk = maximum_increment_per_chunk.max(increment_per_chunk);
        }
        let measured_upper_bound =
            base_size + KNOWN_CHROMOSOME_22_CHUNK_COUNT * maximum_increment_per_chunk + VARIABLE_HEADER_RESERVE_BYTES;

        // Upper-end six-digit identifiers dominate earlier chunks. Measuring
        // the first and subsequent additions for every legal receipt grouping
        // also covers arbitrary mixtures created by repeated interrupted
        // flushes; production part names retain their nine-digit padding.
        assert!(measured_upper_bound > ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES / 2);
        assert!(measured_upper_bound <= ATTEMPT_MANIFEST_MAXIMUM_SIZE_BYTES * 4 / 5);
    }

    #[test]
    fn attempt_manifest_reader_rejects_symlinks_and_special_files() {
        let paths = temporary_attempt_paths("manifest-file-type");
        std::os::unix::fs::symlink("missing-target", &paths.manifest_path).expect("manifest symlink creates");
        let symlink_error =
            read_optional_attempt_manifest_bytes(&paths.manifest_path).expect_err("manifest symlink is rejected");
        assert!(symlink_error.to_string().contains("must not be a symbolic link"));
        std::fs::remove_file(&paths.manifest_path).expect("manifest symlink removes");

        let listener = std::os::unix::net::UnixListener::bind(&paths.manifest_path).expect("manifest socket creates");
        let special_file_error =
            read_optional_attempt_manifest_bytes(&paths.manifest_path).expect_err("manifest socket is rejected");
        assert!(special_file_error.to_string().contains("must be a regular file"));
        drop(listener);

        remove_attempt(&paths);
    }

    #[test]
    fn materialized_manifest_verification_rejects_postpublication_path_replacement() {
        for (replacement, expected_error) in [
            (MaterializedManifestTestReplacement::SymbolicLink, "must not be a symbolic link"),
            (MaterializedManifestTestReplacement::Socket, "must be a regular file"),
        ] {
            let paths = temporary_attempt_paths("postwrite");
            let manifest = valid_schema_test_manifest(&AttemptManifestStatus::Interrupted);
            let expected_sha256 = attempt_manifest_value_sha256(&manifest).expect("staged manifest hashes");
            MATERIALIZED_MANIFEST_TEST_REPLACEMENT.with(|installed| {
                assert!(installed.replace(Some(replacement)).is_none(), "no manifest replacement is installed");
            });

            let error = materialize_attempt_manifest(&paths, &manifest, &expected_sha256)
                .expect_err("postpublication path replacement is rejected");
            assert!(error.to_string().contains(expected_error), "unexpected replacement error: {error}");
            remove_attempt(&paths);
        }
    }

    #[test]
    fn attempt_manifest_schema_rejects_unknown_and_missing_top_level_fields() {
        let mut unknown = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        unknown.as_object_mut().expect("manifest is an object").insert("unknown".to_string(), Value::Null);
        assert!(
            validate_schema_test_manifest(&unknown)
                .expect_err("unknown field is rejected")
                .contains("contains unknown field 'unknown'")
        );

        let mut missing = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        missing.as_object_mut().expect("manifest is an object").remove("committed_chunks");
        assert!(
            validate_schema_test_manifest(&missing)
                .expect_err("missing field is rejected")
                .contains("field 'committed_chunks' is missing")
        );
    }

    #[test]
    fn attempt_manifest_schema_preserves_unsupported_version_error() {
        let mut manifest = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        manifest["attempt_manifest_schema_version"] = Value::from(1);
        assert_eq!(
            validate_schema_test_manifest(&manifest).expect_err("unsupported version is rejected"),
            "Output attempt manifest has an unsupported schema version."
        );

        manifest.as_object_mut().expect("manifest is an object").remove("attempt_manifest_schema_version");
        assert_eq!(
            validate_schema_test_manifest(&manifest).expect_err("missing version is rejected"),
            "Output attempt manifest has an unsupported schema version."
        );
    }

    #[test]
    fn attempt_manifest_schema_accepts_exact_fields_for_all_statuses() {
        for status in [
            AttemptManifestStatus::Running,
            AttemptManifestStatus::Completed,
            AttemptManifestStatus::Interrupted,
            AttemptManifestStatus::Failed,
        ] {
            let manifest = valid_schema_test_manifest(&status);
            assert_eq!(validate_schema_test_manifest(&manifest).expect("status-specific manifest is accepted"), status);
            let expected_field_count = match status {
                AttemptManifestStatus::Running | AttemptManifestStatus::Completed => 15,
                AttemptManifestStatus::Interrupted | AttemptManifestStatus::Failed => 16,
            };
            assert_eq!(manifest.as_object().expect("manifest is an object").len(), expected_field_count);
        }
    }

    #[test]
    fn attempt_manifest_emitter_and_strict_parser_remain_aligned_for_all_statuses() {
        let paths = schema_test_paths();
        let binding = schema_test_binding();
        let header = schema_test_header();
        let run_plan = schema_test_run_plan();
        for (status, interrupted_signal, failure_reason) in [
            (AttemptManifestStatus::Running, None, None),
            (AttemptManifestStatus::Completed, None, None),
            (AttemptManifestStatus::Interrupted, Some("SIGTERM"), None),
            (AttemptManifestStatus::Failed, None, Some("writer failed")),
        ] {
            let expected_status = status.clone();
            let emitted = build_attempt_manifest_value(&AttemptManifestWrite {
                paths: &paths,
                binding: &binding,
                header: &header,
                status,
                interrupted_signal,
                failure_reason,
                receipts: &[],
                run_plan: &run_plan,
            })
            .expect("attempt manifest emits");
            let emitted_object = emitted.as_object().expect("emitted manifest is an object");
            assert_eq!(emitted_object.get("interrupted_signal").and_then(Value::as_str), interrupted_signal);
            assert_eq!(emitted_object.get("failure_reason").and_then(Value::as_str), failure_reason);
            assert_eq!(emitted_object.contains_key("interrupted_signal"), interrupted_signal.is_some());
            assert_eq!(emitted_object.contains_key("failure_reason"), failure_reason.is_some());

            let mut encoded = serde_json::to_vec_pretty(&emitted).expect("emitted manifest serializes");
            encoded.push(b'\n');
            let expected_sha256 = hex::encode(Sha256::digest(&encoded));
            assert_eq!(attempt_manifest_value_sha256(&emitted).expect("emitted manifest hashes"), expected_sha256);
            let parsed =
                parse_attempt_manifest_json(&encoded, &paths.manifest_path).expect("emitted manifest strictly parses");
            assert_eq!(parsed, emitted);
            let validated = validate_attempt_manifest_schema_zero(parsed, &paths, &binding)
                .expect("emitted manifest strictly validates");
            assert_eq!(validated.status, expected_status);
        }
    }

    #[test]
    fn attempt_manifest_schema_rejects_inapplicable_or_wrong_typed_terminal_details() {
        let mut running = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        running
            .as_object_mut()
            .expect("manifest is an object")
            .insert("interrupted_signal".to_string(), Value::String("SIGTERM".to_string()));
        assert!(
            validate_schema_test_manifest(&running)
                .expect_err("running detail is rejected")
                .contains("unknown field 'interrupted_signal'")
        );

        let mut interrupted = valid_schema_test_manifest(&AttemptManifestStatus::Interrupted);
        interrupted["interrupted_signal"] = Value::from(15);
        assert!(
            validate_schema_test_manifest(&interrupted)
                .expect_err("wrong-typed interrupted signal is rejected")
                .contains("schema zero is invalid")
        );

        let mut missing_interrupted = valid_schema_test_manifest(&AttemptManifestStatus::Interrupted);
        missing_interrupted.as_object_mut().expect("manifest is an object").remove("interrupted_signal");
        assert!(
            validate_schema_test_manifest(&missing_interrupted)
                .expect_err("missing interrupted signal is rejected")
                .contains("field 'interrupted_signal' is missing")
        );

        let mut failed = valid_schema_test_manifest(&AttemptManifestStatus::Failed);
        failed["failure_reason"] = Value::Bool(false);
        assert!(
            validate_schema_test_manifest(&failed)
                .expect_err("wrong-typed failure reason is rejected")
                .contains("schema zero is invalid")
        );

        let mut empty_failed = valid_schema_test_manifest(&AttemptManifestStatus::Failed);
        empty_failed["failure_reason"] = Value::String("  ".to_string());
        assert!(
            validate_schema_test_manifest(&empty_failed)
                .expect_err("empty failure reason is rejected")
                .contains("terminal details do not match its status")
        );
    }

    #[test]
    fn attempt_manifest_command_requires_exact_typed_binding_and_path() {
        let mut wrong_interface = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        wrong_interface["command"]["interface"] = Value::from(7);
        assert!(
            validate_schema_test_manifest(&wrong_interface)
                .expect_err("wrong-typed interface is rejected")
                .contains("schema zero is invalid")
        );

        let mut wrong_phenotype = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        wrong_phenotype["command"]["phenotype"] = Value::String("other".to_string());
        assert!(
            validate_schema_test_manifest(&wrong_phenotype)
                .expect_err("wrong phenotype is rejected")
                .contains("does not match its manifest phenotype")
        );

        let mut wrong_path = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        wrong_path["command"]["effective_config"] = Value::String("/other/config.toml".to_string());
        assert!(
            validate_schema_test_manifest(&wrong_path)
                .expect_err("wrong effective config path is rejected")
                .contains("does not match its attempt path")
        );

        let mut unknown = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        unknown["command"].as_object_mut().expect("command is an object").insert("unknown".to_string(), Value::Null);
        assert!(
            validate_schema_test_manifest(&unknown)
                .expect_err("unknown command field is rejected")
                .contains("command contains unknown field 'unknown'")
        );

        let mut missing = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        missing["command"].as_object_mut().expect("command is an object").remove("interface");
        assert!(
            validate_schema_test_manifest(&missing)
                .expect_err("missing command field is rejected")
                .contains("command field 'interface' is missing")
        );

        let mut wrong_object_type = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        wrong_object_type["command"] = Value::Null;
        assert!(
            validate_schema_test_manifest(&wrong_object_type)
                .expect_err("wrong-typed command is rejected")
                .contains("command must contain an object")
        );
    }

    #[test]
    fn attempt_manifest_execution_plan_phenotype_must_match_manifest_phenotype() {
        let mut manifest = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        manifest["execution_plan"]["phenotype_name"] = Value::String("other-phenotype".to_string());
        manifest["execution_plan_hash"] =
            Value::String(build_manifest_value_sha256(&manifest["execution_plan"]).expect("execution plan rehashes"));

        assert!(
            validate_schema_test_manifest(&manifest)
                .expect_err("execution-plan phenotype mismatch is rejected")
                .contains("execution plan phenotype does not match its manifest phenotype")
        );
    }

    #[test]
    fn attempt_manifest_runtime_requires_exact_typed_emitted_fields() {
        let mut positive_cpu_threads = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        positive_cpu_threads["runtime"]["cpu_threads"] = Value::from(2);
        validate_schema_test_manifest(&positive_cpu_threads).expect("positive CPU thread count is accepted");

        let invalid_runtime_fields = [
            ("device", Value::String(String::new())),
            ("cpu_threads", Value::from(0)),
            ("cpu_threads", json!(1.5)),
            ("writer_threads", Value::from(0)),
            ("writer_threads", json!(1.5)),
            ("writer_queue_depth", Value::from(crate::WRITER_QUEUE_DEPTH + 1)),
            ("chunks_per_parquet_file", Value::from(crate::CHUNKS_PER_PARQUET_FILE + 1)),
            ("parquet_compression", Value::String("snappy".to_string())),
        ];
        for (field_name, invalid_value) in invalid_runtime_fields {
            let mut manifest = valid_schema_test_manifest(&AttemptManifestStatus::Running);
            manifest["runtime"]
                .as_object_mut()
                .expect("runtime is an object")
                .insert(field_name.to_string(), invalid_value);
            assert!(
                validate_schema_test_manifest(&manifest).is_err(),
                "invalid runtime field {field_name} is rejected"
            );
        }

        for (field_name, mismatched_value) in
            [("device", Value::String("cpu".to_string())), ("writer_threads", Value::from(2))]
        {
            let mut manifest = valid_schema_test_manifest(&AttemptManifestStatus::Running);
            manifest["runtime"][field_name] = mismatched_value;
            let error = validate_schema_test_manifest(&manifest)
                .expect_err("valid typed runtime metadata must still match the execution plan");
            assert!(
                error.contains(&format!("runtime field '{field_name}' does not match its execution plan")),
                "unexpected mismatch error for {field_name}: {error}"
            );
        }

        let mut missing = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        missing["runtime"].as_object_mut().expect("runtime is an object").remove("writer_threads");
        assert!(
            validate_schema_test_manifest(&missing)
                .expect_err("missing runtime field is rejected")
                .contains("runtime field 'writer_threads' is missing")
        );

        let mut unknown = valid_schema_test_manifest(&AttemptManifestStatus::Running);
        unknown["runtime"].as_object_mut().expect("runtime is an object").insert("unknown".to_string(), Value::Null);
        assert!(
            validate_schema_test_manifest(&unknown)
                .expect_err("unknown runtime field is rejected")
                .contains("runtime contains unknown field 'unknown'")
        );
    }

    #[test]
    fn nonterminal_observation_does_not_parse_or_hash_orphan_bytes() {
        let paths = temporary_attempt_paths("observe-orphan");
        let orphan_path = write_invalid_orphan(&paths.parts_directory);

        reject_or_ignore_uncommitted_parts(&paths, OrphanPartPolicy::Observe, &[])
            .expect("uncommitted orphan is ignored");

        assert!(orphan_path.exists());
        assert_eq!(std::fs::read_dir(&paths.commits_directory).expect("commits read").count(), 0);
        remove_attempt(&paths);
    }

    #[test]
    fn terminal_barriers_reject_receipt_less_parts() {
        let paths = temporary_attempt_paths("reject-orphan");
        let orphan_path = write_invalid_orphan(&paths.parts_directory);
        let receipts = Vec::<OutputPartReceipt>::new();

        for policy in [OrphanPartPolicy::Reject, OrphanPartPolicy::Reconcile] {
            let error = reject_or_ignore_uncommitted_parts(&paths, policy, &receipts)
                .expect_err("terminal barrier rejects orphan");
            assert!(error.to_string().contains("has no immutable receipt"));
        }

        assert!(orphan_path.exists());
        assert_eq!(std::fs::read_dir(&paths.commits_directory).expect("commits read").count(), 0);
        remove_attempt(&paths);
    }
}
