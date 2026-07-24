use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::error::{OutputError, OutputResult};
use crate::manifest::build_manifest_value_sha256;
use crate::persistence::identifier::{AttemptIdentifier, validate_safe_path_component};
use crate::persistence::io::{
    FileIntegrity, clone_file_no_replace_verified, create_directories_durable, file_sha256, sync_directory,
    write_bytes_atomic, write_json_atomic,
};
use crate::persistence::model::{CanonicalChunkPlan, OutputChunkCommit, OutputPartBinding};
use crate::persistence::receipt::{OutputPartReceipt, publish_part_receipt, read_part_receipt, verify_part_receipt};

pub(crate) const ATTEMPT_MANIFEST_SCHEMA_VERSION: u32 = 0;

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
    write_json_atomic(&input.paths.manifest_path, &manifest)?;
    attempt_manifest_value_sha256(&manifest)
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
    Ok(hex::encode(Sha256::digest(bytes)))
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
    let observed_sha256 = file_sha256(&paths.manifest_path)?;
    if observed_sha256 != expected_sha256 {
        return Err(OutputError::Runtime(format!(
            "Materialized output attempt manifest '{}' does not match its staged SHA-256.",
            paths.manifest_path.display()
        )));
    }
    Ok(())
}

pub(crate) fn verify_attempt_run(
    paths: &AttemptRunPaths,
    binding: &AttemptManifestBinding,
    expected_header: &Value,
    canonical_chunk_plan: &CanonicalChunkPlan,
    allowed_producer_attempts: &BTreeSet<AttemptIdentifier>,
    require_terminal_manifest: bool,
    orphan_part_policy: OrphanPartPolicy,
) -> OutputResult<VerifiedAttemptRun> {
    let manifest_bytes = std::fs::read(&paths.manifest_path).map_err(|error| {
        OutputError::Runtime(format!(
            "Failed to read output attempt manifest '{}': {error}",
            paths.manifest_path.display()
        ))
    })?;
    let manifest = serde_json::from_slice::<Value>(&manifest_bytes).map_err(|error| {
        OutputError::InvalidInput(format!(
            "Output attempt manifest '{}' is invalid JSON: {error}",
            paths.manifest_path.display()
        ))
    })?;
    validate_manifest_identity(&manifest, binding, expected_header)?;
    let status = serde_json::from_value::<AttemptManifestStatus>(
        manifest
            .get("status")
            .cloned()
            .ok_or_else(|| OutputError::InvalidInput("Output attempt manifest status is missing.".to_string()))?,
    )
    .map_err(|error| OutputError::InvalidInput(format!("Output attempt manifest status is invalid: {error}")))?;
    let interrupted_signal = manifest.get("interrupted_signal").and_then(Value::as_str);
    let failure_reason = manifest.get("failure_reason").and_then(Value::as_str);
    validate_terminal_details(&status, interrupted_signal, failure_reason)?;
    if require_terminal_manifest && status == AttemptManifestStatus::Running {
        return Err(OutputError::InvalidInput(
            "Durable output terminal references a running attempt manifest.".to_string(),
        ));
    }

    let manifest_receipts =
        serde_json::from_value::<Vec<OutputPartReceipt>>(manifest.get("committed_parts").cloned().ok_or_else(
            || OutputError::InvalidInput("Output attempt manifest committed_parts is missing.".to_string()),
        )?)
        .map_err(|error| {
            OutputError::InvalidInput(format!("Output attempt manifest committed_parts is invalid: {error}"))
        })?;
    let manifest_chunks =
        serde_json::from_value::<Vec<OutputChunkCommit>>(manifest.get("committed_chunks").cloned().ok_or_else(
            || OutputError::InvalidInput("Output attempt manifest committed_chunks is missing.".to_string()),
        )?)
        .map_err(|error| {
            OutputError::InvalidInput(format!("Output attempt manifest committed_chunks is invalid: {error}"))
        })?;
    let mut sorted_manifest_receipts = manifest_receipts;
    sort_and_validate_receipts(&mut sorted_manifest_receipts)?;
    if flatten_receipt_chunks(&sorted_manifest_receipts)? != manifest_chunks {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest committed chunks do not match its committed part receipts.".to_string(),
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
    Ok(VerifiedAttemptRun {
        status,
        receipts,
        committed_chunk_identifiers,
        manifest_sha256: file_sha256(&paths.manifest_path)?,
    })
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

fn validate_manifest_identity(
    manifest: &Value,
    binding: &AttemptManifestBinding,
    expected_header: &Value,
) -> OutputResult<()> {
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| OutputError::InvalidInput("Output attempt manifest must contain an object.".to_string()))?;
    if manifest_object.get("attempt_manifest_schema_version") != Some(&Value::from(ATTEMPT_MANIFEST_SCHEMA_VERSION)) {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest has an unsupported schema version.".to_string(),
        ));
    }
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
    let expected_fields = [
        ("run_set_id", binding.run_set_id.as_str()),
        ("attempt_id", binding.attempt_id.as_str()),
        ("phenotype_name", binding.phenotype_name.as_str()),
        ("output_directory_name", binding.output_directory_name.as_str()),
        ("chunk_plan_hash", binding.chunk_plan_sha256.as_str()),
    ];
    for (field_name, expected_value) in expected_fields {
        if manifest.get(field_name).and_then(Value::as_str) != Some(expected_value) {
            return Err(OutputError::InvalidInput(format!(
                "Output attempt manifest field '{field_name}' does not match its lineage binding."
            )));
        }
    }
    if manifest.get("execution_plan_hash").and_then(Value::as_str) != Some(binding.execution_plan_sha256.as_str()) {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest execution plan hash does not match its lineage binding.".to_string(),
        ));
    }
    let execution_plan = manifest
        .get("execution_plan")
        .ok_or_else(|| OutputError::InvalidInput("Output attempt manifest execution_plan is missing.".to_string()))?;
    if build_manifest_value_sha256(execution_plan)? != binding.execution_plan_sha256 {
        return Err(OutputError::InvalidInput(
            "Output attempt manifest execution plan contents do not match its hash.".to_string(),
        ));
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

    use super::{AttemptRunPaths, OrphanPartPolicy, OutputPartReceipt, reject_or_ignore_uncommitted_parts};
    use crate::persistence::identifier::AttemptIdentifier;

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
