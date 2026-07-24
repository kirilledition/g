use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

use crate::error::{OutputError, OutputResult};
use crate::persistence::identifier::{AttemptIdentifier, validate_run_set_identifier};
use crate::persistence::io::{
    FileIntegrity, NoReplacePublication, file_size_and_sha256, publish_json_no_replace, sync_directory,
};
use crate::persistence::model::{OutputChunkCommit, OutputPartBinding};
use crate::schema;

pub(crate) const PART_RECORD_SCHEMA_VERSION: u32 = 0;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputPartFooter {
    schema_version: u32,
    pub(crate) run_set_id: String,
    pub(crate) attempt_id: AttemptIdentifier,
    pub(crate) phenotype_name: String,
    pub(crate) execution_plan_sha256: String,
    pub(crate) chunk_plan_sha256: String,
    pub(crate) part_id: String,
    pub(crate) part_file_name: String,
    pub(crate) receipt_id: String,
    pub(crate) receipt_file_name: String,
    pub(crate) chunks: Vec<OutputChunkCommit>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputPartReceipt {
    pub(crate) footer: OutputPartFooter,
    pub(crate) part_size_bytes: u64,
    pub(crate) part_sha256: String,
}

impl OutputPartFooter {
    pub(crate) fn new(
        binding: &OutputPartBinding,
        part_file_name: String,
        chunks: Vec<OutputChunkCommit>,
    ) -> OutputResult<Self> {
        let part_id = part_file_name
            .strip_suffix(".parquet")
            .ok_or_else(|| {
                OutputError::InvalidInput(format!("Output part file name '{part_file_name}' must end in .parquet."))
            })?
            .to_string();
        let receipt_id = part_id.clone();
        let receipt_file_name = format!("{receipt_id}.json");
        let footer = Self {
            schema_version: PART_RECORD_SCHEMA_VERSION,
            run_set_id: binding.run_set_id.clone(),
            attempt_id: binding.attempt_id.clone(),
            phenotype_name: binding.phenotype_name.clone(),
            execution_plan_sha256: binding.execution_plan_sha256.clone(),
            chunk_plan_sha256: binding.chunk_plan_sha256.clone(),
            part_id,
            part_file_name,
            receipt_id,
            receipt_file_name,
            chunks,
        };
        footer.validate()?;
        Ok(footer)
    }

    pub(crate) fn to_metadata_text(&self) -> OutputResult<String> {
        serde_json::to_string(self).map_err(OutputError::runtime)
    }

    pub(crate) fn validate(&self) -> OutputResult<()> {
        if self.schema_version != PART_RECORD_SCHEMA_VERSION {
            return Err(OutputError::InvalidInput(format!(
                "Output part footer schema version {} is unsupported.",
                self.schema_version
            )));
        }
        validate_run_set_identifier(&self.run_set_id)?;
        AttemptIdentifier::parse(self.attempt_id.as_str())?;
        if self.phenotype_name.trim().is_empty() {
            return Err(OutputError::InvalidInput("Output part footer phenotype name must not be empty.".to_string()));
        }
        validate_sha256(&self.execution_plan_sha256, "execution plan")?;
        validate_sha256(&self.chunk_plan_sha256, "chunk plan")?;
        validate_path_identifier(&self.part_id, "part")?;
        validate_path_identifier(&self.receipt_id, "receipt")?;
        if self.part_id != self.receipt_id {
            return Err(OutputError::InvalidInput(
                "Output part and receipt identifiers must be identical.".to_string(),
            ));
        }
        if self.part_file_name != format!("{}.parquet", self.part_id)
            || self.receipt_file_name != format!("{}.json", self.receipt_id)
        {
            return Err(OutputError::InvalidInput(
                "Output part footer file names do not match its part identifier.".to_string(),
            ));
        }
        if self.chunks.is_empty() {
            return Err(OutputError::InvalidInput("Output part footer must bind at least one chunk.".to_string()));
        }
        let mut chunk_identifiers = BTreeSet::new();
        let mut previous_identifier = None;
        for chunk in &self.chunks {
            if chunk.chunk_file_name != self.part_file_name {
                return Err(OutputError::InvalidInput(format!(
                    "Output part footer chunk {} names a different part file.",
                    chunk.chunk_identifier
                )));
            }
            if !chunk_identifiers.insert(chunk.chunk_identifier) {
                return Err(OutputError::InvalidInput(format!(
                    "Output part footer contains duplicate chunk identifier {}.",
                    chunk.chunk_identifier
                )));
            }
            if previous_identifier.is_some_and(|previous| previous >= chunk.chunk_identifier) {
                return Err(OutputError::InvalidInput(
                    "Output part footer chunks must be ordered by identifier.".to_string(),
                ));
            }
            previous_identifier = Some(chunk.chunk_identifier);
        }
        Ok(())
    }

    pub(crate) fn binding(&self) -> OutputPartBinding {
        OutputPartBinding {
            run_set_id: self.run_set_id.clone(),
            attempt_id: self.attempt_id.clone(),
            phenotype_name: self.phenotype_name.clone(),
            execution_plan_sha256: self.execution_plan_sha256.clone(),
            chunk_plan_sha256: self.chunk_plan_sha256.clone(),
        }
    }
}

impl OutputPartReceipt {
    pub(crate) fn new(footer: OutputPartFooter, integrity: FileIntegrity) -> OutputResult<Self> {
        let receipt = Self { footer, part_size_bytes: integrity.size_bytes, part_sha256: integrity.sha256 };
        receipt.validate()?;
        Ok(receipt)
    }

    pub(crate) fn validate(&self) -> OutputResult<()> {
        self.footer.validate()?;
        validate_sha256(&self.part_sha256, "part")?;
        if self.part_size_bytes == 0 {
            return Err(OutputError::InvalidInput("Output part receipt byte size must be positive.".to_string()));
        }
        Ok(())
    }
}

pub(crate) fn publish_part_receipt(
    commits_directory: &Path,
    receipt: &OutputPartReceipt,
) -> OutputResult<NoReplacePublication> {
    receipt.validate()?;
    let receipt_path = commits_directory.join(&receipt.footer.receipt_file_name);
    let publication = match publish_json_no_replace(&receipt_path, receipt) {
        Ok(publication) => publication,
        Err(publication_error) => match read_part_receipt(&receipt_path) {
            Ok(existing) if existing == *receipt => NoReplacePublication::Created,
            Ok(_) => {
                return Err(OutputError::InvalidInput(format!(
                    "Output part receipt '{}' conflicts with an existing immutable receipt.",
                    receipt_path.display()
                )));
            }
            Err(_) => return Err(publication_error),
        },
    };
    if publication == NoReplacePublication::AlreadyExists {
        let existing = read_part_receipt(&receipt_path)?;
        if existing != *receipt {
            return Err(OutputError::InvalidInput(format!(
                "Output part receipt '{}' conflicts with an existing immutable receipt.",
                receipt_path.display()
            )));
        }
    }
    sync_directory(commits_directory)?;
    Ok(publication)
}

pub(crate) fn read_part_receipt(receipt_path: &Path) -> OutputResult<OutputPartReceipt> {
    let bytes = std::fs::read(receipt_path).map_err(|error| {
        OutputError::Runtime(format!("Failed to read output part receipt '{}': {error}", receipt_path.display()))
    })?;
    let receipt = serde_json::from_slice::<OutputPartReceipt>(&bytes).map_err(|error| {
        OutputError::InvalidInput(format!("Output part receipt '{}' is invalid JSON: {error}", receipt_path.display()))
    })?;
    receipt.validate()?;
    Ok(receipt)
}

pub(crate) fn read_part_footer(part_path: &Path) -> OutputResult<OutputPartFooter> {
    let input_file = File::open(part_path).map_err(|error| {
        OutputError::Runtime(format!("Failed to open output part '{}': {error}", part_path.display()))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file).map_err(|error| {
        OutputError::InvalidInput(format!(
            "Output part '{}' has an invalid Parquet footer: {error}",
            part_path.display()
        ))
    })?;
    if builder.schema().fields() != schema::REGENIE_STEP2_CHUNK_SCHEMA.fields() {
        return Err(OutputError::InvalidInput(format!(
            "Output part '{}' does not match the canonical output schema.",
            part_path.display()
        )));
    }
    let metadata = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .and_then(|entries| entries.iter().find(|entry| entry.key == schema::PART_BINDING_METADATA_KEY))
        .and_then(|entry| entry.value.as_deref())
        .ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Output part '{}' is missing bound part footer metadata.",
                part_path.display()
            ))
        })?;
    let footer = serde_json::from_str::<OutputPartFooter>(metadata).map_err(|error| {
        OutputError::InvalidInput(format!(
            "Output part '{}' has invalid bound footer metadata: {error}",
            part_path.display()
        ))
    })?;
    footer.validate()?;
    Ok(footer)
}

pub(crate) fn verify_part_receipt(parts_directory: &Path, receipt: &OutputPartReceipt) -> OutputResult<()> {
    receipt.validate()?;
    let part_path = parts_directory.join(&receipt.footer.part_file_name);
    let observed_footer = read_part_footer(&part_path)?;
    if observed_footer != receipt.footer {
        return Err(OutputError::InvalidInput(format!(
            "Output part '{}' footer does not match its immutable receipt.",
            part_path.display()
        )));
    }
    let observed_integrity = file_size_and_sha256(&part_path)?;
    if observed_integrity.size_bytes != receipt.part_size_bytes || observed_integrity.sha256 != receipt.part_sha256 {
        return Err(OutputError::InvalidInput(format!(
            "Output part '{}' raw bytes do not match its immutable receipt.",
            part_path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn receipt_path(commits_directory: &Path, receipt_id: &str) -> OutputResult<PathBuf> {
    validate_path_identifier(receipt_id, "receipt")?;
    Ok(commits_directory.join(format!("{receipt_id}.json")))
}

fn validate_path_identifier(identifier: &str, role: &str) -> OutputResult<()> {
    if identifier.is_empty()
        || identifier.len() > 128
        || !identifier.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(OutputError::InvalidInput(format!(
            "Output {role} identifier must be a non-empty path-safe identifier of at most 128 ASCII characters."
        )));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn footer() -> OutputPartFooter {
        OutputPartFooter::new(
            &OutputPartBinding {
                run_set_id: "run-set-test".to_string(),
                attempt_id: AttemptIdentifier::for_test("attempt-test"),
                phenotype_name: "trait-a".to_string(),
                execution_plan_sha256: digest('a'),
                chunk_plan_sha256: digest('b'),
            },
            "part_000000000.parquet".to_string(),
            vec![OutputChunkCommit {
                chunk_identifier: 0,
                variant_start_index: 0,
                variant_stop_index: 3,
                row_count: 3,
                chunk_file_name: "part_000000000.parquet".to_string(),
            }],
        )
        .expect("footer builds")
    }

    #[test]
    fn footer_and_receipt_repeat_identity_without_self_hashing() {
        let footer = footer();
        assert_eq!(footer.binding().attempt_id.as_str(), "attempt-test");
        assert_eq!(
            receipt_path(Path::new("commits"), &footer.part_id).expect("receipt path builds"),
            Path::new("commits").join("part_000000000.json")
        );
        let receipt = OutputPartReceipt::new(footer.clone(), FileIntegrity { size_bytes: 123, sha256: digest('c') })
            .expect("receipt builds");
        assert_eq!(receipt.footer, footer);
        assert_eq!(receipt.part_size_bytes, 123);
        assert_eq!(receipt.part_sha256, digest('c'));

        let metadata_text = footer.to_metadata_text().expect("footer serializes");
        let decoded: OutputPartFooter = serde_json::from_str(&metadata_text).expect("footer deserializes");
        assert_eq!(decoded, footer);
    }

    #[test]
    fn footer_rejects_duplicate_or_mismatched_chunk_bindings() {
        let original = footer();
        let mut duplicate = original.clone();
        duplicate.chunks.push(original.chunks[0].clone());
        assert!(duplicate.validate().is_err());

        let mut mismatched = original;
        mismatched.chunks[0].chunk_file_name = "other.parquet".to_string();
        assert!(mismatched.validate().is_err());
    }

    #[test]
    fn immutable_receipt_replay_is_idempotent_and_conflicts_fail() {
        let directory = std::env::temp_dir().join(format!(
            "g-output-receipt-{}-{}",
            std::process::id(),
            AttemptIdentifier::generate().as_str()
        ));
        let receipt = OutputPartReceipt::new(footer(), FileIntegrity { size_bytes: 123, sha256: digest('c') })
            .expect("receipt builds");
        assert_eq!(
            publish_part_receipt(&directory, &receipt).expect("first receipt publishes"),
            NoReplacePublication::Created
        );
        assert_eq!(
            publish_part_receipt(&directory, &receipt).expect("identical receipt replays"),
            NoReplacePublication::AlreadyExists
        );
        let conflicting = OutputPartReceipt { part_size_bytes: 124, ..receipt };
        let error = publish_part_receipt(&directory, &conflicting).expect_err("conflicting receipt is rejected");
        assert!(error.to_string().contains("conflicts"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn immutable_part_records_reject_unknown_fields_and_uppercase_hashes() {
        let receipt = OutputPartReceipt::new(footer(), FileIntegrity { size_bytes: 123, sha256: digest('c') })
            .expect("receipt builds");
        let mut forged_receipt = serde_json::to_value(&receipt).expect("receipt serializes");
        forged_receipt
            .as_object_mut()
            .expect("receipt is an object")
            .insert("ignored_authority".to_string(), serde_json::Value::Bool(true));
        let error = serde_json::from_value::<OutputPartReceipt>(forged_receipt)
            .expect_err("unknown receipt fields are rejected");
        assert!(error.to_string().contains("unknown field"));

        let mut forged_footer = serde_json::to_value(&receipt.footer).expect("footer serializes");
        forged_footer
            .as_object_mut()
            .expect("footer is an object")
            .insert("ignored_binding".to_string(), serde_json::Value::Bool(true));
        let error =
            serde_json::from_value::<OutputPartFooter>(forged_footer).expect_err("unknown footer fields are rejected");
        assert!(error.to_string().contains("unknown field"));

        let mut uppercase = receipt;
        uppercase.part_sha256 = digest('C');
        assert!(uppercase.validate().is_err());
    }
}
