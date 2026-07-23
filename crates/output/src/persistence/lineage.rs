use std::collections::BTreeSet;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{OutputError, OutputResult};
use crate::persistence::identifier::{AttemptIdentifier, generate_run_set_identifier, validate_run_set_identifier};
use crate::persistence::io::{NoReplacePublication, create_directories_durable, file_sha256, publish_json_no_replace};

const LINEAGE_SCHEMA_VERSION: u32 = 0;
const GENESIS_FILE_NAME: &str = "genesis.json";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct PhenotypeLineageContract {
    pub(crate) phenotype_name: String,
    pub(crate) output_directory_name: String,
    pub(crate) execution_plan_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct LineageGenesisRecord {
    record_kind: LineageRecordKind,
    schema_version: u32,
    pub(crate) run_set_id: String,
    pub(crate) attempt_id: AttemptIdentifier,
    pub(crate) chunk_plan_sha256: String,
    pub(crate) phenotypes: Vec<PhenotypeLineageContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct LineageSuccessorRecord {
    record_kind: LineageRecordKind,
    schema_version: u32,
    pub(crate) run_set_id: String,
    pub(crate) parent_attempt_id: AttemptIdentifier,
    pub(crate) attempt_id: AttemptIdentifier,
    pub(crate) recovery_kind: LineageRecoveryKind,
    pub(crate) parent_terminal_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LineageRecoveryKind {
    TerminalResume,
    ExactNonterminalRecovery,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttemptTerminalStatus {
    Completed,
    Interrupted,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct TerminalPhenotypeRecord {
    pub(crate) phenotype_name: String,
    pub(crate) output_directory_name: String,
    pub(crate) run_manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct LineageTerminalRecord {
    record_kind: LineageRecordKind,
    schema_version: u32,
    pub(crate) run_set_id: String,
    pub(crate) attempt_id: AttemptIdentifier,
    pub(crate) status: AttemptTerminalStatus,
    pub(crate) interrupted_signal: Option<String>,
    pub(crate) failure_reason: Option<String>,
    pub(crate) phenotypes: Vec<TerminalPhenotypeRecord>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum LineageRecordKind {
    Genesis,
    Successor,
    Terminal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LineageSnapshot {
    pub(crate) genesis: LineageGenesisRecord,
    pub(crate) successor_records: Vec<LineageSuccessorRecord>,
    pub(crate) leaf_attempt_id: AttemptIdentifier,
    pub(crate) leaf_terminal: Option<LineageTerminalRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct OutputLineagePaths {
    pub(crate) output_root: PathBuf,
    pub(crate) control_directory: PathBuf,
    pub(crate) successors_directory: PathBuf,
    pub(crate) terminals_directory: PathBuf,
    pub(crate) attempts_directory: PathBuf,
    pub(crate) genesis_path: PathBuf,
}

impl LineageGenesisRecord {
    pub(crate) fn new(
        attempt_id: AttemptIdentifier,
        chunk_plan_sha256: String,
        phenotypes: Vec<PhenotypeLineageContract>,
    ) -> Self {
        Self {
            record_kind: LineageRecordKind::Genesis,
            schema_version: LINEAGE_SCHEMA_VERSION,
            run_set_id: generate_run_set_identifier(),
            attempt_id,
            chunk_plan_sha256,
            phenotypes,
        }
    }

    fn validate(&self) -> OutputResult<()> {
        if self.record_kind != LineageRecordKind::Genesis || self.schema_version != LINEAGE_SCHEMA_VERSION {
            return Err(OutputError::InvalidInput(
                "Output lineage genesis record has an unsupported record kind or schema version.".to_string(),
            ));
        }
        validate_run_set_identifier(&self.run_set_id)?;
        AttemptIdentifier::parse(self.attempt_id.as_str())?;
        validate_sha256(&self.chunk_plan_sha256, "chunk plan")?;
        validate_phenotype_contracts(&self.phenotypes)
    }
}

impl LineageSuccessorRecord {
    pub(crate) fn new(
        run_set_id: String,
        parent_attempt_id: AttemptIdentifier,
        attempt_id: AttemptIdentifier,
        recovery_kind: LineageRecoveryKind,
        parent_terminal_sha256: Option<String>,
    ) -> OutputResult<Self> {
        let record = Self {
            record_kind: LineageRecordKind::Successor,
            schema_version: LINEAGE_SCHEMA_VERSION,
            run_set_id,
            parent_attempt_id,
            attempt_id,
            recovery_kind,
            parent_terminal_sha256,
        };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> OutputResult<()> {
        if self.record_kind != LineageRecordKind::Successor || self.schema_version != LINEAGE_SCHEMA_VERSION {
            return Err(OutputError::InvalidInput(
                "Output lineage successor record has an unsupported record kind or schema version.".to_string(),
            ));
        }
        validate_run_set_identifier(&self.run_set_id)?;
        AttemptIdentifier::parse(self.parent_attempt_id.as_str())?;
        AttemptIdentifier::parse(self.attempt_id.as_str())?;
        if self.parent_attempt_id == self.attempt_id {
            return Err(OutputError::InvalidInput(
                "Output lineage successor attempt must differ from its parent.".to_string(),
            ));
        }
        match (self.recovery_kind, self.parent_terminal_sha256.as_deref()) {
            (LineageRecoveryKind::TerminalResume, Some(terminal_sha256)) => {
                validate_sha256(terminal_sha256, "parent terminal")
            }
            (LineageRecoveryKind::ExactNonterminalRecovery, None) => Ok(()),
            _ => Err(OutputError::InvalidInput(
                "Output lineage successor terminal binding does not match its recovery kind.".to_string(),
            )),
        }
    }
}

impl LineageTerminalRecord {
    pub(crate) fn completed(
        run_set_id: String,
        attempt_id: AttemptIdentifier,
        phenotypes: Vec<TerminalPhenotypeRecord>,
    ) -> Self {
        Self {
            record_kind: LineageRecordKind::Terminal,
            schema_version: LINEAGE_SCHEMA_VERSION,
            run_set_id,
            attempt_id,
            status: AttemptTerminalStatus::Completed,
            interrupted_signal: None,
            failure_reason: None,
            phenotypes,
        }
    }

    pub(crate) fn interrupted(
        run_set_id: String,
        attempt_id: AttemptIdentifier,
        signal_name: String,
        phenotypes: Vec<TerminalPhenotypeRecord>,
    ) -> Self {
        Self {
            record_kind: LineageRecordKind::Terminal,
            schema_version: LINEAGE_SCHEMA_VERSION,
            run_set_id,
            attempt_id,
            status: AttemptTerminalStatus::Interrupted,
            interrupted_signal: Some(signal_name),
            failure_reason: None,
            phenotypes,
        }
    }

    pub(crate) fn failed(
        run_set_id: String,
        attempt_id: AttemptIdentifier,
        failure_reason: String,
        phenotypes: Vec<TerminalPhenotypeRecord>,
    ) -> Self {
        Self {
            record_kind: LineageRecordKind::Terminal,
            schema_version: LINEAGE_SCHEMA_VERSION,
            run_set_id,
            attempt_id,
            status: AttemptTerminalStatus::Failed,
            interrupted_signal: None,
            failure_reason: Some(failure_reason),
            phenotypes,
        }
    }

    fn validate(&self) -> OutputResult<()> {
        if self.record_kind != LineageRecordKind::Terminal || self.schema_version != LINEAGE_SCHEMA_VERSION {
            return Err(OutputError::InvalidInput(
                "Output lineage terminal record has an unsupported record kind or schema version.".to_string(),
            ));
        }
        validate_run_set_identifier(&self.run_set_id)?;
        AttemptIdentifier::parse(self.attempt_id.as_str())?;
        match (self.status, self.interrupted_signal.as_deref(), self.failure_reason.as_deref()) {
            (AttemptTerminalStatus::Completed, None, None)
            | (AttemptTerminalStatus::Interrupted, Some(_), None)
            | (AttemptTerminalStatus::Failed, None, Some(_)) => {}
            _ => {
                return Err(OutputError::InvalidInput(
                    "Output lineage terminal details do not match its status.".to_string(),
                ));
            }
        }
        if self.phenotypes.is_empty() {
            return Err(OutputError::InvalidInput(
                "Output lineage terminal must bind at least one phenotype manifest.".to_string(),
            ));
        }
        let mut names = BTreeSet::new();
        let mut output_names = BTreeSet::new();
        for phenotype in &self.phenotypes {
            if !names.insert(&phenotype.phenotype_name) || !output_names.insert(&phenotype.output_directory_name) {
                return Err(OutputError::InvalidInput(
                    "Output lineage terminal contains duplicate phenotype bindings.".to_string(),
                ));
            }
            validate_sha256(&phenotype.run_manifest_sha256, "run manifest")?;
        }
        Ok(())
    }
}

impl OutputLineagePaths {
    pub(crate) fn new(output_root: &Path) -> Self {
        let control_directory = output_root.join(".g-output");
        Self {
            output_root: output_root.to_path_buf(),
            successors_directory: control_directory.join("successors"),
            terminals_directory: control_directory.join("terminals"),
            attempts_directory: output_root.join("attempts"),
            genesis_path: control_directory.join(GENESIS_FILE_NAME),
            control_directory,
        }
    }

    pub(crate) fn initialize_directories(&self) -> OutputResult<()> {
        create_directories_durable(&self.output_root)?;
        create_directories_durable(&self.control_directory)?;
        create_directories_durable(&self.successors_directory)?;
        create_directories_durable(&self.terminals_directory)?;
        create_directories_durable(&self.attempts_directory)
    }

    pub(crate) fn attempt_directory(&self, attempt_id: &AttemptIdentifier) -> PathBuf {
        self.attempts_directory.join(attempt_id.as_str())
    }

    pub(crate) fn successor_path(&self, parent_attempt_id: &AttemptIdentifier) -> PathBuf {
        self.successors_directory.join(format!("{}.json", parent_attempt_id.as_str()))
    }

    pub(crate) fn terminal_path(&self, attempt_id: &AttemptIdentifier) -> PathBuf {
        self.terminals_directory.join(format!("{}.json", attempt_id.as_str()))
    }

    pub(crate) fn inspect(&self) -> OutputResult<Option<LineageSnapshot>> {
        let Some(genesis) = read_optional_json::<LineageGenesisRecord>(&self.genesis_path)? else {
            return Ok(None);
        };
        genesis.validate()?;
        let mut successor_records = Vec::new();
        let mut visited_attempts = BTreeSet::from([genesis.attempt_id.clone()]);
        let mut leaf_attempt_id = genesis.attempt_id.clone();
        loop {
            let successor_path = self.successor_path(&leaf_attempt_id);
            let Some(successor) = read_optional_json::<LineageSuccessorRecord>(&successor_path)? else {
                break;
            };
            successor.validate()?;
            if successor.run_set_id != genesis.run_set_id || successor.parent_attempt_id != leaf_attempt_id {
                return Err(OutputError::InvalidInput(format!(
                    "Output lineage successor '{}' is not bound to its traversed parent and run set.",
                    successor_path.display()
                )));
            }
            if !visited_attempts.insert(successor.attempt_id.clone()) {
                return Err(OutputError::InvalidInput("Output lineage successor chain contains a cycle.".to_string()));
            }
            if successor.recovery_kind == LineageRecoveryKind::TerminalResume {
                let parent_terminal_path = self.terminal_path(&successor.parent_attempt_id);
                let observed_terminal_sha256 = file_sha256(&parent_terminal_path)?;
                if successor.parent_terminal_sha256.as_deref() != Some(observed_terminal_sha256.as_str()) {
                    return Err(OutputError::InvalidInput(format!(
                        "Output lineage successor '{}' has a stale parent terminal binding.",
                        successor_path.display()
                    )));
                }
            }
            leaf_attempt_id = successor.attempt_id.clone();
            successor_records.push(successor);
        }
        for attempt_id in &visited_attempts {
            let attempt_directory = self.attempt_directory(attempt_id);
            if !attempt_directory.is_dir() {
                return Err(OutputError::InvalidInput(format!(
                    "Output lineage references missing attempt directory '{}'.",
                    attempt_directory.display()
                )));
            }
        }
        let leaf_terminal = read_optional_json::<LineageTerminalRecord>(&self.terminal_path(&leaf_attempt_id))?;
        if let Some(terminal) = &leaf_terminal {
            terminal.validate()?;
            if terminal.run_set_id != genesis.run_set_id || terminal.attempt_id != leaf_attempt_id {
                return Err(OutputError::InvalidInput(
                    "Output lineage leaf terminal is not bound to the leaf attempt and run set.".to_string(),
                ));
            }
        }
        Ok(Some(LineageSnapshot { genesis, successor_records, leaf_attempt_id, leaf_terminal }))
    }

    pub(crate) fn publish_genesis(&self, genesis: &LineageGenesisRecord) -> OutputResult<NoReplacePublication> {
        genesis.validate()?;
        self.publish_record(&self.genesis_path, genesis)
    }

    pub(crate) fn publish_successor(&self, successor: &LineageSuccessorRecord) -> OutputResult<NoReplacePublication> {
        successor.validate()?;
        self.publish_record(&self.successor_path(&successor.parent_attempt_id), successor)
    }

    pub(crate) fn publish_terminal(&self, terminal: &LineageTerminalRecord) -> OutputResult<NoReplacePublication> {
        terminal.validate()?;
        self.publish_record(&self.terminal_path(&terminal.attempt_id), terminal)
    }

    fn publish_record<RecordType>(&self, path: &Path, record: &RecordType) -> OutputResult<NoReplacePublication>
    where
        RecordType: Serialize + for<'deserialize> Deserialize<'deserialize> + Eq,
    {
        let outcome = publish_json_no_replace(path, record)?;
        if outcome == NoReplacePublication::AlreadyExists {
            let existing = read_required_json::<RecordType>(path)?;
            if existing != *record {
                return Err(OutputError::ConcurrentLineageUpdate { record_path: path.to_path_buf() });
            }
        }
        Ok(outcome)
    }
}

pub(crate) fn terminal_record_sha256(
    paths: &OutputLineagePaths,
    attempt_id: &AttemptIdentifier,
) -> OutputResult<String> {
    file_sha256(&paths.terminal_path(attempt_id))
}

fn validate_phenotype_contracts(phenotypes: &[PhenotypeLineageContract]) -> OutputResult<()> {
    if phenotypes.is_empty() {
        return Err(OutputError::InvalidInput("Output lineage genesis must bind at least one phenotype.".to_string()));
    }
    let mut names = BTreeSet::new();
    let mut output_names = BTreeSet::new();
    for phenotype in phenotypes {
        if phenotype.phenotype_name.is_empty() || phenotype.output_directory_name.is_empty() {
            return Err(OutputError::InvalidInput(
                "Output lineage phenotype names and output directory names must not be empty.".to_string(),
            ));
        }
        if !names.insert(&phenotype.phenotype_name) || !output_names.insert(&phenotype.output_directory_name) {
            return Err(OutputError::InvalidInput(
                "Output lineage genesis contains duplicate phenotype bindings.".to_string(),
            ));
        }
        validate_sha256(&phenotype.execution_plan_sha256, "execution plan")?;
    }
    Ok(())
}

fn validate_sha256(digest: &str, role: &str) -> OutputResult<()> {
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(OutputError::InvalidInput(format!(
            "Output {role} SHA-256 must contain exactly 64 hexadecimal characters."
        )));
    }
    Ok(())
}

fn read_optional_json<ValueType>(path: &Path) -> OutputResult<Option<ValueType>>
where
    ValueType: for<'deserialize> Deserialize<'deserialize>,
{
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|error| {
            OutputError::InvalidInput(format!("Output record '{}' is invalid JSON: {error}", path.display()))
        }),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(OutputError::Runtime(format!("Failed to read output record '{}': {error}", path.display()))),
    }
}

fn read_required_json<ValueType>(path: &Path) -> OutputResult<ValueType>
where
    ValueType: for<'deserialize> Deserialize<'deserialize>,
{
    read_optional_json(path)?.ok_or_else(|| {
        OutputError::Runtime(format!("Output record '{}' disappeared during publication.", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn phenotype_contract() -> PhenotypeLineageContract {
        PhenotypeLineageContract {
            phenotype_name: "trait-a".to_string(),
            output_directory_name: "trait_0001_trait-a".to_string(),
            execution_plan_sha256: digest('a'),
        }
    }

    fn test_paths(label: &str) -> OutputLineagePaths {
        let output_root = std::env::temp_dir().join(format!(
            "g-output-lineage-{label}-{}-{}",
            std::process::id(),
            AttemptIdentifier::generate().as_str()
        ));
        OutputLineagePaths::new(&output_root)
    }

    #[test]
    fn lineage_traverses_immutable_successors_without_head() {
        let paths = test_paths("traverse");
        paths.initialize_directories().expect("lineage directories initialize");
        let first_attempt = AttemptIdentifier::for_test("attempt-first");
        let second_attempt = AttemptIdentifier::for_test("attempt-second");
        create_directories_durable(&paths.attempt_directory(&first_attempt)).expect("first attempt creates");
        let genesis = LineageGenesisRecord::new(first_attempt.clone(), digest('b'), vec![phenotype_contract()]);
        paths.publish_genesis(&genesis).expect("genesis publishes");
        let interrupted = LineageTerminalRecord::interrupted(
            genesis.run_set_id.clone(),
            first_attempt.clone(),
            "SIGTERM".to_string(),
            vec![TerminalPhenotypeRecord {
                phenotype_name: "trait-a".to_string(),
                output_directory_name: "trait_0001_trait-a".to_string(),
                run_manifest_sha256: digest('c'),
            }],
        );
        paths.publish_terminal(&interrupted).expect("terminal publishes");
        create_directories_durable(&paths.attempt_directory(&second_attempt)).expect("second attempt creates");
        let successor = LineageSuccessorRecord::new(
            genesis.run_set_id.clone(),
            first_attempt.clone(),
            second_attempt.clone(),
            LineageRecoveryKind::TerminalResume,
            Some(terminal_record_sha256(&paths, &first_attempt).expect("terminal hashes")),
        )
        .expect("successor builds");
        paths.publish_successor(&successor).expect("successor publishes");

        let snapshot = paths.inspect().expect("lineage inspects").expect("lineage exists");
        assert_eq!(snapshot.genesis, genesis);
        assert_eq!(snapshot.successor_records, [successor]);
        assert_eq!(snapshot.leaf_attempt_id, second_attempt);
        assert_eq!(snapshot.leaf_terminal, None);
    }

    #[test]
    fn competing_successors_have_exactly_one_winner() {
        let paths = test_paths("race");
        paths.initialize_directories().expect("lineage directories initialize");
        let parent_attempt = AttemptIdentifier::for_test("attempt-parent");
        let run_set_id = "run-set-test".to_string();
        create_directories_durable(&paths.attempt_directory(&parent_attempt)).expect("parent attempt creates");
        let winner_count = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for contender in 0..8 {
                let contender_attempt = AttemptIdentifier::for_test(&format!("attempt-contender-{contender}"));
                create_directories_durable(&paths.attempt_directory(&contender_attempt))
                    .expect("candidate attempt creates");
                let successor = LineageSuccessorRecord::new(
                    run_set_id.clone(),
                    parent_attempt.clone(),
                    contender_attempt,
                    LineageRecoveryKind::ExactNonterminalRecovery,
                    None,
                )
                .expect("successor builds");
                let winner_count = &winner_count;
                let paths = &paths;
                scope.spawn(move || match paths.publish_successor(&successor) {
                    Ok(NoReplacePublication::Created) => {
                        winner_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Err(OutputError::ConcurrentLineageUpdate { .. }) => {}
                    outcome => panic!("unexpected race outcome: {outcome:?}"),
                });
            }
        });
        assert_eq!(winner_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn successor_detects_corrupt_parent_terminal_binding() {
        let paths = test_paths("terminal-binding");
        paths.initialize_directories().expect("lineage directories initialize");
        let first_attempt = AttemptIdentifier::for_test("attempt-first");
        let second_attempt = AttemptIdentifier::for_test("attempt-second");
        create_directories_durable(&paths.attempt_directory(&first_attempt)).expect("first attempt creates");
        create_directories_durable(&paths.attempt_directory(&second_attempt)).expect("second attempt creates");
        let genesis = LineageGenesisRecord::new(first_attempt.clone(), digest('b'), vec![phenotype_contract()]);
        paths.publish_genesis(&genesis).expect("genesis publishes");
        let interrupted = LineageTerminalRecord::interrupted(
            genesis.run_set_id.clone(),
            first_attempt.clone(),
            "SIGTERM".to_string(),
            vec![TerminalPhenotypeRecord {
                phenotype_name: "trait-a".to_string(),
                output_directory_name: "trait_0001_trait-a".to_string(),
                run_manifest_sha256: digest('c'),
            }],
        );
        paths.publish_terminal(&interrupted).expect("terminal publishes");
        let successor = LineageSuccessorRecord::new(
            genesis.run_set_id.clone(),
            first_attempt.clone(),
            second_attempt,
            LineageRecoveryKind::TerminalResume,
            Some(digest('d')),
        )
        .expect("successor builds");
        paths.publish_successor(&successor).expect("successor publishes");

        let error = paths.inspect().expect_err("stale terminal binding is rejected");
        assert!(error.to_string().contains("stale parent terminal binding"));
    }

    #[test]
    fn terminal_status_constructors_bind_status_specific_details() {
        let phenotype = TerminalPhenotypeRecord {
            phenotype_name: "trait-a".to_string(),
            output_directory_name: "trait_0001_trait-a".to_string(),
            run_manifest_sha256: digest('c'),
        };
        let completed = LineageTerminalRecord::completed(
            "run-set-test".to_string(),
            AttemptIdentifier::for_test("attempt-completed"),
            vec![phenotype.clone()],
        );
        completed.validate().expect("completed terminal validates");
        assert_eq!(completed.status, AttemptTerminalStatus::Completed);
        let failed = LineageTerminalRecord::failed(
            "run-set-test".to_string(),
            AttemptIdentifier::for_test("attempt-failed"),
            "delivery failed".to_string(),
            vec![phenotype],
        );
        failed.validate().expect("failed terminal validates");
        assert_eq!(failed.status, AttemptTerminalStatus::Failed);
    }
}
