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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "outcome_kind", content = "record", rename_all = "snake_case")]
enum AttemptOutcomeRecord {
    Terminal(LineageTerminalRecord),
    ExactRecoveryClaim(LineageSuccessorRecord),
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
    pub(crate) outcomes_directory: PathBuf,
    pub(crate) successors_directory: PathBuf,
    pub(crate) attempts_directory: PathBuf,
    pub(crate) genesis_path: PathBuf,
    legacy_terminals_directory: PathBuf,
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

impl AttemptOutcomeRecord {
    fn validate(&self) -> OutputResult<()> {
        match self {
            Self::Terminal(terminal) => terminal.validate(),
            Self::ExactRecoveryClaim(successor) => {
                successor.validate()?;
                if successor.recovery_kind != LineageRecoveryKind::ExactNonterminalRecovery {
                    return Err(OutputError::InvalidInput(
                        "Output exact-recovery outcome contains a non-exact successor.".to_string(),
                    ));
                }
                Ok(())
            }
        }
    }

    fn attempt_id(&self) -> &AttemptIdentifier {
        match self {
            Self::Terminal(terminal) => &terminal.attempt_id,
            Self::ExactRecoveryClaim(successor) => &successor.parent_attempt_id,
        }
    }

    fn run_set_id(&self) -> &str {
        match self {
            Self::Terminal(terminal) => &terminal.run_set_id,
            Self::ExactRecoveryClaim(successor) => &successor.run_set_id,
        }
    }
}

impl OutputLineagePaths {
    pub(crate) fn new(output_root: &Path) -> Self {
        let control_directory = output_root.join(".g-output");
        Self {
            output_root: output_root.to_path_buf(),
            outcomes_directory: control_directory.join("outcomes"),
            successors_directory: control_directory.join("successors"),
            attempts_directory: output_root.join("attempts"),
            genesis_path: control_directory.join(GENESIS_FILE_NAME),
            legacy_terminals_directory: control_directory.join("terminals"),
            control_directory,
        }
    }

    pub(crate) fn initialize_directories(&self) -> OutputResult<()> {
        create_directories_durable(&self.output_root)?;
        create_directories_durable(&self.control_directory)?;
        create_directories_durable(&self.outcomes_directory)?;
        create_directories_durable(&self.successors_directory)?;
        create_directories_durable(&self.attempts_directory)
    }

    pub(crate) fn attempt_directory(&self, attempt_id: &AttemptIdentifier) -> PathBuf {
        self.attempts_directory.join(attempt_id.as_str())
    }

    pub(crate) fn normal_successor_path(&self, parent_attempt_id: &AttemptIdentifier) -> PathBuf {
        self.successors_directory.join(format!("{}.json", parent_attempt_id.as_str()))
    }

    pub(crate) fn outcome_path(&self, attempt_id: &AttemptIdentifier) -> PathBuf {
        self.outcomes_directory.join(format!("{}.json", attempt_id.as_str()))
    }

    pub(crate) fn inspect(&self) -> OutputResult<Option<LineageSnapshot>> {
        let Some(genesis) = read_optional_json::<LineageGenesisRecord>(&self.genesis_path)? else {
            return Ok(None);
        };
        genesis.validate()?;
        let mut successor_records = Vec::new();
        let mut visited_attempts = BTreeSet::from([genesis.attempt_id.clone()]);
        let mut leaf_attempt_id = genesis.attempt_id.clone();
        let mut leaf_terminal = None;
        loop {
            self.reject_legacy_terminal(&leaf_attempt_id)?;
            let outcome_path = self.outcome_path(&leaf_attempt_id);
            let outcome = read_optional_json::<AttemptOutcomeRecord>(&outcome_path)?;
            if let Some(outcome) = &outcome {
                outcome.validate()?;
                if outcome.run_set_id() != genesis.run_set_id || outcome.attempt_id() != &leaf_attempt_id {
                    return Err(OutputError::InvalidInput(format!(
                        "Output attempt outcome '{}' is not bound to its traversed attempt and run set.",
                        outcome_path.display()
                    )));
                }
            }
            let normal_successor_path = self.normal_successor_path(&leaf_attempt_id);
            let normal_successor = read_optional_json::<LineageSuccessorRecord>(&normal_successor_path)?;
            let successor = match (outcome, normal_successor) {
                (None, None) => break,
                (None, Some(_)) => {
                    return Err(OutputError::InvalidInput(format!(
                        "Output lineage successor '{}' has no immutable parent terminal outcome.",
                        normal_successor_path.display()
                    )));
                }
                (Some(AttemptOutcomeRecord::ExactRecoveryClaim(successor)), None) => successor,
                (Some(AttemptOutcomeRecord::ExactRecoveryClaim(_)), Some(_)) => {
                    return Err(OutputError::InvalidInput(format!(
                        "Output exact-recovery attempt '{}' also has an incompatible normal successor record.",
                        leaf_attempt_id.as_str()
                    )));
                }
                (Some(AttemptOutcomeRecord::Terminal(terminal)), None) => {
                    leaf_terminal = Some(terminal);
                    break;
                }
                (Some(AttemptOutcomeRecord::Terminal(terminal)), Some(successor)) => {
                    if terminal.status == AttemptTerminalStatus::Completed {
                        return Err(OutputError::InvalidInput(format!(
                            "Completed output attempt '{}' cannot have a successor.",
                            leaf_attempt_id.as_str()
                        )));
                    }
                    successor.validate()?;
                    if successor.recovery_kind != LineageRecoveryKind::TerminalResume {
                        return Err(OutputError::InvalidInput(format!(
                            "Output lineage successor '{}' is not a terminal-resume successor.",
                            normal_successor_path.display()
                        )));
                    }
                    let observed_terminal_sha256 = file_sha256(&outcome_path)?;
                    if successor.parent_terminal_sha256.as_deref() != Some(observed_terminal_sha256.as_str()) {
                        return Err(OutputError::InvalidInput(format!(
                            "Output lineage successor '{}' has a stale parent terminal binding.",
                            normal_successor_path.display()
                        )));
                    }
                    successor
                }
            };
            if successor.run_set_id != genesis.run_set_id || successor.parent_attempt_id != leaf_attempt_id {
                return Err(OutputError::InvalidInput(format!(
                    "Output lineage successor '{}' is not bound to its traversed parent and run set.",
                    normal_successor_path.display()
                )));
            }
            if !visited_attempts.insert(successor.attempt_id.clone()) {
                return Err(OutputError::InvalidInput("Output lineage successor chain contains a cycle.".to_string()));
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
        Ok(Some(LineageSnapshot { genesis, successor_records, leaf_attempt_id, leaf_terminal }))
    }

    pub(crate) fn publish_genesis(&self, genesis: &LineageGenesisRecord) -> OutputResult<NoReplacePublication> {
        genesis.validate()?;
        self.publish_record(&self.genesis_path, genesis)
    }

    pub(crate) fn publish_successor(&self, successor: &LineageSuccessorRecord) -> OutputResult<NoReplacePublication> {
        successor.validate()?;
        match successor.recovery_kind {
            LineageRecoveryKind::ExactNonterminalRecovery => {
                let outcome = AttemptOutcomeRecord::ExactRecoveryClaim(successor.clone());
                self.publish_record(&self.outcome_path(&successor.parent_attempt_id), &outcome)
            }
            LineageRecoveryKind::TerminalResume => {
                let outcome_path = self.outcome_path(&successor.parent_attempt_id);
                let outcome = read_required_json::<AttemptOutcomeRecord>(&outcome_path)?;
                outcome.validate()?;
                let AttemptOutcomeRecord::Terminal(terminal) = outcome else {
                    return Err(OutputError::InvalidInput(format!(
                        "Output terminal-resume successor cannot follow nonterminal attempt '{}'.",
                        successor.parent_attempt_id.as_str()
                    )));
                };
                if terminal.status == AttemptTerminalStatus::Completed {
                    return Err(OutputError::InvalidInput(format!(
                        "Completed output attempt '{}' cannot have a successor.",
                        successor.parent_attempt_id.as_str()
                    )));
                }
                let observed_terminal_sha256 = file_sha256(&outcome_path)?;
                if terminal.run_set_id != successor.run_set_id
                    || terminal.attempt_id != successor.parent_attempt_id
                    || successor.parent_terminal_sha256.as_deref() != Some(observed_terminal_sha256.as_str())
                {
                    return Err(OutputError::InvalidInput(
                        "Output terminal-resume successor has a stale or mismatched parent terminal binding."
                            .to_string(),
                    ));
                }
                self.publish_record(&self.normal_successor_path(&successor.parent_attempt_id), successor)
            }
        }
    }

    pub(crate) fn publish_terminal(&self, terminal: &LineageTerminalRecord) -> OutputResult<NoReplacePublication> {
        terminal.validate()?;
        let outcome = AttemptOutcomeRecord::Terminal(terminal.clone());
        self.publish_record(&self.outcome_path(&terminal.attempt_id), &outcome)
    }

    fn reject_legacy_terminal(&self, attempt_id: &AttemptIdentifier) -> OutputResult<()> {
        let legacy_terminal_path = self.legacy_terminals_directory.join(format!("{}.json", attempt_id.as_str()));
        if legacy_terminal_path.try_exists().map_err(|error| {
            OutputError::Runtime(format!(
                "Failed to inspect legacy output terminal '{}': {error}",
                legacy_terminal_path.display()
            ))
        })? {
            return Err(OutputError::InvalidInput(format!(
                "Output lineage contains unsupported legacy terminal artifact '{}'.",
                legacy_terminal_path.display()
            )));
        }
        Ok(())
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
    let outcome_path = paths.outcome_path(attempt_id);
    let outcome = read_required_json::<AttemptOutcomeRecord>(&outcome_path)?;
    outcome.validate()?;
    if !matches!(outcome, AttemptOutcomeRecord::Terminal(_)) {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt '{}' has no terminal outcome to hash.",
            attempt_id.as_str()
        )));
    }
    file_sha256(&outcome_path)
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
    fn terminal_and_exact_recovery_claim_compete_for_one_outcome() {
        let paths = test_paths("terminal-exact-race");
        paths.initialize_directories().expect("lineage directories initialize");
        let parent_attempt = AttemptIdentifier::for_test("attempt-parent");
        let child_attempt = AttemptIdentifier::for_test("attempt-child");
        let run_set_id = "run-set-test".to_string();
        create_directories_durable(&paths.attempt_directory(&parent_attempt)).expect("parent attempt creates");
        create_directories_durable(&paths.attempt_directory(&child_attempt)).expect("child attempt creates");
        let terminal = LineageTerminalRecord::failed(
            run_set_id.clone(),
            parent_attempt.clone(),
            "writer failed".to_string(),
            vec![TerminalPhenotypeRecord {
                phenotype_name: "trait-a".to_string(),
                output_directory_name: "trait_0001_trait-a".to_string(),
                run_manifest_sha256: digest('c'),
            }],
        );
        let successor = LineageSuccessorRecord::new(
            run_set_id,
            parent_attempt,
            child_attempt,
            LineageRecoveryKind::ExactNonterminalRecovery,
            None,
        )
        .expect("successor builds");
        let winner_count = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let terminal_winner_count = &winner_count;
            let terminal_paths = &paths;
            scope.spawn(move || match terminal_paths.publish_terminal(&terminal) {
                Ok(NoReplacePublication::Created) => {
                    terminal_winner_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Err(OutputError::ConcurrentLineageUpdate { .. }) => {}
                outcome => panic!("unexpected terminal race outcome: {outcome:?}"),
            });
            let exact_winner_count = &winner_count;
            let exact_paths = &paths;
            scope.spawn(move || match exact_paths.publish_successor(&successor) {
                Ok(NoReplacePublication::Created) => {
                    exact_winner_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Err(OutputError::ConcurrentLineageUpdate { .. }) => {}
                outcome => panic!("unexpected exact-recovery race outcome: {outcome:?}"),
            });
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
        publish_json_no_replace(&paths.normal_successor_path(&first_attempt), &successor)
            .expect("corrupt successor artifact publishes");

        let error = paths.inspect().expect_err("stale terminal binding is rejected");
        assert!(error.to_string().contains("stale parent terminal binding"));
    }

    #[test]
    fn exact_recovery_outcome_rejects_incompatible_normal_successor_artifact() {
        let paths = test_paths("dual-transition");
        paths.initialize_directories().expect("lineage directories initialize");
        let first_attempt = AttemptIdentifier::for_test("attempt-first");
        let exact_child = AttemptIdentifier::for_test("attempt-exact-child");
        let normal_child = AttemptIdentifier::for_test("attempt-normal-child");
        create_directories_durable(&paths.attempt_directory(&first_attempt)).expect("first attempt creates");
        create_directories_durable(&paths.attempt_directory(&exact_child)).expect("exact child creates");
        create_directories_durable(&paths.attempt_directory(&normal_child)).expect("normal child creates");
        let genesis = LineageGenesisRecord::new(first_attempt.clone(), digest('b'), vec![phenotype_contract()]);
        paths.publish_genesis(&genesis).expect("genesis publishes");
        let exact_successor = LineageSuccessorRecord::new(
            genesis.run_set_id.clone(),
            first_attempt.clone(),
            exact_child,
            LineageRecoveryKind::ExactNonterminalRecovery,
            None,
        )
        .expect("exact successor builds");
        paths.publish_successor(&exact_successor).expect("exact successor publishes");
        let incompatible_successor = LineageSuccessorRecord::new(
            genesis.run_set_id,
            first_attempt.clone(),
            normal_child,
            LineageRecoveryKind::TerminalResume,
            Some(digest('d')),
        )
        .expect("normal successor builds");
        publish_json_no_replace(&paths.normal_successor_path(&first_attempt), &incompatible_successor)
            .expect("incompatible successor artifact publishes");

        let error = paths.inspect().expect_err("dual transition state is rejected");
        assert!(error.to_string().contains("incompatible normal successor"));
    }

    #[test]
    fn completed_outcome_rejects_forged_successor_artifact() {
        let paths = test_paths("completed-successor");
        paths.initialize_directories().expect("lineage directories initialize");
        let completed_attempt = AttemptIdentifier::for_test("attempt-completed");
        let child_attempt = AttemptIdentifier::for_test("attempt-child");
        create_directories_durable(&paths.attempt_directory(&completed_attempt)).expect("completed attempt creates");
        create_directories_durable(&paths.attempt_directory(&child_attempt)).expect("child attempt creates");
        let genesis = LineageGenesisRecord::new(completed_attempt.clone(), digest('b'), vec![phenotype_contract()]);
        paths.publish_genesis(&genesis).expect("genesis publishes");
        let completed = LineageTerminalRecord::completed(
            genesis.run_set_id.clone(),
            completed_attempt.clone(),
            vec![TerminalPhenotypeRecord {
                phenotype_name: "trait-a".to_string(),
                output_directory_name: "trait_0001_trait-a".to_string(),
                run_manifest_sha256: digest('c'),
            }],
        );
        paths.publish_terminal(&completed).expect("completed outcome publishes");
        let forged_successor = LineageSuccessorRecord::new(
            genesis.run_set_id,
            completed_attempt.clone(),
            child_attempt,
            LineageRecoveryKind::TerminalResume,
            Some(terminal_record_sha256(&paths, &completed_attempt).expect("completed outcome hashes")),
        )
        .expect("forged successor builds");
        let publish_error =
            paths.publish_successor(&forged_successor).expect_err("completed outcome cannot publish a successor");
        assert!(publish_error.to_string().contains("Completed output attempt"));
        publish_json_no_replace(&paths.normal_successor_path(&completed_attempt), &forged_successor)
            .expect("forged successor artifact publishes");

        let error = paths.inspect().expect_err("completed successor is rejected");
        assert!(error.to_string().contains("Completed output attempt"));
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
