use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{OutputError, OutputResult};
use crate::persistence::identifier::{
    AttemptIdentifier, generate_owner_claim_identifier, generate_run_set_identifier, validate_owner_claim_identifier,
    validate_run_set_identifier, validate_safe_path_component,
};
use crate::persistence::io::{
    NoReplacePublication, create_directories_durable, file_sha256, publish_json_no_replace, sync_directory,
    sync_immutable_publication_directory, sync_nearest_existing_directory,
};

pub(crate) const LINEAGE_SCHEMA_VERSION: u32 = 0;
const GENESIS_FILE_NAME: &str = "genesis.json";
const OWNER_CLAIM_FILE_NAME: &str = "session.claim.json";
// A few thousand transitions permit years of repeated resumes while bounding
// corrupt-chain traversal work before any output mutation.
const MAXIMUM_OWNER_TRANSITION_COUNT: usize = 4_096;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct OutputOwnerClaimRecord {
    schema_version: u32,
    claim_id: String,
    host_name: String,
    process_id: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "transition_kind", rename_all = "snake_case", deny_unknown_fields)]
enum OutputOwnerTransitionRecord {
    GracefulRelease { schema_version: u32, predecessor_claim_id: String, released_state_id: String },
    FencedTakeover { schema_version: u32, predecessor_claim_id: String, claim: OutputOwnerClaimRecord },
    AcquireAfterRelease { schema_version: u32, predecessor_released_state_id: String, claim: OutputOwnerClaimRecord },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct OutputOwnerStagingRecord {
    schema_version: u32,
    claim_id: String,
    attempt_id: AttemptIdentifier,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum OutputOwnerAuthorityState {
    Active(OutputOwnerClaimRecord),
    Released { released_state_id: String },
}

struct ResolvedOutputOwnerAuthority {
    state: OutputOwnerAuthorityState,
    claim_identifiers: BTreeSet<String>,
    #[cfg(test)]
    transition_predecessors: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PhenotypeLineageContract {
    pub(crate) phenotype_name: String,
    pub(crate) output_directory_name: String,
    pub(crate) execution_plan_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LineageGenesisRecord {
    record_kind: LineageRecordKind,
    schema_version: u32,
    pub(crate) run_set_id: String,
    pub(crate) attempt_id: AttemptIdentifier,
    pub(crate) chunk_plan_sha256: String,
    pub(crate) phenotypes: Vec<PhenotypeLineageContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub(crate) struct TerminalPhenotypeRecord {
    pub(crate) phenotype_name: String,
    pub(crate) output_directory_name: String,
    pub(crate) run_manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
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
#[serde(tag = "outcome_kind", content = "record", rename_all = "snake_case", deny_unknown_fields)]
enum AttemptOutcomeRecord {
    TerminalClaim(LineageTerminalRecord),
    ExactRecoveryClaim(LineageSuccessorRecord),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct LineageTerminalFinalizationRecord {
    record_kind: LineageRecordKind,
    schema_version: u32,
    run_set_id: String,
    attempt_id: AttemptIdentifier,
    terminal_claim_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum LineageRecordKind {
    Genesis,
    Successor,
    Terminal,
    TerminalFinalization,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LineageSnapshot {
    pub(crate) genesis: LineageGenesisRecord,
    pub(crate) successor_records: Vec<LineageSuccessorRecord>,
    pub(crate) leaf_attempt_id: AttemptIdentifier,
    pub(crate) leaf_terminal: Option<LineageTerminalRecord>,
    pub(crate) pending_terminal: Option<LineageTerminalRecord>,
}

struct LineageTraversal {
    successor_records: Vec<LineageSuccessorRecord>,
    visited_attempts: BTreeSet<AttemptIdentifier>,
    leaf_attempt_id: AttemptIdentifier,
    leaf_terminal: Option<LineageTerminalRecord>,
    pending_terminal: Option<LineageTerminalRecord>,
}

enum InspectedLineageLeaf {
    Active,
    Terminal { record: LineageTerminalRecord, finalized: bool },
    Successor(LineageSuccessorRecord),
}

#[derive(Clone, Debug)]
pub(crate) struct OutputLineagePaths {
    pub(crate) output_root: PathBuf,
    pub(crate) control_directory: PathBuf,
    pub(crate) outcomes_directory: PathBuf,
    pub(crate) successors_directory: PathBuf,
    terminal_finalizations_directory: PathBuf,
    pub(crate) attempts_directory: PathBuf,
    pub(crate) genesis_path: PathBuf,
    pub(crate) owner_claim_path: PathBuf,
    owner_transitions_directory: PathBuf,
    owner_staging_directory: PathBuf,
    legacy_terminals_directory: PathBuf,
}

#[derive(Debug)]
pub(crate) struct OutputOwnerClaim {
    claim_path: PathBuf,
    owner_transitions_directory: PathBuf,
    record: OutputOwnerClaimRecord,
    release_record: OutputOwnerTransitionRecord,
    release_state: OutputOwnerClaimReleaseState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputOwnerClaimReleaseState {
    Published,
    TransitionPublished,
    DurablyReleased,
}

impl OutputOwnerClaimRecord {
    fn new() -> OutputResult<Self> {
        let record = Self {
            schema_version: LINEAGE_SCHEMA_VERSION,
            claim_id: generate_owner_claim_identifier(),
            host_name: current_host_name(),
            process_id: std::process::id(),
        };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> OutputResult<()> {
        if self.schema_version != LINEAGE_SCHEMA_VERSION {
            return Err(OutputError::InvalidInput("Output owner claim has an unsupported schema version.".to_string()));
        }
        validate_owner_claim_identifier(&self.claim_id)?;
        if self.host_name.trim().is_empty()
            || self.host_name.len() > 255
            || self.host_name.chars().any(char::is_control)
        {
            return Err(OutputError::InvalidInput(
                "Output owner claim host name must be non-empty, contain no control characters, and be at most 255 bytes."
                    .to_string(),
            ));
        }
        if self.process_id == 0 {
            return Err(OutputError::InvalidInput(
                "Output owner claim process identifier must be positive.".to_string(),
            ));
        }
        Ok(())
    }

    fn into_error(self, claim_path: PathBuf) -> OutputError {
        OutputError::SurvivingOutputOwnerClaim {
            claim_path,
            claim_id: self.claim_id,
            host_name: self.host_name,
            process_id: self.process_id,
        }
    }
}

impl OutputOwnerTransitionRecord {
    fn graceful_release(predecessor_claim_id: String) -> OutputResult<Self> {
        let record = Self::GracefulRelease {
            schema_version: LINEAGE_SCHEMA_VERSION,
            predecessor_claim_id,
            released_state_id: generate_owner_claim_identifier(),
        };
        record.validate()?;
        Ok(record)
    }

    fn fenced_takeover(predecessor_claim_id: String, claim: OutputOwnerClaimRecord) -> OutputResult<Self> {
        let record = Self::FencedTakeover { schema_version: LINEAGE_SCHEMA_VERSION, predecessor_claim_id, claim };
        record.validate()?;
        Ok(record)
    }

    fn acquire_after_release(
        predecessor_released_state_id: String,
        claim: OutputOwnerClaimRecord,
    ) -> OutputResult<Self> {
        let record =
            Self::AcquireAfterRelease { schema_version: LINEAGE_SCHEMA_VERSION, predecessor_released_state_id, claim };
        record.validate()?;
        Ok(record)
    }

    fn predecessor_state_id(&self) -> &str {
        match self {
            Self::GracefulRelease { predecessor_claim_id, .. } | Self::FencedTakeover { predecessor_claim_id, .. } => {
                predecessor_claim_id
            }
            Self::AcquireAfterRelease { predecessor_released_state_id, .. } => predecessor_released_state_id,
        }
    }

    fn validate(&self) -> OutputResult<()> {
        match self {
            Self::GracefulRelease { schema_version, predecessor_claim_id, released_state_id } => {
                validate_owner_transition_schema(*schema_version)?;
                validate_owner_claim_identifier(predecessor_claim_id)?;
                validate_owner_claim_identifier(released_state_id)?;
                if predecessor_claim_id == released_state_id {
                    return Err(OutputError::InvalidInput(
                        "Output owner release state must differ from its predecessor claim.".to_string(),
                    ));
                }
            }
            Self::FencedTakeover { schema_version, predecessor_claim_id, claim } => {
                validate_owner_transition_schema(*schema_version)?;
                validate_owner_claim_identifier(predecessor_claim_id)?;
                claim.validate()?;
                if predecessor_claim_id == &claim.claim_id {
                    return Err(OutputError::InvalidInput(
                        "Output fenced takeover claim must differ from its predecessor.".to_string(),
                    ));
                }
            }
            Self::AcquireAfterRelease { schema_version, predecessor_released_state_id, claim } => {
                validate_owner_transition_schema(*schema_version)?;
                validate_owner_claim_identifier(predecessor_released_state_id)?;
                claim.validate()?;
                if predecessor_released_state_id == &claim.claim_id {
                    return Err(OutputError::InvalidInput(
                        "Output owner claim must differ from its released predecessor state.".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl OutputOwnerStagingRecord {
    fn new(claim_id: String, attempt_id: AttemptIdentifier) -> OutputResult<Self> {
        let record = Self { schema_version: LINEAGE_SCHEMA_VERSION, claim_id, attempt_id };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> OutputResult<()> {
        validate_owner_transition_schema(self.schema_version)?;
        validate_owner_claim_identifier(&self.claim_id)?;
        AttemptIdentifier::parse(self.attempt_id.as_str())?;
        Ok(())
    }
}

impl OutputOwnerClaim {
    pub(crate) fn claim_id(&self) -> &str {
        &self.record.claim_id
    }

    pub(crate) fn claim_path(&self) -> &Path {
        &self.claim_path
    }

    pub(crate) fn release_transition_is_visible(&self) -> bool {
        self.release_state == OutputOwnerClaimReleaseState::TransitionPublished
    }

    #[cfg(test)]
    pub(crate) fn publish_conflicting_takeover_for_test(&self) -> OutputResult<()> {
        let competing_claim = OutputOwnerClaimRecord::new()?;
        let transition = OutputOwnerTransitionRecord::fenced_takeover(self.record.claim_id.clone(), competing_claim)?;
        match publish_owner_transition(&self.owner_transitions_directory, &transition)? {
            NoReplacePublication::Created => Ok(()),
            NoReplacePublication::AlreadyExists => Err(OutputError::ConcurrentLineageUpdate {
                record_path: owner_transition_path(&self.owner_transitions_directory, self.record.claim_id.as_str()),
            }),
        }
    }

    pub(crate) fn authority_failure(&self, reason: &impl ToString) -> OutputError {
        match resolve_owner_authority(&self.claim_path, &self.owner_transitions_directory) {
            Ok(OutputOwnerAuthorityState::Active(record)) if record == self.record => {
                OutputError::RetainedOutputOwnerClaimRelease {
                    claim_path: self.claim_path.clone(),
                    claim_id: self.record.claim_id.clone(),
                    reason: reason.to_string(),
                }
            }
            Ok(OutputOwnerAuthorityState::Active(record)) => record.into_error(self.claim_path.clone()),
            Ok(OutputOwnerAuthorityState::Released { .. }) => OutputError::Runtime(format!(
                "Output owner claim '{}' is no longer active after an ownership operation failed: {}",
                self.record.claim_id,
                reason.to_string()
            )),
            Err(resolution_error) => OutputError::Runtime(format!(
                "Output ownership operation failed ({}), and current authority could not be resolved: {resolution_error}",
                reason.to_string()
            )),
        }
    }

    pub(crate) fn release(&mut self) -> OutputResult<()> {
        if self.release_state == OutputOwnerClaimReleaseState::DurablyReleased {
            return Ok(());
        }
        let release_path = owner_transition_path(&self.owner_transitions_directory, self.record.claim_id.as_str());
        if let Some(existing_release) = read_optional_json::<OutputOwnerTransitionRecord>(&release_path)? {
            existing_release.validate()?;
            if existing_release != self.release_record {
                return Err(OutputError::ConcurrentLineageUpdate { record_path: release_path });
            }
            sync_immutable_publication_directory(&release_path, &self.owner_transitions_directory)?;
            self.release_state = OutputOwnerClaimReleaseState::DurablyReleased;
            return Ok(());
        }
        let observed = resolve_owner_authority(&self.claim_path, &self.owner_transitions_directory)?;
        if observed != OutputOwnerAuthorityState::Active(self.record.clone()) {
            return Err(OutputError::ConcurrentLineageUpdate { record_path: release_path });
        }
        let publication = match publish_owner_transition(&self.owner_transitions_directory, &self.release_record) {
            Ok(publication) => publication,
            Err(error) => {
                if read_optional_json::<OutputOwnerTransitionRecord>(&release_path)?
                    .is_some_and(|record| record == self.release_record)
                {
                    self.release_state = OutputOwnerClaimReleaseState::TransitionPublished;
                }
                return Err(error);
            }
        };
        if publication == NoReplacePublication::AlreadyExists {
            let existing_release = read_required_json::<OutputOwnerTransitionRecord>(&release_path)?;
            existing_release.validate()?;
            if existing_release != self.release_record {
                return Err(OutputError::ConcurrentLineageUpdate { record_path: release_path });
            }
        }
        self.release_state = OutputOwnerClaimReleaseState::DurablyReleased;
        Ok(())
    }
}

fn validate_owner_transition_schema(schema_version: u32) -> OutputResult<()> {
    if schema_version == LINEAGE_SCHEMA_VERSION {
        Ok(())
    } else {
        Err(OutputError::InvalidInput("Output owner transition has an unsupported schema version.".to_string()))
    }
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
        if self.interrupted_signal.as_deref().is_some_and(|signal| signal.trim().is_empty())
            || self.failure_reason.as_deref().is_some_and(|reason| reason.trim().is_empty())
        {
            return Err(OutputError::InvalidInput(
                "Output lineage terminal signal and failure details must not be empty.".to_string(),
            ));
        }
        if self.phenotypes.is_empty() {
            return Err(OutputError::InvalidInput(
                "Output lineage terminal must bind at least one phenotype manifest.".to_string(),
            ));
        }
        let mut names = BTreeSet::new();
        let mut output_names = BTreeSet::new();
        for phenotype in &self.phenotypes {
            if phenotype.phenotype_name.is_empty() {
                return Err(OutputError::InvalidInput(
                    "Output lineage terminal phenotype name must not be empty.".to_string(),
                ));
            }
            validate_safe_path_component(&phenotype.output_directory_name, "phenotype directory name")?;
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
            Self::TerminalClaim(terminal) => terminal.validate(),
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
            Self::TerminalClaim(terminal) => &terminal.attempt_id,
            Self::ExactRecoveryClaim(successor) => &successor.parent_attempt_id,
        }
    }

    fn run_set_id(&self) -> &str {
        match self {
            Self::TerminalClaim(terminal) => &terminal.run_set_id,
            Self::ExactRecoveryClaim(successor) => &successor.run_set_id,
        }
    }
}

impl LineageTerminalFinalizationRecord {
    fn new(terminal: &LineageTerminalRecord, terminal_claim_sha256: String) -> OutputResult<Self> {
        let record = Self {
            record_kind: LineageRecordKind::TerminalFinalization,
            schema_version: LINEAGE_SCHEMA_VERSION,
            run_set_id: terminal.run_set_id.clone(),
            attempt_id: terminal.attempt_id.clone(),
            terminal_claim_sha256,
        };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> OutputResult<()> {
        if self.record_kind != LineageRecordKind::TerminalFinalization || self.schema_version != LINEAGE_SCHEMA_VERSION
        {
            return Err(OutputError::InvalidInput(
                "Output terminal finalization has an unsupported record kind or schema version.".to_string(),
            ));
        }
        validate_run_set_identifier(&self.run_set_id)?;
        AttemptIdentifier::parse(self.attempt_id.as_str())?;
        validate_sha256(&self.terminal_claim_sha256, "terminal claim")
    }
}

impl OutputLineagePaths {
    pub(crate) fn new(output_root: &Path) -> Self {
        let control_directory = output_root.join(".g-output");
        Self {
            output_root: output_root.to_path_buf(),
            outcomes_directory: control_directory.join("outcomes"),
            successors_directory: control_directory.join("successors"),
            terminal_finalizations_directory: control_directory.join("terminal-finalizations"),
            attempts_directory: output_root.join("attempts"),
            genesis_path: control_directory.join(GENESIS_FILE_NAME),
            owner_claim_path: control_directory.join(OWNER_CLAIM_FILE_NAME),
            owner_transitions_directory: control_directory.join("owner-transitions"),
            owner_staging_directory: control_directory.join("owner-staging"),
            legacy_terminals_directory: control_directory.join("terminals"),
            control_directory,
        }
    }

    pub(crate) fn initialize_directories(&self) -> OutputResult<()> {
        create_directories_durable(&self.output_root)?;
        create_directories_durable(&self.control_directory)?;
        create_directories_durable(&self.outcomes_directory)?;
        create_directories_durable(&self.successors_directory)?;
        create_directories_durable(&self.terminal_finalizations_directory)?;
        create_directories_durable(&self.owner_transitions_directory)?;
        create_directories_durable(&self.owner_staging_directory)?;
        create_directories_durable(&self.attempts_directory)
    }

    pub(crate) fn reestablish_observed_directory_durability(&self) -> OutputResult<()> {
        for directory in [
            &self.owner_transitions_directory,
            &self.owner_staging_directory,
            &self.outcomes_directory,
            &self.successors_directory,
            &self.terminal_finalizations_directory,
            &self.attempts_directory,
            &self.control_directory,
            &self.output_root,
        ] {
            sync_existing_directory(directory)?;
        }
        let output_root_parent = self.output_root.parent().ok_or_else(|| {
            OutputError::InvalidInput(format!("Output root '{}' has no parent directory.", self.output_root.display()))
        })?;
        sync_existing_directory(output_root_parent)
    }

    #[cfg(test)]
    pub(crate) fn reject_surviving_owner_claim(&self) -> OutputResult<()> {
        if !self.owner_claim_path.try_exists().map_err(OutputError::runtime)? {
            return Ok(());
        }
        match resolve_owner_authority(&self.owner_claim_path, &self.owner_transitions_directory)? {
            OutputOwnerAuthorityState::Active(record) => Err(record.into_error(self.owner_claim_path.clone())),
            OutputOwnerAuthorityState::Released { .. } => Ok(()),
        }
    }

    #[cfg(test)]
    pub(crate) fn current_owner_claim_identifier_for_test(&self) -> OutputResult<Option<String>> {
        if !self.owner_claim_path.try_exists().map_err(OutputError::runtime)? {
            return Ok(None);
        }
        match resolve_owner_authority(&self.owner_claim_path, &self.owner_transitions_directory)? {
            OutputOwnerAuthorityState::Active(record) => Ok(Some(record.claim_id)),
            OutputOwnerAuthorityState::Released { .. } => Ok(None),
        }
    }

    pub(crate) fn take_over_fenced_owner_claim(&self, fenced_claim_id: &str) -> OutputResult<OutputOwnerClaim> {
        validate_owner_claim_identifier(fenced_claim_id)?;
        if !self.owner_claim_path.try_exists().map_err(OutputError::runtime)? {
            return Err(OutputError::InvalidInput(format!(
                "Output owner claim '{fenced_claim_id}' was declared fenced, but no owner authority exists at '{}'.",
                self.owner_claim_path.display()
            )));
        }
        let active = match resolve_owner_authority(&self.owner_claim_path, &self.owner_transitions_directory)? {
            OutputOwnerAuthorityState::Active(record) => record,
            OutputOwnerAuthorityState::Released { .. } => {
                return Err(OutputError::InvalidInput(format!(
                    "Output owner claim '{fenced_claim_id}' was declared fenced, but the owner authority is already released."
                )));
            }
        };
        if active.claim_id != fenced_claim_id {
            return Err(active.into_error(self.owner_claim_path.clone()));
        }
        let record = OutputOwnerClaimRecord::new()?;
        let transition = OutputOwnerTransitionRecord::fenced_takeover(active.claim_id, record.clone())?;
        match publish_owner_transition(&self.owner_transitions_directory, &transition) {
            Err(OutputError::ConcurrentLineageUpdate { .. }) => {
                let observed = resolve_owner_authority(&self.owner_claim_path, &self.owner_transitions_directory)?;
                match observed {
                    OutputOwnerAuthorityState::Active(observed_claim) => {
                        Err(observed_claim.into_error(self.owner_claim_path.clone()))
                    }
                    OutputOwnerAuthorityState::Released { .. } => Err(OutputError::ConcurrentLineageUpdate {
                        record_path: owner_transition_path(&self.owner_transitions_directory, fenced_claim_id),
                    }),
                }
            }
            Err(error) => Err(self.owner_claim_publication_error(&record, &error)),
            Ok(NoReplacePublication::Created) => self.owner_claim_from_record(record),
            Ok(NoReplacePublication::AlreadyExists) => {
                let observed = resolve_owner_authority(&self.owner_claim_path, &self.owner_transitions_directory)?;
                match observed {
                    OutputOwnerAuthorityState::Active(observed_claim) => {
                        Err(observed_claim.into_error(self.owner_claim_path.clone()))
                    }
                    OutputOwnerAuthorityState::Released { .. } => Err(OutputError::ConcurrentLineageUpdate {
                        record_path: owner_transition_path(&self.owner_transitions_directory, fenced_claim_id),
                    }),
                }
            }
        }
    }

    pub(crate) fn try_acquire_owner_claim(&self) -> OutputResult<OutputOwnerClaim> {
        let record = OutputOwnerClaimRecord::new()?;
        let root_exists = self.owner_claim_path.try_exists().map_err(OutputError::runtime)?;
        let root_publication_result = if root_exists {
            Ok(NoReplacePublication::AlreadyExists)
        } else {
            create_directories_durable(&self.output_root)?;
            create_directories_durable(&self.control_directory)?;
            publish_json_no_replace_reconciled(&self.owner_claim_path, &record)
        };
        let root_publication = match root_publication_result {
            Ok(publication) => publication,
            Err(OutputError::ConcurrentLineageUpdate { .. }) => NoReplacePublication::AlreadyExists,
            Err(error) => return Err(self.owner_claim_publication_error(&record, &error)),
        };
        match root_publication {
            NoReplacePublication::Created => self.owner_claim_from_record(record),
            NoReplacePublication::AlreadyExists => {
                match resolve_owner_authority(&self.owner_claim_path, &self.owner_transitions_directory)? {
                    OutputOwnerAuthorityState::Active(existing) => {
                        Err(existing.into_error(self.owner_claim_path.clone()))
                    }
                    OutputOwnerAuthorityState::Released { released_state_id } => {
                        let transition =
                            OutputOwnerTransitionRecord::acquire_after_release(released_state_id, record.clone())?;
                        match publish_owner_transition(&self.owner_transitions_directory, &transition) {
                            Ok(NoReplacePublication::Created) => self.owner_claim_from_record(record),
                            Ok(NoReplacePublication::AlreadyExists)
                            | Err(OutputError::ConcurrentLineageUpdate { .. }) => {
                                let observed =
                                    resolve_owner_authority(&self.owner_claim_path, &self.owner_transitions_directory)?;
                                match observed {
                                    OutputOwnerAuthorityState::Active(existing) => {
                                        Err(existing.into_error(self.owner_claim_path.clone()))
                                    }
                                    OutputOwnerAuthorityState::Released { .. } => {
                                        Err(OutputError::ConcurrentLineageUpdate {
                                            record_path: owner_transition_path(
                                                &self.owner_transitions_directory,
                                                transition.predecessor_state_id(),
                                            ),
                                        })
                                    }
                                }
                            }
                            Err(error) => Err(self.owner_claim_publication_error(&record, &error)),
                        }
                    }
                }
            }
        }
    }

    fn owner_claim_from_record(&self, record: OutputOwnerClaimRecord) -> OutputResult<OutputOwnerClaim> {
        let release_record = OutputOwnerTransitionRecord::graceful_release(record.claim_id.clone())?;
        Ok(OutputOwnerClaim {
            claim_path: self.owner_claim_path.clone(),
            owner_transitions_directory: self.owner_transitions_directory.clone(),
            record,
            release_record,
            release_state: OutputOwnerClaimReleaseState::Published,
        })
    }

    fn owner_claim_publication_error(
        &self,
        candidate: &OutputOwnerClaimRecord,
        publication_error: &OutputError,
    ) -> OutputError {
        match resolve_owner_authority(&self.owner_claim_path, &self.owner_transitions_directory) {
            Ok(OutputOwnerAuthorityState::Active(observed)) if observed == *candidate => {
                OutputError::PublishedOutputOwnerClaimDurability {
                    claim_path: self.owner_claim_path.clone(),
                    claim_id: candidate.claim_id.clone(),
                    reason: publication_error.to_string(),
                }
            }
            Ok(OutputOwnerAuthorityState::Active(observed)) => observed.into_error(self.owner_claim_path.clone()),
            Ok(OutputOwnerAuthorityState::Released { .. }) => OutputError::Runtime(format!(
                "Output owner publication failed ({publication_error}), and authority resolves to a released state."
            )),
            Err(resolution_error) => OutputError::Runtime(format!(
                "Output owner publication failed ({publication_error}), and visible authority could not be resolved: {resolution_error}"
            )),
        }
    }

    pub(crate) fn publish_owner_staging_intent(
        &self,
        claim_id: &str,
        attempt_id: &AttemptIdentifier,
    ) -> OutputResult<NoReplacePublication> {
        let record = OutputOwnerStagingRecord::new(claim_id.to_string(), attempt_id.clone())?;
        let staging_path = self.owner_staging_directory.join(format!("{claim_id}.json"));
        Self::publish_record(&staging_path, &record)
    }

    pub(crate) fn owner_staging_attempt(&self, claim_id: &str) -> OutputResult<Option<AttemptIdentifier>> {
        validate_owner_claim_identifier(claim_id)?;
        let staging_path = self.owner_staging_directory.join(format!("{claim_id}.json"));
        let Some(record) = read_optional_json::<OutputOwnerStagingRecord>(&staging_path)? else {
            return Ok(None);
        };
        record.validate()?;
        if record.claim_id != claim_id {
            return Err(OutputError::InvalidInput(format!(
                "Output owner staging record '{}' is not bound to its file name.",
                staging_path.display()
            )));
        }
        if !self.owner_claim_path.try_exists().map_err(OutputError::runtime)?
            || !resolve_owner_authority_with_history(&self.owner_claim_path, &self.owner_transitions_directory)?
                .claim_identifiers
                .contains(claim_id)
        {
            return Err(OutputError::InvalidInput(format!(
                "Output owner staging record '{}' is not bound to a claim in the immutable authority history.",
                staging_path.display()
            )));
        }
        Ok(Some(record.attempt_id))
    }

    pub(crate) fn owner_staging_bindings(&self) -> OutputResult<BTreeMap<AttemptIdentifier, BTreeSet<String>>> {
        let authority_claim_identifiers = if self.owner_claim_path.try_exists().map_err(OutputError::runtime)? {
            resolve_owner_authority_with_history(&self.owner_claim_path, &self.owner_transitions_directory)?
                .claim_identifiers
        } else {
            validate_owner_transition_directory(&self.owner_transitions_directory, &BTreeSet::new())?;
            BTreeSet::new()
        };
        let entries = match std::fs::read_dir(&self.owner_staging_directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(BTreeMap::new()),
            Err(error) => {
                return Err(OutputError::Runtime(format!(
                    "Failed to inspect output owner staging directory '{}': {error}",
                    self.owner_staging_directory.display()
                )));
            }
        };
        let mut bindings = BTreeMap::<AttemptIdentifier, BTreeSet<String>>::new();
        for entry in entries {
            let entry = entry.map_err(OutputError::runtime)?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(OutputError::runtime)?;
            if !file_type.is_file() {
                return Err(OutputError::InvalidInput(format!(
                    "Output owner staging directory contains non-file artifact '{}'.",
                    path.display()
                )));
            }
            let Some(file_name) = entry.file_name().to_str().map(str::to_string) else {
                return Err(OutputError::InvalidInput(format!(
                    "Output owner staging directory contains a non-UTF-8 file name at '{}'.",
                    path.display()
                )));
            };
            if is_immutable_record_temporary_file_name(&file_name) {
                continue;
            }
            let claim_id = file_name.strip_suffix(".json").ok_or_else(|| {
                OutputError::InvalidInput(format!(
                    "Output owner staging directory contains unsupported artifact '{}'.",
                    path.display()
                ))
            })?;
            validate_owner_claim_identifier(claim_id)?;
            let record = read_required_json::<OutputOwnerStagingRecord>(&path)?;
            record.validate()?;
            if record.claim_id != claim_id {
                return Err(OutputError::InvalidInput(format!(
                    "Output owner staging record '{}' is not bound to its file name.",
                    path.display()
                )));
            }
            if !authority_claim_identifiers.contains(claim_id) {
                return Err(OutputError::InvalidInput(format!(
                    "Output owner staging record '{}' is not bound to a claim in the immutable authority history.",
                    path.display()
                )));
            }
            if !bindings.entry(record.attempt_id).or_default().insert(record.claim_id) {
                return Err(OutputError::InvalidInput(
                    "Output owner staging metadata contains a duplicate claim binding.".to_string(),
                ));
            }
        }
        Ok(bindings)
    }

    pub(crate) fn cleanup_obsolete_owner_staging(
        &self,
        current_claim_id: &str,
        referenced_attempts: &BTreeSet<AttemptIdentifier>,
    ) -> OutputResult<()> {
        validate_owner_claim_identifier(current_claim_id)?;
        let authority =
            resolve_owner_authority_with_history(&self.owner_claim_path, &self.owner_transitions_directory)?;
        if !matches!(
            authority.state,
            OutputOwnerAuthorityState::Active(ref record) if record.claim_id == current_claim_id
        ) {
            return Err(OutputError::InvalidInput(format!(
                "Output staging cleanup claim '{current_claim_id}' is not the active owner authority."
            )));
        }
        for (attempt_id, claim_identifiers) in self.owner_staging_bindings()? {
            let obsolete_claim_identifiers = claim_identifiers
                .into_iter()
                .filter(|claim_identifier| claim_identifier != current_claim_id)
                .collect::<Vec<_>>();
            if obsolete_claim_identifiers.is_empty() {
                continue;
            }
            if !referenced_attempts.contains(&attempt_id) {
                let attempt_directory = self.attempt_directory(&attempt_id);
                match std::fs::remove_dir_all(&attempt_directory) {
                    Ok(()) => sync_nearest_existing_directory(&self.attempts_directory)?,
                    Err(error) if error.kind() == ErrorKind::NotFound => {
                        sync_nearest_existing_directory(&self.attempts_directory)?;
                    }
                    Err(error) => {
                        return Err(OutputError::Runtime(format!(
                            "Failed to remove obsolete unreferenced output attempt '{}': {error}",
                            attempt_directory.display()
                        )));
                    }
                }
            }
            for obsolete_claim_identifier in obsolete_claim_identifiers {
                self.retire_owner_staging_intent(&obsolete_claim_identifier, &attempt_id)?;
            }
        }
        Ok(())
    }

    pub(crate) fn retire_owner_staging_intent(
        &self,
        claim_id: &str,
        expected_attempt_id: &AttemptIdentifier,
    ) -> OutputResult<()> {
        validate_owner_claim_identifier(claim_id)?;
        let staging_path = self.owner_staging_directory.join(format!("{claim_id}.json"));
        let Some(record) = read_optional_json::<OutputOwnerStagingRecord>(&staging_path)? else {
            return sync_nearest_existing_directory(&self.owner_staging_directory);
        };
        record.validate()?;
        if record.claim_id != claim_id || record.attempt_id != *expected_attempt_id {
            return Err(OutputError::ConcurrentLineageUpdate { record_path: staging_path });
        }
        match std::fs::remove_file(&staging_path) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(OutputError::Runtime(format!(
                    "Failed to retire output owner staging intent '{}': {error}",
                    staging_path.display()
                )));
            }
        }
        sync_nearest_existing_directory(&self.owner_staging_directory)
    }

    pub(crate) fn require_fenced_owner_claim(&self, fenced_claim_id: &str) -> OutputResult<()> {
        validate_owner_claim_identifier(fenced_claim_id)?;
        if !self.owner_claim_path.try_exists().map_err(OutputError::runtime)? {
            return Err(OutputError::InvalidInput(format!(
                "Output owner claim '{fenced_claim_id}' was declared fenced, but no owner authority exists at '{}'.",
                self.owner_claim_path.display()
            )));
        }
        let record = match resolve_owner_authority(&self.owner_claim_path, &self.owner_transitions_directory)? {
            OutputOwnerAuthorityState::Active(record) => record,
            OutputOwnerAuthorityState::Released { .. } => {
                return Err(OutputError::InvalidInput(format!(
                    "Output owner claim '{fenced_claim_id}' was declared fenced, but owner authority is already released."
                )));
            }
        };
        if record.claim_id == fenced_claim_id {
            Ok(())
        } else {
            Err(OutputError::InvalidInput(format!(
                "Output owner claim '{}' was declared fenced, but the surviving claim is '{}'; no owner transition was published.",
                fenced_claim_id, record.claim_id
            )))
        }
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

    fn terminal_finalization_path(&self, attempt_id: &AttemptIdentifier) -> PathBuf {
        self.terminal_finalizations_directory.join(format!("{}.json", attempt_id.as_str()))
    }

    pub(crate) fn inspect(&self) -> OutputResult<Option<LineageSnapshot>> {
        let Some(genesis) = read_optional_json::<LineageGenesisRecord>(&self.genesis_path)? else {
            return Ok(None);
        };
        genesis.validate()?;
        let mut traversal = LineageTraversal {
            successor_records: Vec::new(),
            visited_attempts: BTreeSet::from([genesis.attempt_id.clone()]),
            leaf_attempt_id: genesis.attempt_id.clone(),
            leaf_terminal: None,
            pending_terminal: None,
        };
        loop {
            match self.inspect_leaf(&genesis.run_set_id, &traversal.leaf_attempt_id)? {
                InspectedLineageLeaf::Active => break,
                InspectedLineageLeaf::Terminal { record, finalized: true } => {
                    traversal.leaf_terminal = Some(record);
                    break;
                }
                InspectedLineageLeaf::Terminal { record, finalized: false } => {
                    traversal.pending_terminal = Some(record);
                    break;
                }
                InspectedLineageLeaf::Successor(successor) => {
                    if !traversal.visited_attempts.insert(successor.attempt_id.clone()) {
                        return Err(OutputError::InvalidInput(
                            "Output lineage successor chain contains a cycle.".to_string(),
                        ));
                    }
                    traversal.leaf_attempt_id = successor.attempt_id.clone();
                    traversal.successor_records.push(successor);
                }
            }
        }
        self.validate_attempt_directories(&traversal)?;
        Ok(Some(LineageSnapshot {
            genesis,
            successor_records: traversal.successor_records,
            leaf_attempt_id: traversal.leaf_attempt_id,
            leaf_terminal: traversal.leaf_terminal,
            pending_terminal: traversal.pending_terminal,
        }))
    }

    fn inspect_leaf(
        &self,
        run_set_id: &str,
        leaf_attempt_id: &AttemptIdentifier,
    ) -> OutputResult<InspectedLineageLeaf> {
        self.reject_legacy_terminal(leaf_attempt_id)?;
        let outcome_path = self.outcome_path(leaf_attempt_id);
        let outcome = read_optional_json::<AttemptOutcomeRecord>(&outcome_path)?;
        if let Some(outcome) = &outcome {
            outcome.validate()?;
            if outcome.run_set_id() != run_set_id || outcome.attempt_id() != leaf_attempt_id {
                return Err(OutputError::InvalidInput(format!(
                    "Output attempt outcome '{}' is not bound to its traversed attempt and run set.",
                    outcome_path.display()
                )));
            }
        }
        let normal_successor_path = self.normal_successor_path(leaf_attempt_id);
        let normal_successor = read_optional_json::<LineageSuccessorRecord>(&normal_successor_path)?;
        let inspected_leaf = match (outcome, normal_successor) {
            (None, None) => InspectedLineageLeaf::Active,
            (None, Some(_)) => {
                return Err(OutputError::InvalidInput(format!(
                    "Output lineage successor '{}' has no immutable parent terminal outcome.",
                    normal_successor_path.display()
                )));
            }
            (Some(AttemptOutcomeRecord::ExactRecoveryClaim(successor)), None) => {
                InspectedLineageLeaf::Successor(successor)
            }
            (Some(AttemptOutcomeRecord::ExactRecoveryClaim(_)), Some(_)) => {
                return Err(OutputError::InvalidInput(format!(
                    "Output exact-recovery attempt '{}' also has an incompatible normal successor record.",
                    leaf_attempt_id.as_str()
                )));
            }
            (Some(AttemptOutcomeRecord::TerminalClaim(terminal)), None) => InspectedLineageLeaf::Terminal {
                finalized: self.terminal_is_finalized(&terminal, &outcome_path)?,
                record: terminal,
            },
            (Some(AttemptOutcomeRecord::TerminalClaim(terminal)), Some(successor)) => {
                self.validate_terminal_successor(
                    leaf_attempt_id,
                    &terminal,
                    &successor,
                    &outcome_path,
                    &normal_successor_path,
                )?;
                InspectedLineageLeaf::Successor(successor)
            }
        };
        if let InspectedLineageLeaf::Successor(successor) = &inspected_leaf
            && (successor.run_set_id != run_set_id || successor.parent_attempt_id != *leaf_attempt_id)
        {
            return Err(OutputError::InvalidInput(format!(
                "Output lineage successor '{}' is not bound to its traversed parent and run set.",
                normal_successor_path.display()
            )));
        }
        Ok(inspected_leaf)
    }

    fn validate_terminal_successor(
        &self,
        leaf_attempt_id: &AttemptIdentifier,
        terminal: &LineageTerminalRecord,
        successor: &LineageSuccessorRecord,
        outcome_path: &Path,
        normal_successor_path: &Path,
    ) -> OutputResult<()> {
        if !self.terminal_is_finalized(terminal, outcome_path)? {
            return Err(OutputError::InvalidInput(format!(
                "Output attempt '{}' has a successor before terminal finalization.",
                leaf_attempt_id.as_str()
            )));
        }
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
        let observed_terminal_sha256 = file_sha256(outcome_path)?;
        if successor.parent_terminal_sha256.as_deref() != Some(observed_terminal_sha256.as_str()) {
            return Err(OutputError::InvalidInput(format!(
                "Output lineage successor '{}' has a stale parent terminal binding.",
                normal_successor_path.display()
            )));
        }
        Ok(())
    }

    fn validate_attempt_directories(&self, traversal: &LineageTraversal) -> OutputResult<()> {
        let unmaterialized_exact_recovery_parents = traversal
            .successor_records
            .iter()
            .filter(|successor| successor.recovery_kind == LineageRecoveryKind::ExactNonterminalRecovery)
            .map(|successor| &successor.parent_attempt_id)
            .collect::<BTreeSet<_>>();
        for attempt_id in &traversal.visited_attempts {
            let attempt_directory = self.attempt_directory(attempt_id);
            let is_unmaterialized_leaf = attempt_id == &traversal.leaf_attempt_id
                && traversal.leaf_terminal.is_none()
                && traversal.pending_terminal.is_none();
            let is_unmaterialized_exact_recovery_parent = unmaterialized_exact_recovery_parents.contains(attempt_id);
            match std::fs::symlink_metadata(&attempt_directory) {
                Ok(metadata) if metadata.is_dir() => {}
                Ok(_) => {
                    return Err(OutputError::InvalidInput(format!(
                        "Output lineage attempt path '{}' is not a directory.",
                        attempt_directory.display()
                    )));
                }
                Err(error)
                    if error.kind() == ErrorKind::NotFound
                        && (is_unmaterialized_leaf || is_unmaterialized_exact_recovery_parent) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    return Err(OutputError::InvalidInput(format!(
                        "Output lineage references missing attempt directory '{}'.",
                        attempt_directory.display()
                    )));
                }
                Err(error) => {
                    return Err(OutputError::Runtime(format!(
                        "Failed to inspect output attempt directory '{}': {error}",
                        attempt_directory.display()
                    )));
                }
            }
        }
        Ok(())
    }

    fn terminal_is_finalized(&self, terminal: &LineageTerminalRecord, outcome_path: &Path) -> OutputResult<bool> {
        let finalization_path = self.terminal_finalization_path(&terminal.attempt_id);
        let Some(finalization) = read_optional_json::<LineageTerminalFinalizationRecord>(&finalization_path)? else {
            return Ok(false);
        };
        finalization.validate()?;
        let observed_claim_sha256 = file_sha256(outcome_path)?;
        if finalization.run_set_id != terminal.run_set_id
            || finalization.attempt_id != terminal.attempt_id
            || finalization.terminal_claim_sha256 != observed_claim_sha256
        {
            return Err(OutputError::InvalidInput(format!(
                "Output terminal finalization '{}' has a stale terminal claim binding.",
                finalization_path.display()
            )));
        }
        Ok(true)
    }

    pub(crate) fn publish_genesis(&self, genesis: &LineageGenesisRecord) -> OutputResult<NoReplacePublication> {
        genesis.validate()?;
        Self::publish_record(&self.genesis_path, genesis)
    }

    pub(crate) fn publish_successor(&self, successor: &LineageSuccessorRecord) -> OutputResult<NoReplacePublication> {
        successor.validate()?;
        match successor.recovery_kind {
            LineageRecoveryKind::ExactNonterminalRecovery => {
                let outcome = AttemptOutcomeRecord::ExactRecoveryClaim(successor.clone());
                Self::publish_record(&self.outcome_path(&successor.parent_attempt_id), &outcome)
            }
            LineageRecoveryKind::TerminalResume => {
                let outcome_path = self.outcome_path(&successor.parent_attempt_id);
                let outcome = read_required_json::<AttemptOutcomeRecord>(&outcome_path)?;
                outcome.validate()?;
                let AttemptOutcomeRecord::TerminalClaim(terminal) = outcome else {
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
                if !self.terminal_is_finalized(&terminal, &outcome_path)? {
                    return Err(OutputError::InvalidInput(format!(
                        "Output terminal-resume successor cannot follow unfinalized attempt '{}'.",
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
                Self::publish_record(&self.normal_successor_path(&successor.parent_attempt_id), successor)
            }
        }
    }

    pub(crate) fn publish_terminal_claim(
        &self,
        terminal: &LineageTerminalRecord,
    ) -> OutputResult<NoReplacePublication> {
        terminal.validate()?;
        let outcome = AttemptOutcomeRecord::TerminalClaim(terminal.clone());
        Self::publish_record(&self.outcome_path(&terminal.attempt_id), &outcome)
    }

    pub(crate) fn finalize_terminal(&self, terminal: &LineageTerminalRecord) -> OutputResult<NoReplacePublication> {
        terminal.validate()?;
        let outcome_path = self.outcome_path(&terminal.attempt_id);
        let outcome = read_required_json::<AttemptOutcomeRecord>(&outcome_path)?;
        outcome.validate()?;
        if outcome != AttemptOutcomeRecord::TerminalClaim(terminal.clone()) {
            return Err(OutputError::ConcurrentLineageUpdate { record_path: outcome_path });
        }
        let finalization = LineageTerminalFinalizationRecord::new(terminal, file_sha256(&outcome_path)?)?;
        Self::publish_record(&self.terminal_finalization_path(&terminal.attempt_id), &finalization)
    }

    #[cfg(test)]
    pub(crate) fn publish_terminal(&self, terminal: &LineageTerminalRecord) -> OutputResult<NoReplacePublication> {
        let claim_publication = self.publish_terminal_claim(terminal)?;
        self.finalize_terminal(terminal)?;
        Ok(claim_publication)
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

    fn publish_record<RecordType>(path: &Path, record: &RecordType) -> OutputResult<NoReplacePublication>
    where
        RecordType: Serialize + for<'deserialize> Deserialize<'deserialize> + Eq,
    {
        publish_json_no_replace_reconciled(path, record)
    }
}

pub(crate) fn terminal_record_sha256(
    paths: &OutputLineagePaths,
    attempt_id: &AttemptIdentifier,
) -> OutputResult<String> {
    let outcome_path = paths.outcome_path(attempt_id);
    let outcome = read_required_json::<AttemptOutcomeRecord>(&outcome_path)?;
    outcome.validate()?;
    let AttemptOutcomeRecord::TerminalClaim(terminal) = outcome else {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt '{}' has no terminal outcome to hash.",
            attempt_id.as_str()
        )));
    };
    if !paths.terminal_is_finalized(&terminal, &outcome_path)? {
        return Err(OutputError::InvalidInput(format!(
            "Output attempt '{}' terminal is not finalized.",
            attempt_id.as_str()
        )));
    }
    file_sha256(&outcome_path)
}

fn owner_transition_path(owner_transitions_directory: &Path, predecessor_state_id: &str) -> PathBuf {
    owner_transitions_directory.join(format!("{predecessor_state_id}.json"))
}

fn publish_owner_transition(
    owner_transitions_directory: &Path,
    transition: &OutputOwnerTransitionRecord,
) -> OutputResult<NoReplacePublication> {
    transition.validate()?;
    let path = owner_transition_path(owner_transitions_directory, transition.predecessor_state_id());
    let publication = publish_json_no_replace_reconciled(&path, transition)?;
    if publication == NoReplacePublication::AlreadyExists {
        let existing = read_required_json::<OutputOwnerTransitionRecord>(&path)?;
        existing.validate()?;
        if existing != *transition {
            return Err(OutputError::ConcurrentLineageUpdate { record_path: path });
        }
        if existing.predecessor_state_id() != transition.predecessor_state_id() {
            return Err(OutputError::InvalidInput(format!(
                "Output owner transition '{}' is not bound to its predecessor file name.",
                path.display()
            )));
        }
    }
    Ok(publication)
}

fn publish_json_no_replace_reconciled<ValueType>(
    destination_path: &Path,
    value: &ValueType,
) -> OutputResult<NoReplacePublication>
where
    ValueType: Serialize + for<'deserialize> Deserialize<'deserialize> + Eq,
{
    for attempt_index in 0..2 {
        match publish_json_no_replace(destination_path, value) {
            Ok(publication) => {
                let existing = read_required_json::<ValueType>(destination_path)?;
                if existing != *value {
                    return Err(OutputError::ConcurrentLineageUpdate { record_path: destination_path.to_path_buf() });
                }
                if publication == NoReplacePublication::AlreadyExists {
                    sync_publication_directory_with_retry(destination_path)?;
                }
                return Ok(publication);
            }
            Err(publication_error) => match read_optional_json::<ValueType>(destination_path)? {
                Some(existing) if existing == *value => {
                    sync_publication_directory_with_retry(destination_path)?;
                    return Ok(NoReplacePublication::Created);
                }
                Some(_) => {
                    return Err(OutputError::ConcurrentLineageUpdate { record_path: destination_path.to_path_buf() });
                }
                None if attempt_index == 0 => {}
                None => return Err(publication_error),
            },
        }
    }
    unreachable!("the no-replace publication retry loop always returns")
}

fn sync_publication_directory_with_retry(destination_path: &Path) -> OutputResult<()> {
    let parent_directory = destination_path.parent().ok_or_else(|| {
        OutputError::InvalidInput(format!("Output publication path '{}' has no parent.", destination_path.display()))
    })?;
    match sync_immutable_publication_directory(destination_path, parent_directory) {
        Ok(()) => Ok(()),
        Err(first_error) => {
            sync_immutable_publication_directory(destination_path, parent_directory).map_err(|retry_error| {
                OutputError::Runtime(format!(
                    "Failed to confirm immutable output publication directory durability after one synchronous retry. First failure: {first_error}; retry failure: {retry_error}"
                ))
            })
        }
    }
}

fn resolve_owner_authority(
    owner_claim_path: &Path,
    owner_transitions_directory: &Path,
) -> OutputResult<OutputOwnerAuthorityState> {
    Ok(resolve_owner_authority_with_history(owner_claim_path, owner_transitions_directory)?.state)
}

fn resolve_owner_authority_with_history(
    owner_claim_path: &Path,
    owner_transitions_directory: &Path,
) -> OutputResult<ResolvedOutputOwnerAuthority> {
    resolve_owner_authority_with_history_observed(owner_claim_path, owner_transitions_directory, &mut || Ok(()))
}

fn resolve_owner_authority_with_history_observed<LeafObserver>(
    owner_claim_path: &Path,
    owner_transitions_directory: &Path,
    leaf_observer: &mut LeafObserver,
) -> OutputResult<ResolvedOutputOwnerAuthority>
where
    LeafObserver: FnMut() -> OutputResult<()>,
{
    let mut preceding_unreachable_snapshot = None;
    for _ in 0..=MAXIMUM_OWNER_TRANSITION_COUNT {
        match resolve_owner_authority_with_history_once(owner_claim_path, owner_transitions_directory, leaf_observer) {
            Err(OutputError::ConcurrentLineageUpdate { record_path }) => {
                let unreachable_snapshot = snapshot_owner_transition_directory(owner_transitions_directory)?;
                if preceding_unreachable_snapshot.as_ref() == Some(&unreachable_snapshot) {
                    return Err(OutputError::InvalidInput(format!(
                        "Output owner transition '{}' is not reachable from the immutable root claim.",
                        record_path.display()
                    )));
                }
                preceding_unreachable_snapshot = Some(unreachable_snapshot);
            }
            result => return result,
        }
    }
    Err(OutputError::InvalidInput(format!(
        "Output owner authority did not stabilize within the supported {MAXIMUM_OWNER_TRANSITION_COUNT} transition limit."
    )))
}

fn snapshot_owner_transition_directory(owner_transitions_directory: &Path) -> OutputResult<Vec<(String, String)>> {
    let entries = match std::fs::read_dir(owner_transitions_directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(OutputError::Runtime(format!(
                "Failed to snapshot output owner transition directory '{}': {error}",
                owner_transitions_directory.display()
            )));
        }
    };
    let mut transition_records = Vec::new();
    for (artifact_index, entry) in entries.enumerate() {
        if artifact_index == MAXIMUM_OWNER_TRANSITION_COUNT {
            return Err(OutputError::InvalidInput(format!(
                "Output owner transition directory exceeds the supported {MAXIMUM_OWNER_TRANSITION_COUNT} artifact limit."
            )));
        }
        let entry = entry.map_err(OutputError::runtime)?;
        let path = entry.path();
        if !entry.file_type().map_err(OutputError::runtime)?.is_file() {
            return Err(OutputError::InvalidInput(format!(
                "Output owner transition directory contains non-file artifact '{}'.",
                path.display()
            )));
        }
        let file_name = entry.file_name();
        let file_name = file_name.to_str().ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Output owner transition directory contains a non-UTF-8 file name at '{}'.",
                path.display()
            ))
        })?;
        if is_immutable_record_temporary_file_name(file_name) {
            continue;
        }
        let predecessor_state_id = file_name.strip_suffix(".json").ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Output owner transition directory contains unsupported artifact '{}'.",
                path.display()
            ))
        })?;
        validate_owner_claim_identifier(predecessor_state_id)?;
        transition_records.push((predecessor_state_id.to_string(), file_sha256(&path)?));
    }
    transition_records.sort_unstable();
    Ok(transition_records)
}

fn resolve_owner_authority_with_history_once(
    owner_claim_path: &Path,
    owner_transitions_directory: &Path,
    leaf_observer: &mut impl FnMut() -> OutputResult<()>,
) -> OutputResult<ResolvedOutputOwnerAuthority> {
    let root = read_required_json::<OutputOwnerClaimRecord>(owner_claim_path)?;
    root.validate()?;
    let mut claim_identifiers = BTreeSet::from([root.claim_id.clone()]);
    let mut state = OutputOwnerAuthorityState::Active(root);
    let mut visited_state_identifiers = BTreeSet::new();
    let mut traversed_transition_predecessors = BTreeSet::new();
    #[cfg(test)]
    let mut transition_predecessors = Vec::new();
    let mut transition_count = 0_usize;
    loop {
        let predecessor_state_id = match &state {
            OutputOwnerAuthorityState::Active(record) => record.claim_id.as_str(),
            OutputOwnerAuthorityState::Released { released_state_id } => released_state_id.as_str(),
        };
        if !visited_state_identifiers.insert(predecessor_state_id.to_string()) {
            return Err(OutputError::InvalidInput(
                "Output owner authority transition chain contains a cycle.".to_string(),
            ));
        }
        let transition_path = owner_transition_path(owner_transitions_directory, predecessor_state_id);
        let Some(transition) = read_optional_json::<OutputOwnerTransitionRecord>(&transition_path)? else {
            leaf_observer()?;
            validate_owner_transition_directory(owner_transitions_directory, &traversed_transition_predecessors)?;
            return Ok(ResolvedOutputOwnerAuthority {
                state,
                claim_identifiers,
                #[cfg(test)]
                transition_predecessors,
            });
        };
        if transition_count == MAXIMUM_OWNER_TRANSITION_COUNT {
            return Err(OutputError::InvalidInput(format!(
                "Output owner authority exceeds the supported {MAXIMUM_OWNER_TRANSITION_COUNT} transition limit."
            )));
        }
        transition.validate()?;
        if transition.predecessor_state_id() != predecessor_state_id {
            return Err(OutputError::InvalidInput(format!(
                "Output owner transition '{}' is not bound to its predecessor state.",
                transition_path.display()
            )));
        }
        traversed_transition_predecessors.insert(predecessor_state_id.to_string());
        #[cfg(test)]
        transition_predecessors.push(predecessor_state_id.to_string());
        transition_count += 1;
        state = match (state, transition) {
            (
                OutputOwnerAuthorityState::Active(_),
                OutputOwnerTransitionRecord::GracefulRelease { released_state_id, .. },
            ) => OutputOwnerAuthorityState::Released { released_state_id },
            (OutputOwnerAuthorityState::Active(_), OutputOwnerTransitionRecord::FencedTakeover { claim, .. })
            | (
                OutputOwnerAuthorityState::Released { .. },
                OutputOwnerTransitionRecord::AcquireAfterRelease { claim, .. },
            ) => {
                claim_identifiers.insert(claim.claim_id.clone());
                OutputOwnerAuthorityState::Active(claim)
            }
            (OutputOwnerAuthorityState::Active(_), OutputOwnerTransitionRecord::AcquireAfterRelease { .. })
            | (OutputOwnerAuthorityState::Released { .. }, OutputOwnerTransitionRecord::GracefulRelease { .. })
            | (OutputOwnerAuthorityState::Released { .. }, OutputOwnerTransitionRecord::FencedTakeover { .. }) => {
                return Err(OutputError::InvalidInput(format!(
                    "Output owner transition '{}' is invalid for its predecessor authority state.",
                    transition_path.display()
                )));
            }
        };
    }
}

fn validate_owner_transition_directory(
    owner_transitions_directory: &Path,
    traversed_transition_predecessors: &BTreeSet<String>,
) -> OutputResult<()> {
    let entries = match std::fs::read_dir(owner_transitions_directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(OutputError::Runtime(format!(
                "Failed to inspect output owner transition directory '{}': {error}",
                owner_transitions_directory.display()
            )));
        }
    };
    for (artifact_index, entry) in entries.enumerate() {
        if artifact_index == MAXIMUM_OWNER_TRANSITION_COUNT {
            return Err(OutputError::InvalidInput(format!(
                "Output owner transition directory exceeds the supported {MAXIMUM_OWNER_TRANSITION_COUNT} artifact limit."
            )));
        }
        let entry = entry.map_err(OutputError::runtime)?;
        let path = entry.path();
        if !entry.file_type().map_err(OutputError::runtime)?.is_file() {
            return Err(OutputError::InvalidInput(format!(
                "Output owner transition directory contains non-file artifact '{}'.",
                path.display()
            )));
        }
        let file_name = entry.file_name();
        let file_name = file_name.to_str().ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Output owner transition directory contains a non-UTF-8 file name at '{}'.",
                path.display()
            ))
        })?;
        if is_immutable_record_temporary_file_name(file_name) {
            continue;
        }
        let predecessor_state_id = file_name.strip_suffix(".json").ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Output owner transition directory contains unsupported artifact '{}'.",
                path.display()
            ))
        })?;
        validate_owner_claim_identifier(predecessor_state_id)?;
        if !traversed_transition_predecessors.contains(predecessor_state_id) {
            return Err(OutputError::ConcurrentLineageUpdate { record_path: path });
        }
    }
    Ok(())
}

fn current_host_name() -> String {
    for environment_name in ["SLURMD_NODENAME", "HOSTNAME"] {
        if let Ok(host_name) = std::env::var(environment_name)
            && !host_name.trim().is_empty()
        {
            return host_name;
        }
    }
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|host_name| host_name.trim().to_string())
        .filter(|host_name| !host_name.is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
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
        validate_safe_path_component(&phenotype.output_directory_name, "phenotype directory name")?;
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
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')) {
        return Err(OutputError::InvalidInput(format!(
            "Output {role} SHA-256 must contain exactly 64 hexadecimal characters."
        )));
    }
    Ok(())
}

fn is_immutable_record_temporary_file_name(file_name: &str) -> bool {
    let Some(identifier) = file_name
        .strip_prefix('.')
        .and_then(|name| name.rsplit_once(".attempt-"))
        .and_then(|(_, suffix)| suffix.strip_suffix(".tmp"))
    else {
        return false;
    };
    identifier.len() == 32 && identifier.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn sync_existing_directory(path: &Path) -> OutputResult<()> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => sync_directory(path),
        Ok(_) => Err(OutputError::InvalidInput(format!(
            "Expected output authority directory '{}' is not a directory.",
            path.display()
        ))),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(OutputError::Runtime(format!(
            "Failed to inspect output authority directory '{}': {error}",
            path.display()
        ))),
    }
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

    const OWNER_CLAIM_HELPER_MODE_ENVIRONMENT: &str = "G_OUTPUT_OWNER_CLAIM_HELPER_MODE";
    const OWNER_CLAIM_HELPER_FENCE_ENVIRONMENT: &str = "G_OUTPUT_OWNER_CLAIM_HELPER_FENCE";
    const OWNER_CLAIM_HELPER_START_ENVIRONMENT: &str = "G_OUTPUT_OWNER_CLAIM_HELPER_START";
    const OWNER_CLAIM_HELPER_OUTCOME_ENVIRONMENT: &str = "G_OUTPUT_OWNER_CLAIM_HELPER_OUTCOME";
    const OWNER_CLAIM_HELPER_ROOT_ENVIRONMENT: &str = "G_OUTPUT_OWNER_CLAIM_HELPER_ROOT";
    const OWNER_CLAIM_HELPER_READY_ENVIRONMENT: &str = "G_OUTPUT_OWNER_CLAIM_HELPER_READY";
    const OWNER_CLAIM_HELPER_EXPECTED_LEAF_ENVIRONMENT: &str = "G_OUTPUT_OWNER_CLAIM_HELPER_EXPECTED_LEAF";
    const OWNER_CLAIM_HELPER_TEST_NAME: &str = "persistence::lineage::tests::owner_claim_subprocess_helper";

    #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum OwnerAuthorityInspectionState {
        Active,
        Released,
    }

    #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(deny_unknown_fields)]
    struct OwnerAuthorityTransitionInspection {
        predecessor_state_id: String,
        record_sha256: String,
    }

    #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(deny_unknown_fields)]
    struct OwnerAuthorityInspection {
        schema_version: u32,
        authority_state: OwnerAuthorityInspectionState,
        final_leaf_state_id: String,
        final_leaf_record_sha256: String,
        root_record_sha256: String,
        transition_records: Vec<OwnerAuthorityTransitionInspection>,
    }

    #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum OwnerClaimHelperAcquisitionStatus {
        Acquired,
        Rejected,
    }

    #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(deny_unknown_fields)]
    struct OwnerClaimHelperAcquisitionOutcome {
        schema_version: u32,
        status: OwnerClaimHelperAcquisitionStatus,
        claim_id: Option<String>,
        error: Option<String>,
    }

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

    fn wait_for_test_path(path: &Path, role: &str) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !path.exists() {
            assert!(std::time::Instant::now() < deadline, "timed out waiting for {role}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    fn inspect_owner_authority(
        paths: &OutputLineagePaths,
        expected_leaf_state_id: Option<&str>,
    ) -> OutputResult<OwnerAuthorityInspection> {
        if let Some(expected_leaf_state_id) = expected_leaf_state_id {
            validate_owner_claim_identifier(expected_leaf_state_id)?;
        }
        let resolved =
            resolve_owner_authority_with_history(&paths.owner_claim_path, &paths.owner_transitions_directory)?;
        let (authority_state, final_leaf_state_id) = match &resolved.state {
            OutputOwnerAuthorityState::Active(record) => {
                (OwnerAuthorityInspectionState::Active, record.claim_id.clone())
            }
            OutputOwnerAuthorityState::Released { released_state_id } => {
                (OwnerAuthorityInspectionState::Released, released_state_id.clone())
            }
        };
        if let Some(expected_leaf_state_id) = expected_leaf_state_id
            && expected_leaf_state_id != final_leaf_state_id
        {
            return Err(OutputError::InvalidInput(format!(
                "Output owner authority inspection expected leaf '{expected_leaf_state_id}', but the current leaf is '{final_leaf_state_id}'."
            )));
        }
        let root_record_sha256 = file_sha256(&paths.owner_claim_path)?;
        let transition_records = resolved
            .transition_predecessors
            .into_iter()
            .map(|predecessor_state_id| {
                let record_sha256 =
                    file_sha256(&owner_transition_path(&paths.owner_transitions_directory, &predecessor_state_id))?;
                Ok(OwnerAuthorityTransitionInspection { predecessor_state_id, record_sha256 })
            })
            .collect::<OutputResult<Vec<_>>>()?;
        let final_leaf_record_sha256 = transition_records
            .last()
            .map_or_else(|| root_record_sha256.clone(), |transition| transition.record_sha256.clone());
        Ok(OwnerAuthorityInspection {
            schema_version: LINEAGE_SCHEMA_VERSION,
            authority_state,
            final_leaf_state_id,
            final_leaf_record_sha256,
            root_record_sha256,
            transition_records,
        })
    }

    fn write_helper_json<ValueType>(path: &Path, value: &ValueType)
    where
        ValueType: Serialize,
    {
        let mut bytes = serde_json::to_vec_pretty(value).expect("owner claim helper JSON serializes");
        bytes.push(b'\n');
        std::fs::write(path, bytes).expect("owner claim helper JSON writes");
    }

    fn run_race_acquisition_helper(paths: &OutputLineagePaths) {
        let outcome = match paths.try_acquire_owner_claim() {
            Ok(claim) => OwnerClaimHelperAcquisitionOutcome {
                schema_version: LINEAGE_SCHEMA_VERSION,
                status: OwnerClaimHelperAcquisitionStatus::Acquired,
                claim_id: Some(claim.claim_id().to_string()),
                error: None,
            },
            Err(error) => OwnerClaimHelperAcquisitionOutcome {
                schema_version: LINEAGE_SCHEMA_VERSION,
                status: OwnerClaimHelperAcquisitionStatus::Rejected,
                claim_id: None,
                error: Some(error.to_string()),
            },
        };
        let outcome_path =
            std::env::var(OWNER_CLAIM_HELPER_OUTCOME_ENVIRONMENT).expect("race acquisition outcome path is configured");
        write_helper_json(Path::new(&outcome_path), &outcome);
    }

    fn run_authority_inspection_helper(paths: &OutputLineagePaths) {
        let expected_leaf_state_id = std::env::var(OWNER_CLAIM_HELPER_EXPECTED_LEAF_ENVIRONMENT).ok();
        let inspection = inspect_owner_authority(paths, expected_leaf_state_id.as_deref())
            .expect("owner authority inspection succeeds");
        let outcome_path = std::env::var(OWNER_CLAIM_HELPER_OUTCOME_ENVIRONMENT)
            .expect("authority inspection outcome path is configured");
        write_helper_json(Path::new(&outcome_path), &inspection);
    }

    fn run_owner_transition_crash_helper(
        paths: &OutputLineagePaths,
        mode: &str,
        failpoint: &str,
        fenced_claim_id: Option<&str>,
    ) -> std::process::Output {
        let mut command = std::process::Command::new(std::env::current_exe().expect("test executable resolves"));
        command
            .args(["--exact", OWNER_CLAIM_HELPER_TEST_NAME, "--nocapture"])
            .env(OWNER_CLAIM_HELPER_MODE_ENVIRONMENT, mode)
            .env(OWNER_CLAIM_HELPER_ROOT_ENVIRONMENT, &paths.output_root)
            .env("G_OUTPUT_TEST_CRASH_POINT", failpoint);
        if let Some(fenced_claim_id) = fenced_claim_id {
            command.env(OWNER_CLAIM_HELPER_FENCE_ENVIRONMENT, fenced_claim_id);
        }
        command.output().expect("owner transition crash helper runs")
    }

    fn assert_owner_transition_crash(output: &std::process::Output) {
        assert_eq!(
            output.status.code(),
            Some(86),
            "owner transition helper did not crash at its failpoint\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn owner_claim_subprocess_helper() {
        let Ok(mode) = std::env::var(OWNER_CLAIM_HELPER_MODE_ENVIRONMENT) else {
            return;
        };
        let output_root =
            std::env::var(OWNER_CLAIM_HELPER_ROOT_ENVIRONMENT).expect("claim helper output root is configured");
        let paths = OutputLineagePaths::new(Path::new(&output_root));
        match mode.as_str() {
            "hold" => {
                let _claim = paths.try_acquire_owner_claim().expect("claim helper acquires owner claim");
                let ready_path =
                    std::env::var(OWNER_CLAIM_HELPER_READY_ENVIRONMENT).expect("claim helper ready path is configured");
                std::fs::write(ready_path, b"ready").expect("claim helper reports readiness");
                loop {
                    std::thread::park_timeout(std::time::Duration::from_mins(1));
                }
            }
            "blocked" => {
                let error = paths.try_acquire_owner_claim().expect_err("surviving claim blocks contender");
                assert!(matches!(error, OutputError::SurvivingOutputOwnerClaim { .. }));
            }
            "release" => {
                let mut claim = paths.try_acquire_owner_claim().expect("claim helper acquires owner claim");
                claim.release().expect("claim helper releases its own claim");
            }
            "crash_release" => {
                let mut claim = paths.try_acquire_owner_claim().expect("crash release acquires root claim");
                claim.release().expect("configured release crash point fires");
                panic!("release crash helper did not reach its configured failpoint");
            }
            "crash_takeover" => {
                let fenced_claim_id =
                    std::env::var(OWNER_CLAIM_HELPER_FENCE_ENVIRONMENT).expect("crash fence is configured");
                let _claim = paths
                    .take_over_fenced_owner_claim(&fenced_claim_id)
                    .expect("configured takeover crash point fires");
                panic!("takeover crash helper did not reach its configured failpoint");
            }
            "crash_reacquire" => {
                let _claim = paths.try_acquire_owner_claim().expect("configured reacquisition crash point fires");
                panic!("reacquisition crash helper did not reach its configured failpoint");
            }
            "race_release" => {
                let mut claim = paths.try_acquire_owner_claim().expect("race owner acquires root claim");
                let ready_path =
                    std::env::var(OWNER_CLAIM_HELPER_READY_ENVIRONMENT).expect("race owner ready path is configured");
                std::fs::write(ready_path, claim.claim_id()).expect("race owner reports its claim");
                let start_path =
                    std::env::var(OWNER_CLAIM_HELPER_START_ENVIRONMENT).expect("race start path is configured");
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while !Path::new(&start_path).exists() {
                    assert!(std::time::Instant::now() < deadline, "race release start timed out");
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                let outcome = match claim.release() {
                    Ok(()) => "ok".to_string(),
                    Err(error) => format!("error:{error}"),
                };
                let outcome_path = std::env::var(OWNER_CLAIM_HELPER_OUTCOME_ENVIRONMENT)
                    .expect("race release outcome path is configured");
                std::fs::write(outcome_path, outcome).expect("race release writes its outcome");
            }
            "race_takeover" => {
                let fenced_claim_id =
                    std::env::var(OWNER_CLAIM_HELPER_FENCE_ENVIRONMENT).expect("race fence is configured");
                let outcome = match paths.take_over_fenced_owner_claim(&fenced_claim_id) {
                    Ok(claim) => format!("ok:{}", claim.claim_id()),
                    Err(error) => format!("error:{error}"),
                };
                let outcome_path = std::env::var(OWNER_CLAIM_HELPER_OUTCOME_ENVIRONMENT)
                    .expect("race takeover outcome path is configured");
                std::fs::write(outcome_path, outcome).expect("race takeover writes its outcome");
            }
            "race_acquire" => run_race_acquisition_helper(&paths),
            "inspect" => run_authority_inspection_helper(&paths),
            "invalid_host" => {
                let error = paths.try_acquire_owner_claim().expect_err("invalid claim metadata is rejected");
                assert!(matches!(error, OutputError::InvalidInput(_)));
                assert!(!paths.output_root.exists(), "invalid claim metadata must not mutate the filesystem");
            }
            unsupported_mode => panic!("unsupported claim helper mode '{unsupported_mode}'"),
        }
    }

    #[test]
    fn invalid_owner_claim_metadata_is_rejected_before_directory_creation() {
        use std::process::{Command, Stdio};

        let paths = test_paths("invalid-owner-host");
        let test_executable = std::env::current_exe().expect("current test executable resolves");
        let status = Command::new(test_executable)
            .args(["--exact", OWNER_CLAIM_HELPER_TEST_NAME, "--nocapture"])
            .env(OWNER_CLAIM_HELPER_MODE_ENVIRONMENT, "invalid_host")
            .env(OWNER_CLAIM_HELPER_ROOT_ENVIRONMENT, &paths.output_root)
            .env("SLURMD_NODENAME", "invalid\nhost")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("invalid-host claim helper runs");
        assert!(status.success());
        assert!(!paths.output_root.exists());
    }

    #[test]
    fn owner_authority_inspection_reports_exact_schema_zero_chain_hashes() {
        let paths = test_paths("owner-authority-inspection");
        let mut root_claim = paths.try_acquire_owner_claim().expect("root claim publishes");
        let root_claim_id = root_claim.claim_id().to_string();
        let root_record_sha256 = file_sha256(&paths.owner_claim_path).expect("root claim hashes");
        root_claim.release().expect("root claim releases");
        let released = inspect_owner_authority(&paths, None).expect("released authority inspects");
        assert_eq!(released.schema_version, 0);
        assert_eq!(released.authority_state, OwnerAuthorityInspectionState::Released);
        assert_eq!(released.root_record_sha256, root_record_sha256);
        assert_eq!(released.transition_records.len(), 1);
        assert_eq!(released.transition_records[0].predecessor_state_id, root_claim_id);
        assert_eq!(
            released.transition_records[0].record_sha256,
            file_sha256(&owner_transition_path(&paths.owner_transitions_directory, &root_claim_id))
                .expect("release transition hashes")
        );
        assert_eq!(released.final_leaf_record_sha256, released.transition_records[0].record_sha256);

        let mut reacquired_claim = paths.try_acquire_owner_claim().expect("released authority reacquires");
        let reacquired_claim_id = reacquired_claim.claim_id().to_string();
        let active =
            inspect_owner_authority(&paths, Some(&reacquired_claim_id)).expect("exact active authority leaf inspects");
        assert_eq!(active.schema_version, 0);
        assert_eq!(active.authority_state, OwnerAuthorityInspectionState::Active);
        assert_eq!(active.final_leaf_state_id, reacquired_claim_id);
        assert_eq!(active.root_record_sha256, root_record_sha256);
        assert_eq!(active.transition_records.len(), 2);
        assert_eq!(
            active.final_leaf_record_sha256,
            active.transition_records.last().expect("reacquisition transition exists").record_sha256
        );
        let inspection_path = paths.output_root.with_extension("inspection.json");
        write_helper_json(&inspection_path, &active);
        let inspection_bytes = std::fs::read(&inspection_path).expect("inspection JSON reads");
        assert_eq!(inspection_bytes.last(), Some(&b'\n'));
        assert_eq!(
            serde_json::from_slice::<OwnerAuthorityInspection>(&inspection_bytes).expect("inspection JSON round trips"),
            active
        );

        let acquisition_outcome = OwnerClaimHelperAcquisitionOutcome {
            schema_version: LINEAGE_SCHEMA_VERSION,
            status: OwnerClaimHelperAcquisitionStatus::Acquired,
            claim_id: Some(reacquired_claim_id),
            error: None,
        };
        assert_eq!(acquisition_outcome.schema_version, 0);
        assert_eq!(
            serde_json::from_slice::<OwnerClaimHelperAcquisitionOutcome>(
                &serde_json::to_vec_pretty(&acquisition_outcome).expect("acquisition outcome serializes")
            )
            .expect("acquisition outcome round trips"),
            acquisition_outcome
        );

        reacquired_claim.release().expect("reacquired claim releases");
        let _ = std::fs::remove_file(inspection_path);
        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn owner_authority_inspection_rejects_stale_expectations_and_malformed_transitions() {
        let paths = test_paths("owner-authority-inspection-rejection");
        let mut root_claim = paths.try_acquire_owner_claim().expect("root claim publishes");
        let root_claim_id = root_claim.claim_id().to_string();
        root_claim.release().expect("root claim releases");
        let transition_snapshot =
            snapshot_owner_transition_directory(&paths.owner_transitions_directory).expect("transition snapshot reads");

        let stale_error =
            inspect_owner_authority(&paths, Some(&root_claim_id)).expect_err("historical root is not the current leaf");
        assert!(stale_error.to_string().contains("expected leaf"));
        assert_eq!(
            snapshot_owner_transition_directory(&paths.owner_transitions_directory)
                .expect("stale inspection leaves transitions readable"),
            transition_snapshot
        );

        let transition_path = owner_transition_path(&paths.owner_transitions_directory, &root_claim_id);
        let mut malformed_transition =
            serde_json::from_slice::<serde_json::Value>(&std::fs::read(&transition_path).expect("transition reads"))
                .expect("transition JSON parses");
        malformed_transition["schema_version"] = serde_json::Value::from(1_u32);
        std::fs::write(
            &transition_path,
            serde_json::to_vec_pretty(&malformed_transition).expect("malformed transition serializes"),
        )
        .expect("test corrupts the immutable transition");
        let malformed_error =
            inspect_owner_authority(&paths, None).expect_err("unsupported transition schema is rejected");
        assert!(malformed_error.to_string().contains("unsupported schema version"));

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn owner_claim_blocks_a_second_process_and_survives_owner_kill() {
        use std::process::{Command, Stdio};

        let paths = test_paths("process-owner-claim");
        let ready_path = paths.output_root.with_extension("ready");
        let test_executable = std::env::current_exe().expect("current test executable resolves");
        let mut owner = Command::new(&test_executable)
            .args(["--exact", OWNER_CLAIM_HELPER_TEST_NAME, "--nocapture"])
            .env(OWNER_CLAIM_HELPER_MODE_ENVIRONMENT, "hold")
            .env(OWNER_CLAIM_HELPER_ROOT_ENVIRONMENT, &paths.output_root)
            .env(OWNER_CLAIM_HELPER_READY_ENVIRONMENT, &ready_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("claim owner starts");
        let readiness_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ready_path.exists() && std::time::Instant::now() < readiness_deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(ready_path.exists(), "claim owner did not report readiness");

        let blocked_status = Command::new(&test_executable)
            .args(["--exact", OWNER_CLAIM_HELPER_TEST_NAME, "--nocapture"])
            .env(OWNER_CLAIM_HELPER_MODE_ENVIRONMENT, "blocked")
            .env(OWNER_CLAIM_HELPER_ROOT_ENVIRONMENT, &paths.output_root)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("blocked claim contender runs");
        assert!(blocked_status.success());

        owner.kill().expect("claim owner is killed");
        owner.wait().expect("killed claim owner is reaped");
        let stale_status = Command::new(&test_executable)
            .args(["--exact", OWNER_CLAIM_HELPER_TEST_NAME, "--nocapture"])
            .env(OWNER_CLAIM_HELPER_MODE_ENVIRONMENT, "blocked")
            .env(OWNER_CLAIM_HELPER_ROOT_ENVIRONMENT, &paths.output_root)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("post-kill claim contender runs");
        assert!(stale_status.success());
        assert!(paths.owner_claim_path.is_file());
        assert!(!paths.attempts_directory.exists(), "a losing claim contender must not create an attempt tree");

        let _ = std::fs::remove_file(ready_path);
        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn owner_claim_release_requires_the_exact_record_and_allows_reacquisition() {
        let paths = test_paths("owner-claim-release");
        let mut claim = paths.try_acquire_owner_claim().expect("first owner claim publishes");
        claim.release().expect("matching owner claim releases");
        assert!(paths.owner_claim_path.exists(), "the immutable root claim remains");
        assert!(matches!(
            resolve_owner_authority(&paths.owner_claim_path, &paths.owner_transitions_directory)
                .expect("released authority resolves"),
            OutputOwnerAuthorityState::Released { .. }
        ));

        let mut replacement = paths.try_acquire_owner_claim().expect("released owner claim can be reacquired");
        let conflicting_record = OutputOwnerClaimRecord::new().expect("conflicting claim builds");
        let conflicting_transition =
            OutputOwnerTransitionRecord::fenced_takeover(replacement.claim_id().to_string(), conflicting_record)
                .expect("conflicting takeover builds");
        assert_eq!(
            publish_owner_transition(&paths.owner_transitions_directory, &conflicting_transition)
                .expect("conflicting transition publishes"),
            NoReplacePublication::Created
        );
        assert!(matches!(replacement.release(), Err(OutputError::ConcurrentLineageUpdate { .. })));
        assert!(paths.owner_claim_path.is_file(), "the immutable root claim must never be removed");

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn owner_claim_release_retries_transition_durability_without_republication() {
        let paths = test_paths("owner-claim-release-sync");
        let mut claim = paths.try_acquire_owner_claim().expect("owner claim publishes");
        let claim_id = claim.claim_id().to_string();
        crate::persistence::io::fail_owner_publication_syncs_for_test(3);
        let error = claim.release().expect_err("injected directory synchronization fails");
        assert!(error.to_string().contains("synchronization failure"));
        assert!(paths.owner_claim_path.exists(), "the immutable root claim remains");
        assert!(
            owner_transition_path(&paths.owner_transitions_directory, &claim_id).is_file(),
            "the first release attempt already published the exact release transition"
        );

        claim.release().expect("release retries only the missing durability step");
        claim.release().expect("durably released claim is idempotent");
        let mut replacement = paths.try_acquire_owner_claim().expect("durable release permits reacquisition");
        replacement.release().expect("replacement claim releases");

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn owner_publication_sync_failures_reconcile_visible_exact_authority() {
        let root_paths = test_paths("owner-root-publication-sync");
        crate::persistence::io::fail_owner_publication_syncs_for_test(1);
        let mut root_claim = root_paths.try_acquire_owner_claim().expect("visible exact root retries durability");
        root_claim.release().expect("root claim releases");

        let typed_paths = test_paths("owner-root-publication-unresolved");
        crate::persistence::io::fail_owner_publication_syncs_for_test(3);
        let typed_error = typed_paths.try_acquire_owner_claim().expect_err("three sync failures remain unresolved");
        let unresolved_claim_id = match typed_error {
            OutputError::PublishedOutputOwnerClaimDurability { claim_id, .. } => claim_id,
            unexpected => panic!("expected typed visible-authority durability error, got: {unexpected}"),
        };
        let mut recovery =
            typed_paths.take_over_fenced_owner_claim(&unresolved_claim_id).expect("exact visible claim is recoverable");
        recovery.release().expect("recovery claim releases");

        let takeover_paths = test_paths("owner-takeover-publication-sync");
        let initial = takeover_paths.try_acquire_owner_claim().expect("initial claim publishes");
        let initial_claim_id = initial.claim_id().to_string();
        drop(initial);
        crate::persistence::io::fail_owner_publication_syncs_for_test(3);
        let takeover_error = takeover_paths
            .take_over_fenced_owner_claim(&initial_claim_id)
            .expect_err("unconfirmed takeover reports its visible claim");
        let visible_takeover_id = match takeover_error {
            OutputError::PublishedOutputOwnerClaimDurability { claim_id, .. } => claim_id,
            unexpected => panic!("expected typed takeover durability error, got: {unexpected}"),
        };
        let mut final_takeover = takeover_paths
            .take_over_fenced_owner_claim(&visible_takeover_id)
            .expect("visible takeover identity supports exact recovery");
        final_takeover.release().expect("final takeover releases");

        let reacquire_paths = test_paths("owner-reacquire-publication-sync");
        let mut released = reacquire_paths.try_acquire_owner_claim().expect("initial claim publishes");
        released.release().expect("initial claim releases");
        crate::persistence::io::fail_owner_publication_syncs_for_test(3);
        let reacquire_error =
            reacquire_paths.try_acquire_owner_claim().expect_err("unconfirmed reacquisition names its visible claim");
        let visible_reacquire_id = match reacquire_error {
            OutputError::PublishedOutputOwnerClaimDurability { claim_id, .. } => claim_id,
            unexpected => panic!("expected typed reacquisition durability error, got: {unexpected}"),
        };
        let mut final_recovery = reacquire_paths
            .take_over_fenced_owner_claim(&visible_reacquire_id)
            .expect("visible reacquisition identity supports exact recovery");
        final_recovery.release().expect("reacquisition recovery releases");

        let _ = std::fs::remove_dir_all(root_paths.output_root);
        let _ = std::fs::remove_dir_all(typed_paths.output_root);
        let _ = std::fs::remove_dir_all(takeover_paths.output_root);
        let _ = std::fs::remove_dir_all(reacquire_paths.output_root);
    }

    #[test]
    fn owner_authority_file_sync_failures_never_reach_the_publication_link() {
        let root_paths = test_paths("owner-root-file-sync");
        crate::persistence::io::fail_owner_publication_file_syncs_for_test(2);
        let root_error =
            root_paths.try_acquire_owner_claim().expect_err("repeated root file-sync failures prevent publication");
        assert!(root_error.to_string().contains("temporary-file synchronization failure"));
        assert!(!root_paths.owner_claim_path.exists());

        let takeover_paths = test_paths("owner-takeover-file-sync");
        let mut initial_claim = takeover_paths.try_acquire_owner_claim().expect("initial claim publishes");
        let initial_claim_id = initial_claim.claim_id().to_string();
        crate::persistence::io::fail_owner_publication_file_syncs_for_test(2);
        let takeover_error = takeover_paths
            .take_over_fenced_owner_claim(&initial_claim_id)
            .expect_err("repeated takeover file-sync failures prevent publication");
        match takeover_error {
            OutputError::SurvivingOutputOwnerClaim { claim_id, .. } => assert_eq!(claim_id, initial_claim_id),
            unexpected => panic!("expected the unchanged surviving claim, got: {unexpected}"),
        }
        assert!(!owner_transition_path(&takeover_paths.owner_transitions_directory, &initial_claim_id).exists());
        initial_claim.release().expect("unchanged initial claim releases");

        let _ = std::fs::remove_dir_all(root_paths.output_root);
        let _ = std::fs::remove_dir_all(takeover_paths.output_root);
    }

    #[test]
    fn ambiguous_candidate_reread_reports_the_latest_authority_leaf() {
        let paths = test_paths("owner-ambiguous-candidate-reread");
        let root_claim = paths.try_acquire_owner_claim().expect("root claim publishes");
        let root_claim_id = root_claim.claim_id().to_string();
        drop(root_claim);
        let candidate = OutputOwnerClaimRecord::new().expect("ambiguous candidate builds");
        let candidate_transition =
            OutputOwnerTransitionRecord::fenced_takeover(root_claim_id, candidate.clone()).expect("takeover builds");
        assert_eq!(
            publish_owner_transition(&paths.owner_transitions_directory, &candidate_transition)
                .expect("candidate publishes"),
            NoReplacePublication::Created
        );
        let successor = OutputOwnerClaimRecord::new().expect("successor claim builds");
        let successor_transition =
            OutputOwnerTransitionRecord::fenced_takeover(candidate.claim_id.clone(), successor.clone())
                .expect("successor takeover builds");
        assert_eq!(
            publish_owner_transition(&paths.owner_transitions_directory, &successor_transition)
                .expect("successor publishes"),
            NoReplacePublication::Created
        );

        let publication_error = OutputError::Runtime("simulated ambiguous candidate publication failure".to_string());
        let reread_error = paths.owner_claim_publication_error(&candidate, &publication_error);
        match reread_error {
            OutputError::SurvivingOutputOwnerClaim { claim_id, .. } => assert_eq!(claim_id, successor.claim_id),
            unexpected => panic!("expected the latest surviving claim, got: {unexpected}"),
        }
        paths
            .owner_claim_from_record(successor)
            .expect("latest claim handle reconstructs")
            .release()
            .expect("latest claim releases");

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn authority_resolution_retraverses_when_the_leaf_advances_before_directory_validation() {
        let paths = test_paths("owner-retraversal");
        let root_claim = paths.try_acquire_owner_claim().expect("root claim publishes");
        let root_claim_id = root_claim.claim_id().to_string();
        drop(root_claim);
        let successor = OutputOwnerClaimRecord::new().expect("successor claim builds");
        let transition =
            OutputOwnerTransitionRecord::fenced_takeover(root_claim_id, successor.clone()).expect("takeover builds");
        let mut leaf_observation_count = 0_usize;
        let resolved = resolve_owner_authority_with_history_observed(
            &paths.owner_claim_path,
            &paths.owner_transitions_directory,
            &mut || {
                leaf_observation_count += 1;
                if leaf_observation_count == 1 {
                    assert_eq!(
                        publish_owner_transition(&paths.owner_transitions_directory, &transition)
                            .expect("transition publishes between traversal and validation"),
                        NoReplacePublication::Created
                    );
                }
                Ok(())
            },
        )
        .expect("authority retraversal resolves the advanced leaf");
        assert_eq!(leaf_observation_count, 2);
        assert_eq!(resolved.state, OutputOwnerAuthorityState::Active(successor.clone()));
        assert!(resolved.claim_identifiers.contains(&successor.claim_id));
        paths
            .owner_claim_from_record(successor)
            .expect("advanced claim handle reconstructs")
            .release()
            .expect("advanced claim releases");

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn authority_resolution_retraverses_across_repeated_valid_leaf_advances() {
        let paths = test_paths("owner-repeated-retraversal");
        let root_claim = paths.try_acquire_owner_claim().expect("root claim publishes");
        let root_claim_id = root_claim.claim_id().to_string();
        drop(root_claim);
        let release = OutputOwnerTransitionRecord::graceful_release(root_claim_id).expect("release transition builds");
        let released_state_id = match &release {
            OutputOwnerTransitionRecord::GracefulRelease { released_state_id, .. } => released_state_id.clone(),
            _ => unreachable!("the release constructor returns a graceful release"),
        };
        let successor = OutputOwnerClaimRecord::new().expect("successor claim builds");
        let reacquisition = OutputOwnerTransitionRecord::acquire_after_release(released_state_id, successor.clone())
            .expect("reacquisition transition builds");
        let mut leaf_observation_count = 0_usize;
        let resolved = resolve_owner_authority_with_history_observed(
            &paths.owner_claim_path,
            &paths.owner_transitions_directory,
            &mut || {
                leaf_observation_count += 1;
                let transition = match leaf_observation_count {
                    1 => Some(&release),
                    2 => Some(&reacquisition),
                    _ => None,
                };
                if let Some(transition) = transition {
                    assert_eq!(
                        publish_owner_transition(&paths.owner_transitions_directory, transition)
                            .expect("valid leaf advance publishes between traversal and validation"),
                        NoReplacePublication::Created
                    );
                }
                Ok(())
            },
        )
        .expect("authority retraversal resolves repeated valid advances");
        assert_eq!(leaf_observation_count, 3);
        assert_eq!(resolved.state, OutputOwnerAuthorityState::Active(successor.clone()));
        assert!(resolved.claim_identifiers.contains(&successor.claim_id));
        paths
            .owner_claim_from_record(successor)
            .expect("reacquired claim handle reconstructs")
            .release()
            .expect("reacquired claim releases");

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn release_recognizes_an_exact_visible_transition_after_sync_errors() {
        let paths = test_paths("owner-release-publication-sync");
        let mut claim = paths.try_acquire_owner_claim().expect("owner claim publishes");
        crate::persistence::io::fail_owner_publication_syncs_for_test(3);
        let error = claim.release().expect_err("publication durability remains unconfirmed");
        assert!(error.to_string().contains("synchronous retry"));
        assert!(claim.release_transition_is_visible());
        claim.release().expect("retry synchronizes the exact visible release");
        assert!(paths.owner_claim_path.is_file());

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn owner_transition_crash_windows_resolve_exact_visible_authority() {
        for failpoint in ["before_owner_transition_link", "after_owner_transition_link"] {
            let release_paths = test_paths(&format!("release-{failpoint}"));
            assert_owner_transition_crash(&run_owner_transition_crash_helper(
                &release_paths,
                "crash_release",
                failpoint,
                None,
            ));
            match resolve_owner_authority(&release_paths.owner_claim_path, &release_paths.owner_transitions_directory)
                .expect("release crash authority resolves")
            {
                OutputOwnerAuthorityState::Active(record) => {
                    assert_eq!(failpoint, "before_owner_transition_link");
                    let mut recovery = release_paths
                        .take_over_fenced_owner_claim(&record.claim_id)
                        .expect("pre-link release crash supports exact takeover");
                    recovery.release().expect("release recovery terminates");
                }
                OutputOwnerAuthorityState::Released { .. } => {
                    assert_eq!(failpoint, "after_owner_transition_link");
                    let mut recovery =
                        release_paths.try_acquire_owner_claim().expect("post-link release crash is reacquired");
                    recovery.release().expect("release recovery terminates");
                }
            }
            let _ = std::fs::remove_dir_all(release_paths.output_root);

            let takeover_paths = test_paths(&format!("takeover-{failpoint}"));
            let initial = takeover_paths.try_acquire_owner_claim().expect("takeover root claim publishes");
            let initial_claim_id = initial.claim_id().to_string();
            drop(initial);
            assert_owner_transition_crash(&run_owner_transition_crash_helper(
                &takeover_paths,
                "crash_takeover",
                failpoint,
                Some(&initial_claim_id),
            ));
            let current_claim_id = takeover_paths
                .current_owner_claim_identifier_for_test()
                .expect("takeover crash authority resolves")
                .expect("takeover crash leaves active authority");
            if failpoint == "before_owner_transition_link" {
                assert_eq!(current_claim_id, initial_claim_id);
            } else {
                assert_ne!(current_claim_id, initial_claim_id);
            }
            let mut recovery = takeover_paths
                .take_over_fenced_owner_claim(&current_claim_id)
                .expect("takeover crash supports exact current-leaf recovery");
            recovery.release().expect("takeover recovery terminates");
            let _ = std::fs::remove_dir_all(takeover_paths.output_root);

            let reacquire_paths = test_paths(&format!("reacquire-{failpoint}"));
            let mut initial = reacquire_paths.try_acquire_owner_claim().expect("reacquire root claim publishes");
            initial.release().expect("reacquire root claim releases");
            assert_owner_transition_crash(&run_owner_transition_crash_helper(
                &reacquire_paths,
                "crash_reacquire",
                failpoint,
                None,
            ));
            if failpoint == "before_owner_transition_link" {
                let mut recovery =
                    reacquire_paths.try_acquire_owner_claim().expect("pre-link reacquisition crash retries");
                recovery.release().expect("reacquisition recovery terminates");
            } else {
                let current_claim_id = reacquire_paths
                    .current_owner_claim_identifier_for_test()
                    .expect("reacquisition authority resolves")
                    .expect("post-link reacquisition leaves active authority");
                let mut recovery = reacquire_paths
                    .take_over_fenced_owner_claim(&current_claim_id)
                    .expect("post-link reacquisition supports exact takeover");
                recovery.release().expect("reacquisition recovery terminates");
            }
            let _ = std::fs::remove_dir_all(reacquire_paths.output_root);
        }
    }

    #[test]
    fn concurrent_fenced_recoverers_publish_exactly_one_successor_claim() {
        let paths = test_paths("owner-fenced-race");
        let owner = paths.try_acquire_owner_claim().expect("initial owner claim publishes");
        let fenced_claim_id = owner.claim_id().to_string();
        drop(owner);
        let winners = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let paths = &paths;
                let fenced_claim_id = &fenced_claim_id;
                let winners = &winners;
                scope.spawn(move || match paths.take_over_fenced_owner_claim(fenced_claim_id) {
                    Ok(claim) => winners.lock().expect("winner lock is available").push(claim),
                    Err(
                        OutputError::SurvivingOutputOwnerClaim { .. } | OutputError::ConcurrentLineageUpdate { .. },
                    ) => {}
                    Err(error) => panic!("unexpected fenced contender error: {error}"),
                });
            }
        });
        let mut winners = winners.into_inner().expect("winner lock unwraps");
        assert_eq!(winners.len(), 1);
        let mut winner = winners.pop().expect("one winner exists");
        assert_ne!(winner.claim_id(), fenced_claim_id);
        winner.release().expect("winning successor releases");

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn graceful_release_and_fenced_takeover_contend_on_one_predecessor_slot() {
        let paths = test_paths("owner-release-takeover-race");
        let mut owner = paths.try_acquire_owner_claim().expect("initial owner claim publishes");
        let fenced_claim_id = owner.claim_id().to_string();
        let takeover_winner = std::sync::Mutex::new(None);
        let release_succeeded = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let paths = &paths;
            let takeover_winner = &takeover_winner;
            let release_succeeded = &release_succeeded;
            scope.spawn(|| {
                if owner.release().is_ok() {
                    release_succeeded.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            });
            scope.spawn(|| {
                if let Ok(claim) = paths.take_over_fenced_owner_claim(&fenced_claim_id) {
                    *takeover_winner.lock().expect("takeover winner lock is available") = Some(claim);
                }
            });
        });
        let mut takeover_winner = takeover_winner.into_inner().expect("takeover winner lock unwraps");
        assert_ne!(
            usize::from(release_succeeded.load(std::sync::atomic::Ordering::SeqCst))
                + usize::from(takeover_winner.is_some()),
            0
        );
        assert_eq!(
            usize::from(release_succeeded.load(std::sync::atomic::Ordering::SeqCst))
                + usize::from(takeover_winner.is_some()),
            1
        );
        if let Some(mut claim) = takeover_winner.take() {
            claim.release().expect("takeover winner releases");
        }

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn two_process_release_and_takeover_have_exactly_one_transition_winner() {
        use std::process::{Command, Stdio};

        let paths = test_paths("owner-release-takeover-process-race");
        let coordination_directory = paths.output_root.with_extension("coordination");
        std::fs::create_dir_all(&coordination_directory).expect("coordination directory creates");
        let owner_ready = coordination_directory.join("owner.ready");
        let release_start = coordination_directory.join("release.start");
        let release_barrier_ready = coordination_directory.join("release.barrier.ready");
        let takeover_barrier_ready = coordination_directory.join("takeover.barrier.ready");
        let barrier_go = coordination_directory.join("barrier.go");
        let release_outcome = coordination_directory.join("release.outcome");
        let takeover_outcome = coordination_directory.join("takeover.outcome");
        let test_executable = std::env::current_exe().expect("current test executable resolves");

        let mut release_process = Command::new(&test_executable)
            .args(["--exact", OWNER_CLAIM_HELPER_TEST_NAME, "--nocapture"])
            .env(OWNER_CLAIM_HELPER_MODE_ENVIRONMENT, "race_release")
            .env(OWNER_CLAIM_HELPER_ROOT_ENVIRONMENT, &paths.output_root)
            .env(OWNER_CLAIM_HELPER_READY_ENVIRONMENT, &owner_ready)
            .env(OWNER_CLAIM_HELPER_START_ENVIRONMENT, &release_start)
            .env(OWNER_CLAIM_HELPER_OUTCOME_ENVIRONMENT, &release_outcome)
            .env("G_OUTPUT_OWNER_TRANSITION_BARRIER_READY", &release_barrier_ready)
            .env("G_OUTPUT_OWNER_TRANSITION_BARRIER_GO", &barrier_go)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("release process starts");
        wait_for_test_path(&owner_ready, "owner claim");
        let fenced_claim_id = std::fs::read_to_string(&owner_ready).expect("owner claim identifier reads");

        let mut takeover_process = Command::new(&test_executable)
            .args(["--exact", OWNER_CLAIM_HELPER_TEST_NAME, "--nocapture"])
            .env(OWNER_CLAIM_HELPER_MODE_ENVIRONMENT, "race_takeover")
            .env(OWNER_CLAIM_HELPER_ROOT_ENVIRONMENT, &paths.output_root)
            .env(OWNER_CLAIM_HELPER_FENCE_ENVIRONMENT, &fenced_claim_id)
            .env(OWNER_CLAIM_HELPER_OUTCOME_ENVIRONMENT, &takeover_outcome)
            .env("G_OUTPUT_OWNER_TRANSITION_BARRIER_READY", &takeover_barrier_ready)
            .env("G_OUTPUT_OWNER_TRANSITION_BARRIER_GO", &barrier_go)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("takeover process starts");
        std::fs::write(&release_start, b"go").expect("release process starts publication");
        wait_for_test_path(&release_barrier_ready, "release publication barrier");
        wait_for_test_path(&takeover_barrier_ready, "takeover publication barrier");
        std::fs::write(&barrier_go, b"go").expect("transition contenders are released together");

        assert!(release_process.wait().expect("release process is reaped").success());
        assert!(takeover_process.wait().expect("takeover process is reaped").success());
        let release_result = std::fs::read_to_string(&release_outcome).expect("release outcome reads");
        let takeover_result = std::fs::read_to_string(&takeover_outcome).expect("takeover outcome reads");
        assert_eq!(
            usize::from(release_result.starts_with("ok")) + usize::from(takeover_result.starts_with("ok")),
            1,
            "release={release_result}; takeover={takeover_result}"
        );
        match resolve_owner_authority(&paths.owner_claim_path, &paths.owner_transitions_directory)
            .expect("winning transition resolves")
        {
            OutputOwnerAuthorityState::Released { .. } => assert_eq!(release_result, "ok"),
            OutputOwnerAuthorityState::Active(record) => {
                assert_eq!(takeover_result, format!("ok:{}", record.claim_id));
            }
        }
        assert!(paths.owner_claim_path.is_file(), "root claim remains immutable");

        let _ = std::fs::remove_dir_all(paths.output_root);
        let _ = std::fs::remove_dir_all(coordination_directory);
    }

    #[test]
    fn released_authority_admits_exactly_one_concurrent_acquisition() {
        let paths = test_paths("owner-acquire-race");
        let mut owner = paths.try_acquire_owner_claim().expect("initial owner claim publishes");
        owner.release().expect("initial owner releases");
        let winners = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let paths = &paths;
                let winners = &winners;
                scope.spawn(move || match paths.try_acquire_owner_claim() {
                    Ok(claim) => winners.lock().expect("winner lock is available").push(claim),
                    Err(
                        OutputError::SurvivingOutputOwnerClaim { .. } | OutputError::ConcurrentLineageUpdate { .. },
                    ) => {}
                    Err(error) => panic!("unexpected acquisition contender error: {error}"),
                });
            }
        });
        let mut winners = winners.into_inner().expect("winner lock unwraps");
        assert_eq!(winners.len(), 1);
        winners.pop().expect("one winner exists").release().expect("winner releases");

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn takeover_chain_requires_fencing_the_current_successor() {
        let paths = test_paths("owner-takeover-chain");
        let first = paths.try_acquire_owner_claim().expect("first owner publishes");
        let first_identifier = first.claim_id().to_string();
        drop(first);
        let second = paths.take_over_fenced_owner_claim(&first_identifier).expect("first takeover publishes");
        let second_identifier = second.claim_id().to_string();
        drop(second);
        let stale_error =
            paths.take_over_fenced_owner_claim(&first_identifier).expect_err("historical fence is rejected");
        assert!(stale_error.to_string().contains(&second_identifier));
        let mut third = paths.take_over_fenced_owner_claim(&second_identifier).expect("second takeover publishes");
        third.release().expect("current successor releases");

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn owner_authority_accepts_exactly_the_documented_transition_limit() {
        let paths = test_paths("owner-transition-limit");
        paths.initialize_directories().expect("lineage directories initialize");
        let root = OutputOwnerClaimRecord {
            schema_version: LINEAGE_SCHEMA_VERSION,
            claim_id: "owner-chain-0000".to_string(),
            host_name: "test-host".to_string(),
            process_id: 1,
        };
        std::fs::write(&paths.owner_claim_path, serde_json::to_vec(&root).expect("root serializes"))
            .expect("root writes");
        let mut predecessor = root.claim_id;
        for transition_index in 1..=MAXIMUM_OWNER_TRANSITION_COUNT {
            let claim = OutputOwnerClaimRecord {
                schema_version: LINEAGE_SCHEMA_VERSION,
                claim_id: format!("owner-chain-{transition_index:04}"),
                host_name: "test-host".to_string(),
                process_id: 1,
            };
            let transition = OutputOwnerTransitionRecord::fenced_takeover(predecessor.clone(), claim.clone())
                .expect("transition builds");
            std::fs::write(
                owner_transition_path(&paths.owner_transitions_directory, &predecessor),
                serde_json::to_vec(&transition).expect("transition serializes"),
            )
            .expect("transition writes");
            predecessor = claim.claim_id;
        }
        assert!(matches!(
            resolve_owner_authority(&paths.owner_claim_path, &paths.owner_transitions_directory)
                .expect("exact transition limit is supported"),
            OutputOwnerAuthorityState::Active(record) if record.claim_id == predecessor
        ));

        let overflow_claim = OutputOwnerClaimRecord {
            schema_version: LINEAGE_SCHEMA_VERSION,
            claim_id: "owner-chain-overflow".to_string(),
            host_name: "test-host".to_string(),
            process_id: 1,
        };
        let overflow_transition =
            OutputOwnerTransitionRecord::fenced_takeover(predecessor.clone(), overflow_claim).expect("overflow builds");
        std::fs::write(
            owner_transition_path(&paths.owner_transitions_directory, &predecessor),
            serde_json::to_vec(&overflow_transition).expect("overflow serializes"),
        )
        .expect("overflow writes");
        let error = resolve_owner_authority(&paths.owner_claim_path, &paths.owner_transitions_directory)
            .expect_err("one transition beyond the limit is rejected");
        assert!(error.to_string().contains("4096 transition limit"));

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn malformed_owner_claim_fails_closed() {
        let paths = test_paths("malformed-owner-claim");
        paths.initialize_directories().expect("lineage directories initialize");
        std::fs::write(
            &paths.owner_claim_path,
            br#"{
                "schema_version": 0,
                "claim_id": "owner-valid",
                "host_name": "node",
                "process_id": 1,
                "unexpected": true
            }"#,
        )
        .expect("malformed owner claim writes");

        let error = paths.reject_surviving_owner_claim().expect_err("unknown claim fields are rejected");
        assert!(matches!(error, OutputError::InvalidInput(_)));
        assert!(paths.owner_claim_path.is_file(), "a malformed surviving claim must not be removed");

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn unanchored_owner_transitions_and_staging_fail_closed() {
        let transition_paths = test_paths("unanchored-owner-transition");
        let _owner = transition_paths.try_acquire_owner_claim().expect("root claim publishes");
        let orphan_transition =
            OutputOwnerTransitionRecord::graceful_release("owner-unanchored".to_string()).expect("transition builds");
        publish_owner_transition(&transition_paths.owner_transitions_directory, &orphan_transition)
            .expect("unanchored transition bytes publish");
        let transition_error =
            transition_paths.reject_surviving_owner_claim().expect_err("unanchored transition is rejected");
        assert!(transition_error.to_string().contains("not reachable"));

        let staging_paths = test_paths("unanchored-owner-staging");
        staging_paths.initialize_directories().expect("lineage directories initialize");
        staging_paths
            .publish_owner_staging_intent("owner-unanchored", &AttemptIdentifier::for_test("attempt-unanchored"))
            .expect("orphan staging bytes publish");
        let staging_error = staging_paths.owner_staging_bindings().expect_err("orphan staging is rejected");
        assert!(staging_error.to_string().contains("authority history"));

        let _ = std::fs::remove_dir_all(transition_paths.output_root);
        let _ = std::fs::remove_dir_all(staging_paths.output_root);
    }

    #[test]
    fn owner_transition_directory_rejects_nonfiles_but_ignores_valid_candidates() {
        let paths = test_paths("owner-transition-artifacts");
        let _owner = paths.try_acquire_owner_claim().expect("root claim publishes");
        std::fs::create_dir(&paths.owner_transitions_directory).expect("owner transition directory creates");
        let candidate_path = paths
            .owner_transitions_directory
            .join(".owner-candidate.json.attempt-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tmp");
        std::fs::write(candidate_path, b"incomplete").expect("candidate writes");
        paths.reject_surviving_owner_claim().expect_err("active owner remains visible");

        let unsupported_directory = paths.owner_transitions_directory.join("unsupported");
        std::fs::create_dir(&unsupported_directory).expect("unsupported directory creates");
        let error = paths.reject_surviving_owner_claim().expect_err("non-file transition artifact is rejected");
        assert!(error.to_string().contains("non-file artifact"));

        let _ = std::fs::remove_dir_all(paths.output_root);
    }

    #[test]
    fn ordinary_reacquisition_sweeps_obsolete_staging_without_deleting_referenced_attempts() {
        let unreferenced_paths = test_paths("obsolete-unreferenced-staging");
        let mut first = unreferenced_paths.try_acquire_owner_claim().expect("first claim publishes");
        let staged_attempt = AttemptIdentifier::for_test("attempt-staged");
        unreferenced_paths
            .publish_owner_staging_intent(first.claim_id(), &staged_attempt)
            .expect("staging intent publishes");
        create_directories_durable(&unreferenced_paths.attempt_directory(&staged_attempt).join("diagnostics"))
            .expect("staged attempt creates");
        first.release().expect("first claim releases without staging cleanup");
        let mut second = unreferenced_paths.try_acquire_owner_claim().expect("second claim acquires");
        unreferenced_paths
            .cleanup_obsolete_owner_staging(second.claim_id(), &BTreeSet::new())
            .expect("ordinary reacquisition sweeps obsolete staging");
        assert!(!unreferenced_paths.attempt_directory(&staged_attempt).exists());
        assert!(unreferenced_paths.owner_staging_bindings().expect("staging bindings inspect").is_empty());
        second.release().expect("second claim releases");

        let referenced_paths = test_paths("obsolete-referenced-staging");
        let mut first = referenced_paths.try_acquire_owner_claim().expect("first claim publishes");
        let referenced_attempt = AttemptIdentifier::for_test("attempt-referenced");
        referenced_paths
            .publish_owner_staging_intent(first.claim_id(), &referenced_attempt)
            .expect("referenced staging intent publishes");
        create_directories_durable(&referenced_paths.attempt_directory(&referenced_attempt).join("diagnostics"))
            .expect("referenced attempt creates");
        first.release().expect("first referenced claim releases");
        let mut second = referenced_paths.try_acquire_owner_claim().expect("second referenced claim acquires");
        referenced_paths
            .cleanup_obsolete_owner_staging(second.claim_id(), &BTreeSet::from([referenced_attempt.clone()]))
            .expect("referenced staging intent retires");
        assert!(referenced_paths.attempt_directory(&referenced_attempt).is_dir());
        assert!(referenced_paths.owner_staging_bindings().expect("staging bindings inspect").is_empty());
        second.release().expect("second referenced claim releases");

        let _ = std::fs::remove_dir_all(unreferenced_paths.output_root);
        let _ = std::fs::remove_dir_all(referenced_paths.output_root);
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

    #[cfg(unix)]
    #[test]
    fn unmaterialized_attempt_exemption_accepts_only_true_absence() {
        let paths = test_paths("unmaterialized-attempt-path-kind");
        paths.initialize_directories().expect("lineage directories initialize");
        let attempt = AttemptIdentifier::for_test("attempt-unmaterialized");
        let genesis = LineageGenesisRecord::new(attempt.clone(), digest('b'), vec![phenotype_contract()]);
        paths.publish_genesis(&genesis).expect("genesis publishes");
        paths.inspect().expect("a genuinely absent active attempt is allowed");

        let attempt_path = paths.attempt_directory(&attempt);
        std::fs::write(&attempt_path, b"not a directory").expect("regular file occupies attempt path");
        let file_error = paths.inspect().expect_err("regular file cannot use the absence exemption");
        assert!(file_error.to_string().contains("is not a directory"));

        std::fs::remove_file(&attempt_path).expect("regular file removes");
        std::os::unix::fs::symlink(paths.output_root.join("missing-target"), &attempt_path)
            .expect("broken symlink occupies attempt path");
        let symlink_error = paths.inspect().expect_err("broken symlink cannot use the absence exemption");
        assert!(symlink_error.to_string().contains("is not a directory"));

        let _ = std::fs::remove_dir_all(paths.output_root);
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

    #[test]
    fn immutable_lineage_records_reject_unknown_fields_uppercase_hashes_and_unsafe_paths() {
        let paths = test_paths("strict-records");
        paths.initialize_directories().expect("lineage directories initialize");
        let attempt = AttemptIdentifier::for_test("attempt-strict");
        create_directories_durable(&paths.attempt_directory(&attempt)).expect("attempt creates");
        let genesis = LineageGenesisRecord::new(attempt.clone(), digest('b'), vec![phenotype_contract()]);
        let mut forged_genesis = serde_json::to_value(&genesis).expect("genesis serializes");
        forged_genesis
            .as_object_mut()
            .expect("genesis is an object")
            .insert("ignored_authority".to_string(), serde_json::Value::Bool(true));
        publish_json_no_replace(&paths.genesis_path, &forged_genesis).expect("forged artifact publishes");
        let error = paths.inspect().expect_err("unknown immutable fields are rejected");
        assert!(error.to_string().contains("unknown field"));

        let uppercase_hash = LineageGenesisRecord::new(attempt.clone(), digest('A'), vec![phenotype_contract()]);
        assert!(uppercase_hash.validate().is_err());
        let mut unsafe_contract = phenotype_contract();
        unsafe_contract.output_directory_name = "../escape".to_string();
        let unsafe_path = LineageGenesisRecord::new(attempt, digest('b'), vec![unsafe_contract]);
        assert!(unsafe_path.validate().is_err());
    }

    #[test]
    fn terminal_signal_and_failure_details_must_not_be_empty() {
        let phenotype = TerminalPhenotypeRecord {
            phenotype_name: "trait-a".to_string(),
            output_directory_name: "trait_0001_trait-a".to_string(),
            run_manifest_sha256: digest('c'),
        };
        let interrupted = LineageTerminalRecord::interrupted(
            "run-set-test".to_string(),
            AttemptIdentifier::for_test("attempt-interrupted"),
            " ".to_string(),
            vec![phenotype.clone()],
        );
        assert!(interrupted.validate().is_err());
        let failed = LineageTerminalRecord::failed(
            "run-set-test".to_string(),
            AttemptIdentifier::for_test("attempt-failed"),
            String::new(),
            vec![phenotype],
        );
        assert!(failed.validate().is_err());
    }
}
