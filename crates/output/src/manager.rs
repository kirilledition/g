//! Run-scoped append-only output lifecycle ownership.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::marker::PhantomData;
use std::ops::Range;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{OutputError, OutputResult};
use crate::manifest::{
    CurrentRunManifestHeaderInput, ManifestFileFingerprintCache, build_current_run_manifest_header_value_with_cache,
};
use crate::persistence::attempt::{
    AttemptManifestBinding, AttemptManifestStatus, AttemptManifestWrite, AttemptRunPaths, OrphanPartPolicy,
    VerifiedAttemptRun, attempt_manifest_value_sha256, build_attempt_manifest_value,
    inspect_unmaterialized_attempt_run, materialize_attempt_manifest, parse_attempt_manifest_json,
    read_optional_attempt_manifest_bytes, reuse_verified_receipts, validate_attempt_manifest_schema_zero,
    verify_attempt_run, write_attempt_manifest, write_effective_config,
};
use crate::persistence::identifier::{AttemptIdentifier, validate_safe_path_component};
use crate::persistence::io::{create_directories_durable, sync_nearest_existing_directory};
use crate::persistence::lineage::{
    AttemptTerminalStatus, LineageGenesisRecord, LineageRecoveryKind, LineageSnapshot, LineageSuccessorRecord,
    LineageTerminalRecord, OutputLineagePaths, OutputOwnerClaim, OutputOwnerConditionalRelease,
    PhenotypeLineageContract, TerminalPhenotypeRecord, terminal_record_sha256,
};
use crate::persistence::model::CanonicalChunkPlan;
use crate::persistence::receipt::OutputPartReceipt;
use crate::session::{
    CreatedOutputWriterSessions, OutputWriterResourceOwner, OutputWriterRunConfig, OutputWriterSession,
    create_output_writer_sessions, finish_interrupted_output_writer_sessions, finish_output_writer_sessions,
    validate_output_writer_settings,
};

/// Output paths returned after a verified completed terminal.
#[derive(Debug, Eq, PartialEq)]
pub struct CompletedOutputRun {
    pub run_directory: std::path::PathBuf,
    pub parts_directory: std::path::PathBuf,
}

/// Completed output artifacts plus optional post-session ownership cleanup.
#[derive(Debug)]
#[must_use = "completed output cleanup must run after claim-scoped diagnostics close"]
pub struct OutputCompletion {
    pub completed_outputs: Vec<CompletedOutputRun>,
    pub post_session_cleanup: Option<OutputPostSessionCleanup>,
}

/// Non-cloneable, retryable cleanup for a verified completed no-op claim.
#[must_use = "completed output cleanup must run after claim-scoped diagnostics close"]
pub struct OutputPostSessionCleanup {
    lineage_paths: OutputLineagePaths,
    owner_release: OutputOwnerConditionalRelease,
    claim_id: String,
    attempt_id: AttemptIdentifier,
    completed: bool,
    #[cfg(test)]
    cleanup_pause: Option<Arc<OutputPostSessionCleanupTestPause>>,
}

impl fmt::Debug for OutputPostSessionCleanup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("OutputPostSessionCleanup").finish_non_exhaustive()
    }
}

/// Terminal output failure plus optional post-session ownership cleanup.
#[derive(Debug, thiserror::Error)]
#[error("{source}")]
#[must_use = "terminal failures can carry completed output cleanup that must run after diagnostics close"]
pub struct OutputTerminalError {
    #[source]
    source: OutputError,
    post_session_cleanup: Option<Box<OutputPostSessionCleanup>>,
}

/// Primary terminal failure plus optional completed-noop cleanup authority.
#[must_use = "terminal failure parts can carry cleanup authority that must run after diagnostics close"]
pub struct OutputTerminalFailureParts {
    /// Primary terminal failure.
    pub source: OutputError,
    /// Cleanup required after claim-scoped diagnostics close.
    pub post_session_cleanup: Option<OutputPostSessionCleanup>,
}

impl OutputTerminalError {
    /// Separate the primary failure from any completed-noop cleanup authority.
    pub fn into_parts(self) -> OutputTerminalFailureParts {
        OutputTerminalFailureParts {
            source: self.source,
            post_session_cleanup: self.post_session_cleanup.map(|cleanup| *cleanup),
        }
    }

    fn with_cleanup(source: OutputError, post_session_cleanup: Option<OutputPostSessionCleanup>) -> Self {
        Self { source, post_session_cleanup: post_session_cleanup.map(Box::new) }
    }
}

impl From<OutputError> for OutputTerminalError {
    fn from(source: OutputError) -> Self {
        Self::with_cleanup(source, None)
    }
}

/// Exclusive unpublished-output ownership retained across session teardown.
#[must_use = "dropping rollback authority leaves the output claim active and requires explicit fencing"]
pub struct OutputClaimRollback(Option<OutputManager<Claimed>>);

impl fmt::Debug for OutputClaimRollback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("OutputClaimRollback").finish_non_exhaustive()
    }
}

impl OutputClaimRollback {
    /// Remove unpublished claim staging and release output ownership.
    ///
    /// Run-scoped diagnostics must be closed before calling this method.
    ///
    /// # Errors
    ///
    /// Returns an error when staging cleanup or owner release fails. Retry this
    /// same value after a transient cleanup or durability failure.
    pub fn abort_before_activation(&mut self) -> OutputResult<()> {
        let Some(manager) = self.0.as_mut() else {
            return Ok(());
        };
        manager.abort_before_activation_in_place()?;
        self.0 = None;
        Ok(())
    }
}

/// Failure while converting an unpublished output claim into active authority.
#[derive(Debug, thiserror::Error)]
#[must_use = "deferred activation failures must be inspected and any rollback consumed after diagnostics close"]
pub enum OutputActivationError {
    /// Attempt authority was not published and ownership must be rolled back
    /// after the claim-scoped diagnostics session closes.
    #[error("{source}")]
    Unpublished {
        #[source]
        source: OutputError,
        rollback: Box<OutputClaimRollback>,
    },
    /// Attempt authority was published and failure recovery already ran.
    #[error(transparent)]
    Published(#[from] OutputError),
}

/// Primary activation failure plus optional deferred rollback authority.
#[must_use = "activation failure parts can carry rollback authority that must be consumed after diagnostics close"]
pub struct OutputActivationFailureParts {
    /// Primary activation failure.
    pub source: OutputError,
    /// Ownership rollback required after claim-scoped diagnostics close.
    pub rollback: Option<OutputClaimRollback>,
}

impl OutputActivationError {
    /// Separate the primary failure from any deferred ownership rollback.
    pub fn into_parts(self) -> OutputActivationFailureParts {
        match self {
            Self::Unpublished { source, rollback } => {
                OutputActivationFailureParts { source, rollback: Some(*rollback) }
            }
            Self::Published(source) => OutputActivationFailureParts { source, rollback: None },
        }
    }
}

#[cfg(test)]
struct OutputPostSessionCleanupTestPause {
    reached_sender: std::sync::mpsc::Sender<()>,
    resume_receiver: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
}

#[cfg(test)]
struct OutputManifestHintTestPause {
    reached_sender: std::sync::mpsc::Sender<()>,
    resume_receiver: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
}

#[cfg(test)]
pub(crate) struct OutputManifestHintTestControl {
    reached_receiver: std::sync::mpsc::Receiver<()>,
    resume_sender: std::sync::mpsc::Sender<()>,
    resumed: std::sync::atomic::AtomicBool,
}

#[cfg(test)]
pub(crate) struct OutputPostSessionCleanupTestControl {
    reached_receiver: std::sync::mpsc::Receiver<()>,
    resume_sender: std::sync::mpsc::Sender<()>,
    resumed: std::sync::atomic::AtomicBool,
}

#[cfg(test)]
const OUTPUT_POST_SESSION_CLEANUP_TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

#[cfg(test)]
const OUTPUT_MANIFEST_HINT_TEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Marker for a read-only output plan.
pub struct Planned;

/// Marker for an exclusively claimed output plan that has no attempt authority.
pub struct Claimed;

/// Marker for an initialized writable or verified read-only attempt.
pub struct Active;

/// Marker proving that every phenotype has exact chunk coverage.
pub struct Covered;

/// Opaque output capability for one phenotype group.
pub struct OutputDeliveryToken {
    writer_sessions: Vec<Option<Arc<OutputWriterSession>>>,
    committed_chunk_identifier_sets: Vec<Arc<BTreeSet<usize>>>,
}

/// Typestate owner for one append-only output lineage.
pub struct OutputManager<State = Planned> {
    core: Option<OutputManagerCore>,
    state: PhantomData<State>,
}

struct OutputManagerCore {
    run_plan: Arc<g_plan::RunPlan>,
    effective_config_toml: String,
    lineage_paths: OutputLineagePaths,
    lineage_snapshot: Option<LineageSnapshot>,
    runs: Vec<ManagedOutputRun>,
    run_indices_by_phenotype: BTreeMap<String, usize>,
    active_attempt: Option<ActiveAttempt>,
    writer_resource_owner: Option<OutputWriterResourceOwner>,
    owner_claim: Option<OutputOwnerClaim>,
    claim_staging_directory: Option<PathBuf>,
    claimed_attempt_id: Option<AttemptIdentifier>,
    completed_noop_cleanup: Option<OutputPostSessionCleanup>,
    canonical_chunk_plan: Option<CanonicalChunkPlan>,
    collect_stage_timings: bool,
    lifecycle_state: OutputManagerLifecycleState,
    #[cfg(test)]
    manifest_hint_pause: Option<Arc<OutputManifestHintTestPause>>,
}

struct OutputManagerLifecycleState {
    attempt_authority_publication: AttemptAuthorityPublication,
    completed_noop_cleanup_policy: CompletedNoopCleanupPolicy,
    terminal: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AttemptAuthorityPublication {
    NotAttempted,
    DefinitelyUnpublished { record_path: PathBuf },
    VisibleOrUnknown { record_path: PathBuf },
}

impl AttemptAuthorityPublication {
    fn rollback_is_safe(&self) -> bool {
        matches!(self, Self::NotAttempted | Self::DefinitelyUnpublished { .. })
    }

    fn ensure_rollback_target_absent(&self) -> OutputResult<()> {
        let Self::DefinitelyUnpublished { record_path } = self else {
            return if matches!(self, Self::NotAttempted) {
                Ok(())
            } else {
                Err(OutputError::InvalidInput(
                    "Output attempt staging cannot be removed after its immutable authority may be visible."
                        .to_string(),
                ))
            };
        };
        match std::fs::symlink_metadata(record_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Ok(_) => Err(OutputError::InvalidInput(format!(
                "Output attempt staging cannot be removed because immutable authority appeared at '{}'.",
                record_path.display()
            ))),
            Err(error) => Err(OutputError::Runtime(format!(
                "Output attempt staging cannot be removed because immutable authority at '{}' could not be disproved: {error}",
                record_path.display()
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompletedNoopCleanupPolicy {
    Immediate,
    Deferred,
}

struct ManagedOutputRun {
    phenotype_name: String,
    output_directory_name: String,
    current_header: Option<Value>,
    binding: Option<AttemptManifestBinding>,
    paths: Option<AttemptRunPaths>,
    receipts: Vec<OutputPartReceipt>,
    committed_chunk_identifiers: Arc<BTreeSet<usize>>,
    writer_session: Option<Arc<OutputWriterSession>>,
}

struct ActiveAttempt {
    run_set_id: String,
    attempt_id: AttemptIdentifier,
    canonical_chunk_plan: CanonicalChunkPlan,
    completed_noop: bool,
    collect_stage_timings: bool,
}

struct PreparedAttempt {
    run_set_id: String,
    attempt_id: AttemptIdentifier,
    runs: Vec<PreparedAttemptRun>,
}

struct PreparedAttemptRun {
    binding: AttemptManifestBinding,
    paths: AttemptRunPaths,
    receipts: Vec<OutputPartReceipt>,
    committed_chunk_identifiers: BTreeSet<usize>,
}

struct StagedTerminalRun {
    manifest: Value,
    manifest_sha256: String,
}

struct StagedTerminalRuns {
    terminal: LineageTerminalRecord,
    runs: Vec<StagedTerminalRun>,
}

impl OutputManager<Planned> {
    /// Inspect one output root without mutating it.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate output names, incompatible resume
    /// controls, or malformed lineage records.
    pub fn open(run_plan: Arc<g_plan::RunPlan>, effective_config_toml: String) -> OutputResult<Self> {
        if contains_cli_line_separator(&run_plan.output.output_run_root) {
            return Err(OutputError::InvalidInput(
                "Output run root must not contain characters that split CLI output lines.".to_string(),
            ));
        }
        let output_root = absolute_lexically_normalized_path(Path::new(&run_plan.output.output_run_root))?;
        let output_root_text = output_root.to_str().ok_or_else(|| {
            OutputError::InvalidInput("Resolved output run root must be valid UTF-8 for CLI publication.".to_string())
        })?;
        if contains_cli_line_separator(output_root_text) {
            return Err(OutputError::InvalidInput(
                "Resolved output run root must not contain characters that split CLI output lines.".to_string(),
            ));
        }
        let lineage_paths = OutputLineagePaths::new(&output_root);
        let mut runs = Vec::with_capacity(run_plan.phenotype_runs.len());
        let mut run_indices_by_phenotype = BTreeMap::new();
        let mut output_directory_names = BTreeSet::new();
        for phenotype_run in &run_plan.phenotype_runs {
            validate_safe_path_component(&phenotype_run.output_directory_name, "phenotype directory name")?;
            let output_index = runs.len();
            if run_indices_by_phenotype.insert(phenotype_run.phenotype_name.clone(), output_index).is_some() {
                return Err(OutputError::InvalidInput(format!(
                    "Duplicate phenotype output name '{}'.",
                    phenotype_run.phenotype_name
                )));
            }
            if !output_directory_names.insert(phenotype_run.output_directory_name.clone()) {
                return Err(OutputError::InvalidInput(format!(
                    "Duplicate phenotype output directory name '{}'.",
                    phenotype_run.output_directory_name
                )));
            }
            runs.push(ManagedOutputRun {
                phenotype_name: phenotype_run.phenotype_name.clone(),
                output_directory_name: phenotype_run.output_directory_name.clone(),
                current_header: None,
                binding: None,
                paths: None,
                receipts: Vec::new(),
                committed_chunk_identifiers: Arc::new(BTreeSet::new()),
                writer_session: None,
            });
        }
        let lineage_snapshot = lineage_paths.inspect()?;
        validate_open_policy(&run_plan, &lineage_paths, lineage_snapshot.as_ref(), &runs)?;
        Ok(Self {
            core: Some(OutputManagerCore {
                run_plan,
                effective_config_toml,
                lineage_paths,
                lineage_snapshot,
                runs,
                run_indices_by_phenotype,
                active_attempt: None,
                writer_resource_owner: None,
                owner_claim: None,
                claim_staging_directory: None,
                claimed_attempt_id: None,
                completed_noop_cleanup: None,
                canonical_chunk_plan: None,
                collect_stage_timings: false,
                lifecycle_state: OutputManagerLifecycleState {
                    attempt_authority_publication: AttemptAuthorityPublication::NotAttempted,
                    completed_noop_cleanup_policy: CompletedNoopCleanupPolicy::Immediate,
                    terminal: false,
                },
                #[cfg(test)]
                manifest_hint_pause: None,
            }),
            state: PhantomData,
        })
    }

    /// Return the genotype representation recorded by an existing leaf manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when the phenotype is unknown or its leaf manifest is
    /// missing or invalid.
    pub fn existing_manifest_gpu_genotype_format(
        &self,
        phenotype_name: &str,
    ) -> OutputResult<Option<g_plan::GpuGenotypeFormat>> {
        let core = self.core()?;
        let Some(snapshot) = core.lineage_paths.inspect()? else {
            return Ok(None);
        };
        let run = core.run(phenotype_name)?;
        let contract = snapshot
            .genesis
            .phenotypes
            .iter()
            .find(|contract| contract.phenotype_name == phenotype_name)
            .ok_or_else(|| {
                OutputError::InvalidInput(format!("Output immutable lineage is missing phenotype '{phenotype_name}'."))
            })?;
        if contract.output_directory_name != run.output_directory_name {
            return Err(OutputError::InvalidInput(format!(
                "Output immutable lineage directory for phenotype '{phenotype_name}' does not match the current plan."
            )));
        }
        let paths = AttemptRunPaths::new(
            &core.lineage_paths.attempts_directory,
            &snapshot.leaf_attempt_id,
            &contract.output_directory_name,
        )?;
        let terminal_authority_exists = snapshot.leaf_terminal.is_some() || snapshot.pending_terminal.is_some();
        let manifest_bytes = match read_optional_attempt_manifest_bytes(&paths.manifest_path)? {
            Some(manifest_bytes) => manifest_bytes,
            None if !terminal_authority_exists => return Ok(None),
            None => {
                return Err(OutputError::InvalidInput(format!(
                    "Terminal output attempt is missing manifest '{}'.",
                    paths.manifest_path.display()
                )));
            }
        };
        let manifest_sha256 = hex::encode(Sha256::digest(&manifest_bytes));
        let manifest = parse_attempt_manifest_json(&manifest_bytes, &paths.manifest_path)?;
        let binding = AttemptManifestBinding {
            run_set_id: snapshot.genesis.run_set_id.clone(),
            attempt_id: snapshot.leaf_attempt_id.clone(),
            phenotype_name: contract.phenotype_name.clone(),
            output_directory_name: contract.output_directory_name.clone(),
            execution_plan_sha256: contract.execution_plan_sha256.clone(),
            chunk_plan_sha256: snapshot.genesis.chunk_plan_sha256.clone(),
        };
        let validated = validate_attempt_manifest_schema_zero(manifest, &paths, &binding)?;
        #[cfg(test)]
        if let Some(pause) = core.manifest_hint_pause.as_ref() {
            pause.wait()?;
        }
        if core.lineage_paths.inspect()?.as_ref() != Some(&snapshot) {
            return Err(OutputError::ConcurrentLineageUpdate { record_path: core.lineage_paths.genesis_path.clone() });
        }
        if let Some(terminal) = snapshot.leaf_terminal.as_ref() {
            validate_manifest_terminal_status(&validated.status, terminal.status)?;
            let terminal_phenotype =
                terminal.phenotypes.iter().find(|record| record.phenotype_name == phenotype_name).ok_or_else(|| {
                    OutputError::InvalidInput(format!(
                        "Output immutable terminal is missing phenotype '{phenotype_name}'."
                    ))
                })?;
            if terminal_phenotype.output_directory_name != contract.output_directory_name
                || terminal_phenotype.run_manifest_sha256 != manifest_sha256
            {
                return Err(OutputError::InvalidInput(format!(
                    "Output immutable terminal manifest binding for phenotype '{phenotype_name}' does not match its bytes."
                )));
            }
        }
        Ok(Some(validated.gpu_genotype_format()))
    }

    #[cfg(test)]
    pub(crate) fn install_manifest_hint_pause_for_test(&mut self) -> OutputResult<OutputManifestHintTestControl> {
        let (reached_sender, reached_receiver) = std::sync::mpsc::channel();
        let (resume_sender, resume_receiver) = std::sync::mpsc::channel();
        self.core_mut()?.manifest_hint_pause = Some(Arc::new(OutputManifestHintTestPause {
            reached_sender,
            resume_receiver: std::sync::Mutex::new(resume_receiver),
        }));
        Ok(OutputManifestHintTestControl {
            reached_receiver,
            resume_sender,
            resumed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Acquire exclusive output ownership without publishing attempt authority.
    ///
    /// # Errors
    ///
    /// Returns an error when writer settings, the canonical chunk plan,
    /// explicit fencing, ownership acquisition, or staging creation fails.
    pub fn claim(
        mut self,
        planned_chunk_ranges: &[Range<usize>],
        collect_stage_timings: bool,
    ) -> OutputResult<OutputManager<Claimed>> {
        let mut core = self.take_core()?;
        validate_output_writer_settings(&core.run_plan.output, core.runs.len())?;
        core.canonical_chunk_plan = Some(CanonicalChunkPlan::try_new(planned_chunk_ranges)?);
        core.collect_stage_timings = collect_stage_timings;

        let claim_result = match core.run_plan.output.fenced_owner_claim_id.as_deref() {
            Some(fenced_claim_id) => core.lineage_paths.take_over_fenced_owner_claim(fenced_claim_id),
            None => core.lineage_paths.try_acquire_owner_claim(),
        };
        core.owner_claim = Some(claim_result?);
        let current_snapshot = match core.lineage_paths.inspect().and_then(|snapshot| {
            validate_claimed_policy(&core.run_plan, &core.lineage_paths, snapshot.as_ref(), &core.runs)?;
            Ok(snapshot)
        }) {
            Ok(snapshot) => snapshot,
            Err(error) => return Err(core.release_after_unmutated_error(error)),
        };
        if let Err(error) = core.reestablish_observed_durability(current_snapshot.as_ref()) {
            return Err(core.release_after_unmutated_error(error));
        }
        let durable_snapshot = match core.lineage_paths.inspect() {
            Ok(snapshot) if snapshot == current_snapshot => snapshot,
            Ok(_) => {
                return Err(core.release_after_unmutated_error(OutputError::InvalidInput(
                    "Output lineage changed while re-establishing observed durability.".to_string(),
                )));
            }
            Err(error) => return Err(core.release_after_unmutated_error(error)),
        };
        core.lineage_snapshot = durable_snapshot;
        let referenced_attempts =
            core.lineage_snapshot.as_ref().map_or_else(BTreeSet::new, lineage_attempt_identifiers);
        let current_claim_identifier = match core.owner_claim.as_ref() {
            Some(owner_claim) => owner_claim.claim_id().to_string(),
            None => {
                return Err(core.release_after_unmutated_error(OutputError::Runtime(
                    "Claimed output lost its owner claim before staging cleanup.".to_string(),
                )));
            }
        };
        if let Err(error) =
            core.lineage_paths.cleanup_obsolete_owner_staging(&current_claim_identifier, &referenced_attempts)
        {
            return Err(core.release_after_unmutated_error(error));
        }
        let staging_result = core.initialize_claim_staging_directory();
        if let Err(error) = staging_result {
            let cleanup_result = core.cleanup_claim_staging();
            let release_result = core.release_owner_claim();
            let primary_error = match cleanup_result {
                Ok(()) => error,
                Err(cleanup_error) => OutputError::OutputOperationAndOwnerClaimRelease {
                    primary: Box::new(error),
                    release: Box::new(cleanup_error),
                },
            };
            return Err(match release_result {
                Ok(()) => primary_error,
                Err(release_error) => OutputError::OutputOperationAndOwnerClaimRelease {
                    primary: Box::new(primary_error),
                    release: Box::new(release_error),
                },
            });
        }
        Ok(OutputManager { core: Some(core), state: PhantomData })
    }

    /// Validate all execution headers and initialize one active attempt.
    ///
    /// # Errors
    ///
    /// Returns an error when header coverage, lineage compatibility, durable
    /// recovery, attempt publication, or writer creation fails.
    pub fn initialize(
        self,
        current_header_inputs: Vec<CurrentRunManifestHeaderInput>,
        planned_chunk_ranges: &[Range<usize>],
        collect_stage_timings: bool,
    ) -> OutputResult<OutputManager<Active>> {
        let claimed_manager = self.claim(planned_chunk_ranges, collect_stage_timings)?;
        claimed_manager.activate(current_header_inputs)
    }
}

impl OutputManager<Claimed> {
    /// Return the ownership-private directory for run-scoped diagnostics.
    ///
    /// This path is allocated only after this process owns the output claim.
    ///
    /// # Errors
    ///
    /// Returns an error if claim staging was not initialized.
    pub fn diagnostics_directory(&self) -> OutputResult<&Path> {
        self.core()?
            .claim_staging_directory
            .as_deref()
            .ok_or_else(|| OutputError::Runtime("Claimed output has no diagnostics staging directory.".to_string()))
    }

    /// Publish immutable attempt authority and start the output writers.
    ///
    /// # Errors
    ///
    /// Returns an error when header coverage, lineage compatibility, durable
    /// recovery, attempt publication, or writer creation fails.
    pub fn activate(
        self,
        current_header_inputs: Vec<CurrentRunManifestHeaderInput>,
    ) -> OutputResult<OutputManager<Active>> {
        self.activate_with_cleanup_policy(current_header_inputs, CompletedNoopCleanupPolicy::Immediate)
            .map_err(resolve_activation_error)
    }

    /// Publish attempt authority while deferring completed-noop cleanup until
    /// the caller closes claim-scoped diagnostics.
    ///
    /// # Errors
    ///
    /// Returns an error when header coverage, lineage compatibility, durable
    /// recovery, attempt publication, or writer creation fails. The caller must
    /// retain any unpublished rollback and consume it only after claim-scoped
    /// diagnostics close. Dropping it deliberately leaves ownership active so
    /// recovery fails closed until an exact external fence is supplied.
    pub fn activate_with_deferred_completed_noop_cleanup(
        self,
        current_header_inputs: Vec<CurrentRunManifestHeaderInput>,
    ) -> Result<OutputManager<Active>, OutputActivationError> {
        self.activate_with_cleanup_policy(current_header_inputs, CompletedNoopCleanupPolicy::Deferred)
    }

    fn activate_with_cleanup_policy(
        mut self,
        current_header_inputs: Vec<CurrentRunManifestHeaderInput>,
        completed_noop_cleanup_policy: CompletedNoopCleanupPolicy,
    ) -> Result<OutputManager<Active>, OutputActivationError> {
        let mut core = self.take_core()?;
        core.lifecycle_state.completed_noop_cleanup_policy = completed_noop_cleanup_policy;
        let Some(canonical_chunk_plan) = core.canonical_chunk_plan.clone() else {
            return Err(unpublished_activation_error(
                core,
                OutputError::Runtime("Claimed output has no canonical chunk plan.".to_string()),
            ));
        };
        let collect_stage_timings = core.collect_stage_timings;
        let phenotype_contracts = match (|| {
            let headers = build_headers(&core, current_header_inputs)?;
            bind_headers(&mut core.runs, headers)?;
            let phenotype_contracts = phenotype_contracts(&core.runs)?;
            if let Some(snapshot) = core.lineage_snapshot.as_ref() {
                preflight_existing_lineage(&core, snapshot, &canonical_chunk_plan, &phenotype_contracts)?;
            }
            Ok(phenotype_contracts)
        })() {
            Ok(phenotype_contracts) => phenotype_contracts,
            Err(error) => return Err(unpublished_activation_error(core, error)),
        };
        #[cfg(test)]
        match core
            .owner_claim
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output activation has no owner claim.".to_string()))
            .and_then(inject_owner_claim_release_conflict_at_test_point)
        {
            Ok(None) => {}
            Ok(Some(error)) | Err(error) => return Err(unpublished_activation_error(core, error)),
        }
        #[cfg(test)]
        if let Err(error) = fail_initialization_at_test_point("after_owner_claim") {
            return Err(unpublished_activation_error(core, error));
        }
        let initialization_result = match core.lineage_snapshot.clone() {
            None => {
                initialize_genesis_attempt(&mut core, &canonical_chunk_plan, phenotype_contracts, collect_stage_timings)
            }
            Some(snapshot) => initialize_existing_lineage(
                &mut core,
                snapshot,
                &canonical_chunk_plan,
                &phenotype_contracts,
                collect_stage_timings,
            ),
        };
        if let Err(error) = initialization_result {
            if core.lifecycle_state.attempt_authority_publication.rollback_is_safe() {
                return Err(unpublished_activation_error(core, error));
            }
            return match core.resolve_initialization_error(&error) {
                Ok(()) => Err(OutputActivationError::Published(error)),
                Err(recovery_error) => {
                    Err(OutputActivationError::Published(OutputError::OutputOperationAndOwnerClaimRelease {
                        primary: Box::new(error),
                        release: Box::new(recovery_error),
                    }))
                }
            };
        }
        Ok(OutputManager { core: Some(core), state: PhantomData })
    }

    /// Remove an unpublished attempt reservation and release ownership.
    ///
    /// Run-scoped diagnostics must be closed before calling this method.
    ///
    /// # Errors
    ///
    /// Returns an error when staging cleanup or owner release fails.
    pub fn abort_before_activation(mut self) -> OutputResult<()> {
        self.abort_before_activation_in_place()
    }

    fn abort_before_activation_in_place(&mut self) -> OutputResult<()> {
        let core = self.core_mut()?;
        if let Err(error) = core.cleanup_claim_staging() {
            return Err(core.retained_owner_claim_error(&error));
        }
        core.release_owner_claim()
    }
}

fn unpublished_activation_error(core: OutputManagerCore, source: OutputError) -> OutputActivationError {
    let manager = OutputManager { core: Some(core), state: PhantomData };
    OutputActivationError::Unpublished { source, rollback: Box::new(OutputClaimRollback(Some(manager))) }
}

fn resolve_activation_error(error: OutputActivationError) -> OutputError {
    let OutputActivationFailureParts { source, rollback } = error.into_parts();
    let Some(mut rollback) = rollback else {
        return source;
    };
    match rollback.abort_before_activation() {
        Ok(()) => source,
        Err(rollback_error) => OutputError::OutputOperationAndOwnerClaimRelease {
            primary: Box::new(source),
            release: Box::new(rollback_error),
        },
    }
}

impl OutputPostSessionCleanup {
    #[cfg(test)]
    pub(crate) fn install_cleanup_pause_for_test(&mut self) -> OutputPostSessionCleanupTestControl {
        let (reached_sender, reached_receiver) = std::sync::mpsc::channel();
        let (resume_sender, resume_receiver) = std::sync::mpsc::channel();
        let pause = Arc::new(OutputPostSessionCleanupTestPause {
            reached_sender,
            resume_receiver: std::sync::Mutex::new(resume_receiver),
        });
        self.cleanup_pause = Some(Arc::clone(&pause));
        OutputPostSessionCleanupTestControl {
            reached_receiver,
            resume_sender,
            resumed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Remove completed-noop diagnostics and release its exact owner claim.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched staging metadata or failed durable
    /// cleanup. Retry this same value after a durability error.
    pub fn cleanup(&mut self) -> OutputResult<()> {
        if self.completed {
            return Ok(());
        }
        if self
            .lineage_paths
            .owner_staging_attempt(&self.claim_id)?
            .is_some_and(|staged_attempt_id| staged_attempt_id != self.attempt_id)
        {
            return Err(OutputError::ConcurrentLineageUpdate {
                record_path: self
                    .lineage_paths
                    .control_directory
                    .join("owner-staging")
                    .join(format!("{}.json", self.claim_id)),
            });
        }
        #[cfg(test)]
        if let Some(cleanup_pause) = self.cleanup_pause.as_ref() {
            cleanup_pause.reached_sender.send(()).map_err(|error| {
                OutputError::Runtime(format!("Failed to report the output cleanup test pause: {error}"))
            })?;
            cleanup_pause
                .resume_receiver
                .lock()
                .map_err(|error| OutputError::Runtime(format!("Output cleanup test pause lock is poisoned: {error}")))?
                .recv_timeout(OUTPUT_POST_SESSION_CLEANUP_TEST_TIMEOUT)
                .map_err(|error| OutputError::Runtime(format!("Output cleanup test pause timed out: {error}")))?;
        }
        let referenced_attempts =
            self.lineage_paths.inspect()?.as_ref().map_or_else(BTreeSet::new, lineage_attempt_identifiers);
        if referenced_attempts.contains(&self.attempt_id) {
            return Err(OutputError::InvalidInput(
                "Completed no-op diagnostics staging unexpectedly became authoritative.".to_string(),
            ));
        }
        let removal_path = self.lineage_paths.attempt_directory(&self.attempt_id);
        match std::fs::remove_dir_all(&removal_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(OutputError::Runtime(format!(
                    "Failed to remove completed no-op diagnostics staging '{}': {error}",
                    removal_path.display()
                )));
            }
        }
        sync_nearest_existing_directory(&self.lineage_paths.attempts_directory)?;
        #[cfg(test)]
        fail_terminal_cleanup_at_test_point("after_post_session_staging_removal")?;
        self.lineage_paths.retire_owner_staging_intent(&self.claim_id, &self.attempt_id)?;
        #[cfg(test)]
        fail_terminal_cleanup_at_test_point("after_post_session_intent_retirement")?;
        #[cfg(test)]
        fail_terminal_cleanup_at_test_point("before_post_session_owner_release")?;
        self.owner_release.release_if_current()?;
        #[cfg(test)]
        fail_terminal_cleanup_at_test_point("after_post_session_owner_release")?;
        self.completed = true;
        Ok(())
    }
}

#[cfg(test)]
impl OutputPostSessionCleanupTestControl {
    pub(crate) fn wait_until_reached(&self) -> Result<(), std::sync::mpsc::RecvTimeoutError> {
        self.reached_receiver.recv_timeout(OUTPUT_POST_SESSION_CLEANUP_TEST_TIMEOUT)
    }

    pub(crate) fn resume(&self) {
        self.resume_cleanup();
    }

    fn resume_cleanup(&self) {
        if !self.resumed.swap(true, std::sync::atomic::Ordering::SeqCst) {
            let _ = self.resume_sender.send(());
        }
    }
}

#[cfg(test)]
impl OutputManifestHintTestPause {
    fn wait(&self) -> OutputResult<()> {
        self.reached_sender.send(()).map_err(|error| {
            OutputError::Runtime(format!("Failed to report the output manifest hint test pause: {error}"))
        })?;
        self.resume_receiver
            .lock()
            .map_err(|error| {
                OutputError::Runtime(format!("Output manifest hint test pause lock is poisoned: {error}"))
            })?
            .recv_timeout(OUTPUT_MANIFEST_HINT_TEST_TIMEOUT)
            .map_err(|error| OutputError::Runtime(format!("Output manifest hint test pause timed out: {error}")))
    }
}

#[cfg(test)]
impl OutputManifestHintTestControl {
    pub(crate) fn wait_until_reached(&self) -> Result<(), std::sync::mpsc::RecvTimeoutError> {
        self.reached_receiver.recv_timeout(OUTPUT_MANIFEST_HINT_TEST_TIMEOUT)
    }

    pub(crate) fn resume(&self) {
        self.resume_hint();
    }

    fn resume_hint(&self) {
        if !self.resumed.swap(true, std::sync::atomic::Ordering::SeqCst) {
            let _ = self.resume_sender.send(());
        }
    }
}

#[cfg(test)]
impl Drop for OutputManifestHintTestControl {
    fn drop(&mut self) {
        self.resume_hint();
    }
}

#[cfg(test)]
impl Drop for OutputPostSessionCleanupTestControl {
    fn drop(&mut self) {
        self.resume_cleanup();
    }
}

impl OutputManager<Active> {
    /// Build an opaque group capability in phenotype order.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or duplicate phenotype names.
    pub fn delivery_token_for_phenotypes(&self, phenotype_names: &[String]) -> OutputResult<OutputDeliveryToken> {
        let core = self.core()?;
        let mut selected_indices = BTreeSet::new();
        let mut writer_sessions = Vec::with_capacity(phenotype_names.len());
        let mut committed_chunk_identifier_sets = Vec::with_capacity(phenotype_names.len());
        for phenotype_name in phenotype_names {
            let output_index =
                core.run_indices_by_phenotype.get(phenotype_name).copied().ok_or_else(|| {
                    OutputError::InvalidInput(format!("Unknown planned phenotype '{phenotype_name}'."))
                })?;
            if !selected_indices.insert(output_index) {
                return Err(OutputError::InvalidInput(format!(
                    "Phenotype '{phenotype_name}' was selected more than once for output delivery."
                )));
            }
            let run = core
                .runs
                .get(output_index)
                .ok_or_else(|| OutputError::Runtime(format!("Output index {output_index} is inconsistent.")))?;
            writer_sessions.push(run.writer_session.as_ref().map(Arc::clone));
            committed_chunk_identifier_sets.push(Arc::clone(&run.committed_chunk_identifiers));
        }
        Ok(OutputDeliveryToken { writer_sessions, committed_chunk_identifier_sets })
    }

    /// Drain all writers and prove exact coverage.
    ///
    /// # Errors
    ///
    /// Returns an error when a writer, timing snapshot, or exact-coverage check
    /// fails. A best-effort failed terminal is published before returning.
    pub fn close_completed(mut self) -> Result<OutputManager<Covered>, OutputTerminalError> {
        let mut core = self.take_core()?;
        let completed_noop = core.active_attempt()?.completed_noop;
        let close_result = if completed_noop { core.reverify_completed_noop() } else { core.close_writers_completed() };
        #[cfg(test)]
        let close_result = close_result.and(fail_lifecycle_at_test_point(if completed_noop {
            "close_completed_noop"
        } else {
            "close_completed"
        }));
        if let Err(error) = close_result {
            if completed_noop {
                return Err(core.completed_noop_terminal_error(
                    error,
                    "Completed output verification failure recovery also failed",
                ));
            }
            let reason = format!("output completion failed: {error}");
            let recovery_result =
                core.publish_terminal_and_release(&AttemptManifestStatus::Failed, None, Some(&reason));
            return return_primary_after_recovery(
                error,
                recovery_result,
                "Output completion failure recovery also failed",
            )
            .map_err(OutputTerminalError::from);
        }
        Ok(OutputManager { core: Some(core), state: PhantomData })
    }

    /// Flush accepted output and publish an interrupted terminal.
    ///
    /// # Errors
    ///
    /// Returns an error when draining, timing, manifest, or terminal publication
    /// fails.
    pub fn finish_interrupted(mut self, signal_name: &str) -> Result<(), OutputTerminalError> {
        let mut core = self.take_core()?;
        let completed_noop = core.active_attempt()?.completed_noop;
        if signal_name.trim().is_empty() {
            let error = OutputError::InvalidInput("Output interruption signal name must not be empty.".to_string());
            if completed_noop {
                return Err(
                    core.completed_noop_terminal_error(error, "Completed no-op interruption rejection cleanup failed")
                );
            }
            return return_primary_after_recovery(
                error,
                core.resolve_rejected_terminal_request("empty interruption signal"),
                "Rejected output interruption recovery failed",
            )
            .map_err(OutputTerminalError::from);
        }
        if completed_noop {
            let error =
                OutputError::InvalidInput("A verified completed output lineage cannot be interrupted.".to_string());
            return Err(
                core.completed_noop_terminal_error(error, "Completed no-op interruption rejection cleanup failed")
            );
        }
        let close_result = core.close_writers_interrupted(signal_name);
        #[cfg(test)]
        let close_result = close_result.and(fail_lifecycle_at_test_point("finish_interrupted"));
        if let Err(error) = close_result {
            let reason = format!("interrupted output flush failed: {error}");
            let recovery_result =
                core.publish_terminal_and_release(&AttemptManifestStatus::Failed, None, Some(&reason));
            return return_primary_after_recovery(
                error,
                recovery_result,
                "Interrupted output flush failure recovery also failed",
            )
            .map_err(OutputTerminalError::from);
        }
        core.publish_terminal_and_release(&AttemptManifestStatus::Interrupted, Some(signal_name), None)
            .map_err(OutputTerminalError::from)
    }

    /// Discard unsubmitted tails and publish a failed terminal.
    ///
    /// # Errors
    ///
    /// Returns an error when writer drain, timing, manifest, or terminal
    /// publication fails.
    pub fn abort(mut self, failure_reason: &str) -> Result<(), OutputTerminalError> {
        let mut core = self.take_core()?;
        let completed_noop = core.active_attempt()?.completed_noop;
        if failure_reason.trim().is_empty() {
            let error = OutputError::InvalidInput("Output failure reason must not be empty.".to_string());
            if completed_noop {
                return Err(core.completed_noop_terminal_error(error, "Completed no-op abort rejection cleanup failed"));
            }
            return return_primary_after_recovery(
                error,
                core.resolve_rejected_terminal_request("empty failure reason"),
                "Rejected output abort recovery failed",
            )
            .map_err(OutputTerminalError::from);
        }
        if completed_noop {
            let error = OutputError::InvalidInput("A verified completed output lineage cannot be aborted.".to_string());
            return Err(core.completed_noop_terminal_error(error, "Completed no-op abort rejection cleanup failed"));
        }
        let abort_result = core.abort_writers();
        let terminal_reason = abort_result.as_ref().err().map_or_else(
            || failure_reason.to_string(),
            |abort_error| format!("{failure_reason}; writer cleanup also reported: {abort_error}"),
        );
        let recovery_result =
            core.publish_terminal_and_release(&AttemptManifestStatus::Failed, None, Some(&terminal_reason));
        combine_terminal_cleanup_result(abort_result, recovery_result, "Output abort recovery also failed")
            .map_err(OutputTerminalError::from)
    }
}

impl OutputManager<Covered> {
    /// Publish completed manifests and the immutable completed terminal.
    ///
    /// Completed no-op attempts are fully reverified and perform no writes.
    ///
    /// # Errors
    ///
    /// Returns an error when final verification, manifest publication, or
    /// terminal publication fails.
    pub fn finish(mut self) -> Result<OutputCompletion, OutputTerminalError> {
        let mut core = self.take_core()?;
        let completed_noop = core.active_attempt()?.completed_noop;
        if completed_noop {
            let completed_outputs = match core.reverify_completed_noop().and_then(|()| core.completed_outputs()) {
                Ok(completed_outputs) => completed_outputs,
                Err(error) => {
                    return Err(core.completed_noop_terminal_error(error, "Completed output verification failed"));
                }
            };
            let post_session_cleanup = core.finalize_completed_noop_claim()?;
            return Ok(OutputCompletion { completed_outputs, post_session_cleanup });
        }

        let completed_outputs_result = core.completed_outputs();
        #[cfg(test)]
        let completed_outputs_result = completed_outputs_result.and_then(|outputs| {
            fail_completion_at_test_point("before_completed_terminal_publication").map(|()| outputs)
        });
        let completed_outputs = match completed_outputs_result {
            Ok(outputs) => outputs,
            Err(error) => {
                let failure_reason = format!("completed output result construction failed: {error}");
                let recovery_result =
                    core.publish_terminal_and_release(&AttemptManifestStatus::Failed, None, Some(&failure_reason));
                return return_primary_after_recovery(
                    error,
                    recovery_result,
                    "Completed output failure recovery also failed",
                )
                .map_err(OutputTerminalError::from);
            }
        };

        if let Err(error) = core.publish_terminal(&AttemptManifestStatus::Completed, None, None) {
            return Err(OutputTerminalError::from(core.retained_owner_claim_error(&error)));
        }
        core.lifecycle_state.terminal = true;
        let release_result = core.release_owner_claim();
        #[cfg(test)]
        combine_terminal_cleanup_result(
            fail_completion_at_test_point("after_completed_terminal_finalization"),
            release_result,
            "Completed output terminal-boundary operation failed",
        )
        .map_err(OutputTerminalError::from)?;
        #[cfg(not(test))]
        release_result.map_err(OutputTerminalError::from)?;
        Ok(OutputCompletion { completed_outputs, post_session_cleanup: None })
    }
}

impl OutputDeliveryToken {
    #[must_use]
    pub fn trait_count(&self) -> usize {
        self.writer_sessions.len()
    }

    #[must_use]
    pub fn committed_chunk_identifier_sets(&self) -> &[Arc<BTreeSet<usize>>] {
        &self.committed_chunk_identifier_sets
    }

    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.writer_sessions.iter().all(Option::is_none)
    }

    pub(crate) fn writer_session(&self, trait_index: usize) -> OutputResult<&OutputWriterSession> {
        self.writer_sessions
            .get(trait_index)
            .ok_or_else(|| OutputError::InvalidInput("Active trait index is out of bounds.".to_string()))?
            .as_deref()
            .ok_or_else(|| {
                OutputError::InvalidInput(
                    "Completed output delivery token is strictly read-only and cannot accept chunks.".to_string(),
                )
            })
    }
}

impl<State> OutputManager<State> {
    fn core(&self) -> OutputResult<&OutputManagerCore> {
        self.core.as_ref().ok_or_else(|| OutputError::Runtime("Output manager core was already consumed.".to_string()))
    }

    fn core_mut(&mut self) -> OutputResult<&mut OutputManagerCore> {
        self.core.as_mut().ok_or_else(|| OutputError::Runtime("Output manager core was already consumed.".to_string()))
    }

    fn take_core(&mut self) -> OutputResult<OutputManagerCore> {
        self.core.take().ok_or_else(|| OutputError::Runtime("Output manager core was already consumed.".to_string()))
    }
}

impl<State> Drop for OutputManager<State> {
    fn drop(&mut self) {
        let Some(core) = self.core.as_mut() else {
            return;
        };
        if !core.lifecycle_state.terminal {
            for run in &core.runs {
                if let Some(writer_session) = run.writer_session.as_ref() {
                    let _ = writer_session.abort();
                }
            }
            let _ = core.shutdown_writer_resources();
        }
    }
}

impl OutputManagerCore {
    fn classify_attempt_authority_publication<ValueType>(
        &mut self,
        record_path: PathBuf,
        publication_result: OutputResult<ValueType>,
    ) -> OutputResult<ValueType> {
        match publication_result {
            Ok(publication) => {
                self.lifecycle_state.attempt_authority_publication =
                    AttemptAuthorityPublication::VisibleOrUnknown { record_path };
                Ok(publication)
            }
            Err(error) => {
                self.lifecycle_state.attempt_authority_publication = match std::fs::symlink_metadata(&record_path) {
                    Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
                        AttemptAuthorityPublication::DefinitelyUnpublished { record_path }
                    }
                    Ok(_) | Err(_) => AttemptAuthorityPublication::VisibleOrUnknown { record_path },
                };
                Err(error)
            }
        }
    }

    fn reestablish_observed_durability(&self, snapshot: Option<&LineageSnapshot>) -> OutputResult<()> {
        if let Some(snapshot) = snapshot {
            for attempt_id in lineage_attempt_identifiers(snapshot) {
                for run in &self.runs {
                    let paths = AttemptRunPaths::new(
                        &self.lineage_paths.attempts_directory,
                        &attempt_id,
                        &run.output_directory_name,
                    )?;
                    paths.reestablish_directory_durability()?;
                }
                let attempt_directory = self.lineage_paths.attempt_directory(&attempt_id);
                if attempt_directory.is_dir() {
                    crate::persistence::io::sync_directory(&attempt_directory)?;
                }
            }
        }
        self.lineage_paths.reestablish_observed_directory_durability()
    }

    fn reestablish_current_durability_and_reinspect(&mut self) -> OutputResult<LineageSnapshot> {
        let observed_snapshot = self
            .lineage_paths
            .inspect()?
            .ok_or_else(|| OutputError::InvalidInput("Owned output lineage disappeared.".to_string()))?;
        self.reestablish_observed_durability(Some(&observed_snapshot))?;
        let durable_snapshot = self.lineage_paths.inspect()?.ok_or_else(|| {
            OutputError::InvalidInput("Owned output lineage disappeared after synchronization.".to_string())
        })?;
        if durable_snapshot != observed_snapshot {
            return Err(OutputError::InvalidInput(
                "Output lineage changed while re-establishing durable authority.".to_string(),
            ));
        }
        self.lineage_snapshot = Some(durable_snapshot.clone());
        Ok(durable_snapshot)
    }

    fn cleanup_claim_staging(&mut self) -> OutputResult<()> {
        self.lifecycle_state.attempt_authority_publication.ensure_rollback_target_absent()?;
        let claimed_attempt_id = self.claimed_attempt_id()?;
        let owner_claim = self
            .owner_claim
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output staging cleanup requires an owner claim.".to_string()))?;
        let claim_identifier = owner_claim.claim_id().to_string();
        let removal_path = self.lineage_paths.attempt_directory(&claimed_attempt_id);
        match std::fs::remove_dir_all(&removal_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(OutputError::Runtime(format!(
                    "Failed to remove unpublished output staging '{}': {error}",
                    removal_path.display()
                )));
            }
        }
        sync_nearest_existing_directory(&self.lineage_paths.attempts_directory)?;
        #[cfg(test)]
        fail_initialization_cleanup_at_test_point("after_claim_staging_removal")?;
        self.lineage_paths.retire_owner_staging_intent(&claim_identifier, &claimed_attempt_id)?;
        self.claim_staging_directory = None;
        self.completed_noop_cleanup = None;
        Ok(())
    }

    fn initialize_claim_staging_directory(&mut self) -> OutputResult<()> {
        let completed_output_claim = self
            .lineage_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.leaf_terminal.as_ref().or(snapshot.pending_terminal.as_ref()))
            .is_some_and(|terminal| terminal.status == AttemptTerminalStatus::Completed);
        let claimed_attempt_id = AttemptIdentifier::generate();
        let owner_claim = self
            .owner_claim
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output claim staging requires an owner claim.".to_string()))?;
        let claim_identifier = owner_claim.claim_id().to_string();
        let claim_staging_directory =
            self.lineage_paths.attempt_directory(&claimed_attempt_id).join("diagnostics").join(&claim_identifier);
        let completed_noop_cleanup = completed_output_claim.then(|| OutputPostSessionCleanup {
            lineage_paths: self.lineage_paths.clone(),
            owner_release: owner_claim.conditional_release(),
            claim_id: claim_identifier.clone(),
            attempt_id: claimed_attempt_id.clone(),
            completed: false,
            #[cfg(test)]
            cleanup_pause: None,
        });
        self.claimed_attempt_id = Some(claimed_attempt_id.clone());
        self.completed_noop_cleanup = completed_noop_cleanup;
        self.claim_staging_directory = Some(claim_staging_directory.clone());
        self.lineage_paths.publish_owner_staging_intent(&claim_identifier, &claimed_attempt_id)?;
        #[cfg(test)]
        fail_initialization_at_test_point("after_owner_staging_intent")?;
        create_directories_durable(&claim_staging_directory)?;
        #[cfg(test)]
        fail_initialization_at_test_point("after_claim_diagnostics_creation")?;
        Ok(())
    }

    fn claimed_attempt_id(&self) -> OutputResult<AttemptIdentifier> {
        self.claimed_attempt_id
            .clone()
            .ok_or_else(|| OutputError::Runtime("Claimed output has no reserved attempt identifier.".to_string()))
    }

    fn retire_claim_staging_intent(&self) -> OutputResult<()> {
        let owner_claim = self
            .owner_claim
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output staging retirement requires an owner claim.".to_string()))?;
        let attempt_id = self
            .claimed_attempt_id
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output staging retirement requires a claimed attempt.".to_string()))?;
        self.lineage_paths.retire_owner_staging_intent(owner_claim.claim_id(), attempt_id)
    }

    fn release_after_unmutated_error(&mut self, error: OutputError) -> OutputError {
        match self.release_owner_claim() {
            Ok(()) => error,
            Err(release_error) => OutputError::OutputOperationAndOwnerClaimRelease {
                primary: Box::new(error),
                release: Box::new(release_error),
            },
        }
    }

    fn run(&self, phenotype_name: &str) -> OutputResult<&ManagedOutputRun> {
        let output_index = self
            .run_indices_by_phenotype
            .get(phenotype_name)
            .ok_or_else(|| OutputError::InvalidInput(format!("Unknown planned phenotype '{phenotype_name}'.")))?;
        self.runs.get(*output_index).ok_or_else(|| {
            OutputError::Runtime(format!("Output index for phenotype '{phenotype_name}' is inconsistent."))
        })
    }

    fn active_attempt(&self) -> OutputResult<&ActiveAttempt> {
        self.active_attempt
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output manager has no active attempt.".to_string()))
    }

    fn install_prepared_attempt(
        &mut self,
        prepared_attempt: PreparedAttempt,
        canonical_chunk_plan: CanonicalChunkPlan,
        completed_noop: bool,
        collect_stage_timings: bool,
    ) -> OutputResult<()> {
        if prepared_attempt.runs.len() != self.runs.len() {
            return Err(OutputError::Runtime("Prepared output attempt run count is inconsistent.".to_string()));
        }
        for (run, prepared_run) in self.runs.iter_mut().zip(prepared_attempt.runs) {
            run.binding = Some(prepared_run.binding);
            run.paths = Some(prepared_run.paths);
            run.receipts = prepared_run.receipts;
            run.committed_chunk_identifiers = Arc::new(prepared_run.committed_chunk_identifiers);
        }
        self.active_attempt = Some(ActiveAttempt {
            run_set_id: prepared_attempt.run_set_id,
            attempt_id: prepared_attempt.attempt_id,
            canonical_chunk_plan,
            completed_noop,
            collect_stage_timings,
        });
        Ok(())
    }

    fn install_claimed_attempt_shell(
        &mut self,
        run_set_id: String,
        attempt_id: AttemptIdentifier,
        canonical_chunk_plan: CanonicalChunkPlan,
        collect_stage_timings: bool,
    ) -> OutputResult<()> {
        for run in &mut self.runs {
            let execution_plan_sha256 = execution_plan_hash(run)?;
            run.binding = Some(AttemptManifestBinding {
                run_set_id: run_set_id.clone(),
                attempt_id: attempt_id.clone(),
                phenotype_name: run.phenotype_name.clone(),
                output_directory_name: run.output_directory_name.clone(),
                execution_plan_sha256,
                chunk_plan_sha256: canonical_chunk_plan.sha256().to_string(),
            });
            run.paths = Some(AttemptRunPaths::new(
                &self.lineage_paths.attempts_directory,
                &attempt_id,
                &run.output_directory_name,
            )?);
            run.receipts.clear();
            run.committed_chunk_identifiers = Arc::new(BTreeSet::new());
        }
        self.active_attempt = Some(ActiveAttempt {
            run_set_id,
            attempt_id,
            canonical_chunk_plan,
            completed_noop: false,
            collect_stage_timings,
        });
        Ok(())
    }

    fn resolve_initialization_error(&mut self, initialization_error: &OutputError) -> OutputResult<()> {
        if self.lifecycle_state.attempt_authority_publication.rollback_is_safe() {
            return self.release_owner_claim();
        }
        let abort_result = if self.runs.iter().any(|run| run.writer_session.is_some()) {
            self.abort_writers()
        } else {
            self.shutdown_writer_resources()
        };
        #[cfg(test)]
        let abort_result = abort_result.and(fail_initialization_cleanup_at_test_point("after_writer_abort"));
        let abort_diagnostic = abort_result.as_ref().err().map(ToString::to_string);
        let support_result = self.materialize_initialization_failure_support();
        let support_succeeded = support_result.is_ok();
        let failure_reason = abort_diagnostic.map_or_else(
            || format!("output initialization failed: {initialization_error}"),
            |abort_error| {
                format!(
                    "output initialization failed: {initialization_error}; writer cleanup also reported: {abort_error}"
                )
            },
        );
        let terminal_result = if support_succeeded {
            self.publish_terminal_and_release(&AttemptManifestStatus::Failed, None, Some(&failure_reason))
        } else {
            Ok(())
        };
        support_result.and(terminal_result)
    }

    fn resolve_rejected_terminal_request(&mut self, rejection_reason: &str) -> OutputResult<()> {
        let abort_result = self.abort_writers();
        let failure_reason = abort_result.as_ref().err().map_or_else(
            || format!("output lifecycle request rejected: {rejection_reason}"),
            |abort_error| {
                format!(
                    "output lifecycle request rejected: {rejection_reason}; writer cleanup also reported: \
                     {abort_error}"
                )
            },
        );
        self.publish_terminal_and_release(&AttemptManifestStatus::Failed, None, Some(&failure_reason))
    }

    fn materialize_initialization_failure_support(&self) -> OutputResult<()> {
        for run in &self.runs {
            let paths = run.paths.as_ref().ok_or_else(|| {
                OutputError::Runtime("Claimed output attempt has no paths during failure recovery.".to_string())
            })?;
            paths.initialize_directories()?;
            write_effective_config(paths, &self.effective_config_toml)?;
        }
        Ok(())
    }

    fn finalize_completed_noop_claim(&mut self) -> Result<Option<OutputPostSessionCleanup>, OutputTerminalError> {
        self.lifecycle_state.terminal = true;
        let cleanup = self.completed_noop_cleanup.take().ok_or_else(|| {
            OutputTerminalError::from(OutputError::Runtime(
                "Completed no-op claim has no post-session cleanup authority.".to_string(),
            ))
        })?;
        if self.lifecycle_state.completed_noop_cleanup_policy == CompletedNoopCleanupPolicy::Deferred {
            Ok(Some(cleanup))
        } else {
            let mut cleanup = cleanup;
            match cleanup.cleanup() {
                Ok(()) => Ok(None),
                Err(error) => Err(OutputTerminalError::with_cleanup(error, Some(cleanup))),
            }
        }
    }

    fn completed_noop_terminal_error(&mut self, source: OutputError, context: &str) -> OutputTerminalError {
        match self.finalize_completed_noop_claim() {
            Ok(post_session_cleanup) => OutputTerminalError::with_cleanup(source, post_session_cleanup),
            Err(recovery_error) => {
                let OutputTerminalFailureParts { source: recovery_source, post_session_cleanup } =
                    recovery_error.into_parts();
                OutputTerminalError::with_cleanup(
                    OutputError::OutputOperationAndOwnerClaimRelease {
                        primary: Box::new(source),
                        release: Box::new(OutputError::Runtime(format!("{context}: {recovery_source}"))),
                    },
                    post_session_cleanup,
                )
            }
        }
    }

    fn start_writers(&mut self) -> OutputResult<()> {
        let canonical_chunk_plan = self.active_attempt()?.canonical_chunk_plan.clone();
        let collect_stage_timings = self.active_attempt()?.collect_stage_timings;
        let run_configs = self
            .runs
            .iter()
            .map(|run| {
                let paths = run
                    .paths
                    .as_ref()
                    .ok_or_else(|| OutputError::Runtime("Output run paths are not initialized.".to_string()))?;
                let binding = run
                    .binding
                    .as_ref()
                    .ok_or_else(|| OutputError::Runtime("Output run binding is not initialized.".to_string()))?;
                Ok(OutputWriterRunConfig {
                    run_directory: paths.run_directory.clone(),
                    parts_directory: paths.parts_directory.clone(),
                    commits_directory: paths.commits_directory.clone(),
                    binding: binding.part_binding(),
                    canonical_chunk_plan: canonical_chunk_plan.clone(),
                    initial_receipts: run.receipts.clone(),
                })
            })
            .collect::<OutputResult<Vec<_>>>()?;
        let CreatedOutputWriterSessions { sessions, resource_owner } =
            create_output_writer_sessions(run_configs, &self.run_plan.output, collect_stage_timings)?;
        for (run, session) in self.runs.iter_mut().zip(sessions) {
            run.writer_session = Some(Arc::new(session));
        }
        self.writer_resource_owner = resource_owner;
        Ok(())
    }

    fn close_writers_completed(&mut self) -> OutputResult<()> {
        let sessions = self.writer_session_references()?;
        let thread_count = finish_thread_count(&self.run_plan.output, sessions.len())?;
        let finish_result = finish_output_writer_sessions(&sessions, thread_count);
        let shutdown_result = self.shutdown_writer_resources();
        finish_result?;
        shutdown_result?;
        self.refresh_receipts_from_sessions()?;
        let canonical_chunk_plan = &self.active_attempt()?.canonical_chunk_plan;
        for run in &self.runs {
            canonical_chunk_plan
                .validate_exact_coverage(run.receipts.iter().flat_map(|receipt| receipt.footer.chunks.iter()))?;
        }
        Ok(())
    }

    fn close_writers_interrupted(&mut self, signal_name: &str) -> OutputResult<()> {
        let sessions = self.writer_session_references()?;
        let thread_count = finish_thread_count(&self.run_plan.output, sessions.len())?;
        let finish_result = finish_interrupted_output_writer_sessions(&sessions, thread_count, signal_name);
        let shutdown_result = self.shutdown_writer_resources();
        finish_result?;
        shutdown_result?;
        self.refresh_receipts_from_sessions()
    }

    fn abort_writers(&mut self) -> OutputResult<()> {
        let mut first_error = None;
        for run in &self.runs {
            if let Some(session) = run.writer_session.as_ref()
                && let Err(error) = session.abort()
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        let shutdown_result = self.shutdown_writer_resources();
        let refresh_result = self.refresh_receipts_from_sessions();
        first_error.map_or(Ok(()), Err).and(shutdown_result).and(refresh_result)
    }

    fn writer_session_references(&self) -> OutputResult<Vec<&OutputWriterSession>> {
        self.runs
            .iter()
            .map(|run| {
                run.writer_session
                    .as_deref()
                    .ok_or_else(|| OutputError::Runtime("Writable output run has no writer session.".to_string()))
            })
            .collect()
    }

    fn refresh_receipts_from_sessions(&mut self) -> OutputResult<()> {
        for run in &mut self.runs {
            if let Some(session) = run.writer_session.as_ref() {
                run.receipts = session.receipt_snapshot()?;
                run.committed_chunk_identifiers = Arc::new(receipt_chunk_identifiers(&run.receipts)?);
            }
        }
        Ok(())
    }

    fn shutdown_writer_resources(&mut self) -> OutputResult<()> {
        self.writer_resource_owner.take().map_or(Ok(()), |mut owner| owner.shutdown_and_join())
    }

    fn release_owner_claim(&mut self) -> OutputResult<()> {
        #[cfg(test)]
        if let Some(owner_claim) = self.owner_claim.as_ref() {
            inject_owner_claim_cleanup_conflict_at_test_point(owner_claim)?;
        }
        let Some(owner_claim) = self.owner_claim.as_mut() else {
            return Err(OutputError::Runtime("Output manager has no owner claim to release.".to_string()));
        };
        let claim_path = owner_claim.claim_path().to_path_buf();
        match owner_claim.release() {
            Ok(()) => {
                self.owner_claim = None;
                Ok(())
            }
            Err(first_error) if owner_claim.release_transition_is_visible() => match owner_claim.release() {
                Ok(()) => {
                    self.owner_claim = None;
                    Ok(())
                }
                Err(retry_error) => Err(OutputError::PublishedOutputOwnerClaimReleaseDurability {
                    claim_path,
                    first_failure: first_error.to_string(),
                    retry_failure: retry_error.to_string(),
                }),
            },
            Err(error) => Err(owner_claim.authority_failure(&error)),
        }
    }

    fn retained_owner_claim_error(&self, reason: &impl ToString) -> OutputError {
        self.owner_claim.as_ref().map_or_else(
            || OutputError::Runtime(reason.to_string()),
            |owner_claim| owner_claim.authority_failure(reason),
        )
    }

    fn publish_terminal_and_release(
        &mut self,
        status: &AttemptManifestStatus,
        interrupted_signal: Option<&str>,
        failure_reason: Option<&str>,
    ) -> OutputResult<()> {
        if let Err(error) = self.publish_terminal(status, interrupted_signal, failure_reason) {
            return Err(self.retained_owner_claim_error(&error));
        }
        self.lifecycle_state.terminal = true;
        self.release_owner_claim()
    }

    fn publish_terminal(
        &mut self,
        status: &AttemptManifestStatus,
        interrupted_signal: Option<&str>,
        failure_reason: Option<&str>,
    ) -> OutputResult<()> {
        #[cfg(test)]
        fail_terminal_cleanup_at_test_point("before_terminal_publication")?;
        self.reestablish_current_durability_and_reinspect()?;
        self.refresh_active_attempt(OrphanPartPolicy::Observe, false)?;
        let run_set_id = self.active_attempt()?.run_set_id.clone();
        let attempt_id = self.active_attempt()?.attempt_id.clone();
        let StagedTerminalRuns { terminal, .. } = stage_terminal_runs(
            &self.runs,
            &self.run_plan,
            run_set_id,
            attempt_id,
            status,
            interrupted_signal,
            failure_reason,
        )?;
        self.lineage_paths.publish_terminal_claim(&terminal)?;
        #[cfg(test)]
        crash_at_test_failpoint("after_terminal_claim");
        let claimed_snapshot = self
            .lineage_paths
            .inspect()?
            .ok_or_else(|| OutputError::InvalidInput("Claimed output lineage disappeared.".to_string()))?;
        if claimed_snapshot.pending_terminal.as_ref() != Some(&terminal) {
            return Err(OutputError::InvalidInput("Output terminal claim changed before materialization.".to_string()));
        }
        self.refresh_active_attempt(OrphanPartPolicy::Reconcile, true)?;
        let staged_terminal_runs = stage_terminal_runs(
            &self.runs,
            &self.run_plan,
            terminal.run_set_id.clone(),
            terminal.attempt_id.clone(),
            status,
            interrupted_signal,
            failure_reason,
        )?;
        if staged_terminal_runs.terminal != terminal {
            return Err(OutputError::InvalidInput(
                "Output receipts changed after terminal authority was claimed.".to_string(),
            ));
        }
        materialize_staged_terminal_runs(&self.runs, &staged_terminal_runs.runs)?;
        self.reestablish_current_durability_and_reinspect()?;
        self.lineage_paths.finalize_terminal(&terminal)?;
        Ok(())
    }

    fn refresh_active_attempt(
        &mut self,
        orphan_part_policy: OrphanPartPolicy,
        allow_pending_terminal: bool,
    ) -> OutputResult<()> {
        let snapshot = self
            .lineage_paths
            .inspect()?
            .ok_or_else(|| OutputError::InvalidInput("Active output lineage disappeared.".to_string()))?;
        let active_attempt_id = self.active_attempt()?.attempt_id.clone();
        if snapshot.leaf_attempt_id != active_attempt_id
            || snapshot.leaf_terminal.is_some()
            || (!allow_pending_terminal && snapshot.pending_terminal.is_some())
        {
            return Err(OutputError::InvalidInput(
                "Active output lineage changed before terminal publication.".to_string(),
            ));
        }
        let canonical_chunk_plan = self.active_attempt()?.canonical_chunk_plan.clone();
        let allowed_attempts = lineage_attempt_identifiers(&snapshot);
        let verified_runs = verify_runs_against_snapshot(
            &self.lineage_paths,
            &self.runs,
            &snapshot,
            &canonical_chunk_plan,
            &allowed_attempts,
            false,
            orphan_part_policy,
        )?;
        if verified_runs.iter().any(|verified_run| verified_run.status != AttemptManifestStatus::Running) {
            return Err(OutputError::InvalidInput(
                "Active nonterminal lineage contains a terminal-status attempt manifest.".to_string(),
            ));
        }
        for (run, verified_run) in self.runs.iter_mut().zip(verified_runs) {
            run.receipts = verified_run.receipts;
            run.committed_chunk_identifiers = Arc::new(verified_run.committed_chunk_identifiers);
        }
        Ok(())
    }

    fn reverify_completed_noop(&mut self) -> OutputResult<()> {
        let snapshot = self
            .lineage_paths
            .inspect()?
            .ok_or_else(|| OutputError::InvalidInput("Completed output lineage disappeared.".to_string()))?;
        let active_attempt_id = self.active_attempt()?.attempt_id.clone();
        let canonical_chunk_plan = self.active_attempt()?.canonical_chunk_plan.clone();
        if snapshot.leaf_attempt_id != active_attempt_id
            || snapshot
                .leaf_terminal
                .as_ref()
                .is_none_or(|terminal| terminal.status != AttemptTerminalStatus::Completed)
        {
            return Err(OutputError::InvalidInput(
                "Completed output lineage changed during read-only verification.".to_string(),
            ));
        }
        let allowed_attempts = lineage_attempt_identifiers(&snapshot);
        verify_runs_against_snapshot(
            &self.lineage_paths,
            &self.runs,
            &snapshot,
            &canonical_chunk_plan,
            &allowed_attempts,
            true,
            OrphanPartPolicy::Reject,
        )?;
        self.lineage_snapshot = Some(snapshot);
        Ok(())
    }

    fn completed_outputs(&self) -> OutputResult<Vec<CompletedOutputRun>> {
        self.runs
            .iter()
            .map(|run| {
                let paths = run
                    .paths
                    .as_ref()
                    .ok_or_else(|| OutputError::Runtime("Completed output run has no paths.".to_string()))?;
                Ok(CompletedOutputRun {
                    run_directory: paths.run_directory.clone(),
                    parts_directory: paths.parts_directory.clone(),
                })
            })
            .collect()
    }
}

fn absolute_lexically_normalized_path(path: &Path) -> OutputResult<PathBuf> {
    let absolute_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map_err(OutputError::runtime)?.join(path)
    };
    let mut normalized_path = PathBuf::new();
    for component in absolute_path.components() {
        match component {
            Component::Prefix(prefix) => normalized_path.push(prefix.as_os_str()),
            Component::RootDir => normalized_path.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized_path.pop();
            }
            Component::Normal(name) => normalized_path.push(name),
        }
    }
    Ok(normalized_path)
}

fn contains_cli_line_separator(text: &str) -> bool {
    text.chars().any(|character| character.is_control() || matches!(character, '\u{2028}' | '\u{2029}'))
}

fn validate_open_policy(
    run_plan: &g_plan::RunPlan,
    lineage_paths: &OutputLineagePaths,
    snapshot: Option<&LineageSnapshot>,
    runs: &[ManagedOutputRun],
) -> OutputResult<()> {
    if snapshot.is_none() && run_plan.output.resume {
        let fenced_claim_id = run_plan.output.fenced_owner_claim_id.as_deref().ok_or_else(|| {
            OutputError::InvalidInput("Output resume requires an existing .g-output lineage.".to_string())
        })?;
        lineage_paths.require_fenced_owner_claim(fenced_claim_id)?;
    }
    validate_claimed_policy(run_plan, lineage_paths, snapshot, runs)
}

fn validate_claimed_policy(
    run_plan: &g_plan::RunPlan,
    lineage_paths: &OutputLineagePaths,
    snapshot: Option<&LineageSnapshot>,
    runs: &[ManagedOutputRun],
) -> OutputResult<()> {
    let Some(snapshot) = snapshot else {
        if run_plan.output.resume {
            if run_plan.output.recover_attempt.is_some() {
                return Err(OutputError::InvalidInput(
                    "Pre-genesis fenced owner recovery cannot name an output attempt.".to_string(),
                ));
            }
            if fresh_root_contains_output_artifacts(lineage_paths)? {
                return Err(OutputError::InvalidInput(format!(
                    "Pre-genesis output root '{}' contains artifacts not bound to owner recovery.",
                    lineage_paths.output_root.display()
                )));
            }
            return Ok(());
        }
        if run_plan.output.recover_attempt.is_some() {
            return Err(OutputError::InvalidInput(
                "Exact output recovery requires an existing nonterminal lineage.".to_string(),
            ));
        }
        if fresh_root_contains_output_artifacts(lineage_paths)? {
            return Err(OutputError::InvalidInput(format!(
                "Output run root '{}' already exists and is not empty.",
                lineage_paths.output_root.display()
            )));
        }
        return Ok(());
    };
    if !run_plan.output.resume {
        return Err(OutputError::InvalidInput("Existing output lineage requires [output].resume=true.".to_string()));
    }
    validate_planned_names(snapshot, runs)?;
    let terminal_authority = snapshot.leaf_terminal.as_ref().or(snapshot.pending_terminal.as_ref());
    match (terminal_authority, run_plan.output.recover_attempt.as_deref()) {
        (Some(_), Some(_)) => Err(OutputError::InvalidInput(
            "Exact output recovery is only valid for a nonterminal leaf attempt.".to_string(),
        )),
        (None, Some(recover_attempt)) if recover_attempt == snapshot.leaf_attempt_id.as_str() => Ok(()),
        (None, Some(recover_attempt)) => Err(OutputError::InvalidInput(format!(
            "Exact output recovery names attempt '{recover_attempt}', but the nonterminal leaf is '{}'.",
            snapshot.leaf_attempt_id.as_str()
        ))),
        (None, None) => Err(OutputError::InvalidInput(format!(
            "Nonterminal output attempt '{}' requires exact output.recover_attempt authorization.",
            snapshot.leaf_attempt_id.as_str()
        ))),
        (Some(_), None) => Ok(()),
    }
}

fn validate_planned_names(snapshot: &LineageSnapshot, runs: &[ManagedOutputRun]) -> OutputResult<()> {
    if snapshot.genesis.phenotypes.len() != runs.len() {
        return Err(OutputError::InvalidInput(
            "Output lineage phenotype count does not match the current run plan.".to_string(),
        ));
    }
    for (contract, run) in snapshot.genesis.phenotypes.iter().zip(runs) {
        if contract.phenotype_name != run.phenotype_name || contract.output_directory_name != run.output_directory_name
        {
            return Err(OutputError::InvalidInput(
                "Output lineage phenotype order or directory names do not match the current run plan.".to_string(),
            ));
        }
    }
    Ok(())
}

fn build_headers(
    core: &OutputManagerCore,
    current_header_inputs: Vec<CurrentRunManifestHeaderInput>,
) -> OutputResult<BTreeMap<String, Value>> {
    let mut headers = BTreeMap::new();
    let mut fingerprint_cache = ManifestFileFingerprintCache::default();
    for input in current_header_inputs {
        let phenotype_name = input.phenotype_name.clone();
        let header =
            build_current_run_manifest_header_value_with_cache(&core.run_plan, &input, &mut fingerprint_cache)?;
        match headers.entry(phenotype_name) {
            Entry::Vacant(entry) => {
                entry.insert(header);
            }
            Entry::Occupied(entry) => {
                return Err(OutputError::InvalidInput(format!(
                    "Duplicate output initialization for phenotype '{}'.",
                    entry.key()
                )));
            }
        }
    }
    if headers.len() != core.runs.len() || core.runs.iter().any(|run| !headers.contains_key(&run.phenotype_name)) {
        return Err(OutputError::InvalidInput(
            "Output initialization headers do not cover planned phenotypes exactly.".to_string(),
        ));
    }
    Ok(headers)
}

fn bind_headers(runs: &mut [ManagedOutputRun], mut headers: BTreeMap<String, Value>) -> OutputResult<()> {
    for run in runs {
        run.current_header = Some(headers.remove(&run.phenotype_name).ok_or_else(|| {
            OutputError::InvalidInput(format!("Missing output initialization for phenotype '{}'.", run.phenotype_name))
        })?);
    }
    Ok(())
}

fn phenotype_contracts(runs: &[ManagedOutputRun]) -> OutputResult<Vec<PhenotypeLineageContract>> {
    runs.iter()
        .map(|run| {
            Ok(PhenotypeLineageContract {
                phenotype_name: run.phenotype_name.clone(),
                output_directory_name: run.output_directory_name.clone(),
                execution_plan_sha256: execution_plan_hash(run)?,
            })
        })
        .collect()
}

fn execution_plan_hash(run: &ManagedOutputRun) -> OutputResult<String> {
    run.current_header
        .as_ref()
        .and_then(|header| header.get("execution_plan_hash"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Output header for phenotype '{}' has no execution plan hash.",
                run.phenotype_name
            ))
        })
}

fn initialize_genesis_attempt(
    core: &mut OutputManagerCore,
    canonical_chunk_plan: &CanonicalChunkPlan,
    phenotype_contracts: Vec<PhenotypeLineageContract>,
    collect_stage_timings: bool,
) -> OutputResult<()> {
    let attempt_id = core.claimed_attempt_id()?;
    let genesis =
        LineageGenesisRecord::new(attempt_id.clone(), canonical_chunk_plan.sha256().to_string(), phenotype_contracts);
    core.lineage_paths.initialize_directories()?;
    let genesis_path = core.lineage_paths.genesis_path.clone();
    let publication_result = core.lineage_paths.publish_genesis(&genesis);
    core.classify_attempt_authority_publication(genesis_path, publication_result)?;
    #[cfg(test)]
    crash_at_test_failpoint("after_genesis_publication_before_staging_retirement");
    core.retire_claim_staging_intent()?;
    #[cfg(test)]
    crash_at_test_failpoint("after_genesis_claim");
    core.install_claimed_attempt_shell(
        genesis.run_set_id.clone(),
        attempt_id.clone(),
        canonical_chunk_plan.clone(),
        collect_stage_timings,
    )?;
    #[cfg(test)]
    fail_initialization_at_test_point("after_attempt_claim")?;
    let prepared_attempt = prepare_attempt(core, genesis.run_set_id.clone(), attempt_id, canonical_chunk_plan, None)?;
    #[cfg(test)]
    fail_initialization_at_test_point("after_attempt_preparation")?;
    core.install_prepared_attempt(prepared_attempt, canonical_chunk_plan.clone(), false, collect_stage_timings)?;
    core.start_writers()?;
    #[cfg(test)]
    fail_initialization_at_test_point("after_writer_start")?;
    core.lineage_snapshot = core.lineage_paths.inspect()?;
    Ok(())
}

fn preflight_existing_lineage(
    core: &OutputManagerCore,
    snapshot: &LineageSnapshot,
    canonical_chunk_plan: &CanonicalChunkPlan,
    phenotype_contracts: &[PhenotypeLineageContract],
) -> OutputResult<()> {
    if snapshot.genesis.chunk_plan_sha256 != canonical_chunk_plan.sha256()
        || snapshot.genesis.phenotypes != phenotype_contracts
    {
        return Err(OutputError::InvalidInput(
            "Output lineage contract does not match the current chunk or execution plan.".to_string(),
        ));
    }
    let allowed_attempts = lineage_attempt_identifiers(snapshot);
    let verified_runs = verify_runs_against_snapshot(
        &core.lineage_paths,
        &core.runs,
        snapshot,
        canonical_chunk_plan,
        &allowed_attempts,
        snapshot.leaf_terminal.is_some(),
        OrphanPartPolicy::Observe,
    )?;
    if snapshot.leaf_terminal.is_none()
        && snapshot.pending_terminal.is_none()
        && verified_runs.iter().any(|verified_run| verified_run.status != AttemptManifestStatus::Running)
    {
        return Err(OutputError::InvalidInput(
            "Nonterminal output lineage has a terminal-status attempt manifest.".to_string(),
        ));
    }
    Ok(())
}

fn initialize_existing_lineage(
    core: &mut OutputManagerCore,
    mut snapshot: LineageSnapshot,
    canonical_chunk_plan: &CanonicalChunkPlan,
    phenotype_contracts: &[PhenotypeLineageContract],
    collect_stage_timings: bool,
) -> OutputResult<()> {
    validate_existing_lineage_contract(&snapshot, canonical_chunk_plan, phenotype_contracts)?;
    if snapshot.pending_terminal.is_some() {
        finalize_pending_terminal(core, &snapshot, canonical_chunk_plan)?;
        snapshot = core
            .lineage_paths
            .inspect()?
            .ok_or_else(|| OutputError::InvalidInput("Finalized output lineage disappeared.".to_string()))?;
    }
    let allowed_attempts = lineage_attempt_identifiers(&snapshot);
    let verified_runs = verify_runs_against_snapshot(
        &core.lineage_paths,
        &core.runs,
        &snapshot,
        canonical_chunk_plan,
        &allowed_attempts,
        snapshot.leaf_terminal.is_some(),
        if snapshot.leaf_terminal.is_some() { OrphanPartPolicy::Reject } else { OrphanPartPolicy::Observe },
    )?;
    if snapshot.leaf_terminal.as_ref().is_some_and(|terminal| terminal.status == AttemptTerminalStatus::Completed) {
        let prepared_attempt = prepared_attempt_from_verified(
            core,
            snapshot.genesis.run_set_id.clone(),
            snapshot.leaf_attempt_id.clone(),
            verified_runs,
        )?;
        core.install_prepared_attempt(prepared_attempt, canonical_chunk_plan.clone(), true, collect_stage_timings)?;
        core.lineage_snapshot = Some(snapshot);
        return Ok(());
    }
    if snapshot.leaf_terminal.is_none()
        && verified_runs.iter().any(|verified_run| verified_run.status != AttemptManifestStatus::Running)
    {
        return Err(OutputError::InvalidInput(
            "Nonterminal output lineage has a terminal-status attempt manifest.".to_string(),
        ));
    }
    let new_attempt_id = core.claimed_attempt_id()?;
    let source_attempt_id = snapshot.leaf_attempt_id.clone();
    let run_set_id = snapshot.genesis.run_set_id.clone();
    let source_receipts = verified_runs.into_iter().map(|verified_run| verified_run.receipts).collect::<Vec<_>>();
    let successor = match snapshot.leaf_terminal {
        Some(_) => LineageSuccessorRecord::new(
            run_set_id.clone(),
            source_attempt_id.clone(),
            new_attempt_id.clone(),
            LineageRecoveryKind::TerminalResume,
            Some(terminal_record_sha256(&core.lineage_paths, &source_attempt_id)?),
        )?,
        None => LineageSuccessorRecord::new(
            run_set_id.clone(),
            source_attempt_id.clone(),
            new_attempt_id.clone(),
            LineageRecoveryKind::ExactNonterminalRecovery,
            None,
        )?,
    };
    core.reestablish_observed_durability(Some(&snapshot))?;
    let durable_parent_snapshot = core
        .lineage_paths
        .inspect()?
        .ok_or_else(|| OutputError::InvalidInput("Output successor parent disappeared.".to_string()))?;
    if durable_parent_snapshot != snapshot {
        return Err(OutputError::InvalidInput(
            "Output successor parent changed while re-establishing durability.".to_string(),
        ));
    }
    let successor_path = match successor.recovery_kind {
        LineageRecoveryKind::TerminalResume => core.lineage_paths.normal_successor_path(&source_attempt_id),
        LineageRecoveryKind::ExactNonterminalRecovery => core.lineage_paths.outcome_path(&source_attempt_id),
    };
    let publication_result = core.lineage_paths.publish_successor(&successor);
    core.classify_attempt_authority_publication(successor_path, publication_result)?;
    #[cfg(test)]
    crash_at_test_failpoint("after_successor_publication_before_staging_retirement");
    core.retire_claim_staging_intent()?;
    #[cfg(test)]
    crash_at_test_failpoint("after_successor_claim");
    core.install_claimed_attempt_shell(
        run_set_id.clone(),
        new_attempt_id.clone(),
        canonical_chunk_plan.clone(),
        collect_stage_timings,
    )?;
    #[cfg(test)]
    fail_initialization_at_test_point("after_attempt_claim")?;
    core.lineage_snapshot = core.lineage_paths.inspect()?;
    let prepared_attempt = prepare_attempt(
        core,
        run_set_id,
        new_attempt_id.clone(),
        canonical_chunk_plan,
        Some((&source_attempt_id, &source_receipts)),
    )?;
    #[cfg(test)]
    fail_initialization_at_test_point("after_attempt_preparation")?;
    core.install_prepared_attempt(prepared_attempt, canonical_chunk_plan.clone(), false, collect_stage_timings)?;
    core.start_writers()?;
    #[cfg(test)]
    fail_initialization_at_test_point("after_writer_start")?;
    Ok(())
}

fn validate_existing_lineage_contract(
    snapshot: &LineageSnapshot,
    canonical_chunk_plan: &CanonicalChunkPlan,
    phenotype_contracts: &[PhenotypeLineageContract],
) -> OutputResult<()> {
    if snapshot.genesis.chunk_plan_sha256 == canonical_chunk_plan.sha256()
        && snapshot.genesis.phenotypes == phenotype_contracts
    {
        Ok(())
    } else {
        Err(OutputError::InvalidInput(
            "Output lineage contract does not match the current chunk or execution plan.".to_string(),
        ))
    }
}

fn finalize_pending_terminal(
    core: &mut OutputManagerCore,
    snapshot: &LineageSnapshot,
    canonical_chunk_plan: &CanonicalChunkPlan,
) -> OutputResult<()> {
    core.reestablish_observed_durability(Some(snapshot))?;
    let durable_snapshot = core
        .lineage_paths
        .inspect()?
        .ok_or_else(|| OutputError::InvalidInput("Pending output lineage disappeared.".to_string()))?;
    if &durable_snapshot != snapshot {
        return Err(OutputError::InvalidInput(
            "Pending output lineage changed while re-establishing durability.".to_string(),
        ));
    }
    let terminal = snapshot.pending_terminal.as_ref().ok_or_else(|| {
        OutputError::Runtime("Pending output terminal finalization is missing its claim.".to_string())
    })?;
    let status = match terminal.status {
        AttemptTerminalStatus::Completed => AttemptManifestStatus::Completed,
        AttemptTerminalStatus::Interrupted => AttemptManifestStatus::Interrupted,
        AttemptTerminalStatus::Failed => AttemptManifestStatus::Failed,
    };
    let allowed_attempts = lineage_attempt_identifiers(snapshot);
    let observed_runs = verify_runs_against_snapshot(
        &core.lineage_paths,
        &core.runs,
        snapshot,
        canonical_chunk_plan,
        &allowed_attempts,
        false,
        OrphanPartPolicy::Observe,
    )?;
    install_verified_leaf_run_state(core, snapshot, canonical_chunk_plan, observed_runs)?;
    let StagedTerminalRuns { terminal: observed_terminal, .. } = stage_terminal_runs(
        &core.runs,
        &core.run_plan,
        terminal.run_set_id.clone(),
        terminal.attempt_id.clone(),
        &status,
        terminal.interrupted_signal.as_deref(),
        terminal.failure_reason.as_deref(),
    )?;
    if &observed_terminal != terminal {
        return Err(OutputError::InvalidInput(
            "Pending output terminal cannot be reconstructed from durable parts.".to_string(),
        ));
    }
    let reconciled_runs = verify_runs_against_snapshot(
        &core.lineage_paths,
        &core.runs,
        snapshot,
        canonical_chunk_plan,
        &allowed_attempts,
        false,
        OrphanPartPolicy::Reconcile,
    )?;
    install_verified_leaf_run_state(core, snapshot, canonical_chunk_plan, reconciled_runs)?;
    let staged_terminal_runs = stage_terminal_runs(
        &core.runs,
        &core.run_plan,
        terminal.run_set_id.clone(),
        terminal.attempt_id.clone(),
        &status,
        terminal.interrupted_signal.as_deref(),
        terminal.failure_reason.as_deref(),
    )?;
    if &staged_terminal_runs.terminal != terminal {
        return Err(OutputError::InvalidInput(
            "Pending output terminal changed during receipt reconciliation.".to_string(),
        ));
    }
    materialize_staged_terminal_runs(&core.runs, &staged_terminal_runs.runs)?;
    core.reestablish_observed_durability(Some(snapshot))?;
    let finalization_snapshot = core.lineage_paths.inspect()?.ok_or_else(|| {
        OutputError::InvalidInput("Pending output lineage disappeared before finalization.".to_string())
    })?;
    if finalization_snapshot.pending_terminal.as_ref() != Some(terminal) {
        return Err(OutputError::InvalidInput("Pending output terminal changed before finalization.".to_string()));
    }
    core.lineage_paths.finalize_terminal(terminal)?;
    Ok(())
}

fn install_verified_leaf_run_state(
    core: &mut OutputManagerCore,
    snapshot: &LineageSnapshot,
    canonical_chunk_plan: &CanonicalChunkPlan,
    verified_runs: Vec<VerifiedAttemptRun>,
) -> OutputResult<()> {
    if verified_runs.len() != core.runs.len() {
        return Err(OutputError::Runtime("Verified output attempt run count is inconsistent.".to_string()));
    }
    for (run, verified_run) in core.runs.iter_mut().zip(verified_runs) {
        run.binding = Some(AttemptManifestBinding {
            run_set_id: snapshot.genesis.run_set_id.clone(),
            attempt_id: snapshot.leaf_attempt_id.clone(),
            phenotype_name: run.phenotype_name.clone(),
            output_directory_name: run.output_directory_name.clone(),
            execution_plan_sha256: execution_plan_hash(run)?,
            chunk_plan_sha256: canonical_chunk_plan.sha256().to_string(),
        });
        run.paths = Some(AttemptRunPaths::new(
            &core.lineage_paths.attempts_directory,
            &snapshot.leaf_attempt_id,
            &run.output_directory_name,
        )?);
        run.receipts = verified_run.receipts;
        run.committed_chunk_identifiers = Arc::new(verified_run.committed_chunk_identifiers);
    }
    Ok(())
}

fn prepare_attempt(
    core: &OutputManagerCore,
    run_set_id: String,
    attempt_id: AttemptIdentifier,
    canonical_chunk_plan: &CanonicalChunkPlan,
    reuse_source: Option<(&AttemptIdentifier, &[Vec<OutputPartReceipt>])>,
) -> OutputResult<PreparedAttempt> {
    let mut prepared_runs = Vec::with_capacity(core.runs.len());
    for (run_index, run) in core.runs.iter().enumerate() {
        let paths =
            AttemptRunPaths::new(&core.lineage_paths.attempts_directory, &attempt_id, &run.output_directory_name)?;
        paths.initialize_directories()?;
        let binding = AttemptManifestBinding {
            run_set_id: run_set_id.clone(),
            attempt_id: attempt_id.clone(),
            phenotype_name: run.phenotype_name.clone(),
            output_directory_name: run.output_directory_name.clone(),
            execution_plan_sha256: execution_plan_hash(run)?,
            chunk_plan_sha256: canonical_chunk_plan.sha256().to_string(),
        };
        let receipts = if let Some((source_attempt_id, source_receipts)) = reuse_source {
            let source_paths = AttemptRunPaths::new(
                &core.lineage_paths.attempts_directory,
                source_attempt_id,
                &run.output_directory_name,
            )?;
            let receipts = source_receipts
                .get(run_index)
                .cloned()
                .ok_or_else(|| OutputError::Runtime("Output reuse receipt count is inconsistent.".to_string()))?;
            reuse_verified_receipts(&source_paths, &paths, &receipts)?;
            receipts
        } else {
            Vec::new()
        };
        write_effective_config(&paths, &core.effective_config_toml)?;
        let header = run
            .current_header
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output run header is not initialized.".to_string()))?;
        write_attempt_manifest(&AttemptManifestWrite {
            paths: &paths,
            binding: &binding,
            header,
            status: AttemptManifestStatus::Running,
            interrupted_signal: None,
            failure_reason: None,
            receipts: &receipts,
            run_plan: &core.run_plan,
        })?;
        let committed_chunk_identifiers = receipt_chunk_identifiers(&receipts)?;
        prepared_runs.push(PreparedAttemptRun { binding, paths, receipts, committed_chunk_identifiers });
    }
    Ok(PreparedAttempt { run_set_id, attempt_id, runs: prepared_runs })
}

fn verify_runs_against_snapshot(
    lineage_paths: &OutputLineagePaths,
    runs: &[ManagedOutputRun],
    snapshot: &LineageSnapshot,
    canonical_chunk_plan: &CanonicalChunkPlan,
    allowed_attempts: &BTreeSet<AttemptIdentifier>,
    require_terminal_manifest: bool,
    orphan_part_policy: OrphanPartPolicy,
) -> OutputResult<Vec<VerifiedAttemptRun>> {
    let terminal_records = snapshot
        .leaf_terminal
        .as_ref()
        .map(|terminal| {
            terminal
                .phenotypes
                .iter()
                .map(|record| (record.phenotype_name.as_str(), record))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let mut verified_runs = Vec::with_capacity(runs.len());
    for run in runs {
        let header = run
            .current_header
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output run header is not initialized.".to_string()))?;
        let binding = AttemptManifestBinding {
            run_set_id: snapshot.genesis.run_set_id.clone(),
            attempt_id: snapshot.leaf_attempt_id.clone(),
            phenotype_name: run.phenotype_name.clone(),
            output_directory_name: run.output_directory_name.clone(),
            execution_plan_sha256: execution_plan_hash(run)?,
            chunk_plan_sha256: canonical_chunk_plan.sha256().to_string(),
        };
        let paths = AttemptRunPaths::new(
            &lineage_paths.attempts_directory,
            &snapshot.leaf_attempt_id,
            &run.output_directory_name,
        )?;
        let manifest_bytes = read_optional_attempt_manifest_bytes(&paths.manifest_path)?;
        let verified = if let Some(manifest_bytes) = manifest_bytes {
            verify_attempt_run(
                &paths,
                &manifest_bytes,
                &binding,
                header,
                canonical_chunk_plan,
                allowed_attempts,
                require_terminal_manifest,
                orphan_part_policy,
            )?
        } else if require_terminal_manifest {
            return Err(OutputError::InvalidInput(format!(
                "Terminal output attempt is missing manifest '{}'.",
                paths.manifest_path.display()
            )));
        } else {
            inspect_unmaterialized_attempt_run(&paths, &binding, canonical_chunk_plan, allowed_attempts)?
        };
        if let Some(terminal) = snapshot.leaf_terminal.as_ref() {
            validate_manifest_terminal_status(&verified.status, terminal.status)?;
            let terminal_record = terminal_records.get(run.phenotype_name.as_str()).ok_or_else(|| {
                OutputError::InvalidInput(format!("Output terminal is missing phenotype '{}'.", run.phenotype_name))
            })?;
            if terminal_record.output_directory_name != run.output_directory_name
                || terminal_record.run_manifest_sha256 != verified.manifest_sha256
            {
                return Err(OutputError::InvalidInput(format!(
                    "Output terminal phenotype '{}' has a stale manifest binding.",
                    run.phenotype_name
                )));
            }
        }
        verified_runs.push(verified);
    }
    if snapshot.leaf_terminal.is_some() && terminal_records.len() != runs.len() {
        return Err(OutputError::InvalidInput("Output terminal phenotype coverage is not exact.".to_string()));
    }
    Ok(verified_runs)
}

fn prepared_attempt_from_verified(
    core: &OutputManagerCore,
    run_set_id: String,
    attempt_id: AttemptIdentifier,
    verified_runs: Vec<VerifiedAttemptRun>,
) -> OutputResult<PreparedAttempt> {
    let mut prepared_runs = Vec::with_capacity(core.runs.len());
    for (run, verified) in core.runs.iter().zip(verified_runs) {
        let binding = AttemptManifestBinding {
            run_set_id: run_set_id.clone(),
            attempt_id: attempt_id.clone(),
            phenotype_name: run.phenotype_name.clone(),
            output_directory_name: run.output_directory_name.clone(),
            execution_plan_sha256: execution_plan_hash(run)?,
            chunk_plan_sha256: core
                .lineage_snapshot
                .as_ref()
                .ok_or_else(|| OutputError::Runtime("Output lineage snapshot is missing.".to_string()))?
                .genesis
                .chunk_plan_sha256
                .clone(),
        };
        let paths =
            AttemptRunPaths::new(&core.lineage_paths.attempts_directory, &attempt_id, &run.output_directory_name)?;
        prepared_runs.push(PreparedAttemptRun {
            binding,
            paths,
            receipts: verified.receipts,
            committed_chunk_identifiers: verified.committed_chunk_identifiers,
        });
    }
    Ok(PreparedAttempt { run_set_id, attempt_id, runs: prepared_runs })
}

fn stage_terminal_runs(
    runs: &[ManagedOutputRun],
    run_plan: &g_plan::RunPlan,
    run_set_id: String,
    attempt_id: AttemptIdentifier,
    status: &AttemptManifestStatus,
    interrupted_signal: Option<&str>,
    failure_reason: Option<&str>,
) -> OutputResult<StagedTerminalRuns> {
    let mut phenotype_records = Vec::with_capacity(runs.len());
    let mut staged_runs = Vec::with_capacity(runs.len());
    for run in runs {
        let paths = run
            .paths
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output run paths are not initialized.".to_string()))?;
        let binding = run
            .binding
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output run binding is not initialized.".to_string()))?;
        let header = run
            .current_header
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output run header is not initialized.".to_string()))?;
        let manifest = build_attempt_manifest_value(&AttemptManifestWrite {
            paths,
            binding,
            header,
            status: status.clone(),
            interrupted_signal,
            failure_reason,
            receipts: &run.receipts,
            run_plan,
        })?;
        let manifest_sha256 = attempt_manifest_value_sha256(&manifest)?;
        phenotype_records.push(TerminalPhenotypeRecord {
            phenotype_name: run.phenotype_name.clone(),
            output_directory_name: run.output_directory_name.clone(),
            run_manifest_sha256: manifest_sha256.clone(),
        });
        staged_runs.push(StagedTerminalRun { manifest, manifest_sha256 });
    }
    let terminal = match status {
        AttemptManifestStatus::Completed => LineageTerminalRecord::completed(run_set_id, attempt_id, phenotype_records),
        AttemptManifestStatus::Interrupted => LineageTerminalRecord::interrupted(
            run_set_id,
            attempt_id,
            interrupted_signal
                .ok_or_else(|| OutputError::Runtime("Interrupted terminal is missing its signal.".to_string()))?
                .to_string(),
            phenotype_records,
        ),
        AttemptManifestStatus::Failed => LineageTerminalRecord::failed(
            run_set_id,
            attempt_id,
            failure_reason
                .ok_or_else(|| OutputError::Runtime("Failed terminal is missing its reason.".to_string()))?
                .to_string(),
            phenotype_records,
        ),
        AttemptManifestStatus::Running => {
            return Err(OutputError::InvalidInput(
                "A running attempt manifest cannot be staged as a terminal.".to_string(),
            ));
        }
    };
    Ok(StagedTerminalRuns { terminal, runs: staged_runs })
}

fn materialize_staged_terminal_runs(runs: &[ManagedOutputRun], staged_runs: &[StagedTerminalRun]) -> OutputResult<()> {
    if runs.len() != staged_runs.len() {
        return Err(OutputError::Runtime("Staged output terminal run count is inconsistent.".to_string()));
    }
    for (run, staged_run) in runs.iter().zip(staged_runs) {
        let paths = run
            .paths
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Output run paths are not initialized.".to_string()))?;
        materialize_attempt_manifest(paths, &staged_run.manifest, &staged_run.manifest_sha256)?;
        #[cfg(test)]
        crash_at_test_failpoint("after_terminal_run_materialization");
    }
    Ok(())
}

fn lineage_attempt_identifiers(snapshot: &LineageSnapshot) -> BTreeSet<AttemptIdentifier> {
    let mut attempts = BTreeSet::from([snapshot.genesis.attempt_id.clone()]);
    attempts.extend(snapshot.successor_records.iter().map(|successor| successor.attempt_id.clone()));
    attempts
}

fn validate_manifest_terminal_status(
    manifest_status: &AttemptManifestStatus,
    terminal_status: AttemptTerminalStatus,
) -> OutputResult<()> {
    let matches = matches!(
        (manifest_status, terminal_status),
        (AttemptManifestStatus::Completed, AttemptTerminalStatus::Completed)
            | (AttemptManifestStatus::Interrupted, AttemptTerminalStatus::Interrupted)
            | (AttemptManifestStatus::Failed, AttemptTerminalStatus::Failed)
    );
    if matches {
        Ok(())
    } else {
        Err(OutputError::InvalidInput("Output terminal status does not match its attempt manifests.".to_string()))
    }
}

fn receipt_chunk_identifiers(receipts: &[OutputPartReceipt]) -> OutputResult<BTreeSet<usize>> {
    let mut identifiers = BTreeSet::new();
    for receipt in receipts {
        for chunk in &receipt.footer.chunks {
            let identifier = usize::try_from(chunk.chunk_identifier).map_err(|_| {
                OutputError::InvalidInput(format!(
                    "Output chunk identifier {} does not fit the platform index width.",
                    chunk.chunk_identifier
                ))
            })?;
            if !identifiers.insert(identifier) {
                return Err(OutputError::InvalidInput(format!(
                    "Output receipts contain duplicate chunk identifier {identifier}."
                )));
            }
        }
    }
    Ok(identifiers)
}

fn directory_exists_and_is_non_empty(directory: &Path) -> OutputResult<bool> {
    match std::fs::read_dir(directory) {
        Ok(mut entries) => entries.next().transpose().map(|entry| entry.is_some()).map_err(OutputError::runtime),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(OutputError::runtime(error)),
    }
}

fn fresh_root_contains_output_artifacts(lineage_paths: &OutputLineagePaths) -> OutputResult<bool> {
    let owner_staging_bindings = lineage_paths.owner_staging_bindings()?;
    let root_entries = match std::fs::read_dir(&lineage_paths.output_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(OutputError::runtime(error)),
    };
    for root_entry in root_entries {
        let root_entry = root_entry.map_err(OutputError::runtime)?;
        let file_name = root_entry.file_name();
        if file_name == ".g-output" {
            if control_directory_contains_output_artifacts(&lineage_paths.control_directory)? {
                return Ok(true);
            }
        } else if file_name == "attempts" {
            if attempts_directory_contains_non_staging_artifacts(&root_entry.path(), &owner_staging_bindings)? {
                return Ok(true);
            }
        } else {
            return Ok(true);
        }
    }
    Ok(false)
}

fn attempts_directory_contains_non_staging_artifacts(
    attempts_directory: &Path,
    owner_staging_bindings: &BTreeMap<AttemptIdentifier, BTreeSet<String>>,
) -> OutputResult<bool> {
    let entries = match std::fs::read_dir(attempts_directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(OutputError::runtime(error)),
    };
    for entry in entries {
        let entry = entry.map_err(OutputError::runtime)?;
        if !entry.file_type().map_err(OutputError::runtime)?.is_dir() {
            return Ok(true);
        }
        let Some(attempt_name) = entry.file_name().to_str().map(str::to_string) else {
            return Ok(true);
        };
        let Ok(attempt_id) = AttemptIdentifier::parse(&attempt_name) else {
            return Ok(true);
        };
        let Some(claim_identifiers) = owner_staging_bindings.get(&attempt_id) else {
            return Ok(true);
        };
        let attempt_entries = std::fs::read_dir(entry.path()).map_err(OutputError::runtime)?;
        for attempt_entry in attempt_entries {
            let attempt_entry = attempt_entry.map_err(OutputError::runtime)?;
            if attempt_entry.file_name() != "diagnostics"
                || !attempt_entry.file_type().map_err(OutputError::runtime)?.is_dir()
            {
                return Ok(true);
            }
            let diagnostic_entries = std::fs::read_dir(attempt_entry.path()).map_err(OutputError::runtime)?;
            for diagnostic_entry in diagnostic_entries {
                let diagnostic_entry = diagnostic_entry.map_err(OutputError::runtime)?;
                let Some(claim_identifier) = diagnostic_entry.file_name().to_str().map(str::to_string) else {
                    return Ok(true);
                };
                if !claim_identifiers.contains(&claim_identifier)
                    || !diagnostic_entry.file_type().map_err(OutputError::runtime)?.is_dir()
                {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
fn crash_at_test_failpoint(expected_failpoint: &str) {
    if std::env::var("G_OUTPUT_TEST_CRASH_POINT").as_deref() == Ok(expected_failpoint) {
        std::process::exit(86);
    }
}

#[cfg(test)]
std::thread_local! {
    static INITIALIZATION_FAILURE_POINT: std::cell::RefCell<Option<String>> = const {
        std::cell::RefCell::new(None)
    };
    static INITIALIZATION_CLEANUP_FAILURE_POINT: std::cell::RefCell<Option<String>> = const {
        std::cell::RefCell::new(None)
    };
    static LIFECYCLE_FAILURE_POINT: std::cell::RefCell<Option<String>> = const {
        std::cell::RefCell::new(None)
    };
    static TERMINAL_CLEANUP_FAILURE_POINT: std::cell::RefCell<Option<String>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
pub(crate) struct InitializationFailureGuard;

#[cfg(test)]
pub(crate) struct InitializationCleanupFailureGuard;

#[cfg(test)]
pub(crate) struct LifecycleFailureGuard;

#[cfg(test)]
pub(crate) struct TerminalCleanupFailureGuard;

#[cfg(test)]
impl Drop for InitializationFailureGuard {
    fn drop(&mut self) {
        INITIALIZATION_FAILURE_POINT.with(|failure_point| {
            *failure_point.borrow_mut() = None;
        });
    }
}

#[cfg(test)]
impl Drop for InitializationCleanupFailureGuard {
    fn drop(&mut self) {
        INITIALIZATION_CLEANUP_FAILURE_POINT.with(|failure_point| {
            *failure_point.borrow_mut() = None;
        });
    }
}

#[cfg(test)]
impl Drop for LifecycleFailureGuard {
    fn drop(&mut self) {
        LIFECYCLE_FAILURE_POINT.with(|failure_point| {
            *failure_point.borrow_mut() = None;
        });
    }
}

#[cfg(test)]
impl Drop for TerminalCleanupFailureGuard {
    fn drop(&mut self) {
        TERMINAL_CLEANUP_FAILURE_POINT.with(|failure_point| {
            *failure_point.borrow_mut() = None;
        });
    }
}

#[cfg(test)]
pub(crate) fn install_initialization_failure_for_test(failure_point: &str) -> InitializationFailureGuard {
    INITIALIZATION_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        assert!(installed_failure_point.is_none(), "an initialization failure point is already installed");
        *installed_failure_point = Some(failure_point.to_string());
    });
    InitializationFailureGuard
}

#[cfg(test)]
pub(crate) fn install_initialization_cleanup_failure_for_test(
    failure_point: &str,
) -> InitializationCleanupFailureGuard {
    INITIALIZATION_CLEANUP_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        assert!(installed_failure_point.is_none(), "an initialization cleanup failure point is already installed");
        *installed_failure_point = Some(failure_point.to_string());
    });
    InitializationCleanupFailureGuard
}

#[cfg(test)]
pub(crate) fn install_lifecycle_failure_for_test(failure_point: &str) -> LifecycleFailureGuard {
    LIFECYCLE_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        assert!(installed_failure_point.is_none(), "a lifecycle failure point is already installed");
        *installed_failure_point = Some(failure_point.to_string());
    });
    LifecycleFailureGuard
}

#[cfg(test)]
pub(crate) fn install_terminal_cleanup_failure_for_test(failure_point: &str) -> TerminalCleanupFailureGuard {
    TERMINAL_CLEANUP_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        assert!(installed_failure_point.is_none(), "a terminal cleanup failure point is already installed");
        *installed_failure_point = Some(failure_point.to_string());
    });
    TerminalCleanupFailureGuard
}

#[cfg(test)]
fn fail_initialization_at_test_point(observed_failure_point: &str) -> OutputResult<()> {
    INITIALIZATION_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        if installed_failure_point.as_deref() == Some(observed_failure_point) {
            *installed_failure_point = None;
            Err(OutputError::Runtime(format!("Injected output initialization failure at '{observed_failure_point}'.")))
        } else {
            Ok(())
        }
    })
}

#[cfg(test)]
fn fail_initialization_cleanup_at_test_point(observed_failure_point: &str) -> OutputResult<()> {
    INITIALIZATION_CLEANUP_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        if installed_failure_point.as_deref() == Some(observed_failure_point) {
            *installed_failure_point = None;
            Err(OutputError::Runtime(format!(
                "Injected output initialization cleanup failure at '{observed_failure_point}'."
            )))
        } else {
            Ok(())
        }
    })
}

#[cfg(test)]
fn fail_lifecycle_at_test_point(observed_failure_point: &str) -> OutputResult<()> {
    LIFECYCLE_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        if installed_failure_point.as_deref() == Some(observed_failure_point) {
            *installed_failure_point = None;
            Err(OutputError::Runtime(format!("Injected output lifecycle failure at '{observed_failure_point}'.")))
        } else {
            Ok(())
        }
    })
}

#[cfg(test)]
fn fail_terminal_cleanup_at_test_point(observed_failure_point: &str) -> OutputResult<()> {
    TERMINAL_CLEANUP_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        if installed_failure_point.as_deref() == Some(observed_failure_point) {
            *installed_failure_point = None;
            Err(OutputError::Runtime(format!(
                "Injected output terminal cleanup failure at '{observed_failure_point}'."
            )))
        } else {
            Ok(())
        }
    })
}

#[cfg(test)]
fn inject_owner_claim_cleanup_conflict_at_test_point(owner_claim: &OutputOwnerClaim) -> OutputResult<()> {
    TERMINAL_CLEANUP_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        if installed_failure_point.as_deref() != Some("owner_claim_release_conflict") {
            return Ok(());
        }
        *installed_failure_point = None;
        owner_claim.publish_conflicting_takeover_for_test()
    })
}

#[cfg(test)]
fn inject_owner_claim_release_conflict_at_test_point(
    owner_claim: &OutputOwnerClaim,
) -> OutputResult<Option<OutputError>> {
    INITIALIZATION_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        if installed_failure_point.as_deref() != Some("after_owner_claim_release_conflict") {
            return Ok(None);
        }
        *installed_failure_point = None;
        owner_claim.publish_conflicting_takeover_for_test()?;
        Ok(Some(OutputError::Runtime("Injected output initialization failure before owner-claim release.".to_string())))
    })
}

fn control_directory_contains_output_artifacts(control_directory: &Path) -> OutputResult<bool> {
    let entries = match std::fs::read_dir(control_directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(false);
        }
        Err(error) => return Err(OutputError::runtime(error)),
    };
    for entry in entries {
        let entry = entry.map_err(OutputError::runtime)?;
        let file_name = entry.file_name();
        if file_name == "session.claim.json" || is_owner_claim_candidate_temporary_file_name(&file_name) {
            continue;
        }
        if matches!(file_name.to_str(), Some("owner-transitions" | "owner-staging"))
            && entry.file_type().map_err(OutputError::runtime)?.is_dir()
        {
            continue;
        }
        if matches!(file_name.to_str(), Some("outcomes" | "successors" | "terminal-finalizations"))
            && entry.file_type().map_err(OutputError::runtime)?.is_dir()
            && !directory_exists_and_is_non_empty(&entry.path())?
        {
            continue;
        }
        if file_name.to_str().is_some_and(is_immutable_record_candidate_temporary_file_name) {
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

fn is_immutable_record_candidate_temporary_file_name(file_name: &str) -> bool {
    let Some(identifier) = file_name
        .strip_prefix('.')
        .and_then(|name| name.rsplit_once(".attempt-"))
        .and_then(|(_, suffix)| suffix.strip_suffix(".tmp"))
    else {
        return false;
    };
    identifier.len() == 32 && identifier.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn is_owner_claim_candidate_temporary_file_name(file_name: &std::ffi::OsStr) -> bool {
    let Some(identifier) = file_name
        .to_str()
        .and_then(|name| name.strip_prefix(".session.claim.json."))
        .and_then(|name| name.strip_suffix(".tmp"))
        .and_then(|name| name.strip_prefix("attempt-"))
    else {
        return false;
    };
    identifier.len() == 32 && identifier.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn finish_thread_count(output_plan: &g_plan::OutputPlan, writer_count: usize) -> OutputResult<usize> {
    let requested = usize::try_from(output_plan.writer_thread_count).map_err(OutputError::runtime)?;
    if requested == 0 && writer_count != 0 {
        return Err(OutputError::InvalidInput("Writer finish thread count must be positive.".to_string()));
    }
    Ok(requested.min(writer_count))
}

fn combine_terminal_cleanup_result<ValueType>(
    primary_result: OutputResult<ValueType>,
    release_result: OutputResult<()>,
    context: &str,
) -> OutputResult<ValueType> {
    match (primary_result, release_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(release_error)) => Err(release_error),
        (Err(error), Err(release_error)) => Err(OutputError::OutputOperationAndOwnerClaimRelease {
            primary: Box::new(OutputError::Runtime(format!("{context}: {error}"))),
            release: Box::new(release_error),
        }),
    }
}

fn return_primary_after_recovery<ValueType>(
    primary_error: OutputError,
    recovery_result: OutputResult<()>,
    context: &str,
) -> OutputResult<ValueType> {
    match recovery_result {
        Ok(()) => Err(primary_error),
        Err(recovery_error) => Err(OutputError::OutputOperationAndOwnerClaimRelease {
            primary: Box::new(OutputError::Runtime(format!("{context}: {primary_error}"))),
            release: Box::new(recovery_error),
        }),
    }
}

#[cfg(test)]
std::thread_local! {
    static COMPLETION_FAILURE_POINT: std::cell::RefCell<Option<String>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
pub(crate) struct CompletionFailureGuard;

#[cfg(test)]
impl Drop for CompletionFailureGuard {
    fn drop(&mut self) {
        COMPLETION_FAILURE_POINT.with(|failure_point| {
            *failure_point.borrow_mut() = None;
        });
    }
}

#[cfg(test)]
pub(crate) fn install_completion_failure_for_test(failure_point: &str) -> CompletionFailureGuard {
    COMPLETION_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        assert!(installed_failure_point.is_none(), "a completion failure point is already installed");
        *installed_failure_point = Some(failure_point.to_string());
    });
    CompletionFailureGuard
}

#[cfg(test)]
fn fail_completion_at_test_point(observed_failure_point: &str) -> OutputResult<()> {
    COMPLETION_FAILURE_POINT.with(|installed_failure_point| {
        let mut installed_failure_point = installed_failure_point.borrow_mut();
        if installed_failure_point.as_deref() == Some(observed_failure_point) {
            *installed_failure_point = None;
            Err(OutputError::Runtime(format!("Injected output completion failure at '{observed_failure_point}'.")))
        } else {
            Ok(())
        }
    })
}
