//! Prepared native association-run ownership.

use std::collections::BTreeSet;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use g_genotype::{BgenContentSelector, BgenOpenRequest, Packed8Compatibility};
use g_input::{AlignedPhenotypeGroup, PhenotypeGroupLoadRequest};
use g_output::{
    Active, Claimed, CompletedOutputRun, ManifestFileFingerprintCache, OutputDeliveryToken, OutputManager, Planned,
};
use g_plan::{GpuGenotypeFormat, RunPlan};

use crate::backend::AssociationBackend;
use crate::delivery::{AssociationDeliveryRequest, AssociationDeliverySettings, PreparedGenotypeInput};
use crate::delivery_execution::{AssociationDeliveryReport, DeliveryError, run_association_delivery};
use crate::output_manifest::build_prediction_loco_file_fingerprints_with_cache;
use crate::preflight::{PreflightError, validate_jax_index_capacity, validate_multi_trait_preflight_values};
use crate::preparation::{
    PipelineOutputPreparationError, RuntimeOutputGroupInput, RuntimeOutputPlan, build_runtime_output_initializations,
};
use crate::progress::RunProgressReporter;

/// Failure while converting a run plan into a fully prepared native run.
#[derive(Debug, thiserror::Error)]
pub(crate) enum RunPreparationError {
    #[error(transparent)]
    Bgen(#[from] g_genotype::BgenError),
    #[error(transparent)]
    Genotype(#[from] g_genotype::GenotypeError),
    #[error(transparent)]
    Input(#[from] g_input::InputError),
    #[error(transparent)]
    Output(#[from] g_output::OutputError),
    #[error("Output delivery-token construction failed: {token}; active output abort also failed: {abort}")]
    OutputDeliveryTokenAbort {
        #[source]
        token: g_output::OutputError,
        abort: Box<g_output::OutputError>,
    },
    #[error(transparent)]
    OutputPreparation(#[from] PipelineOutputPreparationError),
    #[error(transparent)]
    Preflight(#[from] PreflightError),
    #[error("The run plan contains no phenotype outputs.")]
    EmptyPhenotypePlan,
    #[error("Aligned input produced no phenotype groups.")]
    EmptyPhenotypeGroups,
    #[error("{field_name} must be positive.")]
    NonPositiveCapacity { field_name: &'static str },
    #[error("{field_name} overflowed the native usize representation.")]
    CapacityOverflow { field_name: &'static str },
    #[error("{field_name} exceeds the JAX int32 domain.")]
    JaxIntegerOverflow { field_name: &'static str },
    #[error("The resumed output requires packed8 delivery, but the current BGEN requires dosage delivery.")]
    ResumedPacked8Incompatible,
    #[error(
        "Configured BGEN content SHA-256 {configured_content_sha256} does not match existing output authority {persisted_content_sha256}."
    )]
    BgenContentSelectorMismatch {
        configured_content_sha256: g_genotype_contracts::BgenContentSha256,
        persisted_content_sha256: g_genotype_contracts::BgenContentSha256,
    },
}

/// Failure while converting a claimed run into active output authority.
#[derive(Debug, thiserror::Error)]
pub(crate) enum RunActivationError {
    #[error("Output activation failed before attempt publication: {source}")]
    Unpublished {
        #[source]
        source: g_output::OutputError,
        rollback: Box<g_output::OutputClaimRollback>,
    },
    #[error(transparent)]
    Published(#[from] RunPreparationError),
}

/// Why an engine-owned post-session cleanup obligation exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnginePostSessionCleanupPurpose {
    /// Output activation never published attempt authority.
    PreActivationRollback,
    /// A completed read-only attempt produced removable claim diagnostics.
    CompletedNoop,
}

/// Opaque output authority retained until claim-scoped diagnostics close.
#[must_use = "post-session output cleanup authority must be handled after diagnostics close"]
pub struct EnginePostSessionCleanup {
    authority: EnginePostSessionCleanupAuthority,
}

enum EnginePostSessionCleanupAuthority {
    PreActivationRollback(Box<g_output::OutputClaimRollback>),
    CompletedNoop(Box<g_output::OutputPostSessionCleanup>),
    #[cfg(test)]
    RetrySentinel {
        failed_once: bool,
    },
}

/// Failure while applying engine-owned post-session cleanup.
#[derive(Debug)]
pub struct EnginePostSessionCleanupError {
    message: String,
    source: Option<g_output::OutputError>,
}

impl std::fmt::Debug for EnginePostSessionCleanup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("EnginePostSessionCleanup").field("purpose", &self.purpose()).finish_non_exhaustive()
    }
}

impl std::fmt::Display for EnginePostSessionCleanupError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for EnginePostSessionCleanupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|source| source as &(dyn std::error::Error + 'static))
    }
}

impl EnginePostSessionCleanup {
    pub(crate) fn pre_activation_rollback(rollback: Box<g_output::OutputClaimRollback>) -> Self {
        Self { authority: EnginePostSessionCleanupAuthority::PreActivationRollback(rollback) }
    }

    pub(crate) fn completed_noop(cleanup: g_output::OutputPostSessionCleanup) -> Self {
        Self { authority: EnginePostSessionCleanupAuthority::CompletedNoop(Box::new(cleanup)) }
    }

    /// Return the cleanup obligation category without exposing its authority.
    #[must_use]
    pub const fn purpose(&self) -> EnginePostSessionCleanupPurpose {
        match &self.authority {
            EnginePostSessionCleanupAuthority::PreActivationRollback(_) => {
                EnginePostSessionCleanupPurpose::PreActivationRollback
            }
            EnginePostSessionCleanupAuthority::CompletedNoop(_) => EnginePostSessionCleanupPurpose::CompletedNoop,
            #[cfg(test)]
            EnginePostSessionCleanupAuthority::RetrySentinel { .. } => EnginePostSessionCleanupPurpose::CompletedNoop,
        }
    }

    /// Apply post-session cleanup.
    ///
    /// Both completed-noop cleanup and pre-activation rollback remain retryable
    /// after a failed attempt and idempotent after success.
    ///
    /// # Errors
    ///
    /// Returns an engine-owned error when durable output cleanup fails.
    pub fn cleanup(&mut self) -> Result<(), EnginePostSessionCleanupError> {
        match &mut self.authority {
            EnginePostSessionCleanupAuthority::PreActivationRollback(rollback) => {
                rollback.abort_before_activation().map_err(EnginePostSessionCleanupError::from_output)
            }
            EnginePostSessionCleanupAuthority::CompletedNoop(cleanup) => {
                cleanup.cleanup().map_err(EnginePostSessionCleanupError::from_output)
            }
            #[cfg(test)]
            EnginePostSessionCleanupAuthority::RetrySentinel { failed_once } => {
                if *failed_once {
                    Ok(())
                } else {
                    *failed_once = true;
                    Err(EnginePostSessionCleanupError::message("injected first cleanup failure"))
                }
            }
        }
    }

    #[cfg(test)]
    fn retry_sentinel() -> Self {
        Self { authority: EnginePostSessionCleanupAuthority::RetrySentinel { failed_once: false } }
    }
}

impl EnginePostSessionCleanupError {
    fn from_output(source: g_output::OutputError) -> Self {
        Self { message: source.to_string(), source: Some(source) }
    }

    #[cfg(test)]
    fn message(message: &str) -> Self {
        Self { message: message.to_string(), source: None }
    }
}

/// Binding-owned hooks used at the native run boundary.
pub trait RunHooks {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Check whether execution must stop.
    ///
    /// # Errors
    ///
    /// Returns the binding-specific interruption error.
    fn check_interruption(&mut self) -> Result<(), Self::Error>;

    /// Return the process signal name associated with an interruption error.
    fn interruption_signal_name(error: &Self::Error) -> Option<&str>;
}

/// Completed native execution and its output artifacts.
pub(crate) struct RunExecution {
    pub completed_outputs: Vec<CompletedOutputRun>,
    pub delivery_reports: Vec<AssociationDeliveryReport>,
}

/// One fixed activation result plus optional completed-noop cleanup authority.
#[must_use = "the activation result and post-session cleanup must both be handled"]
pub(crate) struct RunActivationOutcome {
    pub result: Result<PreparedRun, RunActivationError>,
    pub post_session_cleanup: Option<g_output::OutputPostSessionCleanup>,
}

/// One fixed execution result plus optional completed-noop cleanup authority.
#[must_use = "the execution result and post-session cleanup must both be handled"]
pub(crate) struct RunExecutionOutcome<BackendError, HookError, PostSessionCleanup = g_output::OutputPostSessionCleanup>
{
    pub result: Result<RunExecution, RunExecutionError<BackendError, HookError>>,
    pub post_session_cleanup: Option<PostSessionCleanup>,
}

struct RunPreparationFailureOutcome<PostSessionCleanup = g_output::OutputPostSessionCleanup> {
    error: RunPreparationError,
    post_session_cleanup: Option<PostSessionCleanup>,
}

/// Failure while executing a fully prepared run.
#[derive(Debug, thiserror::Error)]
pub(crate) enum RunExecutionError<BackendError, HookError> {
    #[error("Association delivery failed: {0}")]
    Delivery(#[source] DeliveryError<BackendError, HookError>),
    #[error("Association delivery was interrupted.")]
    Interrupted(#[source] HookError),
    #[error("Association delivery was interrupted and queued output could not be flushed.")]
    InterruptedOutputFlush {
        interruption: HookError,
        #[source]
        output: g_output::OutputError,
    },
    #[error("Association delivery failed: {delivery}; output abort also failed: {output}")]
    DeliveryAbort {
        delivery: Box<DeliveryError<BackendError, HookError>>,
        #[source]
        output: g_output::OutputError,
    },
    #[error("Output completion failed: {0}")]
    OutputFinish(#[source] g_output::OutputError),
}

/// Unprepared native run owning its canonical plan and output lifecycle.
pub(crate) struct RunEngine {
    run_plan: Arc<RunPlan>,
    output_manager: OutputManager<Planned>,
}

impl RunEngine {
    /// Open the output lifecycle for a canonical run plan.
    ///
    /// # Errors
    ///
    /// Returns an error when planned output paths cannot be opened.
    pub(crate) fn open(run_plan: RunPlan, effective_config_toml: String) -> Result<Self, RunPreparationError> {
        if run_plan.phenotype_runs.is_empty() {
            return Err(RunPreparationError::EmptyPhenotypePlan);
        }
        validate_jax_integer_domain(&run_plan)?;
        let run_plan = Arc::new(run_plan);
        let output_manager = OutputManager::open(Arc::clone(&run_plan), effective_config_toml)?;
        Ok(Self { run_plan, output_manager })
    }

    /// Resolve genotype policy and prepare all native input and output state.
    ///
    /// # Errors
    ///
    /// Returns a typed error when BGEN preparation, input alignment, preflight,
    /// manifest construction, resume validation, or output initialization fails.
    pub(crate) fn prepare(self) -> Result<ClaimedRun, RunPreparationError> {
        let Self { run_plan, output_manager } = self;
        let existing_output_resume_agreement = output_manager.existing_output_resume_agreement()?;
        let bgen_open_request = build_bgen_open_request(
            Path::new(&run_plan.input.bgen_path),
            run_plan.input.bgen_content_sha256,
            existing_output_resume_agreement.as_ref(),
        )?;
        let chunk_size = positive_u32_as_usize(run_plan.chunk_size, "analysis chunk size")?;
        let bgen_reader = g_genotype::BgenReaderCore::open_request(bgen_open_request)?;
        let genotype_input = PreparedGenotypeInput { reader: bgen_reader, chunk_size };
        if genotype_input.reader.variant_count() == 0 {
            return Err(PreflightError::EmptyBgenInput.into());
        }
        let bgen_content_evidence = Arc::new(genotype_input.reader.content_evidence().clone());
        let planned_chunk_ranges = genotype_input
            .reader
            .plan_chromosome_homogeneous_chunks(genotype_input.chunk_size, &BTreeSet::new())?
            .into_iter()
            .map(|chunk_spec| chunk_spec.variant_start_index..chunk_spec.variant_stop_index)
            .collect::<Vec<Range<usize>>>();
        let resolved_gpu_genotype_format = resolve_gpu_genotype_format(
            run_plan.compute.device,
            existing_output_resume_agreement.as_ref(),
            &genotype_input,
        )?;
        let phenotype_names = run_plan
            .phenotype_runs
            .iter()
            .map(|phenotype_run| phenotype_run.phenotype_name.clone())
            .collect::<Vec<_>>();
        let prediction_loco_paths =
            g_input::resolve_prediction_loco_paths(Path::new(&run_plan.input.prediction_list_path), &phenotype_names)?;
        let groups = load_groups(&run_plan, &genotype_input, &phenotype_names, &prediction_loco_paths)?;
        validate_groups(&run_plan, &groups)?;
        let ClaimedOutputGroups { output_manager } = prepare_output_claim(
            &run_plan,
            &groups,
            &prediction_loco_paths,
            output_manager,
            resolved_gpu_genotype_format,
            bgen_content_evidence,
            genotype_input.reader.variant_count(),
            &planned_chunk_ranges,
        )?;
        Ok(ClaimedRun { run_plan, resolved_gpu_genotype_format, genotype_input, groups, output_manager })
    }
}

struct PreparedAssociationGroup {
    group: AlignedPhenotypeGroup,
    output: OutputDeliveryToken,
}

struct ClaimedOutputGroups {
    output_manager: OutputManager<Claimed>,
}

/// Read-only prepared inputs with exclusive output ownership but no attempt authority.
pub(crate) struct ClaimedRun {
    run_plan: Arc<RunPlan>,
    resolved_gpu_genotype_format: GpuGenotypeFormat,
    genotype_input: PreparedGenotypeInput,
    groups: Vec<AlignedPhenotypeGroup>,
    output_manager: OutputManager<Claimed>,
}

/// Fully prepared run awaiting one association backend.
pub(crate) struct PreparedRun {
    run_plan: Arc<RunPlan>,
    resolved_gpu_genotype_format: GpuGenotypeFormat,
    genotype_input: PreparedGenotypeInput,
    groups: Vec<PreparedAssociationGroup>,
    output_manager: OutputManager<Active>,
}

impl ClaimedRun {
    /// Return the canonical run plan bound to this claim.
    #[must_use]
    pub(crate) fn run_plan(&self) -> &RunPlan {
        &self.run_plan
    }

    /// Return the ownership-private diagnostics path.
    ///
    /// # Errors
    ///
    /// Returns an error if claim staging is inconsistent.
    pub(crate) fn diagnostics_directory(&self) -> Result<&Path, RunPreparationError> {
        self.output_manager.diagnostics_directory().map_err(Into::into)
    }

    /// Remove unpublished diagnostics and release output ownership.
    ///
    /// # Errors
    ///
    /// Returns an error when staging cleanup or claim release fails.
    pub(crate) fn abort_before_activation(self) -> Result<(), g_output::OutputError> {
        self.output_manager.abort_before_activation()
    }

    /// Reject backend compatibility before publishing attempt authority.
    pub(crate) fn reject_activation(self, source: g_output::OutputError) -> RunActivationOutcome {
        let error = self.output_manager.reject_activation(source);
        RunActivationOutcome { result: Err(run_activation_error(error)), post_session_cleanup: None }
    }

    /// Publish attempt authority after backend construction.
    ///
    /// # Errors
    ///
    /// The result records any output activation or delivery-token construction
    /// failure. Completed-noop cleanup remains orthogonal to that result.
    pub(crate) fn activate(
        self,
        association_implementation: g_output::AssociationImplementationCompatibility,
    ) -> RunActivationOutcome {
        let Self { run_plan, resolved_gpu_genotype_format, genotype_input, groups, output_manager } = self;
        let output_manager = match output_manager
            .activate_with_deferred_completed_noop_cleanup(association_implementation)
        {
            Ok(output_manager) => output_manager,
            Err(error) => {
                return RunActivationOutcome { result: Err(run_activation_error(error)), post_session_cleanup: None };
            }
        };
        let prepared_groups_result = groups
            .into_iter()
            .map(|group| {
                let output = output_manager.delivery_token_for_phenotypes(&group.phenotype_group.phenotype_names)?;
                Ok(PreparedAssociationGroup { group, output })
            })
            .collect::<Result<Vec<_>, g_output::OutputError>>();
        let prepared_groups = match prepared_groups_result {
            Ok(prepared_groups) => prepared_groups,
            Err(token_error) => {
                let outcome = abort_active_output_after_delivery_token_error(token_error, |failure_reason| {
                    output_manager.abort(failure_reason)
                });
                return RunActivationOutcome {
                    result: Err(RunActivationError::Published(outcome.error)),
                    post_session_cleanup: outcome.post_session_cleanup,
                };
            }
        };
        RunActivationOutcome {
            result: Ok(PreparedRun {
                run_plan,
                resolved_gpu_genotype_format,
                genotype_input,
                groups: prepared_groups,
                output_manager,
            }),
            post_session_cleanup: None,
        }
    }
}

fn run_activation_error(error: g_output::OutputActivationError) -> RunActivationError {
    let g_output::OutputActivationFailureParts { source, rollback } = error.into_parts();
    match rollback {
        Some(rollback) => RunActivationError::Unpublished { source, rollback: Box::new(rollback) },
        None => RunActivationError::Published(RunPreparationError::Output(source)),
    }
}

fn abort_active_output_after_delivery_token_error<AbortOutput>(
    token_error: g_output::OutputError,
    abort_output: AbortOutput,
) -> RunPreparationFailureOutcome
where
    AbortOutput: FnOnce(&str) -> Result<(), g_output::OutputTerminalError>,
{
    let failure_reason = format!("output delivery-token construction failed: {token_error}");
    match abort_output(&failure_reason) {
        Ok(()) => {
            RunPreparationFailureOutcome { error: RunPreparationError::Output(token_error), post_session_cleanup: None }
        }
        Err(abort) => {
            let g_output::OutputTerminalFailureParts { source, post_session_cleanup } = abort.into_parts();
            delivery_token_abort_failure_outcome(token_error, source, post_session_cleanup)
        }
    }
}

fn delivery_token_abort_failure_outcome<PostSessionCleanup>(
    token_error: g_output::OutputError,
    abort_error: g_output::OutputError,
    post_session_cleanup: Option<PostSessionCleanup>,
) -> RunPreparationFailureOutcome<PostSessionCleanup> {
    if post_session_cleanup.is_some() {
        RunPreparationFailureOutcome { error: RunPreparationError::Output(token_error), post_session_cleanup }
    } else {
        RunPreparationFailureOutcome {
            error: RunPreparationError::OutputDeliveryTokenAbort { token: token_error, abort: Box::new(abort_error) },
            post_session_cleanup,
        }
    }
}

impl PreparedRun {
    /// Return the resolved association backend contract.
    #[must_use]
    pub(crate) const fn resolved_gpu_genotype_format(&self) -> GpuGenotypeFormat {
        self.resolved_gpu_genotype_format
    }

    /// Execute every prepared group with optional throttled progress reporting.
    ///
    /// # Errors
    ///
    /// The result records any typed backend, hook, delivery, or output
    /// completion error. Completed-noop cleanup remains orthogonal to that
    /// result.
    pub(crate) fn execute_with_progress<Backend, Hooks>(
        self,
        backend: Arc<Backend>,
        hooks: &mut Hooks,
        progress_reporter: Option<&Arc<RunProgressReporter>>,
    ) -> RunExecutionOutcome<Backend::Error, Hooks::Error>
    where
        Backend: AssociationBackend + 'static,
        Hooks: RunHooks,
    {
        let PreparedRun { run_plan, resolved_gpu_genotype_format, genotype_input, groups, output_manager } = self;
        let statistics_policy = match run_plan.association_mode {
            g_plan::AssociationMode::Regenie2Linear => g_genotype::ChunkStatisticsPolicy {
                retain_imputed_dosage_square_sum: true,
                collect_sparse_candidate_mask: false,
            },
            g_plan::AssociationMode::Regenie2Binary => g_genotype::ChunkStatisticsPolicy {
                retain_imputed_dosage_square_sum: false,
                collect_sparse_candidate_mask: run_plan.correction.method
                    == g_plan::BinaryFallbackMethod::FirthApproximate,
            },
        };
        let delivery_result: Result<Vec<AssociationDeliveryReport>, DeliveryError<Backend::Error, Hooks::Error>> =
            (|| {
                let progress_context = progress_reporter.and_then(|reporter| {
                    reporter
                        .totals_from_chunk_plan(|| {
                            genotype_input
                                .reader
                                .plan_chromosome_homogeneous_chunks(genotype_input.chunk_size, &BTreeSet::new())
                        })
                        .map(|totals| (reporter, totals))
                });
                let mut reports = Vec::with_capacity(groups.len());
                for prepared_group in groups {
                    let progress = progress_context.map(|(reporter, totals)| {
                        reporter
                            .register_delivery(prepared_group.group.phenotype_group.phenotype_names.join(","), totals)
                    });
                    let request = AssociationDeliveryRequest {
                        group: prepared_group.group,
                        settings: AssociationDeliverySettings {
                            output: prepared_group.output,
                            null_logistic_nonconvergence_policy: run_plan
                                .compute
                                .kernels
                                .binary_null
                                .nonconvergence_policy,
                            progress,
                            gpu_genotype_format: resolved_gpu_genotype_format,
                            statistics_policy,
                        },
                    };
                    reports.push(run_association_delivery(&genotype_input, &backend, request, || {
                        hooks.check_interruption()
                    })?);
                }
                Ok(reports)
            })();
        drop(backend);
        finish_execution::<Backend::Error, Hooks>(delivery_result, output_manager)
    }
}

fn positive_u32_as_usize(value: u32, field_name: &'static str) -> Result<usize, RunPreparationError> {
    if value == 0 {
        return Err(RunPreparationError::NonPositiveCapacity { field_name });
    }
    usize::try_from(value).map_err(|_| RunPreparationError::CapacityOverflow { field_name })
}

/// Validate every host-plan integer consumed by JAX.
///
/// # Errors
///
/// Returns an error when a required count is zero or exceeds the signed int32
/// domain used by JAX indices, reductions, and loop state.
pub(crate) fn validate_jax_integer_domain(run_plan: &RunPlan) -> Result<(), RunPreparationError> {
    let kernels = &run_plan.compute.kernels;
    let positive_values = [
        ("analysis chunk size", run_plan.chunk_size),
        ("binary null maximum iterations", kernels.binary_null.maximum_iterations),
        ("Firth batch size", kernels.firth.batch_size),
        ("Firth candidate capacity", kernels.firth.candidate_capacity),
        ("Firth maximum iterations", kernels.firth.maximum_iterations),
        ("Firth pseudo maximum iterations", kernels.firth.pseudo_maximum_iterations),
        ("Firth pseudo inner maximum iterations", kernels.firth.pseudo_inner_maximum_iterations),
        ("Firth line-search maximum attempts", kernels.firth.line_search_maximum_attempts),
        ("null Firth maximum iterations", kernels.null_firth.maximum_iterations),
        ("null Firth fallback iteration multiplier", kernels.null_firth.fallback_iteration_multiplier),
        ("null Firth line-search maximum attempts", kernels.null_firth.line_search_maximum_attempts),
    ];
    for (field_name, value) in positive_values {
        if value == 0 {
            return Err(RunPreparationError::NonPositiveCapacity { field_name });
        }
        i32::try_from(value).map_err(|_| RunPreparationError::JaxIntegerOverflow { field_name })?;
    }
    let fallback_iteration_limit = u64::from(kernels.null_firth.maximum_iterations)
        .checked_mul(u64::from(kernels.null_firth.fallback_iteration_multiplier))
        .ok_or(RunPreparationError::JaxIntegerOverflow { field_name: "null Firth fallback iteration limit" })?;
    let maximum_jax_integer = u64::from(i32::MAX.unsigned_abs());
    if fallback_iteration_limit > maximum_jax_integer {
        return Err(RunPreparationError::JaxIntegerOverflow { field_name: "null Firth fallback iteration limit" });
    }
    Ok(())
}

fn resolve_gpu_genotype_format(
    device: g_plan::Device,
    existing_output_resume_agreement: Option<&g_output::ExistingOutputResumeAgreement>,
    genotype_input: &PreparedGenotypeInput,
) -> Result<GpuGenotypeFormat, RunPreparationError> {
    if let Some(existing_output_resume_agreement) = existing_output_resume_agreement {
        let manifest_format = existing_output_resume_agreement.gpu_genotype_format;
        if manifest_format == GpuGenotypeFormat::Packed8
            && genotype_input.reader.packed8_compatibility_with_cache()? == Packed8Compatibility::RequiresDosage
        {
            return Err(RunPreparationError::ResumedPacked8Incompatible);
        }
        return Ok(manifest_format);
    }
    if device != g_plan::Device::Gpu {
        return Ok(GpuGenotypeFormat::Dosage);
    }
    match genotype_input.reader.packed8_compatibility_with_cache()? {
        Packed8Compatibility::Compatible => Ok(GpuGenotypeFormat::Packed8),
        Packed8Compatibility::RequiresDosage => Ok(GpuGenotypeFormat::Dosage),
    }
}

fn build_bgen_open_request<'path>(
    locator: &'path Path,
    configured_content_sha256: Option<g_genotype_contracts::BgenContentSha256>,
    existing_output_resume_agreement: Option<&g_output::ExistingOutputResumeAgreement>,
) -> Result<BgenOpenRequest<'path>, RunPreparationError> {
    let persisted_content_fingerprint =
        existing_output_resume_agreement.map(|agreement| agreement.bgen_content_fingerprint);
    let content_selector = match (configured_content_sha256, persisted_content_fingerprint) {
        (Some(configured_content_sha256), Some(persisted_content_fingerprint))
            if configured_content_sha256 != persisted_content_fingerprint.content_sha256 =>
        {
            return Err(RunPreparationError::BgenContentSelectorMismatch {
                configured_content_sha256,
                persisted_content_sha256: persisted_content_fingerprint.content_sha256,
            });
        }
        (Some(content_sha256), Some(persisted_content_fingerprint)) => Some(BgenContentSelector {
            content_sha256,
            expected_byte_count: Some(persisted_content_fingerprint.byte_count),
        }),
        (None, Some(persisted_content_fingerprint)) => Some(BgenContentSelector {
            content_sha256: persisted_content_fingerprint.content_sha256,
            expected_byte_count: Some(persisted_content_fingerprint.byte_count),
        }),
        (Some(content_sha256), None) => Some(BgenContentSelector { content_sha256, expected_byte_count: None }),
        (None, None) => None,
    };
    Ok(BgenOpenRequest { locator, content_selector })
}

fn load_groups(
    run_plan: &RunPlan,
    genotype_input: &PreparedGenotypeInput,
    phenotype_names: &[String],
    prediction_loco_paths: &[g_input::PredictionLocoPath],
) -> Result<Vec<AlignedPhenotypeGroup>, RunPreparationError> {
    let sample_identifiers = g_input::load_sample_identifier_data_from_sample_file(
        Path::new(&run_plan.input.sample_path),
        genotype_input.reader.sample_count(),
    )?;
    let groups = g_input::load_aligned_phenotype_groups(&PhenotypeGroupLoadRequest {
        sample_identifiers: &sample_identifiers,
        phenotype_path: &run_plan.input.phenotype_path,
        prediction_loco_paths,
        phenotype_names,
        covariate_path: run_plan.input.covariate_path.as_deref(),
        covariate_names: Some(&run_plan.input.covariate_names),
        is_binary_trait: run_plan.association_mode == g_plan::AssociationMode::Regenie2Binary,
        sample_mode: run_plan.compute.multi_phenotype_sample_mode,
    })?;
    if groups.is_empty() {
        return Err(RunPreparationError::EmptyPhenotypeGroups);
    }
    Ok(groups)
}

fn validate_groups(run_plan: &RunPlan, groups: &[AlignedPhenotypeGroup]) -> Result<(), RunPreparationError> {
    let is_binary_trait = run_plan.association_mode == g_plan::AssociationMode::Regenie2Binary;
    let chunk_size = positive_u32_as_usize(run_plan.chunk_size, "analysis chunk size")?;
    let firth_candidate_capacity =
        positive_u32_as_usize(run_plan.compute.kernels.firth.candidate_capacity, "Firth candidate capacity")?;
    let firth_batch_size = positive_u32_as_usize(run_plan.compute.kernels.firth.batch_size, "Firth batch size")?;
    for group in groups {
        let trait_count = group.phenotype_group.phenotype_names.len();
        let sample_count = group.sample_indices.len();
        validate_jax_index_capacity(
            trait_count,
            sample_count,
            chunk_size,
            firth_candidate_capacity,
            firth_batch_size,
            is_binary_trait,
        )?;
        validate_multi_trait_preflight_values(
            trait_count,
            sample_count,
            &group.phenotype_values,
            sample_count,
            group.covariate_names.len(),
            &group.covariate_values,
            is_binary_trait,
        )?;
    }
    Ok(())
}

fn prepare_output_claim(
    run_plan: &RunPlan,
    groups: &[AlignedPhenotypeGroup],
    prediction_loco_paths: &[g_input::PredictionLocoPath],
    output_manager: OutputManager<Planned>,
    resolved_gpu_genotype_format: GpuGenotypeFormat,
    bgen_content_evidence: Arc<g_genotype_contracts::BgenContentEvidence>,
    variant_count: usize,
    planned_chunk_ranges: &[Range<usize>],
) -> Result<ClaimedOutputGroups, RunPreparationError> {
    let runtime_output_plan = RuntimeOutputPlan { variant_count, resolved_gpu_genotype_format, bgen_content_evidence };
    let mut fingerprint_cache = ManifestFileFingerprintCache::default();
    let all_prediction_loco_files: Arc<[g_output::PredictionLocoFileFingerprint]> =
        build_prediction_loco_file_fingerprints_with_cache(prediction_loco_paths, &mut fingerprint_cache)?.into();
    let mut run_initializations = Vec::with_capacity(run_plan.phenotype_runs.len());
    for group in groups {
        run_initializations.extend(build_runtime_output_initializations(
            &RuntimeOutputGroupInput {
                phenotype_group: &group.phenotype_group,
                covariate_names: &group.covariate_names,
                sample_count: group.sample_indices.len(),
            },
            &runtime_output_plan,
            &all_prediction_loco_files,
        )?);
    }
    let collect_stage_timings = matches!(run_plan.telemetry, g_plan::TelemetryMode::Profile);
    let output_manager = output_manager.claim(run_initializations, planned_chunk_ranges, collect_stage_timings)?;
    Ok(ClaimedOutputGroups { output_manager })
}

fn finish_execution<BackendError, Hooks>(
    delivery_result: Result<Vec<AssociationDeliveryReport>, DeliveryError<BackendError, Hooks::Error>>,
    output_manager: OutputManager<Active>,
) -> RunExecutionOutcome<BackendError, Hooks::Error>
where
    BackendError: std::error::Error,
    Hooks: RunHooks,
{
    match delivery_result {
        Ok(delivery_reports) => match output_manager.close_completed().and_then(OutputManager::finish) {
            Ok(completion) => RunExecutionOutcome {
                result: Ok(RunExecution { completed_outputs: completion.completed_outputs, delivery_reports }),
                post_session_cleanup: completion.post_session_cleanup,
            },
            Err(output) => {
                let g_output::OutputTerminalFailureParts { source, post_session_cleanup } = output.into_parts();
                RunExecutionOutcome { result: Err(RunExecutionError::OutputFinish(source)), post_session_cleanup }
            }
        },
        Err(DeliveryError::Interrupted(interruption)) => {
            let signal_name = Hooks::interruption_signal_name(&interruption).unwrap_or("unknown");
            match output_manager.finish_interrupted(signal_name) {
                Ok(()) => RunExecutionOutcome {
                    result: Err(RunExecutionError::Interrupted(interruption)),
                    post_session_cleanup: None,
                },
                Err(output) => {
                    let g_output::OutputTerminalFailureParts { source, post_session_cleanup } = output.into_parts();
                    interrupted_output_failure_outcome(interruption, source, post_session_cleanup)
                }
            }
        }
        Err(delivery) => {
            let failure_reason = format!("association delivery failed: {delivery}");
            match output_manager.abort(&failure_reason) {
                Ok(()) => RunExecutionOutcome {
                    result: Err(RunExecutionError::Delivery(delivery)),
                    post_session_cleanup: None,
                },
                Err(output) => {
                    let g_output::OutputTerminalFailureParts { source, post_session_cleanup } = output.into_parts();
                    delivery_abort_failure_outcome(delivery, source, post_session_cleanup)
                }
            }
        }
    }
}

fn interrupted_output_failure_outcome<BackendError, HookError, PostSessionCleanup>(
    interruption: HookError,
    output: g_output::OutputError,
    post_session_cleanup: Option<PostSessionCleanup>,
) -> RunExecutionOutcome<BackendError, HookError, PostSessionCleanup> {
    let result = if post_session_cleanup.is_some() {
        Err(RunExecutionError::Interrupted(interruption))
    } else {
        Err(RunExecutionError::InterruptedOutputFlush { interruption, output })
    };
    RunExecutionOutcome { result, post_session_cleanup }
}

fn delivery_abort_failure_outcome<BackendError, HookError, PostSessionCleanup>(
    delivery: DeliveryError<BackendError, HookError>,
    output: g_output::OutputError,
    post_session_cleanup: Option<PostSessionCleanup>,
) -> RunExecutionOutcome<BackendError, HookError, PostSessionCleanup> {
    let result = if post_session_cleanup.is_some() {
        Err(RunExecutionError::Delivery(delivery))
    } else {
        Err(RunExecutionError::DeliveryAbort { delivery: Box::new(delivery), output })
    };
    RunExecutionOutcome { result, post_session_cleanup }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use g_genotype_contracts::{
        BgenContentEvidence, BgenContentFingerprint, BgenContentSha256, BgenSnapshotResolution, BgenSourceIdentity,
    };

    use super::*;

    struct BgenIntegrationTestDirectory {
        path: PathBuf,
    }

    #[derive(Debug, Eq, PartialEq)]
    struct OutputTreeFile {
        relative_path: PathBuf,
        contents: Vec<u8>,
    }

    impl BgenIntegrationTestDirectory {
        fn new(label: &str) -> Self {
            static DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);
            let sequence = DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("test time should follow the Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("g-engine-bgen-integration-{label}-{}-{timestamp}-{sequence}", std::process::id()));
            std::fs::create_dir_all(&path).expect("BGEN integration test directory should be created");
            Self { path }
        }

        fn write(&self, file_name: &str, contents: &[u8]) -> PathBuf {
            let path = self.path.join(file_name);
            std::fs::write(&path, contents).expect("BGEN integration test file should be written");
            path
        }
    }

    impl Drop for BgenIntegrationTestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn append_bgen_string(bytes: &mut Vec<u8>, value: &[u8]) {
        let value_length = u16::try_from(value.len()).expect("test BGEN string length should fit uint16");
        bytes.extend_from_slice(&value_length.to_le_bytes());
        bytes.extend_from_slice(value);
    }

    fn single_variant_bgen_bytes(sample_ploidy_and_missingness: &[u8], probability_bytes: &[u8]) -> Vec<u8> {
        let sample_count =
            u32::try_from(sample_ploidy_and_missingness.len()).expect("test BGEN sample count should fit uint32");
        let mut probability_block = Vec::new();
        probability_block.extend_from_slice(&sample_count.to_le_bytes());
        probability_block.extend_from_slice(&2_u16.to_le_bytes());
        probability_block.push(2);
        probability_block.push(2);
        probability_block.extend_from_slice(sample_ploidy_and_missingness);
        probability_block.push(0);
        probability_block.push(8);
        probability_block.extend_from_slice(probability_bytes);

        let mut bytes = vec![0_u8; 24];
        bytes[0..4].copy_from_slice(&20_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&20_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&1_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&sample_count.to_le_bytes());
        bytes[16..20].copy_from_slice(b"bgen");
        bytes[20..24].copy_from_slice(&(2_u32 << 2).to_le_bytes());
        append_bgen_string(&mut bytes, b"variant");
        append_bgen_string(&mut bytes, b"rs-test");
        append_bgen_string(&mut bytes, b"22");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.push(b'A');
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.push(b'G');
        let probability_block_length =
            u32::try_from(probability_block.len()).expect("test BGEN probability block should fit uint32");
        bytes.extend_from_slice(&probability_block_length.to_le_bytes());
        bytes.extend_from_slice(&probability_block);
        bytes
    }

    fn owned_content_fingerprint(reader: &g_genotype::BgenReaderCore) -> BgenContentFingerprint {
        match reader.content_evidence() {
            BgenContentEvidence::OwnedSnapshot(content_fingerprint) => *content_fingerprint,
            BgenContentEvidence::PositionedUnattested(_) => {
                panic!("small BGEN integration fixture should use an owned snapshot")
            }
        }
    }

    fn bgen_resume_agreement(
        content_fingerprint: BgenContentFingerprint,
        gpu_genotype_format: GpuGenotypeFormat,
    ) -> g_output::ExistingOutputResumeAgreement {
        g_output::ExistingOutputResumeAgreement {
            bgen_content_fingerprint: content_fingerprint,
            gpu_genotype_format,
            association_implementation: bgen_test_association_implementation(),
        }
    }

    fn bgen_test_association_implementation() -> g_output::AssociationImplementationCompatibility {
        g_output::AssociationImplementationCompatibility::new(
            "0.11.0".to_string(),
            "0.11.0".to_string(),
            Some(g_output::FirthComponentsCompatibility::jax()),
        )
        .expect("test association implementation is valid")
    }

    fn bgen_test_kernel_plan() -> g_plan::KernelPlan {
        g_plan::KernelPlan {
            linear: g_plan::LinearKernelPlan {
                minimum_variance: g_plan::PositiveF32::try_from(1.0e-8).expect("valid test minimum variance"),
                relative_variance_tolerance: g_plan::PositiveF32::try_from(1.0e-6)
                    .expect("valid test relative variance tolerance"),
            },
            binary_null: g_plan::BinaryNullKernelPlan {
                maximum_iterations: 50,
                coefficient_tolerance: g_plan::PositiveF32::try_from(1.0e-6).expect("valid test coefficient tolerance"),
                nonconvergence_policy: g_plan::NullLogisticNonconvergencePolicy::Fail,
                minimum_probability: g_plan::ProbabilityFloor::try_from(1.0e-6)
                    .expect("valid test minimum probability"),
                minimum_variance: g_plan::PositiveF32::try_from(1.0e-8).expect("valid test binary minimum variance"),
                relative_variance_tolerance: g_plan::PositiveF32::try_from(1.0e-6)
                    .expect("valid test binary relative variance tolerance"),
            },
            firth: g_plan::FirthKernelPlan {
                batch_size: 2,
                candidate_capacity: 4,
                maximum_iterations: 25,
                gradient_tolerance: g_plan::PositiveF64::try_from(1.0e-5).expect("valid test Firth gradient tolerance"),
                maximum_step_size: g_plan::PositiveF64::try_from(5.0).expect("valid test Firth maximum step"),
                pseudo_maximum_iterations: 10,
                pseudo_inner_maximum_iterations: 5,
                line_search_maximum_attempts: 5,
                sparse_carrier_dosage_threshold: g_plan::DosageThreshold::try_from(0.1)
                    .expect("valid test sparse dosage threshold"),
            },
            null_firth: g_plan::NullFirthKernelPlan {
                maximum_iterations: 50,
                gradient_tolerance: g_plan::PositiveF64::try_from(1.0e-5)
                    .expect("valid test null Firth gradient tolerance"),
                maximum_step_size: g_plan::PositiveF64::try_from(10.0).expect("valid test null Firth maximum step"),
                fallback_iteration_multiplier: 2,
                fallback_step_divisor: g_plan::PositiveF64::try_from(2.0)
                    .expect("valid test null Firth fallback divisor"),
                line_search_maximum_attempts: 5,
                step_halving_scale: g_plan::StepScale::try_from(0.5).expect("valid test step-halving scale"),
            },
        }
    }

    fn output_authority_run_plan(
        directory: &BgenIntegrationTestDirectory,
        configured_content_sha256: Option<BgenContentSha256>,
        resume: bool,
    ) -> RunPlan {
        let sample_path = directory.path.join("input.sample");
        let phenotype_path = directory.path.join("phenotypes.tsv");
        let prediction_list_path = directory.path.join("predictions.list");
        RunPlan {
            association_mode: g_plan::AssociationMode::Regenie2Binary,
            chunk_size: 1,
            input: g_plan::InputPlan {
                bgen_path: directory.path.join("missing-input.bgen").display().to_string(),
                bgen_content_sha256: configured_content_sha256,
                sample_path: sample_path.display().to_string(),
                phenotype_path: phenotype_path.display().to_string(),
                prediction_list_path: prediction_list_path.display().to_string(),
                covariate_path: None,
                covariate_names: Vec::new(),
            },
            compute: g_plan::ComputePlan {
                device: g_plan::Device::Gpu,
                cpu_thread_count: None,
                jax_cache_directory: None,
                multi_phenotype_sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
                kernels: bgen_test_kernel_plan(),
            },
            correction: g_plan::CorrectionPlan {
                method: g_plan::BinaryFallbackMethod::FirthApproximate,
                p_threshold: g_plan::Probability::try_from(0.05).expect("valid test correction threshold"),
                firth_se: false,
            },
            output: g_plan::OutputPlan {
                output_run_root: directory.path.join("results").display().to_string(),
                resume,
                recover_attempt: None,
                fenced_owner_claim_id: None,
                writer_thread_count: 1,
            },
            telemetry: g_plan::TelemetryMode::Off,
            phenotype_runs: vec![g_plan::PhenotypeRunPlan {
                phenotype_name: "trait".to_string(),
                output_directory_name: "trait_0001_trait".to_string(),
            }],
        }
    }

    fn write_output_authority_inputs(directory: &BgenIntegrationTestDirectory) {
        directory.write("input.sample", b"ID_1 ID_2\n0 0\nfamily sample\n");
        directory.write("phenotypes.tsv", b"FID\tIID\ttrait\nfamily\tsample\t1\n");
        directory.write("predictions.list", b"");
    }

    fn publish_interrupted_output_authority(run_plan: RunPlan, evidence: BgenContentEvidence) {
        let header = g_output::CurrentRunManifestHeaderInput {
            phenotype_name: "trait".to_string(),
            bgen_content_evidence: Arc::new(evidence),
            covariate_names: Arc::from(Vec::<String>::new()),
            prediction_loco_files: Arc::from(Vec::new()),
            sample_count: 1,
            variant_count: 1,
            resolved_gpu_genotype_format: GpuGenotypeFormat::Packed8,
            sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
            phenotype_compute_group_id: Arc::from("test-group"),
            sample_set_fingerprint: Arc::from("test-sample-set"),
            covariate_design_fingerprint: Arc::from("test-covariate-design"),
            phenotype_design_fingerprint: Arc::from("test-phenotype-design"),
            prediction_alignment_fingerprint: Arc::from("test-prediction-alignment"),
        };
        let planned_chunk_range = 0..1;
        g_output::OutputManager::open(Arc::new(run_plan), "# test output authority\n".to_string())
            .expect("test output authority should plan")
            .initialize(
                vec![header],
                std::slice::from_ref(&planned_chunk_range),
                false,
                bgen_test_association_implementation(),
            )
            .expect("test output authority should initialize")
            .finish_interrupted("SIGTERM")
            .expect("test output authority should publish an interrupted terminal");
    }

    fn collect_output_tree_files(root_path: &Path, current_path: &Path, files: &mut Vec<OutputTreeFile>) {
        let mut entries = std::fs::read_dir(current_path)
            .expect("output authority directory should be readable")
            .map(|entry| entry.expect("output authority directory entry should be readable").path())
            .collect::<Vec<_>>();
        entries.sort();
        for entry_path in entries {
            let metadata =
                std::fs::symlink_metadata(&entry_path).expect("output authority metadata should be readable");
            if metadata.is_dir() {
                collect_output_tree_files(root_path, &entry_path, files);
            } else {
                assert!(metadata.is_file(), "output authority should contain only directories and regular files");
                files.push(OutputTreeFile {
                    relative_path: entry_path
                        .strip_prefix(root_path)
                        .expect("output authority file should remain below its root")
                        .to_path_buf(),
                    contents: std::fs::read(&entry_path).expect("output authority file should be readable"),
                });
            }
        }
    }

    fn output_tree_snapshot(root_path: &Path) -> Vec<OutputTreeFile> {
        let mut files = Vec::new();
        collect_output_tree_files(root_path, root_path, &mut files);
        files
    }

    fn prepare_run_error(run_plan: RunPlan) -> RunPreparationError {
        let run_engine = RunEngine::open(run_plan, "# failed BGEN preparation\n".to_string())
            .expect("test run should pass read-only output planning");
        match run_engine.prepare() {
            Ok(_) => panic!("test run should fail during BGEN authority preparation"),
            Err(error) => error,
        }
    }

    #[test]
    fn bgen_open_request_reconciles_fresh_and_persisted_content_authority() {
        let locator = Path::new("configured-input.bgen");
        let configured_content_sha256 = BgenContentSha256::from_bytes([0x11; 32]);
        let persisted_content_fingerprint =
            BgenContentFingerprint { content_sha256: configured_content_sha256, byte_count: 912_345 };
        let persisted_agreement = bgen_resume_agreement(persisted_content_fingerprint, GpuGenotypeFormat::Packed8);

        assert_eq!(
            build_bgen_open_request(locator, None, None).expect("unselected request should build"),
            BgenOpenRequest { locator, content_selector: None }
        );
        assert_eq!(
            build_bgen_open_request(locator, Some(configured_content_sha256), None)
                .expect("fresh selected request should build"),
            BgenOpenRequest {
                locator,
                content_selector: Some(BgenContentSelector {
                    content_sha256: configured_content_sha256,
                    expected_byte_count: None,
                }),
            }
        );
        for configured_selector in [None, Some(configured_content_sha256)] {
            assert_eq!(
                build_bgen_open_request(locator, configured_selector, Some(&persisted_agreement))
                    .expect("persisted selected request should build"),
                BgenOpenRequest {
                    locator,
                    content_selector: Some(BgenContentSelector {
                        content_sha256: configured_content_sha256,
                        expected_byte_count: Some(persisted_content_fingerprint.byte_count),
                    }),
                }
            );
        }
    }

    #[test]
    fn configured_bgen_mismatch_precedes_missing_locator_and_output_owner_mutation() {
        let directory = BgenIntegrationTestDirectory::new("configured-mismatch");
        write_output_authority_inputs(&directory);
        let persisted_content_fingerprint =
            BgenContentFingerprint { content_sha256: BgenContentSha256::from_bytes([0x22; 32]), byte_count: 731 };
        let fresh_plan = output_authority_run_plan(&directory, None, false);
        publish_interrupted_output_authority(
            fresh_plan,
            BgenContentEvidence::OwnedSnapshot(persisted_content_fingerprint),
        );
        let output_root = directory.path.join("results");
        let output_before = output_tree_snapshot(&output_root);

        let configured_content_sha256 = BgenContentSha256::from_bytes([0x33; 32]);
        let error = prepare_run_error(output_authority_run_plan(&directory, Some(configured_content_sha256), true));

        assert!(matches!(
            error,
            RunPreparationError::BgenContentSelectorMismatch {
                configured_content_sha256: observed_configured,
                persisted_content_sha256: observed_persisted,
            } if observed_configured == configured_content_sha256
                && observed_persisted == persisted_content_fingerprint.content_sha256
        ));
        assert!(
            !directory.path.join("missing-input.bgen").exists(),
            "error precedence test locator should remain absent"
        );
        assert_eq!(output_tree_snapshot(&output_root), output_before);
    }

    #[test]
    fn unattested_resume_authority_precedes_missing_locator_and_output_owner_mutation() {
        let directory = BgenIntegrationTestDirectory::new("unattested-authority");
        write_output_authority_inputs(&directory);
        let fresh_plan = output_authority_run_plan(&directory, None, false);
        publish_interrupted_output_authority(
            fresh_plan,
            BgenContentEvidence::PositionedUnattested(BgenSourceIdentity {
                configured_path: directory.path.join("historical-input.bgen"),
                canonical_path: None,
                device_identifier: 0,
                inode_identifier: 0,
                change_time_nanoseconds: 0,
                modification_time_nanoseconds: 0,
                file_size: 731,
            }),
        );
        let output_root = directory.path.join("results");
        let output_before = output_tree_snapshot(&output_root);

        let error = prepare_run_error(output_authority_run_plan(
            &directory,
            Some(BgenContentSha256::from_bytes([0x44; 32])),
            true,
        ));

        assert!(matches!(
            error,
            RunPreparationError::Output(g_output::OutputError::ExistingOutputUnattestedBgenContent { .. })
        ));
        assert!(
            !directory.path.join("missing-input.bgen").exists(),
            "error precedence test locator should remain absent"
        );
        assert_eq!(output_tree_snapshot(&output_root), output_before);
    }

    #[test]
    fn persisted_selector_reuses_authenticated_reader_without_locator_access() {
        let directory = BgenIntegrationTestDirectory::new("locator-free-cache-hit");
        let bgen_bytes = single_variant_bgen_bytes(&[2, 2, 2], &[0, 0, 255, 0, 0, 255]);
        let bgen_path = directory.write("selected-input.bgen", &bgen_bytes);
        let fingerprint_reader =
            g_genotype::BgenReaderCore::open(&bgen_path).expect("small unselected BGEN should provide a fingerprint");
        let fingerprint = owned_content_fingerprint(&fingerprint_reader);
        drop(fingerprint_reader);

        let priming_request = build_bgen_open_request(&bgen_path, Some(fingerprint.content_sha256), None)
            .expect("fresh selected request should build");
        let priming_reader =
            g_genotype::BgenReaderCore::open_request(priming_request).expect("selected reader should seed the cache");
        assert_eq!(priming_reader.source_provenance().resolution, BgenSnapshotResolution::CapturedFromLocator);
        drop(priming_reader);
        std::fs::remove_file(&bgen_path).expect("selected locator should be removed after cache priming");

        let persisted_agreement = bgen_resume_agreement(fingerprint, GpuGenotypeFormat::Packed8);
        let resumed_request = build_bgen_open_request(&bgen_path, None, Some(&persisted_agreement))
            .expect("persisted selector request should build");
        assert_eq!(
            resumed_request.content_selector,
            Some(BgenContentSelector {
                content_sha256: fingerprint.content_sha256,
                expected_byte_count: Some(fingerprint.byte_count),
            })
        );
        let resumed_reader = g_genotype::BgenReaderCore::open_request(resumed_request)
            .expect("persisted selector should reuse the cache without its locator");
        assert_eq!(resumed_reader.source_provenance().resolution, BgenSnapshotResolution::ProcessSnapshotCache);
        assert_eq!(resumed_reader.content_evidence(), &BgenContentEvidence::OwnedSnapshot(fingerprint));
    }

    #[test]
    fn persisted_packed8_authority_rejects_dosage_required_bgen() {
        let directory = BgenIntegrationTestDirectory::new("packed8-incompatible");
        let bgen_bytes = single_variant_bgen_bytes(&[0x82], &[0, 0]);
        let bgen_path = directory.write("missing-genotype.bgen", &bgen_bytes);
        let reader = g_genotype::BgenReaderCore::open(&bgen_path).expect("missing-value BGEN should open");
        let content_fingerprint = owned_content_fingerprint(&reader);
        let genotype_input = PreparedGenotypeInput { reader, chunk_size: 1 };
        let persisted_agreement = bgen_resume_agreement(content_fingerprint, GpuGenotypeFormat::Packed8);

        assert!(matches!(
            resolve_gpu_genotype_format(g_plan::Device::Gpu, Some(&persisted_agreement), &genotype_input),
            Err(RunPreparationError::ResumedPacked8Incompatible)
        ));
    }

    #[test]
    fn delivery_token_failure_explicitly_aborts_active_output_and_remains_primary() {
        let token_error = g_output::OutputError::InvalidInput("token failure".to_string());
        let mut observed_failure_reason = None;
        let outcome = abort_active_output_after_delivery_token_error(token_error, |failure_reason| {
            observed_failure_reason = Some(failure_reason.to_string());
            Ok(())
        });

        assert!(observed_failure_reason.is_some_and(|reason| reason.contains("token failure")));
        assert!(outcome.post_session_cleanup.is_none());
        assert!(
            matches!(outcome.error, RunPreparationError::Output(g_output::OutputError::InvalidInput(message)) if message == "token failure")
        );
    }

    #[test]
    fn delivery_token_failure_keeps_abort_failure_as_secondary_context() {
        let outcome = abort_active_output_after_delivery_token_error(
            g_output::OutputError::InvalidInput("token failure".to_string()),
            |_failure_reason| {
                Err(g_output::OutputTerminalError::from(g_output::OutputError::InvalidInput(
                    "abort failure".to_string(),
                )))
            },
        );

        assert!(outcome.post_session_cleanup.is_none());
        let RunPreparationError::OutputDeliveryTokenAbort { token, abort } = outcome.error else {
            panic!("token and abort failures should retain combined context");
        };
        assert!(matches!(token, g_output::OutputError::InvalidInput(message) if message == "token failure"));
        assert_eq!(abort.to_string(), "abort failure");
    }

    #[test]
    fn completed_noop_token_abort_rejection_keeps_token_failure_primary() {
        let outcome = delivery_token_abort_failure_outcome(
            g_output::OutputError::InvalidInput("token failure".to_string()),
            g_output::OutputError::InvalidInput("completed output cannot abort".to_string()),
            Some("cleanup"),
        );

        assert_eq!(outcome.post_session_cleanup, Some("cleanup"));
        assert!(
            matches!(outcome.error, RunPreparationError::Output(g_output::OutputError::InvalidInput(message)) if message == "token failure")
        );
    }

    #[test]
    fn completed_noop_interrupt_rejection_keeps_interruption_primary() {
        let outcome: RunExecutionOutcome<g_output::OutputError, g_output::OutputError, &str> =
            interrupted_output_failure_outcome(
                g_output::OutputError::InvalidInput("SIGTERM".to_string()),
                g_output::OutputError::InvalidInput("completed output cannot interrupt".to_string()),
                Some("cleanup"),
            );

        assert_eq!(outcome.post_session_cleanup, Some("cleanup"));
        assert!(matches!(
            outcome.result,
            Err(RunExecutionError::Interrupted(g_output::OutputError::InvalidInput(message)))
                if message == "SIGTERM"
        ));
    }

    #[test]
    fn writable_interrupt_flush_failure_keeps_output_context() {
        let outcome: RunExecutionOutcome<g_output::OutputError, g_output::OutputError, ()> =
            interrupted_output_failure_outcome(
                g_output::OutputError::InvalidInput("SIGTERM".to_string()),
                g_output::OutputError::InvalidInput("flush failure".to_string()),
                None,
            );

        assert!(outcome.post_session_cleanup.is_none());
        assert!(matches!(
            outcome.result,
            Err(RunExecutionError::InterruptedOutputFlush {
                interruption: g_output::OutputError::InvalidInput(interruption),
                output: g_output::OutputError::InvalidInput(output),
            }) if interruption == "SIGTERM" && output == "flush failure"
        ));
    }

    #[test]
    fn completed_noop_abort_rejection_keeps_delivery_failure_primary() {
        let outcome = delivery_abort_failure_outcome(
            DeliveryError::<g_output::OutputError, g_output::OutputError>::InvalidInput("delivery failure".to_string()),
            g_output::OutputError::InvalidInput("completed output cannot abort".to_string()),
            Some("cleanup"),
        );

        assert_eq!(outcome.post_session_cleanup, Some("cleanup"));
        assert!(matches!(
            outcome.result,
            Err(RunExecutionError::Delivery(DeliveryError::InvalidInput(message)))
                if message == "delivery failure"
        ));
    }

    #[test]
    fn writable_abort_failure_keeps_delivery_and_output_context() {
        let outcome = delivery_abort_failure_outcome(
            DeliveryError::<g_output::OutputError, g_output::OutputError>::InvalidInput("delivery failure".to_string()),
            g_output::OutputError::InvalidInput("abort failure".to_string()),
            Option::<()>::None,
        );

        assert!(outcome.post_session_cleanup.is_none());
        assert!(matches!(
            outcome.result,
            Err(RunExecutionError::DeliveryAbort {
                delivery,
                output: g_output::OutputError::InvalidInput(output),
            }) if matches!(*delivery, DeliveryError::InvalidInput(ref message) if message == "delivery failure")
                && output == "abort failure"
        ));
    }

    #[test]
    fn opaque_post_session_cleanup_retains_retry_authority_after_failure() {
        let mut cleanup = EnginePostSessionCleanup::retry_sentinel();
        assert_eq!(cleanup.purpose(), EnginePostSessionCleanupPurpose::CompletedNoop);
        let first_error = cleanup.cleanup().expect_err("the first cleanup attempt is injected to fail");
        assert_eq!(first_error.to_string(), "injected first cleanup failure");
        cleanup.cleanup().expect("the same opaque cleanup authority remains retryable");
    }
}
