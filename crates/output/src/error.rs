//! Public output error boundary.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("Another process published a conflicting immutable output lineage record at '{}'.", record_path.display())]
    ConcurrentLineageUpdate { record_path: PathBuf },
    #[error(
        "Output owner claim '{claim_id}' at '{}' survives from process {process_id} on host '{host_name}'. Its owner may be live or may have crashed; this BeeGFS mount cannot distinguish those states safely. First use an external coordinator to fence the recorded owner, then resume with --fenced-output-owner-claim '{claim_id}' (or output.fenced_owner_claim_id). Do not remove the claim manually or infer fencing from age or PID state.",
        claim_path.display()
    )]
    SurvivingOutputOwnerClaim { claim_path: PathBuf, claim_id: String, host_name: String, process_id: u32 },
    #[error(
        "Output owner claim '{claim_id}' at '{}' could not be released and remains authoritative. Do not remove or take over the claim without an external coordinator that has fenced this exact process claim. Release failure: {reason}",
        claim_path.display()
    )]
    RetainedOutputOwnerClaimRelease { claim_path: PathBuf, claim_id: String, reason: String },
    #[error(
        "Output owner claim at '{}' has a visible graceful-release transition, but transition durability could not be confirmed after one synchronous directory-sync retry. First failure: {first_failure}; retry failure: {retry_failure}",
        claim_path.display()
    )]
    PublishedOutputOwnerClaimReleaseDurability { claim_path: PathBuf, first_failure: String, retry_failure: String },
    #[error(
        "Output owner claim '{claim_id}' became the visible authority at '{}', but publication durability could not be confirmed. The run did not start; externally fence this exact claim before recovery. Publication failure: {reason}",
        claim_path.display()
    )]
    PublishedOutputOwnerClaimDurability { claim_path: PathBuf, claim_id: String, reason: String },
    #[error("{primary} Output owner cleanup also failed: {release}")]
    OutputOperationAndOwnerClaimRelease { primary: Box<OutputError>, release: Box<OutputError> },
    #[error("{0}")]
    Runtime(String),
}

impl OutputError {
    // `Result::map_err` transfers each concrete source error into this adapter;
    // accepting it by value matches that ownership boundary directly.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn runtime(error: impl ToString) -> Self {
        Self::Runtime(error.to_string())
    }
}

pub(crate) type OutputResult<T> = Result<T, OutputError>;
