//! Native null-logistic nonconvergence policy.

const NULL_LOGISTIC_POLICY_FAIL: &str = "fail";
const NULL_LOGISTIC_POLICY_WARN: &str = "warn";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NullLogisticNonconvergenceAction {
    Continue,
    Warn,
    Fail,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct NullLogisticNonconvergencePlan {
    pub action: NullLogisticNonconvergenceAction,
    pub failed_trait_indices: Vec<usize>,
    pub message: Option<String>,
    pub warning_message: Option<String>,
    pub nonconverged_count: usize,
    pub scalar_convergence: bool,
    pub total_fit_count: usize,
}

#[derive(Debug, Eq, PartialEq)]
struct NullLogisticNonconvergenceMessage {
    failed_trait_indices: Vec<usize>,
    message: String,
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum NullLogisticPolicyError {
    #[error("Null logistic convergence flags must contain at least one value.")]
    EmptyConvergenceFlags,
    #[error("Scalar null logistic convergence must contain exactly one flag, observed {observed_count}.")]
    ScalarConvergenceFlagCount { observed_count: usize },
    #[error(
        "Null logistic phenotype name count ({phenotype_name_count}) must match convergence flag count ({convergence_flag_count})."
    )]
    PhenotypeNameCountMismatch { phenotype_name_count: usize, convergence_flag_count: usize },
    #[error("Unsupported null logistic nonconvergence policy: {policy}")]
    UnsupportedNullLogisticPolicy { policy: String },
}

/// Plan how a binary null-logistic nonconvergence observation should be handled.
///
/// # Errors
///
/// Returns an error when the convergence flags are empty, scalar mode receives
/// anything other than one flag, phenotype names do not match multi-trait flag
/// count, or the policy value is unsupported.
pub(crate) fn plan_null_logistic_nonconvergence(
    chromosome: &str,
    convergence_flags: &[bool],
    scalar_convergence: bool,
    phenotype_names: Option<&[String]>,
    policy: &str,
) -> Result<NullLogisticNonconvergencePlan, NullLogisticPolicyError> {
    if convergence_flags.is_empty() {
        return Err(NullLogisticPolicyError::EmptyConvergenceFlags);
    }
    let action = match policy {
        NULL_LOGISTIC_POLICY_FAIL => NullLogisticNonconvergenceAction::Fail,
        NULL_LOGISTIC_POLICY_WARN => NullLogisticNonconvergenceAction::Warn,
        _ => return Err(NullLogisticPolicyError::UnsupportedNullLogisticPolicy { policy: policy.to_string() }),
    };
    let NullLogisticNonconvergenceMessage { failed_trait_indices, message } =
        build_null_logistic_nonconvergence_message(chromosome, convergence_flags, scalar_convergence, phenotype_names)?;
    let nonconverged_count = failed_trait_indices.len();
    let total_fit_count = convergence_flags.len();
    if failed_trait_indices.is_empty() {
        return Ok(NullLogisticNonconvergencePlan {
            action: NullLogisticNonconvergenceAction::Continue,
            failed_trait_indices,
            message: None,
            warning_message: None,
            nonconverged_count,
            scalar_convergence,
            total_fit_count,
        });
    }
    let warning_message = (action == NullLogisticNonconvergenceAction::Warn)
        .then(|| format!("{message} Continuing because --null_logistic_nonconvergence_policy=warn."));
    Ok(NullLogisticNonconvergencePlan {
        action,
        failed_trait_indices,
        message: Some(message),
        warning_message,
        nonconverged_count,
        scalar_convergence,
        total_fit_count,
    })
}

fn build_null_logistic_nonconvergence_message(
    chromosome: &str,
    convergence_flags: &[bool],
    scalar_convergence: bool,
    phenotype_names: Option<&[String]>,
) -> Result<NullLogisticNonconvergenceMessage, NullLogisticPolicyError> {
    if scalar_convergence {
        if convergence_flags.len() != 1 {
            return Err(NullLogisticPolicyError::ScalarConvergenceFlagCount {
                observed_count: convergence_flags.len(),
            });
        }
        if convergence_flags[0] {
            return Ok(NullLogisticNonconvergenceMessage { failed_trait_indices: Vec::new(), message: String::new() });
        }
        return Ok(NullLogisticNonconvergenceMessage {
            failed_trait_indices: vec![0],
            message: format!("Binary null logistic model did not converge for chromosome {chromosome}."),
        });
    }

    if let Some(phenotype_names) = phenotype_names
        && phenotype_names.len() != convergence_flags.len()
    {
        return Err(NullLogisticPolicyError::PhenotypeNameCountMismatch {
            phenotype_name_count: phenotype_names.len(),
            convergence_flag_count: convergence_flags.len(),
        });
    }
    let failed_trait_indices = convergence_flags
        .iter()
        .enumerate()
        .filter_map(|(trait_index, converged)| if *converged { None } else { Some(trait_index) })
        .collect::<Vec<_>>();
    if failed_trait_indices.is_empty() {
        return Ok(NullLogisticNonconvergenceMessage { failed_trait_indices, message: String::new() });
    }
    let failed_traits = failed_trait_indices
        .iter()
        .map(|trait_index| {
            phenotype_names
                .and_then(|names| names.get(*trait_index))
                .cloned()
                .unwrap_or_else(|| trait_index.to_string())
        })
        .collect::<Vec<_>>()
        .join(", ");
    Ok(NullLogisticNonconvergenceMessage {
        failed_trait_indices,
        message: format!("Binary null logistic model did not converge for chromosome {chromosome}: {failed_traits}."),
    })
}
