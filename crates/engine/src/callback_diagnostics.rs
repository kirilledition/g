//! Native callback diagnostics policy helpers.

const NULL_LOGISTIC_POLICY_FAIL: &str = "fail";
const NULL_LOGISTIC_POLICY_WARN: &str = "warn";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NullLogisticNonconvergenceAction {
    Continue,
    Warn,
    Fail,
}

impl NullLogisticNonconvergenceAction {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::Warn => "warn",
            Self::Fail => "fail",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NullLogisticNonconvergencePlan {
    pub action: NullLogisticNonconvergenceAction,
    pub failed_trait_indices: Vec<usize>,
    pub message: Option<String>,
    pub warning_message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NullLogisticNonconvergenceMessage {
    failed_trait_indices: Vec<usize>,
    message: String,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CallbackDiagnosticsError {
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
pub fn plan_null_logistic_nonconvergence(
    chromosome: &str,
    convergence_flags: &[bool],
    scalar_convergence: bool,
    phenotype_names: Option<&[String]>,
    policy: &str,
) -> Result<NullLogisticNonconvergencePlan, CallbackDiagnosticsError> {
    if convergence_flags.is_empty() {
        return Err(CallbackDiagnosticsError::EmptyConvergenceFlags);
    }
    let action = match policy {
        NULL_LOGISTIC_POLICY_FAIL => NullLogisticNonconvergenceAction::Fail,
        NULL_LOGISTIC_POLICY_WARN => NullLogisticNonconvergenceAction::Warn,
        _ => return Err(CallbackDiagnosticsError::UnsupportedNullLogisticPolicy { policy: policy.to_string() }),
    };
    let NullLogisticNonconvergenceMessage { failed_trait_indices, message } =
        build_null_logistic_nonconvergence_message(chromosome, convergence_flags, scalar_convergence, phenotype_names)?;
    if failed_trait_indices.is_empty() {
        return Ok(NullLogisticNonconvergencePlan {
            action: NullLogisticNonconvergenceAction::Continue,
            failed_trait_indices,
            message: None,
            warning_message: None,
        });
    }
    let warning_message = (action == NullLogisticNonconvergenceAction::Warn)
        .then(|| format!("{message} Continuing because --null_logistic_nonconvergence_policy=warn."));
    Ok(NullLogisticNonconvergencePlan { action, failed_trait_indices, message: Some(message), warning_message })
}

fn build_null_logistic_nonconvergence_message(
    chromosome: &str,
    convergence_flags: &[bool],
    scalar_convergence: bool,
    phenotype_names: Option<&[String]>,
) -> Result<NullLogisticNonconvergenceMessage, CallbackDiagnosticsError> {
    if scalar_convergence {
        if convergence_flags.len() != 1 {
            return Err(CallbackDiagnosticsError::ScalarConvergenceFlagCount {
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
        return Err(CallbackDiagnosticsError::PhenotypeNameCountMismatch {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_continue_when_null_logistic_converges() {
        assert_eq!(
            plan_null_logistic_nonconvergence("22", &[true], true, None, "fail").unwrap(),
            NullLogisticNonconvergencePlan {
                action: NullLogisticNonconvergenceAction::Continue,
                failed_trait_indices: Vec::new(),
                message: None,
                warning_message: None,
            },
        );
        assert_eq!(
            plan_null_logistic_nonconvergence("22", &[true, true], false, None, "warn").unwrap().action,
            NullLogisticNonconvergenceAction::Continue,
        );
    }

    #[test]
    fn plans_scalar_fail_policy() {
        assert_eq!(
            plan_null_logistic_nonconvergence("22", &[false], true, None, "fail").unwrap(),
            NullLogisticNonconvergencePlan {
                action: NullLogisticNonconvergenceAction::Fail,
                failed_trait_indices: vec![0],
                message: Some("Binary null logistic model did not converge for chromosome 22.".to_string()),
                warning_message: None,
            },
        );
    }

    #[test]
    fn plans_multi_trait_warn_policy_with_phenotype_names() {
        assert_eq!(
            plan_null_logistic_nonconvergence(
                "22",
                &[true, false, false],
                false,
                Some(&["trait_a".to_string(), "trait_b".to_string(), "trait_c".to_string()]),
                "warn",
            )
            .unwrap(),
            NullLogisticNonconvergencePlan {
                action: NullLogisticNonconvergenceAction::Warn,
                failed_trait_indices: vec![1, 2],
                message: Some(
                    "Binary null logistic model did not converge for chromosome 22: trait_b, trait_c.".to_string()
                ),
                warning_message: Some(
                    "Binary null logistic model did not converge for chromosome 22: trait_b, trait_c. Continuing because --null_logistic_nonconvergence_policy=warn.".to_string()
                ),
            },
        );
    }

    #[test]
    fn plans_multi_trait_message_with_trait_indices_without_names() {
        let plan = plan_null_logistic_nonconvergence("22", &[false, true, false], false, None, "fail").unwrap();

        assert_eq!(plan.failed_trait_indices, vec![0, 2]);
        assert_eq!(
            plan.message.as_deref(),
            Some("Binary null logistic model did not converge for chromosome 22: 0, 2."),
        );
    }

    #[test]
    fn rejects_invalid_null_logistic_policy_inputs() {
        assert_eq!(
            plan_null_logistic_nonconvergence("22", &[], false, None, "fail").unwrap_err(),
            CallbackDiagnosticsError::EmptyConvergenceFlags,
        );
        assert_eq!(
            plan_null_logistic_nonconvergence("22", &[true, false], true, None, "fail").unwrap_err(),
            CallbackDiagnosticsError::ScalarConvergenceFlagCount { observed_count: 2 },
        );
        assert_eq!(
            plan_null_logistic_nonconvergence("22", &[false, true], false, Some(&["trait".to_string()]), "fail")
                .unwrap_err(),
            CallbackDiagnosticsError::PhenotypeNameCountMismatch { phenotype_name_count: 1, convergence_flag_count: 2 },
        );
        assert_eq!(
            plan_null_logistic_nonconvergence("22", &[false], true, None, "ignore").unwrap_err(),
            CallbackDiagnosticsError::UnsupportedNullLogisticPolicy { policy: "ignore".to_string() },
        );
    }
}
