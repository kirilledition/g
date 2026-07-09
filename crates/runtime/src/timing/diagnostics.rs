use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use super::payloads::NullLogisticDiagnosticValue;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimingDiagnosticError {
    ExpectedSingleValue { value_label: &'static str },
    NullLogisticValueCountMismatch { convergence_count: usize, iteration_count: usize },
    NullLogisticPhenotypeNameCountMismatch { phenotype_name_count: usize, convergence_count: usize },
}

impl fmt::Display for TimingDiagnosticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExpectedSingleValue { value_label } => {
                write!(formatter, "{value_label} must contain exactly one value.")
            }
            Self::NullLogisticValueCountMismatch { convergence_count, iteration_count } => write!(
                formatter,
                "Null logistic convergence value count ({convergence_count}) must match iteration count value count ({iteration_count}).",
            ),
            Self::NullLogisticPhenotypeNameCountMismatch { phenotype_name_count, convergence_count } => write!(
                formatter,
                "Null logistic phenotype name count ({phenotype_name_count}) must match convergence value count ({convergence_count}).",
            ),
        }
    }
}

impl Error for TimingDiagnosticError {}

/// Build scalar null-logistic diagnostics for timing/profile output.
///
/// # Errors
///
/// Returns an error when scalar diagnostic vectors do not contain exactly one value.
pub fn build_scalar_null_logistic_diagnostics(
    chromosome: String,
    convergence_flags: Vec<bool>,
    iteration_counts: Vec<i64>,
    firth_iteration_counts: Vec<i64>,
    firth_convergence_reason_codes: Vec<i64>,
    correction_method: String,
) -> Result<BTreeMap<String, NullLogisticDiagnosticValue>, TimingDiagnosticError> {
    let converged = require_single_value(convergence_flags, "Scalar null logistic convergence values")?;
    let iteration_count = require_single_value(iteration_counts, "Scalar null logistic iteration counts")?;
    let firth_iteration_count = require_single_value(firth_iteration_counts, "Scalar null Firth iteration counts")?;
    let firth_convergence_reason_code =
        require_single_value(firth_convergence_reason_codes, "Scalar null Firth convergence reason codes")?;
    let mut diagnostics = BTreeMap::new();
    diagnostics.insert("chromosome".to_string(), NullLogisticDiagnosticValue::Text(chromosome));
    diagnostics.insert("iteration_count".to_string(), NullLogisticDiagnosticValue::Integer(iteration_count));
    diagnostics.insert("converged".to_string(), NullLogisticDiagnosticValue::Integer(i64::from(converged)));
    diagnostics
        .insert("firth_iteration_count".to_string(), NullLogisticDiagnosticValue::Integer(firth_iteration_count));
    diagnostics.insert(
        "firth_convergence_reason_code".to_string(),
        NullLogisticDiagnosticValue::Integer(firth_convergence_reason_code),
    );
    diagnostics.insert("correction_method".to_string(), NullLogisticDiagnosticValue::Text(correction_method));
    Ok(diagnostics)
}

/// Build per-phenotype null-logistic diagnostics for timing/profile output.
///
/// # Errors
///
/// Returns an error when convergence, iteration, and phenotype-name counts do
/// not match.
pub fn build_multi_null_logistic_diagnostics(
    chromosome: &str,
    convergence_flags: Vec<bool>,
    iteration_counts: Vec<i64>,
    phenotype_names: Vec<String>,
    correction_method: &str,
) -> Result<Vec<BTreeMap<String, NullLogisticDiagnosticValue>>, TimingDiagnosticError> {
    if convergence_flags.len() != iteration_counts.len() {
        return Err(TimingDiagnosticError::NullLogisticValueCountMismatch {
            convergence_count: convergence_flags.len(),
            iteration_count: iteration_counts.len(),
        });
    }
    if phenotype_names.len() != convergence_flags.len() {
        return Err(TimingDiagnosticError::NullLogisticPhenotypeNameCountMismatch {
            phenotype_name_count: phenotype_names.len(),
            convergence_count: convergence_flags.len(),
        });
    }
    convergence_flags
        .into_iter()
        .zip(iteration_counts)
        .zip(phenotype_names)
        .map(|((converged, iteration_count), phenotype_name)| {
            let mut diagnostics = BTreeMap::new();
            diagnostics.insert("chromosome".to_string(), NullLogisticDiagnosticValue::Text(chromosome.to_string()));
            diagnostics.insert("phenotype".to_string(), NullLogisticDiagnosticValue::Text(phenotype_name));
            diagnostics.insert("iteration_count".to_string(), NullLogisticDiagnosticValue::Integer(iteration_count));
            diagnostics.insert("converged".to_string(), NullLogisticDiagnosticValue::Integer(i64::from(converged)));
            diagnostics.insert(
                "correction_method".to_string(),
                NullLogisticDiagnosticValue::Text(correction_method.to_string()),
            );
            Ok(diagnostics)
        })
        .collect()
}

fn require_single_value<T>(values: Vec<T>, value_label: &'static str) -> Result<T, TimingDiagnosticError> {
    if values.len() != 1 {
        return Err(TimingDiagnosticError::ExpectedSingleValue { value_label });
    }
    values.into_iter().next().ok_or(TimingDiagnosticError::ExpectedSingleValue { value_label })
}
