//! Native callback summary counter state.

#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
pub struct BinaryChunkDiagnosticsInput {
    pub score_only_count: i64,
    pub score_test_candidate_count: i64,
    pub firth_candidate_count: i64,
    pub firth_converged_count: i64,
    pub firth_failed_count: i64,
    pub firth_numerical_failure_count: i64,
    pub firth_max_iteration_failure_count: i64,
    pub firth_invalid_statistic_failure_count: i64,
    pub firth_step_halving_failure_count: i64,
    pub pseudo_firth_attempt_count: i64,
    pub pseudo_firth_success_count: i64,
    pub nr_zero_start_attempt_count: i64,
    pub nr_zero_start_success_count: i64,
    pub nr_warm_start_attempt_count: i64,
    pub nr_warm_start_success_count: i64,
    pub sparse_correction_count: i64,
    pub dense_correction_count: i64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
pub struct BinaryCorrectionSummaryState {
    pub chunk_count: i64,
    pub score_only_count: i64,
    pub score_test_candidate_count: i64,
    pub firth_attempted_count: i64,
    pub firth_success_count: i64,
    pub firth_failed_count: i64,
    pub firth_numerical_failure_count: i64,
    pub firth_max_iteration_failure_count: i64,
    pub firth_invalid_statistic_failure_count: i64,
    pub firth_step_halving_failure_count: i64,
    pub pseudo_firth_attempt_count: i64,
    pub pseudo_firth_success_count: i64,
    pub nr_zero_start_attempt_count: i64,
    pub nr_zero_start_success_count: i64,
    pub nr_warm_start_attempt_count: i64,
    pub nr_warm_start_success_count: i64,
    pub sparse_correction_count: i64,
    pub dense_correction_count: i64,
    pub null_model_failure_count: i64,
}

impl BinaryCorrectionSummaryState {
    pub fn add_null_model_failure_count(&mut self, failure_count: i64) {
        self.null_model_failure_count += failure_count;
    }

    pub fn add_diagnostics(&mut self, diagnostics: &BinaryChunkDiagnosticsInput) {
        self.add_diagnostics_totals(1, diagnostics);
    }

    pub fn add_diagnostics_totals(&mut self, chunk_count: i64, diagnostics: &BinaryChunkDiagnosticsInput) {
        self.chunk_count += chunk_count;
        self.score_only_count += diagnostics.score_only_count;
        self.score_test_candidate_count += diagnostics.score_test_candidate_count;
        self.firth_attempted_count += diagnostics.firth_candidate_count;
        self.firth_success_count += diagnostics.firth_converged_count;
        self.firth_failed_count += diagnostics.firth_failed_count;
        self.firth_numerical_failure_count += diagnostics.firth_numerical_failure_count;
        self.firth_max_iteration_failure_count += diagnostics.firth_max_iteration_failure_count;
        self.firth_invalid_statistic_failure_count += diagnostics.firth_invalid_statistic_failure_count;
        self.firth_step_halving_failure_count += diagnostics.firth_step_halving_failure_count;
        self.pseudo_firth_attempt_count += diagnostics.pseudo_firth_attempt_count;
        self.pseudo_firth_success_count += diagnostics.pseudo_firth_success_count;
        self.nr_zero_start_attempt_count += diagnostics.nr_zero_start_attempt_count;
        self.nr_zero_start_success_count += diagnostics.nr_zero_start_success_count;
        self.nr_warm_start_attempt_count += diagnostics.nr_warm_start_attempt_count;
        self.nr_warm_start_success_count += diagnostics.nr_warm_start_success_count;
        self.sparse_correction_count += diagnostics.sparse_correction_count;
        self.dense_correction_count += diagnostics.dense_correction_count;
    }

    #[must_use]
    pub const fn should_emit(&self) -> bool {
        self.chunk_count != 0 || self.null_model_failure_count != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagnostics_input() -> BinaryChunkDiagnosticsInput {
        BinaryChunkDiagnosticsInput {
            score_only_count: 1,
            score_test_candidate_count: 2,
            firth_candidate_count: 3,
            firth_converged_count: 4,
            firth_failed_count: 5,
            firth_numerical_failure_count: 6,
            firth_max_iteration_failure_count: 7,
            firth_invalid_statistic_failure_count: 8,
            firth_step_halving_failure_count: 9,
            pseudo_firth_attempt_count: 10,
            pseudo_firth_success_count: 11,
            nr_zero_start_attempt_count: 12,
            nr_zero_start_success_count: 13,
            nr_warm_start_attempt_count: 14,
            nr_warm_start_success_count: 15,
            sparse_correction_count: 16,
            dense_correction_count: 17,
        }
    }

    #[test]
    fn accumulates_binary_correction_summary_counts() {
        let diagnostics = diagnostics_input();
        let mut summary = BinaryCorrectionSummaryState::default();

        summary.add_diagnostics(&diagnostics);
        summary.add_diagnostics_totals(3, &diagnostics);
        summary.add_null_model_failure_count(5);

        assert_eq!(
            summary,
            BinaryCorrectionSummaryState {
                chunk_count: 4,
                score_only_count: 2,
                score_test_candidate_count: 4,
                firth_attempted_count: 6,
                firth_success_count: 8,
                firth_failed_count: 10,
                firth_numerical_failure_count: 12,
                firth_max_iteration_failure_count: 14,
                firth_invalid_statistic_failure_count: 16,
                firth_step_halving_failure_count: 18,
                pseudo_firth_attempt_count: 20,
                pseudo_firth_success_count: 22,
                nr_zero_start_attempt_count: 24,
                nr_zero_start_success_count: 26,
                nr_warm_start_attempt_count: 28,
                nr_warm_start_success_count: 30,
                sparse_correction_count: 32,
                dense_correction_count: 34,
                null_model_failure_count: 5,
            }
        );
    }

    #[test]
    fn emits_when_chunks_or_null_model_failures_exist() {
        let mut null_failure_summary = BinaryCorrectionSummaryState::default();
        assert!(!null_failure_summary.should_emit());

        null_failure_summary.add_null_model_failure_count(1);
        assert!(null_failure_summary.should_emit());

        let mut chunk_summary = BinaryCorrectionSummaryState::default();
        chunk_summary.add_diagnostics_totals(1, &BinaryChunkDiagnosticsInput::default());
        assert!(chunk_summary.should_emit());
    }
}
