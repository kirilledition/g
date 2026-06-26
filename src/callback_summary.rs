//! Native callback summary counter state.

#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
pub(crate) struct BinaryChunkDiagnosticsInput {
    pub(crate) score_only_count: i64,
    pub(crate) score_test_candidate_count: i64,
    pub(crate) firth_candidate_count: i64,
    pub(crate) firth_converged_count: i64,
    pub(crate) firth_failed_count: i64,
    pub(crate) firth_numerical_failure_count: i64,
    pub(crate) firth_max_iteration_failure_count: i64,
    pub(crate) firth_invalid_statistic_failure_count: i64,
    pub(crate) firth_step_halving_failure_count: i64,
    pub(crate) pseudo_firth_attempt_count: i64,
    pub(crate) pseudo_firth_success_count: i64,
    pub(crate) nr_zero_start_attempt_count: i64,
    pub(crate) nr_zero_start_success_count: i64,
    pub(crate) nr_warm_start_attempt_count: i64,
    pub(crate) nr_warm_start_success_count: i64,
    pub(crate) sparse_correction_count: i64,
    pub(crate) dense_correction_count: i64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
pub(crate) struct BinaryCorrectionSummaryState {
    pub(crate) chunk_count: i64,
    pub(crate) score_only_count: i64,
    pub(crate) score_test_candidate_count: i64,
    pub(crate) firth_attempted_count: i64,
    pub(crate) firth_success_count: i64,
    pub(crate) firth_failed_count: i64,
    pub(crate) firth_numerical_failure_count: i64,
    pub(crate) firth_max_iteration_failure_count: i64,
    pub(crate) firth_invalid_statistic_failure_count: i64,
    pub(crate) firth_step_halving_failure_count: i64,
    pub(crate) pseudo_firth_attempt_count: i64,
    pub(crate) pseudo_firth_success_count: i64,
    pub(crate) nr_zero_start_attempt_count: i64,
    pub(crate) nr_zero_start_success_count: i64,
    pub(crate) nr_warm_start_attempt_count: i64,
    pub(crate) nr_warm_start_success_count: i64,
    pub(crate) sparse_correction_count: i64,
    pub(crate) dense_correction_count: i64,
    pub(crate) null_model_failure_count: i64,
}

impl BinaryCorrectionSummaryState {
    pub(crate) fn add_null_model_failure_count(&mut self, failure_count: i64) {
        self.null_model_failure_count += failure_count;
    }

    pub(crate) fn add_diagnostics(&mut self, diagnostics: &BinaryChunkDiagnosticsInput) {
        self.add_diagnostics_totals(1, diagnostics);
    }

    pub(crate) fn add_diagnostics_totals(&mut self, chunk_count: i64, diagnostics: &BinaryChunkDiagnosticsInput) {
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

    pub(crate) fn should_emit(&self) -> bool {
        self.chunk_count != 0 || self.null_model_failure_count != 0
    }
}
