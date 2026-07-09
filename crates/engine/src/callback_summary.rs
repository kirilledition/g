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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BinaryCorrectionDiagnosticsRecordPlan {
    pub should_record: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BinaryCorrectionSummaryEmitPlan {
    pub should_flush_pending_diagnostics: bool,
    pub should_emit_summary: bool,
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

    #[must_use]
    pub const fn chunk_count_with_pending(&self, pending_diagnostics_count: i64) -> i64 {
        self.chunk_count + pending_diagnostics_count
    }

    #[must_use]
    pub const fn plan_diagnostics_record(
        has_telemetry_session: bool,
        has_diagnostics: bool,
    ) -> BinaryCorrectionDiagnosticsRecordPlan {
        BinaryCorrectionDiagnosticsRecordPlan { should_record: has_telemetry_session && has_diagnostics }
    }

    #[must_use]
    pub const fn plan_summary_emit(
        &self,
        has_telemetry_session: bool,
        pending_diagnostics_count: i64,
    ) -> BinaryCorrectionSummaryEmitPlan {
        let should_flush_pending_diagnostics = has_telemetry_session && pending_diagnostics_count > 0;
        BinaryCorrectionSummaryEmitPlan {
            should_flush_pending_diagnostics,
            should_emit_summary: has_telemetry_session && (self.should_emit() || should_flush_pending_diagnostics),
        }
    }
}
