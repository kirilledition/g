#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallbackWorkerStartAction {
    StartDosageWorker,
    StartResultWorker,
}

impl CallbackWorkerStartAction {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::StartDosageWorker => "start_dosage_worker",
            Self::StartResultWorker => "start_result_worker",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallbackWorkerFinishAction {
    CompleteProgress,
    EmitBinaryCorrectionSummary,
    JoinDosageWorker,
    JoinResultWorker,
    RaiseWorkerError,
    StopDosageWorker,
    StopResultWorker,
}

impl CallbackWorkerFinishAction {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::CompleteProgress => "complete_progress",
            Self::EmitBinaryCorrectionSummary => "emit_binary_correction_summary",
            Self::JoinDosageWorker => "join_dosage_worker",
            Self::JoinResultWorker => "join_result_worker",
            Self::RaiseWorkerError => "raise_worker_error",
            Self::StopDosageWorker => "stop_dosage_worker",
            Self::StopResultWorker => "stop_result_worker",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackWorkerStartPlan {
    pub(super) start_actions: Vec<CallbackWorkerStartAction>,
}

impl CallbackWorkerStartPlan {
    #[must_use]
    pub fn start_actions(&self) -> &[CallbackWorkerStartAction] {
        &self.start_actions
    }

    #[must_use]
    pub fn should_start(&self) -> bool {
        !self.start_actions.is_empty()
    }

    #[must_use]
    pub fn start_result_worker(&self) -> bool {
        self.contains_start_action(CallbackWorkerStartAction::StartResultWorker)
    }

    #[must_use]
    pub fn start_dosage_worker(&self) -> bool {
        self.contains_start_action(CallbackWorkerStartAction::StartDosageWorker)
    }

    fn contains_start_action(&self, start_action: CallbackWorkerStartAction) -> bool {
        self.start_actions.contains(&start_action)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackWorkerStartAttemptPlan {
    pub(super) start_actions: Vec<CallbackWorkerStartAction>,
    pub has_marked_started: bool,
    pub has_start_error: bool,
    pub error_message: Option<String>,
}

impl CallbackWorkerStartAttemptPlan {
    #[must_use]
    pub fn start_actions(&self) -> &[CallbackWorkerStartAction] {
        &self.start_actions
    }

    #[must_use]
    pub fn should_start(&self) -> bool {
        !self.start_actions.is_empty()
    }

    #[must_use]
    pub fn start_result_worker(&self) -> bool {
        self.contains_start_action(CallbackWorkerStartAction::StartResultWorker)
    }

    #[must_use]
    pub fn start_dosage_worker(&self) -> bool {
        self.contains_start_action(CallbackWorkerStartAction::StartDosageWorker)
    }

    fn contains_start_action(&self, start_action: CallbackWorkerStartAction) -> bool {
        self.start_actions.contains(&start_action)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CallbackWorkerLifecycleState {
    started: bool,
}

impl CallbackWorkerLifecycleState {
    #[must_use]
    pub const fn new() -> Self {
        Self { started: false }
    }

    #[must_use]
    pub const fn has_started(&self) -> bool {
        self.started
    }

    pub fn mark_started(&mut self) -> bool {
        if self.started {
            return false;
        }
        self.started = true;
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackWorkerShutdownTimeouts {
    pub dosage_worker_join_timeout_seconds: f64,
    pub result_worker_join_timeout_seconds: f64,
    pub graceful_dosage_worker_join_timeout_seconds: f64,
    pub graceful_result_worker_join_timeout_seconds: f64,
    pub worker_abort_stop_timeout_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackWorkerJoinPlan {
    pub should_join: bool,
    pub timeout_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackWorkerStopPlan {
    pub should_stop: bool,
    pub timeout_seconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallbackWorkerFinishPlan {
    pub(super) finish_actions: Vec<CallbackWorkerFinishAction>,
    pub dosage_stop_timeout_seconds: f64,
    pub dosage_join_timeout_seconds: f64,
    pub result_stop_timeout_seconds: f64,
    pub result_join_timeout_seconds: f64,
}

impl CallbackWorkerFinishPlan {
    #[must_use]
    pub fn finish_actions(&self) -> &[CallbackWorkerFinishAction] {
        &self.finish_actions
    }

    #[must_use]
    pub fn stop_dosage_worker(&self) -> bool {
        self.contains_finish_action(CallbackWorkerFinishAction::StopDosageWorker)
    }

    #[must_use]
    pub fn join_dosage_worker(&self) -> bool {
        self.contains_finish_action(CallbackWorkerFinishAction::JoinDosageWorker)
    }

    #[must_use]
    pub fn stop_result_worker(&self) -> bool {
        self.contains_finish_action(CallbackWorkerFinishAction::StopResultWorker)
    }

    #[must_use]
    pub fn join_result_worker(&self) -> bool {
        self.contains_finish_action(CallbackWorkerFinishAction::JoinResultWorker)
    }

    #[must_use]
    pub fn raise_worker_error(&self) -> bool {
        self.contains_finish_action(CallbackWorkerFinishAction::RaiseWorkerError)
    }

    #[must_use]
    pub fn complete_progress(&self) -> bool {
        self.contains_finish_action(CallbackWorkerFinishAction::CompleteProgress)
    }

    #[must_use]
    pub fn emit_binary_correction_summary(&self) -> bool {
        self.contains_finish_action(CallbackWorkerFinishAction::EmitBinaryCorrectionSummary)
    }

    fn contains_finish_action(&self, finish_action: CallbackWorkerFinishAction) -> bool {
        self.finish_actions.contains(&finish_action)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallbackWorkerAbortPlan {
    pub(super) abort_actions: Vec<CallbackWorkerFinishAction>,
    pub dosage_stop_timeout_seconds: f64,
    pub result_stop_timeout_seconds: f64,
}

impl CallbackWorkerAbortPlan {
    #[must_use]
    pub fn abort_actions(&self) -> &[CallbackWorkerFinishAction] {
        &self.abort_actions
    }

    #[must_use]
    pub fn stop_dosage_worker(&self) -> bool {
        self.contains_abort_action(CallbackWorkerFinishAction::StopDosageWorker)
    }

    #[must_use]
    pub fn stop_result_worker(&self) -> bool {
        self.contains_abort_action(CallbackWorkerFinishAction::StopResultWorker)
    }

    fn contains_abort_action(&self, abort_action: CallbackWorkerFinishAction) -> bool {
        self.abort_actions.contains(&abort_action)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackWorkerStopPollPlan {
    pub should_stop: bool,
    pub poll_timeout_seconds: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackWorkerErrorRaisePlan {
    pub should_raise: bool,
    pub raise_dosage_worker_error: bool,
    pub raise_result_worker_error: bool,
    pub error_message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackWorkerErrorUpdatePlan {
    pub had_error: bool,
    pub has_error: bool,
    pub error_message: Option<String>,
}
