use super::ScheduleError;

const RESULT_WRITE_ITEM_KIND_SINGLE_RESULT: &str = "single_result";
const RESULT_WRITE_ITEM_KIND_MULTI_RESULT: &str = "multi_result";
const RESULT_WRITE_ITEM_KIND_STOP_SIGNAL: &str = "stop_signal";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResultWriteItemKind {
    MultiResult,
    SingleResult,
    StopSignal,
}

impl ResultWriteItemKind {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::MultiResult => RESULT_WRITE_ITEM_KIND_MULTI_RESULT,
            Self::SingleResult => RESULT_WRITE_ITEM_KIND_SINGLE_RESULT,
            Self::StopSignal => RESULT_WRITE_ITEM_KIND_STOP_SIGNAL,
        }
    }

    /// Parse a result-write work item kind from its serialized value.
    ///
    /// # Errors
    ///
    /// Returns an error when the work item kind is unsupported.
    pub fn from_value(result_work_item_kind: &str) -> Result<Self, ScheduleError> {
        match result_work_item_kind {
            RESULT_WRITE_ITEM_KIND_SINGLE_RESULT => Ok(Self::SingleResult),
            RESULT_WRITE_ITEM_KIND_MULTI_RESULT => Ok(Self::MultiResult),
            RESULT_WRITE_ITEM_KIND_STOP_SIGNAL => Ok(Self::StopSignal),
            _ => Err(ScheduleError::UnsupportedResultWriteItemKind {
                result_work_item_kind: result_work_item_kind.to_owned(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultWriteItemResourceReleasePlan {
    pub should_release_host_buffer: bool,
    pub should_release_result_in_flight_slot: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultWriteHandoffPlan {
    pub should_enqueue: bool,
    pub has_result_work_item: bool,
    pub is_stop_signal: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultWriteDrainCompletionPlan {
    pub should_stop: bool,
    pub should_flush_binary_correction_diagnostics: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultWriteItemDispatchPlan {
    result_work_item_kind: ResultWriteItemKind,
    expected_result_work_item_kind: ResultWriteItemKind,
    pub should_process_result_write_item: bool,
    pub should_process_multi_result_write_item: bool,
    pub has_dispatch_error: bool,
    pub error_message: Option<String>,
}

impl ResultWriteItemDispatchPlan {
    #[must_use]
    pub const fn result_work_item_kind(&self) -> ResultWriteItemKind {
        self.result_work_item_kind
    }

    #[must_use]
    pub const fn expected_result_work_item_kind(&self) -> ResultWriteItemKind {
        self.expected_result_work_item_kind
    }
}

#[must_use]
pub const fn plan_result_write_handoff(has_result_work_item: bool) -> ResultWriteHandoffPlan {
    ResultWriteHandoffPlan { should_enqueue: true, has_result_work_item, is_stop_signal: !has_result_work_item }
}

/// Plan which result write processing path should consume a dequeued item.
///
/// # Errors
///
/// Returns an error when either work-item kind is unsupported.
pub fn plan_result_write_item_dispatch(
    result_work_item_kind: &str,
    expected_result_work_item_kind: &str,
) -> Result<ResultWriteItemDispatchPlan, ScheduleError> {
    let result_work_item_kind = ResultWriteItemKind::from_value(result_work_item_kind)?;
    let expected_result_work_item_kind = ResultWriteItemKind::from_value(expected_result_work_item_kind)?;
    Ok(plan_result_write_item_dispatch_for_kinds(result_work_item_kind, expected_result_work_item_kind))
}

#[must_use]
pub fn plan_result_write_item_dispatch_for_kinds(
    result_work_item_kind: ResultWriteItemKind,
    expected_result_work_item_kind: ResultWriteItemKind,
) -> ResultWriteItemDispatchPlan {
    if result_work_item_kind == ResultWriteItemKind::StopSignal {
        return ResultWriteItemDispatchPlan {
            result_work_item_kind,
            expected_result_work_item_kind,
            should_process_result_write_item: false,
            should_process_multi_result_write_item: false,
            has_dispatch_error: true,
            error_message: Some("Native result write dispatch plan continued without a work item.".to_owned()),
        };
    }

    if result_work_item_kind != expected_result_work_item_kind {
        return ResultWriteItemDispatchPlan {
            result_work_item_kind,
            expected_result_work_item_kind,
            should_process_result_write_item: false,
            should_process_multi_result_write_item: false,
            has_dispatch_error: true,
            error_message: Some(format!(
                "Native result write dispatch plan expected {} but received {}.",
                expected_result_work_item_kind.as_value(),
                result_work_item_kind.as_value(),
            )),
        };
    }

    ResultWriteItemDispatchPlan {
        result_work_item_kind,
        expected_result_work_item_kind,
        should_process_result_write_item: result_work_item_kind == ResultWriteItemKind::SingleResult,
        should_process_multi_result_write_item: result_work_item_kind == ResultWriteItemKind::MultiResult,
        has_dispatch_error: false,
        error_message: None,
    }
}
