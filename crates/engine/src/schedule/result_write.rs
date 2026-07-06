use super::{
    RESULT_WRITE_ITEM_KIND_MULTI_RESULT, RESULT_WRITE_ITEM_KIND_SINGLE_RESULT, RESULT_WRITE_ITEM_KIND_STOP_SIGNAL,
    ScheduleError,
};

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
    pub result_work_item_kind: String,
    pub expected_result_work_item_kind: String,
    pub should_process_result_write_item: bool,
    pub should_process_multi_result_write_item: bool,
    pub has_dispatch_error: bool,
    pub error_message: Option<String>,
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
    validate_result_write_item_kind(result_work_item_kind)?;
    validate_result_write_item_kind(expected_result_work_item_kind)?;

    if result_work_item_kind == RESULT_WRITE_ITEM_KIND_STOP_SIGNAL {
        return Ok(ResultWriteItemDispatchPlan {
            result_work_item_kind: result_work_item_kind.to_owned(),
            expected_result_work_item_kind: expected_result_work_item_kind.to_owned(),
            should_process_result_write_item: false,
            should_process_multi_result_write_item: false,
            has_dispatch_error: true,
            error_message: Some("Native result write dispatch plan continued without a work item.".to_owned()),
        });
    }

    if result_work_item_kind != expected_result_work_item_kind {
        return Ok(ResultWriteItemDispatchPlan {
            result_work_item_kind: result_work_item_kind.to_owned(),
            expected_result_work_item_kind: expected_result_work_item_kind.to_owned(),
            should_process_result_write_item: false,
            should_process_multi_result_write_item: false,
            has_dispatch_error: true,
            error_message: Some(format!(
                "Native result write dispatch plan expected {expected_result_work_item_kind} but received {result_work_item_kind}."
            )),
        });
    }

    Ok(ResultWriteItemDispatchPlan {
        result_work_item_kind: result_work_item_kind.to_owned(),
        expected_result_work_item_kind: expected_result_work_item_kind.to_owned(),
        should_process_result_write_item: result_work_item_kind == RESULT_WRITE_ITEM_KIND_SINGLE_RESULT,
        should_process_multi_result_write_item: result_work_item_kind == RESULT_WRITE_ITEM_KIND_MULTI_RESULT,
        has_dispatch_error: false,
        error_message: None,
    })
}

fn validate_result_write_item_kind(result_work_item_kind: &str) -> Result<(), ScheduleError> {
    match result_work_item_kind {
        RESULT_WRITE_ITEM_KIND_SINGLE_RESULT
        | RESULT_WRITE_ITEM_KIND_MULTI_RESULT
        | RESULT_WRITE_ITEM_KIND_STOP_SIGNAL => Ok(()),
        _ => Err(ScheduleError::UnsupportedResultWriteItemKind {
            result_work_item_kind: result_work_item_kind.to_owned(),
        }),
    }
}
