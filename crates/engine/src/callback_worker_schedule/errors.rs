use super::types::{CallbackWorkerErrorRaisePlan, CallbackWorkerErrorUpdatePlan};

#[must_use]
pub fn format_dosage_callback_worker_error_message(error_message: &str) -> String {
    format!("native pipeline callback worker failed: {error_message}")
}

#[must_use]
pub fn format_result_callback_worker_error_message(error_message: &str) -> String {
    format!("native pipeline result writer worker failed: {error_message}")
}

pub(crate) fn update_callback_worker_error(
    worker_error_message: &mut Option<String>,
    error_message: Option<&str>,
    format_worker_error_message: fn(&str) -> String,
) -> CallbackWorkerErrorUpdatePlan {
    let had_error = worker_error_message.is_some();
    *worker_error_message = error_message.map(format_worker_error_message);
    CallbackWorkerErrorUpdatePlan {
        had_error,
        has_error: worker_error_message.is_some(),
        error_message: worker_error_message.clone(),
    }
}

pub(crate) fn plan_callback_worker_error_raise(
    dosage_worker_error_message: Option<&str>,
    result_worker_error_message: Option<&str>,
) -> CallbackWorkerErrorRaisePlan {
    if let Some(error_message) = dosage_worker_error_message {
        return CallbackWorkerErrorRaisePlan {
            should_raise: true,
            raise_dosage_worker_error: true,
            raise_result_worker_error: false,
            error_message: Some(error_message.to_string()),
        };
    }
    if let Some(error_message) = result_worker_error_message {
        return CallbackWorkerErrorRaisePlan {
            should_raise: true,
            raise_dosage_worker_error: false,
            raise_result_worker_error: true,
            error_message: Some(error_message.to_string()),
        };
    }
    CallbackWorkerErrorRaisePlan {
        should_raise: false,
        raise_dosage_worker_error: false,
        raise_result_worker_error: false,
        error_message: None,
    }
}
