pub(super) const RUN_FAILED_EVENT_NAME: &str = "run_failed";
pub(super) const EXECUTION_PLAN_PREPARED_EVENT_NAME: &str = "execution_plan_prepared";
pub(super) const WRITER_FINISHED_EVENT_NAME: &str = "writer_finished";
pub(super) const ASSOCIATION_BACKEND_SELECTED_EVENT_NAME: &str = "association_backend_selected";
pub(super) const RUN_LIFECYCLE_INFO_LEVEL: &str = "info";
pub(super) const RUN_LIFECYCLE_WARN_LEVEL: &str = "warn";
pub(super) const RUN_LIFECYCLE_ERROR_LEVEL: &str = "error";
pub(super) const NATIVE_CLI_INTERRUPTED_LINE_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_interrupted_line";
pub(super) const NATIVE_CLI_FAILED_LINE_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_failed_line";
pub(super) const NATIVE_CLI_COMPLETED_LINE_DIAGNOSTIC_EVENT_NAME: &str = "native_cli_completed_line";
pub(super) const RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_execution_plan_build_started";
pub(super) const RUNNER_EXECUTION_PLAN_PREPARED_DIAGNOSTIC_EVENT_NAME: &str = "runner_execution_plan_prepared";
pub(super) const RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_execution_plan_dispatch_started";
pub(super) const RUNNER_EXECUTION_PLAN_BUILD_STARTED_DIAGNOSTIC_MESSAGE: &str = "Building REGENIE execution plan.";
pub(super) const RUNNER_EXECUTION_PLAN_DISPATCH_STARTED_DIAGNOSTIC_MESSAGE: &str =
    "Dispatching REGENIE execution plan.";
pub(super) const NATIVE_DISPATCH_DELIVERY_FINISHED_DIAGNOSTIC_EVENT_NAME: &str = "native_dispatch_delivery_finished";
pub(super) const NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_EVENT_NAME: &str = "native_runtime_knobs_configured";
pub(super) const NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_MESSAGE: &str = "Configuring native runtime knobs.";
pub(super) const RUNNER_METADATA_ARTIFACTS_COMPLETED_DIAGNOSTIC_EVENT_NAME: &str =
    "runner_metadata_artifacts_completed";
