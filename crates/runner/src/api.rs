//! Public runner facade consumed by the native Python host.

pub use crate::run::{CliRunResult, NativeRunHost, NativeRunInterruption, run_cli};
pub use g_runtime::{
    JaxDeviceObservation, JaxRuntimeConfigUpdatePayload, JaxRuntimeConfigValue, RunFailedEventPayload,
    build_run_failed_event_payload, sigterm_shutdown_requested,
};
