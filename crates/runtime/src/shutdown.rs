//! Deterministic graceful-shutdown signal metadata helpers.

mod controller;
mod error;
mod signal;

pub use controller::{
    ShutdownControllerState, ShutdownHandlerInstallPlan, ShutdownHandlerRestorePlan, ShutdownHandlerSession,
    ShutdownRequestAction, ShutdownRequestDecisionPayload,
};
pub use error::ShutdownError;
pub use signal::{
    SecondSignalExceptionPlan, ShutdownSignalPayload, build_shutdown_signal, default_shutdown_signal_numbers,
    plan_second_signal_exception,
};
