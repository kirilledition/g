//! Crate-private telemetry support for the native CLI and run engine.

pub(crate) mod logging;
pub(crate) mod run_events;
pub(crate) mod session;
pub(crate) mod telemetry_policy;

pub(crate) use crate::binding::errors;
