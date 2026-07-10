//! Deterministic graceful-shutdown signal metadata helpers.

mod error;
mod process;
mod signal;

pub use error::ShutdownError;
pub use process::{SigtermShutdownScope, begin_sigterm_shutdown_scope, sigterm_shutdown_requested};
pub use signal::{ShutdownSignalPayload, build_shutdown_signal};
