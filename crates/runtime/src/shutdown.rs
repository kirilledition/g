//! Deterministic graceful-shutdown signal metadata helpers.

mod error;
mod process;

pub use error::ShutdownError;
pub use process::{SigtermShutdownScope, begin_sigterm_shutdown_scope, sigterm_shutdown_requested};
