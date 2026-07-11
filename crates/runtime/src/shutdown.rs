//! Deterministic graceful-shutdown signal metadata helpers.

mod error;
mod process;

pub use error::ShutdownError;
pub use process::sigterm_shutdown_requested;
pub(crate) use process::{SigtermShutdownScope, begin_sigterm_shutdown_scope};
