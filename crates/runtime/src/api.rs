//! Public runtime crate facade.

pub use crate::diagnostics::{DiagnosticEventError, emit_diagnostic_event};
pub use crate::error::RuntimeCompatibilityError;
pub use crate::logging_sink::LoggingSinkError;
pub use crate::native_run_session::{NativeRunSession, NativeRunSessionError};
pub use crate::runtime_policy::NativeRunSessionPolicy;
pub use crate::runtime_state::{ProcessRuntimeState, RayonThreadPoolConfigurationError};
pub use crate::shutdown::{ShutdownError, sigterm_shutdown_requested};
pub use crate::telemetry_session::{TelemetryRunError, TelemetryRunSession};
pub use crate::timing::{StageTimingRecorder, TimingFileError};
