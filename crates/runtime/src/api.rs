//! Public runtime crate facade.

pub use crate::error::{RuntimeCompatibilityError, RuntimeError, RuntimeResult};
pub use crate::jax_runtime::{
    JaxRuntimeDiagnosticEventPayload, JaxRuntimeSetupPayload, JaxRuntimeSetupSession, JaxRuntimeSetupSideEffectPlan,
    plan_jax_runtime_setup_side_effects, resolve_jax_runtime_setup,
};
pub use crate::logging_sink::{LoggingSinkConfig, initialize_logging_sinks, shutdown_logging_sinks};
pub use crate::runtime_state::{
    JaxRuntimePolicyPayload, ProcessRuntimeState, RunRuntime, RuntimeCompatibilityToken, RuntimePolicyPayload,
};
pub use crate::shutdown::{ShutdownHandlerSession, ShutdownSignalPayload};
pub use crate::telemetry_session::{TelemetryEventEnvelope, TelemetryRunSessionState};
pub use crate::timing::{StageTimingRecorder, StageTimingSnapshotPayload};
