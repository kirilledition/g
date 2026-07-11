//! Public runner facade consumed by the native Python host.

pub use crate::backend_plan::JaxAssociationBackendPlan;
pub use crate::cli_output::CliRunResult;
pub use crate::jax_runtime::{JaxDevice, JaxRuntimeConfigUpdate, JaxRuntimeConfigValue};
pub use crate::run::{NativeRunFailure, NativeRunHost, NativeRunInterruption, run_cli};
