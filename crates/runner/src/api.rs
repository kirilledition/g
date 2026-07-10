//! Public runner facade consumed by the native Python host.

pub use crate::run::{
    CliRunResult, JaxDevice, JaxRuntimeConfigUpdate, JaxRuntimeConfigValue, NativeRunFailure, NativeRunHost,
    NativeRunInterruption, run_cli,
};
