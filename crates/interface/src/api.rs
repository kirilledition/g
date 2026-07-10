//! Public facade for frontend configuration normalization.

pub use crate::cli::{CliDispatch, CompiledCliRun, dispatch_cli};
pub use crate::resolved::{
    BinaryConfigData, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData, InputConfigData,
    RegenieConfigData, TraitConfigData,
};
