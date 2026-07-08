//! Public facade for frontend configuration normalization.

pub use crate::cli::{CliOutcomeData, dispatch_cli};
pub use crate::defaults::load_packaged_config_data;
pub use crate::domain::{
    ArrowCompressionValue, DeviceValue, FloatingPointDtypeValue, GpuGenotypeFormatValue, JaxMatmulPrecisionValue,
    MultiPhenotypeSampleModeValue, NullLogisticNonconvergencePolicyValue, OutputFormatValue, ParquetCompressionValue,
    RegenieTraitTypeValue, ResumeModeValue, SampleKeyModeValue, TelemetryModeValue, TrustedBgenValidationModeValue,
};
pub use crate::native_cli::{
    NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE, NATIVE_EXECUTION_UNAVAILABLE_MESSAGE,
    NATIVE_PYTHON_BRIDGE_ENVIRONMENT_VARIABLE, NATIVE_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE, NativeCliOutcome,
    dispatch_native_cli,
};
pub use crate::options::{ConfigOptionMetadata, ConfigOptionValueKind, config_option_metadata};
pub use crate::plan_request::compile_run_request;
pub use crate::resolved::{
    BinaryConfigData, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData, InputConfigData,
    RegenieConfigData, TraitConfigData,
};
pub use crate::run_validation::validate_config_for_run;
pub use crate::toml::{dumps_toml, from_options, from_toml_path, write_toml};
pub use crate::validation::validate_config;
