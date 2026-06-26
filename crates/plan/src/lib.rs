#![warn(clippy::pedantic)]

pub mod host_policy;
pub mod prepared;
pub mod request;

pub use host_policy::{
    AssociationBackendPlanPayload, BinaryCorrectionPlanPayload, HostPolicyError, JaxRuntimeSetupPayload,
    PhenotypeComputeGroupPayload, build_phenotype_compute_group_id, build_phenotype_compute_groups,
    build_phenotype_output_directory_name, normalize_binary_correction, plan_association_backend,
    resolve_association_mode, resolve_jax_runtime_setup,
};
pub use prepared::{
    AssociationBackendKind, AssociationBackendPlan, JaxPolicyPlan, ManifestFileFingerprint, PredictionInputsIdentity,
    PredictionLocoFileFingerprint, PreparedComputePlan, PreparedInputIdentity, PreparedOutputWriterPlan,
    PreparedPhenotypeComputeGroup, PreparedRunPlan, PreparedSampleMode,
};
pub use request::{
    ArrowCompression, AssociationMode, BinaryFallbackMethod, ComputeRequest, CorrectionPlan, Device,
    FloatingPointDtype, GpuGenotypeFormat, InputRequest, JaxMatmulPrecision, MultiPhenotypeSampleMode, OutputFormat,
    OutputWriterPlan, ParquetCompression, PhenotypeComputeGroup, PhenotypeComputeGroupMode, PhenotypeRunPlan,
    RegenieTraitType, ResumeMode, RunRequest, RuntimePlan, SampleKeyMode, TraitRequest, TrustedBgenValidationMode,
};
