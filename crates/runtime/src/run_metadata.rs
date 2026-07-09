//! Deterministic run metadata and artifact payload construction.

mod artifacts;
mod error;
mod manifest;
mod types;

pub use artifacts::{
    build_execution_run_artifacts, build_execution_run_artifacts_from_sequences, build_multi_run_artifacts,
    build_phenotype_run_artifacts,
};
pub use error::RunMetadataError;
pub use manifest::build_run_manifest_extension;
pub use types::{
    ExecutionRunArtifactsInput, ExecutionRunArtifactsSequenceInput, PhenotypeRunArtifactsInput,
    RunManifestCommandPayload, RunManifestExtensionInput, RunManifestExtensionPayload, RunManifestRuntimePayload,
};
