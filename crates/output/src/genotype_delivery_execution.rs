//! Output-owned genotype-delivery execution evidence.

use crate::error::OutputError;

/// Effective genotype-delivery path used by one phenotype compute group.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenotypeDeliveryEffectivePath {
    /// Decode and materialize genotype values on the host.
    Host,
    /// Deliver raw-DEFLATE BGEN payloads through nvCOMP and packed8 finalization.
    RawDeflateNvcomp,
}

/// Stable identity of the raw-DEFLATE packed8 typed-XLA artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawDeflatePacked8Artifact {
    ffi_target: String,
    ffi_api_version: u32,
    handler_sha256: String,
    ptx_sha256: String,
    ptx_isa: String,
    ptx_target: String,
    capability_requirements: RawDeflatePacked8CapabilityRequirements,
}

/// Minimum CUDA runtime capabilities encoded in raw packed8 execution evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawDeflatePacked8CapabilityRequirements {
    cuda_driver_version: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
}

/// Immutable execution evidence for one phenotype compute group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenotypeDeliveryExecution {
    phenotype_compute_group_id: String,
    effective_path: GenotypeDeliveryEffectivePath,
    processed_chunk_count: u64,
    raw_deflate_nvcomp_chunk_count: u64,
    host_chunk_count: u64,
    raw_deflate_packed8_artifact: Option<RawDeflatePacked8Artifact>,
}

impl RawDeflatePacked8CapabilityRequirements {
    /// Build validated minimum CUDA runtime requirements.
    ///
    /// # Errors
    ///
    /// Returns an error when the driver version or compute capability is
    /// outside its valid domain.
    pub fn new(
        minimum_cuda_driver_version: i32,
        minimum_compute_capability_major: i32,
        minimum_compute_capability_minor: i32,
    ) -> Result<Self, OutputError> {
        if minimum_cuda_driver_version <= 0 {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE packed8 artifact minimum CUDA driver version must be positive.".to_string(),
            ));
        }
        if minimum_compute_capability_major <= 0 || minimum_compute_capability_minor < 0 {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE packed8 artifact minimum compute capability must have a positive major and nonnegative minor."
                    .to_string(),
            ));
        }
        Ok(Self {
            cuda_driver_version: minimum_cuda_driver_version,
            compute_capability_major: minimum_compute_capability_major,
            compute_capability_minor: minimum_compute_capability_minor,
        })
    }

    /// Return the reviewed minimum CUDA driver API version.
    #[must_use]
    pub const fn minimum_cuda_driver_version(self) -> i32 {
        self.cuda_driver_version
    }

    /// Return the minimum compute-capability major version.
    #[must_use]
    pub const fn minimum_compute_capability_major(self) -> i32 {
        self.compute_capability_major
    }

    /// Return the minimum compute-capability minor version.
    #[must_use]
    pub const fn minimum_compute_capability_minor(self) -> i32 {
        self.compute_capability_minor
    }
}

impl RawDeflatePacked8Artifact {
    /// Build a validated raw-DEFLATE packed8 artifact identity.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty stable identifier, a zero FFI API
    /// version, a noncanonical handler or PTX SHA-256, or a PTX target that
    /// contradicts its minimum capability.
    pub fn new(
        ffi_target: String,
        ffi_api_version: u32,
        handler_sha256: String,
        ptx_sha256: String,
        ptx_isa: String,
        ptx_target: String,
        capability_requirements: RawDeflatePacked8CapabilityRequirements,
    ) -> Result<Self, OutputError> {
        if ffi_target.is_empty() {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE packed8 artifact FFI target must not be empty.".to_string(),
            ));
        }
        if ffi_api_version == 0 {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE packed8 artifact FFI API version must be a positive integer.".to_string(),
            ));
        }
        if !crate::digest::is_canonical_sha256(&handler_sha256) {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE packed8 artifact handler SHA-256 must contain exactly 64 lowercase hexadecimal characters."
                    .to_string(),
            ));
        }
        if !crate::digest::is_canonical_sha256(&ptx_sha256) {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE packed8 artifact PTX SHA-256 must contain exactly 64 lowercase hexadecimal characters."
                    .to_string(),
            ));
        }
        if ptx_isa.is_empty() {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE packed8 artifact PTX ISA must not be empty.".to_string(),
            ));
        }
        if ptx_target.is_empty() {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE packed8 artifact PTX target must not be empty.".to_string(),
            ));
        }
        if !crate::association_implementation::ptx_target_matches_compute_capability(
            &ptx_target,
            capability_requirements.minimum_compute_capability_major(),
            capability_requirements.minimum_compute_capability_minor(),
        ) {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE packed8 artifact PTX target must match its minimum compute capability.".to_string(),
            ));
        }
        Ok(Self {
            ffi_target,
            ffi_api_version,
            handler_sha256,
            ptx_sha256,
            ptx_isa,
            ptx_target,
            capability_requirements,
        })
    }

    /// Return the exact typed-XLA FFI target.
    #[must_use]
    pub fn ffi_target(&self) -> &str {
        &self.ffi_target
    }

    /// Return the typed-XLA FFI API version.
    #[must_use]
    pub const fn ffi_api_version(&self) -> u32 {
        self.ffi_api_version
    }

    /// Return the framed native handler and ABI-input SHA-256.
    #[must_use]
    pub fn handler_sha256(&self) -> &str {
        &self.handler_sha256
    }

    /// Return the canonical SHA-256 of the embedded PTX.
    #[must_use]
    pub fn ptx_sha256(&self) -> &str {
        &self.ptx_sha256
    }

    /// Return the embedded PTX ISA version.
    #[must_use]
    pub fn ptx_isa(&self) -> &str {
        &self.ptx_isa
    }

    /// Return the embedded PTX compilation target.
    #[must_use]
    pub fn ptx_target(&self) -> &str {
        &self.ptx_target
    }

    /// Return the reviewed minimum CUDA driver API version.
    #[must_use]
    pub const fn minimum_cuda_driver_version(&self) -> i32 {
        self.capability_requirements.minimum_cuda_driver_version()
    }

    /// Return the embedded PTX target's minimum compute-capability major version.
    #[must_use]
    pub const fn minimum_compute_capability_major(&self) -> i32 {
        self.capability_requirements.minimum_compute_capability_major()
    }

    /// Return the embedded PTX target's minimum compute-capability minor version.
    #[must_use]
    pub const fn minimum_compute_capability_minor(&self) -> i32 {
        self.capability_requirements.minimum_compute_capability_minor()
    }
}

impl GenotypeDeliveryExecution {
    /// Record host delivery for one compute group.
    ///
    /// # Errors
    ///
    /// Returns an error when the compute-group identifier is empty.
    pub fn host(phenotype_compute_group_id: String, processed_chunk_count: u64) -> Result<Self, OutputError> {
        validate_compute_group_identifier(&phenotype_compute_group_id)?;
        Ok(Self {
            phenotype_compute_group_id,
            effective_path: GenotypeDeliveryEffectivePath::Host,
            processed_chunk_count,
            raw_deflate_nvcomp_chunk_count: 0,
            host_chunk_count: processed_chunk_count,
            raw_deflate_packed8_artifact: None,
        })
    }

    /// Record raw-DEFLATE nvCOMP delivery for one compute group.
    ///
    /// # Errors
    ///
    /// Returns an error when the compute-group identifier is empty or no chunk
    /// was processed through the raw path.
    pub fn raw_deflate_nvcomp(
        phenotype_compute_group_id: String,
        processed_chunk_count: u64,
        artifact: RawDeflatePacked8Artifact,
    ) -> Result<Self, OutputError> {
        validate_compute_group_identifier(&phenotype_compute_group_id)?;
        if processed_chunk_count == 0 {
            return Err(OutputError::InvalidInput(
                "Raw-DEFLATE nvCOMP genotype delivery must process at least one chunk.".to_string(),
            ));
        }
        Ok(Self {
            phenotype_compute_group_id,
            effective_path: GenotypeDeliveryEffectivePath::RawDeflateNvcomp,
            processed_chunk_count,
            raw_deflate_nvcomp_chunk_count: processed_chunk_count,
            host_chunk_count: 0,
            raw_deflate_packed8_artifact: Some(artifact),
        })
    }

    /// Return the stable phenotype compute-group identifier.
    #[must_use]
    pub fn phenotype_compute_group_id(&self) -> &str {
        &self.phenotype_compute_group_id
    }

    /// Return the effective genotype-delivery path.
    #[must_use]
    pub const fn effective_path(&self) -> GenotypeDeliveryEffectivePath {
        self.effective_path
    }

    /// Return the number of chunks processed in this lifecycle.
    #[must_use]
    pub const fn processed_chunk_count(&self) -> u64 {
        self.processed_chunk_count
    }

    /// Return the number of chunks delivered through raw-DEFLATE nvCOMP.
    #[must_use]
    pub const fn raw_deflate_nvcomp_chunk_count(&self) -> u64 {
        self.raw_deflate_nvcomp_chunk_count
    }

    /// Return the number of chunks decoded on the host.
    #[must_use]
    pub const fn host_chunk_count(&self) -> u64 {
        self.host_chunk_count
    }

    /// Return the raw-DEFLATE packed8 artifact when that path was effective.
    #[must_use]
    pub const fn raw_deflate_packed8_artifact(&self) -> Option<&RawDeflatePacked8Artifact> {
        self.raw_deflate_packed8_artifact.as_ref()
    }
}

fn validate_compute_group_identifier(phenotype_compute_group_id: &str) -> Result<(), OutputError> {
    if phenotype_compute_group_id.is_empty() {
        return Err(OutputError::InvalidInput(
            "Genotype-delivery phenotype compute-group identifier must not be empty.".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capability_requirements() -> RawDeflatePacked8CapabilityRequirements {
        RawDeflatePacked8CapabilityRequirements::new(12_020, 7, 0)
            .expect("test raw-DEFLATE packed8 requirements are valid")
    }

    fn artifact() -> RawDeflatePacked8Artifact {
        RawDeflatePacked8Artifact::new(
            "g.bgen.packed8_deflate.v1".to_string(),
            1,
            "a".repeat(64),
            "b".repeat(64),
            "8.2".to_string(),
            "sm_70".to_string(),
            capability_requirements(),
        )
        .expect("test raw-DEFLATE packed8 artifact is valid")
    }

    #[test]
    fn constructors_derive_path_counts_and_artifact_presence() {
        let host = GenotypeDeliveryExecution::host("host-group".to_string(), 3).expect("host execution is valid");
        assert_eq!(host.effective_path(), GenotypeDeliveryEffectivePath::Host);
        assert_eq!(host.processed_chunk_count(), 3);
        assert_eq!(host.raw_deflate_nvcomp_chunk_count(), 0);
        assert_eq!(host.host_chunk_count(), 3);
        assert_eq!(host.raw_deflate_packed8_artifact(), None);

        let raw = GenotypeDeliveryExecution::raw_deflate_nvcomp("raw-group".to_string(), 2, artifact())
            .expect("raw-nvCOMP execution is valid");
        assert_eq!(raw.effective_path(), GenotypeDeliveryEffectivePath::RawDeflateNvcomp);
        assert_eq!(raw.processed_chunk_count(), 2);
        assert_eq!(raw.raw_deflate_nvcomp_chunk_count(), 2);
        assert_eq!(raw.host_chunk_count(), 0);
        let raw_artifact = raw.raw_deflate_packed8_artifact().expect("raw execution carries its artifact");
        assert_eq!(raw_artifact.minimum_cuda_driver_version(), 12_020);
        assert_eq!(raw_artifact.minimum_compute_capability_major(), 7);
        assert_eq!(raw_artifact.minimum_compute_capability_minor(), 0);
        assert!(
            GenotypeDeliveryExecution::raw_deflate_nvcomp("raw-group".to_string(), 0, artifact()).is_err(),
            "zero-work execution is canonicalized to host"
        );
    }

    #[test]
    fn identities_reject_empty_fields_zero_api_and_noncanonical_hashes() {
        assert!(GenotypeDeliveryExecution::host(String::new(), 0).is_err());
        for (ffi_target, ffi_api_version, handler_sha256, ptx_sha256, ptx_isa, ptx_target) in [
            ("", 1, "a".repeat(64), "b".repeat(64), "8.2", "sm_70"),
            ("target", 0, "a".repeat(64), "b".repeat(64), "8.2", "sm_70"),
            ("target", 1, "A".repeat(64), "b".repeat(64), "8.2", "sm_70"),
            ("target", 1, "a".repeat(64), "short".to_string(), "8.2", "sm_70"),
            ("target", 1, "a".repeat(64), "b".repeat(64), "", "sm_70"),
            ("target", 1, "a".repeat(64), "b".repeat(64), "8.2", ""),
        ] {
            assert!(
                RawDeflatePacked8Artifact::new(
                    ffi_target.to_string(),
                    ffi_api_version,
                    handler_sha256,
                    ptx_sha256,
                    ptx_isa.to_string(),
                    ptx_target.to_string(),
                    capability_requirements(),
                )
                .is_err()
            );
        }
        assert!(RawDeflatePacked8CapabilityRequirements::new(0, 7, 0).is_err());
        assert!(RawDeflatePacked8CapabilityRequirements::new(12_020, 0, 0).is_err());
        assert!(RawDeflatePacked8CapabilityRequirements::new(12_020, 7, -1).is_err());
        let mismatched_requirements = RawDeflatePacked8CapabilityRequirements::new(12_020, 8, 0)
            .expect("mismatched requirements are well formed");
        assert!(
            RawDeflatePacked8Artifact::new(
                "target".to_string(),
                1,
                "a".repeat(64),
                "b".repeat(64),
                "8.2".to_string(),
                "sm_70".to_string(),
                mismatched_requirements,
            )
            .is_err()
        );
    }
}
