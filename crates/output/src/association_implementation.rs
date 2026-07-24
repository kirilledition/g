//! Output-owned association implementation compatibility.

use crate::error::OutputError;

/// Runtime implementation state that must remain exact across output resume.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssociationImplementationCompatibility {
    jax_version: String,
    jaxlib_version: String,
    firth_components: Option<FirthComponentsCompatibility>,
}

/// Concrete implementation used for approximate-Firth component reductions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FirthComponentsImplementationCompatibility {
    /// Use the portable JAX implementation.
    Jax,
    /// Use the raw-CUDA typed-XLA FFI implementation.
    RawCuda,
}

/// Recoverable reason a requested raw-CUDA implementation used JAX instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FirthComponentsFallbackReasonCompatibility {
    /// The platform cannot load the Linux CUDA ABI.
    UnsupportedPlatform,
    /// The CUDA driver shared object is unavailable.
    CudaDriverUnavailable,
    /// A required CUDA driver symbol is unavailable.
    RequiredSymbolUnavailable,
    /// The CUDA driver is too old for the embedded PTX ISA.
    CudaDriverTooOld,
    /// The requested CUDA-visible device is unavailable.
    CudaDeviceUnavailable,
    /// The requested device cannot execute the embedded kernel.
    UnsupportedComputeCapability,
}

/// Stable identity of the raw-CUDA Firth artifact requested by runtime policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawCudaFirthArtifactCompatibility {
    ffi_target: String,
    ffi_api_version: u32,
    handler_sha256: String,
    ptx_sha256: String,
    ptx_isa: String,
    ptx_target: String,
    capability_requirements: RawCudaFirthCapabilityRequirementsCompatibility,
}

/// Minimum CUDA runtime capabilities encoded in output compatibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawCudaFirthCapabilityRequirementsCompatibility {
    cuda_driver_version: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
}

/// Validated requested and effective approximate-Firth implementation state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FirthComponentsCompatibility {
    requested: FirthComponentsImplementationCompatibility,
    effective: FirthComponentsImplementationCompatibility,
    fallback_reason: Option<FirthComponentsFallbackReasonCompatibility>,
    raw_cuda_artifact: Option<RawCudaFirthArtifactCompatibility>,
}

impl AssociationImplementationCompatibility {
    /// Build exact JAX runtime compatibility with optional Firth state.
    ///
    /// # Errors
    ///
    /// Returns an error when either runtime version is empty.
    pub fn new(
        jax_version: String,
        jaxlib_version: String,
        firth_components: Option<FirthComponentsCompatibility>,
    ) -> Result<Self, OutputError> {
        if jax_version.is_empty() {
            return Err(OutputError::InvalidInput(
                "Association implementation JAX version must not be empty.".to_string(),
            ));
        }
        if jaxlib_version.is_empty() {
            return Err(OutputError::InvalidInput(
                "Association implementation JAXlib version must not be empty.".to_string(),
            ));
        }
        Ok(Self { jax_version, jaxlib_version, firth_components })
    }

    /// Return the exact observed JAX version.
    #[must_use]
    pub fn jax_version(&self) -> &str {
        &self.jax_version
    }

    /// Return the exact observed `JAXlib` version.
    #[must_use]
    pub fn jaxlib_version(&self) -> &str {
        &self.jaxlib_version
    }

    /// Return approximate-Firth implementation state when it is applicable.
    #[must_use]
    pub const fn firth_components(&self) -> Option<&FirthComponentsCompatibility> {
        self.firth_components.as_ref()
    }
}

impl RawCudaFirthCapabilityRequirementsCompatibility {
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
                "Raw-CUDA Firth artifact minimum CUDA driver version must be positive.".to_string(),
            ));
        }
        if minimum_compute_capability_major <= 0 || minimum_compute_capability_minor < 0 {
            return Err(OutputError::InvalidInput(
                "Raw-CUDA Firth artifact minimum compute capability must have a positive major and nonnegative minor."
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

impl RawCudaFirthArtifactCompatibility {
    /// Build the exact raw-CUDA artifact identity requested by runtime policy.
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
        capability_requirements: RawCudaFirthCapabilityRequirementsCompatibility,
    ) -> Result<Self, OutputError> {
        if ffi_target.is_empty() {
            return Err(OutputError::InvalidInput("Raw-CUDA Firth artifact FFI target must not be empty.".to_string()));
        }
        if ffi_api_version == 0 {
            return Err(OutputError::InvalidInput(
                "Raw-CUDA Firth artifact FFI API version must be a positive integer.".to_string(),
            ));
        }
        if !crate::digest::is_canonical_sha256(&handler_sha256) {
            return Err(OutputError::InvalidInput(
                "Raw-CUDA Firth artifact handler SHA-256 must contain exactly 64 lowercase hexadecimal characters."
                    .to_string(),
            ));
        }
        if !crate::digest::is_canonical_sha256(&ptx_sha256) {
            return Err(OutputError::InvalidInput(
                "Raw-CUDA Firth artifact PTX SHA-256 must contain exactly 64 lowercase hexadecimal characters."
                    .to_string(),
            ));
        }
        if ptx_isa.is_empty() {
            return Err(OutputError::InvalidInput("Raw-CUDA Firth artifact PTX ISA must not be empty.".to_string()));
        }
        if ptx_target.is_empty() {
            return Err(OutputError::InvalidInput("Raw-CUDA Firth artifact PTX target must not be empty.".to_string()));
        }
        if !ptx_target_matches_compute_capability(
            &ptx_target,
            capability_requirements.minimum_compute_capability_major(),
            capability_requirements.minimum_compute_capability_minor(),
        ) {
            return Err(OutputError::InvalidInput(
                "Raw-CUDA Firth artifact PTX target must match its minimum compute capability.".to_string(),
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

    /// Return the stable typed-XLA FFI target.
    #[must_use]
    pub fn ffi_target(&self) -> &str {
        &self.ffi_target
    }

    /// Return the typed-XLA FFI API version.
    #[must_use]
    pub const fn ffi_api_version(&self) -> u32 {
        self.ffi_api_version
    }

    /// Return the framed native handler and ABI input SHA-256.
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

pub(crate) fn ptx_target_matches_compute_capability(target: &str, expected_major: i32, expected_minor: i32) -> bool {
    let Some(compute_capability) = target.strip_prefix("sm_") else {
        return false;
    };
    if compute_capability.len() < 2 || !compute_capability.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    let split_index = compute_capability.len() - 1;
    let Ok(major) = compute_capability[..split_index].parse::<i32>() else {
        return false;
    };
    let Ok(minor) = compute_capability[split_index..].parse::<i32>() else {
        return false;
    };
    major == expected_major && minor == expected_minor
}

impl FirthComponentsCompatibility {
    /// Record a requested and effective portable JAX implementation.
    #[must_use]
    pub const fn jax() -> Self {
        Self {
            requested: FirthComponentsImplementationCompatibility::Jax,
            effective: FirthComponentsImplementationCompatibility::Jax,
            fallback_reason: None,
            raw_cuda_artifact: None,
        }
    }

    /// Record a requested and effective raw-CUDA implementation.
    #[must_use]
    pub const fn raw_cuda(raw_cuda_artifact: RawCudaFirthArtifactCompatibility) -> Self {
        Self {
            requested: FirthComponentsImplementationCompatibility::RawCuda,
            effective: FirthComponentsImplementationCompatibility::RawCuda,
            fallback_reason: None,
            raw_cuda_artifact: Some(raw_cuda_artifact),
        }
    }

    /// Record a recoverable raw-CUDA request that uses portable JAX.
    #[must_use]
    pub const fn raw_cuda_fallback(
        raw_cuda_artifact: RawCudaFirthArtifactCompatibility,
        fallback_reason: FirthComponentsFallbackReasonCompatibility,
    ) -> Self {
        Self {
            requested: FirthComponentsImplementationCompatibility::RawCuda,
            effective: FirthComponentsImplementationCompatibility::Jax,
            fallback_reason: Some(fallback_reason),
            raw_cuda_artifact: Some(raw_cuda_artifact),
        }
    }

    /// Return the implementation requested by runtime policy.
    #[must_use]
    pub const fn requested(&self) -> FirthComponentsImplementationCompatibility {
        self.requested
    }

    /// Return the implementation selected for execution.
    #[must_use]
    pub const fn effective(&self) -> FirthComponentsImplementationCompatibility {
        self.effective
    }

    /// Return the stable recoverable fallback reason, when fallback occurred.
    #[must_use]
    pub const fn fallback_reason(&self) -> Option<FirthComponentsFallbackReasonCompatibility> {
        self.fallback_reason
    }

    /// Return raw-CUDA artifact identity when raw CUDA was requested.
    #[must_use]
    pub const fn raw_cuda_artifact(&self) -> Option<&RawCudaFirthArtifactCompatibility> {
        self.raw_cuda_artifact.as_ref()
    }
}
