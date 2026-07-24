//! Runtime-selected association implementation state.

/// Exact JAX runtime versions used by a production JAX backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeVersions {
    jax_version: String,
    jaxlib_version: String,
}

impl JaxRuntimeVersions {
    /// Build runtime-version state when both observed versions are nonempty.
    #[must_use]
    pub fn new(jax_version: String, jaxlib_version: String) -> Option<Self> {
        if jax_version.is_empty() || jaxlib_version.is_empty() {
            return None;
        }
        Some(Self { jax_version, jaxlib_version })
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
}

/// Concrete implementation used for approximate-Firth component reductions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FirthComponentsImplementation {
    /// Use the portable JAX reduction.
    Jax,
    /// Use the registered raw-CUDA typed-XLA FFI handler.
    RawCuda,
}

impl FirthComponentsImplementation {
    /// Return the stable output-facing implementation name.
    #[must_use]
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::Jax => "jax",
            Self::RawCuda => "raw_cuda",
        }
    }
}

/// Typed reason a requested raw-CUDA Firth implementation was unavailable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FirthComponentsFallbackReason {
    /// The platform cannot load the Linux CUDA ABI.
    UnsupportedPlatform,
    /// The CUDA driver shared object could not be loaded.
    CudaDriverUnavailable,
    /// A required CUDA driver symbol was unavailable.
    RequiredSymbolUnavailable,
    /// The installed driver cannot consume the embedded PTX.
    CudaDriverTooOld,
    /// The selected CUDA-visible device ordinal was unavailable.
    CudaDeviceUnavailable,
    /// The selected device cannot execute the embedded kernel.
    UnsupportedComputeCapability,
}

impl FirthComponentsFallbackReason {
    /// Return the stable output-facing fallback-reason name.
    #[must_use]
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::UnsupportedPlatform => "unsupported_platform",
            Self::CudaDriverUnavailable => "cuda_driver_unavailable",
            Self::RequiredSymbolUnavailable => "required_symbol_unavailable",
            Self::CudaDriverTooOld => "cuda_driver_too_old",
            Self::CudaDeviceUnavailable => "cuda_device_unavailable",
            Self::UnsupportedComputeCapability => "unsupported_compute_capability",
        }
    }
}

/// Stable identity of the raw-CUDA approximate-Firth artifact and XLA ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawCudaFirthArtifactIdentity {
    ffi_target: &'static str,
    ffi_api_version: u32,
    handler_sha256: &'static str,
    ptx_sha256: &'static str,
    ptx_isa: &'static str,
    ptx_target: &'static str,
    capability_requirements: RawCudaFirthCapabilityRequirements,
}

/// Minimum CUDA runtime capabilities required by a raw-CUDA Firth artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawCudaFirthCapabilityRequirements {
    cuda_driver_version: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
}

impl RawCudaFirthCapabilityRequirements {
    /// Build validated minimum CUDA runtime requirements.
    #[must_use]
    pub const fn new(
        minimum_cuda_driver_version: i32,
        minimum_compute_capability_major: i32,
        minimum_compute_capability_minor: i32,
    ) -> Option<Self> {
        if minimum_cuda_driver_version <= 0
            || minimum_compute_capability_major <= 0
            || minimum_compute_capability_minor < 0
        {
            return None;
        }
        Some(Self {
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

impl RawCudaFirthArtifactIdentity {
    /// Build a validated raw-CUDA artifact identity.
    #[must_use]
    pub const fn new(
        ffi_target: &'static str,
        ffi_api_version: u32,
        handler_sha256: &'static str,
        ptx_sha256: &'static str,
        ptx_isa: &'static str,
        ptx_target: &'static str,
        capability_requirements: RawCudaFirthCapabilityRequirements,
    ) -> Option<Self> {
        if ffi_target.is_empty()
            || ffi_api_version == 0
            || !is_canonical_sha256(handler_sha256)
            || !is_canonical_sha256(ptx_sha256)
            || ptx_isa.is_empty()
            || ptx_target.is_empty()
            || !ptx_target_matches_compute_capability(
                ptx_target,
                capability_requirements.minimum_compute_capability_major(),
                capability_requirements.minimum_compute_capability_minor(),
            )
        {
            return None;
        }
        Some(Self {
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
    pub const fn ffi_target(self) -> &'static str {
        self.ffi_target
    }

    /// Return the typed-XLA FFI API version.
    #[must_use]
    pub const fn ffi_api_version(self) -> u32 {
        self.ffi_api_version
    }

    /// Return the framed native handler and ABI input SHA-256.
    #[must_use]
    pub const fn handler_sha256(self) -> &'static str {
        self.handler_sha256
    }

    /// Return the canonical lower-case SHA-256 of the embedded PTX.
    #[must_use]
    pub const fn ptx_sha256(self) -> &'static str {
        self.ptx_sha256
    }

    /// Return the PTX ISA version used by the embedded artifact.
    #[must_use]
    pub const fn ptx_isa(self) -> &'static str {
        self.ptx_isa
    }

    /// Return the PTX target used by the embedded artifact.
    #[must_use]
    pub const fn ptx_target(self) -> &'static str {
        self.ptx_target
    }

    /// Return the reviewed minimum CUDA driver API version.
    #[must_use]
    pub const fn minimum_cuda_driver_version(self) -> i32 {
        self.capability_requirements.minimum_cuda_driver_version()
    }

    /// Return the embedded PTX target's minimum compute-capability major version.
    #[must_use]
    pub const fn minimum_compute_capability_major(self) -> i32 {
        self.capability_requirements.minimum_compute_capability_major()
    }

    /// Return the embedded PTX target's minimum compute-capability minor version.
    #[must_use]
    pub const fn minimum_compute_capability_minor(self) -> i32 {
        self.capability_requirements.minimum_compute_capability_minor()
    }
}

/// Diagnostic-only CUDA observations made while selecting the Firth backend.
///
/// These fields are emitted once per run but are deliberately excluded from
/// output compatibility and resume hashes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawCudaFirthRuntimeObservation {
    kind: RawCudaFirthRuntimeObservationKind,
    cuda_driver_version: Option<i32>,
    device_ordinal: Option<i32>,
    compute_capability_major: Option<i32>,
    compute_capability_minor: Option<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RawCudaFirthRuntimeObservationKind {
    Qualified,
    Fallback(FirthComponentsFallbackReason),
}

impl RawCudaFirthRuntimeObservation {
    /// Build complete observations from successful native qualification.
    #[must_use]
    pub const fn qualified(
        cuda_driver_version: i32,
        device_ordinal: i32,
        compute_capability_major: i32,
        compute_capability_minor: i32,
    ) -> Option<Self> {
        if cuda_driver_version <= 0
            || device_ordinal < 0
            || compute_capability_major <= 0
            || compute_capability_minor < 0
        {
            return None;
        }
        Some(Self {
            kind: RawCudaFirthRuntimeObservationKind::Qualified,
            cuda_driver_version: Some(cuda_driver_version),
            device_ordinal: Some(device_ordinal),
            compute_capability_major: Some(compute_capability_major),
            compute_capability_minor: Some(compute_capability_minor),
        })
    }

    /// Record that raw CUDA is unavailable on the current platform.
    #[must_use]
    pub const fn unsupported_platform() -> Self {
        Self::early_fallback(FirthComponentsFallbackReason::UnsupportedPlatform)
    }

    /// Record that the CUDA driver shared object is unavailable.
    #[must_use]
    pub const fn cuda_driver_unavailable() -> Self {
        Self::early_fallback(FirthComponentsFallbackReason::CudaDriverUnavailable)
    }

    /// Record that a required CUDA driver symbol is unavailable.
    #[must_use]
    pub const fn required_symbol_unavailable() -> Self {
        Self::early_fallback(FirthComponentsFallbackReason::RequiredSymbolUnavailable)
    }

    /// Record a CUDA driver that is too old for the embedded PTX.
    #[must_use]
    pub const fn cuda_driver_too_old(cuda_driver_version: i32) -> Option<Self> {
        if cuda_driver_version <= 0 {
            return None;
        }
        Some(Self {
            kind: RawCudaFirthRuntimeObservationKind::Fallback(FirthComponentsFallbackReason::CudaDriverTooOld),
            cuda_driver_version: Some(cuda_driver_version),
            device_ordinal: None,
            compute_capability_major: None,
            compute_capability_minor: None,
        })
    }

    /// Record an unavailable device after successful driver qualification.
    #[must_use]
    pub const fn cuda_device_unavailable(cuda_driver_version: i32, device_ordinal: i32) -> Option<Self> {
        if cuda_driver_version <= 0 || device_ordinal < 0 {
            return None;
        }
        Some(Self {
            kind: RawCudaFirthRuntimeObservationKind::Fallback(FirthComponentsFallbackReason::CudaDeviceUnavailable),
            cuda_driver_version: Some(cuda_driver_version),
            device_ordinal: Some(device_ordinal),
            compute_capability_major: None,
            compute_capability_minor: None,
        })
    }

    /// Record a device whose compute capability cannot execute the PTX.
    #[must_use]
    pub const fn unsupported_compute_capability(
        cuda_driver_version: i32,
        device_ordinal: i32,
        compute_capability_major: i32,
        compute_capability_minor: i32,
    ) -> Option<Self> {
        if cuda_driver_version <= 0
            || device_ordinal < 0
            || compute_capability_major <= 0
            || compute_capability_minor < 0
        {
            return None;
        }
        Some(Self {
            kind: RawCudaFirthRuntimeObservationKind::Fallback(
                FirthComponentsFallbackReason::UnsupportedComputeCapability,
            ),
            cuda_driver_version: Some(cuda_driver_version),
            device_ordinal: Some(device_ordinal),
            compute_capability_major: Some(compute_capability_major),
            compute_capability_minor: Some(compute_capability_minor),
        })
    }

    /// Return the observed CUDA driver API version.
    #[must_use]
    pub const fn cuda_driver_version(self) -> Option<i32> {
        self.cuda_driver_version
    }

    /// Return the selected CUDA-visible device ordinal.
    #[must_use]
    pub const fn device_ordinal(self) -> Option<i32> {
        self.device_ordinal
    }

    /// Return the selected device compute-capability major version.
    #[must_use]
    pub const fn compute_capability_major(self) -> Option<i32> {
        self.compute_capability_major
    }

    /// Return the selected device compute-capability minor version.
    #[must_use]
    pub const fn compute_capability_minor(self) -> Option<i32> {
        self.compute_capability_minor
    }

    const fn is_qualified(self) -> bool {
        matches!(self.kind, RawCudaFirthRuntimeObservationKind::Qualified)
    }

    const fn is_qualified_for(self, artifact: RawCudaFirthArtifactIdentity) -> bool {
        self.is_qualified()
            && matches!(
                (
                    self.cuda_driver_version,
                    self.compute_capability_major,
                    self.compute_capability_minor,
                ),
                (Some(driver), Some(major), Some(minor))
                    if driver >= artifact.minimum_cuda_driver_version()
                        && compute_capability_is_at_least(
                            major,
                            minor,
                            artifact.minimum_compute_capability_major(),
                            artifact.minimum_compute_capability_minor(),
                        )
            )
    }

    fn is_fallback_for(self, reason: FirthComponentsFallbackReason) -> bool {
        matches!(self.kind, RawCudaFirthRuntimeObservationKind::Fallback(observed) if observed == reason)
    }

    fn is_fallback_for_artifact(
        self,
        reason: FirthComponentsFallbackReason,
        artifact: RawCudaFirthArtifactIdentity,
    ) -> bool {
        if !self.is_fallback_for(reason) {
            return false;
        }
        match reason {
            FirthComponentsFallbackReason::UnsupportedPlatform
            | FirthComponentsFallbackReason::CudaDriverUnavailable
            | FirthComponentsFallbackReason::RequiredSymbolUnavailable => true,
            FirthComponentsFallbackReason::CudaDriverTooOld => {
                self.cuda_driver_version.is_some_and(|driver| driver < artifact.minimum_cuda_driver_version())
            }
            FirthComponentsFallbackReason::CudaDeviceUnavailable => {
                self.cuda_driver_version.is_some_and(|driver| driver >= artifact.minimum_cuda_driver_version())
            }
            FirthComponentsFallbackReason::UnsupportedComputeCapability => {
                matches!(
                    (
                        self.cuda_driver_version,
                        self.compute_capability_major,
                        self.compute_capability_minor,
                    ),
                    (Some(driver), Some(major), Some(minor))
                        if driver >= artifact.minimum_cuda_driver_version()
                            && !compute_capability_is_at_least(
                                major,
                                minor,
                                artifact.minimum_compute_capability_major(),
                                artifact.minimum_compute_capability_minor(),
                            )
                )
            }
        }
    }

    const fn early_fallback(reason: FirthComponentsFallbackReason) -> Self {
        Self {
            kind: RawCudaFirthRuntimeObservationKind::Fallback(reason),
            cuda_driver_version: None,
            device_ordinal: None,
            compute_capability_major: None,
            compute_capability_minor: None,
        }
    }
}

const fn compute_capability_is_at_least(major: i32, minor: i32, minimum_major: i32, minimum_minor: i32) -> bool {
    major > minimum_major || (major == minimum_major && minor >= minimum_minor)
}

const fn ptx_target_matches_compute_capability(target: &str, expected_major: i32, expected_minor: i32) -> bool {
    let bytes = target.as_bytes();
    if bytes.len() < 5 || bytes[0] != b's' || bytes[1] != b'm' || bytes[2] != b'_' {
        return false;
    }
    let final_index = bytes.len() - 1;
    let minor_byte = bytes[final_index];
    let Some(minor) = ascii_decimal_digit_value(minor_byte) else {
        return false;
    };
    let mut major = 0_i32;
    let mut index = 3;
    while index < final_index {
        let byte = bytes[index];
        let Some(digit) = ascii_decimal_digit_value(byte) else {
            return false;
        };
        if major > (i32::MAX - digit) / 10 {
            return false;
        }
        major = major * 10 + digit;
        index += 1;
    }
    major == expected_major && minor == expected_minor
}

const fn ascii_decimal_digit_value(byte: u8) -> Option<i32> {
    match byte {
        b'0' => Some(0),
        b'1' => Some(1),
        b'2' => Some(2),
        b'3' => Some(3),
        b'4' => Some(4),
        b'5' => Some(5),
        b'6' => Some(6),
        b'7' => Some(7),
        b'8' => Some(8),
        b'9' => Some(9),
        _ => None,
    }
}

const fn is_canonical_sha256(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 64 {
        return false;
    }
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if !((byte >= b'0' && byte <= b'9') || (byte >= b'a' && byte <= b'f')) {
            return false;
        }
        index += 1;
    }
    true
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum FirthComponentsSelection {
    Jax,
    RawCuda {
        artifact: RawCudaFirthArtifactIdentity,
        observation: RawCudaFirthRuntimeObservation,
    },
    RawCudaFallback {
        artifact: RawCudaFirthArtifactIdentity,
        observation: RawCudaFirthRuntimeObservation,
        reason: FirthComponentsFallbackReason,
        detail: String,
    },
}

/// Validated runtime selection for approximate-Firth component reductions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FirthComponentsImplementationState {
    selection: FirthComponentsSelection,
}

impl FirthComponentsImplementationState {
    /// Record a requested and effective portable JAX implementation.
    #[must_use]
    pub const fn jax() -> Self {
        Self { selection: FirthComponentsSelection::Jax }
    }

    /// Record a requested and effective raw-CUDA implementation.
    ///
    #[must_use]
    pub const fn raw_cuda(
        artifact: RawCudaFirthArtifactIdentity,
        observation: RawCudaFirthRuntimeObservation,
    ) -> Option<Self> {
        if !observation.is_qualified_for(artifact) {
            return None;
        }
        Some(Self { selection: FirthComponentsSelection::RawCuda { artifact, observation } })
    }

    /// Record raw CUDA falling back to the portable JAX implementation.
    #[must_use]
    pub fn raw_cuda_fallback(
        artifact: RawCudaFirthArtifactIdentity,
        observation: RawCudaFirthRuntimeObservation,
        reason: FirthComponentsFallbackReason,
        detail: String,
    ) -> Option<Self> {
        if !observation.is_fallback_for_artifact(reason, artifact) {
            return None;
        }
        Some(Self { selection: FirthComponentsSelection::RawCudaFallback { artifact, observation, reason, detail } })
    }

    /// Return the implementation requested by runtime policy.
    #[must_use]
    pub const fn requested(&self) -> FirthComponentsImplementation {
        match &self.selection {
            FirthComponentsSelection::Jax => FirthComponentsImplementation::Jax,
            FirthComponentsSelection::RawCuda { .. } | FirthComponentsSelection::RawCudaFallback { .. } => {
                FirthComponentsImplementation::RawCuda
            }
        }
    }

    /// Return the implementation selected for execution.
    #[must_use]
    pub const fn effective(&self) -> FirthComponentsImplementation {
        match &self.selection {
            FirthComponentsSelection::Jax | FirthComponentsSelection::RawCudaFallback { .. } => {
                FirthComponentsImplementation::Jax
            }
            FirthComponentsSelection::RawCuda { .. } => FirthComponentsImplementation::RawCuda,
        }
    }

    /// Return the exact typed-XLA FFI target when raw CUDA is effective.
    #[must_use]
    pub const fn ffi_target(&self) -> Option<&'static str> {
        match &self.selection {
            FirthComponentsSelection::RawCuda { artifact, .. } => Some(artifact.ffi_target()),
            FirthComponentsSelection::Jax | FirthComponentsSelection::RawCudaFallback { .. } => None,
        }
    }

    /// Return raw-CUDA artifact identity whenever raw CUDA was requested.
    #[must_use]
    pub const fn raw_cuda_artifact(&self) -> Option<RawCudaFirthArtifactIdentity> {
        match &self.selection {
            FirthComponentsSelection::Jax => None,
            FirthComponentsSelection::RawCuda { artifact, .. }
            | FirthComponentsSelection::RawCudaFallback { artifact, .. } => Some(*artifact),
        }
    }

    /// Return diagnostic-only CUDA observations whenever raw CUDA was requested.
    #[must_use]
    pub const fn raw_cuda_observation(&self) -> Option<RawCudaFirthRuntimeObservation> {
        match &self.selection {
            FirthComponentsSelection::Jax => None,
            FirthComponentsSelection::RawCuda { observation, .. }
            | FirthComponentsSelection::RawCudaFallback { observation, .. } => Some(*observation),
        }
    }

    /// Return the stable typed fallback reason, when fallback occurred.
    #[must_use]
    pub const fn fallback_reason(&self) -> Option<FirthComponentsFallbackReason> {
        match &self.selection {
            FirthComponentsSelection::RawCudaFallback { reason, .. } => Some(*reason),
            FirthComponentsSelection::Jax | FirthComponentsSelection::RawCuda { .. } => None,
        }
    }

    /// Return diagnostic-only fallback detail.
    ///
    /// This free text must not participate in compatibility or resume hashes.
    #[must_use]
    pub fn fallback_detail(&self) -> Option<&str> {
        match &self.selection {
            FirthComponentsSelection::RawCudaFallback { detail, .. } => Some(detail),
            FirthComponentsSelection::Jax | FirthComponentsSelection::RawCuda { .. } => None,
        }
    }
}

/// Runtime-selected association implementations that affect reproducibility.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssociationImplementationState {
    runtime_versions: JaxRuntimeVersions,
    firth_components: Option<FirthComponentsImplementationState>,
}

impl AssociationImplementationState {
    /// Record a production JAX backend and its optional Firth implementation.
    #[must_use]
    pub fn jax(
        runtime_versions: JaxRuntimeVersions,
        firth_components: Option<FirthComponentsImplementationState>,
    ) -> Self {
        Self { runtime_versions, firth_components }
    }

    /// Return the exact JAX runtime versions.
    #[must_use]
    pub const fn jax_runtime_versions(&self) -> &JaxRuntimeVersions {
        &self.runtime_versions
    }

    /// Return the approximate-Firth implementation state, when applicable.
    #[must_use]
    pub const fn firth_components(&self) -> Option<&FirthComponentsImplementationState> {
        self.firth_components.as_ref()
    }

    /// Project runtime selection into the output-owned resume contract.
    ///
    /// # Errors
    ///
    /// Returns an error when the output contract rejects the projection.
    pub fn output_compatibility(
        &self,
    ) -> Result<g_output::AssociationImplementationCompatibility, g_output::OutputError> {
        let firth_components =
            self.firth_components.as_ref().map(FirthComponentsImplementationState::output_compatibility).transpose()?;
        g_output::AssociationImplementationCompatibility::new(
            self.runtime_versions.jax_version().to_owned(),
            self.runtime_versions.jaxlib_version().to_owned(),
            firth_components,
        )
    }
}

impl FirthComponentsImplementationState {
    fn output_compatibility(&self) -> Result<g_output::FirthComponentsCompatibility, g_output::OutputError> {
        match &self.selection {
            FirthComponentsSelection::Jax => Ok(g_output::FirthComponentsCompatibility::jax()),
            FirthComponentsSelection::RawCuda { artifact, .. } => {
                Ok(g_output::FirthComponentsCompatibility::raw_cuda(output_raw_cuda_artifact(*artifact)?))
            }
            FirthComponentsSelection::RawCudaFallback { artifact, reason, .. } => {
                Ok(g_output::FirthComponentsCompatibility::raw_cuda_fallback(
                    output_raw_cuda_artifact(*artifact)?,
                    output_fallback_reason(*reason),
                ))
            }
        }
    }
}

fn output_raw_cuda_artifact(
    artifact: RawCudaFirthArtifactIdentity,
) -> Result<g_output::RawCudaFirthArtifactCompatibility, g_output::OutputError> {
    let capability_requirements = g_output::RawCudaFirthCapabilityRequirementsCompatibility::new(
        artifact.minimum_cuda_driver_version(),
        artifact.minimum_compute_capability_major(),
        artifact.minimum_compute_capability_minor(),
    )?;
    g_output::RawCudaFirthArtifactCompatibility::new(
        artifact.ffi_target().to_owned(),
        artifact.ffi_api_version(),
        artifact.handler_sha256().to_owned(),
        artifact.ptx_sha256().to_owned(),
        artifact.ptx_isa().to_owned(),
        artifact.ptx_target().to_owned(),
        capability_requirements,
    )
}

const fn output_fallback_reason(
    reason: FirthComponentsFallbackReason,
) -> g_output::FirthComponentsFallbackReasonCompatibility {
    match reason {
        FirthComponentsFallbackReason::UnsupportedPlatform => {
            g_output::FirthComponentsFallbackReasonCompatibility::UnsupportedPlatform
        }
        FirthComponentsFallbackReason::CudaDriverUnavailable => {
            g_output::FirthComponentsFallbackReasonCompatibility::CudaDriverUnavailable
        }
        FirthComponentsFallbackReason::RequiredSymbolUnavailable => {
            g_output::FirthComponentsFallbackReasonCompatibility::RequiredSymbolUnavailable
        }
        FirthComponentsFallbackReason::CudaDriverTooOld => {
            g_output::FirthComponentsFallbackReasonCompatibility::CudaDriverTooOld
        }
        FirthComponentsFallbackReason::CudaDeviceUnavailable => {
            g_output::FirthComponentsFallbackReasonCompatibility::CudaDeviceUnavailable
        }
        FirthComponentsFallbackReason::UnsupportedComputeCapability => {
            g_output::FirthComponentsFallbackReasonCompatibility::UnsupportedComputeCapability
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAW_CUDA_TARGET: &str = "g.firth.components.test.v0";
    const RAW_CUDA_HANDLER_SHA256: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const RAW_CUDA_PTX_SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const RAW_CUDA_REQUIREMENTS: RawCudaFirthCapabilityRequirements =
        RawCudaFirthCapabilityRequirements::new(12_020, 7, 0)
            .expect("the test raw-CUDA capability requirements are valid");
    const RAW_CUDA_ARTIFACT: RawCudaFirthArtifactIdentity = RawCudaFirthArtifactIdentity::new(
        RAW_CUDA_TARGET,
        1,
        RAW_CUDA_HANDLER_SHA256,
        RAW_CUDA_PTX_SHA256,
        "8.2",
        "sm_70",
        RAW_CUDA_REQUIREMENTS,
    )
    .expect("the test raw-CUDA artifact identity is valid");

    fn qualified_observation() -> RawCudaFirthRuntimeObservation {
        RawCudaFirthRuntimeObservation::qualified(12_090, 2, 8, 0)
            .expect("the test qualification observations are valid")
    }

    fn fallback_observation(reason: FirthComponentsFallbackReason) -> RawCudaFirthRuntimeObservation {
        match reason {
            FirthComponentsFallbackReason::UnsupportedPlatform => {
                RawCudaFirthRuntimeObservation::unsupported_platform()
            }
            FirthComponentsFallbackReason::CudaDriverUnavailable => {
                RawCudaFirthRuntimeObservation::cuda_driver_unavailable()
            }
            FirthComponentsFallbackReason::RequiredSymbolUnavailable => {
                RawCudaFirthRuntimeObservation::required_symbol_unavailable()
            }
            FirthComponentsFallbackReason::CudaDriverTooOld => {
                RawCudaFirthRuntimeObservation::cuda_driver_too_old(12_010).expect("the test driver version is valid")
            }
            FirthComponentsFallbackReason::CudaDeviceUnavailable => {
                RawCudaFirthRuntimeObservation::cuda_device_unavailable(12_090, 2)
                    .expect("the test driver and ordinal are valid")
            }
            FirthComponentsFallbackReason::UnsupportedComputeCapability => {
                RawCudaFirthRuntimeObservation::unsupported_compute_capability(12_090, 2, 6, 1)
                    .expect("the test driver, ordinal, and capability are valid")
            }
        }
    }

    #[test]
    fn runtime_versions_require_both_exact_values() {
        let versions = JaxRuntimeVersions::new("0.11.0".to_owned(), "0.11.0".to_owned()).expect("nonempty versions");
        assert_eq!(versions.jax_version(), "0.11.0");
        assert_eq!(versions.jaxlib_version(), "0.11.0");
        assert_eq!(JaxRuntimeVersions::new(String::new(), "0.11.0".to_owned()), None);
        assert_eq!(JaxRuntimeVersions::new("0.11.0".to_owned(), String::new()), None);
    }

    #[test]
    fn firth_states_admit_only_valid_target_and_fallback_combinations() {
        let jax = FirthComponentsImplementationState::jax();
        assert_eq!(jax.requested(), FirthComponentsImplementation::Jax);
        assert_eq!(jax.effective(), FirthComponentsImplementation::Jax);
        assert_eq!(jax.ffi_target(), None);
        assert_eq!(jax.raw_cuda_artifact(), None);
        assert_eq!(jax.fallback_reason(), None);

        let raw_cuda = FirthComponentsImplementationState::raw_cuda(RAW_CUDA_ARTIFACT, qualified_observation())
            .expect("qualified observations admit raw CUDA");
        assert_eq!(raw_cuda.requested(), FirthComponentsImplementation::RawCuda);
        assert_eq!(raw_cuda.effective(), FirthComponentsImplementation::RawCuda);
        assert_eq!(raw_cuda.ffi_target(), Some(RAW_CUDA_TARGET));
        assert_eq!(raw_cuda.raw_cuda_artifact(), Some(RAW_CUDA_ARTIFACT));
        assert_eq!(raw_cuda.fallback_reason(), None);

        let fallback = FirthComponentsImplementationState::raw_cuda_fallback(
            RAW_CUDA_ARTIFACT,
            fallback_observation(FirthComponentsFallbackReason::CudaDriverTooOld),
            FirthComponentsFallbackReason::CudaDriverTooOld,
            "driver predates required PTX support".to_owned(),
        )
        .expect("reason-matched fallback observations are valid");
        assert_eq!(fallback.requested(), FirthComponentsImplementation::RawCuda);
        assert_eq!(fallback.effective(), FirthComponentsImplementation::Jax);
        assert_eq!(fallback.ffi_target(), None);
        assert_eq!(fallback.raw_cuda_artifact(), Some(RAW_CUDA_ARTIFACT));
        assert_eq!(fallback.fallback_reason(), Some(FirthComponentsFallbackReason::CudaDriverTooOld));
    }

    #[test]
    fn raw_cuda_artifact_identity_rejects_ambiguous_values() {
        let identity = |target, api_version, handler_sha256, ptx_sha256, ptx_isa, ptx_target, requirements| {
            RawCudaFirthArtifactIdentity::new(
                target,
                api_version,
                handler_sha256,
                ptx_sha256,
                ptx_isa,
                ptx_target,
                requirements,
            )
        };
        let valid = || {
            identity(
                RAW_CUDA_TARGET,
                1,
                RAW_CUDA_HANDLER_SHA256,
                RAW_CUDA_PTX_SHA256,
                "8.2",
                "sm_70",
                RAW_CUDA_REQUIREMENTS,
            )
        };
        assert!(valid().is_some());
        assert_eq!(
            identity("", 1, RAW_CUDA_HANDLER_SHA256, RAW_CUDA_PTX_SHA256, "8.2", "sm_70", RAW_CUDA_REQUIREMENTS),
            None
        );
        assert_eq!(
            identity(
                RAW_CUDA_TARGET,
                0,
                RAW_CUDA_HANDLER_SHA256,
                RAW_CUDA_PTX_SHA256,
                "8.2",
                "sm_70",
                RAW_CUDA_REQUIREMENTS
            ),
            None
        );
        assert_eq!(
            identity(RAW_CUDA_TARGET, 1, "abcd", RAW_CUDA_PTX_SHA256, "8.2", "sm_70", RAW_CUDA_REQUIREMENTS),
            None
        );
        assert_eq!(
            identity(
                RAW_CUDA_TARGET,
                1,
                RAW_CUDA_HANDLER_SHA256,
                "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef",
                "8.2",
                "sm_70",
                RAW_CUDA_REQUIREMENTS,
            ),
            None
        );
        let mismatched_requirements =
            RawCudaFirthCapabilityRequirements::new(12_020, 8, 0).expect("mismatched requirements are well formed");
        assert_eq!(
            identity(
                RAW_CUDA_TARGET,
                1,
                RAW_CUDA_HANDLER_SHA256,
                RAW_CUDA_PTX_SHA256,
                "8.2",
                "sm_70",
                mismatched_requirements,
            ),
            None
        );
        assert_eq!(
            identity(
                RAW_CUDA_TARGET,
                1,
                RAW_CUDA_HANDLER_SHA256,
                RAW_CUDA_PTX_SHA256,
                "",
                "sm_70",
                RAW_CUDA_REQUIREMENTS
            ),
            None
        );
        assert_eq!(
            identity(
                RAW_CUDA_TARGET,
                1,
                RAW_CUDA_HANDLER_SHA256,
                RAW_CUDA_PTX_SHA256,
                "8.2",
                "",
                RAW_CUDA_REQUIREMENTS
            ),
            None
        );
        assert_eq!(RawCudaFirthCapabilityRequirements::new(0, 7, 0), None);
        assert_eq!(RawCudaFirthCapabilityRequirements::new(12_020, 0, 0), None);
        assert_eq!(RawCudaFirthCapabilityRequirements::new(12_020, 7, -1), None);
    }

    #[test]
    fn production_jax_state_always_retains_versions() {
        let versions = JaxRuntimeVersions::new("0.11.0".to_owned(), "0.11.0".to_owned()).expect("nonempty versions");
        let state = AssociationImplementationState::jax(versions, None);

        let observed = state.jax_runtime_versions();
        assert_eq!(observed.jax_version(), "0.11.0");
        assert_eq!(observed.jaxlib_version(), "0.11.0");
        assert_eq!(state.firth_components(), None);
        let compatibility = state.output_compatibility().expect("production state projects to output compatibility");
        assert_eq!(compatibility.jax_version(), "0.11.0");
        assert_eq!(compatibility.jaxlib_version(), "0.11.0");
        assert_eq!(compatibility.firth_components(), None);
    }

    #[test]
    fn free_text_detail_is_separate_from_stable_projection() {
        let firth = FirthComponentsImplementationState::raw_cuda_fallback(
            RAW_CUDA_ARTIFACT,
            fallback_observation(FirthComponentsFallbackReason::CudaDriverUnavailable),
            FirthComponentsFallbackReason::CudaDriverUnavailable,
            "host-specific loader detail".to_owned(),
        )
        .expect("reason-matched fallback observations are valid");
        let versions = JaxRuntimeVersions::new("0.11.0".to_owned(), "0.11.0".to_owned()).expect("nonempty versions");
        let state = AssociationImplementationState::jax(versions, Some(firth));
        let firth = state.firth_components().expect("the test state includes Firth components");

        assert_eq!(firth.requested().stable_name(), "raw_cuda");
        assert_eq!(firth.effective().stable_name(), "jax");
        assert_eq!(
            firth.fallback_reason().map(FirthComponentsFallbackReason::stable_name),
            Some("cuda_driver_unavailable")
        );
        assert_eq!(firth.ffi_target(), None);
        assert_eq!(firth.raw_cuda_artifact(), Some(RAW_CUDA_ARTIFACT));
        assert_eq!(firth.fallback_detail(), Some("host-specific loader detail"));

        let compatibility = state.output_compatibility().expect("stable selection projects to output compatibility");
        let compatibility = compatibility.firth_components().expect("Firth compatibility is present");
        assert_eq!(compatibility.requested(), g_output::FirthComponentsImplementationCompatibility::RawCuda);
        assert_eq!(compatibility.effective(), g_output::FirthComponentsImplementationCompatibility::Jax);
        assert_eq!(
            compatibility.fallback_reason(),
            Some(g_output::FirthComponentsFallbackReasonCompatibility::CudaDriverUnavailable)
        );
        let artifact = compatibility.raw_cuda_artifact().expect("raw-CUDA request retains artifact identity");
        assert_eq!(artifact.ffi_target(), RAW_CUDA_TARGET);
        assert_eq!(artifact.ffi_api_version(), 1);
        assert_eq!(artifact.handler_sha256(), RAW_CUDA_HANDLER_SHA256);
        assert_eq!(artifact.ptx_sha256(), RAW_CUDA_PTX_SHA256);
    }

    #[test]
    fn stable_projection_names_are_explicit() {
        assert_eq!(FirthComponentsImplementation::Jax.stable_name(), "jax");
        assert_eq!(FirthComponentsImplementation::RawCuda.stable_name(), "raw_cuda");
        assert_eq!(
            FirthComponentsFallbackReason::UnsupportedComputeCapability.stable_name(),
            "unsupported_compute_capability"
        );
    }

    #[test]
    fn raw_cuda_runtime_observations_are_causal_and_not_persisted() {
        let observation = qualified_observation();
        assert_eq!(observation.cuda_driver_version(), Some(12_090));
        assert_eq!(observation.device_ordinal(), Some(2));
        assert_eq!(observation.compute_capability_major(), Some(8));
        assert_eq!(observation.compute_capability_minor(), Some(0));
        assert_eq!(RawCudaFirthRuntimeObservation::qualified(0, 2, 8, 0), None);
        assert_eq!(RawCudaFirthRuntimeObservation::qualified(12_090, -1, 8, 0), None);
        assert_eq!(RawCudaFirthRuntimeObservation::cuda_driver_too_old(0), None);
        assert_eq!(RawCudaFirthRuntimeObservation::cuda_device_unavailable(12_090, -1), None);
        assert_eq!(RawCudaFirthRuntimeObservation::unsupported_compute_capability(12_090, 2, 0, 1), None);

        let state = FirthComponentsImplementationState::raw_cuda(RAW_CUDA_ARTIFACT, observation)
            .expect("qualified observations admit raw CUDA");
        assert_eq!(state.raw_cuda_observation(), Some(observation));
        let compatibility = AssociationImplementationState::jax(
            JaxRuntimeVersions::new("0.11.0".to_owned(), "0.11.0".to_owned()).expect("valid versions"),
            Some(state),
        )
        .output_compatibility()
        .expect("observations do not prevent output projection");
        let artifact = compatibility
            .firth_components()
            .and_then(g_output::FirthComponentsCompatibility::raw_cuda_artifact)
            .expect("the compatibility retains artifact identity");
        assert_eq!(artifact.ffi_target(), RAW_CUDA_TARGET);
    }

    #[test]
    fn selection_constructors_reject_mismatched_observation_kinds() {
        let fallback = fallback_observation(FirthComponentsFallbackReason::CudaDriverTooOld);
        assert_eq!(FirthComponentsImplementationState::raw_cuda(RAW_CUDA_ARTIFACT, fallback), None);
        assert_eq!(
            FirthComponentsImplementationState::raw_cuda_fallback(
                RAW_CUDA_ARTIFACT,
                qualified_observation(),
                FirthComponentsFallbackReason::CudaDriverTooOld,
                "driver predates required PTX support".to_owned(),
            ),
            None
        );
        assert_eq!(
            FirthComponentsImplementationState::raw_cuda_fallback(
                RAW_CUDA_ARTIFACT,
                fallback,
                FirthComponentsFallbackReason::CudaDriverUnavailable,
                "wrong reason".to_owned(),
            ),
            None
        );
        assert_eq!(
            FirthComponentsImplementationState::raw_cuda_fallback(
                RAW_CUDA_ARTIFACT,
                fallback,
                FirthComponentsFallbackReason::CudaDriverTooOld,
                String::new(),
            ),
            Some(FirthComponentsImplementationState {
                selection: FirthComponentsSelection::RawCudaFallback {
                    artifact: RAW_CUDA_ARTIFACT,
                    observation: fallback,
                    reason: FirthComponentsFallbackReason::CudaDriverTooOld,
                    detail: String::new(),
                },
            })
        );
    }

    #[test]
    fn selection_validates_observations_against_artifact_thresholds() {
        let old_driver_qualification =
            RawCudaFirthRuntimeObservation::qualified(12_010, 2, 8, 0).expect("positive observations are shaped");
        assert_eq!(FirthComponentsImplementationState::raw_cuda(RAW_CUDA_ARTIFACT, old_driver_qualification), None);
        let old_compute_capability =
            RawCudaFirthRuntimeObservation::qualified(12_090, 2, 6, 1).expect("positive observations are shaped");
        assert_eq!(FirthComponentsImplementationState::raw_cuda(RAW_CUDA_ARTIFACT, old_compute_capability), None);

        let current_driver_reported_as_old =
            RawCudaFirthRuntimeObservation::cuda_driver_too_old(12_020).expect("positive driver is shaped");
        assert_eq!(
            FirthComponentsImplementationState::raw_cuda_fallback(
                RAW_CUDA_ARTIFACT,
                current_driver_reported_as_old,
                FirthComponentsFallbackReason::CudaDriverTooOld,
                "inconsistent threshold".to_owned(),
            ),
            None
        );
        let old_driver_device_failure = RawCudaFirthRuntimeObservation::cuda_device_unavailable(12_010, 2)
            .expect("positive device observations are shaped");
        assert_eq!(
            FirthComponentsImplementationState::raw_cuda_fallback(
                RAW_CUDA_ARTIFACT,
                old_driver_device_failure,
                FirthComponentsFallbackReason::CudaDeviceUnavailable,
                "inconsistent gate order".to_owned(),
            ),
            None
        );
        let supported_compute_capability =
            RawCudaFirthRuntimeObservation::unsupported_compute_capability(12_090, 2, 7, 0)
                .expect("positive capability observations are shaped");
        assert_eq!(
            FirthComponentsImplementationState::raw_cuda_fallback(
                RAW_CUDA_ARTIFACT,
                supported_compute_capability,
                FirthComponentsFallbackReason::UnsupportedComputeCapability,
                "inconsistent capability threshold".to_owned(),
            ),
            None
        );
    }
}
