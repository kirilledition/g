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
    /// CUDA driver initialization or device inspection failed.
    CudaDriverFailure,
    /// The installed driver cannot consume the embedded PTX.
    CudaDriverTooOld,
    /// The selected CUDA-visible device ordinal was unavailable.
    CudaDeviceUnavailable,
    /// The selected device cannot execute the embedded kernel.
    UnsupportedComputeCapability,
    /// Native initialization failed outside the expected capability taxonomy.
    NativeInitializationFailure,
    /// JAX rejected registration of the native typed-XLA target.
    JaxRegistrationFailure,
}

impl FirthComponentsFallbackReason {
    /// Return the stable output-facing fallback-reason name.
    #[must_use]
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::UnsupportedPlatform => "unsupported_platform",
            Self::CudaDriverUnavailable => "cuda_driver_unavailable",
            Self::RequiredSymbolUnavailable => "required_symbol_unavailable",
            Self::CudaDriverFailure => "cuda_driver_failure",
            Self::CudaDriverTooOld => "cuda_driver_too_old",
            Self::CudaDeviceUnavailable => "cuda_device_unavailable",
            Self::UnsupportedComputeCapability => "unsupported_compute_capability",
            Self::NativeInitializationFailure => "native_initialization_failure",
            Self::JaxRegistrationFailure => "jax_registration_failure",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum FirthComponentsSelection {
    Jax,
    RawCuda { ffi_target: &'static str },
    RawCudaFallback { reason: FirthComponentsFallbackReason, detail: String },
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
    /// An empty target is rejected so effective raw CUDA always carries the
    /// exact ABI target registered with JAX.
    #[must_use]
    pub const fn raw_cuda(ffi_target: &'static str) -> Option<Self> {
        if ffi_target.is_empty() {
            return None;
        }
        Some(Self { selection: FirthComponentsSelection::RawCuda { ffi_target } })
    }

    /// Record raw CUDA falling back to the portable JAX implementation.
    #[must_use]
    pub fn raw_cuda_fallback(reason: FirthComponentsFallbackReason, detail: String) -> Self {
        Self { selection: FirthComponentsSelection::RawCudaFallback { reason, detail } }
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
            FirthComponentsSelection::RawCuda { ffi_target } => Some(*ffi_target),
            FirthComponentsSelection::Jax | FirthComponentsSelection::RawCudaFallback { .. } => None,
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

#[derive(Clone, Debug, Eq, PartialEq)]
enum AssociationImplementationSelection {
    NotApplicable,
    Jax { runtime_versions: JaxRuntimeVersions, firth_components: Option<FirthComponentsImplementationState> },
}

/// Runtime-selected association implementations that affect reproducibility.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssociationImplementationState {
    selection: AssociationImplementationSelection,
}

impl AssociationImplementationState {
    /// Record a production JAX backend and its optional Firth implementation.
    #[must_use]
    pub fn jax(
        runtime_versions: JaxRuntimeVersions,
        firth_components: Option<FirthComponentsImplementationState>,
    ) -> Self {
        Self { selection: AssociationImplementationSelection::Jax { runtime_versions, firth_components } }
    }

    /// Return the exact JAX runtime versions, when the backend is JAX-backed.
    #[must_use]
    pub const fn jax_runtime_versions(&self) -> Option<&JaxRuntimeVersions> {
        match &self.selection {
            AssociationImplementationSelection::Jax { runtime_versions, .. } => Some(runtime_versions),
            AssociationImplementationSelection::NotApplicable => None,
        }
    }

    /// Return the approximate-Firth implementation state, when applicable.
    #[must_use]
    pub const fn firth_components(&self) -> Option<&FirthComponentsImplementationState> {
        match &self.selection {
            AssociationImplementationSelection::Jax { firth_components, .. } => firth_components.as_ref(),
            AssociationImplementationSelection::NotApplicable => None,
        }
    }
}

impl Default for AssociationImplementationState {
    fn default() -> Self {
        Self { selection: AssociationImplementationSelection::NotApplicable }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAW_CUDA_TARGET: &str = "g.firth.components.test.v0";

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
        assert_eq!(jax.fallback_reason(), None);

        let raw_cuda =
            FirthComponentsImplementationState::raw_cuda(RAW_CUDA_TARGET).expect("the test FFI target is nonempty");
        assert_eq!(raw_cuda.requested(), FirthComponentsImplementation::RawCuda);
        assert_eq!(raw_cuda.effective(), FirthComponentsImplementation::RawCuda);
        assert_eq!(raw_cuda.ffi_target(), Some(RAW_CUDA_TARGET));
        assert_eq!(raw_cuda.fallback_reason(), None);
        assert_eq!(FirthComponentsImplementationState::raw_cuda(""), None);

        let fallback = FirthComponentsImplementationState::raw_cuda_fallback(
            FirthComponentsFallbackReason::CudaDriverTooOld,
            "driver predates required PTX support".to_owned(),
        );
        assert_eq!(fallback.requested(), FirthComponentsImplementation::RawCuda);
        assert_eq!(fallback.effective(), FirthComponentsImplementation::Jax);
        assert_eq!(fallback.ffi_target(), None);
        assert_eq!(fallback.fallback_reason(), Some(FirthComponentsFallbackReason::CudaDriverTooOld));
    }

    #[test]
    fn production_jax_state_always_retains_versions() {
        let versions = JaxRuntimeVersions::new("0.11.0".to_owned(), "0.11.0".to_owned()).expect("nonempty versions");
        let state = AssociationImplementationState::jax(versions, None);

        let observed = state.jax_runtime_versions().expect("production JAX state retains versions");
        assert_eq!(observed.jax_version(), "0.11.0");
        assert_eq!(observed.jaxlib_version(), "0.11.0");
        assert_eq!(state.firth_components(), None);
    }

    #[test]
    fn free_text_detail_is_separate_from_stable_projection() {
        let firth = FirthComponentsImplementationState::raw_cuda_fallback(
            FirthComponentsFallbackReason::CudaDriverUnavailable,
            "host-specific loader detail".to_owned(),
        );
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
        assert_eq!(firth.fallback_detail(), Some("host-specific loader detail"));
    }

    #[test]
    fn stable_projection_names_are_explicit() {
        assert_eq!(FirthComponentsImplementation::Jax.stable_name(), "jax");
        assert_eq!(FirthComponentsImplementation::RawCuda.stable_name(), "raw_cuda");
        assert_eq!(FirthComponentsFallbackReason::JaxRegistrationFailure.stable_name(), "jax_registration_failure");
    }

    #[test]
    fn not_applicable_is_the_only_default() {
        let state = AssociationImplementationState::default();
        assert_eq!(state.jax_runtime_versions(), None);
        assert_eq!(state.firth_components(), None);
    }
}
