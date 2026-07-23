//! Canonical association implementation provenance.

use serde::{Deserialize, Serialize};

/// Concrete implementation used for approximate-Firth component reductions.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum FirthComponentsImplementation {
    /// Use the portable JAX reduction.
    #[serde(rename = "jax")]
    Jax,
    /// Use the registered raw-CUDA typed-XLA FFI handler.
    #[serde(rename = "raw_cuda")]
    RawCuda,
}

/// Typed reason a requested raw-CUDA Firth implementation was unavailable.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum FirthComponentsFallbackReason {
    /// The platform cannot load the Linux CUDA ABI.
    #[serde(rename = "unsupported_platform")]
    UnsupportedPlatform,
    /// The CUDA driver shared object could not be loaded.
    #[serde(rename = "cuda_driver_unavailable")]
    CudaDriverUnavailable,
    /// A required CUDA driver symbol was unavailable.
    #[serde(rename = "required_symbol_unavailable")]
    RequiredSymbolUnavailable,
    /// CUDA driver initialization or device inspection failed.
    #[serde(rename = "cuda_driver_failure")]
    CudaDriverFailure,
    /// The installed driver cannot consume the embedded PTX.
    #[serde(rename = "cuda_driver_too_old")]
    CudaDriverTooOld,
    /// The requested CUDA-visible device ordinal was unavailable.
    #[serde(rename = "cuda_device_unavailable")]
    CudaDeviceUnavailable,
    /// The selected device cannot execute the embedded kernel.
    #[serde(rename = "unsupported_compute_capability")]
    UnsupportedComputeCapability,
    /// Native initialization failed outside the expected capability taxonomy.
    #[serde(rename = "native_initialization_failure")]
    NativeInitializationFailure,
    /// JAX rejected registration of the native typed-XLA target.
    #[serde(rename = "jax_registration_failure")]
    JaxRegistrationFailure,
}

/// Structured fallback evidence retained with the effective implementation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FirthComponentsFallback {
    pub reason: FirthComponentsFallbackReason,
    pub detail: String,
}

/// Requested and effective implementation for Firth component reductions.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FirthComponentsImplementationProvenance {
    pub requested: FirthComponentsImplementation,
    pub effective: FirthComponentsImplementation,
    pub fallback: Option<FirthComponentsFallback>,
}

/// Runtime-selected association implementations that affect reproducibility.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct AssociationImplementationProvenance {
    pub firth_components: Option<FirthComponentsImplementationProvenance>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implementation_provenance_serializes_with_stable_names() {
        let provenance = AssociationImplementationProvenance {
            firth_components: Some(FirthComponentsImplementationProvenance {
                requested: FirthComponentsImplementation::RawCuda,
                effective: FirthComponentsImplementation::Jax,
                fallback: Some(FirthComponentsFallback {
                    reason: FirthComponentsFallbackReason::CudaDriverTooOld,
                    detail: "driver 12010 predates PTX ISA 8.2 support".to_string(),
                }),
            }),
        };

        assert_eq!(
            serde_json::to_value(provenance).expect("provenance serializes"),
            serde_json::json!({
                "firth_components": {
                    "requested": "raw_cuda",
                    "effective": "jax",
                    "fallback": {
                        "reason": "cuda_driver_too_old",
                        "detail": "driver 12010 predates PTX ISA 8.2 support"
                    }
                }
            })
        );
    }

    #[test]
    fn default_provenance_marks_firth_components_not_applicable() {
        assert_eq!(
            serde_json::to_value(AssociationImplementationProvenance::default())
                .expect("default provenance serializes"),
            serde_json::json!({"firth_components": null})
        );
    }
}
