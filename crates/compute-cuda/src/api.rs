//! Crate-owned CUDA compute capability and FFI surface.

use std::error::Error;
#[cfg(target_os = "linux")]
use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
#[cfg(target_os = "linux")]
use std::ptr::NonNull;

/// Stable approximate-Firth component target registered by the private binding.
pub const FIRTH_COMPONENTS_FFI_TARGET: &str = "g.firth.components.v1";

include!(concat!(env!("OUT_DIR"), "/firth_components_artifact_identity.rs"));

#[cfg(target_os = "linux")]
const INITIALIZATION_SUCCESS: c_int = 0;
#[cfg(target_os = "linux")]
const CUDA_DRIVER_UNAVAILABLE: c_int = 1;
#[cfg(target_os = "linux")]
const REQUIRED_SYMBOL_UNAVAILABLE: c_int = 2;
#[cfg(target_os = "linux")]
const CUDA_DRIVER_FAILURE: c_int = 3;
#[cfg(target_os = "linux")]
const CUDA_DRIVER_TOO_OLD: c_int = 4;
#[cfg(target_os = "linux")]
const CUDA_DEVICE_UNAVAILABLE: c_int = 5;
#[cfg(target_os = "linux")]
const COMPUTE_CAPABILITY_UNSUPPORTED: c_int = 6;

/// Proof that the selected CUDA device can JIT the embedded compute PTX.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub struct FirthComponentsCapability {
    cuda_driver_version: i32,
    device_ordinal: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    private: (),
}

impl FirthComponentsCapability {
    /// Returns the CUDA driver API version observed during qualification.
    #[must_use]
    pub const fn cuda_driver_version(&self) -> i32 {
        self.cuda_driver_version
    }

    /// Returns the CUDA-visible device ordinal qualified for this capability.
    #[must_use]
    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    /// Returns the qualified device's compute-capability major version.
    #[must_use]
    pub const fn compute_capability_major(&self) -> i32 {
        self.compute_capability_major
    }

    /// Returns the qualified device's compute-capability minor version.
    #[must_use]
    pub const fn compute_capability_minor(&self) -> i32 {
        self.compute_capability_minor
    }
}

/// Reason the optional CUDA Firth component path cannot be selected.
#[derive(Debug, Eq, PartialEq)]
pub enum FirthComponentsInitializationError {
    /// The crate was built for an operating system without the Linux CUDA ABI.
    UnsupportedPlatform,
    /// The CUDA driver shared object could not be loaded.
    CudaDriverUnavailable { detail: String },
    /// A required stable CUDA driver symbol is unavailable.
    RequiredSymbolUnavailable { detail: String },
    /// A CUDA driver operation failed while validating the selected device.
    CudaDriverFailure { detail: String },
    /// The driver predates CUDA 12.2 and cannot consume PTX ISA 8.2.
    CudaDriverTooOld { version: i32, detail: String },
    /// The requested CUDA-visible device ordinal is unavailable.
    CudaDeviceUnavailable { cuda_driver_version: i32, device_ordinal: i32, detail: String },
    /// The requested device predates compute capability 7.0.
    UnsupportedComputeCapability {
        cuda_driver_version: i32,
        device_ordinal: i32,
        major: i32,
        minor: i32,
        detail: String,
    },
    /// An unexpected native initialization failure occurred.
    Internal { detail: String },
}

impl fmt::Display for FirthComponentsInitializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => formatter.write_str("CUDA Firth components are supported only on Linux"),
            Self::CudaDriverUnavailable { detail }
            | Self::RequiredSymbolUnavailable { detail }
            | Self::CudaDriverFailure { detail }
            | Self::Internal { detail } => formatter.write_str(detail),
            Self::CudaDriverTooOld { version, detail } => {
                write!(formatter, "CUDA driver API version {version} is too old: {detail}")
            }
            Self::CudaDeviceUnavailable { device_ordinal, detail, .. } => {
                write!(formatter, "CUDA device {device_ordinal} is unavailable: {detail}")
            }
            Self::UnsupportedComputeCapability { device_ordinal, major, minor, detail, .. } => write!(
                formatter,
                "CUDA device {device_ordinal} has unsupported compute capability {major}.{minor}: {detail}"
            ),
        }
    }
}

impl Error for FirthComponentsInitializationError {}

#[repr(C)]
#[derive(Default)]
#[cfg(target_os = "linux")]
struct NativeCapability {
    cuda_driver_version: i32,
    device_ordinal: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
}

/// Validates one CUDA-visible device for the optional Firth component kernel.
///
/// No CUDA context is created and no module is loaded during validation. The
/// embedded PTX is loaded lazily on the XLA execution context.
///
/// # Errors
///
/// Returns a precise capability error when the driver or selected device cannot
/// support the checked-in PTX artifact.
#[cfg(target_os = "linux")]
pub fn initialize_firth_components_runtime(
    device_ordinal: i32,
) -> Result<FirthComponentsCapability, FirthComponentsInitializationError> {
    let mut native_capability = NativeCapability::default();
    let mut native_detail = std::ptr::null();
    // SAFETY: Both out-pointers remain valid for the duration of the native call.
    let status = unsafe {
        g_compute_cuda_initialize_firth_components_runtime(
            device_ordinal,
            &raw mut native_capability,
            &raw mut native_detail,
        )
    };
    if status == INITIALIZATION_SUCCESS {
        return Ok(firth_components_capability_from_native(&native_capability));
    }
    let detail = if native_detail.is_null() {
        "native CUDA compute initialization returned no diagnostic".to_owned()
    } else {
        // SAFETY: Native code retains the NUL-terminated thread-local detail until
        // the next initialization call on this thread; copy it before returning.
        unsafe { CStr::from_ptr(native_detail) }.to_string_lossy().into_owned()
    };
    Err(initialization_error_from_native_status(status, &native_capability, device_ordinal, detail))
}

#[cfg(target_os = "linux")]
fn firth_components_capability_from_native(native_capability: &NativeCapability) -> FirthComponentsCapability {
    FirthComponentsCapability {
        cuda_driver_version: native_capability.cuda_driver_version,
        device_ordinal: native_capability.device_ordinal,
        compute_capability_major: native_capability.compute_capability_major,
        compute_capability_minor: native_capability.compute_capability_minor,
        private: (),
    }
}

#[cfg(target_os = "linux")]
fn initialization_error_from_native_status(
    status: c_int,
    native_capability: &NativeCapability,
    device_ordinal: i32,
    detail: String,
) -> FirthComponentsInitializationError {
    match status {
        CUDA_DRIVER_UNAVAILABLE => FirthComponentsInitializationError::CudaDriverUnavailable { detail },
        REQUIRED_SYMBOL_UNAVAILABLE => FirthComponentsInitializationError::RequiredSymbolUnavailable { detail },
        CUDA_DRIVER_FAILURE => FirthComponentsInitializationError::CudaDriverFailure { detail },
        CUDA_DRIVER_TOO_OLD => FirthComponentsInitializationError::CudaDriverTooOld {
            version: native_capability.cuda_driver_version,
            detail,
        },
        CUDA_DEVICE_UNAVAILABLE => FirthComponentsInitializationError::CudaDeviceUnavailable {
            cuda_driver_version: native_capability.cuda_driver_version,
            device_ordinal,
            detail,
        },
        COMPUTE_CAPABILITY_UNSUPPORTED => FirthComponentsInitializationError::UnsupportedComputeCapability {
            cuda_driver_version: native_capability.cuda_driver_version,
            device_ordinal,
            major: native_capability.compute_capability_major,
            minor: native_capability.compute_capability_minor,
            detail,
        },
        _ => FirthComponentsInitializationError::Internal { detail },
    }
}

/// Reports that raw CUDA Firth components are unavailable off Linux.
///
/// # Errors
///
/// Always returns [`FirthComponentsInitializationError::UnsupportedPlatform`].
#[cfg(not(target_os = "linux"))]
pub fn initialize_firth_components_runtime(
    _device_ordinal: i32,
) -> Result<FirthComponentsCapability, FirthComponentsInitializationError> {
    Err(FirthComponentsInitializationError::UnsupportedPlatform)
}

/// Returns the process-lifetime typed-XLA FFI handler address.
///
/// # Panics
///
/// Panics if the platform reports a null address for the linked handler
/// function, which violates Rust's function-pointer representation contract.
#[cfg(target_os = "linux")]
#[must_use]
pub fn firth_components_ffi_handler(_capability: &FirthComponentsCapability) -> NonNull<c_void> {
    let handler = g_firth_components_ffi as *mut c_void;
    NonNull::new(handler).expect("the linked Firth typed-XLA handler symbol must be non-null")
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn g_compute_cuda_initialize_firth_components_runtime(
        device_ordinal: c_int,
        capability: *mut NativeCapability,
        detail: *mut *const c_char,
    ) -> c_int;
    fn g_firth_components_ffi(call_frame: *mut c_void) -> *mut c_void;
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

    struct DisplayCase {
        error: FirthComponentsInitializationError,
        expected: &'static str,
    }

    #[cfg(target_os = "linux")]
    struct MappingCase {
        status: c_int,
        expected: FirthComponentsInitializationError,
    }

    #[cfg(target_os = "linux")]
    const NATIVE_INTERNAL_FAILURE: c_int = 7;

    #[test]
    fn ffi_and_embedded_ptx_identity_are_stable() {
        assert_eq!(std::hint::black_box(FIRTH_COMPONENTS_FFI_TARGET), "g.firth.components.v1");
        assert_eq!(std::hint::black_box(FIRTH_COMPONENTS_FFI_API_VERSION), 1);
        assert_eq!(
            std::hint::black_box(FIRTH_COMPONENTS_HANDLER_SHA256),
            "005f72c4d5ab3d81f16db305bd94bc7bd1eb9febf0e0ba9e10a486122d935ff8"
        );
        assert_eq!(std::hint::black_box(FIRTH_COMPONENTS_MINIMUM_CUDA_DRIVER_VERSION), 12_020);
        assert_eq!(std::hint::black_box(FIRTH_COMPONENTS_MINIMUM_COMPUTE_CAPABILITY_MAJOR), 7);
        assert_eq!(std::hint::black_box(FIRTH_COMPONENTS_MINIMUM_COMPUTE_CAPABILITY_MINOR), 0);
        assert_eq!(
            std::hint::black_box(FIRTH_COMPONENTS_PTX_SHA256),
            "a22c9866447f21c7f7cd484ec1e12c3c249a5a84acf3850cb3eb3a56697c736f"
        );
        assert_eq!(std::hint::black_box(FIRTH_COMPONENTS_PTX_ISA), "8.2");
        assert_eq!(std::hint::black_box(FIRTH_COMPONENTS_PTX_TARGET), "sm_70");
        let ptx = include_str!("../native/firth_components_kernel.compute_70.ptx");
        let declared_isa = ptx.lines().find_map(|line| line.trim().strip_prefix(".version "));
        let declared_target = ptx.lines().find_map(|line| line.trim().strip_prefix(".target "));
        assert_eq!(declared_isa, Some(FIRTH_COMPONENTS_PTX_ISA));
        assert_eq!(declared_target, Some(FIRTH_COMPONENTS_PTX_TARGET));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_status_and_capability_abi_match_the_linked_boundary() {
        let status_values = std::hint::black_box([
            INITIALIZATION_SUCCESS,
            CUDA_DRIVER_UNAVAILABLE,
            REQUIRED_SYMBOL_UNAVAILABLE,
            CUDA_DRIVER_FAILURE,
            CUDA_DRIVER_TOO_OLD,
            CUDA_DEVICE_UNAVAILABLE,
            COMPUTE_CAPABILITY_UNSUPPORTED,
            NATIVE_INTERNAL_FAILURE,
        ]);

        assert_eq!(status_values, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(std::mem::size_of::<NativeCapability>(), 16);
        assert_eq!(std::mem::align_of::<NativeCapability>(), std::mem::align_of::<i32>());
        assert_eq!(NativeCapability::default().cuda_driver_version, 0);
    }

    #[test]
    fn initialization_errors_render_every_diagnostic_field() {
        let cases = [
            DisplayCase {
                error: FirthComponentsInitializationError::UnsupportedPlatform,
                expected: "CUDA Firth components are supported only on Linux",
            },
            DisplayCase {
                error: FirthComponentsInitializationError::CudaDriverUnavailable {
                    detail: "driver unavailable".to_owned(),
                },
                expected: "driver unavailable",
            },
            DisplayCase {
                error: FirthComponentsInitializationError::RequiredSymbolUnavailable {
                    detail: "symbol unavailable".to_owned(),
                },
                expected: "symbol unavailable",
            },
            DisplayCase {
                error: FirthComponentsInitializationError::CudaDriverFailure { detail: "driver failure".to_owned() },
                expected: "driver failure",
            },
            DisplayCase {
                error: FirthComponentsInitializationError::CudaDriverTooOld {
                    version: 12_010,
                    detail: "PTX ISA 8.2 requires CUDA 12.2".to_owned(),
                },
                expected: "CUDA driver API version 12010 is too old: PTX ISA 8.2 requires CUDA 12.2",
            },
            DisplayCase {
                error: FirthComponentsInitializationError::CudaDeviceUnavailable {
                    cuda_driver_version: 12_090,
                    device_ordinal: 3,
                    detail: "ordinal is not visible".to_owned(),
                },
                expected: "CUDA device 3 is unavailable: ordinal is not visible",
            },
            DisplayCase {
                error: FirthComponentsInitializationError::UnsupportedComputeCapability {
                    cuda_driver_version: 12_090,
                    device_ordinal: 4,
                    major: 6,
                    minor: 1,
                    detail: "compute_70 is required".to_owned(),
                },
                expected: "CUDA device 4 has unsupported compute capability 6.1: compute_70 is required",
            },
            DisplayCase {
                error: FirthComponentsInitializationError::Internal { detail: "internal failure".to_owned() },
                expected: "internal failure",
            },
        ];

        for case in cases {
            assert_eq!(case.error.to_string(), case.expected);
            assert!(case.error.source().is_none());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn successful_native_capability_observations_are_retained() {
        let native_capability = NativeCapability {
            cuda_driver_version: 12_090,
            device_ordinal: 2,
            compute_capability_major: 7,
            compute_capability_minor: 5,
        };

        let capability = firth_components_capability_from_native(&native_capability);

        assert_eq!(capability.cuda_driver_version(), 12_090);
        assert_eq!(capability.device_ordinal(), 2);
        assert_eq!(capability.compute_capability_major(), 7);
        assert_eq!(capability.compute_capability_minor(), 5);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn every_native_failure_status_maps_to_the_public_error_taxonomy() {
        let requested_device_ordinal = 3;
        let cases = [
            MappingCase {
                status: CUDA_DRIVER_UNAVAILABLE,
                expected: FirthComponentsInitializationError::CudaDriverUnavailable { detail: "detail".to_owned() },
            },
            MappingCase {
                status: REQUIRED_SYMBOL_UNAVAILABLE,
                expected: FirthComponentsInitializationError::RequiredSymbolUnavailable { detail: "detail".to_owned() },
            },
            MappingCase {
                status: CUDA_DRIVER_FAILURE,
                expected: FirthComponentsInitializationError::CudaDriverFailure { detail: "detail".to_owned() },
            },
            MappingCase {
                status: CUDA_DRIVER_TOO_OLD,
                expected: FirthComponentsInitializationError::CudaDriverTooOld {
                    version: 12_010,
                    detail: "detail".to_owned(),
                },
            },
            MappingCase {
                status: CUDA_DEVICE_UNAVAILABLE,
                expected: FirthComponentsInitializationError::CudaDeviceUnavailable {
                    cuda_driver_version: 12_090,
                    device_ordinal: requested_device_ordinal,
                    detail: "detail".to_owned(),
                },
            },
            MappingCase {
                status: COMPUTE_CAPABILITY_UNSUPPORTED,
                expected: FirthComponentsInitializationError::UnsupportedComputeCapability {
                    cuda_driver_version: 12_090,
                    device_ordinal: requested_device_ordinal,
                    major: 6,
                    minor: 1,
                    detail: "detail".to_owned(),
                },
            },
            MappingCase {
                status: NATIVE_INTERNAL_FAILURE,
                expected: FirthComponentsInitializationError::Internal { detail: "detail".to_owned() },
            },
            MappingCase {
                status: i32::MAX,
                expected: FirthComponentsInitializationError::Internal { detail: "detail".to_owned() },
            },
        ];

        for case in cases {
            let native_capability = NativeCapability {
                cuda_driver_version: if case.status == CUDA_DRIVER_TOO_OLD { 12_010 } else { 12_090 },
                device_ordinal: 99,
                compute_capability_major: 6,
                compute_capability_minor: 1,
            };
            let observed = initialization_error_from_native_status(
                case.status,
                &native_capability,
                requested_device_ordinal,
                "detail".to_owned(),
            );
            assert_eq!(observed, case.expected);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn capability_gates_a_stable_non_null_handler_address() {
        let native_capability = NativeCapability {
            cuda_driver_version: 12_090,
            device_ordinal: 2,
            compute_capability_major: 7,
            compute_capability_minor: 0,
        };
        let capability = firth_components_capability_from_native(&native_capability);

        let first_handler = firth_components_ffi_handler(&capability);
        let second_handler = firth_components_ffi_handler(&capability);

        assert_eq!(first_handler, second_handler);
        assert_eq!(capability.cuda_driver_version(), 12_090);
        assert_eq!(capability.device_ordinal(), 2);
        assert_eq!(capability.compute_capability_major(), 7);
        assert_eq!(capability.compute_capability_minor(), 0);
        assert!(format!("{capability:?}").contains("FirthComponentsCapability"));
    }
}
