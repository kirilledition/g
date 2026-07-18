//! Crate-owned CUDA compute capability and FFI surface.

use std::error::Error;
#[cfg(target_os = "linux")]
use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
#[cfg(target_os = "linux")]
use std::ptr::NonNull;

/// Stable approximate-Firth component target registered by the private binding.
pub const FIRTH_COMPONENTS_FFI_TARGET: &str = "g.firth.components.v1";

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
    private: (),
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
    CudaDeviceUnavailable { device_ordinal: i32, detail: String },
    /// The requested device predates compute capability 7.0.
    UnsupportedComputeCapability { device_ordinal: i32, major: i32, minor: i32, detail: String },
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
            Self::CudaDeviceUnavailable { device_ordinal, detail } => {
                write!(formatter, "CUDA device {device_ordinal} is unavailable: {detail}")
            }
            Self::UnsupportedComputeCapability { device_ordinal, major, minor, detail } => write!(
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
        return Ok(FirthComponentsCapability { private: () });
    }
    let detail = if native_detail.is_null() {
        "native CUDA compute initialization returned no diagnostic".to_owned()
    } else {
        // SAFETY: Native code retains the NUL-terminated thread-local detail until
        // the next initialization call on this thread; copy it before returning.
        unsafe { CStr::from_ptr(native_detail) }.to_string_lossy().into_owned()
    };
    Err(match status {
        CUDA_DRIVER_UNAVAILABLE => FirthComponentsInitializationError::CudaDriverUnavailable { detail },
        REQUIRED_SYMBOL_UNAVAILABLE => FirthComponentsInitializationError::RequiredSymbolUnavailable { detail },
        CUDA_DRIVER_FAILURE => FirthComponentsInitializationError::CudaDriverFailure { detail },
        CUDA_DRIVER_TOO_OLD => FirthComponentsInitializationError::CudaDriverTooOld {
            version: native_capability.cuda_driver_version,
            detail,
        },
        CUDA_DEVICE_UNAVAILABLE => FirthComponentsInitializationError::CudaDeviceUnavailable { device_ordinal, detail },
        COMPUTE_CAPABILITY_UNSUPPORTED => FirthComponentsInitializationError::UnsupportedComputeCapability {
            device_ordinal,
            major: native_capability.compute_capability_major,
            minor: native_capability.compute_capability_minor,
            detail,
        },
        _ => FirthComponentsInitializationError::Internal { detail },
    })
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
#[cfg(target_os = "linux")]
#[must_use]
pub fn firth_components_ffi_handler(_capability: &FirthComponentsCapability) -> NonNull<c_void> {
    let handler = g_firth_components_ffi as *mut c_void;
    // SAFETY: A linked function symbol always has a non-null address.
    unsafe { NonNull::new_unchecked(handler) }
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
