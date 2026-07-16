//! Public CUDA genotype crate facade.

use std::error::Error;
#[cfg(target_os = "linux")]
use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
#[cfg(target_os = "linux")]
use std::ptr::NonNull;

/// Stable JAX FFI target registered by the private Python binding boundary.
pub const PACKED8_DEFLATE_FFI_TARGET: &str = "g.bgen.packed8_deflate.v1";

#[cfg(target_os = "linux")]
const INITIALIZATION_SUCCESS: c_int = 0;
#[cfg(target_os = "linux")]
const CUDA_DRIVER_UNAVAILABLE: c_int = 1;
#[cfg(target_os = "linux")]
const NVCOMP_LIBRARY_UNAVAILABLE: c_int = 2;
#[cfg(target_os = "linux")]
const REQUIRED_SYMBOL_UNAVAILABLE: c_int = 3;
#[cfg(target_os = "linux")]
const NVCOMP_VERSION_UNSUPPORTED: c_int = 4;
#[cfg(target_os = "linux")]
const CUDA_DRIVER_FAILURE: c_int = 5;
#[cfg(target_os = "linux")]
const CUDA_DRIVER_TOO_OLD: c_int = 6;
#[cfg(target_os = "linux")]
const CUDA_DEVICE_UNAVAILABLE: c_int = 7;
#[cfg(target_os = "linux")]
const COMPUTE_CAPABILITY_UNSUPPORTED: c_int = 8;
#[cfg(target_os = "linux")]
const NVCOMP_INPUT_ALIGNMENT_UNSUPPORTED: c_int = 9;

/// Proof that CUDA, nvCOMP, the selected device, and buffer alignment were validated.
///
/// The embedded PTX module is validated lazily against the actual XLA CUDA
/// context on that context's first handler invocation.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub struct NvcompCapability {
    private: (),
}

/// Reason the nvCOMP JAX FFI path cannot be selected for a run.
#[derive(Debug, Eq, PartialEq)]
pub enum NvcompInitializationError {
    /// The crate was built for an operating system without the Linux CUDA ABI.
    UnsupportedPlatform,
    /// The CUDA driver shared object could not be loaded.
    CudaDriverUnavailable { detail: String },
    /// The nvCOMP shared object could not be loaded.
    NvcompLibraryUnavailable { detail: String },
    /// A required stable C ABI symbol is missing from a loaded shared object.
    RequiredSymbolUnavailable { detail: String },
    /// The loaded nvCOMP library is outside the supported 5.2 <= version < 6 range.
    UnsupportedNvcompVersion { version: u32, detail: String },
    /// A CUDA driver operation failed while validating the requested device.
    CudaDriverFailure { detail: String },
    /// The installed CUDA driver predates CUDA 12.2 and PTX ISA 8.2 support.
    CudaDriverTooOld { version: i32, detail: String },
    /// The requested CUDA-visible device ordinal is unavailable.
    CudaDeviceUnavailable { device_ordinal: i32, detail: String },
    /// The requested device predates compute capability 7.0.
    UnsupportedComputeCapability { device_ordinal: i32, major: i32, minor: i32, detail: String },
    /// The loaded nvCOMP runtime requires stricter input alignment than slab planning provides.
    UnsupportedNvcompInputAlignment { required_alignment: usize, member_alignment: usize, detail: String },
    /// An unexpected native initialization failure occurred.
    Internal { detail: String },
}

impl fmt::Display for NvcompInitializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => formatter.write_str("the nvCOMP path is supported only on Linux"),
            Self::CudaDriverUnavailable { detail }
            | Self::NvcompLibraryUnavailable { detail }
            | Self::RequiredSymbolUnavailable { detail }
            | Self::CudaDriverFailure { detail }
            | Self::Internal { detail } => formatter.write_str(detail),
            Self::UnsupportedNvcompVersion { version, detail } => {
                write!(formatter, "unsupported nvCOMP version {version}: {detail}")
            }
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
            Self::UnsupportedNvcompInputAlignment { required_alignment, member_alignment, detail } => write!(
                formatter,
                "nvCOMP requires {required_alignment}-byte DEFLATE input alignment, but slab members use \
                 {member_alignment}-byte alignment: {detail}"
            ),
        }
    }
}

impl Error for NvcompInitializationError {}

#[repr(C)]
#[derive(Default)]
#[cfg(target_os = "linux")]
struct NativeCapability {
    nvcomp_version: u32,
    nvcomp_cuda_runtime_version: u32,
    cuda_driver_version: i32,
    device_ordinal: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    nvcomp_input_alignment: usize,
}

/// Loads and validates the CUDA driver and nvCOMP runtime for one visible device.
///
/// The official `nvidia-libnvcomp-cu12` loader must load `libnvcomp.so.5` before
/// this call. No CUDA context is created and no module is loaded during validation.
///
/// # Errors
///
/// Returns a precise capability error when the runtime, driver, device, or
/// buffer-alignment contract cannot support the fused handler. PTX loading is
/// deferred until the first invocation on each XLA CUDA context.
#[cfg(target_os = "linux")]
pub fn initialize_nvcomp_runtime(device_ordinal: i32) -> Result<NvcompCapability, NvcompInitializationError> {
    let mut native_capability = NativeCapability::default();
    let mut native_detail = std::ptr::null();
    // SAFETY: Both out-pointers remain valid for the duration of the native call.
    let status = unsafe {
        g_genotype_cuda_initialize_nvcomp_runtime(
            device_ordinal,
            g_genotype_contracts::RAW_DEFLATE_MEMBER_ALIGNMENT,
            &raw mut native_capability,
            &raw mut native_detail,
        )
    };

    if status == INITIALIZATION_SUCCESS {
        return Ok(NvcompCapability { private: () });
    }

    let detail = if native_detail.is_null() {
        "native CUDA initialization returned no diagnostic".to_owned()
    } else {
        // SAFETY: The native boundary keeps this NUL-terminated diagnostic alive until the next
        // initialization call on this thread; it is copied before this function returns.
        unsafe { CStr::from_ptr(native_detail) }.to_string_lossy().into_owned()
    };
    Err(match status {
        CUDA_DRIVER_UNAVAILABLE => NvcompInitializationError::CudaDriverUnavailable { detail },
        NVCOMP_LIBRARY_UNAVAILABLE => NvcompInitializationError::NvcompLibraryUnavailable { detail },
        REQUIRED_SYMBOL_UNAVAILABLE => NvcompInitializationError::RequiredSymbolUnavailable { detail },
        NVCOMP_VERSION_UNSUPPORTED => {
            NvcompInitializationError::UnsupportedNvcompVersion { version: native_capability.nvcomp_version, detail }
        }
        CUDA_DRIVER_FAILURE => NvcompInitializationError::CudaDriverFailure { detail },
        CUDA_DRIVER_TOO_OLD => {
            NvcompInitializationError::CudaDriverTooOld { version: native_capability.cuda_driver_version, detail }
        }
        CUDA_DEVICE_UNAVAILABLE => NvcompInitializationError::CudaDeviceUnavailable { device_ordinal, detail },
        COMPUTE_CAPABILITY_UNSUPPORTED => NvcompInitializationError::UnsupportedComputeCapability {
            device_ordinal,
            major: native_capability.compute_capability_major,
            minor: native_capability.compute_capability_minor,
            detail,
        },
        NVCOMP_INPUT_ALIGNMENT_UNSUPPORTED => NvcompInitializationError::UnsupportedNvcompInputAlignment {
            required_alignment: native_capability.nvcomp_input_alignment,
            member_alignment: g_genotype_contracts::RAW_DEFLATE_MEMBER_ALIGNMENT,
            detail,
        },
        _ => NvcompInitializationError::Internal { detail },
    })
}

/// Reports that the CUDA FFI path is unavailable on non-Linux builds.
///
/// # Errors
///
/// Always returns [`NvcompInitializationError::UnsupportedPlatform`].
#[cfg(not(target_os = "linux"))]
pub fn initialize_nvcomp_runtime(_device_ordinal: i32) -> Result<NvcompCapability, NvcompInitializationError> {
    Err(NvcompInitializationError::UnsupportedPlatform)
}

/// Returns the non-null typed-XLA FFI handler address for capsule registration.
///
/// Requiring the opaque capability proof prevents callers from obtaining a
/// handler before [`initialize_nvcomp_runtime`] succeeds for the selected device.
#[cfg(target_os = "linux")]
#[must_use]
pub fn packed8_deflate_ffi_handler(_capability: &NvcompCapability) -> NonNull<c_void> {
    let handler = g_nvcomp_decode_packed8_ffi as *mut c_void;
    // SAFETY: A linked function symbol always has a non-null address.
    unsafe { NonNull::new_unchecked(handler) }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn g_genotype_cuda_initialize_nvcomp_runtime(
        device_ordinal: c_int,
        member_alignment: usize,
        capability: *mut NativeCapability,
        detail: *mut *const c_char,
    ) -> c_int;

    fn g_nvcomp_decode_packed8_ffi(call_frame: *mut c_void) -> *mut c_void;
}
