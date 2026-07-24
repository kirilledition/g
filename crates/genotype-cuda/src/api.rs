//! Public CUDA genotype crate facade.

use std::error::Error;
#[cfg(target_os = "linux")]
use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
#[cfg(target_os = "linux")]
use std::ptr::NonNull;

/// Stable JAX FFI target registered by the private Python binding boundary.
pub const PACKED8_DEFLATE_FFI_TARGET: &str = "g.bgen.packed8_deflate.v1";
/// Device status bit emitted when packed8 descriptor bounds or alignment validation fails.
pub const PACKED8_DESCRIPTOR_FAILURE_STATUS: u32 = 0x0000_0800;

include!(concat!(env!("OUT_DIR"), "/packed8_artifact_identity.rs"));

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
    nvcomp_version: u32,
    nvcomp_cuda_runtime_version: u32,
    cuda_driver_version: i32,
    device_ordinal: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    input_alignment: usize,
    private: (),
}

impl NvcompCapability {
    /// Returns the loaded nvCOMP encoded version.
    #[must_use]
    pub const fn nvcomp_version(&self) -> u32 {
        self.nvcomp_version
    }

    /// Returns the CUDA runtime version reported by nvCOMP.
    #[must_use]
    pub const fn nvcomp_cuda_runtime_version(&self) -> u32 {
        self.nvcomp_cuda_runtime_version
    }

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

    /// Returns the required alignment of each compressed DEFLATE member.
    #[must_use]
    pub const fn input_alignment(&self) -> usize {
        self.input_alignment
    }
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
        return Ok(nvcomp_capability_from_native(&native_capability));
    }

    let detail = if native_detail.is_null() {
        "native CUDA initialization returned no diagnostic".to_owned()
    } else {
        // SAFETY: The native boundary keeps this NUL-terminated diagnostic alive until the next
        // initialization call on this thread; it is copied before this function returns.
        unsafe { CStr::from_ptr(native_detail) }.to_string_lossy().into_owned()
    };
    Err(initialization_error_from_native_status(status, &native_capability, device_ordinal, detail))
}

#[cfg(target_os = "linux")]
fn nvcomp_capability_from_native(native_capability: &NativeCapability) -> NvcompCapability {
    NvcompCapability {
        nvcomp_version: native_capability.nvcomp_version,
        nvcomp_cuda_runtime_version: native_capability.nvcomp_cuda_runtime_version,
        cuda_driver_version: native_capability.cuda_driver_version,
        device_ordinal: native_capability.device_ordinal,
        compute_capability_major: native_capability.compute_capability_major,
        compute_capability_minor: native_capability.compute_capability_minor,
        input_alignment: native_capability.nvcomp_input_alignment,
        private: (),
    }
}

#[cfg(target_os = "linux")]
fn initialization_error_from_native_status(
    status: c_int,
    native_capability: &NativeCapability,
    device_ordinal: i32,
    detail: String,
) -> NvcompInitializationError {
    match status {
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
    }
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
///
/// # Panics
///
/// Panics if the platform reports a null address for the linked handler
/// function, which violates Rust's function-pointer representation contract.
#[cfg(target_os = "linux")]
#[must_use]
pub fn packed8_deflate_ffi_handler(_capability: &NvcompCapability) -> NonNull<c_void> {
    let handler = g_nvcomp_decode_packed8_ffi as *mut c_void;
    NonNull::new(handler).expect("the linked packed8 typed-XLA handler symbol must be non-null")
}

/// Returns the handler address without a capability proof for one negative test.
///
/// This function exists only under the repository-private test-support feature.
/// Production registration must use [`packed8_deflate_ffi_handler`].
///
/// # Panics
///
/// Panics if the linked handler address is null.
#[cfg(all(target_os = "linux", feature = "private-test-support"))]
#[must_use]
pub fn unqualified_packed8_deflate_ffi_handler_for_test() -> NonNull<c_void> {
    let handler = g_nvcomp_decode_packed8_ffi as *mut c_void;
    NonNull::new(handler).expect("the linked packed8 typed-XLA handler symbol must be non-null")
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

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

    struct DisplayCase {
        error: NvcompInitializationError,
        expected: &'static str,
    }

    #[cfg(target_os = "linux")]
    struct MappingCase {
        status: c_int,
        expected: NvcompInitializationError,
    }

    #[cfg(target_os = "linux")]
    const NATIVE_INTERNAL_FAILURE: c_int = 10;

    #[test]
    fn ffi_artifact_identity_and_member_alignment_are_stable() {
        assert_eq!(std::hint::black_box(PACKED8_DEFLATE_FFI_TARGET), "g.bgen.packed8_deflate.v1");
        assert_eq!(std::hint::black_box(PACKED8_DEFLATE_FFI_API_VERSION), 1);
        assert_eq!(
            std::hint::black_box(PACKED8_DEFLATE_HANDLER_SHA256),
            "12e3dc7dc43d2929047bbee9469750f61923667805d10c1d1757d22253788d94"
        );
        assert_eq!(std::hint::black_box(PACKED8_DEFLATE_MINIMUM_CUDA_DRIVER_VERSION), 12_020);
        assert_eq!(std::hint::black_box(PACKED8_DEFLATE_MINIMUM_COMPUTE_CAPABILITY_MAJOR), 7);
        assert_eq!(std::hint::black_box(PACKED8_DEFLATE_MINIMUM_COMPUTE_CAPABILITY_MINOR), 0);
        assert_eq!(
            std::hint::black_box(PACKED8_DEFLATE_PTX_SHA256),
            "a4b7b84171b6a78e6677a5fe1ba84fa6b4fd5a307eef198a5573fb83381ed088"
        );
        assert_eq!(std::hint::black_box(PACKED8_DEFLATE_PTX_ISA), "8.2");
        assert_eq!(std::hint::black_box(PACKED8_DEFLATE_PTX_TARGET), "sm_70");
        assert_eq!(std::hint::black_box(g_genotype_contracts::RAW_DEFLATE_MEMBER_ALIGNMENT), 4);
        assert_eq!(std::hint::black_box(PACKED8_DESCRIPTOR_FAILURE_STATUS), 2048);
        let ptx = include_str!("../native/packed8_kernel.compute_70.ptx");
        let declared_isa = ptx.lines().find_map(|line| line.trim().strip_prefix(".version "));
        let declared_target = ptx.lines().find_map(|line| line.trim().strip_prefix(".target "));
        assert_eq!(declared_isa, Some(PACKED8_DEFLATE_PTX_ISA));
        assert_eq!(declared_target, Some(PACKED8_DEFLATE_PTX_TARGET));
        assert!(
            include_str!("../native/packed8_kernel.cu").contains("#define STATUS_DESCRIPTOR 2048u"),
            "Rust and CUDA descriptor status values must remain identical"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn successful_native_capability_observations_are_retained() {
        let native_capability = NativeCapability {
            nvcomp_version: 50_300,
            nvcomp_cuda_runtime_version: 12_200,
            cuda_driver_version: 12_090,
            device_ordinal: 2,
            compute_capability_major: 7,
            compute_capability_minor: 5,
            nvcomp_input_alignment: 4,
        };

        let capability = nvcomp_capability_from_native(&native_capability);

        assert_eq!(capability.nvcomp_version(), 50_300);
        assert_eq!(capability.nvcomp_cuda_runtime_version(), 12_200);
        assert_eq!(capability.cuda_driver_version(), 12_090);
        assert_eq!(capability.device_ordinal(), 2);
        assert_eq!(capability.compute_capability_major(), 7);
        assert_eq!(capability.compute_capability_minor(), 5);
        assert_eq!(capability.input_alignment(), 4);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_status_and_capability_abi_match_the_linked_boundary() {
        let status_values = std::hint::black_box([
            INITIALIZATION_SUCCESS,
            CUDA_DRIVER_UNAVAILABLE,
            NVCOMP_LIBRARY_UNAVAILABLE,
            REQUIRED_SYMBOL_UNAVAILABLE,
            NVCOMP_VERSION_UNSUPPORTED,
            CUDA_DRIVER_FAILURE,
            CUDA_DRIVER_TOO_OLD,
            CUDA_DEVICE_UNAVAILABLE,
            COMPUTE_CAPABILITY_UNSUPPORTED,
            NVCOMP_INPUT_ALIGNMENT_UNSUPPORTED,
            NATIVE_INTERNAL_FAILURE,
        ]);

        assert_eq!(status_values, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(std::mem::size_of::<NativeCapability>(), 32);
        assert_eq!(std::mem::align_of::<NativeCapability>(), std::mem::align_of::<usize>());
        assert_eq!(NativeCapability::default().nvcomp_input_alignment, 0);
    }

    #[test]
    fn initialization_errors_render_every_diagnostic_field() {
        let cases = [
            DisplayCase {
                error: NvcompInitializationError::UnsupportedPlatform,
                expected: "the nvCOMP path is supported only on Linux",
            },
            DisplayCase {
                error: NvcompInitializationError::CudaDriverUnavailable { detail: "driver unavailable".to_owned() },
                expected: "driver unavailable",
            },
            DisplayCase {
                error: NvcompInitializationError::NvcompLibraryUnavailable { detail: "nvCOMP unavailable".to_owned() },
                expected: "nvCOMP unavailable",
            },
            DisplayCase {
                error: NvcompInitializationError::RequiredSymbolUnavailable { detail: "symbol unavailable".to_owned() },
                expected: "symbol unavailable",
            },
            DisplayCase {
                error: NvcompInitializationError::UnsupportedNvcompVersion {
                    version: 50_100,
                    detail: "5.2 or newer is required".to_owned(),
                },
                expected: "unsupported nvCOMP version 50100: 5.2 or newer is required",
            },
            DisplayCase {
                error: NvcompInitializationError::CudaDriverFailure { detail: "driver failure".to_owned() },
                expected: "driver failure",
            },
            DisplayCase {
                error: NvcompInitializationError::CudaDriverTooOld {
                    version: 12_010,
                    detail: "PTX ISA 8.2 requires CUDA 12.2".to_owned(),
                },
                expected: "CUDA driver API version 12010 is too old: PTX ISA 8.2 requires CUDA 12.2",
            },
            DisplayCase {
                error: NvcompInitializationError::CudaDeviceUnavailable {
                    device_ordinal: 3,
                    detail: "ordinal is not visible".to_owned(),
                },
                expected: "CUDA device 3 is unavailable: ordinal is not visible",
            },
            DisplayCase {
                error: NvcompInitializationError::UnsupportedComputeCapability {
                    device_ordinal: 4,
                    major: 6,
                    minor: 1,
                    detail: "compute_70 is required".to_owned(),
                },
                expected: "CUDA device 4 has unsupported compute capability 6.1: compute_70 is required",
            },
            DisplayCase {
                error: NvcompInitializationError::UnsupportedNvcompInputAlignment {
                    required_alignment: 8,
                    member_alignment: 4,
                    detail: "runtime requirement exceeds slab alignment".to_owned(),
                },
                expected: "nvCOMP requires 8-byte DEFLATE input alignment, but slab members use 4-byte alignment: runtime requirement exceeds slab alignment",
            },
            DisplayCase {
                error: NvcompInitializationError::Internal { detail: "internal failure".to_owned() },
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
    fn every_native_failure_status_maps_to_the_public_error_taxonomy() {
        let native_capability = NativeCapability {
            nvcomp_version: 50_300,
            nvcomp_cuda_runtime_version: 12_200,
            cuda_driver_version: 12_010,
            device_ordinal: 99,
            compute_capability_major: 6,
            compute_capability_minor: 1,
            nvcomp_input_alignment: 8,
        };
        let requested_device_ordinal = 3;
        let cases = [
            MappingCase {
                status: CUDA_DRIVER_UNAVAILABLE,
                expected: NvcompInitializationError::CudaDriverUnavailable { detail: "detail".to_owned() },
            },
            MappingCase {
                status: NVCOMP_LIBRARY_UNAVAILABLE,
                expected: NvcompInitializationError::NvcompLibraryUnavailable { detail: "detail".to_owned() },
            },
            MappingCase {
                status: REQUIRED_SYMBOL_UNAVAILABLE,
                expected: NvcompInitializationError::RequiredSymbolUnavailable { detail: "detail".to_owned() },
            },
            MappingCase {
                status: NVCOMP_VERSION_UNSUPPORTED,
                expected: NvcompInitializationError::UnsupportedNvcompVersion {
                    version: 50_300,
                    detail: "detail".to_owned(),
                },
            },
            MappingCase {
                status: CUDA_DRIVER_FAILURE,
                expected: NvcompInitializationError::CudaDriverFailure { detail: "detail".to_owned() },
            },
            MappingCase {
                status: CUDA_DRIVER_TOO_OLD,
                expected: NvcompInitializationError::CudaDriverTooOld { version: 12_010, detail: "detail".to_owned() },
            },
            MappingCase {
                status: CUDA_DEVICE_UNAVAILABLE,
                expected: NvcompInitializationError::CudaDeviceUnavailable {
                    device_ordinal: requested_device_ordinal,
                    detail: "detail".to_owned(),
                },
            },
            MappingCase {
                status: COMPUTE_CAPABILITY_UNSUPPORTED,
                expected: NvcompInitializationError::UnsupportedComputeCapability {
                    device_ordinal: requested_device_ordinal,
                    major: 6,
                    minor: 1,
                    detail: "detail".to_owned(),
                },
            },
            MappingCase {
                status: NVCOMP_INPUT_ALIGNMENT_UNSUPPORTED,
                expected: NvcompInitializationError::UnsupportedNvcompInputAlignment {
                    required_alignment: 8,
                    member_alignment: 4,
                    detail: "detail".to_owned(),
                },
            },
            MappingCase {
                status: NATIVE_INTERNAL_FAILURE,
                expected: NvcompInitializationError::Internal { detail: "detail".to_owned() },
            },
            MappingCase {
                status: i32::MAX,
                expected: NvcompInitializationError::Internal { detail: "detail".to_owned() },
            },
        ];

        for case in cases {
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
            nvcomp_version: 50_300,
            nvcomp_cuda_runtime_version: 12_200,
            cuda_driver_version: 12_090,
            device_ordinal: 2,
            compute_capability_major: 7,
            compute_capability_minor: 0,
            nvcomp_input_alignment: 4,
        };
        let capability = nvcomp_capability_from_native(&native_capability);

        let first_handler = packed8_deflate_ffi_handler(&capability);
        let second_handler = packed8_deflate_ffi_handler(&capability);

        assert_eq!(first_handler, second_handler);
        assert_eq!(capability.nvcomp_version(), 50_300);
        assert_eq!(capability.nvcomp_cuda_runtime_version(), 12_200);
        assert_eq!(capability.cuda_driver_version(), 12_090);
        assert_eq!(capability.device_ordinal(), 2);
        assert_eq!(capability.compute_capability_major(), 7);
        assert_eq!(capability.compute_capability_minor(), 0);
        assert_eq!(capability.input_alignment(), 4);
        assert!(format!("{capability:?}").contains("NvcompCapability"));
    }
}
