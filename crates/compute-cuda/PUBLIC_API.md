# g-compute-cuda public API

The crate owns the stable internal Firth FFI target name and JAX registration
API integer. It also exposes the build-verified embedded artifact identity:
the PTX SHA-256, declared PTX ISA, and declared PTX target. The build script
generates these PTX constants from the verified checked-in artifact so runtime
registration and output compatibility do not repeat provenance literals.
It also generates a framed source/ABI SHA-256 over the native FFI wrapper,
embedded PTX, shared CUDA-driver support, and vendored XLA FFI ABI headers.
Wrapper or launch-contract changes therefore change resume identity even when
the PTX itself is unchanged. This is deliberately a semantic source-set
identity, not a hash of compiler-dependent native-library bytes.

Successful capability initialization retains the CUDA driver API version,
CUDA-visible device ordinal, and compute-capability major and minor versions
returned by the native C ABI. Read-only accessors make those observations
available for diagnostics without granting access to driver or module state.
The capability remains required to obtain the handler address.
CUDA driver device lookup treats only `CUDA_ERROR_INVALID_DEVICE` as a
recoverable unavailable selected ordinal. Invalid values, uninitialized or
deinitialized driver state, invalid contexts, and every unexpected driver
status remain fatal driver failures.

Capability initialization is bound to the local hardware ordinal selected by
JAX. Before first PTX load in any XLA context, the private handler verifies that
the context's CUDA device is the exact qualified device. Driver and module
state intentionally remains process-lifetime because JAX owns context teardown.

Public identity constants:

- `FIRTH_COMPONENTS_FFI_TARGET`
- `FIRTH_COMPONENTS_FFI_API_VERSION`
- `FIRTH_COMPONENTS_HANDLER_SHA256`
- `FIRTH_COMPONENTS_MINIMUM_CUDA_DRIVER_VERSION`
- `FIRTH_COMPONENTS_MINIMUM_COMPUTE_CAPABILITY_MAJOR`
- `FIRTH_COMPONENTS_MINIMUM_COMPUTE_CAPABILITY_MINOR`
- `FIRTH_COMPONENTS_PTX_SHA256`
- `FIRTH_COMPONENTS_PTX_ISA`
- `FIRTH_COMPONENTS_PTX_TARGET`

The repository-private `native/cuda-driver` header is implementation support,
not a Rust crate dependency or public contract.
