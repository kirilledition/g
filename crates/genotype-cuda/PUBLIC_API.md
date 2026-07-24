# g-genotype-cuda public API

The crate owns the stable internal packed8 DEFLATE FFI target name and JAX
registration API integer. It also exposes the build-verified embedded artifact
identity: the PTX SHA-256, declared PTX ISA, and declared PTX target. The build
script generates these PTX constants and the reviewed minimum CUDA driver and
compute-capability requirements from the verified checked-in artifact so
runtime registration and diagnostic provenance do not repeat literals.
It also generates a framed source/ABI SHA-256 over the native FFI wrapper,
nvCOMP ABI, embedded PTX, shared CUDA-driver support, and vendored XLA FFI ABI
headers. This is deliberately a semantic source-set identity, not a hash of
compiler-dependent native-library bytes.

Packed8 delivery is semantics-preserving and may fall back to host decoding,
so this artifact identity is diagnostic and does not participate in output
resume compatibility. Approximate-Firth raw CUDA has a separate, output-bound
artifact contract owned by `g-engine` and `g-output`.

Successful capability initialization retains the nvCOMP version, nvCOMP CUDA
runtime version, CUDA driver API version, CUDA-visible device ordinal,
compute-capability major and minor versions, and required DEFLATE input
alignment returned by the native C ABI. Read-only accessors make those
observations available for diagnostics without granting access to driver,
nvCOMP, or module state. The capability remains required to obtain the handler
address. The descriptor-failure status bit remains public for the private
binding diagnostic.
CUDA driver device lookup treats only `CUDA_ERROR_INVALID_DEVICE` as an
unavailable selected ordinal. Invalid values, uninitialized or deinitialized
driver state, invalid contexts, and every unexpected driver status remain
driver failures; packed8 fallback policy is applied only after that exact
native classification.

The repository-private `private-test-support` feature additionally exposes the
same handler address without a capability proof solely to verify that direct
use fails with `FailedPrecondition`. Production builds do not compile that
escape hatch.

Capability initialization is bound to the local hardware ordinal selected by
JAX. Before first PTX load in any XLA context, the private handler verifies that
the context's CUDA device is the exact qualified device. Every handler access
to nvCOMP also requires the qualified-device proof. Driver, nvCOMP, and module
state intentionally remains process-lifetime because JAX owns context teardown.

Public identity constants:

- `PACKED8_DEFLATE_FFI_TARGET`
- `PACKED8_DEFLATE_FFI_API_VERSION`
- `PACKED8_DEFLATE_HANDLER_SHA256`
- `PACKED8_DEFLATE_MINIMUM_CUDA_DRIVER_VERSION`
- `PACKED8_DEFLATE_MINIMUM_COMPUTE_CAPABILITY_MAJOR`
- `PACKED8_DEFLATE_MINIMUM_COMPUTE_CAPABILITY_MINOR`
- `PACKED8_DEFLATE_PTX_SHA256`
- `PACKED8_DEFLATE_PTX_ISA`
- `PACKED8_DEFLATE_PTX_TARGET`

The repository-private `native/cuda-driver` header is implementation support,
not a Rust crate dependency or public contract.
