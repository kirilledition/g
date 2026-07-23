# g-genotype-cuda public API

The crate exposes only the stable internal packed8 DEFLATE FFI target name, the
descriptor-failure status bit used by the private binding diagnostic,
opaque nvCOMP capability proof, capability initialization, its error type, and
the capability's required DEFLATE input alignment, and the capability-gated
handler address. CUDA driver and nvCOMP loading, embedded PTX modules, kernel
launches, workspace construction, descriptor validation, and XLA buffer
validation remain private.

The repository-private `private-test-support` feature additionally exposes the
same handler address without a capability proof solely to verify that direct
use fails with `FailedPrecondition`. Production builds do not compile that
escape hatch.

Capability initialization is bound to the local hardware ordinal selected by
JAX. Before first PTX load in any XLA context, the private handler verifies that
the context's CUDA device is the exact qualified device. Every handler access
to nvCOMP also requires the qualified-device proof. Driver, nvCOMP, and module
state intentionally remains process-lifetime because JAX owns context teardown.

The repository-private `native/cuda-driver` header is implementation support,
not a Rust crate dependency or public contract.
