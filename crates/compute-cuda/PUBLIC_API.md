# g-compute-cuda public API

The crate exposes only the stable internal Firth FFI target name, an opaque
device-capability proof, capability initialization, its error type, and the
capability-gated handler address. CUDA driver loading, PTX modules, kernel
launches, and XLA buffer validation remain private.

Capability initialization is bound to the local hardware ordinal selected by
JAX. Before first PTX load in any XLA context, the private handler verifies that
the context's CUDA device is the exact qualified device. Driver and module
state intentionally remains process-lifetime because JAX owns context teardown.

The repository-private `native/cuda-driver` header is implementation support,
not a Rust crate dependency or public contract.
