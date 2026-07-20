# g-compute-cuda public API

The crate exposes only the stable internal Firth FFI target name, an opaque
device-capability proof, capability initialization, its error type, and the
capability-gated handler address. CUDA driver loading, PTX modules, kernel
launches, and XLA buffer validation remain private.

The repository-private `native/cuda-driver` header is implementation support,
not a Rust crate dependency or public contract.
