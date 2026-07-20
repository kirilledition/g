# g-genotype-cuda public API

The crate exposes only the stable internal packed8 DEFLATE FFI target name, an
opaque nvCOMP capability proof, capability initialization, its error type, and
the capability-gated handler address. CUDA driver and nvCOMP loading, embedded
PTX modules, kernel launches, workspace construction, descriptor validation,
and XLA buffer validation remain private.

The repository-private `native/cuda-driver` header is implementation support,
not a Rust crate dependency or public contract.
