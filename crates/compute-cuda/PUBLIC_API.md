# g-compute-cuda public API

The crate exposes only the stable internal Firth FFI target name, an opaque
device-capability proof, capability initialization, its error type, and the
capability-gated handler address. CUDA driver loading, PTX modules, kernel
launches, and XLA buffer validation remain private.
