# Approximate-Firth CUDA kernel provenance

`firth_components_kernel.cu` is the maintained source for the embedded
`firth_components_kernel.compute_70.ptx` artifact. The frozen files have these
hashes:

- source: `sha256:1d15fd1aad609023c849942478764c8d2c67a74ff5acd0909652f2dfa180fce0`
- PTX: `sha256:a22c9866447f21c7f7cd484ec1e12c3c249a5a84acf3850cb3eb3a56697c736f`

The PTX was generated twice reproducibly with the official
`nvidia-cuda-nvrtc-cu12==12.2.140` wheel using NVRTC options
`--gpu-architecture=compute_70`, `--std=c++17`, and `--fmad=true`. The loaded
`libnvrtc.so.12` has
`sha256:0a18d98e687e24fa33cceb56a0c1d25b4e70a0723bc4e1975a5b78ebb1bf4813`.
The artifact declares PTX ISA 8.2 and target `sm_70`. NVRTC is a generation
tool only and is not a build-time or runtime dependency. Native initialization
requires CUDA driver API version 12.2 or newer and compute capability 7.0 or
newer; unsupported configurations retain the pure-JAX implementation.
The build verifies the pinned PTX digest, parses the `.version` and `.target`
directives, and generates both the public Rust artifact identity and the native
driver-qualification constants. Registration, compatibility output, and native
diagnostics therefore consume one crate-owned identity rather than repeating
PTX metadata literals.
The public handler digest additionally frames the FFI wrapper, PTX, shared
CUDA-driver support, and vendored XLA FFI headers so wrapper-only
launch-contract changes cannot retain the same resume identity. It identifies
that semantic source/ABI set rather than compiler-dependent native-library
bytes.

CUDA 12.9.86 `ptxas` reports 38 registers, one barrier, 256 bytes of static
shared memory, no stack frame, and no spills when compiling the artifact for
`sm_70`. The PTX is loaded once per process-long XLA CUDA context, only after
the context's CUDA device is proven equal to the device obtained from the
JAX-selected local hardware ordinal. A partially constructed module is unloaded
if symbol lookup fails. Successfully cached driver and module state is retained
until process exit; it is not destroyed after JAX may have torn down contexts.

The kernel evaluates clipped logistic probability, genotype information,
score adjustment, score, and penalized deviance with `f64` accumulation. It
accepts any row-major batch prefix and treats the final input dimension as the
sample dimension. JAX `broadcast_all` batching gives every scalar operand the
same batch prefix, including lane-specific minimum-variance values.

The original V100 experiment and its numerical and application gates are
recorded in `documentation/scratchpad/performance.md`. Production validation
must update that ledger after rebuilding this source-owned PTX implementation.
