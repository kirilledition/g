# Approximate-Firth CUDA kernel provenance

`firth_components_kernel.cu` is the maintained source for the embedded
`firth_components_kernel.compute_70.ptx` artifact. The frozen files have these
hashes:

- source: `sha256:4a823918e8b198ef8079cf54e159467c0942ee3d59c99924558d413f7c43585c`
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

CUDA 12.9.86 `ptxas` reports 38 registers, one barrier, 256 bytes of static
shared memory, no stack frame, and no spills when compiling the artifact for
`sm_70`. The PTX is loaded once per process-long XLA CUDA context. A partially
constructed module is unloaded if symbol lookup fails.

The kernel evaluates clipped logistic probability, genotype information,
score adjustment, score, and penalized deviance with `f64` accumulation. It
accepts any row-major batch prefix and treats the final input dimension as the
sample dimension. JAX `broadcast_all` batching gives every scalar operand the
same batch prefix, including lane-specific minimum-variance values.

The original V100 experiment and its numerical and application gates are
recorded in `documentation/scratchpad/performance.md`. Production validation
must update that ledger after rebuilding this source-owned PTX implementation.
