# Packed8 CUDA kernel provenance

`packed8_kernel.cu` is the maintained source for the embedded
`packed8_kernel.compute_70.ptx` artifact. The frozen files have these hashes:

- source: `sha256:e66b6c708d0fa2523213f7519f9313f6fee04b4fe62046785c441032ef74c1e4`
- PTX: `sha256:cb13cce735d0d44c932f342cb50a25e9cd350ad6abb7495282d664b517f26300`

The PTX was generated twice reproducibly with CUDA NVRTC 12.2.140 for
`compute_70`; it declares PTX ISA 8.2 and target `sm_70`. NVRTC is a generation
tool only and is not a build-time or runtime dependency of this crate. Native
initialization requires CUDA driver API version 12020 or newer because the
embedded artifact uses the CUDA 12.2 PTX ISA.

The private FFI accepts compressed bytes only from `g-genotype`'s trusted
packed8 transport. That transport is selected after the exact-source
compatibility scan has successfully decompressed and validated every member;
the read session retains the matching source identity so the BGEN cannot be
substituted during delivery. The device descriptor kernel still treats offsets,
sizes, and alignment as adversarial: it checks them before pointer formation
and redirects invalid metadata to a known-valid aligned empty-DEFLATE sentinel.
Arbitrary compressed bytes must not be passed directly to nvCOMP because its
API does not guarantee memory safety for corrupt streams.

Loaded modules are cached by CUDA context for the process lifetime. This relies
on JAX's process-long CUDA contexts; destroying a context and reusing its handle
within the same process is outside the supported lifecycle.

The exact embedded artifact was execution-validated on a V100 with the R535
driver against nvCOMP 5.3 in SLURM job 45171 and the official
`nvidia-libnvcomp-cu12==5.2.0.13` package in job 45172. Both runs matched the
CPU reference for identity, contiguous, and nonmonotonic indexed selection,
integer summaries, Adler-32 error reporting, and neutral compute-tail rows.

The hardened descriptor and row gates were execution-validated with this exact
PTX in SLURM job 45216 against nvCOMP 5.2.0.13 and 5.3.0.16. The standalone
proof covered identity and nonmonotonic indexed selection, compute-tail rows,
out-of-range and misaligned offsets, zero-length members, valid short output,
an injected nvCOMP failure status, and full-length Adler corruption. Invalid or
short rows produced neutral outputs without being read by the finalizer.
