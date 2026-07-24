# Packed8 CUDA kernel provenance

`packed8_kernel.cu` is the maintained source for the embedded
`packed8_kernel.compute_70.ptx` artifact. The frozen files have these hashes:

- source: `sha256:673df9629dcb5fec1fc9d688f16349eba7d75bb8a942724f7bcdcd0a0c5dbf1d`
- PTX: `sha256:a4b7b84171b6a78e6677a5fe1ba84fa6b4fd5a307eef198a5573fb83381ed088`

The crate build verifies both hashes before embedding the PTX. A source or PTX
change therefore requires an explicit provenance-hash update after regeneration
and review.

The PTX was generated twice reproducibly with CUDA NVRTC 12.2.140 for
`compute_70`; it declares PTX ISA 8.2 and target `sm_70`. NVRTC is a generation
tool only and is not a build-time or runtime dependency of this crate. Native
initialization requires CUDA driver API version 12020 or newer because the
embedded artifact uses the CUDA 12.2 PTX ISA.
The build verifies the pinned PTX digest, parses the `.version` and `.target`
directives, and generates both the public Rust artifact identity and the native
driver-qualification constants. Registration and native diagnostics therefore
consume one crate-owned identity rather than repeating PTX metadata literals.
The public handler digest additionally frames the FFI wrapper, nvCOMP ABI, PTX,
shared CUDA-driver support, and vendored XLA FFI headers. It identifies that
semantic source/ABI set rather than compiler-dependent native-library bytes.
Packed8 delivery remains semantics-preserving and can fall back to host
decoding, so this identity is diagnostic rather than output resume authority.

The finalizer computes the packed8 genotype mean with explicit
`cvt.rn.f32.u64`, `mul.rn.f32`, and `div.rn.f32` instructions. This preserves
the host's sequential float32 conversion and operations instead of allowing an
XLA consumer to reassociate the scale and sample-count division.

The finalizer partitions every BGEN row byte exactly once among the CUDA
threads, computes Adler-32 from unreduced integer byte and weighted-byte sums,
and reduces those sums and the packed8 statistics through warp shuffles. An
identity sample selection emits probabilities and accumulates statistics during
that same source pass; other selection modes retain the indexed gather pass.
The private FFI rejects source sample counts above 126,789,562, the largest
count for which the unreduced Adler weighted sum is proven to fit in `uint64_t`
for a `3 * sample_count + 10` byte packed8 row.

CUDA 12.4 `ptxas` reports 40 registers, 360 bytes of static shared memory, no
stack frame, and no spills for the finalizer on `sm_70`. The generated finalizer
contains two block barriers and no integer divide or remainder instructions;
the descriptor kernel retains its separate dynamic-alignment remainder.

The private FFI accepts compressed bytes only from `g-genotype`'s trusted
packed8 transport. That transport is selected after the exact-source
compatibility scan has successfully decompressed and validated every member;
the read session retains the matching source identity so the BGEN cannot be
substituted during delivery. The device descriptor kernel still treats offsets,
sizes, and alignment as adversarial: it checks them before pointer formation
and redirects invalid metadata to a known-valid aligned empty-DEFLATE sentinel.
Arbitrary compressed bytes must not be passed directly to nvCOMP because its
API does not guarantee memory safety for corrupt streams.

Loaded modules are cached by CUDA context for the process lifetime, after the
context's CUDA device is proven equal to the device obtained from the
JAX-selected local hardware ordinal. The handler establishes that proof before
accessing nvCOMP or allocating decode workspace; direct use before capability
qualification fails with `FailedPrecondition`. Driver, nvCOMP, and module state
is intentionally retained until process exit because JAX may destroy its
contexts before C++ static teardown. Destroying a context and reusing its handle
within the same process remains outside the supported lifecycle.

The compressed decoder was execution-validated on a V100 with the R535 driver
against nvCOMP 5.3 in SLURM job 45171 and the official
`nvidia-libnvcomp-cu12==5.2.0.13` package in job 45172. Both runs matched the
CPU reference for identity, contiguous, and nonmonotonic indexed selection,
integer summaries, Adler-32 error reporting, and neutral compute-tail rows.

The hardened descriptor and row gates were execution-validated in SLURM job
45216 against nvCOMP 5.2.0.13 and 5.3.0.16. The standalone
proof covered identity and nonmonotonic indexed selection, compute-tail rows,
out-of-range and misaligned offsets, zero-length members, valid short output,
an injected nvCOMP failure status, and full-length Adler corruption. Invalid or
short rows produced neutral outputs without being read by the finalizer.

The original seven-result FFI was execution-validated in SLURM job 45263.
Full and partial 16,384-variant batches matched the canonical host decoder
bit-for-bit for probability bytes, integer summaries, status values, neutral
compute-tail rows, and genotype means.

The current fused finalizer PTX was execution-validated on a V100 in SLURM
jobs 45291 and 45293. The production-path diagnostic matched canonical host
inputs, decoded dosage, score results, and full approximate-Firth results
bit-for-bit. The direct FFI diagnostic covered full and tail batches,
contiguous and nonmonotonic indexed selections, an out-of-range selected index,
Adler-32 corruption, and an invalid descriptor; valid results matched exactly,
and error rows retained their established status and neutral-output contracts.
