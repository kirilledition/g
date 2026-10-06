# Packed8 CUDA kernel provenance

`packed8_kernel.cu` is the maintained source for the embedded
`packed8_kernel.compute_70.ptx` artifact. The frozen files have these hashes:

- source: `sha256:5d08dac719a190b201f9c11cdf4a031fb69e4f02b4a20adef1e4837b6842b180`
- PTX: `sha256:3e89b8d8277de8b10113c4d0d27979a1449e34593c4141c9dc1a230137fdba32`

The crate build verifies both hashes before embedding the PTX. A source or PTX
change therefore requires an explicit provenance-hash update after regeneration
and review.

The PTX was generated twice reproducibly with CUDA NVRTC 12.2.140 for
`compute_70`; it declares PTX ISA 8.2 and target `sm_70`. NVRTC is a generation
tool only and is not a build-time or runtime dependency of this crate. Native
initialization requires CUDA driver API version 12020 or newer because the
embedded artifact uses the CUDA 12.2 PTX ISA.

The association finalizer returns six buffers: probability pairs, exact dosage
sums, exact dosage-square sums, zero counts, homozygous-alternate counts, and
validation statuses. Floating-point means are derived in Python from the exact
integer totals; the old float32 native mean was unused and is no longer emitted.
The private target version is `g.bgen.packed8_deflate.v2`, so previously cached
seven-result executables cannot be paired with its changed result signature.

The source-only target `g.bgen.packed8_source_deflate.v1` returns just full-source
probability pairs and validation statuses. Its finalizer specializes the same
row validation code at compile time, omitting moment and count accumulation,
reductions, and stores while preserving Adler-32, descriptor, length, ploidy,
header, and probability validation. The host handler enforces identity source
selection. Group selection computes its own private statistics afterward.

The finalizer partitions every BGEN row byte exactly once among the CUDA
threads, computes Adler-32 from unreduced integer byte and weighted-byte sums,
and reduces those sums and the packed8 statistics through warp shuffles. An
identity sample selection emits probabilities and accumulates statistics during
that same source pass; other selection modes retain the indexed gather pass.
The private FFI rejects source sample counts above 126,789,562, the largest
count for which the unreduced Adler weighted sum is proven to fit in `uint64_t`
for a `3 * sample_count + 10` byte packed8 row.

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


The six-result and source-only handlers were qualified on Landau V100 in SLURM
job 52101. Full (16,384 variants) and tail (9,343 logical variants) batches,
identity, contiguous, duplicate indexed, and invalid indexed selections matched
the prior decoder exactly for every retained field. The source specialization
matched probability bytes and statuses for every row gate, header/ploidy/pair
failure, Adler failure, and neutral tail. A resident diagnostic passed null
statistic pointers and injected nvCOMP failure statuses without accesses.

CUDA 12.9 `ptxas` reports 40 registers and 360 bytes of shared memory for the
association finalizer, and 43 registers and 168 bytes for the source finalizer;
both have no stack frame or spills. The warmed full-FFI stage measured 6.385 ms
for both association ABIs and 6.308 ms for source-only decoding in this campaign.
These stage measurements do not establish a whole-application speedup.

SLURM job 52114 screened source-only blocks of 64, 128, 256, and 512 threads
against full/tail 2,504-sample batches and a 100,000-sample batch. Every variant
preserved exact probabilities and statuses, including injected row gates. The
128-thread variant improved the full/tail resident finalizer but regressed the
large cohort; 512 threads reversed that tradeoff. The maintained source retains
256 threads to avoid an unsupported sample-count-dependent launch policy.
