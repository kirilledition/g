# Public API

## This crate owns

BGEN mmap/index/decode for uncompressed, zlib, and Zstandard Layout 2 files,
immutable per-delivery read sessions,
chromosome-homogeneous chunk planning, owned decoded genotype batches, and
genotype preprocessing summaries. It also owns validated zlib-to-raw-DEFLATE
host packing for the compressed GPU delivery path.

## Public types

`BgenReaderCore`, `BgenReadSession`, `BgenError`,
`GenotypeError`/`GenotypeResult`, chunk and statistics contracts,
`DecodedGenotypeBatch`, `OwnedGenotypeBuffer`, `PooledPacked8Buffer`, and the typed
`Packed8Compatibility` negotiation result. Compressed delivery exposes
`CompressedPacked8Batch`, its opaque run-planned layout, and one immutable
session transfer descriptor with contiguous or indexed sample selection. Shared output-facing
metadata and columns are owned by `g-genotype-contracts` and are not
re-exported here.

## Public functions

`BgenReaderCore` opens and indexes a BGEN, reads metadata, plans
full-scan chromosome-homogeneous chunks, resolves packed8 compatibility through
a best-effort persistent fingerprint cache tied to the exact opened file, and
creates a `BgenReadSession` from one sample selection. The immutable session
uses the crate-owned 32-variant decode-tile policy and decodes owned
variant-major dosage or validated packed8 batches. Sample selection is
session-local, decode-tile policy is private and fixed, and no process-global
decode state is exposed. Source metadata is captured from the opened file and
rechecked after indexing, before each session, and after delivery. The exact
opened-file identity is also exposed to the engine for strict output-manifest
compatibility; callers do not restat it from the configured path. The cache
stores typed compatible and dosage-required outcomes so an incompatible source
is not rescanned on every run.

For compatible zlib sources, `BgenReaderCore` derives one opaque compressed slab
layout from the actual pending chunk plan. `BgenReadSession` exposes invariant
sample-transfer geometry once and packs each logical chunk as aligned raw-DEFLATE
members plus interleaved offset, size, and Adler-32 metadata. Compute-only tails
remain scalar geometry and are not fabricated as compressed members.

## This crate must not expose

Sample/phenotype alignment, output schema/writers, runtime/JAX policy, engine
scheduling, callback queues, PyO3 classes, or public `debug`, `ffi`, or
`internal` modules.

## Performance constraints

Batch-oriented APIs only. Owned decode reserves the final typed vector,
initializes its spare capacity, and publishes its length only after every tile
succeeds. Packed8 delivery reuses a session-owned bounded pool; the immutable
pooled buffer can outlive its read session and returns its allocation only when
the final downstream owner drops. No raw address/count wrapper crosses the crate boundary. Produce
immutable contract metadata slices, store variant IDs in a contiguous UTF-8
arena, dictionary-code repeated chromosome/allele text, reuse the dictionary
code across each contiguous chromosome run, and avoid per-variant heap
ownership, public calls, or JSON conversion. Normal
preprocessing reuses the dosage-sum allocation for the final genotype-mean
vector after INFO and sparse-candidate calculations; output observation counts
remain output-only and are not cloned or transferred to JAX. Nullable INFO
statistics use a contiguous `f32` value column and an Arrow-compatible packed
validity bitmap instead of per-value `Option` storage.
Compressed batches use the run-planned maximum actual slab length so every hot
JAX submission retains one input shape. The bounded session pool keeps that
storage initialized at full length across drops; each pack overwrites only real
member bytes and logical metadata, without clearing alignment gaps or the unused
suffix. Indexed sample conversion is allocated once per session, while identity
selection is represented as a zero-start contiguous range.

## Allowed downstream users

`g-engine` and the private root PyO3 `AssociationBackend` adapter. The adapter
consumes genotype-owned buffer and statistics types directly without invoking
reader services.
