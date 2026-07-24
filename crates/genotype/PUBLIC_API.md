# Public API

## This crate owns

BGEN source ownership, index, and decode for uncompressed, zlib, and Zstandard
Layout 2 files,
immutable per-delivery read sessions, chromosome-homogeneous chunk planning,
owned decoded genotype batches, and genotype preprocessing summaries. It also
owns validated zlib-to-raw-DEFLATE host packing, compressed transfer
descriptors, and exact device-summary conversion for the compressed GPU
delivery path.

## Public types

`BgenReaderCore`, `BgenReadSession`, `BgenOpenRequest`,
`BgenContentSelector`, `BgenError`,
`GenotypeError`/`GenotypeResult`, chunk and statistics contracts, the canonical
`GenotypeBatch`/`GenotypeBatchPayload`, `OwnedGenotypeBuffer`,
`PooledPacked8Buffer`, and the typed `Packed8Compatibility` negotiation result.
Compressed delivery exposes `CompressedPacked8Batch`, its opaque run-planned
layout, one immutable session transfer descriptor with contiguous or indexed
sample selection, and the canonical exact-integer device statistics payload.
Backend capability and group preparation belong to `g-engine`. Shared
output-facing metadata and columns are owned by `g-genotype-contracts` and are
not re-exported here.

## Public functions

`BgenReaderCore::open` is the temporary unselected compatibility entry point:
it always acquires and parses the supplied locator and never consults or
publishes the process snapshot cache. `BgenReaderCore::open_request`
additionally accepts an authoritative SHA-256 selector and optional expected
byte count. Selected small-file opens reuse an already authenticated matching
process snapshot without resolving, opening, stating, or canonicalizing the
supplied locator. A selected miss captures the locator bytes, verifies the
selector, and publishes only after the full header and index parse succeeds.
Selected inputs that cannot fit the 256 MiB owned-snapshot ceiling are rejected
with a typed error rather than silently falling back to unattested positioned
I/O.

The reader exposes content evidence and request provenance separately.
`content_evidence()` returns the authority for the source actually opened or
reused. `source_provenance()` records both the original capture identity and
the locator supplied by the current request. On a selected cache hit that
request locator was not accessed and remains provenance rather than content
authority.

`BgenReaderCore` reads metadata, plans full-scan chromosome-homogeneous chunks,
resolves packed8 compatibility through a best-effort persistent fingerprint
cache tied to authoritative content, and validates the header sample count
against aligned sample metadata through `validate_expected_sample_count`. It
creates a `BgenReadSession` from one sample selection. The immutable session
uses the crate-owned 32-variant decode-tile policy and decodes owned
variant-major dosage or validated packed8 batches. Sample selection is
session-local, decode-tile policy is private and fixed, and no process-global
decode state is exposed. Source metadata is captured from the opened descriptor
and verified while indexing. Positioned sources are also rechecked before each
session and after delivery. The compatibility cache stores typed compatible and
dosage-required outcomes for authoritative owned content so an incompatible
snapshot is not rescanned on every run. Positioned unattested sources do not
read or write persistent compatibility markers.

Files no larger than 256 MiB are copied once into a private immutable byte
vector; indexing, decode, compatibility validation, and raw-DEFLATE packing
borrow directly from that snapshot. Larger files retain the exact opened
descriptor and use positioned reads into reusable owned windows capped at 8
MiB. One positioned window is read per byte-bounded range and borrowed by its
parallel 32-variant compute tiles; compatibility validation uses the same
coalescing policy. Snapshot indexing uses a concrete bounds-checked slice
cursor, while positioned indexing streams bounded source windows. The reader
never exposes a safe file-backed memory map.
The opened descriptor identity is checked after an owned snapshot capture.
Deletion or retargeting of the configured locator after that capture is
provenance-only: parsing continues from the immutable bytes already read. Once
published, that payload isolates its readers from later in-place mutation or
configured-path replacement, so snapshot sessions deliberately require no
per-delivery or terminal restat. Positioned batches recheck both the descriptor
and configured-path identities after every delivered batch and at session
finish. Concurrent truncation or replacement of a positioned source therefore
becomes a typed error rather than an invalid memory access.

A private one-entry process registry strongly owns the most recent completely
parsed selected small-file payload under a revision-0 key containing its full
SHA-256 and byte count. A digest-only lookup inherits the stored byte count; an
explicit byte count is an additional assertion. Selected hits share the
canonical payload even after all earlier readers close. Unselected opens never
consult or replace the entry, including when their bytes are identical. A
different selected fingerprint replaces the entry atomically only after
capture, digest verification, and open-time header/index validation; a rejected
candidate does not evict the valid entry. Concurrent misses parse independently
outside the registry lock and canonicalize the matching payload at publication.
Readers of an older fingerprint keep their payload alive across a replacement.

The registry retains up to 256 MiB of source bytes plus parsed index and
metadata allocations until a replacement passes open/index validation or the
process exits. Those parsed allocations are additional to the source-byte
ceiling and adversarial high-cardinality metadata can make them substantially
larger. Capturing and parsing a different identity overlaps the candidate with
the still-retained old entry. Each concurrent cold miss owns a separate
candidate outside the registry lock, so transient candidate memory scales with
in-flight opens. After publication, live readers of the previous identity can
extend the overlap between old and new payloads.

For authoritative owned content, the persistent packed8 marker uses a
revision-0 fingerprint over the content digest, content byte count, sample
count, and variant count. It contains no path or filesystem metadata. For
compatible zlib sources, `BgenReaderCore` derives one opaque compressed slab
layout from the actual pending chunk plan; non-zlib sources return no compressed
layout so callers can retain host delivery. `BgenReadSession` exposes invariant
sample-transfer geometry once and packs each logical chunk as aligned raw-DEFLATE
members plus interleaved offset, size, and Adler-32 metadata. Compute-only tails
remain in the canonical batch geometry and are not fabricated as compressed
members. Device raw sums, square sums, and status codes are validated and
converted to output allele frequency, observation count, and INFO through the
same per-variant formula used by host preprocessing.

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
Batch geometry exists only on `GenotypeBatch`; compressed storage contains only
the slab and member metadata it owns.

BGEN indexing constructs shared metadata through the always-on contracts
validator once. A parser-created invariant failure is reported contextually as
`BgenError::InvalidFormat`; successful chunk access retains the existing shared
store and performs only constant-time range validation. Header offsets, sample
block framing, reserved flags, source length, and variant-count capacity are
checked before allocation; all BGEN variant identifiers, RSIDs, chromosome
labels, and alleles must contain valid UTF-8.

## Allowed downstream users

`g-engine` and the private root PyO3 `AssociationBackend` adapter. The adapter
consumes genotype-owned buffer and statistics types directly without invoking
reader services.
