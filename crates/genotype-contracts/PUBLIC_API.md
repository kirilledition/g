# Public API

## This crate owns

Canonical, allocation-preserving data-plane contracts shared between native
genotype producers, the engine, and output consumers.

## Public contracts

The exact opened BGEN source identity, authoritative owned-snapshot content
fingerprint, per-request source provenance, compact shared variant metadata
storage and slices, output-facing chunk statistics, packed nullable `f32`
columns, and the canonical raw-DEFLATE member alignment used by slab producers
and consumers.
`BgenContentSha256` stores exactly 32 private digest bytes and accepts only the
canonical 64-character lowercase hexadecimal representation through parsing
and Serde. `BgenContentFingerprint` binds that digest to the exact source byte
count. `BgenContentEvidence` distinguishes an authenticated owned snapshot from
a positioned source whose content is unattested. `BgenSourceProvenance`
separates the request locator from descriptor metadata recorded when the source
was opened and records whether the request acquired its locator or reused a
process snapshot. That descriptor metadata is provenance rather than content
authority; positioned sources remain mutable and unattested.
`VariantMetadataInvariantError` is the dependency-free typed construction error
for malformed parallel columns, identifier offsets, dictionary codes, and slice
ranges. `VariantMetadataStore::from_parts` and `VariantMetadataColumns::new`
return `Result`; there is no public unchecked construction path.

## This crate must not expose

BGEN readers or decoding policy, compute-only statistics, Arrow arrays or
Parquet writers, engine scheduling, runtime policy, or Python bindings.

## Performance constraints

Metadata slices retain one immutable dictionary-coded store through
`Arc` plus a range. Nullable columns retain dense values and an Arrow-compatible
packed validity bitmap. Consumers must move or share these allocations; they
must not introduce row-wise mirrors or crate-boundary copies. Store construction
performs one-time linear validation before publication. Range construction is
constant-time, and accessors do not repeat invariant checks on the hot path.

## Allowed downstream users

`g-interface`, `g-plan`, `g-genotype`, `g-genotype-cuda`, `g-engine`, and
`g-output`.
