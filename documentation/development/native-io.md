# Native I/O

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; development contract | Rust I/O and output runtime as of 2026-07-22 | Native runtime maintainers |

Native I/O owns the parts of the hot path that should not depend on Python
DataFrame libraries: BGEN decode, sample/covariate/phenotype alignment, chunk
delivery, output writing, manifest handling, and resume.

Integer boundary decisions for native I/O are maintained in
[Integer Policy](integer-policy.md) and [Integer Type Audit](integer-type-audit.md).

## Source Map

| Path | Responsibility |
| --- | --- |
| `crates/genotype/src/` | BGEN source ownership/index/decode/preprocess and genotype source planning. |
| `crates/input/src/` | Sample, phenotype, covariate, prediction-list, and LOCO prediction alignment. |
| `crates/output/src/` | Parquet dataset parts, manifests, resume, and bounded writer sessions. |
| `crates/engine/src/` | Production coordination of input, genotype delivery, output, telemetry facts, and cleanup. |
| `src/binding/` | NumPy/Python adaptation at the JAX backend boundary; no I/O policy. |

## BGEN Contract

The supported public genotype input is Layout 2 BGEN, including uncompressed,
zlib, and BGEN v1.3 Zstandard variant blocks. Native code owns:

- index/open path;
- sample-count validation against identities loaded from the required Oxford
  `.sample` file;
- variant metadata;
- dosage decode;
- missing-value representation;
- cached no-missing diploid packed8 compatibility validation and fast paths;
- chunk delivery in variant-major shape.

Python/JAX kernels should receive already aligned dosage or validated packed8
chunks and metadata, not parse file formats.

The reader copies BGEN files up to 256 MiB into one immutable owned snapshot.
Larger files retain the exact opened descriptor and use positioned reads into
one reusable owned window per active batch, never larger than 8 MiB. Byte
windows are independent of the 32-variant compute tile: each positioned window
is read once, then its borrowed bytes are decoded by parallel compute tiles.
Compatibility validation reuses the same byte-window policy. Decode and
raw-DEFLATE packing borrow directly from the snapshot when present; no safe API
is backed by a mutable file mapping. Snapshot indexing uses a concrete
bounds-checked slice cursor; positioned indexing streams bounded 8 MiB windows
instead of issuing one metadata read per variant. The opened descriptor and
configured path
are identified by device, inode, size, modification time, and change time and
rechecked around indexing and before a captured snapshot is published. A
published immutable payload isolates its readers from later in-place mutation
or configured-path replacement, so snapshot delivery and session finish do not
restat the source. Positioned batches recheck both the descriptor and
configured path after every delivered batch and at session finish. Concurrent
truncation, mutation, or path replacement of a positioned source therefore
returns an error without exposing mapped memory.

One private process-wide registry entry strongly owns the latest completely
parsed small-file payload by that full identity. The payload contains the
snapshot bytes, header properties, variant records, validated metadata, and
chromosome boundaries. An unchanged reopen shares this canonical payload even
after every previous reader closes. A different identity replaces the entry
atomically only after capture, open-time header/index validation, and final
identity verification; a candidate rejected before publication does not evict
the valid entry. Probability-semantic corruption discovered later during
compatibility validation or decode still fails safely, but its published
identity may already have replaced the earlier entry. Live readers may keep a
replaced payload alive.

The registry therefore retains up to 256 MiB of source bytes plus parsed index
and metadata allocations until a replacement passes open/index validation or
the process exits. The parsed allocations are outside the source-byte ceiling
and can be substantially larger for adversarial high-cardinality metadata.
Capturing and parsing a different identity overlaps the candidate with the
still-retained old entry. Each concurrent cold miss owns a separate candidate
outside the registry lock, so transient candidate memory scales with in-flight
opens. After publication, readers of the previous identity can extend the
overlap between old and new payloads.

Header length and first-variant offsets, zero-variant end-of-file placement,
reserved flag bits, embedded sample-block framing/counts, and variant-count
capacity are validated before the index allocates. Variant identifiers, RSIDs,
chromosome labels, and alleles must be valid UTF-8. Probability-pair validation
covers every stored sample, including samples omitted from a delivery
selection.

The BGEN index publishes variant metadata only through the validated
`g-genotype-contracts` constructors. Store construction performs one-time,
linear, always-on validation of parallel columns, identifier offsets, UTF-8
boundaries, and dictionary codes. Chunk slicing then performs only constant-time range
validation. A parser-created invariant failure is surfaced as contextual
`BgenError::InvalidFormat`; there is no public unchecked metadata constructor.

Optional `.bgi` input is deliberately deferred. The current runtime must remain
fully functional without an external index; future support is tracked in the
[Roadmap](roadmap.md) and must retain exact BGEN/index identity validation.

## Alignment Contract

Native alignment resolves:

- identities loaded from the required Oxford `.sample` file;
- phenotype and covariate rows;
- Step 1 prediction rows;
- fixed `(FID, IID)` identity matching;
- complete-case rows for phenotype/covariate data;
- per-phenotype or complete-case multi-phenotype grouping.

Changing alignment behavior is a public input contract change. Update
[Input Files](../public/input-files.md), tests, and parity expectations.

## Output Contract

Native output writes one result representation:

| Dataset | Directory | File names |
| --- | --- | --- |
| Parquet | `parts/` | `part_<first>[_<last>].parquet` |

Run directories also contain `run_manifest.json` and `effective_config.toml`.
The parts directory is the completed dataset; output does not require a
consolidation pass.

Output planning is read-only. The engine resolves every phenotype run path and
inspects any resume manifest before input preparation without creating output
directories. Directory and initial-manifest creation begins only during output
initialization, after the complete path plan, prepared inputs, and manifest
headers validate.

## Manifest And Resume Contract

`run_manifest.json` is the resume authority. Pre-release manifest schema version `0`
stores prepared-run fields, input fingerprints, and Parquet writer settings in
one canonical `execution_plan` object plus `execution_plan_hash`. Top-level
state is limited to schema and mutable lifecycle fields such as committed
chunks. The pre-release Parquet output schema is version `0`; `INFO` is nullable
when its expected-variance denominator is undefined.

Each compute group records a required fingerprint of its trait-major aligned
phenotype matrix, including phenotype names, shape, and float32 values. Resume
therefore rejects phenotype-value changes even if a source path changes between
input loading and manifest preparation.

Manifest construction records `association_backend.kind` from the
engine-resolved delivery format so resume and review tooling can distinguish
`jax_dosage` and `jax_packed8`; it is not a `RunPlan` field.

Resume always reconciles manifest chunk commits with files on disk. It reads
chunk commit metadata from Parquet footer metadata, compares every part to the
canonical output schema, and checks every commit against the current
chromosome-aware BGEN chunk plan. Parts without the current metadata are
rejected rather than reconstructed from result columns. BGEN compatibility is
bound to the exact opened file's device, inode, size, modification time, and
change time.

Compatibility validation must fail loudly on mismatched result-affecting inputs
or output schema assumptions.

## Testing

Native I/O changes usually need tests in:

- Rust unit tests under `crates/genotype/src/`, `crates/input/src/`, or `crates/output/src/`;
- integration or pipeline coverage in the owning Rust crate when backend delivery or writer sessions change.

Output contract changes also require [Output Files](../public/output-files.md)
and [Resume and Manifest](../public/resume-and-manifest.md) updates.

For BGEN performance work, the process-registry-empty and
process-registry-primed open contracts are deliberately separate:

```bash
GWAS_ENGINE_BGEN_BENCHMARK_PATH=/path/to/input.bgen \
  cargo bench -p g-genotype --features benchmark-internals --bench bgen_open_once

GWAS_ENGINE_BGEN_BENCHMARK_PATH=/path/to/input.bgen \
  cargo bench -p g-genotype --bench bgen_read -- \
  bgen_warmed_open_and_index/sequential_index

GWAS_ENGINE_BGEN_BENCHMARK_PATH=/path/to/input.bgen \
  cargo bench -p g-genotype --features benchmark-internals \
  --bench bgen_lifecycle_once
```

`bgen_open_once` starts with an empty process registry and reports first-open
capture/index time, same-process reopen/index time, a second reopen after both
readers have dropped, strong canonical hit status, retained source bytes, and
process resident memory before open and after reader drop. The retained byte
counter covers source bytes; the accompanying flag records that the canonical
payload also retains its parsed index and metadata. The Criterion group keeps
one priming reader alive and measures process-registry-primed reopen/index work.
For inputs at or below the owned-snapshot ceiling this is an unchanged
canonical-payload reopen; larger inputs use positioned I/O.
`bgen_lifecycle_once` reports open/index, preparation, aggregate time for all
full 16,384-variant packed8 batches, the final logical tail using the same fixed
production compute shape, policy-specific session finish, and their end-to-end
sum. Snapshot finish is a no-op because the payload is already immutable;
positioned finish verifies the source identity. Use the lifecycle total—not an
isolated open or batch result—when comparing source policies. Set
`GWAS_ENGINE_BGEN_BENCHMARK_PRIMED_REOPEN=1` to open and drop one priming
reader before timing the focused same-process reopen lifecycle. This leaves
only process-persistent state—not an extra live reader or index—during the
measurement.

The lifecycle input must be packed8-compatible: biallelic, diploid, unphased,
eight-bit, and without missing samples. The benchmark asserts that contract
instead of switching to dosage delivery. A paired compatibility-cache-primed
comparison must set `XDG_CACHE_HOME` to one fresh campaign-local directory, run
one unmeasured prewarm lifecycle, verify that it wrote a `compatible` marker,
and reuse that same marker state for every baseline and candidate process. An
empty compatibility-cache comparison must instead give every measured process
its own empty cache directory. Never let the first measured design create a
compatibility marker that only later positioned runs consume.

These process-registry and compatibility-cache labels say nothing about the
kernel filesystem page cache. A same-input source-policy campaign must give
both designs equivalent filesystem-cache state. The paired protocol performs
an untimed full read of the source immediately before every individual
baseline or candidate process, verifies the source size and digest on that
read, and alternates process order across blocks. Record that conditioning
method with the results. Without explicit cache eviction and supporting
evidence, describe these measurements as page-cache-primed rather than
filesystem-cold.

For an exact same-input comparison against positioned I/O, build `bgen_read`
with `--features benchmark-positioned-source`. This benchmark-only feature
exposes and selects an explicit zero-snapshot-limit benchmark constructor and
labels its open group `bgen_positioned_open_and_index`. Enabling the feature
does not change the production `BgenReaderCore::open` policy. Add
`benchmark-positioned-source` to the `bgen_lifecycle_once` feature list for the
corresponding end-to-end positioned measurement.
