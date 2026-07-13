# Native I/O

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; development contract | Rust I/O and output runtime as of 2026-07-10 | Native runtime maintainers |

Native I/O owns the parts of the hot path that should not depend on Python
DataFrame libraries: BGEN decode, sample/covariate/phenotype alignment, chunk
delivery, output writing, manifest handling, and resume.

Integer boundary decisions for native I/O are maintained in
[Integer Policy](integer-policy.md) and [Integer Type Audit](integer-type-audit.md).

## Source Map

| Path | Responsibility |
| --- | --- |
| `crates/genotype/src/` | BGEN mmap/index/decode/preprocess and genotype source planning. |
| `crates/input/src/` | Sample, phenotype, covariate, prediction-list, and LOCO prediction alignment. |
| `crates/output/src/` | Parquet dataset parts, manifests, resume, and bounded writer sessions. |
| `crates/engine/src/` | Production coordination of input, genotype delivery, output, telemetry facts, and cleanup. |
| `src/binding/` | NumPy/Python adaptation at the JAX backend boundary; no I/O policy. |

## BGEN Contract

The supported public genotype input is BGEN 1.2. Native code owns:

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

## Manifest And Resume Contract

`run_manifest.json` is the resume authority. Manifest schema version `14`
stores prepared-run fields, input fingerprints, and Parquet writer settings in
one canonical `execution_plan` object plus `execution_plan_hash`. Top-level
state is limited to schema and mutable lifecycle fields such as committed
chunks. The Parquet output schema remains version `3`.

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
