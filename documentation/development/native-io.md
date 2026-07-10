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
| `crates/genotype/src/` | BGEN mmap/index/decode/preprocess/profile and genotype source planning. |
| `crates/input/src/` | Sample, phenotype, covariate, prediction-list, and LOCO prediction alignment. |
| `crates/output/src/` | Arrow IPC chunks, Parquet parts/finalization, REGENIE text, manifests, resume, and writer sessions. |
| `crates/engine/src/` | Production coordination of input, genotype delivery, output, and cleanup. |
| `src/binding/` | NumPy/Python adaptation at the JAX backend boundary; no I/O policy. |

## BGEN Contract

The supported public genotype input is BGEN 1.2. Native code owns:

- index/open path;
- sample identifier extraction;
- variant metadata;
- dosage decode;
- missing-value representation;
- trusted no-missing diploid validation and fast paths;
- chunk delivery in variant-major shape.

Python/JAX kernels should receive already aligned dosage chunks and metadata,
not parse file formats.

## Alignment Contract

Native alignment resolves:

- sample file identities and identifier views supplied by genotype readers;
- phenotype and covariate rows;
- Step 1 prediction rows;
- sample-key mode;
- complete-case rows for phenotype/covariate data;
- per-phenotype or complete-case multi-phenotype grouping.

Changing alignment behavior is a public input contract change. Update
[Input Files](../public/input-files.md), tests, and parity expectations.

## Output Contract

Native output owns chunk persistence and finalization:

| Format | Directory | File names |
| --- | --- | --- |
| Arrow | `chunks/` | `chunk_<first>[_<last>].arrow` |
| Parquet | `parts/` | `part_<first>[_<last>].parquet` |
| REGENIE text | `regenie/` | `part_<first>[_<last>].regenie` plus sidecar JSON |

Run directories also contain `run_manifest.json` and `effective_config.toml`.
Finalization can write `final.parquet` or `final.regenie` depending on output
format and options.

## Manifest And Resume Contract

`run_manifest.json` is the resume authority. It records prepared-run fields,
input fingerprints, output writer settings, committed chunks, schema versions,
and finalization metadata.

The immutable prepared run plan includes `association_backend.kind` so resume and
review tooling can distinguish `jax_dosage` and `jax_packed8` execution without
inferring from lower-level genotype or device fields.

Resume modes:

- `fast` trusts manifest committed chunks after compatibility validation;
- `strict` reconciles manifest chunk commits with files on disk.

Strict resume reads chunk commit metadata from Arrow schema metadata, Parquet
footer metadata, or REGENIE text sidecars. Do not add metadata-free Arrow
fallbacks; pre-release output formats are allowed to require current metadata.

Compatibility validation must fail loudly on mismatched result-affecting inputs
or output schema assumptions.

## Testing

Native I/O changes usually need tests in:

- Rust unit tests under `crates/genotype/src/`, `crates/input/src/`, or `crates/output/src/`;
- integration or pipeline coverage in the owning Rust crate when backend delivery or writer sessions change.

Output contract changes also require [Output Files](../public/output-files.md)
and [Resume and Manifest](../public/resume-and-manifest.md) updates.
