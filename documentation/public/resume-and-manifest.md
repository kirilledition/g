# Resume And Manifest

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-06-30 resume and manifest behavior | Public user docs |

This page is the canonical user-facing reference for resumable output runs.

For output layout and schemas, see [Output Files](output-files.md). For config
merge behavior, see [Configuration](configuration.md).

## Per-Phenotype Metadata

Every phenotype output run writes:

```text
effective_config.toml
run_manifest.json
```

`effective_config.toml` is the final merged config after packaged defaults, the
optional TOML file, and explicit CLI overrides.

`run_manifest.json` records execution-plan-affecting state, including:

- manifest and output schema versions;
- association mode;
- BGEN, sample, phenotype, covariate, prediction-list, and selected LOCO
  prediction-file fingerprints;
- phenotype name, covariate names, sample count, variant count, chunk size, and
  variant limit;
- multi-phenotype sample mode, phenotype compute-group identifier, sample-set
  fingerprint, covariate-design fingerprint, and prediction-alignment
  fingerprint;
- selected association backend such as `jax_dosage` or `jax_packed8`, with the
  resolved concrete GPU genotype format;
- binary correction plan and binary kernel settings when applicable;
- trusted BGEN policy, sample-key mode, JAX device/precision policy, dtype
  choices;
- output writer settings;
- committed chunk identifiers and Parquet part metadata.

Manifest schema version `11` stores immutable compatibility state once under
`execution_plan`, with its SHA-256 digest in `execution_plan_hash`. Top-level
fields are limited to manifest/output schema versions and mutable lifecycle
metadata such as status, committed chunks, command, runtime, and interruption
state. The Parquet output schema remains version `3`.

The manifest is the resume authority. It is intentionally stricter than a file
name check.

File fingerprints include resolved path, file size, and `mtime_ns`. Smaller
control files also include a SHA-256 content hash: sample, phenotype, covariate,
prediction-list, and LOCO prediction files referenced by the selected
phenotype or compute group. BGEN input fingerprints are metadata-only to avoid
hashing large genotype files during normal startup; their manifest field records
that metadata-only policy explicitly.

## Starting A New Run

Without `[output].resume = true`, `g` refuses to reuse a non-empty output run directory:

```text
Output run directory '<path>' already exists and is not empty. Enable [output].resume or choose a new output path.
```

Choose a new `--out` prefix, delete stale local output intentionally, or run
with `[output].resume = true` when the existing manifest belongs to the same planned run.

## Resume Controls

```toml
[output]
resume = true
resume_mode = "fast" # or "strict"
```

| Mode | Behavior |
| --- | --- |
| `fast` | Trust committed chunk identifiers recorded in `run_manifest.json` after manifest compatibility passes. |
| `strict` | Reconcile manifest chunk commits with chunk files on disk before resuming. |

Use `fast` for normal interruption recovery. Use `strict` after manual file
movement, storage failures, or any situation where the manifest and chunk files
might disagree.

Strict resume requires current chunk commit metadata in every Parquet part.
Parts without the native writer's `g.output.chunk_commits` footer metadata are
rejected instead of being reconstructed from data columns.

## Compatibility Checks

Resume first requires an existing schema-v11 `run_manifest.json`. It then
compares the current requested run against the canonical `execution_plan` and
its hash. A mismatch fails with a message naming the first incompatible
manifest field. Earlier manifest layouts are not adapted because the
application has no released legacy output contract.

Incompatible resume attempts are non-mutating: `run_manifest.json` remains
unchanged and `effective_config.toml` is not newly created or overwritten until
all selected phenotype output runs pass compatibility checks.

Common mismatch causes:

- changed BGEN, sample, phenotype, covariate, prediction-list, or selected
  LOCO prediction file;
- changed sample, phenotype, covariate, prediction-list, or selected LOCO
  content even when path, size, and `mtime_ns` are preserved;
- changed phenotype or covariate columns;
- changed trait mode, binary correction plan, or Firth settings;
- changed selected association backend;
- changed sample-key mode, multi-phenotype sample mode, aligned sample set,
  covariate design, or prediction alignment;
- changed chunk size, variant limit, public statistic output dtype, Parquet
  compression, writer grouping, or schema version;
- changed JAX precision/dtype or trusted BGEN policy.

Resume is not a way to combine different analyses into one output directory.

## Graceful Interruption

During `g regenie`, the first SIGINT or SIGTERM requests graceful shutdown. The
engine flushes queued chunks, saves committed output for resume, prints an
interruption message, and exits with `128 + signal_number` such as `130` for
SIGINT.

After that, rerun the same command with a config containing:

```toml
[output]
resume = true
resume_mode = "strict"
```

or use `fast` when the previous interruption was clean and storage is trusted.

## Parquet Parts And Resume

Committed Parquet parts are both the resumable unit and the completed dataset.
After interruption, resume writes only missing chunks and does not perform a
separate dataset consolidation step.
