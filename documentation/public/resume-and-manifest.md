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
- phenotype name, covariate names, sample count, variant count, and chunk size;
- multi-phenotype sample mode, phenotype compute-group identifier, sample-set
  fingerprint, covariate-design fingerprint, aligned phenotype-design
  fingerprint, and prediction-alignment fingerprint;
- selected association backend such as `jax_dosage` or `jax_packed8`, with the
  resolved genotype delivery format;
- binary correction plan and binary kernel settings when applicable;
- JAX device/precision policy and dtype choices;
- output writer settings;
- committed chunk identifiers and Parquet part metadata.

Manifest schema version `17` stores immutable compatibility state once under
`execution_plan`, with its SHA-256 digest in `execution_plan_hash`. Top-level
fields are limited to manifest/output schema versions and mutable lifecycle
metadata such as status, committed chunks, command, runtime, and interruption
state. The Parquet output schema remains version `3`.

The manifest is the resume authority. It is intentionally stricter than a file
name check.

Sample identity is always the non-empty `(FID, IID)` pair, so there is no
identity-mode field in the execution plan. Sample-file fingerprints and aligned
sample-set fingerprints cover the concrete identity data used by the run.
Prediction-alignment fingerprints bind the LOCO headers and chromosome rows
seen during indexing plus the concrete per-trait sample-alignment recipe.

File fingerprints include resolved path, file size, and `mtime_ns`. Smaller
control files also include a SHA-256 content hash: sample, phenotype, covariate,
prediction-list, and LOCO prediction files referenced by the selected
phenotype or compute group. BGEN identity comes from the exact file opened by
the native reader and includes device, inode, size, modification time, and
change time. This rejects replacement or in-place mutation without hashing the
large genotype file during normal startup.

## Starting A New Run

Without `[output].resume = true`, `g` refuses to reuse a non-empty output run directory:

```text
Output run directory '<path>' already exists and is not empty. Enable [output].resume or choose a new output path.
```

Choose a new `--out` prefix, delete stale local output intentionally, or run
with `[output].resume = true` when the existing manifest belongs to the same planned run.

## Enabling Resume

```toml
[output]
resume = true
```

Resume is always strict: after manifest compatibility passes, `g` reconciles
committed chunk identifiers with the chunk files on disk before continuing.
There is no public resume-validation mode.

Resume requires current chunk commit metadata in every Parquet part.
Parts without the native writer's `g.output.chunk_commits` footer metadata are
rejected instead of being reconstructed from data columns. Every part must use
the current production Parquet schema, and its footer commits must have unique
chunk identifiers, exact row counts, and ranges matching the current BGEN
chunk plan.

## Compatibility Checks

Resume first requires an existing schema-v17 `run_manifest.json`. It then
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
- changed multi-phenotype sample mode, aligned sample set, aligned phenotype or
  covariate design, or prediction alignment;
- changed chunk size or schema version;
- an application upgrade that changed fixed output-writer, result-dtype, or JAX
  policy recorded in the execution plan.

Resume is not a way to combine different analyses into one output directory.
Approximate-Firth manifests fingerprint the fixed inner proposal policy as
`float32_elementwise_float64_reduction`. Pre-v17 runs used the prior
all-`float64` inner policy and cannot be resumed into or mixed with current
output parts.

In particular, runs whose execution plan still contains the removed
`firth_newton_raphson_zero_start_iterations` field have a different plan schema
and hash. Start a new output directory instead of resuming those pre-release
binary runs with a newer build.

The scalar approximate-Firth solver is the only supported correction path.
Pre-release configurations and manifests must remove the experimental
`use_block_firth_math` option and its block-only coefficient, likelihood,
step-halving, and initial-response settings. Effective configuration metadata
uses option schema version `0`, and these fields are not accepted by manifest
schema version `17`; start a new output directory instead of resuming an older
block-Firth-compatible run.

## Graceful Interruption

During `g regenie`, the first SIGINT or SIGTERM requests graceful shutdown. The
engine flushes queued chunks, saves committed output for resume, prints an
interruption message, and exits with `128 + signal_number` such as `130` for
SIGINT.

After that, rerun the same command with a config containing:

```toml
[output]
resume = true
```

## Parquet Parts And Resume

Committed Parquet parts are both the resumable unit and the completed dataset.
After interruption, resume writes only missing chunks and does not perform a
separate dataset consolidation step.
