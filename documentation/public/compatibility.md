# Compatibility

This page is the canonical public compatibility and scope reference.

`g` is pre-release. Backward compatibility is not guaranteed until a stable
release line exists.

## Supported Surface

| Area | Status |
| --- | --- |
| REGENIE Step 2 quantitative traits | Supported with `--step 2 --qt`. |
| REGENIE Step 2 binary score test | Supported with `--step 2 --bt`. |
| Binary approximate Firth fallback | Experimental with `--bt --firth --approx`; not production-stable until upstream golden parity is added. |
| BGEN 1.2 input | Supported. |
| Oxford `.sample` files | Supported. |
| Embedded BGEN sample identifiers | Supported when compatible with sample-key mode. |
| Multiple phenotypes | Supported with per-phenotype semantics by default. |
| Output formats | Arrow, Parquet dataset parts, optional finalized Parquet, and REGENIE Step 2-style text. |
| GPU execution | Supported through JAX when the environment exposes a compatible accelerator. |
| TOML config | Supported through `--config`. |
| Python API | Supported as a small execution wrapper; see [Python API](api-python.md). |

## Not Implemented

| Area | Behavior |
| --- | --- |
| REGENIE Step 1 | Not implemented. Use upstream `regenie` and pass `--pred`. |
| PLINK BED input | `--bed` is outside the current option surface and fails as unknown. |
| PLINK2 PGEN input | `--pgen` is outside the current option surface and fails as unknown. |
| Sample/variant filters | `--keep`, `--remove`, `--extract`, and `--exclude` fail as unknown. |
| Categorical covariates | `--catCovarList` fails as unknown. |
| SPA fallback | `--spa` fails as unknown. |
| Exact Firth without `--approx` | Recognized and rejected. |
| Alternative tests and time-to-event traits | `--test` and `--t2e` fail as unknown. |

Unsupported flags fail loudly so REGENIE command migration does not silently
drop scientific intent.

## REGENIE Command Migration

The supported command is intentionally close to REGENIE Step 2:

```bash
g regenie --step 2 --qt --bgen ... --phenoFile ... --phenoCol ... --pred ... --out ...
```

Important migration limits:

- Replace Step 1 commands with upstream `regenie`, not `g`.
- Keep BGEN Step 2 inputs; BED/PGEN Step 2 inputs are not accepted.
- Compare equivalent statistical modes only. A binary score-only `g` run should
  not be compared to upstream REGENIE output that used approximate Firth.
- Treat `--bt --firth --approx` as experimental until the pre-release parity
  suite includes an upstream golden approximate-Firth fixture.
- REGENIE text output is selected with `--format regenie`; Arrow and
  Parquet are the performance-oriented defaults for this engine.

## Versioning Expectations

Until the project declares a stable release:

- CLI and TOML behavior in the current checkout is authoritative.
- Defaults can change; use `src/interface/config.default.toml` for the exact
  current values.
- Output schema changes are guarded by manifest/schema versions but may still
  evolve.
- Performance assumptions are workload-dependent and should be re-measured on
  the target machine.
