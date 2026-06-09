# Testing And Parity

| Status | Applies to | Owner |
| --- | --- | --- |
| Development contract | Correctness tests, parity checks, CLI/config validation, and native/JAX boundaries | Development maintainers |

This page defines how to validate changes without confusing fast local checks,
full correctness runs, and external REGENIE parity work.

## Test Tiers

| Tier | Commands | Use when |
| --- | --- | --- |
| Local focused | `just check-local`, `just test-local-focused` | Fast agent iteration and documentation-adjacent changes. |
| Local full Python | `just test-local` | Broader Python behavior without large native/GPU work. |
| Repository checks | `just format`, `just lint`, `just typecheck`, `just check` | Pre-merge quality gates on a suitable host. |
| Full tests | `just test` | Full suite on a host where native builds and larger tests are appropriate. |
| SLURM CPU/GPU | Server recipes in [Server Gauss SLURM](server-gauss-slurm.md) | Heavy CPU validation, GPU validation, large suites, and benchmarks. |

Do not run GPU workloads, heavy compilation, large test suites, or benchmark
sweeps on a login node.

## Correctness Ownership

| Area | Representative tests |
| --- | --- |
| CLI and lifecycle | `tests/test_cli.py`, `tests/test_cli_smoke.py` |
| TOML/options/config | `tests/test_interface.py`, `tests/test_api.py`, `tests/test_preflight.py` |
| Input parsing and sample alignment | `tests/test_io_source.py`, `tests/test_io_sample.py`, `tests/test_tabular.py` |
| Output writer, schema, manifest, resume | `tests/test_io_output.py`, Rust tests under `src/output/` |
| Quantitative kernels | `tests/test_regenie2_linear.py`, `tests/test_regenie2_parity.py` |
| Binary score and Firth kernels | `tests/test_regenie2_binary*.py`, `tests/test_regenie_binary_correction_contract.py` |
| Pipeline orchestration | `tests/test_regenie2_pipeline.py`, `tests/test_callback_lifecycle.py` |
| JAX runtime | `tests/test_jax_setup.py`, `tests/test_warm_cache.py` |
| Telemetry and timing | `tests/test_telemetry.py`, `tests/test_timing.py` |
| Development tooling | `tests/test_tooling_architecture.py`, `tests/test_regenie_comparison_scripts.py` |

Use the narrowest test that covers the changed contract, then broaden when a
change crosses module boundaries.

## REGENIE Parity Rules

Parity checks must compare equivalent statistical modes:

- quantitative Step 2 to quantitative Step 2;
- binary score-only to binary score-only;
- approximate Firth only when both tools use approximate Firth with the same
  fallback threshold;
- same phenotype, covariates, Step 1 predictions, sample identity mode, and
  genotype source.

Do not treat differences caused by different complete-case sample sets as kernel
bugs until input alignment has been verified.

## Numerical Expectations

Result statistics are public `float32` outputs. Some internal kernels can use
wider dtypes for parity-sensitive work. When validating a numerical change,
record:

- command and commit;
- input paths and phenotype/covariate columns;
- trait mode and binary correction plan;
- device and dtype settings;
- tolerance used for comparison;
- whether differences are isolated to invalid or `TEST_FAIL` rows.

## Output And Resume Tests

Output changes must cover:

- output path derivation;
- Arrow/Parquet/REGENIE text chunk naming;
- finalization behavior;
- `run_manifest.json` compatibility;
- `effective_config.toml`;
- fast and strict resume behavior when chunk files and manifest disagree.

Any output schema change must update public [Output Files](../public/output-files.md)
and keep manifest/schema versions coherent.

## Documentation Changes

For documentation-only changes, run:

```bash
just docs-build
```

Run code tests as well when docs expose behavior that was changed in code or
when examples depend on newly changed CLI/config semantics.
