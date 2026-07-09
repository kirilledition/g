# Testing And Parity

| Status | Applies to | Owner |
| --- | --- | --- |
| Math-only Python suite | main branch as of 2026-07-09 tests and parity workflow | Correctness maintainers |

This page defines how to validate mathematical correctness work without
confusing fast local checks, full correctness runs, and external REGENIE parity
work.

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

## Python Correctness Ownership

| Area | Representative tests |
| --- | --- |
| Quantitative kernels | `tests/test_regenie2_linear.py`, `tests/test_regenie2_parity.py` |
| Binary score and Firth kernels | `tests/test_regenie2_binary.py`, `tests/test_regenie2_binary_firth_null.py`, `tests/test_regenie2_binary_full_model.py`, `tests/test_regenie2_binary_scalar_firth.py` |
| REGENIE parity metadata and comparison helpers | `tests/parity/` |

The Python suite intentionally excludes non-mathematical product contracts such
as CLI behavior, configuration validation, output/resume behavior, telemetry,
pipeline orchestration, architecture checks, and tooling automation.

## Rust Correctness Ownership

The active Rust test surface is math-only and lives in `g-genotype`. Keep
`cargo test -p g-genotype` focused on BGEN dosage decoding, imputation,
variant summaries, and SIMD/scalar numerical equivalence.

Rust product, runtime, CLI, IO, scheduling, and planning contracts stay out of
the test suite until those APIs stabilize.

## REGENIE Parity Rules

The pre-release Step 2 parity gate is documented in
[REGENIE Parity Suite](regenie-parity-suite.md). Its machine-readable coverage
matrix lives in `tests/parity/golden_metadata.json`, and the lightweight harness
checks live under `tests/parity/`.

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

Result statistics are public `float32` outputs by default. Some internal kernels
can use wider dtypes for parity-sensitive work, and parity runs can set
`[output].output_statistic_dtype = "float64"` to persist wider public statistics.
When validating a numerical change, record:

- command and commit;
- input paths and phenotype/covariate columns;
- trait mode and binary correction plan;
- device and dtype settings;
- tolerance used for comparison;
- whether differences are isolated to invalid or `TEST_FAIL` rows.

## Documentation Changes

For documentation-only changes, run:

```bash
just docs-build
```

Run code tests as well when docs expose behavior that was changed in code or
when examples depend on newly changed CLI/config semantics.
