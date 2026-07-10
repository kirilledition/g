# Compute Kernels

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; development contract | JAX association kernels and target bridge as of 2026-07-10 | Compute maintainers |

JAX kernels own statistical computation after native I/O has aligned samples and
decoded genotype chunks. Public mathematical behavior is documented in
[Algorithm](../public/algorithm.md).

## Source Map

| Path | Responsibility |
| --- | --- |
| `src/g/compute/common/` | Shared dtype, genotype, linear algebra, and p-value helpers. |
| `src/g/compute/regenie2_linear/` | Quantitative Step 2 state preparation and score/statistic kernels. |
| `src/g/compute/regenie2_binary/` | Binary null model, score test, candidate selection, correction, diagnostics, and Firth paths. |
| `src/g/jax_backend.py` | Coarse group/chromosome/compute/materialize adapter called by Rust. |
| `crates/engine/src/` | Rust owner of aligned inputs, batching, scheduling, and host-result writing. |

## Kernel Boundary

Kernels should receive:

- aligned phenotype/covariate/prediction state;
- variant-major genotype dosage chunks or packed GPU-compatible genotype data;
- immutable numerical configuration;
- explicit dtype and device policy from the typed JAX backend config.

Kernels should not:

- read CLI/TOML/environment directly;
- parse BGEN, phenotype, covariate, or prediction files;
- write output files;
- introduce hidden constants that affect public statistics or compiled shapes.

## Quantitative Contract

Quantitative kernels implement REGENIE Step 2 linear association:

- covariate projection/residualization;
- LOCO-adjusted phenotype residuals;
- additive allele-one dosage effect;
- one-degree-of-freedom chi-squared statistic and `LOG10P`;
- numerical masks for low residualized genotype variance.

Changing this path requires parity tests and public algorithm updates when
statistics or accepted failure conditions change.

## Binary Contract

Binary kernels implement:

- chromosome-specific logistic null fitting with LOCO offset;
- binary score statistic;
- optional approximate-Firth candidate selection and correction;
- Firth and null-Firth solver diagnostics;
- correction method and status labels for successful and failed rows.

Changing binary correction behavior requires tests for score-only, approximate
Firth, diagnostics, failure-code mapping, and public documentation updates when
user-visible statistics or tuning semantics change.

## Numerical Configuration

Numerical settings are canonical TOML options and must be threaded from
`crates/interface/src/config.default.toml` through the Rust run request and
typed JAX backend config into kernel config dataclasses.

Examples:

- linear variance floors;
- binary probability and variance floors;
- null logistic nonconvergence policy;
- Firth iteration/tolerance/line-search limits;
- score and Firth dtypes;
- JAX matmul precision.

Do not add module-level runtime constants for configurable numerical behavior.

## JAX Runtime

JAX runtime policy is configured before backend initialization. Code that imports
JAX-heavy modules should stay behind explicit runtime boundaries when needed to
preserve device/cache setup order.

Profile mode may intentionally synchronize device work for timing. Production
progress logging must not add synchronization only for observability.

## Testing

Relevant tests include:

- `tests/test_regenie2_linear.py`;
- `tests/test_regenie2_binary.py`;
- `tests/test_regenie2_binary_firth_null.py`;
- `tests/test_regenie2_binary_full_model.py`;
- `tests/test_regenie2_binary_scalar_firth.py`;
- `tests/test_regenie2_parity.py`;
- `tests/parity/`.

Use [Testing and Parity](testing-and-parity.md) for validation tier selection.
