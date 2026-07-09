# Python Math Test Suite Audit

| Status | Applies to | Owner |
| --- | --- | --- |
| Math-only suite | Python tests after the 2026-07-09 pruning pass | Correctness maintainers |

The checked-in Python tests are limited to mathematical correctness coverage:
quantitative kernels, binary score/Firth kernels, and REGENIE parity comparison
helpers. Product, CLI, lifecycle, output/resume, telemetry, architecture,
tooling, warm-cache, and pipeline orchestration tests were removed from
`tests/`.

## Retained Tests

| Path | Reason |
| --- | --- |
| `tests/test_regenie2_linear.py` | Quantitative association statistics, p-value conversion, residualization, LOCO behavior, packed8/dosage equivalence, and numerical edge cases. |
| `tests/test_regenie2_binary.py` | Binary score, candidate selection, approximate Firth correction, sparse/dense paths, packed8/dosage equivalence, and CPU/GPU numerical parity. |
| `tests/test_regenie2_binary_firth_null.py` | Null Firth fallback attempts and convergence behavior. |
| `tests/test_regenie2_binary_full_model.py` | Full-model Firth matrix blocks, score components, log-likelihood behavior, and solver outcomes. |
| `tests/test_regenie2_binary_scalar_firth.py` | Scalar approximate-Firth formulas, Newton-Raphson fallback behavior, sparse flags, and numerical failure labels. |
| `tests/test_regenie2_parity.py` | External REGENIE golden comparisons for public association statistics. |
| `tests/parity/` | Metadata and helper checks that keep parity comparisons tied to current numerical coverage. |

## Removed Categories

The pruning pass removed Python tests that validated non-mathematical contracts:
API import boundaries, CLI behavior, configuration validation, BGEN bridge
plumbing, preflight checks, output writers, resume manifests, callbacks,
telemetry, timing, JAX runtime policy, warm-cache planning, development tooling,
architecture rules, Symphony automation, and benchmark/report tooling.
