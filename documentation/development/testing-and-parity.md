# Testing And Parity

| Status | Applies to | Owner |
| --- | --- | --- |
| Active foundation | Main branch as of 2026-07-20 | Correctness maintainers |

This page separates fast mathematical checks from full external REGENIE
comparisons. A comparison with an earlier `g` build is useful for locating a
regression, but it is not the primary correctness oracle: two `g` versions can
share the same defect.

## Test Tiers

| Tier | Command | Execution policy |
| --- | --- | --- |
| External-contract harness | `just test-local-focused` | Login-node safe; reads metadata and tiny in-memory frames only. |
| Active non-data Python suite | `just test-local` | Run on an appropriate CPU allocation when JAX compilation would be material. |
| Optional external parity | `just test-parity` | Full chr22 GPU work when fixtures exist; missing local fixtures skip. |
| Blocking required-fixture parity | `just slurm-gpu-test-parity-required` | Serialized GPU-node release gate; missing fixtures fail the test. |
| Diagnostic required-fixture parity | `just slurm-gpu-test-parity-diagnostic-required` | Serialized GPU-node qualification of workflows that are not yet release gates. |
| Full repository suite | `just test-full` | GPU allocation only; CPU and parity tests run in separate processes. |

Do not run GPU workloads, heavy compilation, large suites, or benchmark sweeps
on a login node. GitHub-hosted CI runs the active non-data tests and the
login-safe parity harness. It does not claim to run the protected full chr22
fixture.

## Correctness Oracle

Use an independently generated, version-pinned upstream REGENIE output as the
primary oracle whenever the project claims REGENIE compatibility. The current
goldens use upstream REGENIE v4.1 and are recorded in
`tests/parity/golden_metadata.json` with their commands, row counts, and
SHA-256 digests. Each workflow separately records whether it is blocking or
diagnostic. Binary score-only and approximate Firth remain diagnostic until
their full comparisons are reproduced on current HEAD.

The comparison contract is:

- align all rows by `(CHROM, GENPOS, ID, ALLELE0, ALLELE1)`; `ID` alone is not
  unique in the full chr22 fixture;
- require both inputs and the one-to-one join to contain all 418,943 rows;
- for each finite statistic, require `abs(g_value - regenie_value) < tolerance`;
- require identical `NaN`, positive-infinity, and negative-infinity masks;
- require exact `N` and exact significance decisions at `p < 0.05` and
  `p < 5e-8`;
- verify every BGEN, sample, phenotype, covariate, Step-1 prediction list and
  referenced `.loco` file, REGENIE output, and REGENIE log hash before
  execution;
- parse correction and correction-failure counts from the pinned upstream log
  and require the production aggregate to match it.

The hashes identify the external oracle artifacts. They are not a request for
byte-for-byte equality between REGENIE text and `g` Parquet output.

Binary approximate-Firth labels need a narrower statement. Upstream REGENIE's
saved table does not expose a per-row successful-Firth label. The comparison
therefore checks its recorded aggregate correction and failure counts, checks `g`'s
allowed method/status values, and compares every public statistic and
significance decision. It does not claim per-row correction-label parity that
the upstream artifact cannot establish.

## Supported Production Boundary

Full parity invokes only the supported `g._core.cli.run` binding. Results are
loaded from the production `*.run/parts/*.parquet` dataset. The test does not
restore legacy Python orchestration or expect a post-run `final.parquet`.

Local missing data are a skip so contributors can run the harness without the
protected fixture. Required scheduled runs set
`G_REGENIE_PARITY_REQUIRE_DATA=1`; every missing BGEN, sample, phenotype,
covariate, prediction, output, or log artifact is then a hard failure.

Each completed full comparison writes an ignored qualification report below
`results/parity/qualification/<workflow>/`. It records source, native library,
dependencies, device/configuration, all input/reference/output hashes,
per-statistic maxima, correction counts, and pass/failure status. Set
`G_REGENIE_PARITY_REPORT_DIRECTORY` to place reports in another ignored or
temporary directory.

## Coverage Reports

`just coverage-python` and `just coverage-rust` generate reports for the active
test surface. They are not advertised as percentage gates while the product
suite is being rebuilt around the current native boundary. Restore a threshold
only with a measured baseline and tests for real supported contracts; do not
meet a number by adding placeholder product tests.

## Documentation Changes

For documentation-only changes, run:

```bash
just docs-build
```

Run code tests as well when documentation describes behavior changed in code.
