# REGENIE Parity Suite

| Status | Applies to | Owner |
| --- | --- | --- |
| Quantitative gate; binary qualification | Full 1KG chromosome 22 Step 2 workflows as of 2026-07-20 | Correctness maintainers |

The parity suite compares production `g` results with independently generated
upstream REGENIE v4.1 outputs. Earlier `g` output may be used as a secondary
regression diagnostic, never as the sole oracle for a compatibility claim. A
workflow becomes release-blocking only when its metadata `gate_status` is
`blocking`.

The machine-readable record is `tests/parity/golden_metadata.json`. It pins the
reference version, exact commands, artifact hashes, row counts, supported native
CLI configuration, hashes for every workflow input, and statistic-specific
tolerances.

## Commands

The metadata and comparison-helper tests are login-node safe:

```bash
just test-local-focused
```

The release-blocking quantitative run is a serialized GPU workload and belongs
on `landau`:

```bash
just slurm-gpu-test-parity-required
```

The required recipe sets `G_REGENIE_PARITY_REQUIRE_DATA=1`, so missing fixture
or oracle files fail loudly. Binary score-only and approximate-Firth
qualification run separately with
`just slurm-gpu-test-parity-diagnostic-required`; their failures do not redefine
the quantitative release gate. Neither full-data job runs on GitHub-hosted
runners because the protected fixture is unavailable there.
`GWAS_ENGINE_DATA_DIR` can point at a fixture tree outside the repository's
ignored `data/` directory. `G_REGENIE_PARITY_DEVICE` may override the recorded
`gpu` device only for a deliberate diagnostic run on an appropriate allocation.

`just test-parity` is useful for an explicitly requested local diagnostic: it
skips workflows with missing data. Do not run it on a login node merely because
the fixture happens to be mounted there.

## Golden Workflows

The quantitative oracle was generated with:

```bash
regenie --step 1 --bed data/1kg_chr22_full --phenoFile data/pheno_cont.txt --covarFile data/covariates.txt --qt --force-step1 --bsize 1000 --out data/baselines/regenie_step1_qt
regenie --step 2 --bgen data/1kg_chr22_full.bgen --sample data/1kg_chr22_full.sample --ref-first --phenoFile data/pheno_cont.txt --covarFile data/covariates.txt --qt --bsize 400 --pred data/baselines/regenie_step1_qt_pred.list --out data/baselines/regenie_step2_qt
```

Its 418,943-row output is
`data/baselines/regenie_step2_qt_phenotype_continuous.regenie` with SHA-256
`0c4782540b992d9f2163e2d1732ea0a9781e1816b23d80b8c893c3ad4ffab7b0`.

The binary score-only oracle was generated on `hilbert` with REGENIE v4.1. The
log records a start time of `Mon Jul 20 13:45:00 2026` and an end time of
`Mon Jul 20 13:45:10 2026`; REGENIE did not record the time zone.

```bash
regenie --step 1 --bed data/1kg_chr22_full --phenoFile data/pheno_bin.txt --covarFile data/covariates.txt --bt --cc12 --force-step1 --bsize 1000 --out data/baselines/regenie_step1
regenie --step 2 --bgen data/1kg_chr22_full.bgen --sample data/1kg_chr22_full.sample --ref-first --phenoFile data/pheno_bin.txt --covarFile data/covariates.txt --bt --cc12 --bsize 400 --pred data/baselines/regenie_step1_pred.list --out data/baselines/regenie_step2_score_only
```

Its 418,943-row output is
`data/baselines/regenie_step2_score_only_phenotype_binary.regenie` with SHA-256
`ba7278541d211a8ca446f5af3d45beba06030ad40f8124651db3038c196dac33`.
The pinned log SHA-256 is
`c4002866c86dd67ebe23fcb563f17488635b59547cc30baa3a8566730e2e0e5b`.
The workflow remains diagnostic until current HEAD completes the full
comparison.

The binary approximate-Firth oracle was generated with:

```bash
regenie --step 1 --bed data/1kg_chr22_full --phenoFile data/pheno_bin.txt --covarFile data/covariates.txt --bt --cc12 --force-step1 --bsize 1000 --out data/baselines/regenie_step1
regenie --step 2 --bgen data/1kg_chr22_full.bgen --sample data/1kg_chr22_full.sample --ref-first --phenoFile data/pheno_bin.txt --covarFile data/covariates.txt --bt --cc12 --firth --approx --bsize 400 --pred data/baselines/regenie_step1_pred.list --out data/baselines/regenie_step2
```

Its 418,943-row output is
`data/baselines/regenie_step2_phenotype_binary.regenie` with SHA-256
`0b9dc124525b6fec63e1b0d3f446263c05f690862235bd84f51b1b3c77b6ed72`.
The pinned log records 17,938 corrections and zero correction failures.

The binary tolerance qualification was observed at merge commit `bf5439aa`
with JAX and jaxlib 0.11.0 and native-library SHA-256
`de785821de0e3558a7dfada4cd7dbe8eacced67ffa02006feacdab5db4db39c6`.
The standard `bench-binary-hot-gpu-smoke` production configuration produced
all-row maximum absolute differences of `0.0006930` (`BETA`), `0.0002226`
(`SE`), `0.0013471` (`CHISQ`), and `0.0003654` (`LOG10P`), with exact
significance classifications and 17,938/0 correction/failure counts. That
temporary output was removed with its completed worktree, as required by the
worktree cleanup policy. The workflow therefore remains diagnostic until the
same result is reproduced on current HEAD; the absence of the old path is not
presented as surviving evidence.

Before launching `g`, the suite verifies SHA-256 for the BGEN, Oxford sample
file, phenotype, covariates, Step-1 prediction list, and every `.loco` file
referenced by that list, as well as the upstream output and log. Referenced
paths must resolve within the configured data root. Approximate-Firth
correction and failure counts are parsed from the hash-pinned log; metadata is
checked against that parsed summary, and the production aggregate is compared
with the parsed values.

## Comparison Contract

Production is invoked through `g._core.cli.run` with 16,384-variant chunks,
eight output writers, telemetry off, and direct Parquet parts. The test reads
`*.run/parts/*.parquet`; it does not require the removed finalization path.

Rows are joined one-to-one by the composite key
`(CHROM, GENPOS, ID, ALLELE0, ALLELE1)`. This is required because IDs repeat in
the full chromosome fixture. Both source tables and the joined result must have
exactly 418,943 unique rows.

For finite values the assertion is strictly:

```text
abs(g_value - regenie_value) < absolute_tolerance
```

`NaN`, positive-infinity, and negative-infinity masks must match exactly. `N`
must match exactly. The `p < 0.05` and `p < 5e-8` classifications derived from
`LOG10P` must also match exactly.

| Statistic | Quantitative | Binary score-only | Binary approximate-Firth |
| --- | ---: | ---: | ---: |
| `BETA` | `1.0e-3` | `1.0e-3` | `2.0e-3` |
| `SE` | `1.0e-3` | `1.0e-3` | `1.0e-3` |
| `CHISQ` | `1.5e-2` | `2.0e-2` | `3.0e-3` |
| `LOG10P` | `1.5e-2` | `2.0e-2` | `1.0e-3` |

These tolerances are exclusive bounds. A difference equal to the tolerance
fails.

Upstream REGENIE's binary result table labels every row as `TEST=ADD` and does
not identify successful approximate-Firth rows individually. Consequently, the
external contract can assert the exact aggregate correction/failure counts from
the pinned log, but not a per-row upstream correction mask. `g` method/status
labels are still checked for valid combinations and consistent failure masks.

## Qualification Reports

After a completed full comparison, the suite writes one JSON artifact to:

```text
results/parity/qualification/<workflow>/<UTC timestamp>_<process ID>.json
```

`results/` is ignored. Set `G_REGENIE_PARITY_REPORT_DIRECTORY` to use another
ignored or temporary root. The report records the git commit and dirty-state
hashes, native-library and lockfile hashes, JAX/jaxlib versions, configured
device and TOML hash, input and reference hashes, individual and aggregate
Parquet hashes, output schema and column order, run-manifest/metadata hashes,
row/correction counts, and every observed maximum absolute difference with its
exclusive tolerance. Failed qualifications retain the assertion message in the
report before pytest re-raises the failure.
