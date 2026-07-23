# REGENIE Parity Suite

| Status | Applies to | Owner |
| --- | --- | --- |
| Exact-head qualification required | Full 1KG chromosome 22 Step 2 workflows as of 2026-07-23 | Correctness maintainers |

The parity suite compares production `g` results with independently generated
upstream REGENIE v4.1 outputs. Earlier `g` output may be used as a secondary
regression diagnostic, never as the sole oracle for a compatibility claim. A
workflow is not release-qualified merely because an older source revision
passed. Every checked-in workflow has first-class `required` status but keeps
its evidence null; a trusted status publisher consumes the sanitized bundle
produced from the exact release commit.

The machine-readable record is `tests/parity/golden_metadata.json`. It pins the
reference version, exact commands, artifact hashes, row counts, supported
native CLI configuration, hashes for every workflow input, and
statistic-specific tolerances. It does not carry a self-referential claim that
the commit containing the metadata qualified itself.

## Commands

The metadata and comparison-helper tests are login-node safe:

```bash
just test-local-focused
```

The required quantitative, binary score-only, and binary approximate-Firth
runs are serialized GPU workloads and belong on `landau`:

```bash
G_REGENIE_PARITY_EXPECTED_GIT_COMMIT=<full scheduler-selected SHA> \
  just slurm-gpu-test-parity-required
```

The SLURM recipe creates one GPU allocation, installs the GPU dependency set,
computes the clean checkout's science-source SHA-256, builds and installs the
release extension with that commit and fingerprint embedded, and then runs all
tests marked `parity_required` in the same allocation. Native artifacts are
isolated below the current worktree at `target/qualification/<node>/`, while
the explicit JAX cache is node- and job-specific below `/tmp`; the recipe logs
both resolved locations. This prevents `target-cpu=native` or compiled JAX
artifacts from crossing heterogeneous nodes. The test preflight
rejects a missing scheduler-selected SHA, a dirty checkout, a changed SHA or
science fingerprint, a non-release build, or an extension stamped from other
source. Evidence is accepted only from the workflow's allowed qualification
host (`landau`) and records the observed JAX platform, homogeneous device kind
and count, CUDA backend version, NVIDIA driver, and CUDA runtime package. A
non-CUDA backend is rejected. The recipe sets
`G_REGENIE_PARITY_REQUIRE_DATA=1`, so missing fixture or oracle files fail
loudly. The full-data gate does not run on GitHub-hosted runners because the
protected fixture is unavailable there.
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
This oracle remains required input to exact-head qualification.

The binary approximate-Firth oracle was generated with:

```bash
regenie --step 1 --bed data/1kg_chr22_full --phenoFile data/pheno_bin.txt --covarFile data/covariates.txt --bt --cc12 --force-step1 --bsize 1000 --out data/baselines/regenie_step1
regenie --step 2 --bgen data/1kg_chr22_full.bgen --sample data/1kg_chr22_full.sample --ref-first --phenoFile data/pheno_bin.txt --covarFile data/covariates.txt --bt --cc12 --firth --approx --bsize 400 --pred data/baselines/regenie_step1_pred.list --out data/baselines/regenie_step2
```

Its 418,943-row output is
`data/baselines/regenie_step2_phenotype_binary.regenie` with SHA-256
`0b9dc124525b6fec63e1b0d3f446263c05f690862235bd84f51b1b3c77b6ed72`.
The pinned log records 17,938 corrections and zero correction failures.

Earlier binary reports targeted commit `68f831f9...` and did not prove that
their loaded extension came from that source. They are historical diagnostics,
not current release evidence. The exact-head runner must reproduce all three
workflows and emit a new ignored bundle; no current-source maxima are asserted
until that external run completes.

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
Every part must expose exactly this ordered Arrow schema:
`CHROM String`, `GENPOS Int64`, `ID String`, `ALLELE0 String`,
`ALLELE1 String`, `A1FREQ Float32`, nullable `INFO Float32`, `N Int32`,
`BETA Float32`, `SE Float32`, `CHISQ Float32`, `LOG10P Float32`,
`CORRECTION_METHOD String`, and `CORRECTION_STATUS String`. The pinned
REGENIE text schema, column order, and inferred dtypes are also asserted before
numeric comparison.

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
ignored or temporary root. The report records the exact git commit, canonical
science-source fingerprint, embedded native commit/fingerprint/clean/profile
identity, native-library and lockfile hashes, JAX/jaxlib versions, configured
device, observed CUDA device/runtime identity, and TOML hash, input and
reference hashes, individual and aggregate Parquet hashes, ordered output
schema, run-manifest/metadata hashes,
row/correction counts, and every observed maximum absolute difference with its
exclusive tolerance. Failed qualifications retain the assertion message before
pytest re-raises.

After all three reports pass, the final required test writes:

```text
results/parity/qualification/qualification_bundle_<exact Git SHA>.json
```

The bundle contains only digests, versions, counts, typed schema/statistic
evidence, and relative oracle labels. It contains no protected records or
absolute protected-data paths. A trusted post-job identity can validate and
attach this ignored bundle to the exact SHA without a metadata commit, avoiding
the impossible requirement for a commit to contain its own hash. The trusted
status publisher and repository rule remain external deployment dependencies.

The current reader discovers one `*.run/parts/` transaction. If the separate
output-transaction change lands first, rebase this branch and adapt
`direct_parquet_paths()` to its finalized layout before the first exact-head
run; do not treat the pre-rebase layout probe as qualification evidence.
