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

### Trusted Scheduler Launch

The required quantitative, binary score-only, and binary approximate-Firth
runs are serialized GPU workloads and belong on `landau`. Qualification must
start from a full commit selected by the trusted scheduler, not from a recipe
read from the mutable worktree. The scheduler writes the selected commit's
bootstrap blob to a private temporary executable visible on `landau`, computes
its SHA-256, and launches that exact file.

The following is the launch pattern. The `SCHEDULER_*` values must come from
trusted scheduler configuration. The staging directory must be private to the
launching user and visible at the same absolute path on the login and compute
nodes. Add site-specific `--partition`, `--account`, or reservation arguments
to `srun` as required.

```bash
set -euo pipefail

source_repository="${SCHEDULER_SOURCE_REPOSITORY:?absolute source repository}"
selected_git_commit="${SCHEDULER_SELECTED_GIT_COMMIT:?full lowercase Git SHA}"
trusted_uv_path="${SCHEDULER_TRUSTED_UV_PATH:?absolute uv executable}"
trusted_just_path="${SCHEDULER_TRUSTED_JUST_PATH:?absolute just executable}"
trusted_cargo_path="${SCHEDULER_TRUSTED_CARGO_PATH:?absolute cargo executable}"
trusted_cargo_cache_home="${SCHEDULER_TRUSTED_CARGO_CACHE_HOME:?absolute Cargo cache home}"
trusted_mold_path="${SCHEDULER_TRUSTED_MOLD_PATH:?absolute mold executable}"
trusted_python_path="${SCHEDULER_TRUSTED_PYTHON_PATH:?absolute Python 3.14 executable}"
trusted_rustc_path="${SCHEDULER_TRUSTED_RUSTC_PATH:?absolute rustc executable}"
trusted_rustup_home="${SCHEDULER_TRUSTED_RUSTUP_HOME:?absolute Rustup home}"
trusted_cuda_library_path="${SCHEDULER_TRUSTED_CUDA_LIBRARY_PATH:-}"
data_directory="${SCHEDULER_PARITY_DATA_DIRECTORY:?absolute protected fixture directory}"
report_base="${SCHEDULER_PARITY_REPORT_BASE:?absolute report base}"
staging_directory="${SCHEDULER_PARITY_STAGING_DIRECTORY:?private shared staging directory}"

bootstrap_path="$(/usr/bin/mktemp \
  "${staging_directory}/exact-parity-bootstrap.XXXXXX")"
trap '/usr/bin/rm -f -- "${bootstrap_path}"' EXIT
/usr/bin/env -i \
  HOME="${HOME:?}" \
  GIT_CONFIG_COUNT=0 \
  GIT_CONFIG_GLOBAL=/dev/null \
  GIT_CONFIG_NOSYSTEM=1 \
  GIT_NO_REPLACE_OBJECTS=1 \
  /usr/bin/git --no-replace-objects -C "${source_repository}" show \
  "${selected_git_commit}:tooling/server/exact_parity_bootstrap.sh" \
  > "${bootstrap_path}"
/usr/bin/chmod 0500 "${bootstrap_path}"
bootstrap_sha256="$(
  /usr/bin/sha256sum "${bootstrap_path}" | /usr/bin/cut -d ' ' -f 1
)"

/usr/bin/srun \
  --export=NONE \
  --nodes=1 \
  --ntasks=1 \
  --nodelist=landau \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=04:00:00 \
  /usr/bin/bash --noprofile --norc -c '
    set -euo pipefail
    exec /usr/bin/env -i \
      HOME="$1" \
      SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:?}" \
      SLURM_JOB_ID="${SLURM_JOB_ID:?}" \
      SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:?}" \
      SLURM_JOB_UID="${SLURM_JOB_UID:?}" \
      SLURM_JOB_USER="${SLURM_JOB_USER:?}" \
      SLURM_STEP_ID="${SLURM_STEP_ID:?}" \
      SLURM_STEP_NODELIST="${SLURM_STEP_NODELIST:?}" \
      SLURMD_NODENAME="${SLURMD_NODENAME:?}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}" \
      GWAS_ENGINE_DATA_DIR="$2" \
      G_REGENIE_PARITY_REPORT_DIRECTORY="$3" \
      G_REGENIE_PARITY_UV_PATH="$4" \
      G_REGENIE_PARITY_JUST_PATH="$5" \
      G_REGENIE_PARITY_CARGO_PATH="$6" \
      G_REGENIE_PARITY_CARGO_CACHE_HOME="$7" \
      G_REGENIE_PARITY_RUSTC_PATH="$8" \
      G_REGENIE_PARITY_RUSTUP_HOME="$9" \
      G_REGENIE_PARITY_LD_LIBRARY_PATH="${10}" \
      G_REGENIE_PARITY_BOOTSTRAP_SHA256="${11}" \
      G_REGENIE_PARITY_MOLD_PATH="${15}" \
      G_REGENIE_PARITY_PYTHON_PATH="${16}" \
      /usr/bin/bash --noprofile --norc "${12}" "${13}" "${14}"
  ' exact-parity-launch \
  "${HOME:?}" \
  "${data_directory}" \
  "${report_base}" \
  "${trusted_uv_path}" \
  "${trusted_just_path}" \
  "${trusted_cargo_path}" \
  "${trusted_cargo_cache_home}" \
  "${trusted_rustc_path}" \
  "${trusted_rustup_home}" \
  "${trusted_cuda_library_path}" \
  "${bootstrap_sha256}" \
  "${bootstrap_path}" \
  "${source_repository}" \
  "${selected_git_commit}" \
  "${trusted_mold_path}" \
  "${trusted_python_path}"
```

The outer clean environment supplies only the required home, Slurm, CUDA,
fixture, report, tool, and bootstrap identities. If the installation requires
additional scheduler-owned CUDA variables, add only those specific names; do
not replace `env -i` with wholesale environment inheritance. `uv`, `just`,
`mold`, Python 3.14, `cargo`, and `rustc` must be scheduler-selected absolute
executables. The Cargo and Rust binaries must be direct pinned toolchain
binaries, not Rustup proxies. The Cargo cache and Rustup homes must be absolute
existing scheduler-trusted snapshots; the selected Python installation tree is
also trusted. When CUDA
requires an explicit dynamic-library search path, supply it through the
dedicated trusted scheduler value shown above; direct inherited
`LD_LIBRARY_PATH` is rejected. The bootstrap invokes itself through
`/usr/bin/bash --noprofile --norc`. It rejects inherited shell, Python, pytest,
uv, Cargo, Rustup, linker-loader, Git, and compiler override variables rather
than treating them as trusted inputs.

In other words, the scheduler runs
`/usr/bin/git --no-replace-objects show
<selected SHA>:tooling/server/exact_parity_bootstrap.sh` in the configured
source repository; it never copies the bootstrap from the worktree.

The bootstrap uses `/usr/bin/git`, `/usr/bin/bash`, and
`/usr/bin/scontrol`. It verifies its executed-file digest against both the
scheduler-provided digest and the blob at the selected commit, then asks
`scontrol` for the live job and step. The job, step, user, `landau` node, and
running state must agree with the process environment. It creates a
mode-`0700` unique root directly below sticky `/tmp`, named
`g-parity-qualification-<uid>-<job>-<nonce>.<random>/`, and places a detached,
non-local clone there without local hardlinks or working-tree checkout
filters. Build output, copied Cargo source caches, the private home, uv's
no-cache temporary state, the JAX cache, pytest base directory, and the virtual
environment are private children. After the second independent bundle
validation, an ownership- and mode-checked EXIT trap removes the whole
qualification root on success or failure. The protected fixture and durable
reports remain external to it.

System and global Git configuration, replacement objects, hooks, external
attributes files, alternate object stores, and templates are neutralized. The
clone starts with no checkout and is materialized from the selected tree.
Shell startup files, Python user and import environments, uv configuration and
environment files, Cargo/Rust compiler overrides, and pytest arguments and
plugin autoloading are also neutralized. The bootstrap finally re-executes the
exact inner Just recipe under a new `env -i` environment containing only its
explicit scheduler, CUDA, build-tool, and qualification allowlist. Pytest
receives exactly the three workflow node IDs and the final bundle node.
The locked uv sync first verifies that `.venv/bin/python` resolves to the
scheduler-selected interpreter. The recipe then binds both `VIRTUAL_ENV` and
`UV_PYTHON` to that private environment before the literal
`maturin develop --profile release --uv` command, so installation cannot target
uv's externally managed base interpreter. The fixed build PATH must not expose
an unselected `patchelf`; its appearance fails qualification instead of
silently changing Maturin's extension rewriting.

The mutable `just test-parity-required`,
`just test-parity-required-exact`, and
`just slurm-gpu-test-parity-required` entrypoints deliberately exit without
running or publishing evidence. Generic worktree recipes are useful only for
diagnostics; they cannot produce a qualifying bundle.

The source preflight reads blobs from the replacement-disabled committed HEAD
tree and byte-compares their path set, mode, object identity, and contents with
both the index and disk. It rejects `assume-unchanged`, `skip-worktree`, and
fsmonitor-valid index flags as well as symbolic links in any science-source
path. All tracked root Python modules and the full `tests/parity/` package are
fingerprinted; isolated Python startup and explicit pytest configuration avoid
untracked import/config discovery. A clean-looking `git status` is therefore
not trusted by itself. The release extension embeds the exact commit, science fingerprint,
clean-source bit, and a 128-bit run nonce. After `maturin develop`, the recipe
captures the discovered extension path and SHA-256; the test requires the
imported module to be exactly that artifact with the same nonce.

Evidence is accepted only from the workflow's allowed qualification host
(`landau`) and records the Slurm job, step, nonce, UTC run start, bootstrap
committed path and SHA-256, and the path, version, and SHA-256 of the selected
`bash`, `ar`, `as`, `cc`, GCC `cc1`/`cc1plus`/`collect2`, `cargo`, `c++`,
`env`, `git`, `just`, Maturin, Mold, the scheduler-selected and private-venv
Python interpreters, `ranlib`, `rustc`, `scontrol`, and `uv` executables. It
records the effective `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, `RUSTC_WRAPPER`,
and `CARGO_BUILD_RUSTC_WRAPPER` values as empty strings, rather than leaving
their absence implicit. It also records the observed JAX platform, homogeneous
device kind and count, CUDA backend version, NVIDIA driver, and CUDA runtime
package. A non-CUDA backend is rejected. The exact inner recipe sets
`G_REGENIE_PARITY_REQUIRE_DATA=1`, so missing fixture or oracle files fail
loudly. The full-data gate does not run on GitHub-hosted runners because the
protected fixture is unavailable there.
`GWAS_ENGINE_DATA_DIR` can point at a fixture tree outside the repository's
ignored `data/` directory. `G_REGENIE_PARITY_DEVICE` may override the recorded
`gpu` device only for a deliberate diagnostic run on an appropriate allocation.

`just test-parity` is useful for an explicitly requested local diagnostic: it
skips workflows with missing data and never emits an exact-source bundle when
the required qualification environment is absent. All three comparisons still
run when local data are present. Do not run it on a login node merely because
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
with the parsed values. Every one of those artifacts is rehashed after the
production run and protected-oracle reads, then again before bundle assembly;
any mid-run mutation invalidates qualification.

## Comparison Contract

Production is invoked through `g._core.cli.run` with 16,384-variant chunks,
eight output writers, telemetry off, and direct Parquet parts. The test
requires exactly one stdout line beginning `Parquet dataset saved to `, requires
the reported path to be absolute and resolve to a directory strictly below the
requested output root, and reads only direct `*.parquet` children of that
selected directory. It performs no recursive `*.run` discovery. Every part
must expose exactly this ordered Arrow schema:
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
results/parity/qualification/<Slurm job>/<Slurm step>/<run nonce>/<workflow>/<UTC timestamp>_<run nonce>_<process ID>.json
```

`results/` is ignored. Set `G_REGENIE_PARITY_REPORT_DIRECTORY` to use another
ignored or temporary base. The report records the exact git commit, canonical
science-source fingerprint, Slurm job/step/start/nonce, bootstrap digest,
selected-tool paths/versions/digests, embedded native
commit/fingerprint/clean/profile/nonce identity, native-library and lockfile
hashes, Cargo configuration and Rust toolchain hashes, JAX/jaxlib versions,
configured device, observed CUDA device/runtime identity, TOML hash, input and
reference hashes, individual and aggregate Parquet hashes, ordered output
schema, run-manifest/metadata hashes, row/correction counts, and every observed
maximum absolute difference with its exclusive tolerance. Failed
qualifications retain the assertion message before pytest re-raises.

After all three reports pass, the final required test writes:

```text
results/parity/qualification/<Slurm job>/<Slurm step>/<run nonce>/qualification_bundle_<exact Git SHA>_<Slurm job>_<Slurm step>_<run nonce>.json
```

The bundle contains only digests, versions, counts, typed schema/statistic
evidence, relative report paths, and relative oracle labels. It binds each
report SHA-256 and direct Parquet dataset SHA-256, but contains no protected
records or absolute protected-data paths. Publication validates the complete
payload and dependencies before an exclusive atomic link and never replaces an
existing path. The writer validates the linked bundle again, and the recipe
validates it in a second Python process after pytest. A trusted post-job
identity can attach this
ignored bundle to the exact SHA without a metadata commit. The trusted status
publisher and repository rule remain external deployment dependencies.

This is a provenance boundary, not self-authentication against a hostile
machine. Qualification's execution trust boundary is the host, scheduler
control plane, selected executable files, and scheduler-selected Cargo-cache,
Rustup, and Python installation snapshots. The twenty recorded tools are not
the entire executable boundary: hardcoded host helpers including coreutils,
`tar`, Python 3 for nonce generation, and `git-upload-pack` are trusted as
part of the host image and are probed by
`just doctor-server`. The selected commit and protected inputs are hash-bound
evidence subjects, not independent trust anchors. A hostile host, scheduler,
shell, Git implementation, or executable can fabricate its own checks and
evidence; the bootstrap does not claim to detect that condition.

After the output-transaction change is merged, retain this unambiguous
completion-line selection but validate the selected path against the durable
completed lineage, finalization, and manifests. Bind the relevant `.g-output`
artifact hashes into the qualification report and bundle before the first
exact-head run; path selection alone is not qualification evidence for the new
layout.
