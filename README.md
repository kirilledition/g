# `g` — GPU-oriented REGENIE Step 2 GWAS engine

[![PR CI](https://github.com/kirilledition/g/actions/workflows/pr-ci.yml/badge.svg)](https://github.com/kirilledition/g/actions/workflows/pr-ci.yml)
[![Science Monthly](https://github.com/kirilledition/g/actions/workflows/science-monthly.yml/badge.svg)](https://github.com/kirilledition/g/actions/workflows/science-monthly.yml)

`g` is a pre-release, performance-focused GWAS engine for **BGEN-backed REGENIE Step 2** association scans. It combines a REGENIE-style user interface with a Rust/JAX execution engine:

```text
CLI / TOML / Python API
        ↓
RegenieConfig
        ↓
ExecutionPlan
        ↓
Rust BGEN decode + sample alignment + output writer
        ↓
JAX quantitative / binary association kernels
        ↓
Arrow chunks + optional finalized Parquet
```

The package is designed for reproducible, large-scale association runs where the hot path is explicit: Rust owns file formats, alignment, chunking, manifests, resume, SIMD-sensitive BGEN decode, and Arrow/Parquet output; Python owns the public API, config normalization, orchestration, telemetry, and JAX dispatch.

`g` is not a replacement for REGENIE Step 1. Use upstream `regenie` to produce Step 1 prediction lists, then use `g` for Step 2 BGEN scans.

---

## Current status

`g` is under active development and has not had a stable public release yet. Backward compatibility is not guaranteed.

| Area | Status |
|---|---|
| Quantitative REGENIE Step 2 (`--qt`) | Primary supported workflow |
| Binary score-test Step 2 (`--bt`) | Supported, evolving |
| Binary approximate Firth fallback (`--bt --firth --approx`) | Implemented, still parity/performance sensitive |
| SPA fallback (`--spa`) | Recognized, not implemented |
| Exact Firth without `--approx` | Recognized, not implemented |
| REGENIE Step 1 | Not implemented |
| BGEN 1.2 input | Supported |
| `.sample` files and embedded BGEN samples | Supported |
| BED/PGEN input | Recognized as REGENIE options, not implemented |
| Output | Arrow run chunks and Parquet finalization |
| REGENIE text output | Not the default output path; use Arrow/Parquet today |

The public surface is intentionally narrow:

- CLI: `g regenie ...`, `g-regenie ...`, and `g config init|validate|explain`
- Python API: `g.regenie(config)` and `g.regenie.from_options({...})`
- Config: TOML files using the same option names as the CLI flags

---

## Why `g` exists

REGENIE is already a very strong CPU implementation. `g` focuses on architecture that can exploit acceleration where it matters:

- decode and preprocess BGEN data in Rust;
- keep sample alignment and output writing out of Python DataFrame libraries;
- use JAX for accelerator-resident statistical kernels;
- batch multiple phenotypes when sample semantics allow it;
- preserve run manifests and effective configs for reproducibility;
- expose profiling-grade telemetry for performance work.

For single-phenotype scans, speedups can be limited by BGEN decode, host-device transfer, and output. The long-term performance path is to decode each genotype chunk once and reuse it across more useful work, especially multi-phenotype quantitative scans.

---

## Requirements

The project is managed with `uv`, `just`, Rust/Cargo, and `maturin`.

- Python: `>=3.14,<3.15`
- Runtime Python dependencies: Click, JAX, NumPy
- Native extension: Rust 2024 + PyO3 ABI3 for Python 3.14
- Optional GPU workflow: CUDA-capable JAX environment
- Development/benchmark tools: `plink`, `plink2`, `regenie`, `zstd`, and optionally Nix/SLURM

Install-time runtime dependency policy is intentionally small. Python table libraries such as Polars and PyArrow are development/inspection dependencies, not core runtime dependencies.

---

## Setup

### Nix development shell

```bash
nix develop
```

### CPU-oriented local setup

```bash
just bootstrap
just doctor
just check-local
```

### GPU-capable setup

```bash
just bootstrap-gpu
just doctor-jax
```

### Ubuntu / SLURM server setup

On the server, bootstrap repo-local tools and source the generated environment script:

```bash
UV_CACHE_DIR=/tmp/g-uv-cache uv run --no-project python scripts/bootstrap_server_tools.py
source scripts/server_env.sh
```

Useful server checks:

```bash
just doctor-server
just doctor-baselines
```

See `docs/NO_NIX_DEVELOPMENT.md` and `docs/UBUNTU_SLURM_DEVELOPMENT.md` for environment-specific workflows.

---

## Quick start

Prepare the local 1000 Genomes chromosome 22 benchmark data and simulated phenotypes:

```bash
just setup-data
```

Generate binary REGENIE Step 1 baseline predictions for binary Step 2 examples:

```bash
just setup-binary-baseline
```

Run a quantitative Step 2 scan:

```bash
uv run g regenie \
  --step 2 \
  --qt \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --phenoFile data/pheno_cont.txt \
  --phenoCol phenotype_continuous \
  --covarFile data/covariates.txt \
  --covarColList age,sex \
  --pred data/baselines/regenie_step1_qt_pred.list \
  --out data/example_regenie2
```

Run a binary score-only Step 2 scan:

```bash
uv run g regenie \
  --step 2 \
  --bt \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --phenoFile data/pheno_bin.txt \
  --phenoCol phenotype_binary \
  --covarFile data/covariates.txt \
  --covarColList age,sex \
  --pred data/baselines/regenie_step1_pred.list \
  --out data/example_regenie2_binary_score
```

Run a binary scan with approximate Firth fallback:

```bash
uv run g regenie \
  --step 2 \
  --bt \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --phenoFile data/pheno_bin.txt \
  --phenoCol phenotype_binary \
  --covarFile data/covariates.txt \
  --covarColList age,sex \
  --pred data/baselines/regenie_step1_pred.list \
  --firth \
  --approx \
  --pThresh 0.01 \
  --out data/example_regenie2_binary_firth
```

The direct executable form is also available:

```bash
g-regenie --step 2 --qt --bgen ... --phenoFile ... --phenoCol ... --pred ... --out ...
```

---

## REGENIE-compatible CLI model

`g regenie` uses REGENIE-style option names where possible:

| REGENIE-style option | Meaning |
|---|---|
| `--step 2` | Step 2 association scan; Step 1 is not implemented |
| `--qt` / `--bt` | Quantitative or binary trait mode |
| `--bgen` / `--sample` | BGEN genotype file and optional Oxford sample file |
| `--phenoFile` | Phenotype table |
| `--phenoCol` / `--phenoColList` | One or more phenotype columns |
| `--covarFile` | Covariate table |
| `--covarCol` / `--covarColList` | One or more covariate columns |
| `--pred` | REGENIE Step 1 prediction list |
| `--bsize` | Variant block size; default `8192` |
| `--threads` | Requested native CPU thread count |
| `--out` | User output prefix |
| `--firth --approx` | Binary approximate Firth fallback |
| `--pThresh` | Score-test p-value threshold for binary fallback |
| `--firth-se` | Firth-derived standard error behavior |

`g`-specific runtime and engine options are explicitly namespaced with `--g-*`, for example:

```bash
--g-device gpu
--g-staging-depth 2
--g-output-format parquet
--g-resume
--g-resume-mode strict
--g-trusted-no-missing-diploid
--g-trusted-bgen-validation-mode cache_on_miss
--g-telemetry profile
```

Recognized but unsupported REGENIE options, such as `--bed`, `--pgen`, `--spa`, and categorical covariate flags, fail loudly instead of being silently ignored.

---

## Configuration files

Config files use the same option names as the CLI, grouped by section. The merge order is:

```text
packaged defaults in src/g/config.default.toml
        < values in --config
        < explicit CLI flags
```

Create and validate a starter config:

```bash
uv run g config init --out regenie.toml
uv run g config validate regenie.toml
```

Example quantitative config:

```toml
[input]
bgen = "data/1kg_chr22_full.bgen"
sample = "data/1kg_chr22_full.sample"
phenoFile = "data/pheno_cont.txt"
phenoCol = "phenotype_continuous"
covarFile = "data/covariates.txt"
covarColList = "age,sex"
pred = "data/baselines/regenie_step1_qt_pred.list"

[trait]
step = 2
qt = true
bt = false
bsize = 8192

[output]
out = "data/example_regenie2"

[g.compute]
device = "gpu"
staging-depth = 2
trusted-no-missing-diploid = false
trusted-bgen-validation-mode = "cache_on_miss"
sample-key-mode = "iid"
multi-phenotype-sample-mode = "per-phenotype"

[g.output]
format = "parquet"
writer-threads = 4
writer-queue-depth = 16
chunks-per-arrow-file = 16
arrow-compression = "zstd"
resume = false
resume-mode = "fast"

[g.diagnostics]
telemetry = "progress"
log-stderr = true
```

Run with config and override a single option:

```bash
uv run g regenie --config regenie.toml --g-device cpu
```

Explain any known option:

```bash
uv run g config explain pThresh
uv run g config explain g-telemetry
```

---

## Python API

The public Python entrypoint mirrors the CLI/config path.

```python
from pathlib import Path

import g

artifacts = g.regenie.from_options(
    {
        "step": 2,
        "qt": True,
        "bgen": Path("data/1kg_chr22_full.bgen"),
        "sample": Path("data/1kg_chr22_full.sample"),
        "phenoFile": Path("data/pheno_cont.txt"),
        "phenoCol": "phenotype_continuous",
        "covarFile": Path("data/covariates.txt"),
        "covarColList": "age,sex",
        "pred": Path("data/baselines/regenie_step1_qt_pred.list"),
        "out": Path("data/example_regenie2"),
        "g-device": "cpu",
    }
)

print(artifacts.output_run_directory)
print(artifacts.final_parquet)
```

Loading TOML from Python:

```python
import g
from g.interface.config import RegenieConfig

config = RegenieConfig.from_toml("regenie.toml")
artifacts = g.regenie(config)
```

For multiple phenotypes, `artifacts.phenotype_artifacts` contains one `RunArtifacts` object per phenotype.

---

## Input conventions

### Genotypes

- BGEN 1.2 is the active supported genotype source.
- `.sample` files are supported; embedded BGEN sample IDs are also supported when available.
- The performance-oriented trusted path is enabled with `--g-trusted-no-missing-diploid` and controlled by `--g-trusted-bgen-validation-mode`.

Trusted validation modes:

| Mode | Meaning |
|---|---|
| `cache_on_miss` | Validate on cache miss, then reuse validation result |
| `force_validate` | Always validate before using trusted fast path |
| `assume_validated` | Skip validation; expert mode only |

### Phenotypes and covariates

- Phenotype and covariate tables are parsed natively in Rust.
- Tables are expected to include `IID`; `FID` is required when `--g-sample-key-mode fid_iid` is used.
- `--g-sample-key-mode iid` requires globally unique non-empty IIDs.
- `--g-sample-key-mode fid_iid` requires unique `(FID, IID)` pairs.
- Binary phenotypes use REGENIE-style coding: `1 = control`, `2 = case`; internally these are recoded to `0/1`.
- Missing tokens include empty string, `NA`, `NaN`, `nan`, and `-9`.

---

## Multi-phenotype execution

`g` accepts multiple phenotypes through repeated `--phenoCol` or `--phenoColList`.

Default behavior:

```text
--g-multi-phenotype-sample-mode per-phenotype
```

This preserves per-phenotype sample inclusion semantics and may run separate engine passes for phenotypes with different complete-case sample sets.

Opt-in batching mode:

```bash
--g-multi-phenotype-sample-mode complete-case
```

This runs a shared complete-case sample intersection for all requested phenotypes and can reuse genotype decoding/transfer across traits. It is faster when applicable, but it is **not statistically equivalent** to running each phenotype with its own sample set. Use it only when the shared complete-case intersection is intended.

---

## Output layout

Given:

```bash
--out data/example_regenie2
--phenoCol phenotype_continuous
```

`g` writes a run directory like:

```text
data/example_regenie2.g/
  logs/
    events.jsonl
    progress.jsonl
  trait_0001_phenotype_continuous.regenie2_linear.run/
    chunks/
      chunk_000000000.arrow
      chunk_000000001.arrow
    effective_config.toml
    run_manifest.json
    output_stage_timings.json        # when output timing collection is enabled
    final.parquet                    # when Parquet finalization is enabled
```

Binary runs use `.regenie2_binary.run`.

The final table schema follows REGENIE Step 2-style association fields:

```text
CHROM, GENPOS, ID, ALLELE0, ALLELE1, A1FREQ, INFO, N,
TEST, BETA, SE, CHISQ, LOG10P, EXTRA
```

`BETA`, `SE`, `CHISQ`, and `LOG10P` are persisted as float32 in Arrow and
Parquet outputs. This matches the current REGENIE-compatible writer schema; any
float64 internal arrays are narrowed before writing, and the run manifest records
`output_writer.result_statistic_dtype = "float32"`.

`EXTRA` is null for ordinary successful rows and `TEST_FAIL` for failed binary correction/statistic rows.

---

## Resume and reproducibility

Every run writes:

```text
effective_config.toml
run_manifest.json
```

The manifest records execution-plan-affecting state such as input file fingerprints, sample/covariate/prediction choices, chunk size, binary correction plan, trusted BGEN policy, output writer settings, and an execution-plan hash.

Resume controls:

```bash
--g-resume
--g-resume-mode fast|strict
```

Use `strict` when correctness is more important than fast startup; use `fast` for normal uninterrupted/manifest-backed resumes.

---

## Telemetry and profiling

Telemetry is configured through `[g.diagnostics]` or `--g-*` CLI flags.

| Mode | Use case | Behavior |
|---|---|---|
| `off` | Minimal runtime overhead | No run telemetry files |
| `progress` | Normal runs | Lifecycle events and throttled progress |
| `profile` | Benchmarks | Adds stage timings and profile summaries; may synchronize JAX work for measurement |
| `trace` | Deep debugging | Adds high-volume native/JAX traces; use with small runs or variant caps |

Example:

```bash
uv run g regenie \
  --config regenie.toml \
  --g-telemetry profile \
  --g-log-dir data/example_regenie2.g/logs \
  --g-progress-interval-seconds 5 \
  --g-progress-interval-chunks 10
```

Default telemetry paths are resolved under:

```text
<out>.g/logs/
  events.jsonl
  progress.jsonl
  profile.summary.json       # profile/trace modes
  stage-timings.json         # profile/trace modes
  trace.jsonl                # trace mode
```

Important performance rule: production logging avoids host-device synchronization. Profile and trace modes may intentionally synchronize JAX work to produce accurate stage timings.

---

## Performance notes

The most important runtime knobs are:

| Option | Purpose |
|---|---|
| `--bsize` | Variants per chunk; default `8192` |
| `--g-device cpu|gpu` | JAX execution device |
| `--g-staging-depth` | Native callback staging depth |
| `--g-trusted-no-missing-diploid` | Enables trusted BGEN fast path after validation policy |
| `--g-bgen-decode-tile-variant-count` | Native BGEN decode tile size |
| `--g-writer-threads` | Output writer worker count |
| `--g-writer-queue-depth` | Output writer queue depth |
| `--g-output-chunks-per-arrow-file` | Number of compute chunks grouped into one Arrow IPC file |
| `--g-firth-batch-size` | Binary approximate-Firth batch size |
| `--g-firth-candidate-capacity` | Fixed device candidate capacity before fallback |
| `--g-jax-persistent-cache` | Enable JAX persistent compilation cache |

Benchmark commands:

```bash
just benchmark-rust
just benchmark-bgen-reader
just benchmark-regenie-comparison
just benchmark-regenie-comparison-gpu
just profile-regenie-comparison
just profile-regenie-comparison-gpu
```

Binary-specific GPU runs:

```bash
just setup-regenie2-binary-gpu-inputs
just verify-regenie2-binary-gpu-inputs
just regenie2-binary-gpu-smoke
just regenie2-binary-gpu
```

Fair performance comparisons require equivalent statistical modes. Compare score-only to score-only, and compare approximate Firth only when both tools are using approximate Firth with the same fallback threshold.

---

## Architecture

```text
src/g/
  api.py                         # thin public Python API
  cli.py                         # Click CLI generated from OptionSpec
  interface/
    options.py                   # single option registry for CLI/TOML/Python names
    config.py                    # typed config, TOML load/dump, validation
  execution_plan.py              # immutable normalized run plans
  runner.py                      # runtime setup, telemetry, dispatch, artifacts
  engine/
    regenie2_pipeline.py         # native-driven BGEN pipeline wrappers
    callbacks.py                 # JAX callback workers and result materialization
    native_dispatch.py           # Rust engine / alignment / prediction-source bridge
    telemetry.py                 # JSONL run telemetry
    timing.py                    # synchronized profile summaries
    preflight.py                 # pre-run validation
  compute/
    regenie2_linear/             # quantitative state and score kernels
    regenie2_binary/             # binary score, candidates, Firth, diagnostics
  io/
    output.py                    # output path, manifest, writer-session bridge
    source.py                    # genotype source config
```

Rust native modules:

```text
src/genotype/                    # BGEN mmap/index/decode/preprocess/profile
src/sample.rs                    # native sample/phenotype/covariate alignment
src/output/                      # Arrow IPC chunks, Parquet finalization, manifests
src/prediction/                  # REGENIE Step 1 prediction loading/alignment
src/python/                      # PyO3 bindings and logging bridge
```

Design principles:

- no Python DataFrame library in the core execution path;
- no hidden runtime constants inside JAX kernels when they affect compiled shapes;
- no silent acceptance of unsupported REGENIE flags;
- run manifests must protect resume correctness;
- profiling should be structured and reproducible;
- performance work should be benchmarked end to end, not just in microbenchmarks.

---

## Development workflow

Common checks:

```bash
just format
just lint
just typecheck
just check
just test
```

Reduced-toolchain local checks:

```bash
just check-local
just test-local
just test-local-focused
```

Native extension and Rust benchmarks:

```bash
just install-perf-extension
just benchmark-rust
just benchmark-bgen-reader
```

Coverage:

```bash
just coverage-python
just coverage-rust
just coverage
```

Codex task-farm helpers:

```bash
just codex-tasks-sync
just codex-tasks-doctor
just codex-tasks-list
just codex-tasks-run --jobs 4
just codex-tasks-status
just codex-tasks-review 1
just codex-tasks-integrate 1
just codex-tasks-integrate-ready
```

The task farm uses isolated Git worktrees for worker branches and an integration workflow for reviewed branches. See `docs/codex-task-farm.md`.

---

## Documentation map

Useful docs:

- `docs/STYLEGUIDE.md` — coding rules and review expectations
- `docs/NO_NIX_DEVELOPMENT.md` — local reduced-toolchain workflow
- `docs/UBUNTU_SLURM_DEVELOPMENT.md` — server and SLURM workflow
- `docs/linear-regenie-step2-learning.md` — quantitative Step 2 math notes
- `docs/binary-regenie-step2-learning.md` — binary Step 2 / approximate Firth notes
- `docs/simd-optimization-reference.md` — SIMD decisions and BGEN decode optimization notes
- `docs/logging-setup.md` — telemetry and logging design notes
- `docs/codex-task-farm.md` — Codex worktree automation

Some older docs may be historical or pending cleanup. Prefer this README, `ROADMAP.md`, and the focused learning/reference docs for current orientation.

---

## Known limitations

- `g` is pre-release and actively changing.
- Only BGEN-backed REGENIE Step 2 is in active scope.
- REGENIE Step 1 is not implemented.
- BED/PGEN inputs are not implemented.
- SPA is not implemented.
- Exact Firth without `--approx` is not implemented.
- Binary approximate Firth is implemented but remains the most numerically sensitive path.
- Multi-phenotype `complete-case` mode changes sample inclusion semantics by design.
- GPU acceleration is workload-dependent; single-trait runs may be limited by native decode, transfer, or output rather than JAX compute.

---

## Citation / acknowledgement

`g` follows the REGENIE Step 2 workflow and option vocabulary where possible. Use upstream REGENIE for Step 1 prediction generation and as the primary external parity reference while `g` is under active development.
