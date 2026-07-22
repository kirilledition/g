# ✨ g

GWAS engine with GPU acceleration via JAX

[![PR CI](https://github.com/kirilledition/g/actions/workflows/pr-ci.yml/badge.svg)](https://github.com/kirilledition/g/actions/workflows/pr-ci.yml)
[![Science Monthly](https://github.com/kirilledition/g/actions/workflows/science-monthly.yml/badge.svg)](https://github.com/kirilledition/g/actions/workflows/science-monthly.yml)

`g` is a pre-release GWAS engine for BGEN-backed REGENIE Step 2 association
scans. It is a Rust host application with a Python/JAX numerical backend: Rust
owns the CLI, configuration, planning, input, scheduling, runtime, telemetry,
resume, and Parquet output, while JAX and optional CUDA kernels perform the
association mathematics. Python embedding is not a supported API.

`g` does not implement REGENIE Step 1. Use upstream `regenie` to produce Step 1
prediction lists, then use `g` for Step 2 scans.

## Current Capabilities

| Area | Status |
| --- | --- |
| Commands | `g regenie` runs one Step 2 request; `g batch` runs complete TOML configurations sequentially in one process. |
| Quantitative REGENIE Step 2 (`--qt`) | Supported. |
| Binary score-test Step 2 (`--bt`) | Supported. |
| Binary approximate Firth fallback (`--bt --binary-fallback firth_approximate`) | Experimental; upstream golden parity is not complete. |
| REGENIE Step 1 | Not implemented |
| Genotype input | Layout 2 BGEN with uncompressed, zlib, or BGEN v1.3 Zstandard blocks; an Oxford `.sample` file is required. |
| Sample identity | Strict non-empty, unique `(FID, IID)` pairs across aligned inputs. |
| Multiple phenotypes | Supported; per-phenotype complete-case samples are the default, with a shared complete-case mode available. |
| Result output | One chunked Parquet dataset per phenotype, plus effective configuration, manifest, logs, and telemetry. |
| Resume | Strict manifest-validated resume from committed Parquet chunks. |
| Execution | CPU by default; JAX GPU execution and eligible automatic CUDA fast paths are supported. |
| Unsupported surface | BED/PGEN, sample and variant filters, categorical covariates, SPA, exact Firth, and time-to-event traits. |

Unsupported REGENIE options are rejected rather than silently ignored. See
[Compatibility](documentation/public/compatibility.md) for the exact current
surface.

## Installation

`g` is installed from a Git checkout because it is not published on PyPI.
The primary supported path is Linux with Python 3.14, Rust 1.97.1, `uv`, a
C/C++ compiler, and Mold. GPU execution additionally requires a compatible
NVIDIA driver and node.

```bash
git clone https://github.com/kirilledition/g.git
cd g
uv python install 3.14
uv sync --python 3.14 --no-dev --frozen
uv run g --help
uv run g regenie --help
```

The supported runtime is exactly `jax==0.11.0` with `jaxlib==0.11.0`; GPU
installs use its CUDA 12 extra for the project's NVIDIA V100/R535 deployment.
For CPU and GPU prerequisites, cluster installs, and development setup, use
[Installation](documentation/public/installation.md).

## First Step 2 Run

```bash
uv run g regenie \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --covarFile /path/to/covariates.tsv \
  --covarCol age --covarCol sex \
  --pred /path/to/regenie_step1_pred.list \
  --out /path/to/output/g_regenie2
```

The packaged configuration selects CPU execution by default. `--out` is a
prefix: this example creates `/path/to/output/g_regenie2.g/`, with a separate
run directory and `parts/` Parquet dataset for each requested phenotype.

For quantitative, binary, approximate-Firth, multi-phenotype, GPU, and output
examples, continue with [Quickstart](documentation/public/quickstart.md).

## Batch Runs

Use `g batch` to run compatible complete TOML configurations sequentially while
reusing one Python/JAX process:

```bash
uv run g batch \
  --config chromosome_21.toml \
  --config chromosome_22.toml
```

Batch mode validates shared process policy and output-root separation before
starting work, executes configurations in argument order, and stops at the
first failure. See [CLI](documentation/public/cli.md#batch-runs) and
[Configuration](documentation/public/configuration.md) for the complete
contract.

## Outputs And Resume

A typical run creates:

```text
<out>.g/
  logs/
    events.jsonl
  trait_0001_<phenotype>.regenie2_linear.run/
    effective_config.toml
    run_manifest.json
    parts/
      part_000000000_000000007.parquet
```

Binary run directories use `.regenie2_binary.run`. Parquet `parts/` is the
completed result dataset; there is no consolidation step. Resume is opt-in with
`[output].resume = true` and rejects changes to result-affecting inputs or
execution policy. See [Output Files](documentation/public/output-files.md) and
[Resume and Manifest](documentation/public/resume-and-manifest.md).

## Documentation Map

Start with the [User Guide](documentation/public/index.md). Canonical
user-facing contracts live under `documentation/public/`:

| Need | Page |
| --- | --- |
| Orientation and installation | [Getting Started](documentation/public/getting-started.md), [Installation](documentation/public/installation.md) |
| Commands and worked examples | [Quickstart](documentation/public/quickstart.md), [CLI](documentation/public/cli.md) |
| Supported REGENIE surface | [Compatibility](documentation/public/compatibility.md) |
| TOML settings and merge rules | [Configuration](documentation/public/configuration.md) |
| Input and output contracts | [Input Files](documentation/public/input-files.md), [Output Files](documentation/public/output-files.md) |
| Statistical models and result interpretation | [Algorithm](documentation/public/algorithm.md) |
| Restarting interrupted work | [Resume and Manifest](documentation/public/resume-and-manifest.md) |
| GPU and cluster operation | [GPU and Clusters](documentation/public/gpu-and-clusters.md) |
| Tuning and diagnosis | [Performance Guide](documentation/public/performance-guide.md), [Troubleshooting](documentation/public/troubleshooting.md) |

Contributor guidance starts at [Development](documentation/development/index.md):

- [Architecture](documentation/development/architecture.md)
- [Configuration frontend](documentation/development/configuration-frontend.md)
- [Testing and parity](documentation/development/testing-and-parity.md)
- [Benchmarking](documentation/development/benchmarking.md)
- [Development tooling](documentation/development/tooling.md)
- [Justfile command reference](documentation/development/justfile.md)
- [Style guide](documentation/development/style-guide.md)

Internal scratchpad notes live under `documentation/scratchpad/`. They are not
part of the primary published navigation and may be stale.

## Development And Documentation

Use `just help` to inspect project recipes. The standard development entry
points are:

```bash
just dev-bootstrap
just doctor
just check
just test
```

Build or serve the documentation with:

```bash
just docs-build
just docs-serve
```

The published documentation site is expected at:

```text
https://kirilledition.github.io/g/
```

<!-- code-size-summary:start -->
## Code Size

Generated from Git-tracked files under `crates/` and `src/` using [`cloc`](https://github.com/AlDanial/cloc).

| Language | Files | Blank | Comment | Code |
| --- | ---: | ---: | ---: | ---: |
| Rust | 175 | 2,660 | 973 | 26,567 |
| Python | 36 | 644 | 453 | 3,915 |
| C++ | 2 | 116 | 6 | 1,036 |
| Markdown | 13 | 130 | 0 | 438 |
| CUDA | 2 | 23 | 4 | 382 |
| TOML | 12 | 37 | 5 | 234 |
| C/C++ Header | 1 | 12 | 3 | 56 |
| **Total** | 241 | 3,622 | 1,444 | 32,628 |

`cloc` version: `2.10`.
<!-- code-size-summary:end -->
