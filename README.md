# `g` - GPU-oriented REGENIE Step 2 GWAS engine

[![PR CI](https://github.com/kirilledition/g/actions/workflows/pr-ci.yml/badge.svg)](https://github.com/kirilledition/g/actions/workflows/pr-ci.yml)
[![Science Monthly](https://github.com/kirilledition/g/actions/workflows/science-monthly.yml/badge.svg)](https://github.com/kirilledition/g/actions/workflows/science-monthly.yml)

`g` is a pre-release GWAS engine for BGEN-backed REGENIE Step 2 association
scans. It exposes a REGENIE-style CLI, TOML configuration, and a small Python
API while using Rust for native file handling and JAX for quantitative and
binary association kernels.

`g` does not implement REGENIE Step 1. Use upstream `regenie` to produce Step 1
prediction lists, then use `g` for Step 2 scans.

## Current Scope

| Area | Status |
| --- | --- |
| Quantitative REGENIE Step 2 (`--qt`) | Primary supported workflow |
| Binary score-test Step 2 (`--bt`) | Supported, evolving |
| Binary approximate Firth fallback (`--bt --firth --approx`) | Implemented, parity and performance sensitive |
| REGENIE Step 1 | Not implemented |
| BGEN 1.2 input | Supported |
| BED/PGEN input | Recognized, not implemented |
| Output | Arrow, Parquet, and REGENIE Step 2-style text artifacts |
| GPU execution | Supported through JAX when the environment is configured |

## Install

`g` is installed from a Git checkout because it is not published on PyPI.

```bash
git clone https://github.com/kirilledition/g.git
cd g
uv python install 3.14
uv sync --python 3.14 --no-dev
uv run g --help
```

For GPU installs, cluster installs, and development setup, use
[Installation](documentation/public/installation.md).

## Minimal Run Shape

```bash
uv run g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_name \
  --covarFile /path/to/covariates.tsv \
  --covarColList age,sex \
  --pred /path/to/regenie_step1_pred.list \
  --out /path/to/output/g_regenie2
```

See [Quickstart](documentation/public/quickstart.md) for quantitative, binary,
approximate-Firth, GPU, and REGENIE-text examples.

## Documentation Map

User-facing behavior lives under `documentation/public/`:

- [Public guide index](documentation/public/index.md)
- [Compatibility and current scope](documentation/public/compatibility.md)
- [CLI reference](documentation/public/cli.md)
- [Configuration reference](documentation/public/configuration.md)
- [Input files](documentation/public/input-files.md)
- [Output files](documentation/public/output-files.md)
- [Resume and manifests](documentation/public/resume-and-manifest.md)
- [Python API](documentation/public/api-python.md)
- [Performance guide](documentation/public/performance-guide.md)
- [Troubleshooting](documentation/public/troubleshooting.md)

Implementation and contributor guidance lives under `documentation/development/`:

- [Development index](documentation/development/index.md)
- [Architecture](documentation/development/architecture.md)
- [Configuration frontend](documentation/development/configuration-frontend.md)
- [Testing and parity](documentation/development/testing-and-parity.md)
- [Benchmarking](documentation/development/benchmarking.md)
- [Style guide](documentation/development/style-guide.md)

Internal scratchpad notes live under `documentation/scratchpad/`. They are not
part of the primary published navigation and may be stale.

## Documentation Build

```bash
just docs-serve
just docs-build
```

The published documentation site is expected at:

```text
https://kirilledition.github.io/g/
```
