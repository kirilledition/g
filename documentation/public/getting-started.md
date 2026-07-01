# Getting Started

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-06-30 public quick orientation | Public user docs |

This page is a short orientation for a first local run. The project is still pre-release, so prefer the repository README and these docs over assumptions from older branches or external examples.

## Install First

Install `g` from source with the consumer flow in [Installation](installation.md). That page keeps
the Python environment inside the checkout's `.venv/` and separates runtime setup from development
setup.

After installation, the basic command checks are:

```bash
uv run g --help
uv run g regenie --help
```

## What You Can Run

The active user workflow is REGENIE Step 2 over BGEN input:

1. Generate Step 1 prediction lists with upstream `regenie`.
2. Run `g regenie --step 2` with quantitative (`--qt`) or binary (`--bt`) trait mode.
3. Inspect the run directory, manifest, chunks, logs, and optional Parquet output.

`g` currently recognizes some REGENIE flags that are not implemented, including BED/PGEN inputs, SPA, and exact Firth without `--approx`. These fail explicitly rather than being silently ignored.

## First Data-backed Run

Prepare or locate the files that a REGENIE Step 2 run needs:

- BGEN genotype file and Oxford `.sample` file unless the BGEN embeds usable sample IDs.
- Phenotype table.
- Optional covariate table.
- REGENIE Step 1 prediction list from upstream `regenie`.

Then follow [Quickstart](quickstart.md) for concrete CPU and GPU command shapes.
Use [Input Files](input-files.md), [Output Files](output-files.md), and
[Resume and Manifest](resume-and-manifest.md) when you need exact contracts.

## Development Setup

If you are changing code, building documentation, or using repository fixture-data recipes, use the
separate [Development Installation](installation.md#development-installation) section.
