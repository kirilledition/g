# Getting Started

This page is a short orientation for a first local run. The project is still pre-release, so prefer the repository README and these docs over assumptions from older branches or external examples.

## What You Can Run

The active user workflow is REGENIE Step 2 over BGEN input:

1. Generate Step 1 prediction lists with upstream `regenie`.
2. Run `g regenie --step 2` with quantitative (`--qt`) or binary (`--bt`) trait mode.
3. Inspect the run directory, manifest, chunks, logs, and optional Parquet output.

`g` currently recognizes some REGENIE flags that are not implemented, including BED/PGEN inputs, SPA, and exact Firth without `--approx`. These fail explicitly rather than being silently ignored.

## Local Setup

For a CPU-oriented local environment:

```bash
just bootstrap
just doctor
just check-local
```

For a GPU-capable environment:

```bash
just bootstrap-gpu
just doctor-jax
```

The project requires Python `>=3.14,<3.15`, Rust/Cargo, `uv`, `just`, and `maturin`. Baseline comparisons and data-preparation workflows also use tools such as `plink`, `plink2`, `regenie`, and `zstd`.

## First Data-backed Run

The repository provides data-preparation recipes for local 1000 Genomes chromosome 22 fixture data and simulated phenotypes:

```bash
just setup-data
```

Binary examples also need REGENIE Step 1 baseline predictions:

```bash
just setup-binary-baseline
```

Then follow [Quickstart](quickstart.md) for concrete commands.

## Documentation Commands

The documentation site is built from `docs/`:

```bash
just docs-serve
just docs-build
```

Generated `site/` output is local build output and is not committed.
