# Getting Started

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-07-01 public quick orientation | Public user docs |

Use this page to choose the shortest path through the user guide. The project is
still pre-release, so prefer the repository README and these docs over
assumptions from older branches or external examples.

## If You Are New To `g`

Start with [Installation](installation.md), then run the command checks from the
same checkout and environment that will run your scan:

```bash
uv run g --help
uv run g regenie --help
```

Then use [Quickstart](quickstart.md) for concrete quantitative, binary,
approximate-Firth, GPU, and REGENIE-text command shapes.

## If You Have A REGENIE Workflow

The active user workflow is REGENIE Step 2 over BGEN input:

1. Generate Step 1 prediction lists with upstream `regenie`.
2. Run `g regenie --step 2` with quantitative (`--qt`) or binary (`--bt`) trait
   mode.
3. Inspect the run directory, manifest, chunks, logs, and optional Parquet
   output.

Use [Compatibility](compatibility.md) to check whether the workflow is currently
supported. `g` does not implement Step 1, BED/PGEN inputs, SPA, categorical
covariates, or exact Firth without `--approx`.

## If You Need Exact Contracts

Read the reference page for the contract you are touching:

| Need | Page |
| --- | --- |
| Required input files and sample alignment | [Input Files](input-files.md) |
| Output directories, formats, and schema | [Output Files](output-files.md) |
| Resume behavior and manifests | [Resume and Manifest](resume-and-manifest.md) |
| CLI flags and exit behavior | [CLI](cli.md) |
| TOML merge order and effective config | [Configuration](configuration.md) |
| Statistical model and output interpretation | [Algorithm](algorithm.md) |

## If You Are Debugging

First confirm the command surface:

```bash
uv run g --help
uv run g regenie --help
```

Then use [Troubleshooting](troubleshooting.md) for symptom-specific checks. For
GPU jobs, also verify the target node with [GPU and Clusters](gpu-and-clusters.md).

## Development Setup

If you are changing code, building documentation, or using repository fixture
data recipes, use the separate
[Development Installation](installation.md#development-installation) section.
