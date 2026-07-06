# g Documentation

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-07-01 public docs | Public user docs |

`g` is a pre-release GWAS engine focused on BGEN-backed REGENIE Step 2
association scans. It exposes a REGENIE-style CLI and TOML configuration while
using Rust for native file handling and JAX for quantitative and binary
association kernels.

`g` does not implement REGENIE Step 1. Use upstream `regenie` to produce Step 1
prediction lists, then use `g` for Step 2 scans.

## Choose Your Path

| I want to... | Read |
| --- | --- |
| understand what `g` can run | [Getting Started](getting-started.md) |
| install on a workstation or cluster | [Installation](installation.md) |
| run a first Step 2 scan | [Quickstart](quickstart.md) |
| port an existing REGENIE Step 2 workflow | [Compatibility](compatibility.md), then [CLI](cli.md) |
| understand required input files | [Input Files](input-files.md) |
| find and interpret output files | [Output Files](output-files.md), then [Algorithm](algorithm.md) |
| resume or inspect an interrupted run | [Resume and Manifest](resume-and-manifest.md) |
| run on GPU or SLURM | [GPU and Clusters](gpu-and-clusters.md) |
| tune or measure performance | [Performance Guide](performance-guide.md) |
| fix an error | [Troubleshooting](troubleshooting.md) |

## Reference Pages

| Topic | Canonical page |
| --- | --- |
| CLI grammar, flags, boolean semantics, and exit behavior | [CLI](cli.md) |
| TOML sections, merge order, effective config, and defaults policy | [Configuration](configuration.md) |
| Genotype, sample, phenotype, covariate, and prediction files | [Input Files](input-files.md) |
| Run directories, output formats, schema, and telemetry files | [Output Files](output-files.md) |
| Resume modes, manifest compatibility, and graceful interruption | [Resume and Manifest](resume-and-manifest.md) |
| Statistical models and result interpretation | [Algorithm](algorithm.md) |
| GPU, cluster, and SLURM operation | [GPU and Clusters](gpu-and-clusters.md) |
| Performance tuning and measurement | [Performance Guide](performance-guide.md) |
| Common failures and first checks | [Troubleshooting](troubleshooting.md) |

Development-team documentation starts at [Development](../development/index.md).
Internal scratchpad notes are kept under `documentation/scratchpad/` and may be
stale.
