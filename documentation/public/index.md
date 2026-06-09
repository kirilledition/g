# g Documentation

`g` is a pre-release GWAS engine focused on BGEN-backed REGENIE Step 2
association scans. It exposes a REGENIE-style CLI, TOML configuration, and a
Python API while using Rust for native file handling and JAX for quantitative
and binary association kernels.

`g` does not implement REGENIE Step 1. Use upstream `regenie` to produce Step 1
prediction lists, then use `g` for Step 2 scans.

## Start Here

- [Getting Started](getting-started.md) gives the shortest orientation from
  install to first run.
- [Installation](installation.md) covers CPU, GPU, cluster, and development
  setup paths.
- [Quickstart](quickstart.md) shows quantitative, binary, approximate-Firth,
  GPU, and REGENIE-text command examples.
- [Compatibility](compatibility.md) lists supported and recognized-but-unsupported
  REGENIE behavior.

## Reference Pages

| Topic | Canonical page |
| --- | --- |
| CLI grammar, flags, boolean semantics, and exit behavior | [CLI](cli.md) |
| TOML sections, merge order, effective config, and defaults policy | [Configuration](configuration.md) |
| Genotype, sample, phenotype, covariate, and prediction files | [Input Files](input-files.md) |
| Run directories, output formats, schema, and telemetry files | [Output Files](output-files.md) |
| Resume modes, manifest compatibility, and graceful interruption | [Resume and Manifest](resume-and-manifest.md) |
| Public Python wrapper | [Python API](api-python.md) |
| Statistical models and result interpretation | [Algorithm](algorithm.md) |
| GPU, cluster, and SLURM operation | [GPU and Clusters](gpu-and-clusters.md) |
| Performance tuning and measurement | [Performance Guide](performance-guide.md) |
| Common failures and first checks | [Troubleshooting](troubleshooting.md) |

Development-team documentation starts at [Development](../development/index.md).
Internal scratchpad notes are kept under `documentation/scratchpad/` and may be
stale.
