# g Documentation

`g` is a pre-release GWAS engine focused on BGEN-backed REGENIE Step 2 association scans. It exposes a REGENIE-style CLI, TOML configuration, and a Python API while using Rust for native file handling and JAX for quantitative and binary association kernels.

`g` does not implement REGENIE Step 1. Use upstream `regenie` to produce Step 1 prediction lists, then use `g` for Step 2 scans.

## Current Scope

| Area | Status |
| --- | --- |
| Quantitative REGENIE Step 2 (`--qt`) | Primary supported workflow |
| Binary score-test Step 2 (`--bt`) | Supported, evolving |
| Binary approximate Firth fallback (`--bt --firth --approx`) | Implemented, parity and performance sensitive |
| REGENIE Step 1 | Not implemented |
| BGEN 1.2 input | Supported |
| BED/PGEN input | Recognized, not implemented |
| Output | Arrow chunks and Parquet run outputs |
| GPU execution | Supported through JAX when the environment is configured |

## Start Here

- [Getting Started](getting-started.md) explains the shortest path through setup and a first scan.
- [Installation](installation.md) covers local, GPU, and server bootstrap commands.
- [Quickstart](quickstart.md) shows quantitative, binary, and approximate-Firth examples.
- [CLI](cli.md) and [Configuration](configuration.md) document the main user interface.
- [Input and Output](input-output.md) describes file expectations and run artifacts.
- [GPU and SLURM](gpu-and-slurm.md) covers accelerator and cluster notes.

## Reference / Internal Notes

The repository also contains design notes, learning documents, and planning references. These are preserved as source material for development and may be more implementation-focused than the user guide:

- [Style Guide](STYLEGUIDE.md)
- [Ubuntu SLURM Development](UBUNTU_SLURM_DEVELOPMENT.md)
- [No-Nix Development](NO_NIX_DEVELOPMENT.md)
- [Configuration and CLI Architecture](configuration_cli_architecture.md)
- [Agent Learning](agent-learning.md)
- [Linear REGENIE Step 2 Learning](linear-regenie-step2-learning.md)
- [Binary REGENIE Step 2 Learning](binary-regenie-step2-learning.md)
- [Performance Discovery Playbook](performance-discovery.md)
- [SIMD Optimization Reference](simd-optimization-reference.md)
- [Codex Task Farm](codex-task-farm.md)
