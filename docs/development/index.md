# Development

Use `uv` for dependency management and `just` for project commands.

## Setup

```bash
just bootstrap
just doctor
```

For GPU-capable development:

```bash
just bootstrap-gpu
just doctor-jax
```

For server-specific setup, see [Ubuntu SLURM Development](UBUNTU_SLURM_DEVELOPMENT.md). For reduced-toolchain local setup, see [No-Nix Development](NO_NIX_DEVELOPMENT.md).

## Checks

Common checks:

```bash
just format
just lint
just typecheck
```

Reduced-toolchain local checks:

```bash
just check-local
just test-local
just test-local-focused
```

Full CPU validation belongs on a CPU SLURM node on the gauss server:

```bash
just slurm-cpu-check
just slurm-cpu-test
just slurm-cpu-test-full
just slurm-cpu-rust-build
just slurm-cpu-rust-test
```

`just check` and `just test` remain available as direct recipes, but do not run
the full versions on a login node when they will compile Rust dependencies or
execute the large Python suite. Use [Ubuntu SLURM Development](UBUNTU_SLURM_DEVELOPMENT.md)
for the CPU/GPU routing rules and allocation environment variables.

## Documentation

Serve and build the Zensical site:

```bash
just docs-serve
just docs-build
```

When changing user-facing CLI behavior, configuration, input/output contracts, runtime behavior, performance assumptions, or deployment workflow, update the relevant page under `docs/public/` in the same branch. When changing development workflows or docs infrastructure, update the relevant page under `docs/development/`. Run `just docs-build` before finishing documentation changes.

Generated `site/` output is local build output and is not committed.

See [Documentation Operations](documentation.md) for publishing setup, theme configuration, GitHub Pages settings, and documentation workflow behavior.

## Coding Rules

Follow [Style Guide](STYLEGUIDE.md). Important project rules include:

- full-word variable names;
- strict type coverage;
- module-qualified imports by default;
- dataclasses instead of bare tuples for structured returns;
- Google-style docstrings without duplicated type information.

## Task and Worktree Notes

Symphony work happens in issue-specific worktrees under `/mnt/beegfs/kirill/Projects/g-worktrees/symphony`. Keep changes scoped to the Linear issue and do not commit `data/`, `results/`, local caches, logs, build artifacts, or generated benchmark outputs.

See [Symphony](symphony.md) for Linear-backed multi-agent orchestration.
