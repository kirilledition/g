# Development

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-06-30 development workflow index | Development maintainers |

Use `uv` for dependency management and `just` for project commands. Run
`just help` before choosing a task-specific recipe.

## Start Here

```bash
just dev-bootstrap
just doctor
```

For GPU-capable development:

```bash
just dev-bootstrap-gpu
just doctor-jax
```

Server-specific CPU/GPU routing, SLURM nodes, caches, and environment variables
belong in [Server Gauss SLURM](server-gauss-slurm.md). Reduced-toolchain local
setup belongs in [No-Nix Development](no-nix-development.md).

## Common Checks

Login-node-safe local checks:

```bash
just check-local
just test-local-focused
```

Repository-wide checks:

```bash
just format
just lint
just typecheck
```

CI checks the README code-size summary but does not push refresh commits to
protected `main`. After changing tracked files under `crates/`, `src/`, or
`README.md`, run `scripts/update_readme_code_summary.py` with `cloc` available
and commit the updated README in the same branch.

Full CPU validation, GPU validation, large test suites, and native builds should
run through the appropriate local or SLURM workflow for the current host. See
[Testing and Parity](testing-and-parity.md) and [Server Gauss SLURM](server-gauss-slurm.md).

## Documentation

Serve and build the Zensical site:

```bash
just docs-serve
just docs-build
just docs-check
```

When changing user-facing CLI behavior, configuration, input/output contracts,
runtime behavior, performance assumptions, or deployment workflow, update the
relevant page under `documentation/public/` in the same branch. When changing
development workflows or docs infrastructure, update the relevant page under
`documentation/development/`.

Generated `documentation_rendered_website/` output is local build output and is
not committed.

See [Documentation Operations](documentation.md) for publishing setup, theme
configuration, GitHub Pages settings, and documentation workflow behavior.

## Development Contracts

| Topic | Page |
| --- | --- |
| Code style and review rules | [Style Guide](style-guide.md) |
| Architecture map | [Architecture](architecture.md) |
| Architecture cleanup and Rust migration | [Architecture Cleanup](architecture-cleanup.md) |
| CLI/TOML/Python configuration frontend | [Configuration Frontend](configuration-frontend.md) |
| Native BGEN, sample, output, and manifest boundaries | [Native I/O](native-io.md) |
| Native integer boundary policy | [Integer Policy](integer-policy.md) |
| Native integer type audit | [Integer Type Audit](integer-type-audit.md) |
| JAX quantitative, binary, and Firth kernels | [Compute Kernels](compute-kernels.md) |
| Testing, correctness, and parity expectations | [Testing and Parity](testing-and-parity.md) |
| Pre-release REGENIE Step 2 parity gate | [REGENIE Parity Suite](regenie-parity-suite.md) |
| Benchmark taxonomy and protocols | [Benchmarking](benchmarking.md) |
| Telemetry and logging architecture | [Telemetry](telemetry.md) |
| Development tooling | [Tooling](tooling.md) |
| Justfile recipes | [Justfile Command Reference](justfile.md) |
| Roadmap | [Roadmap](roadmap.md) |

Internal scratchpad notes are direct-path development material and may be stale.
