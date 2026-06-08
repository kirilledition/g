# Development Tooling Architecture

Development-only benchmark and profiling entrypoints live in the top-level `tooling/` package. This package is intentionally not listed in `tool.maturin.python-packages` and is not exposed through `[project.scripts]`, so packaged consumers continue to receive only `src/g` and the existing `g` entrypoints.

Hydra is used for development tool parameters, saved development configurations, benchmark campaigns, machine profiles, telemetry defaults, and sweep profiles. Production REGENIE configuration remains in `src/g` and continues to use the TOML-backed `RegenieConfig` and `ExecutionPlan` flow.

The base tooling config is `tooling/configs/config.yaml`. It sets:

```yaml
hydra:
  job:
    chdir: false
```

This preserves the current repository-relative behavior of benchmark commands. Dataset paths still honor `GWAS_ENGINE_DATA_DIR`.

The long-form usage and extension guide is `docs/development/tooling.md`.

The migrated benchmark and profiling commands are invoked through module execution. There is no compatibility-wrapper layer under `scripts/` for these entrypoints. Justfile recipes call the modules directly, for example:

```bash
uv run --no-sync python -m tooling.cli.benchmark_bgen_reader
```

Optional GPU smoke validation should run through SLURM rather than on the head node:

```bash
just slurm-gpu-run 'uv run --no-sync python -m tooling.cli.benchmark_regenie2_binary_hot machine=landau_gpu tool.variant_limit=1000 tool.include_cold_process=false tool.include_finalized_hot=false'
```
