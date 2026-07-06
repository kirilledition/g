# Architecture

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; durable implementation map | main branch as of 2026-06-30 runtime architecture | Development maintainers |

`g` separates the user interface, execution plan, native I/O, JAX kernels, and output writer so performance-sensitive behavior is explicit.

## High-level Flow

```text
CLI / TOML
        |
RegenieConfig
        |
ExecutionPlan
        |
Rust BGEN decode + sample alignment + output writer
        |
JAX quantitative / binary association kernels
        |
Arrow chunks + optional finalized Parquet
```

## Python Runtime

```text
src/g/
  cli.py                         thin Python dispatcher into the Rust CLI frontend
  execution_plan.py              immutable normalized run plans
  runner/                        runtime orchestration, telemetry, dispatch, artifacts
  jax_runtime.py                 JAX runtime policy, resolution, diagnostics, setup
  engine/
    regenie2_pipeline/           native-driven BGEN pipeline wrappers
      backend.py                 association backend selection before dispatch
    callbacks/                   JAX callback workers and result materialization
    native_dispatch/             Rust bridge for engine/alignment/predictions
    telemetry.py                 JSONL run telemetry
    timing.py                    synchronized profile summaries
    preflight.py                 pre-run validation
  compute/
    regenie2_linear/             quantitative state and score kernels
    regenie2_binary/             binary score, candidates, Firth, diagnostics
  io.py                          output paths, manifest, writer bridge
```

## Native Runtime

```text
crates/plan/src/                 Rust execution-plan policy payloads and deterministic plan IDs
crates/interface/src/            Rust CLI/config frontend, clap, toml/Serde, option specs
crates/genotype/src/             BGEN mmap/index/decode/preprocess/profile
crates/input/src/                sample/phenotype/covariate and prediction alignment
crates/output/src/               Arrow IPC chunks, Parquet finalization, manifests
src/python/                      PyO3 config/plan/runtime bindings and logging bridge
```

## Design Principles

- No Python DataFrame library in the core execution path.
- No hidden runtime constants inside JAX kernels when they affect compiled shapes.
- Unsupported REGENIE flags should stay out of the CLI/config schema until implemented.
- Run manifests protect resume correctness.
- Profiling should be structured and reproducible.
- Performance work should be benchmarked end to end.

More detailed internal notes:

- [Configuration Frontend](configuration-frontend.md)
- [Native I/O](native-io.md)
- [Compute Kernels](compute-kernels.md)
- [Testing and Parity](testing-and-parity.md)
- [Benchmarking](benchmarking.md)
- [SIMD Optimization Reference](simd-optimization-reference.md)
