# Architecture

`g` separates the user interface, execution plan, native I/O, JAX kernels, and output writer so performance-sensitive behavior is explicit.

## High-level Flow

```text
CLI / TOML / Python API
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
  api.py                         public Python API
  cli.py                         thin Python dispatcher into the Rust CLI frontend
  interface/
    config.py                    compatibility wrappers around Rust-owned config objects
  execution_plan.py              immutable normalized run plans
  runner.py                      runtime orchestration, telemetry, dispatch, artifacts
  jax_runtime.py                 JAX runtime policy, reports, diagnostics
  jax_setup.py                   order-sensitive JAX config mutation and GPU validation
  runtime_paths.py               node-local and cluster path policy
  engine/
    regenie2_pipeline.py         native-driven BGEN pipeline wrappers
    callbacks.py                 JAX callback workers and result materialization
    native_dispatch.py           Rust bridge for engine/alignment/predictions
    telemetry.py                 JSONL run telemetry
    timing.py                    synchronized profile summaries
    preflight.py                 pre-run validation
  compute/
    regenie2_linear/             quantitative state and score kernels
    regenie2_binary/             binary score, candidates, Firth, diagnostics
  io/
    output.py                    output paths, manifest, writer bridge
```

## Native Runtime

```text
src/config_frontend/             Rust CLI/config frontend, clap, toml/Serde, option specs
src/genotype/                    BGEN mmap/index/decode/preprocess/profile
src/sample.rs                    sample/phenotype/covariate alignment
src/output/                      Arrow IPC chunks, Parquet finalization, manifests
src/python/                      PyO3 config/runtime bindings and logging bridge
```

## Design Principles

- No Python DataFrame library in the core execution path.
- No hidden runtime constants inside JAX kernels when they affect compiled shapes.
- Unsupported REGENIE flags should stay out of the CLI/config schema until implemented.
- Run manifests protect resume correctness.
- Profiling should be structured and reproducible.
- Performance work should be benchmarked end to end.

More detailed internal notes:

- [Configuration and CLI Architecture](configuration_cli_architecture.md)
- [Agent Learning](../scratchpad/agent-learning.md)
- [Linear REGENIE Step 2 Learning](../scratchpad/linear-regenie-step2-learning.md)
- [Binary REGENIE Step 2 Learning](../scratchpad/binary-regenie-step2-learning.md)
- [SIMD Optimization Reference](simd-optimization-reference.md)
