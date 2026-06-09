# Architecture

| Status | Applies to | Owner |
| --- | --- | --- |
| Durable implementation map | `src/g`, Rust native modules, and public execution flow | Development maintainers |

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
  cli.py                         Click CLI generated from OptionSpec
  interface/
    options.py                   option registry for CLI/TOML/Python names
    config.py                    typed config, TOML load/dump, validation
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
src/genotype/                    BGEN mmap/index/decode/preprocess/profile
src/sample.rs                    sample/phenotype/covariate alignment
src/output/                      Arrow IPC chunks, Parquet finalization, manifests
src/python/                      PyO3 bindings and logging bridge
```

## Design Principles

- No Python DataFrame library in the core execution path.
- No hidden runtime constants inside JAX kernels when they affect compiled shapes.
- Unsupported REGENIE flags should fail loudly.
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
