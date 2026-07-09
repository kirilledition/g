# Architecture

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; durable implementation map | main branch as of 2026-07-09 runtime architecture | Development maintainers |

`g` separates the user interface, immutable run contracts, native I/O, JAX kernels, and output writer so performance-sensitive behavior is explicit.

## High-level Flow

```text
CLI / TOML
        |
g-interface (RegenieConfig)
        |
NativeRunEngineSession (g-engine orchestration)
        |
Rust BGEN decode + sample alignment + output writer
        |
Python JAX callbacks (compute + materialize)
        |
Arrow chunks + optional finalized Parquet
```

## Python Runtime

```text
src/g/
  runner/                        CLI shell, JAX setup wiring, telemetry session, run entry
  engine/
    callbacks/                   JAX compute callbacks, thin glue over native runtime resources
      factory.py                 callback factory for native-owned runs
      base.py                    local state + write/compute side effects
      linear.py / binary.py      association callbacks
    timing.py                    stage timing recorder bridge
  compute/
    regenie2_linear/             quantitative state and score kernels
    regenie2_binary/             binary score, candidates, Firth, diagnostics
  jax_runtime.py                 JAX runtime policy, resolution, diagnostics, setup
  types.py                       shared Python enums/types
```

## Native Runtime

```text
crates/plan/src/                 immutable run/config/prepared-plan contracts
crates/interface/src/            CLI/config frontend, clap, toml/Serde
crates/genotype/src/             BGEN mmap/index/decode/preprocess
crates/input/src/                sample/phenotype/covariate and prediction alignment
crates/output/src/               Arrow IPC chunks, Parquet finalization, manifests
crates/runtime/src/              logging, telemetry, timing, shutdown, Rayon/JAX policy
crates/engine/src/               orchestration, preflight, schedule plans, delivery policy
src/binding/engine/              PyO3 adaptation including callback runtime resources
```

## Design Principles

- No Python DataFrame library in the core execution path.
- No hidden runtime constants inside JAX kernels when they affect compiled shapes.
- Unsupported REGENIE flags should stay out of the CLI/config schema until implemented.
- Run manifests protect resume correctness.
- Profiling should be structured and reproducible.
- Performance work should be benchmarked end to end.
- Python does not own domain orchestration; `NativeRunEngineSession` owns the run.
- Callback queues, workers, slots, and buffer pools are native-owned.

More detailed internal notes:

- [Architecture Cleanup](architecture-cleanup.md)
- [Configuration Frontend](configuration-frontend.md)
- [Native I/O](native-io.md)
- [Compute Kernels](compute-kernels.md)
- [Testing and Parity](testing-and-parity.md)
- [Benchmarking](benchmarking.md)
- [SIMD Optimization Reference](simd-optimization-reference.md)
