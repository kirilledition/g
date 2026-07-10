# Roadmap

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | project direction as of 2026-07-10 | Development maintainers |

## Product Goal

Build a REGENIE-compatible, BGEN-backed Step 2 engine with Rust I/O, JAX
compute, reproducible config-driven runs, structured telemetry, and resumable
Arrow/Parquet output.

## Active Supported Scope

- Quantitative REGENIE Step 2 through `g regenie --step 2 --qt`.
- Binary score-only Step 2 through `g regenie --step 2 --bt`.
- Binary `--firth --approx` through scalar approximate Firth, with parity work
  still treated as active engineering surface.
- BGEN 1.2 input with Oxford `.sample` metadata.
- Native TOML/CLI input compiled into one Rust run request.
- Arrow chunk output and final Parquet materialization.
- REGENIE-compatible Step 2 text output through `[output].format = "regenie"`.
- Runtime telemetry, progress logging, profile summaries, and trace mode.

## Not Yet Supported

- REGENIE Step 1.
- SPA.
- Exact Firth without `--approx`.
- BED/PGEN input.
- Full public support for all REGENIE Step 2 flags.

## Tracked Follow-Ups

Roadmap work should be tracked in Linear rather than as unchecked tasks in this
file. The current docs-task audit generated focused follow-ups for packed8
custom-kernel profiling, trace-mode event caps, and binary benchmark
diagnostics. See
[Agent Memory](../scratchpad/memory.md) for the audit summary and Linear links.

## Performance Direction

- Group phenotypes by identical sample, covariate, and prediction alignment so
  multi-phenotype runs can preserve per-trait semantics while reducing BGEN
  rereads.
- Keep AVX2 as the production SIMD target for trusted BGEN decode and
  preprocessing. AVX-512 and arbitrary selected-subset SIMD are deferred unless
  new measurements justify them.
- Use native genotype sums and square sums in linear and binary kernels to
  avoid redundant GPU reductions.
- Reduce output-writer copies and clarify ownership of chunk metadata and
  result buffers.
- Measure synchronization points explicitly in profiling mode while keeping
  production telemetry low overhead.

## Architecture Direction

- Route CLI and TOML through one Rust-owned production path:

```text
g-interface -> g-plan -> g-engine::RunEngine
                              |
                         Python/JAX backend
```

- Keep runtime core code free of DataFrame dependencies.
- Reduce Python to the console bootstrap, JAX setup, JAX backend, and JAX
  kernels. Rust owns all surrounding orchestration and I/O.
- Use only the coarse `prepare_group`, `prepare_chromosome`, `compute_batch`,
  and `materialize_batch` Python boundary.
- Keep JAX imports behind explicit runtime boundaries where they are needed to
  preserve process-global runtime policy.
- Treat prepared-run hashes and manifest metadata as the source of resume
  compatibility.
- Keep production telemetry low overhead and free of accidental JAX
  synchronization.
- Keep profile and trace modes explicitly diagnostic because they may perturb
  performance.
- Follow [Architecture Cleanup](architecture-cleanup.md) for the ordered
  migration and Rust cleanup program.
