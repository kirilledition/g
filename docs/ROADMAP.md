# Roadmap

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
- TOML, CLI, and Python entry points normalized into `RegenieConfig`.
- Arrow chunk output and final Parquet materialization.
- Runtime telemetry, progress logging, profile summaries, and trace mode.

## Not Yet Supported

- REGENIE Step 1.
- SPA.
- Exact Firth without `--approx`.
- BED/PGEN input.
- REGENIE-compatible text output.
- Full public support for all REGENIE Step 2 flags.

## Near-Term P0

- JIT and fuse the binary variant-major score-only path.
- Remove the multi-binary `traits x variants x samples` intermediate.
- Fix O(T x N^2) complete-case multi-phenotype alignment.
- Fail by default on binary null logistic non-convergence.
- Output NaN for invalid binary score statistics instead of
  `CHISQ = 0` and `LOG10P = 0`.

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
- Measure synchronization points explicitly, especially Firth candidate
  counting and profile-mode timing.

## Architecture Direction

- Route CLI, TOML, and Python through:

```text
RegenieConfig -> ExecutionPlan -> runner -> pipeline
```

- Keep runtime core code free of DataFrame dependencies.
- Keep JAX imports behind explicit runtime boundaries where they are needed to
  preserve process-global runtime policy.
- Treat execution-plan hashes and manifest metadata as the source of resume
  compatibility.
- Keep production telemetry low overhead and free of accidental JAX
  synchronization.
- Keep profile and trace modes explicitly diagnostic because they may perturb
  performance.
