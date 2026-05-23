# Current-State Architecture Review

Review date: 2026-05-23

Commit reviewed: `c4e45c39e48dd38b78b88b8207327caa2495b8e0`

Branch state during evidence collection: `main` at `c4e45c39`, one local commit
ahead of the observed local `origin/main` ref `879aca53`. There were no tracked
dirty changes when the review evidence was collected. The branch later advanced
to `475f9318` with additional trusted BGEN SIMD dispatch changes; those later
changes are outside the evidence below. Generated benchmark outputs were written
under the project-local ignored `data/` tree.

Toolchain evidence:

- Host: `gauss`
- Python: `3.14.3`
- `uv`: `0.10.9`
- Rust: `rustc 1.95.0`, `cargo 1.95.0`
- JAX: `jax 0.10.1`, `jaxlib 0.10.1`
- Current GitHub Actions status for this commit: PR CI run `26322236070`
  started at `2026-05-23T03:25:48Z` and completed successfully.

## Verdict

The repository is an active REGENIE step 2 engine with a real hybrid
Python/JAX and Rust/PyO3 architecture. The shipped surface supports CLI and
Python entry points, BGEN dosage input, sample and phenotype alignment, LOCO
prediction files, quantitative step 2 association, binary score testing with
approximate Firth fallback paths, Arrow chunk output, optional final Parquet
output, resumability through manifests, and run telemetry/logging.

The quantitative step 2 path is the most mature association path. On the
representative CPU comparison run, the project output agreed with REGENIE on the
merged 50,000-variant subset for `beta`, `se`, `chisq`, and `log10p`. The
limited runtime comparison was essentially tied in this run: REGENIE took 13.31
seconds for its baseline invocation, while the project CPU run took 13.16
seconds for the 50,000-variant project invocation.

The binary path is substantially implemented but still less production-proven
than the quantitative path. Score-test kernels, approximate Firth fallback,
diagnostic codes, and scalar/full Firth helper code exist and have broad local
unit coverage. Exact Firth without `--approx` and SPA are still intentionally
unsupported. GPU binary smoke evidence was not collected successfully because
the environment did not expose a CUDA-enabled JAX backend.

CI health is currently good for the checked-out commit: `ci-lint`,
`ci-typecheck`, `ci-test`, and GitHub PR CI all pass. The previous
high-priority API import-boundary failure is fixed in this checkout: the focused
public API import tests pass, and `g.execution_plan` now lazy-loads binary
kernel config types with `importlib`.

The main production-readiness risks are now GPU environment reproducibility,
binary parity breadth, multi-phenotype performance semantics, resume/finalization
failure-injection coverage, and benchmark reproducibility. These are tractable
engineering risks, not signs that the architecture is hollow.

## Architecture Map

### Public API, CLI, and Configuration

The public Python entry point is `g.api`. It exposes a small `RegenieApi` facade
and delegates option normalization to `g.interface.config` and execution to
`g.runner`. This keeps the user-facing API intentionally narrow.

The CLI surface lives in `g.cli`; the Python/TOML/CLI normalization layer lives
under `g.interface`; shared enums and lightweight immutable settings live in
`g.types`. `g.execution_plan` translates a validated `RegenieConfig` into
immutable execution plans: kernel settings, output settings, per-phenotype run
plans, genotype source configuration, binary correction settings, and optional
stage-timing paths.

The current import boundary is in much better shape than earlier review notes.
`g.execution_plan` imports `g.compute.regenie2_binary.types` only under
`typing.TYPE_CHECKING` for annotations and uses `importlib.import_module()` only
when a binary kernel config must actually be built. The focused API tests confirm
that importing `g.api` no longer imports JAX-heavy modules.

### Runtime Orchestration

`g.runner` is the process-level orchestration layer. It configures logging,
native runtime knobs, JAX runtime policy, output directories, metadata, execution
plans, stage timing, profile summaries, and run telemetry sessions.

The runner uses lazy imports for heavy runtime modules:
`g.engine.regenie2_pipeline`, `g.engine.timing`, `g.engine.telemetry`, and
`g.jax_setup`. That boundary matters because JAX platform selection and backend
initialization are process-global. The intended model is now clear: public API,
configuration, and planning remain light; JAX and native execution are entered
through the runner.

The JAX runtime policy records device, cache directory, matmul precision,
persistent cache settings, autotune cache behavior, and transfer guard settings.
The first run fixes those process-global choices, and incompatible later runs in
the same Python process are rejected.

### Engine and End-to-End Data Flow

The hot path is a hybrid streaming pipeline:

1. User input arrives through CLI flags, TOML config, or Python options.
2. `g.interface.config` validates and normalizes it into `RegenieConfig`.
3. `g.execution_plan` builds a `RegenieExecutionPlan` with genotype source,
   trait, kernel, output, binary correction, and per-phenotype run settings.
4. `g.runner` initializes logging, telemetry, native runtime settings, JAX
   policy, effective config output, and timing/profile recorders.
5. `g.engine.regenie2_pipeline` opens the native Rust engine and aligns samples,
   phenotypes, covariates, and REGENIE LOCO prediction files.
6. Rust opens the BGEN input and streams decoded dosage chunks plus variant
   metadata to Python callbacks.
7. Python callback code converts native chunk buffers into JAX arrays and calls
   quantitative or binary kernels.
8. The callback packages association results and enqueues them to the native
   output writer.
9. Rust writer threads persist grouped Arrow chunk files, commit progress into
   the manifest, and optionally finalize `final.parquet`.
10. Resume mode validates manifest fingerprints, execution-plan hashes, schema
    metadata, and committed chunks before skipping or recomputing work.

This split is coherent. Rust owns BGEN decoding, sample alignment, writer
coordination, manifest commits, and finalization. Python/JAX owns the
matrix-heavy association statistics and accelerator placement.

The active optimized input path is variant-major buffered BGEN delivery.
Sample-major support still exists, but the pipeline defaults toward
variant-major chunks. Binary score-only work can stay variant-major; binary
fallback correction still has more layout-dependent complexity.

### Compute Kernels

Quantitative step 2 code lives under `g.compute.regenie2_linear`. It implements
covariate projection, null residual structure, score statistics, effect sizes,
standard errors, chi-square statistics, and log10 p-values in a chunkable
variant-major shape.

Binary step 2 code lives under `g.compute.regenie2_binary`. The package now has
a clearer internal split: score testing, result construction, logistic helpers,
state preparation, and Firth code under `firth`. Score-only binary output and
approximate Firth fallback are active paths; exact Firth and SPA are rejected
rather than silently approximated.

Multi-phenotype behavior is explicit. The default is per-phenotype execution,
which preserves each trait's own sample mask. The optimized shared native
multi-phenotype path requires explicit `complete-case` mode and uses the shared
intersection of valid samples. That is correctness-preserving as a default, but
the fastest shared-decode strategy is not yet available for many traits with
different missingness patterns.

### Native Extension and Output Runtime

The PyO3 extension is registered from `src/lib.rs` through the Rust `python`
module tree. Native responsibilities include BGEN decode, trusted/no-missing
fast paths, SIMD decode selection controls, chunk planning, sample alignment,
prediction-source handling, output sessions, writer coordination, grouped Arrow
chunk writing, manifest updates, and final Parquet construction.

The output layer has a strong production shape: manifest schema versions, output
schema versions, input fingerprints, execution-plan hashes, writer settings,
binary configuration, JAX policy, atomic manifest writes, committed chunk lists,
interrupted-run state, and strict resume validation. The next step is not to
invent a resume model; it is to test it under hostile interruptions.

## File Structure by Responsibility

Public/configuration code is concentrated in `src/g/api.py`, `src/g/cli.py`,
`src/g/interface`, `src/g/execution_plan.py`, `src/g/types.py`, and
`src/g/config.default.toml`.

Runtime orchestration is in `src/g/runner.py`, `src/g/jax_setup.py`, and
`src/g/engine`. The engine package includes pipeline entry points, native
dispatch, callbacks, timing, and telemetry.

JAX association kernels are split by domain:
`src/g/compute/regenie2_linear` for quantitative step 2,
`src/g/compute/regenie2_binary` for binary score/Firth work, and
`src/g/compute/common` for shared helpers.

Native Rust code sits directly under `src`: BGEN/genotype code in
`src/genotype`, Python bindings under `src/python`, sample alignment in
`src/sample.rs`, REGENIE prediction support in `src/regenie.rs`, output writing
and finalization under `src/output`, and pipeline/runtime glue under
`src/pipeline`.

Tests live in `tests` and cover API behavior, CLI/config normalization,
interface contracts, JAX setup, IO/output, pipeline behavior, numerical kernels,
binary correction contracts, scripts, Rust architecture, telemetry, timing, and
warm-cache behavior. Scripts in `scripts` cover data preparation, runtime
probes, benchmarks, and development diagnostics.

`archive/direct_association` should be treated as historical/reference code, not
as part of the shipped runtime. Existing review docs such as `docs/code-review.md`
remain useful history, but some findings are now stale because the compute
package structure, import boundary, binary configuration flow, telemetry, and
multi-phenotype defaults have changed.

## Evidence Collected

### GitHub and Local Test Health

Recent GitHub Actions state:

- Reviewed commit `c4e45c39`: PR CI `26322236070`, completed successfully.
- Previous observed commit `879aca53`: PR CI was cancelled.
- Earlier commit `845bae00`: PR CI succeeded.
- Intermediate commits `fef5c759`, `c536cc3d`, `04e5b53b`, and `eb7fcb09`: PR
  CI failed.

Focused API import-boundary tests now pass:

```text
uv run --no-sync pytest \
  tests/test_api.py::test_importing_api_does_not_import_jax_heavy_modules \
  tests/test_api.py::test_quantitative_kernel_config_does_not_import_binary_runtime \
  -q

2 passed in 0.72s
```

Project CI-style checks:

```text
just ci-lint       passed
just ci-typecheck  passed
just ci-test       passed: 380 passed, 1 skipped, 5 deselected
```

Rust workspace tests were run on `cantor`:

```text
srun --nodelist=cantor --cpus-per-task=16 --mem=64G --time=01:00:00 \
  --chdir=/mnt/beegfs/kirill/Projects/g cargo test --workspace --locked
```

Rust unit tests and native coverage tests passed:

- `src/lib.rs`: 47 passed.
- `tests/rust_native_coverage.rs`: 12 passed.

The workspace test command still failed overall because
`tests/rust_python_bindings.rs` could not load `libpython3.14.so.1.0` on
`cantor`. That is a worker-node linker/runtime environment problem, not a Rust
assertion failure.

### Benchmark Evidence

`just doctor-server` passed after allowing `just` to create its temporary recipe
directory outside the default sandbox. Required local BGEN, sample, phenotype,
covariate, and baseline prediction files already existed under `data`, so data
generation was not rerun on the login node.

Quantitative REGENIE comparison, CPU only:

```text
uv run --no-sync python scripts/benchmark_regenie_comparison.py \
  --cpu-only \
  --only-quantitative-step2 \
  --variant-limit 50000 \
  --output-dir data/benchmarks/current_state/regenie_comparison_cpu_qt_step2_c4e45c39
```

Results:

- REGENIE quantitative step 2: success, 13.3139 seconds, 418,943 output rows.
- Project quantitative CPU step 2: success, 13.1648 seconds, 50,000 output rows.
- Runtime ratio: 1.0113x, with the project run 0.1492 seconds faster in this
  limited comparison.
- Merged comparison variants: 50,000.
- Numeric agreement: `beta`, `se`, `chisq`, and `log10p` all passed allclose.
- Maximum absolute errors on the merged subset: `beta` 3.50e-05, `se` 1.14e-05,
  `chisq` 1.07e-03, `log10p` 2.59e-04.

The row-count difference matters: REGENIE produced its full baseline output,
while the project invocation was variant-limited. The correctness comparison was
on the merged 50,000-variant subset.

BGEN reader benchmark, CPU worker node:

```text
uv run --no-sync python scripts/benchmark_bgen_reader.py \
  --variant-limit 16384 \
  --repeat-count 5 \
  --path-modes variant_major_buffered \
  --chunk-sizes 2048,8192 \
  --trusted-no-missing-diploid
```

Results:

- Variant-major buffered, chunk size 2048: mean 3.3419 seconds.
- Variant-major buffered, chunk size 8192: mean 3.3597 seconds.
- Both cases reported checksum `7854816.0`.

Output-stage benchmark, CPU worker node:

```text
uv run --no-sync python scripts/benchmark_output_stages.py \
  --device cpu \
  --variant-limit 5000 \
  --trials 1 \
  --writer-thread-counts 1,4 \
  --writer-queue-depth-multipliers 1 \
  --chunks-per-arrow-file-values 4,16 \
  --arrow-compressions zstd,none \
  --output-dir data/benchmarks/current_state/output_stages_cpu_c4e45c39
```

Results from 64 case summaries:

- Fastest case: Arrow chunks, single phenotype, batch size 1024, one writer,
  queue depth 1, 16 chunks per file, no compression: 0.5299 seconds.
- Slowest case: Arrow chunks, single phenotype, batch size 1024, one writer,
  queue depth 1, four chunks per file, zstd: 7.8065 seconds.
- Arrow-only single phenotype range: 0.5299 to 7.8065 seconds.
- Arrow-only eight phenotype range: 4.0296 to 4.9204 seconds.
- Final Parquet single phenotype range: 0.5415 to 0.6561 seconds.
- Final Parquet eight phenotype range: 4.1848 to 4.8261 seconds.

This is a single-trial smoke matrix, not a final performance model. The slowest
single-phenotype Arrow case looks like a configuration/cold-run outlier and
should not be overinterpreted without repeated trials. The broad signal is that
phenotype count dominates output cost, and writer/compression settings matter.

### GPU Evidence

GPU commands were run through the project SLURM recipe on `landau`, not on the
login node.

The JAX runtime probe completed but found only CPU devices:

```text
just slurm-gpu-run uv run --no-sync python scripts/probe_jax_runtime.py
```

The default and GPU-driver-path probes reported:

```text
{"default_backend": "cpu", "devices": ["cpu:0"]}
```

They also warned that a GPU may be present, but a CUDA-enabled `jaxlib` is not
installed.

The binary GPU smoke benchmark did not run:

```text
just slurm-gpu-run uv run --no-sync python scripts/benchmark_regenie2_binary_hot.py \
  --device gpu \
  --variant-limit 1000 \
  --no-include-cold-process \
  --no-include-finalized-hot \
  --output-dir data/benchmarks/current_state/binary_hot_gpu_smoke_c4e45c39
```

It failed during JAX runtime initialization because backend `cuda` was unknown;
the known backends were `cpu` and `tpu`. The environment needs the project GPU
dependency group installed and validated before GPU benchmark results can be
used as evidence.

## Main Engineering Risks

The public API import boundary is currently fixed and tested, but it should stay
on the risk register because this project has many legitimate reasons to import
JAX-heavy compute code. The practical rule should remain: public API,
configuration, planning, and metadata modules are JAX-free unless execution has
started.

CI health is green for the reviewed commit, both locally and in GitHub Actions.
Several preceding main-branch runs failed, so the repo should still keep the
current API-boundary and CI checks prominent until this stays stable across a few
more commits.

GPU reproducibility is the most concrete environment risk. The project has GPU
commands and GPU runtime checks, but the active environment only exposes CPU JAX.
Benchmark reports should not accept GPU numbers unless they record CUDA backend
availability before the run.

Binary maturity is real but incomplete. Approximate Firth fallback and score
testing are implemented and tested, but binary parity needs broader evidence
across separation, low MAC, missingness, multiple phenotypes, fallback
thresholds, and GPU execution. Exact Firth and SPA should stay documented as
unsupported until they are implemented and compared.

Multi-phenotype semantics are safe by default but not yet optimal. Per-phenotype
execution preserves missingness semantics. Complete-case shared execution is
explicit and faster in principle, but it changes the sample set. A production
many-trait path probably needs shared BGEN decode with per-trait masks or
gathers.

Output and resume machinery are strong architecturally but need hostile testing.
Manifest fingerprints, schema versions, execution-plan hashes, committed chunk
tracking, and strict resume are good foundations. Confidence now depends on
failure-injection tests around Arrow writes, manifest commits, writer
backpressure, process interruption, and final Parquet generation.

Benchmark reproducibility needs tightening. CPU evidence is available, but the
output-stage matrix was single-trial and GPU evidence was blocked by dependency
state. The current head improves trusted BGEN SIMD benchmarking controls, but
benchmark reports should still record dependency groups, JAX backend state,
commit SHA, node name, and whether data/baselines were regenerated or reused.

Documentation drift remains visible. `docs/code-review.md` and older architecture
notes are useful history, but several findings have changed. Current-state
documents should be dated and tied to commits so readers do not treat stale
findings as live issues.

## Remediation Roadmap

### Architecture

1. Keep `g.api`, `g.interface`, `g.execution_plan`, and metadata modules JAX-free
   with explicit import-boundary tests.
2. Keep lightweight public enums and dataclasses in `g.types` or other JAX-free
   modules when planning code needs them.
3. Preserve `g.runner` as the runtime boundary for JAX setup, native execution,
   stage timing, and telemetry.
4. Document which modules are allowed to import JAX and native pipeline code.

### Correctness

1. Expand quantitative parity coverage beyond the current representative subset,
   including missingness, multiple chromosomes, resume, and finalization paths.
2. Expand binary parity tests around low MAC, separation, missingness, fallback
   thresholds, multiple traits, and complete-case versus per-phenotype masks.
3. Add failure-injection tests for output interruption around Arrow creation,
   manifest commit, writer shutdown, and final Parquet generation.
4. Keep unsupported exact Firth and SPA behavior explicit in CLI/API docs,
   errors, and output metadata.

### Performance

1. Split benchmark timing into decode, callback conversion, JAX compute, enqueue,
   Arrow write, manifest commit, and Parquet finalization stages.
2. Add repeated-trial output-stage benchmarks before making writer/compression
   decisions from outlier cases.
3. Build a reproducible GPU benchmark preflight that fails before benchmarking if
   CUDA JAX is unavailable.
4. Investigate shared-decode multi-phenotype execution that preserves per-trait
   missingness semantics.

### Developer Workflow

1. Keep main release-clean after the successful GitHub CI run for `c4e45c39`.
2. Fix the Rust Python-binding test environment on worker nodes so
   `libpython3.14.so.1.0` is discoverable during `cargo test`.
3. Document CPU-only, native-test, and GPU dependency groups separately.
4. Keep review documents dated and cross-linked so older findings are treated as
   historical context rather than current behavior.
