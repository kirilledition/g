# Agent Learning

This document consolidates task-oriented planning notes that used to be spread
across multiple markdown files. It is intended to preserve useful engineering
context without keeping stale task lists in the docs tree.

## Audit Result

The June 2026 task-doc audit found that most historical plans were already
implemented:

- Configuration defaults now come from a packaged TOML file, and the active
  `rust-cli-config-pyo3` branch moves CLI/config ownership into Rust.
- Native grouped alignment, prediction-source grouping, telemetry stream
  ownership, trusted packed8 decode improvements, output streaming, and setup
  reuse work have landed.
- Non-custom-kernel compute optimizations have landed: production timing no
  longer synchronizes by default, public result statistics narrow on device,
  warm-cache coverage reaches production entrypoints, score kernels stack shared
  genotype products, chunk stats are bundled through PyO3, Firth candidate
  dispatch has tiny/small tiers, and sparse approximate Firth can use compact
  carrier lanes.
- Firth hot-path capacity selection uses fixed-shape tiny/small/bounded tiers,
  multi-binary approximate Firth is batched over flattened trait-variant lanes,
  and the benchmark harness can sweep trait counts, storage modes, Firth batch
  sizes, candidate capacities, and fallback densities.

The standalone task-plan documents were removed after their durable lessons were
merged here. Current implementation work should be tracked in Linear, not as
free-floating task sections in docs.

## Generated Follow-Ups

The audit created these bounded Linear follow-ups:

- [GLA-23](https://linear.app/glaphyra/issue/GLA-23/profile-packed8-score-custom-kernel-gate): profile the packed8 score custom-kernel gate before any Pallas/CUDA prototype.
- [GLA-24](https://linear.app/glaphyra/issue/GLA-24/add-regenie-compatible-step-2-text-output): add typed REGENIE Step 2-compatible text output.
- [GLA-25](https://linear.app/glaphyra/issue/GLA-25/add-bounded-trace-mode-telemetry-event-caps): added bounded trace-mode telemetry event caps.
- [GLA-26](https://linear.app/glaphyra/issue/GLA-26/persist-binary-benchmark-diagnostics-in-summaries): persist richer binary benchmark diagnostics in summary JSON.

## Configuration

The live configuration contract is documented in
[Configuration Frontend](../development/configuration-frontend.md). Historical
config rewrite notes reduce to these rules for the Rust frontend branch:

- `crates/interface/src/config.default.toml` owns user-tunable defaults.
- Rust owns CLI parsing, TOML decoding, config layering, default loading,
  validation, and effective TOML serialization; root PyO3 config classes are adapters.
- `crates/plan/src/` owns deterministic host policy payloads plus requested and
  prepared run contracts. `crates/interface/src/plan_request.rs` compiles resolved
  config into `RunRequest`; manifest headers now serialize through a Rust-built
  `PreparedRunPlan` consumed by `g-output`. `g-plan` owns backend planning
  validation, the prepared-plan input builder, and prepared manifest backend-kind
  derivation from the resolved genotype format; Python still supplies other
  transitional header fields from legacy execution dataclasses until dynamic
  preparation moves to Rust.
- `crates/input/src/` owns native sample, phenotype, covariate, prediction-list,
  and LOCO prediction alignment; genotype readers keep BGEN-embedded sample
  retrieval and pass identifier views into input code.
- `crates/runtime/src/` now owns pure runtime policy/state helpers: logging
  runtime policy, telemetry path/counter policy, telemetry session
  cap/counter/envelope state, run lifecycle event payload/rendering policy,
  shutdown signal metadata and controller state, stage timing state, run
  metadata payloads, profile summary payloads, and the logging/Rayon/JAX process
  runtime state handle. Root PyO3 adapters still own side effects and
  Python-only JAX setup until runtime handles move fully into Rust.
- `crates/engine/src/` now owns the first native coordinator scaffold:
  `RunPhase`, the `AssociationBackend` trait, typed batch/prediction/group
  views, a deterministic fake backend, single-batch phase progression,
  injected failure handling for every entered phase, backend trait-method
  failure handling, interruption handling, a Criterion coordinator-overhead
  benchmark, the BGEN-backed `Regenie2RunEngineCore`/chunk planning core that
  used to live under `src/pipeline/`, native required-chromosome resolution,
  native preflight report/warning/scan-count helpers, and native
  committed-chunk intersection for multi-output resume scheduling. Native
  callback batch-size delivery policy, grouped-union callback batch-size
  policy, callback queue-limit policy, variant-major dosage batch handoff
  planning, result in-flight slot accounting, dosage-buffer pool accounting,
  dosage-buffer reuse shape planning, writer-finish thread cleanup policy,
  callback worker lifecycle start state, callback worker shutdown timeout
  policy, callback worker stop poll-timeout policy, callback worker
  stop-attempt decision policy, and BGEN delivery method selection also live in
  `g-engine`; the same crate also owns the callback worker backpressure
  poll-timeout policy, binary correction summary counter accumulation, and
  callback progress/chunk identity state. Native multi-trait committed-chunk
  write selection is also owned by `g-engine`.
  Python only extracts callback/input object attributes, acquires semaphores,
  allocates/slices/owns NumPy buffers, owns thread objects and join calls,
  invokes writer sessions, owns summary payload emission, and calls the
  selected PyO3 engine method. Python still owns NumPy array
  finite/rank/binary-shape validation while those array contracts are being
  migrated. Production queues, output writer lifecycle, cleanup, telemetry
  emission, and the PyO3/JAX association backend remain later migration work.
- `crates/interface/src/partial.rs` defines the typed partial TOML surface with Serde,
  optional fields, aliases, and unknown-key rejection.
- `crates/interface/src/overlay.rs` decodes provenance and overlays defaults, user
  TOML, and CLI/Python overrides before resolving complete config data.
- Runtime subsystems should receive resolved `RegenieConfig` or
  `ExecutionPlan` values, not raw CLI dictionaries, environment variables, or
  packaged default views.
- Config tests should continue guarding against reintroducing configurable
  `DEFAULT_*` constants outside the default catalog path.

## Native Boundaries

The active Rust frontend ownership split is:

```text
Rust:
  CLI parsing
  TOML decoding and default overlay
  config validation and effective TOML serialization
  PyO3 config objects
  BGEN decode/preprocessing
  sample, covariate, phenotype, and prediction alignment primitives
  output writing/finalization
  prepared-plan manifest header construction and manifest/resume validation
  low-overhead telemetry streams

Python:
  public API shim
  execution planning
  JAX runtime setup
  high-level orchestration

JAX:
  multi-phenotype kernels
  binary Firth and approximate Firth
  dense linear algebra where batching is large enough
```

Production association paths should stay variant-major. Row-major BGEN decode
paths are useful for parity/reference tests, but production BGEN dispatch uses
variant-major dosage or packed8 delivery.

Rust optimization work improved local native paths, but app-level GPU hot
benchmarks were effectively flat on the measured workloads because JAX,
worker, and output orchestration dominated. Future native performance work
should start with an app-level bottleneck signal rather than isolated native
microbenchmarks alone.

## Binary And Firth

Binary approximate Firth remains the most performance-sensitive scientific
surface. Durable constraints from the completed work:

- Keep the rare full-chunk overflow path in a separate executable. A 2026-06-09
  GLA-42 attempt to fold bounded-vs-overflow routing into the common JAX
  dispatch removed one scalar candidate-count synchronization, but regressed
  chr22 binary headline performance because the common executable then carried
  the full-chunk overflow branch.
- Capacity selection inside the common executable should stay fixed-shape across
  tiny, small, and bounded tiers.
- Multi-binary capacity is based on flattened trait-variant lanes.
- Single-trait and multi-trait correction paths must preserve the same overflow
  semantics and failure labels.
- Sparse masks are modifiers for already-selected Firth lanes, not candidate
  selectors.
- Larger Firth batch/capacity sweeps did not justify changing the current
  defaults: `firth_batch_size = 1024` and `firth_candidate_capacity = 2048`.
- Binary null logistic non-convergence fails by default; warning mode is an
  explicit opt-in.
- Invalid binary score statistics should emit NaN public statistics, not
  misleading zero chi-square or p-value fields.

Exact Firth without `--approx` and SPA remain unsupported public features.
Those are scientific parity projects and should be scoped as explicit Linear
issues before implementation.

## Custom Kernels

Custom kernels are still possible, but they need profiling proof first. The
current order of attack is:

1. Profile packed8 score-only runs to see whether packed8 decode plus score
   reductions are memory-traffic limited.
2. Prefer a Pallas prototype before CUDA FFI because it stays inside JAX
   tracing and shape specialization.
3. Keep the current JAX implementation as a runtime fallback.
4. Add parity tests before making any benchmark claims.
5. Move to Firth kernels only after traces show solver reductions, compact
   sparse carrier work, or candidate preparation dominates high-Firth runs.

Stop custom-kernel work if it improves microbenchmarks but not chr10/chr22 hot
runs after warmup, requires wider numerical tolerances than current packed8
versus variant-major tests, or makes CPU/non-CUDA test runs fragile.

## Output

Arrow chunks and Parquet run outputs remain the fast path. Public association
statistics are currently written as float32. A future float64 public output
schema would need schema/versioning, writer, manifest, resume, and native
dispatch changes, not just a compute dtype option.

REGENIE-compatible text output is still a useful compatibility gap. It should
be implemented as a typed writer/finalizer, not as Python table formatting.
That work is tracked in GLA-24.

## Telemetry And Benchmarking

Production timing should not force JAX synchronization by default. Exact stage
timings are diagnostic and should be requested deliberately when the user needs
synchronized attribution.

Trace mode is also diagnostic and can perturb performance. It has a bounded
event-cap policy so accidental high-volume tracing fails clearly or drops
events only under the documented lossy policy.

Binary hot benchmark summaries should persist enough diagnostic information to
interpret Firth performance across runs: score candidates, Firth candidates,
correction branch counts, failure/correction codes, and the relevant stage
timing paths. That work is tracked in GLA-26.

## Validation Anchors

Useful commands from the completed campaigns:

```bash
uv run pytest tests/test_interface.py tests/test_cli.py -q
uv run pytest tests/test_regenie2_binary.py tests/test_regenie2_pipeline.py -q
uv run pytest tests/test_regenie2_linear.py tests/test_regenie2_binary.py tests/test_timing.py tests/test_telemetry.py tests/test_warm_cache.py tests/test_regenie2_pipeline.py -q
uv run pytest tests/test_regenie_comparison_scripts.py -q -k binary_hot
uv run ty check src/g tests
uv run ruff check src tests
. scripts/server_env.sh && cargo test --lib --quiet
```

GPU work must run through SLURM on `landau`. Use production timing defaults for
throughput and exact stage attribution only when investigating timings.
