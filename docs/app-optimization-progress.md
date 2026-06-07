# App optimization progress

Last updated: 2026-06-07

This tracks the implementation campaign for
`docs/app-src-optimization-review-findings.md`.

## Execution defaults

- Strategy: staged waves, correctness before performance rewrites.
- Benchmark gate: targeted tests and microbenchmarks first, chr10 after the
  benchmark harness can separate compile, warm-cache, and execution time.
- API policy: public Python API can break when doing so materially removes
  transfers or clarifies Rust/JAX ownership; CLI behavior should remain stable
  except for bug fixes.
- Worktrees: use `/home/kirill/Projects/g-worktrees` for parallel work.
- Heavy work: use Slurm compute/GPU nodes, not the head node.

## Wave status

- Wave 0 campaign setup: complete.
- Wave 1 correctness and benchmark integrity: complete for F-001, F-002,
  F-003, F-004, F-005, F-023, F-024, F-026, F-031.
- Wave 2 native boundaries and startup overhead: partial; F-006, F-010,
  F-007, F-008, F-016, F-021, F-022, and F-041 are complete.
- Wave 3 JAX numerics and Firth performance: partial; F-013, F-014, and
  F-015 are complete.
- Wave 4 writer/output throughput: partial; F-018, F-019, F-020, F-030,
  F-038, and F-039 are complete.
- Wave 5 Rust decode and larger architecture: partial; F-017, F-025, F-036,
  F-037, and F-040 are integrated.
- Wave 6 warmup and performance proof: partial; F-009 and F-028 are
  integrated.

## Current implementation notes

- The existing untracked `scripts/run_chr10_benchmark_profile.py` is unrelated
  and must stay untouched unless explicitly requested.
- Implemented strict manifest-driven finalization, duplicate commit conflict
  checks, grouped-file strict resume validation, and repair-safe manifest
  commit recovery.
- Implemented strict Python boolean coercion while preserving specific
  recognized-unsupported option diagnostics.
- Implemented relative LOCO path resolution, canonical LOCO cache keys, and
  empty IID rejection for IID alignment.
- Implemented scalar approximate-Firth line-search exhaustion handling without
  accepting unaccepted beta updates.
- Implemented native required-chromosome discovery and scalar chunk chromosome
  lookup to avoid full metadata-column cloning.
- Implemented preflight-before-writer initialization and avoided reloading a
  manifest already returned by output preparation.
- Implemented active-trait device slicing before multi-trait output
  materialization and skip-all-committed behavior.
- Implemented public writer copy-on-enqueue safety and `EXTRA` all-null/success
  fast paths.
- Implemented shared native output writer pools keyed by the configured writer
  cap, so multi-trait runs keep per-trait queues/manifests while actual chunk
  file writes no longer allocate `trait_count * writer_thread_count` native
  worker threads.
- Implemented bounded concurrent multi-writer finish/finalization through the
  same writer cap, with PyO3 writer lifecycle calls releasing the GIL while
  Rust drains or finalizes output.
- Locked in Parquet parts as the default fast final dataset. The existing
  single `final.parquet` rewrite remains available only when
  `--g-finalize-parquet` is explicitly enabled.
- Implemented linear residual variance floors, selective high-frequency
  float64 shifted sum-of-squares, and production routing of those linear
  numerical settings.
- Implemented lazy null-Firth fallback execution with staged `jax.lax.cond`.
- Implemented row-major selected-sample BGEN all-present fast paths for
  identity, contiguous, sparse selected-index, and dense-mask selections.
- Implemented parallel trusted no-missing diploid validation and cache-hit
  validation bypass. Trusted decode now skips ploidy/missingness rescans only
  after the reader has been validated, with debug assertions preserving
  validation hooks.
- Implemented direct Rayon reduction of variant-major tile profile snapshots,
  removing the intermediate per-chunk tile result vector.
- Implemented shared backing storage for generated `dosage_sum` and
  `allele_count` chunk stats while preserving both field names at the Rust and
  Python binding boundary.
- In `appopt-hotloop-warmup`, added chunk-scoped stage timing records for
  native Rust-to-Python delivery callback time, Python callback worker time,
  host-to-device transfers, JAX compute dispatch/synchronization time,
  device-to-host materialization, and output write time. True Rust decode-only
  per-chunk timing still requires native per-chunk profile export; the Python
  branch records the observable delivery callback boundary.
- In `appopt-hotloop-warmup`, warm-cache enumeration now uses every unique
  production chunk shape in plan order instead of truncating to two sizes, and
  `WarmCacheReport` records warmed signatures with genotype format, trait
  count, correction method, Firth capacity settings, score dtype, and
  dosage/packed8 entrypoint path.
- Implemented F-017 union-sample per-phenotype group delivery for trusted
  no-missing dosage runs when overlapping sample groups make one BGEN pass
  cheaper than repeated per-group passes. The old per-group path remains the
  fallback for packed8, untrusted missingness, single groups, and disjoint
  groups where the union sample count is not cheaper.

## Validation

- `uv run pytest tests/test_interface.py tests/test_io_output.py tests/test_api.py tests/test_preflight.py tests/test_core.py tests/test_regenie2_pipeline.py tests/test_regenie2_linear.py tests/test_regenie2_binary_scalar_firth.py tests/test_regenie2_binary_firth_null.py -q`
  passed: 319 tests, 1 existing JAX donation warning.
- `. scripts/server_env.sh && cargo test --lib --quiet` passed: 101 tests.
- `. scripts/server_env.sh && cargo test --test rust_python_bindings --quiet`
  passed: 1 test.
- `uv run ruff check src tests` passed.
- Scoped `uv run ruff format --check ...` over modified Python files passed.
  Full-tree format is currently blocked by an unrelated unmodified compute
  file, so this wave avoided changing it while another engineer is working in
  that area.
- `uv run ty check src tests` passed.
- `cargo fmt --check` passed.
- `git diff --check` passed.

## Remaining larger work

- Pipeline architecture and batching: F-032, F-033.
- Packed8 and Firth compute rewrites: F-011, F-012, F-034.
- Score-kernel/state setup rewrites: F-035.
- Rust decode and allocation reuse: F-025, F-036, F-037, F-040 are implemented
  and integrated; deeper reusable stats-buffer ownership remains a future
  design item because returned chunk stats can outlive the next reader call.
- Scientific parity projects: F-029.

## Integration branch updates

- F-007: multi-trait preflight now validates shared covariates, required
  chromosomes, binary phenotype coding, and trait-major LOCO prediction
  matrices once per compatible phenotype group instead of once per trait.
- F-008: native BGEN run inputs now keep aligned phenotype and covariate arrays
  as host NumPy arrays through preflight; callbacks explicitly place them on
  the active JAX device during state preparation.
- F-021: native chunk delivery can use Rust-owned aligned sample handles, so the
  hot delivery call no longer has to pass Python `sample_indices` back into
  Rust. The Python `sample_indices` view remains for shape metadata and tests.
- F-022: multi-run family and individual identifiers are exposed lazily from the
  native aligned-sample object instead of being cloned into the runtime
  dataclass during construction.
- F-027: identity-aligned LOCO prediction vectors now reuse shared immutable
  buffers. Multi-trait chromosome prediction matrix construction preallocates
  its final matrix and caches assembled chromosome matrices for repeated
  requests.
- F-009: chunk-scoped stage timing now records native delivery callback,
  Python worker, host-to-device transfer, JAX compute dispatch/synchronization,
  device-to-host materialization, and output write time.
- F-028: warm-cache enumeration now covers every unique production chunk shape
  in plan order, and `WarmCacheReport` records warmed signatures with genotype
  format, trait count, correction method, Firth capacity settings, score dtype,
  and dosage/packed8 entrypoint path.
- F-017: grouped per-phenotype runs can decode an ordered union sample set once
  and fan out projected group buffers plus native chunk statistics to existing
  group callbacks. This path is selected only for trusted no-missing dosage
  delivery when `union_sample_count < sum(group_sample_count)`; otherwise the
  previous per-group delivery path is preserved.
