# App src optimization review findings

Review date: 2026-06-07
Reviewed branch: `main`
Reviewed commit: `58264565`

This is a read-only review of `src/` focused on correctness bugs that can
pollute performance results, CPU/GPU throughput limits, avoidable allocations,
Python/Rust/JAX transfer overhead, and larger rewrites that could move the
engine closer to HPC-scale throughput. No code was changed in this pass.

Existing documents such as `docs/rust-optimization-progress.md`,
`docs/rust-optimization-opportunities.md`, and
`docs/compute-optimization-opportunities.md` already cover several completed
optimizations. This document focuses on issues still visible in the current
source.

## Severity guide

- P1: correctness or result-integrity risk, or a performance issue large enough
  to block reliable benchmarking.
- P2: likely material speedup or scalability improvement for real runs.
- P3: narrower optimization, cleanup, or feature-parity issue worth tracking
  after the P1/P2 work.

## Executive summary

The largest remaining performance levers are not isolated casts. They are
pipeline-level:

1. Preflight currently exports too much metadata from Rust to Python and repeats
   the work for every trait in multi-phenotype runs.
2. The hot loop still crosses Rust -> Python -> JAX -> Python -> Rust once per
   chunk, with per-chunk device puts, device gets, metadata object creation, and
   writer calls.
3. The packed8 binary score-only path is fused enough to avoid a host-side
   float dosage matrix, but the approximate-Firth path still materializes dense
   decoded dosage before correction.
4. Approximate-Firth and null-Firth fallback code eagerly evaluates fallback
   attempts that are often unused.
5. Output manifest/finalization edge cases can include stale chunks or ignore
   conflicting duplicate commits. These should be fixed before treating resume
   or finalization benchmarks as trustworthy.

## Findings

### F-001 - Final Parquet finalization scans the directory instead of the manifest

Priority: P1 correctness / benchmarking integrity

References:

- `src/output/finalization.rs:87`
- `src/output/finalization.rs:114`
- `src/output/finalization.rs:151`
- `src/output/finalization.rs:179`

Observation:

`finalize_regenie_step2_parquet` builds the final file from
`sorted_output_chunk_file_paths(chunks_directory, output_format)`. That helper
uses `read_dir` and includes every matching `chunk_*.arrow` or `part_*.parquet`
file in the directory. It does not restrict the read set to manifest
`committed_chunks`. After writing those files into `final.parquet`, it marks the
manifest finalized.

Impact:

A stale chunk left in the run directory can be included in the final output even
if it is not in the manifest for the current run. That can produce duplicated or
wrong rows and can make resume/finalization timings look better or worse for the
wrong reason.

Implementation direction:

Finalize from validated manifest commits, not from directory globbing. Either
fail on unmanifested matching files or quarantine them before finalization. The
manifest should provide the ordered file list and expected row counts.

Validation:

Add a regression that creates a valid manifest plus an extra matching chunk file
and asserts finalization rejects the run or excludes the extra file.

### F-002 - Duplicate manifest chunk commits are silently ignored

Priority: P1 correctness / resume integrity

References:

- `src/output/manifest.rs:37`
- `src/output/manifest.rs:41`
- `src/output/manifest.rs:43`

Observation:

`append_run_manifest_chunk_commits` deduplicates existing commits only by
`chunk_identifier`. A duplicate identifier with different range, file name,
compression, format, or row count is ignored instead of rejected.

Impact:

If a writer retry or resume repair produces conflicting metadata for the same
chunk identifier, the manifest can keep the old entry while the chunk directory
contains the new file. That creates a silent mismatch between resume state and
physical output.

Implementation direction:

When an existing chunk identifier is seen, compare the entire commit object.
Allow exact idempotent duplicates, but raise on conflicting metadata.

Validation:

Add unit tests for exact duplicate append, conflicting duplicate append, and
append ordering.

### F-003 - Python API boolean coercion turns `"false"` into `true`

Priority: P1 correctness / configuration integrity

References:

- `src/g/interface/config_layers.py:284`
- `src/g/interface/config_layers.py:296`

Observation:

For boolean options, `coerce_option_value` returns `bool(option_value)`. In the
Python API or raw option dictionaries, non-empty strings such as `"false"`,
`"0"`, and `"no"` become `True`.

Impact:

This can silently enable expensive modes such as Firth, approximate Firth,
telemetry, trusted validation, output finalization, or JAX transfer guard when a
caller intended to disable them. It can also corrupt benchmark comparisons.

Implementation direction:

Accept native `bool` values directly. For strings, parse a strict vocabulary
such as `true/false`, `1/0`, `yes/no`, `on/off`, and reject ambiguous values.

Validation:

Add config-layer tests that pass Python booleans and string booleans through the
public API path.

### F-004 - Scalar approximate-Firth line search can move beta after no accepted step

Priority: P1 numerical correctness

References:

- `src/g/compute/regenie2_binary/firth/scalar_approx.py:352`
- `src/g/compute/regenie2_binary/firth/scalar_approx.py:369`
- `src/g/compute/regenie2_binary/firth/scalar_approx.py:461`
- `src/g/compute/regenie2_binary/firth/scalar_approx.py:476`

Observation:

`run_scalar_line_search` stops when the attempt count is exhausted, when a step
is accepted, or when the state becomes invalid. If all attempted components stay
valid but no step is accepted, `line_search_state.accepted` is false and
`line_search_state.valid` can still be true. The caller then computes
`accepted_step_size = line_search_state.step_size + 1.0e-6` and updates beta.

Impact:

This can advance the Newton state using an explicitly unaccepted step. It risks
non-monotonic approximate-Firth updates and can mark some variants as valid when
the line search should have failed or held the previous beta.

Implementation direction:

Treat exhausted non-accepted line search as failure, or preserve the previous
beta when `accepted` is false. The failure condition should include
`~line_search_state.accepted` unless convergence has already been reached.

Validation:

Add a small deterministic scalar-Firth case where step-halving exhausts without
acceptance and assert the result is failed or unchanged.

### F-005 - Relative LOCO paths are resolved against process cwd, not the list file

Priority: P1 correctness / reproducibility

References:

- `src/regenie.rs:371`
- `src/regenie.rs:392`
- `src/regenie.rs:404`

Observation:

`parse_prediction_list_file` stores the second field as `PathBuf::from(...)`.
Relative LOCO file paths are later opened directly. That makes them relative to
the current process working directory, not to the prediction-list file location.

Impact:

The same prediction list can work or fail depending on where the command is
launched. Worse, it can load the wrong same-named relative file from another
directory.

Implementation direction:

Resolve relative LOCO paths against `prediction_list_path.parent()`. Canonicalize
cache keys after resolution so repeated entries share cache state.

Validation:

Add a prediction-list test with a relative LOCO path from a different process
cwd.

### F-006 - Preflight exports all variant metadata just to discover chromosomes

Priority: P2 startup and Python/Rust transfer

References:

- `src/g/engine/preflight.py:57`
- `src/g/engine/preflight.py:122`
- `src/g/engine/preflight.py:132`
- `src/python/mod.rs:628`
- `src/genotype/bgen/metadata.rs:15`

Observation:

`collect_required_chromosomes` calls `engine.variant_metadata_slice(0,
scanned_variant_count)` and unpacks all five metadata columns while using only
chromosome values. The Rust side builds cloned vectors for chromosome, variant
identifier, position, counted allele, and reference allele.

Impact:

For a full chromosome-scale run, this creates millions of string clones and
large Python objects before any compute begins. It is pure startup overhead and
is repeated in multi-trait preflight.

Implementation direction:

Expose a native `required_chromosomes(variant_limit)` method based on chromosome
boundary indices and boundary labels. It should return one chromosome label per
represented chromosome without per-variant metadata export.

Validation:

Benchmark preflight on chr10 before/after and assert the same chromosome set for
full and limited variant ranges.

### F-007 - Multi-trait preflight repeats shared checks for every trait

Priority: P2 startup and multi-phenotype scalability

References:

- `src/g/engine/regenie2_pipeline.py:1238`
- `src/g/engine/regenie2_pipeline.py:1294`
- `src/g/engine/regenie2_pipeline.py:1303`
- `src/g/engine/preflight.py:91`
- `src/g/engine/preflight.py:122`

Observation:

`run_multi_preflight` loops over every trait and calls the single-trait
preflight path. This repeats chromosome collection, covariate shape/rank checks,
and prediction validation for each trait.

Impact:

The cost scales with trait count even when most data are shared. For large
phenotype batches this can become a visible startup tax and can also trigger
repeated host/device copies.

Implementation direction:

Split preflight into shared checks and per-trait checks. Validate covariates and
required chromosomes once per group. Validate the prediction matrix per
chromosome in one batched operation.

Validation:

Add a multi-phenotype benchmark with 1, 10, and 100 traits and report preflight
time separately.

### F-008 - Preflight pulls JAX arrays back to NumPy

Priority: P2 startup and synchronization

References:

- `src/g/engine/native_dispatch.py:110`
- `src/g/engine/native_dispatch.py:113`
- `src/g/engine/native_dispatch.py:114`
- `src/g/engine/preflight.py:47`
- `src/g/engine/preflight.py:48`

Observation:

`build_native_bgen_run_input` immediately wraps native phenotype and covariate
data with `jnp.asarray`. Preflight then calls `np.asarray` on those arrays.
Depending on device placement, that can force device-to-host transfers and
synchronization before compute.

Impact:

This fights the intended JAX data path and makes preflight timing sensitive to
device placement and JAX async dispatch state.

Implementation direction:

Run finite/rank/shape validation on host arrays before `jnp.asarray`, or expose
host validation directly from Rust aligned sample data. Only put arrays on
device after preflight succeeds.

Validation:

Enable JAX transfer guard in tests for preflight paths and assert no accidental
device-to-host copy occurs.

### F-009 - The hot loop still crosses Rust/Python/JAX per chunk

Priority: P2 throughput ceiling

References:

- `src/python/mod.rs:872`
- `src/python/mod.rs:875`
- `src/python/mod.rs:902`
- `src/python/mod.rs:914`
- `src/python/mod.rs:923`
- `src/python/mod.rs:931`
- `src/python/mod.rs:943`
- `src/python/mod.rs:971`
- `src/python/mod.rs:985`
- `src/python/mod.rs:994`
- `src/g/engine/callbacks.py:238`
- `src/g/engine/callbacks.py:359`
- `src/g/engine/callbacks.py:371`

Observation:

For every chunk, Rust asks Python for a buffer, Rust fills it, Rust builds
metadata, Python puts data on device, JAX computes, Python materializes the
result, and Python calls back into Rust writer code. Buffer reuse helps, but the
control plane still pays per-chunk Python overhead and per-chunk transfers.

Impact:

As kernel time improves, this boundary overhead becomes a fixed throughput
ceiling. It also limits opportunities for overlapping decode, transfer, compute,
and output.

Implementation direction:

Consider a native-driven execution controller with a ring of preallocated
buffers, explicit prefetch, and batched callback submission. Longer term, move
more orchestration into Rust and call a small number of stable JAX entrypoints
per chromosome or chunk batch.

Validation:

Add per-chunk counters for Python callback time, host-to-device transfer time,
device compute time, device-to-host materialization, and writer time. Report
the percentage of wall time outside JAX kernels.

### F-010 - Chunk chromosome lookup clones the whole chromosome column

Priority: P2 allocation and Python/Rust transfer

References:

- `src/g/engine/callbacks.py:456`
- `src/python/mod.rs:348`
- `src/genotype/planner.rs:14`

Observation:

`get_metadata_chromosome` reads `metadata.chromosome[0]`. The Python getter for
`metadata.chromosome` clones the entire `Vec<String>` from Rust before Python
takes the first element. Chunks are planned to be chromosome-homogeneous.

Impact:

Every chunk pays a full chromosome-column clone for a scalar value that is
needed for chromosome-state transitions and progress.

Implementation direction:

Expose `metadata.first_chromosome`, `metadata.chromosome_label`, or include the
chunk chromosome label in `ChunkSpec`. Keep the full column getter for output
only.

Validation:

Add an allocation/profile assertion around a multi-chunk run and verify the
chromosome getter no longer clones per-variant strings.

### F-011 - Packed8 approximate-Firth path materializes dense decoded dosage

Priority: P2 GPU memory bandwidth and Firth throughput

References:

- `src/g/compute/regenie2_binary/api.py:201`
- `src/g/compute/regenie2_binary/api.py:216`
- `src/g/compute/regenie2_binary/api.py:342`
- `src/g/compute/regenie2_binary/api.py:352`
- `src/g/compute/regenie2_binary/api.py:418`
- `src/g/compute/regenie2_binary/api.py:428`
- `src/g/compute/common/genotype.py:50`

Observation:

The score-only packed8 binary path decodes probability pairs inside a jitted
donating entrypoint and immediately computes score statistics. When correction
is enabled, the code decodes packed8 probabilities to a dense variant-major
dosage matrix and then calls the canonical variant-major correction path.

Impact:

Approximate-Firth runs lose much of the packed8 memory advantage. Dense dosage
materialization increases device memory pressure and bandwidth, especially when
only a small candidate subset needs correction.

Implementation direction:

Add a packed8 correction entrypoint that keeps score-only work fused and decodes
only candidate rows for Firth. For sparse Firth, decode/gather compact carrier
slots directly from packed8 bytes. A custom JAX/Pallas or XLA custom-call kernel
may be justified if JAX fusion does not remove the dense intermediate.

Validation:

Profile peak device memory and HLO for packed8 score-only vs packed8 Firth.
Track correction-candidate count and decoded-row count.

### F-012 - Approximate-Firth fallback attempts are evaluated eagerly

Priority: P2 GPU compute

References:

- `src/g/compute/regenie2_binary/firth/scalar_approx.py:681`
- `src/g/compute/regenie2_binary/firth/scalar_approx.py:695`
- `src/g/compute/regenie2_binary/firth/scalar_approx.py:698`
- `src/g/compute/regenie2_binary/firth/scalar_approx.py:712`
- `src/g/compute/regenie2_binary/firth/scalar_approx.py:713`
- `src/g/compute/regenie2_binary/firth/scalar_approx.py:727`

Observation:

The scalar approximate-Firth path computes pseudo-Firth, zero-start Newton, and
warm-start Newton before selecting a result. The `run_zero_start` and
`run_warm_start` masks are computed, but the expensive fallback calls have
already run.

Impact:

For variants where pseudo-Firth succeeds, the kernel still pays for fallback
Newton paths. That is especially expensive in rare-variant correction-heavy
runs.

Implementation direction:

Use a `jax.lax.cond` cascade so zero-start Newton is evaluated only when the
pseudo result fails, and warm-start Newton only when pseudo and zero-start do
not produce a valid result. If lane-level divergence defeats this inside a vmap,
consider grouping candidate lanes by fallback stage.

Validation:

Record pseudo/zero/warm attempt counts and compare kernel runtime on synthetic
data where pseudo success is common vs rare.

### F-013 - Null Firth fallback attempts are also evaluated eagerly

Priority: P2 GPU compute

References:

- `src/g/compute/regenie2_binary/firth/null.py:276`
- `src/g/compute/regenie2_binary/firth/null.py:288`
- `src/g/compute/regenie2_binary/firth/null.py:306`
- `src/g/compute/regenie2_binary/firth/null.py:318`
- `src/g/compute/regenie2_binary/firth/null.py:330`

Observation:

`fit_covariate_only_firth_null_model` always executes four fits before choosing
the first converged result.

Impact:

Null model preparation can be several times more expensive than necessary when
the first or second attempt usually converges.

Implementation direction:

Convert the selection into staged `jax.lax.cond` calls. Run the second fit only
if the first fails, then the longer fallback fits only if needed.

Validation:

Profile null-model compile/runtime on binary traits with easy and hard
convergence cases.

### F-014 - Linear residual variance lacks the binary path's stability floor

Priority: P2 numerical correctness and tail performance

References:

- `src/g/compute/regenie2_linear/score.py:92`
- `src/g/compute/regenie2_linear/score.py:97`
- `src/g/compute/regenie2_linear/score.py:98`
- `src/g/compute/regenie2_binary/score.py:17`
- `src/g/compute/regenie2_binary/score.py:33`

Observation:

The linear score path clamps residual genotype sum of squares at zero and then
uses a `> 0.0` mask. The binary path has an explicit absolute/relative variance
floor.

Impact:

Near-collinear variants can produce tiny positive residual variances that pass
the mask and inflate beta, standard error, or test statistics. These tails are
scientifically important and can also create slow debug cycles when results
diverge from REGENIE.

Implementation direction:

Add a linear numerical policy with absolute and relative variance thresholds,
ideally reusing the same concepts as the binary score path.

Validation:

Add tests for near-constant and covariate-collinear genotype vectors.

### F-015 - Linear shifted sum-of-squares can subtract large float32 terms

Priority: P2 numerical stability

References:

- `src/g/compute/regenie2_linear/score.py:14`
- `src/g/compute/regenie2_linear/score.py:24`
- `src/g/compute/regenie2_linear/score.py:30`
- `src/g/compute/regenie2_linear/score.py:31`

Observation:

For high-frequency variants, normalized genotype sum of squares is computed
from native stats as `sum(x^2) - 2 * offset * sum(x) + n * offset^2`. In
float32, variants close to the high-frequency boundary can subtract large
similar values.

Impact:

Cancellation can create inaccurate small variances and interact with F-014.

Implementation direction:

Carry complementary native stats for high-frequency shifted coding, such as
`sum(2 - dosage)` and `sum((2 - dosage)^2)`, or compute this statistic in
float64/compensated arithmetic only for boundary-sensitive variants.

Validation:

Compare float32 and float64 sum-of-squares for mostly-homozygous alternate
variants and assert bounded relative error.

### F-016 - Writer sessions are created before preflight succeeds

Priority: P2 correctness hygiene and wasted startup work

References:

- `src/g/engine/regenie2_pipeline.py:587`
- `src/g/engine/regenie2_pipeline.py:594`
- `src/g/engine/regenie2_pipeline.py:599`
- `src/g/engine/regenie2_pipeline.py:605`
- `src/g/engine/regenie2_pipeline.py:1224`
- `src/g/engine/regenie2_pipeline.py:1232`
- `src/g/engine/regenie2_pipeline.py:1240`

Observation:

Single-trait and multi-trait pipelines initialize output runs and create writer
sessions before preflight validation completes.

Impact:

Invalid inputs can still create or mutate run directories and manifests. This
adds cleanup noise and can interfere with resume/finalization benchmarks.

Implementation direction:

Run preflight before writer-session creation where possible. For checks that
need committed chunk sets, split manifest validation from writer thread startup.

Validation:

Add tests that preflight failures leave no writer session and no new manifest
commit.

### F-017 - Multi-phenotype groups may re-read/decompress BGEN for each group

Priority: P2 multi-trait throughput

References:

- `src/g/engine/regenie2_pipeline.py:1168`
- `src/g/engine/regenie2_pipeline.py:1172`
- `src/g/engine/regenie2_pipeline.py:1176`
- `src/g/engine/regenie2_pipeline.py:1197`
- `src/g/engine/regenie2_pipeline.py:1282`
- `src/g/engine/regenie2_pipeline.py:1283`

Observation:

`run_prepared_multi_phenotype_bgen_pipeline` loops over compatible phenotype
groups and calls `run_prepared_multi_phenotype_bgen_group`. Each group then
drives the BGEN engine through its own chunk loop.

Impact:

If phenotypes are split into several complete-case groups, the same genotype
file can be reread and decompressed once per group. That can dominate runtime
for many traits.

Implementation direction:

Explore one decode pass that fans out to multiple sample-selection groups, or a
grouping strategy that pads/masks sample sets to reduce repeated BGEN scans.
This is a larger architectural rewrite, but it is one of the biggest
multi-trait levers.

Validation:

Benchmark multi-trait runs with intentionally fragmented missingness patterns
and report BGEN decode time per group.

### F-018 - Multi-trait output materializes all traits before resume filtering

Priority: P2 device-to-host transfer

References:

- `src/g/engine/callbacks.py:399`
- `src/g/engine/callbacks.py:400`
- `src/g/engine/callbacks.py:411`
- `src/g/engine/callbacks.py:413`
- `src/g/engine/callbacks.py:419`

Observation:

`write_regenie2_multi_native_chunk_with_optional_timing` materializes the full
trait-by-variant result dictionary with `jax.device_get` before computing
`active_trait_indices` from committed chunk sets.

Impact:

During resume, traits that already have the chunk still contribute to
device-to-host transfer and host memory pressure.

Implementation direction:

Compute active trait indices before materialization and slice result arrays on
device before `jax.device_get`. For heavily resumed runs, group active traits so
the transfer shape matches the write set.

Validation:

Create a resume benchmark where only one of many traits is missing a chunk and
measure device-to-host bytes.

### F-019 - Writer threads scale by trait count

Priority: P2 scalability

References:

- `src/g/config.default.toml:66`
- `src/g/engine/regenie2_pipeline.py:344`
- `src/g/engine/regenie2_pipeline.py:352`
- `src/g/engine/regenie2_pipeline.py:1232`

Observation:

The default writer thread count is four per writer session, and the multi-trait
pipeline creates one writer session per phenotype output.

Impact:

A 100-trait run can request hundreds of writer threads, independent of disk
bandwidth or CPU core availability. Oversubscription can slow compute and make
benchmark results noisy.

Implementation direction:

Use a shared writer pool or impose a global writer-thread cap across sessions.
Keep per-trait queues if needed, but schedule actual writes through a bounded
pool.

Validation:

Benchmark multi-trait output on local scratch and network filesystem with 10,
50, and 100 traits while varying the global writer cap.

### F-020 - Parquet finalization rewrites already-written Parquet parts

Priority: P2 optional output overhead

References:

- `src/output/writer.rs:487`
- `src/output/writer.rs:520`
- `src/output/finalization.rs:114`
- `src/output/finalization.rs:130`
- `src/output/finalization.rs:136`
- `src/g/config.default.toml:73`

Observation:

The writer can already emit Parquet part files. When `finalize-parquet` is
enabled, finalization reads those parts back, projects batches, and writes a new
single `final.parquet`.

Impact:

This doubles output I/O for finalized runs. The default is currently false, so
this is opt-in overhead, but it matters for users who want a single file.

Implementation direction:

Prefer treating the Parquet parts directory as the final dataset. If a single
file is required, add a single-final-writer mode rather than rewriting parts at
the end.

Validation:

Report output wall time and bytes read/written with and without finalization on
chr10.

### F-021 - Sample indices round-trip through Python and are copied back into Rust

Priority: P2/P3 boundary overhead

References:

- `src/g/engine/native_dispatch.py:112`
- `src/python/mod.rs:653`
- `src/python/mod.rs:657`
- `src/python/mod.rs:661`

Observation:

Native aligned sample data exposes `sample_indices` to Python as a NumPy array.
When BGEN delivery starts, PyO3 reads that NumPy array and copies it back into a
Rust `Vec`.

Impact:

This is startup overhead proportional to sample count and another example of
state bouncing across the Python boundary.

Implementation direction:

Let the native engine keep a prepared sample-selection handle from the aligned
sample data, or pass an opaque native aligned-sample object to delivery instead
of a NumPy copy.

Validation:

Track allocations and startup time for large sample counts.

### F-022 - Multi-run input clones sample identifiers into Python but runtime does not use them

Priority: P2/P3 allocation

References:

- `src/g/engine/native_dispatch.py:123`
- `src/g/engine/native_dispatch.py:127`
- `src/g/engine/native_dispatch.py:128`

Observation:

`build_native_bgen_multi_run_input` clones family and individual identifiers into
Python tuples. A search of `src/g` shows these fields are not used outside the
dataclass definition and construction path.

Impact:

For large cohorts, this creates many Python string objects with no current
runtime benefit.

Implementation direction:

Keep identifiers in the native aligned-sample handle. Expose them lazily for
debug/test/public-inspection paths only.

Validation:

Measure Python allocation count when loading a large multi-phenotype input.

### F-023 - Output path suffix handling can drop user suffixes

Priority: P2/P3 user-visible correctness

References:

- `src/g/io/output.py:66`
- `src/g/io/output.py:72`

Observation:

`resolve_output_run_paths` uses `output_root.with_suffix(f".{association_mode}.run")`.
For an output root such as `/tmp/run.v1`, this produces
`/tmp/run.regenie2_linear.run`, dropping `.v1`.

Impact:

Users can accidentally write to unexpected directories. It can also collide
with outputs from roots that differ only by suffix.

Implementation direction:

If the desired behavior is appending, use `Path(str(output_root) +
f".{association_mode}.run")`. If replacing is intentional, document it and add
tests.

Validation:

Add path derivation tests for roots with and without suffixes.

### F-024 - Process-global runtime state can misreport effective thread count

Priority: P2/P3 benchmarking integrity

References:

- `src/g/runner.py:266`
- `src/g/runner.py:271`
- `src/g/runner.py:721`
- `src/g/runner.py:724`

Observation:

The Rayon global thread pool is configured once per process. Later runs with a
different requested `--threads` value log a warning and continue. The run
manifest records the requested configured thread count, not necessarily the
effective Rayon thread count.

Impact:

Back-to-back benchmarks in one Python process can silently run with a different
thread count than the manifest says.

Implementation direction:

Reject incompatible repeated requests in benchmarking paths, or record both
requested and effective thread counts. Consider a process-level guard that makes
thread-count changes explicit.

Validation:

Add a test that runs two plans with different thread counts in one process and
asserts the second manifest cannot misstate the effective runtime.

### F-025 - Row-major selected-sample BGEN decode still scans every file sample

Priority: P3 SIMD / CPU decode opportunity

References:

- `src/genotype/bgen/decode.rs:881`
- `src/genotype/bgen/decode.rs:883`
- `src/genotype/bgen/decode.rs:887`

Observation:

In the `all_samples_present` selected-sample path, the decoder iterates over
every file sample, decodes the probability pair, looks up the selected index,
and only writes when selected.

Impact:

This path is lower priority than the production variant-major buffered paths,
but it is inefficient when a small subset of samples is selected. It also leaves
SIMD/vectorization potential on the table.

Implementation direction:

Add specialized paths for identity, contiguous selected range, dense selected
mask, and sparse selected-index lists. For all-present non-missing data, consider
SIMD decoding of packed probability pairs into f32 dosages and gather/scatter
only selected lanes.

Validation:

Microbenchmark row-major APIs with 1%, 10%, 50%, and 100% sample selection.

### F-026 - Strict resume validates the same grouped output file repeatedly

Priority: P3 resume startup

References:

- `src/output/resume.rs:26`
- `src/output/resume.rs:33`
- `src/output/resume.rs:41`
- `src/output/resume.rs:80`
- `src/output/resume.rs:89`

Observation:

Strict resume iterates manifest chunk commits and inspects each referenced file.
For grouped output where one physical Arrow/Parquet part contains multiple
chunk commits, the same file can be opened and scanned repeatedly.

Impact:

Strict resume startup scales with commit count rather than file count.

Implementation direction:

Group manifest commits by `chunk_file_name`, inspect each physical file once,
and validate all contained commits from the file metadata/batches in that pass.

Validation:

Benchmark strict resume validation with `chunks-per-arrow-file` set to 1, 16,
and 64.

### F-027 - LOCO prediction matrices are copied even for identity alignment

Priority: P3 allocation / chromosome-state preparation

References:

- `src/regenie.rs:269`
- `src/regenie.rs:272`
- `src/regenie.rs:295`
- `src/regenie.rs:356`
- `src/regenie.rs:357`
- `src/regenie.rs:358`

Observation:

LOCO predictions are aligned into owned `Vec<f32>` values per chromosome and
trait. Even identity alignment returns `prediction_values.to_vec()`.
`chromosome_prediction_matrix` then creates another `Vec` and extends it with
each trait's prediction values.

Impact:

Chromosome-state preparation allocates and copies prediction data that may
already be in the correct order. This is usually smaller than genotype data, but
it scales with sample count, chromosome count, and trait count.

Implementation direction:

Use borrowed/shared buffers for identity alignment, preallocate the matrix with
`trait_count * sample_count`, and cache per-chromosome matrices when they are
requested repeatedly.

Validation:

Profile allocations during prediction loading and first-chromosome state
preparation.

### F-028 - Warm-cache coverage is shape-limited and not a full pipeline warmup

Priority: P3 benchmarking UX

References:

- `src/g/engine/warm_cache.py:54`
- `src/g/engine/warm_cache.py:62`
- `src/g/engine/warm_cache.py:75`

Observation:

`build_warm_cache_shapes` warms at most two unique variant counts, typically the
full chunk and one tail shape. This is useful, but it is still separate from the
normal run path and does not guarantee every production shape/configuration is
compiled before timing.

Impact:

Benchmark comparisons can still include compile time for unexpected shapes,
trait counts, correction modes, or grouped multi-phenotype layouts.

Implementation direction:

Make warmup part of the benchmark/run harness: enumerate the exact chunk shapes,
trait-group shapes, genotype format, correction plan, and score dtype that will
be timed. Record which compiled signatures were warmed.

Validation:

Run the same benchmark twice with persistent compilation cache enabled and
assert the timed run has no JAX compile events.

### F-029 - Exact Firth and SPA remain unsupported feature-parity gaps

Priority: P3 scientific parity

References:

- `src/g/execution_plan.py:146`
- `src/g/execution_plan.py:158`
- `src/g/interface/config.py:831`
- `src/g/interface/config.py:834`
- `src/g/compute/regenie2_binary/correction.py:26`
- `src/g/compute/regenie2_binary/correction.py:28`
- `tests/test_regenie_binary_correction_contract.py:44`
- `tests/test_regenie_binary_correction_contract.py:55`

Observation:

The public config accepts REGENIE-style binary options, but exact Firth and SPA
fallbacks are rejected. This is not a raw speed issue, but it affects scientific
coverage and parity comparisons against original REGENIE.

Impact:

Users needing exact Firth or SPA cannot use this engine for those result sets,
and benchmark comparisons must clearly state that the binary correction set is
not identical.

Implementation direction:

Keep approximate-Firth optimization first if that is the target workload. Track
exact Firth and SPA as separate parity projects with explicit acceptance tests
against REGENIE.

Validation:

Add documented benchmark/result matrices that state which correction modes are
implemented for each comparison.

### F-030 - Async writer buffers can borrow mutable NumPy memory

Priority: P1/P2 output API safety

References:

- `src/python/output.rs:45`
- `src/python/output.rs:51`
- `src/python/output.rs:64`
- `src/python/output.rs:69`

Observation:

The Python output bridge builds Arrow buffers from NumPy result arrays with a
custom allocation that keeps the Python array alive. That prevents use-after-free
while the writer thread is still using the memory, but it does not make the
source array immutable.

Impact:

The internal pipeline passes fresh materialized arrays that are not mutated
after enqueue, so the normal path is probably safe. The lower-level writer API,
however, can enqueue an Arrow buffer backed by a mutable NumPy array. If a caller
mutates the array before the writer consumes it, output can change after the
write call returns.

Implementation direction:

Either copy arrays at enqueue, transfer exclusive ownership of immutable buffers,
or make the public writer API synchronous for borrowed arrays. For the internal
hot path, a no-copy path is still desirable, but it should be backed by buffers
that cannot be mutated from Python after enqueue.

Validation:

Add a stress test that calls the writer directly, mutates the source NumPy array
after enqueue, and verifies the output is stable.

### F-031 - IID-mode LOCO alignment skips empty identifiers during uniqueness checks

Priority: P1 correctness

References:

- `src/regenie.rs:532`
- `src/regenie.rs:537`
- `src/regenie.rs:549`
- `src/regenie.rs:552`
- `src/regenie.rs:584`
- `src/regenie.rs:588`
- `src/regenie.rs:596`

Observation:

IID-mode uniqueness validation ignores empty target and LOCO individual
identifiers. The actual IID lookup then inserts every LOCO IID, including empty
strings, and aligns every target IID, including empty strings.

Impact:

Duplicate empty IIDs can pass validation and then align to whichever empty LOCO
entry was inserted last. This can silently attach the wrong LOCO prediction to
samples with missing IIDs.

Implementation direction:

Reject empty IIDs in IID mode, or treat empty IIDs as missing and fail alignment
with an explicit message.

Validation:

Add IID-mode LOCO alignment tests for empty target IIDs, empty LOCO IIDs, and
duplicate empty IIDs.

### F-032 - Multi-binary Firth chunk execution is not wrapped as one jitted entrypoint

Priority: P2 dispatch/fusion

References:

- `src/g/compute/regenie2_binary/api.py:292`
- `src/g/compute/regenie2_binary/api.py:310`
- `src/g/compute/regenie2_binary/api.py:319`
- `src/g/compute/regenie2_binary/variant_major_correction.py:455`
- `src/g/compute/regenie2_binary/variant_major_correction.py:468`

Observation:

The multi-trait binary chunk function for variant-major inputs is a plain Python
function. It calls the score helper and then the correction helper. The fixed
capacity correction dispatchers are jitted, but the full score-plus-correction
chunk is not a single jitted function.

Impact:

This can add Python dispatch between score and correction and can reduce XLA's
ability to fuse or schedule shared work across the full chunk path.

Implementation direction:

Add a jitted full multi-binary chunk entrypoint for the Firth path, analogous to
the score-only donating wrappers. Keep static args explicit for correction plan,
kernel config, score dtype, and capacity policy.

Validation:

Compare HLO/module count and per-chunk dispatch count for multi-binary Firth
before and after.

### F-033 - Tiered Firth dispatch can still bloat compile time

Priority: P2 compile time / kernel size

References:

- `src/g/compute/regenie2_binary/variant_major_correction.py:135`
- `src/g/compute/regenie2_binary/variant_major_correction.py:166`
- `src/g/compute/regenie2_binary/variant_major_correction.py:195`
- `src/g/compute/regenie2_binary/variant_major_correction.py:209`
- `src/g/compute/regenie2_binary/variant_major_correction.py:223`
- `src/g/compute/regenie2_binary/variant_major_correction.py:248`

Observation:

The tiered candidate dispatch is a good runtime shape-control strategy, but the
jitted dispatcher still contains tiny, small, bounded, and overflow branches.
XLA generally compiles both branches of `lax.cond` into the program.

Impact:

Compile time and HLO size can remain high even if most chunks take only the tiny
or small path. Overflow capacity in particular can make the executable much
larger than the common case needs.

Implementation direction:

Split rare overflow handling into a separate executable or a separate pass.
Consider separate entrypoints for common tiny/small capacities so the main path
does not compile the largest branch.

Validation:

Record compile time, executable size if available, and HLO operation count for
different capacity settings and fallback-count distributions.

### F-034 - Firth full null deviance is recomputed per chunk

Priority: P2 GPU compute

References:

- `src/g/compute/regenie2_binary/firth/batch.py:254`
- `src/g/compute/regenie2_binary/firth/batch.py:256`
- `src/g/compute/regenie2_binary/firth/batch.py:257`
- `src/g/compute/regenie2_binary/firth/batch.py:375`
- `src/g/compute/regenie2_binary/firth/batch.py:385`

Observation:

Candidate batch preparation recomputes full null deviance from chromosome-state
phenotype and offset data. Those values are invariant for a chromosome and
trait, but the code runs inside per-chunk Firth preparation.

Impact:

Correction-heavy runs pay repeated logistic probability/deviance work that
could be computed once during chromosome-state preparation.

Implementation direction:

Store scalar and multi-trait full null deviance in the chromosome state and pass
those arrays into Firth batch preparation.

Validation:

Compare per-chunk Firth preparation time before/after on a chunk set with many
candidate corrections.

### F-035 - Stacked projection matrices are rebuilt per chunk

Priority: P2/P3 allocation and JAX graph size

References:

- `src/g/compute/regenie2_linear/score.py:80`
- `src/g/compute/regenie2_linear/score.py:81`
- `src/g/compute/regenie2_binary/score.py:117`
- `src/g/compute/regenie2_binary/score.py:121`

Observation:

The optimized score paths fuse projection products by stacking matrices, but
the stacked matrices themselves are assembled inside the per-chunk score
functions.

Impact:

This adds repeated concatenation work and keeps more constant-like setup in the
chunk executable.

Implementation direction:

Precompute the stacked left/right hand matrices during chromosome-state
preparation and store them in state. Keep chunk functions focused on genotype
normalization and matrix multiply.

Validation:

Inspect HLO before/after and benchmark small chunks where setup overhead is
more visible.

### F-036 - Trusted BGEN validation is serial and trusted decode still scans ploidy bytes

Priority: P2/P3 CPU decode

References:

- `src/genotype/bgen/reader.rs:296`
- `src/genotype/bgen/reader.rs:299`
- `src/genotype/bgen/trusted.rs:139`
- `src/genotype/bgen/trusted.rs:141`

Observation:

Trusted no-missing diploid validation loops over variants serially. Even after
validation, trusted decode still reads and scans per-sample ploidy/missingness
bytes to confirm all samples are present diploid.

Impact:

Validation startup and trusted decode retain CPU work proportional to
variant-count times sample-count. If a validation cache already proves the
property for the file, repeated per-sample checks during decode are redundant.

Implementation direction:

Parallelize initial validation. After a cache hit or successful validation,
trusted decode should advance the cursor over ploidy/missingness bytes without
rescanning them, while preserving debug/assert validation modes.

Validation:

Benchmark trusted validation and trusted decode separately on chr10. Add a test
that validation cache hits skip the expensive validation pass.

### F-037 - Variant-major decode allocates stats buffers and result vectors per chunk

Priority: P3 allocation / CPU decode

References:

- `src/genotype/bgen/reader.rs:470`
- `src/genotype/bgen/reader.rs:478`
- `src/genotype/bgen/reader.rs:628`

Observation:

Each variant-major dosage read allocates new `VariantMajorStatsBuffers` and
collects per-tile decode results into a `Vec`.

Impact:

For small chunks or many chunks, allocator traffic can show up after the decode
kernel itself is optimized.

Implementation direction:

Reuse stats buffers per prepared reader/session where possible. Reduce tile
profiles directly into a shared accumulator instead of collecting an
intermediate result vector.

Validation:

Use allocator profiling on many small chunks and compare allocations/chunk.

### F-038 - Multi-writer finish/finalization is serial

Priority: P3 output latency

References:

- `src/g/engine/native_dispatch.py:408`
- `src/g/engine/native_dispatch.py:416`
- `src/g/engine/native_dispatch.py:417`
- `src/g/engine/native_dispatch.py:418`

Observation:

`finish_writer_sessions` loops through writer sessions and calls
`writer_session.finish()` one at a time.

Impact:

For multi-trait runs, writer draining and optional finalization cannot overlap
across trait outputs.

Implementation direction:

If finalization remains per-trait, finish sessions concurrently with a bounded
pool. A shared writer pool from F-019 may make this unnecessary.

Validation:

Benchmark final drain/finalization time for 10+ trait outputs.

### F-039 - EXTRA string array construction allocates per chunk

Priority: P3 output allocation

References:

- `src/output/schema.rs:13`
- `src/output/schema.rs:24`
- `src/output/schema.rs:39`

Observation:

When an `extra_code` array is provided, `build_extra_string_array` allocates a
`Vec<Option<&str>>` and builds a new `StringArray` for every chunk.

Impact:

Most chunks likely have null/no-extra output. Per-row string-array work is
avoidable on the common path.

Implementation direction:

Fast-path all-null or all-zero `extra_code` to `new_null_array`, and consider a
dictionary-encoded or cached representation for rare `TEST_FAIL` rows.

Validation:

Profile output batch construction with score-only and Firth runs.

### F-040 - Chunk stats duplicate dosage sum as allele count

Priority: P3 memory

References:

- `src/genotype/preprocess.rs:391`
- `src/genotype/preprocess.rs:395`
- `src/genotype/preprocess.rs:400`

Observation:

`ChunkStats` stores `dosage_sum` and `allele_count` with identical values by
cloning the dosage-sum vector.

Impact:

This doubles memory for one per-variant stats column and copies it once per
chunk.

Implementation direction:

If the public API still needs both field names, derive one from the other at the
binding/API boundary or store shared backing data internally.

Validation:

Confirm no downstream writer or Python code mutates either field independently,
then add a regression for both public names.

### F-041 - Output run initialization reloads a manifest already loaded by preparation

Priority: P3 startup I/O

References:

- `src/g/io/output.py:404`
- `src/g/io/output.py:415`
- `src/g/io/output.py:444`
- `src/g/io/output.py:461`

Observation:

`prepare_output_run` loads the manifest and returns it as `existing_manifest`.
`initialize_output_run` then calls `load_run_manifest` again and merges the
already loaded manifest state.

Impact:

This is minor I/O, but it also widens the race window between preparation and
initialization.

Implementation direction:

Use `existing_manifest` as the source of truth inside initialization unless an
explicit reload/lock step is needed.

Validation:

Add a unit test for resume initialization and remove the redundant read.

## Recommended implementation order

1. Fix P1 correctness issues first: manifest/finalization consistency, duplicate
   commits, boolean coercion, scalar line-search non-acceptance, relative LOCO
   paths, and IID-mode empty-sample handling.
2. Remove startup waste: native required-chromosome API, shared multi-preflight,
   host-side preflight before JAX device placement, and delayed writer session
   creation.
3. Reduce hot-loop boundary overhead: scalar chromosome metadata getter,
   precomputed native sample selection, and active-trait device slicing before
   host materialization.
4. Attack packed8 Firth: avoid dense decoded dosage for candidate correction and
   inspect HLO/peak memory.
5. Make Firth fallback execution lazy and benchmark by candidate/fallback counts.
6. Revisit larger architecture: one BGEN pass across phenotype groups, shared
   writer pool, and native-driven chunk batching.

## Benchmarking notes for follow-up

For each implemented optimization, report at least:

- Wall time split into preflight, decode, host-to-device transfer, JAX compute,
  device-to-host materialization, output write, and finalization.
- Number of chunks, variants, samples, traits, candidate corrections, and
  fallback attempts.
- JAX compile time separately from execution time.
- Effective Rayon thread count and writer thread count, not only requested
  values.
- Device memory peak for dosage vs packed8 vs packed8-Firth paths.

No benchmarks were run for this document. The findings are source-review
findings and should be validated with targeted microbenchmarks before large
rewrites.
