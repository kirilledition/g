The absolute best setup for `g` is:

```text
one logical telemetry system
one stable JSON event schema
Rust tracing for native hot paths
Python structured event helper for orchestration
non-blocking lossy writers by default
aggregate summaries for performance numbers
progress events throttled by time/chunk count
no logging that forces JAX synchronization in production mode
```

Your current implementation is a good foundation: Rust already initializes a `tracing_subscriber` with optional compact stderr and JSON file layers, uses non-blocking writers, and bridges Python logging through `pyo3-pylogger`.  The Python config currently exposes `g-stage-timings-json`, `g-log-filter`, `g-log-file`, and `g-log-stderr`.  The key remaining work is to turn this from “logging exists” into **run telemetry**.

---

# Best logging layout

For each run, write:

```text
<run_dir>/
  logs/
    events.jsonl
    progress.jsonl
    trace.jsonl                  # optional, only in trace mode
    profile.summary.json
  output_stage_timings.json       # current Rust writer timing file, eventually merge into profile.summary.json
  effective_config.toml
  run_manifest.json
```

I would keep **one logical telemetry schema**, but split physical streams by purpose:

| File                   | Purpose                                            |       Volume |
| ---------------------- | -------------------------------------------------- | -----------: |
| `events.jsonl`         | lifecycle, chunk, chromosome, writer, error events |   low/medium |
| `progress.jsonl`       | user progress, safe to `tail -f`                   |     very low |
| `trace.jsonl`          | deep native/JAX trace events                       | high, opt-in |
| `profile.summary.json` | aggregate timings/counters for benchmarks          |     one file |

Do not put everything into one giant file by default. It is convenient at first, but it makes progress checking, profile summaries, and trace-heavy debugging less ergonomic.

---

# Recommended config surface

Extend current diagnostics config from:

```toml
[g.diagnostics]
stage-timings-json = "..."
log-filter = "info"
log-file = "..."
log-stderr = true
```

to this:

```toml
[g.diagnostics]
telemetry = "progress"              # off | progress | profile | trace
log-dir = "results/bmi.g/logs"
log-filter = "g=info"
log-file = "results/bmi.g/logs/events.jsonl"
log-stderr = true

progress-interval-seconds = 5
progress-interval-chunks = 10

profile-summary-json = "results/bmi.g/logs/profile.summary.json"
stage-timings-json = "results/bmi.g/logs/stage-timings.json"

trace-file = "results/bmi.g/logs/trace.jsonl"
trace-filter = "g.native.bgen=trace,g.output=debug"

log-queue-size = 65536
log-lossy = true
include-source-location = false
include-span-events = false
```

CLI equivalents:

```bash
--g-telemetry progress|profile|trace|off
--g-log-dir PATH
--g-log-filter FILTER
--g-log-file PATH
--g-log-stderr / --no-g-log-stderr
--g-progress-interval-seconds 5
--g-progress-interval-chunks 10
--g-log-queue-size 65536
--g-log-lossy / --no-g-log-lossy
```

Current logging config is too small for profiling-grade work. Add these knobs before you rely on logs for performance work.

---

# Recommended modes

## 1. Production default

This is what normal users should get.

```toml
[g.diagnostics]
telemetry = "progress"
log-filter = "g=info"
log-stderr = true
log-lossy = true
progress-interval-seconds = 5
progress-interval-chunks = 10
stage-timings-json = ""
```

Events:

```text
run_started
config_resolved
execution_plan_prepared
bgen_engine_opened
sample_alignment_completed
prediction_source_loaded
preflight_completed
chromosome_started
progress_tick
chromosome_completed
writer_finished
run_completed
run_failed
```

No forced JAX synchronization. No per-stage blocking. No binary diagnostics device pulls.

This should be essentially invisible compared with BGEN decode/JAX/output time.

---

## 2. Benchmark profile mode

Use this when comparing to REGENIE or testing optimization PRs.

```toml
[g.diagnostics]
telemetry = "profile"
log-filter = "g=debug,g.native=debug,g.output=debug"
log-file = "results/run/logs/events.jsonl"
log-stderr = false
stage-timings-json = "results/run/logs/stage-timings.json"
profile-summary-json = "results/run/logs/profile.summary.json"
log-lossy = true
```

Events:

```text
per run
per chromosome
per chunk
per stage
per writer batch
native BGEN profile snapshot
binary candidate/Firth aggregate diagnostics
output finalization timing
```

This mode can perturb timings because your current timing path intentionally synchronizes JAX work. For example, `put_genotype_matrix_on_device()` only blocks on the transfer when a stage timing recorder exists, and compute blocking is also gated on the recorder.  The timing recorder is only created when a stage timing path is requested. 

That is good design, but it means:

```text
profile mode measures detailed synchronized stages;
production mode measures real throughput.
```

Use both when benchmarking.

---

## 3. Deep trace mode

Use only for small runs, one chromosome, or `--g-variant-limit`.

```toml
[g.diagnostics]
telemetry = "trace"
log-filter = "g=debug,g.native.bgen=trace,g.output=debug"
trace-file = "results/run/logs/trace.jsonl"
log-stderr = false
log-lossy = true
stage-timings-json = "results/run/logs/stage-timings.json"
```

Events:

```text
per decode tile
per native writer batch
per queue-blocking episode
per BGEN decompression/profile substage
per JAX compile/cache event if available
```

Do not use trace mode for full production-scale scans unless you are specifically diagnosing a performance bug.

---

# Granularity: what is safe?

## Safe in production

These are safe and useful:

```text
1 event per run lifecycle transition
1 event per chromosome start/end
1 progress event every 5 seconds or every 10 chunks
1 event per output writer finish/finalization
1 event per fatal/recoverable error
```

This is tiny volume. It should not affect compute performance.

## Safe in profile mode

These are usually fine:

```text
1–5 events per chunk
1 event per native BGEN delivery chunk
1 event per JAX compute chunk
1 event per D2H materialization chunk
1 event per writer batch / Arrow file
1 aggregate binary diagnostic per chunk
```

With `bsize=8192`, even 10 million variants is roughly 1,220 chunks. Five events per chunk is only about 6,100 events. That is nothing for a non-blocking JSONL writer.

The expensive part is not the number of JSON lines; it is whether the event requires:

```text
jax.device_get
block_until_ready
large string formatting
large arrays
Python callbacks from Rust hot loops
```

Avoid those in production.

## Acceptable only in trace mode

These can be okay for focused debugging:

```text
1 event per BGEN decode tile
1 event per writer queue flush
1 event per decompression block category
1 event per native profile snapshot interval
```

If tile size is 64 variants, then 10 million variants creates about 156,250 tile events. That is acceptable for a focused trace file, but not a default production log.

## Never log by default

Do **not** log:

```text
per variant
per sample
per genotype
per probability byte
per Firth iteration for every candidate
full sample IDs
phenotype values
covariate values
genotype arrays
large metadata arrays
```

Per-variant logging at biobank scale means millions to tens of millions of JSON objects. It will distort performance and create huge logs.

For Firth, log aggregate histograms per chunk:

```json
{
  "event": "firth_chunk_summary",
  "candidate_count": 122,
  "converged_count": 119,
  "failed_count": 3,
  "iteration_min": 4,
  "iteration_median": 11,
  "iteration_max": 41
}
```

not one event per candidate per iteration.

---

# The most important performance rule

The biggest logging foot-gun in this app is **not JSON serialization**. It is accidentally forcing device synchronization.

These are expensive:

```python
jax.device_get(...)
array.block_until_ready()
bool(jax_array)
int(jax_array)
np.asarray(jax_array)
```

Your code already reflects this: timing-mode calls block on H2D and JAX compute only when the stage timing recorder is enabled.  Binary chunk diagnostics also use `jax.device_get(...)`, but only when the stage timing recorder exists. 

Keep that rule:

```text
production logging:
  never device_get just for logs

profile logging:
  may device_get aggregates

trace logging:
  may synchronize, but must be opt-in and documented as perturbing
```

---

# Best event schema

Every event should have these fields:

```json
{
  "schema_version": 1,
  "run_id": "01HY...",
  "ts": "2026-05-22T10:12:30.123456Z",
  "level": "INFO",
  "source": "python|rust|jax",
  "target": "g.engine.callbacks",
  "event": "chunk_completed",
  "association_mode": "regenie2_linear",
  "trait_type": "quantitative",
  "phenotype": "BMI",
  "chromosome": "22",
  "chunk_identifier": 139264,
  "variant_start_index": 139264,
  "variant_stop_index": 147456,
  "variant_count": 8192,
  "sample_count": 408123,
  "duration_ms": 83.4
}
```

Important: use `event` as a stable machine-readable field. Do not rely on free-form messages.

Recommended common fields:

```text
schema_version
run_id
event
source
target
level
ts
pid
thread_name
association_mode
trait_type
phenotype
chromosome
chunk_identifier
variant_start_index
variant_stop_index
sample_count
variant_count
duration_ms
bytes
queue_depth
device
error
```

Never log raw biological data arrays.

---

# Current implementation issue: process-global logging

Your Rust logging initialization is process-global. If logging guards already exist, `initialize_logging()` sets up Python logging and returns `false`; it does not switch to a new log file. 

That is fine for CLI runs because one process usually means one run. It is less good for the Python API, where a user may call:

```python
g.regenie(config1)  # logs to run1/events.jsonl
g.regenie(config2)  # expects logs to run2/events.jsonl
```

Currently, the second run may still log to the first configured sink.

## Fix

Choose one policy:

### Simple policy

Make logging process-global and reject incompatible log configuration after first run.

```text
If first run configured log_file=A
and second run asks for log_file=B:
  raise RuntimeError("Logging is process-global; start a new process or reuse A.")
```

### Best policy

Initialize the process subscriber once, but make run telemetry a separate session:

```python
telemetry_session = _core.start_run_telemetry(
    run_id=run_id,
    event_file=...,
    progress_file=...,
    trace_file=...,
)
```

Then events carry `run_id`, and sinks are managed by the telemetry session rather than by reinitializing global tracing.

Given this app’s long-running Python API aspirations, I would implement the “best policy” eventually. For now, the simple policy is acceptable.

---

# Rust tracing setup

Your current Rust setup is already close:

```rust
tracing_subscriber::registry()
    .with(environment_filter)
    .with(stderr_layer)
    .with(file_layer)
```

with non-blocking writers and JSON output for file logs. 

I would change/add:

```text
1. Add configurable queue size.
2. Add configurable lossy/blocking mode.
3. Add source location only in trace mode.
4. Add span events only in trace mode.
5. Add run_id/event schema fields.
6. Add dropped-log counters if supported by the writer.
7. Add separate trace file layer for high-volume native events.
```

Recommended default:

```rust
lossy = true
queue_size = 65536
include_source_location = false
include_span_events = false
```

Use blocking mode only when debugging rare correctness failures where losing logs is unacceptable:

```toml
[g.diagnostics]
log-lossy = false
```

Do not use non-lossy/blocking logging in performance benchmarks unless you are measuring logging overhead.

---

# Python logging setup

I would not add a heavy Python logging stack. You already have Rust tracing and Python logging bridged through the native layer. Keep runtime dependencies small.

Add a small Python helper:

```python
def log_event(event: str, level: str = "info", **fields: object) -> None:
    ...
```

Usage:

```python
telemetry.log_event(
    "execution_plan_prepared",
    phenotype_count=len(plan.phenotype_run_plans),
    association_mode=plan.association_mode.value,
)
```

Do not rely only on `logger.info("Prepared plan for %s phenotypes", n)`. That is readable, but it is not enough for structured profiling.

For Python, the right event places are:

```text
runner:
  run_started
  config_validated
  execution_plan_prepared
  effective_config_written
  run_completed
  run_failed

pipeline:
  bgen_engine_opened
  sample_alignment_completed
  prediction_source_loaded
  preflight_completed
  native_delivery_started
  native_delivery_completed

callbacks:
  chromosome_state_prepared
  chunk_compute_submitted
  chunk_compute_completed
  result_materialized
  chunk_enqueued_for_write
  queue_blocked
```

---

# Native/Rust event places

Add Rust events at these places:

```text
BGEN:
  bgen_open_started/completed
  bgen_index_loaded
  chunk_planned
  native_chunk_decode_started/completed
  decode_tile_completed only in trace mode
  profile_snapshot

Sample:
  sample_alignment_started/completed
  phenotype_table_scanned
  covariate_table_scanned
  duplicate_key_rejected

Output:
  writer_session_started
  writer_batch_flushed
  arrow_chunk_written
  manifest_commits_recorded
  parquet_finalization_started/completed
  writer_session_finished
```

The current output writer already collects detailed stage timing internally and writes `output_stage_timings.json`.   Eventually, roll that into the same `profile.summary.json` path so users do not have to correlate multiple summary files manually.

---

# How granular can you log?

Here is the practical threshold table.

| Granularity                         | Production | Profile |              Trace | Notes                                         |
| ----------------------------------- | ---------: | ------: | -----------------: | --------------------------------------------- |
| Run lifecycle                       |        yes |     yes |                yes | Always safe                                   |
| Config/execution-plan hash          |        yes |     yes |                yes | No sensitive values beyond paths/fingerprints |
| Progress every 5s                   |        yes |     yes |                yes | Very cheap                                    |
| Chromosome start/end                |        yes |     yes |                yes | Safe                                          |
| Per chunk lifecycle                 |      maybe |     yes |                yes | Fine if no `device_get`                       |
| Per chunk timings with JAX blocking |         no |     yes |                yes | Perturbs performance                          |
| Per writer batch                    |      maybe |     yes |                yes | Good for output bottlenecks                   |
| Per BGEN decode tile                |         no |      no |                yes | Use only for focused native profiling         |
| Per Firth candidate aggregate       |         no |     yes |                yes | Aggregate only                                |
| Per Firth candidate event           |         no |      no |            limited | Only tiny fixtures                            |
| Per Firth iteration                 |         no |      no | tiny fixtures only | Otherwise too much                            |
| Per variant                         |         no |      no |                 no | Use sampled tracing if absolutely needed      |
| Per sample/genotype                 |         no |      no |                 no | Never for real runs                           |

A good production target is:

```text
< 1% overhead for progress mode
1–5% overhead for profile mode without deep JAX/device sync
arbitrary overhead acceptable for trace mode, because trace mode is diagnostic
```

But profile mode with forced JAX synchronization is not “1–5% overhead”; it can change the schedule materially. Treat it as a measurement mode, not normal runtime.

---

# Recommended event volume limits

Add hard caps:

```toml
[g.diagnostics]
max-events-per-run = 1000000
max-trace-events-per-run = 5000000
max-events-per-chunk = 20
sampled-variant-event-rate = 0
```

In trace mode, if logs exceed the cap:

```json
{
  "event": "telemetry_event_cap_reached",
  "dropped_or_suppressed_events": 123456
}
```

This prevents accidental per-variant logging from producing a 100 GB log file.

---

# Progress logging

Progress should be independent of debug logging.

Example `progress.jsonl`:

```json
{"event":"run_started","run_id":"...","phenotype_count":1,"total_variants":1200000}
{"event":"progress_tick","run_id":"...","completed_chunks":50,"completed_variants":409600,"elapsed_seconds":41.2,"variants_per_second":9941.7}
{"event":"chromosome_completed","run_id":"...","chromosome":"22","duration_ms":30120.4}
{"event":"run_completed","run_id":"...","duration_ms":32100.1}
```

Emit progress by:

```text
time interval: every 5 seconds
or chunk interval: every 10 chunks
or lifecycle boundary: chromosome/run start/end
```

This gives users confidence without turning logs into a profiler.

---

# Profile summary

Keep a summary file. Raw JSONL is for debugging; summary JSON is for benchmark comparisons.

Example:

```json
{
  "schema_version": 1,
  "run_id": "01HY...",
  "association_mode": "regenie2_linear",
  "phenotype_count": 1,
  "sample_count": 408123,
  "variant_count": 1200000,
  "stage_totals_seconds": {
    "bgen_engine_open_index_setup": 1.31,
    "sample_phenotype_covariate_alignment": 0.44,
    "prediction_source_load": 0.88,
    "native_engine_delivery": 14.94,
    "host_to_device_transfer": 1.77,
    "jax_compute": 6.26,
    "device_to_host_materialization": 0.54,
    "output_write": 0.91,
    "writer_finish_and_parquet_finalization": 2.10
  },
  "native_bgen_profile": {
    "variant_decode_count": 1200000,
    "decompression_ns": 123456789,
    "probability_decode_ns": 123456789,
    "output_write_ns": 123456789
  },
  "derived_metrics": {
    "variants_per_second": 37142.0,
    "dosage_values_per_second": 15100000000.0
  }
}
```

Your current timing recorder already produces stage totals, native BGEN profile, binary chunk diagnostics, null logistic diagnostics, and derived metrics.  Keep that, but make it part of the telemetry system.

---

# Logging filters I would use

## Normal run

```text
g=info,g.native=info,g.output=info
```

## Benchmark

```text
g=debug,g.native=debug,g.output=debug
```

## Native BGEN deep dive

```text
g=info,g.native.bgen=trace,g.output=debug
```

## Output bottleneck deep dive

```text
g=info,g.output=trace
```

## Binary/Firth deep dive

```text
g=debug,g.compute.binary=debug,g.compute.firth=trace
```

But keep Firth trace aggregate-first. Do not log every candidate iteration except on tiny fixtures.

---

# What to change in the current code

## 1. Add run context

Generate a `run_id` at the beginning of `runner.regenie(...)`, before logging starts. Current runner initializes logging and starts the run, but there is no visible run context. 

Add:

```python
run_id = create_run_id()
```

Bind it to all events.

## 2. Separate “stage timings” from “logging”

Currently, requesting `g-stage-timings-json` changes runtime behavior by enabling synchronization for accurate timing. That is okay, but users must know it.

Rename or document modes:

```text
telemetry=progress:
  no JAX sync

telemetry=profile:
  enables timing sync and detailed diagnostics

telemetry=trace:
  enables timing sync + high-volume native events
```

## 3. Add structured Python events

Use a helper instead of only `logger.info("Starting REGENIE run.")`.

```python
telemetry.event("run_started", association_mode=..., phenotype_count=...)
```

## 4. Add queue/cap controls to Rust logging

Current Rust logging uses `NonBlockingBuilder::default().lossy(true)` but does not expose queue size/lossy config.  Add:

```text
g-log-lossy
g-log-queue-size
```

## 5. Add hard policy for multiple Python API runs

Because current logging initialization is global and does not switch log files after first initialization, either:

```text
raise on incompatible log_file after first run
```

or implement per-run telemetry sessions. 

## 6. Add progress file

The current system has stage timing summaries but no dedicated progress stream. Add `progress.jsonl`.

## 7. Merge output timing summary

Rust writer has its own `output_stage_timings.json`.  Either keep it but link it from `profile.summary.json`, or merge the numbers into the main profile summary.

---

# Final recommendation

For `g`, the best setup is:

```text
Default:
  progress-level telemetry
  non-blocking lossy JSONL
  progress every 5 seconds / 10 chunks
  no JAX sync
  no per-chunk device_get diagnostics

Profile:
  per-chunk stage timings
  binary aggregate diagnostics
  native BGEN profile snapshot
  output writer timings
  profile.summary.json
  acceptable timing perturbation

Trace:
  per-tile native BGEN events
  writer queue events
  targeted module filters
  explicit small-scope diagnostic mode only
```

The granularity ceiling is:

```text
Production:
  run/chromosome/progress, maybe lightweight per chunk

Profile:
  per chunk and per writer batch

Trace:
  per decode tile

Never:
  per variant, per sample, per genotype, or per Firth iteration at full scale
```

The main thing to protect is not log volume itself. It is **avoiding accidental synchronization and host materialization**. Any log event that calls `device_get`, `block_until_ready`, converts JAX arrays to Python scalars, or builds large strings must be profile/trace-only.
