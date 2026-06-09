# Output Performance Results

Historical profiling from `feature/profile-output-hotspots` showed that output cost is dominated by
the Rust Arrow writer and optional Parquet finalization, not by the Python/JAX handoff. These
numbers were collected before the Review 2 output refactor, so treat them as directional context and
refresh the benchmark before changing defaults.

## Representative Findings

- Best 8-trait GPU Parquet output spent about 6.1 s in Rust Arrow writing and 4.7 s in Parquet
  finalization, with less than 0.7 s in device-to-host materialization plus Python handoff.
- Best 8-trait CPU Parquet output spent about 6.0 s in Rust Arrow writing and 6.9 s in Parquet
  finalization, with less than 0.2 s in device-to-host materialization plus Python handoff.
- Larger chunks helped consistently: `bsize=8192` was faster than `bsize=1024` for the measured
  8-trait Arrow and Parquet cases.
- Fewer intermediate Arrow files helped: `chunks_per_arrow_file=16` was faster than `4` in the
  measured 8-trait cases.
- Intermediate Arrow `zstd` compression was not a clear speed win for Arrow-only output.

## Implications

Direct Parquet output remains the highest-confidence structural optimization candidate for final
Parquet mode because the current path writes Arrow chunks, reopens them, reads them back, and writes
Parquet. A wide multi-trait output layout is the next candidate if repeated per-trait metadata and
file work dominate refreshed profiles.

Use `python -m tooling.cli.benchmark_output_stages` to refresh the evidence on current `main`. The module now
emits detailed Rust writer/finalization substage timings, byte counters, throughput summaries,
ranked bottlenecks, and optional JAX profiler traces.
