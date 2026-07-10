# Performance Guide

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft; guidance, not a benchmark guarantee | main branch as of 2026-07-01 CPU and GPU Step 2 runs | Public user docs |

Performance depends on genotype format, trait mode, phenotype count, BGEN
decode cost, host-device transfer, JAX compilation, Parquet writing, storage, and
cluster placement. Treat numbers from one machine as local evidence, not a
portable guarantee.

## What To Check First

1. Confirm you are running on the intended node.
2. Confirm JAX sees the intended device:

   ```bash
   uv run python -c "import jax; print(jax.devices())"
   ```

3. Confirm the statistical modes are comparable. Compare score-only to
   score-only, and approximate Firth to approximate Firth with the same
   threshold.
4. Check whether the run is dominated by BGEN decode, host-device transfer,
   JAX compute, or output writing before changing knobs.
5. Use telemetry/profile output from the same command shape you intend to tune.

## Common Bottlenecks

| Symptom | Likely first checks |
| --- | --- |
| GPU visible but little speedup | Single phenotype, small chunks, BGEN decode, host-device transfer, or output may dominate. |
| First run slow but repeated work faster | JAX backend initialization and compilation are likely visible. |
| Approximate Firth much slower than binary score test | Candidate density from `--pThresh` and Firth solver work dominate. |
| Output stage slow | Storage throughput, writer threads, queue depth, part grouping, and Parquet compression. |
| Resume startup slow | Manifest validation mode and strict chunk reconciliation. |

## Cold, Warm, And Hot Runs

Do not compare timing modes as if they measured the same thing:

| Mode | Includes |
| --- | --- |
| Cold process | Python process startup, JAX backend initialization, first compilation, BGEN decode, compute, and output. |
| Warm JAX cache | New process startup plus cache lookup/reuse when persistent cache entries match. |
| Hot same process | Already-live Python process, JAX backend, and compiled functions for compatible shapes. |

Use cold-process timing for batch-job wall-clock expectations. Use warm-cache
or hot same-process timing only when that is the workflow being measured.

## Runtime Knobs

Use the current packaged defaults first. Override only with measurements.

| Option | Main effect |
| --- | --- |
| `--bsize` | Variants per chunk; affects memory, JAX shapes, compilation, and per-chunk overhead. |
| `--threads` | Native Rayon thread request for Rust-owned work. |
| `[compute] device` | JAX execution target, `cpu` or `gpu`. |
| `[compute] staging_depth` | Decoded batches allowed ahead of device compute. |
| `[compute] result_in_flight_limit` | Device results allowed ahead of host materialization. |
| `[compute] bgen_decode_tile_variant_count` | Native BGEN decode tile size. |
| `[compute] gpu_genotype_format` | Host-to-device representation; `auto` upgrades eligible single-trait binary GPU runs to packed8. |
| `[output] writer_threads` | Output writer worker count. |
| `[output] writer_queue_depth` | Output writer queue depth. |
| `[output] chunks_per_parquet_file` | Engine chunks grouped into each Parquet part. |
| `[output] parquet_compression` | Parquet compression, `none` or `zstd`. |
| `[compute] firth_batch_size` | Approximate-Firth candidate batch size. |
| `[compute] firth_candidate_capacity` | Candidate capacity for binary fallback staging. |
| `[compute] jax_persistent_cache`, `jax_cache_dir` | JAX compilation cache behavior. |
| `[diagnostics] telemetry` | Progress, profile, and trace modes; profile/trace can perturb timing. |

`gpu_genotype_format = "auto"` resolves to packed8 only for single-trait binary
GPU REGENIE Step 2 runs when trusted no-missing diploid BGEN validation passes.
It falls back to dosage for CPU, linear, grouped, multi-phenotype, and
incompatible BGEN cases. Explicit `packed8` keeps fail-fast validation behavior.

Current default values are in `crates/interface/src/config.default.toml`.

Project profiling recipes isolate CPU JAX caches by host and CPU feature
fingerprint under `/tmp/g-jax-cpu-profile-cache` by default. This avoids reusing
CPU AOT artifacts across SLURM nodes when profile outputs are stored on shared
filesystems. GPU profile caches remain node-local under `/tmp/g-jax-profile-cache`
or `/tmp/g-jax-binary-hot-cache` unless `G_PROFILE_GPU_JAX_CACHE_PARENT` is
overridden.

## CPU Runs

CPU runs exercise native BGEN decode, sample alignment, output writing, and JAX
CPU kernels. For large scans on a cluster, submit to a compute node instead of a
login node and pass the scheduler CPU count to `--threads`.

```bash
uv run --no-sync g regenie \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_cpu_regenie2 \
  --threads "${SLURM_CPUS_PER_TASK:-16}"
```

## GPU Runs

GPU runs require a CUDA-capable JAX environment and a GPU allocation. Single
phenotype scans can still be limited by BGEN decode, transfer, and output rather
than accelerator compute.

Useful GPU checks:

```bash
uv run python -c "import jax; print(jax.devices())"
uv run --no-sync g regenie --config gpu.toml ...
```

Multi-phenotype quantitative runs can amortize BGEN decode and process startup
when sample semantics allow it. Use `per-phenotype` for separate-run semantics
and `complete-case` only when the shared complete-case intersection is intended.

As a practical rule of thumb:

- `per-phenotype` maximizes per-trait usable samples and is the conservative
  choice when missingness differs by phenotype.
- `complete-case` tends to reduce repeated alignment work because one shared mask
  is used across traits, but it can reduce sample size for each trait and change
  statistics when missingness differs.

If speed is the only concern, test both modes on a small representative subset
before changing production scripts.

## Parquet Output

Each phenotype run writes a `parts/` Parquet dataset. Tune writer threads,
queue depth, part grouping, and compression against the target filesystem.
Larger `chunks_per_parquet_file` values reduce file counts but delay each file
commit and increase the amount of work repeated after an interruption.

## Measuring

Production mode:

```toml
[diagnostics]
telemetry = "progress"
```

Profile mode:

```toml
[diagnostics]
telemetry = "profile"
log_dir = "/path/to/logs"
```

Trace mode is for small or capped runs:

```toml
[compute]
variant_limit = 1000

[diagnostics]
telemetry = "trace"
trace_event_cap = 1000000
```

Trace can perturb performance and generate high-volume logs. Use it to diagnose
a specific small case, not as a production timing mode.

Development benchmark protocols and repository-specific SLURM recipes live in
[Benchmarking](../development/benchmarking.md) and
[Server Gauss SLURM](../development/server-gauss-slurm.md).
