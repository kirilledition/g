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

Binary score-only runs prepare only the null-logistic and score state. They do
not pay the chromosome-level null-Firth fit, full-null deviance, or Firth-state
memory cost; those operations are exclusive to approximate-Firth runs.

## Cold, Warm, And Hot Runs

Do not compare timing modes as if they measured the same thing:

| Mode | Includes |
| --- | --- |
| Cold process | Python process startup, JAX backend initialization, first compilation, BGEN decode, compute, and output. |
| Warm JAX cache | New process startup plus cache lookup/reuse when persistent cache entries match. |
| Hot same process | Already-live Python process, JAX backend, and compiled functions for compatible shapes. |

Use cold-process timing for batch-job wall-clock expectations. Use warm-cache
or hot same-process timing only when that is the workflow being measured.

For several compatible production scans, use the native batch command to make
the hot same-process path explicit:

```bash
uv run --no-sync g batch \
  --config chromosome_21.toml \
  --config chromosome_22.toml
```

Batch mode constructs every frontend config and checks output-root and
process-policy compatibility before starting. Run-owned input, output, and
resume preflight remains per entry. The process reuses only process-global
JAX/CUDA state and compiled executables. Shape changes may compile an additional
executable, so group configs with stable shapes when throughput is the priority.

Native decode buffers submitted to JAX transfer their allocation into NumPy;
there is no full genotype memcpy at the binding boundary. Grouped-union runs
retain only the union source buffer for projection reuse. Phenotype, covariate,
and single-use LOCO matrices also transfer their allocations directly into
NumPy. Native input indexes each selected LOCO file once using byte offsets and
line numbers. Files with identical headers share one loader-side identifier
index and one alignment recipe per group; those large header strings are
discarded after source construction. Resume planning validates only chromosome
blocks that still need output. When execution reaches one of those blocks,
input reads, parses, finite-validates, and aligns just that chromosome into its
final trait-major matrix. File metadata snapshots and raw-row SHA-256 digests
reject changes between indexing and deferred reading. Fully committed
chromosomes therefore never allocate or parse prediction values. A repeated
noncontiguous chromosome block alone keeps one matrix for safe clones until its
final planned use.

Packed8 GPU delivery keeps one device-compute variant shape across a run. A
short chromosome-boundary or final chunk is padded only after exact BGEN decode
to `min(bsize, effective scan size)`; metadata, statistics written to output,
and result rows retain the logical variant count. This trades a bounded amount
of final-chunk transfer and compute for one reusable JAX executable. A small
`variant_limit` also caps the compute shape, so smoke runs do not expand to the
configured production chunk size.

The native association scheduler starts one compute thread, one host-result
materialization thread, and one bounded channel set for the delivery, independent
of active phenotype-group count. Per-group state and counters preserve result
routing. At a chromosome boundary, all results drain and the compute worker
acknowledges destruction of each replaced JAX state before its successor is built,
which avoids both per-chromosome worker churn and overlapping chromosome-state
device memory. Group-level device state is created at first use and released
after its final chromosome preparation. Fully resumed groups initialize progress but do not select BGEN
samples, prepare JAX state, or start scheduler workers. Partial resume rebuilds
the union from groups with pending output; one remaining group or a union without
sample overlap uses direct delivery instead of union projection.

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
| `[diagnostics] telemetry` | `off`, `progress`, or `profile`; profile mode can perturb timing. |

`gpu_genotype_format = "auto"` resolves to packed8 only for single-trait binary
GPU REGENIE Step 2 runs when trusted no-missing diploid BGEN validation passes.
It falls back to dosage for CPU, linear, grouped, multi-phenotype, and
incompatible BGEN cases. Explicit `packed8` keeps fail-fast validation behavior.

Current default values are in `crates/interface/src/config.default.toml`.
The current Firth batch default is tuned for hot same-process throughput on the
V100 reference node. Larger batches can reduce some separate-process cache-load
overhead while increasing device execution time, so compare complete workflows
rather than choosing from compile or kernel timing alone.

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
```

Development benchmark protocols and repository-specific SLURM recipes live in
[Benchmarking](../development/benchmarking.md) and
[Server Gauss SLURM](../development/server-gauss-slurm.md).
