# Performance Guide

| Status | Applies to | Owner |
| --- | --- | --- |
| Public performance guidance, not a benchmark guarantee | CPU and GPU `g regenie` Step 2 runs | Public interface |

Performance depends on genotype format, trait mode, phenotype count, BGEN
decode cost, host-device transfer, JAX compilation, output format, storage, and
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
| Output stage slow | Storage throughput, writer threads, queue depth, output format, compression, and finalization. |
| Resume startup slow | Manifest validation mode and strict chunk reconciliation. |

## Runtime Knobs

Use the current packaged defaults first. Override only with measurements.

| Option | Main effect |
| --- | --- |
| `--bsize` | Variants per chunk; affects memory, JAX shapes, compilation, and per-chunk overhead. |
| `--threads` | Native Rayon thread request for Rust-owned work. |
| `--device` | JAX execution target, `cpu` or `gpu`. |
| `--staging_depth` | Native callback staging depth. |
| `--result_in_flight_limit` | Optional cap for result chunks awaiting materialization. |
| `--dosage_buffer_limit` | Optional cap for reusable native dosage decode buffers. |
| `--bgen_decode_tile_variant_count` | Native BGEN decode tile size. |
| `--gpu_genotype_format` | Host-to-device genotype representation for GPU-compatible paths. |
| `--format` | Arrow, Parquet, or REGENIE text materialization. |
| `--writer_threads` | Output writer worker count. |
| `--writer_queue_depth` | Output writer queue depth. |
| `--chunks_per_arrow_file` | Number of engine chunks grouped into each Arrow/Parquet/text part. |
| `--firth_batch_size` | Approximate-Firth candidate batch size. |
| `--firth_candidate_capacity` | Candidate capacity for binary fallback staging. |
| `--jax_persistent_cache` and `--jax_cache_dir` | JAX compilation cache behavior. |
| `--telemetry` | Progress, profile, and trace modes. Profile/trace can perturb timing. |

Current default values are in `src/interface/config.default.toml`.

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
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_cpu_regenie2 \
  --threads "${SLURM_CPUS_PER_TASK:-16}" \
  --device cpu
```

## GPU Runs

GPU runs require a CUDA-capable JAX environment and a GPU allocation. Single
phenotype scans can still be limited by BGEN decode, transfer, and output rather
than accelerator compute.

Useful GPU checks:

```bash
uv run python -c "import jax; print(jax.devices())"
uv run --no-sync g regenie --device gpu ...
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

## Output Choices

| Format | Use when |
| --- | --- |
| `parquet` | You want dataset parts and efficient downstream analytics. |
| `arrow` | You want Arrow IPC chunks for inspection or intermediate workflows. |
| `regenie` | You need REGENIE Step 2-style text compatibility. |

Finalizing a single `final.parquet` adds work after chunk output. Keep
`--finalize_parquet` off when the parts dataset is sufficient.

## Measuring

Production mode:

```bash
--telemetry progress
```

Profile mode:

```bash
--telemetry profile
--log_dir /path/to/logs
```

Trace mode is for small or capped runs:

```bash
--telemetry trace
--variant_limit 1000
--trace_event_cap 1000000
```

Trace can perturb performance and generate high-volume logs. Use it to diagnose
a specific small case, not as a production timing mode.

Development benchmark protocols and repository-specific SLURM recipes live in
[Benchmarking](../development/benchmarking.md) and
[Server Gauss SLURM](../development/server-gauss-slurm.md).
