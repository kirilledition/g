# GPU And Clusters

`g` executes statistical kernels through JAX. Choose the target device with:

```bash
--device cpu
--device gpu
```

CPU support is installed by the base runtime dependencies. GPU support requires the
[GPU installation flow](installation.md#gpu-install-from-source) and a scheduler allocation on a GPU
node.

GPU acceleration is workload-dependent. Single-trait runs can be limited by BGEN decode, host-device
transfer, or output writing rather than JAX compute.

## What To Check First

Run these checks inside the same environment and allocation where the scan will
run:

```bash
hostname
uv run python -c "import jax; print(jax.devices())"
uv run g regenie --help
```

If JAX does not list the expected GPU, fix the node allocation, driver, CUDA, or
JAX install before tuning `g`.

## Probe the JAX Runtime

Run this check in the same environment and on the same kind of node where the scan will run:

```bash
uv run python -c "import jax; print(jax.devices())"
```

If JAX cannot see the expected accelerator, fix the driver, CUDA, or JAX wheel environment before
measuring `g` performance. On a login node without NVIDIA devices, a CPU-only result can be
expected even when the GPU environment is otherwise valid.

## Generic SLURM GPU Job

Install and sync the checkout before submitting the job. Inside the batch script, use
`uv run --no-sync` so the job uses the already-created `.venv/` instead of trying to change the
environment while the scan is running:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=g-regenie2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd /path/to/g
export UV_CACHE_DIR="${SCRATCH:-$HOME}/.cache/uv"
export UV_LINK_MODE=copy

uv run --no-sync g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_gpu_regenie2 \
  --device gpu
```

Adjust `#SBATCH` options for your site's partitions, accounts, GPU resource syntax, and memory
policy.

## Generic SLURM CPU Job

Large CPU scans should also run on a compute node rather than a login node:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=g-regenie2-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd /path/to/g
export UV_CACHE_DIR="${SCRATCH:-$HOME}/.cache/uv"
export UV_LINK_MODE=copy

uv run --no-sync g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_cpu_regenie2 \
  --threads "${SLURM_CPUS_PER_TASK:-16}" \
  --device cpu
```

## Cluster Notes

- Do not run GPU scans or large CPU scans on a login node.
- Keep `UV_CACHE_DIR` and run output on user-writable storage with enough quota.
- Use `UV_LINK_MODE=copy` on shared filesystems where hardlinks or reflinks are unreliable.
- Run `uv sync` before submitting production jobs, then use `uv run --no-sync` inside batch jobs.
- Use upstream `regenie` to create Step 1 predictions before running `g` Step 2.

The gauss development-server recipes and benchmark wrappers are documented in
[Server Gauss SLURM](../development/server-gauss-slurm.md).

## Runtime Knobs

Important runtime knobs include:

| Option | Purpose |
| --- | --- |
| `--bsize` | Variants per chunk. |
| `--device` | JAX execution target. |
| `--staging_depth` | Native callback staging depth. |
| `--trusted_no_missing_diploid` | Enables trusted BGEN fast path after validation policy. |
| `--bgen_decode_tile_variant_count` | Native BGEN decode tile size. |
| `--writer_threads` | Output writer worker count. |
| `--writer_queue_depth` | Output writer queue depth. |
| `--firth_batch_size` | Binary approximate-Firth batch size. |
| `--jax_persistent_cache` | Enable JAX persistent compilation cache. |
| `--jax_cache_dir` | Persistent JAX compilation cache directory. |
| `--jax_xla_autotune_cache` | Enable XLA auxiliary autotune caches when the persistent cache is enabled. Prefer a local `--jax_cache_dir` for this cache. |
| `--jax_transfer_guard` | Enable JAX transfer guard diagnostics. |

Fair performance comparisons require equivalent statistical modes. Compare score-only to score-only,
and compare approximate Firth only when both tools use approximate Firth with the same fallback
threshold.

For broader tuning and measurement guidance, see [Performance Guide](performance-guide.md).
