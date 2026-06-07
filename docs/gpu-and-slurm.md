# GPU and SLURM

`g` can execute statistical kernels on CPU or GPU through JAX:

```bash
--g-device cpu
--g-device gpu
```

GPU acceleration is workload-dependent. Single-trait runs can be limited by BGEN decode, host-device transfer, or output writing rather than JAX compute.

## Local GPU Checks

Bootstrap and probe the JAX runtime:

```bash
just bootstrap-gpu
just doctor-jax
```

If JAX cannot see the expected accelerator, fix the environment before measuring `g` performance.

## Server Rules

On the gauss server, do not run heavy computation, large test suites, or GPU workloads on the login node. Use SLURM recipes for GPU work. The default GPU node is configured through `GWAS_ENGINE_GPU_NODE` and defaults to `landau`.

Useful recipes:

```bash
just slurm-gpu-shell
just slurm-gpu-run 'uv run --no-sync python scripts/probe_jax_runtime.py'
just slurm-gpu-just regenie2-binary-gpu-smoke
```

Binary GPU examples:

```bash
just setup-regenie2-binary-gpu-inputs
just verify-regenie2-binary-gpu-inputs
just slurm-regenie2-binary-gpu-smoke
just slurm-regenie2-binary-gpu
```

## Performance Notes

Important runtime knobs include:

| Option | Purpose |
| --- | --- |
| `--bsize` | Variants per chunk. |
| `--g-device` | JAX execution target. |
| `--g-staging-depth` | Native callback staging depth. |
| `--g-trusted-no-missing-diploid` | Enables trusted BGEN fast path after validation policy. |
| `--g-bgen-decode-tile-variant-count` | Native BGEN decode tile size. |
| `--g-writer-threads` | Output writer worker count. |
| `--g-writer-queue-depth` | Output writer queue depth. |
| `--g-firth-batch-size` | Binary approximate-Firth batch size. |
| `--g-jax-persistent-cache` | Enable JAX persistent compilation cache. |

Fair performance comparisons require equivalent statistical modes. Compare score-only to score-only, and compare approximate Firth only when both tools use approximate Firth with the same fallback threshold.
