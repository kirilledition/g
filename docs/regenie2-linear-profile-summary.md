# REGENIE2 Linear Profile Summary

## Run

Saved locally with:

```bash
nix develop --command uv run python scripts/profile_regenie2_linear_detailed.py \
  --device gpu \
  --chunk-size 1024 \
  --warmup-pass-count 1 \
  --enable-jax-trace \
  --enable-memory-profile \
  --output-dir data/profiles/regenie2_linear \
  --report-name full_chr22_gpu_chunk1024
```

## Dataset

- BGEN: `data/1kg_chr22_full.bgen`
- Sample file: `data/1kg_chr22_full.sample`
- Phenotype: `data/pheno_cont.txt`
- Covariates: `data/covariates.txt`
- Prediction list: `data/baselines/regenie_step1_qt_pred.list`

## Output

- Variants: `418,943`
- Chunks: `410`
- Wall time: `23.416s`
- Throughput: `17,891 variants/s`

## Key Timing Breakdown

| Stage | Total | Mean per event | Share of wall |
|---|---:|---:|---:|
| `bgen_read_host` | 12.325s | 30.062ms | 52.64% |
| `write_chunk_to_disk` | 1.006s | 2.453ms | 4.30% |
| `get_variant_table_arrays` | 0.925s | 2.255ms | 3.95% |
| `device_put_genotypes` | 0.373s | 0.910ms | 1.59% |
| `split_chunk_by_chromosome` | 0.155s | 0.377ms | 0.66% |
| `build_chunk_payload` | 0.122s | 0.297ms | 0.52% |
| `preprocess_genotypes` | 0.083s | 0.203ms | 0.36% |
| `compute_regenie2_linear_chunk` | 0.079s | 0.194ms | 0.34% |

## cProfile Highlights

Hottest functions in the saved run:

- `cbgen._ffi.bgen_file_open_genotype`
- `cbgen._ffi.bgen_genotype_read32`
- `_bgen_file.py:211(read_probability)`
- `bgen.py:294(read)`
- `_bgen2.py:348(read)`

Interpretation:

- the current implementation is not GPU compute bound
- it is mostly paying for host-side BGEN decode and backend call overhead

## Perfetto Highlights

The trace also shows the same pattern:

- BGEN backend worker activity dominates meaningful runtime
- chunk compute is very small
- writer lifetime is long, but actual write cost is small relative to BGEN reads

## Main Conclusion

The fastest path to materially improving REGENIE step 2 performance is:

1. reduce BGEN ingestion overhead
2. reduce repeated per-chunk metadata work
3. batch host materialization to reduce device-to-host sync frequency

Optimizing the JAX kernel itself should not be the priority yet.
