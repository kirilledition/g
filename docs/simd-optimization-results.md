# Trusted BGEN SIMD Optimization Results

Date: 2026-05-23
Host: `cantor`

## Summary

The remaining trusted identity decode modes from `docs/simd-optimization-plan.md` were implemented and benchmarked.

- Keep `auto` dispatching to raw AVX2 on AVX2-capable x86 CPUs.
- Keep lookup scalar as the `auto` fallback on non-AVX2 CPUs.
- Keep raw scalar only as an internal benchmark/test override because it regressed larger reader chunks.
- Do not pursue AVX-512, selected-subset SIMD, or row-major SIMD for this path.

## Reader Benchmark

Command shape:

```bash
srun --nodelist=cantor --cpus-per-task=40 --mem=16G --time=00:45:00 bash -lc \
  'cd /home/kirill/Projects/g-worktrees/bgen-simd-plan-completion &&
   . scripts/server_env.sh &&
   for mode in lookup raw_scalar raw_avx2 auto; do
     G_BGEN_SIMD=${mode} RUSTFLAGS="-C target-cpu=native" \
       cargo bench --bench bgen_read bgen_preprocessed_variant_major_trusted_no_missing_diploid \
       -- --sample-size 10 --measurement-time 1 --warm-up-time 1
   done'
```

Median times:

| Chunk size | lookup | raw scalar | raw AVX2 | auto |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 4.3336 ms | 4.1988 ms | 3.0116 ms | 2.9622 ms |
| 2048 | 4.0504 ms | 4.1825 ms | 2.9495 ms | 2.9606 ms |
| 4096 | 6.5958 ms | 6.6772 ms | 5.0574 ms | 5.0130 ms |
| 8192 | 12.089 ms | 12.366 ms | 9.0766 ms | 9.0681 ms |
| 16384 | 21.904 ms | 22.138 ms | 16.731 ms | 16.715 ms |

Criterion reported raw AVX2 improved reader time by about 25-27% for chunk sizes 4096-16384. Raw scalar regressed those larger chunks by about 1.7-2.6%, so it is not a production fallback.

## Native Profile Counters

Command shape:

```bash
srun --nodelist=cantor --cpus-per-task=40 --mem=16G --time=00:15:00 bash -lc \
  'cd /home/kirill/Projects/g-worktrees/bgen-simd-plan-completion &&
   . scripts/server_env.sh &&
   for mode in lookup raw_scalar raw_avx2 auto; do
     G_BGEN_SIMD=${mode} G_BGEN_PROFILE_CHUNK_SIZE=16384 RUSTFLAGS="-C target-cpu=native" \
       cargo run --release --example bgen_trusted_profile
   done'
```

Single 16,384-variant chunk profile:

| Mode | decompression_ns | probability_decode_ns | output_write_ns |
| --- | ---: | ---: | ---: |
| lookup | 414,107,129 | 765,949,602 | 763,091,426 |
| raw_scalar | 426,626,205 | 786,003,517 | 783,146,020 |
| raw_avx2 | 455,462,499 | 578,556,384 | 575,467,948 |
| auto | 450,862,963 | 601,849,412 | 598,829,025 |

## Microbenchmark

The synthetic trusted identity Criterion benchmark covered 10k, 100k, and 500k samples across all-zero, all-two, alternating raw, rare-variant-like, and deterministic valid random probability patterns.

Result:

- Raw AVX2 and auto were consistently faster than lookup.
- Raw scalar was consistently slower than lookup.
- Auto and raw AVX2 were effectively equivalent on `cantor`.
