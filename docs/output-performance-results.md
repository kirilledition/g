# Output Performance Results

This report compares the representative GPU output benchmark with a fresh CPU run from the
`feature/profile-output-hotspots` branch.

## Data Sources

- GPU: `data/benchmarks/output_handoff_representative_20260523_123420/output_stage_benchmark_summary.json`
  from the main worktree. This is a 64-case, one-trial representative run.
- CPU: `data/benchmarks/output_hotspots_cpu_20260523_143313/summary.json`
  from this profiling worktree. This is a 64-case, three-trial run executed on SLURM node
  `cantor` with `--device cpu` and `--data-dir /mnt/beegfs/kirill/Projects/g/data`.

The GPU and CPU runs have different trial counts, so compare large directional effects rather
than small differences.

## Representative Best Output Cases

These rows pick the fastest measured output path per device, phenotype count, and final format.
`Full output` is device/host materialization plus Python handoff overhead, Rust Arrow writing,
and Parquet finalization where present.

| Device | Case | Wall | Full output | Rust writer | Parquet finalization | Device-to-host/materialize | Python output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU | 1 trait Arrow | 3.152 s | 0.848 s | 0.740 s | 0.000 s | 0.102 s | 0.003 s |
| GPU | 1 trait Parquet | 3.731 s | 1.594 s | 0.850 s | 0.592 s | 0.134 s | 0.009 s |
| GPU | 8 traits Arrow | 24.005 s | 6.941 s | 6.204 s | 0.000 s | 0.685 s | 0.026 s |
| GPU | 8 traits Parquet | 29.115 s | 11.443 s | 6.067 s | 4.673 s | 0.655 s | 0.024 s |
| CPU | 1 trait Arrow | 8.830 s | 0.661 s | 0.626 s | 0.000 s | 0.022 s | 0.005 s |
| CPU | 1 trait Parquet | 9.318 s | 1.234 s | 0.636 s | 0.567 s | 0.022 s | 0.005 s |
| CPU | 8 traits Arrow | 72.027 s | 6.332 s | 6.096 s | 0.000 s | 0.160 s | 0.038 s |
| CPU | 8 traits Parquet | 78.136 s | 13.121 s | 5.990 s | 6.913 s | 0.148 s | 0.034 s |

For the best 8-trait Parquet output case, output is about 39.3% of GPU wall time and 16.8% of
CPU wall time. CPU is dominated by non-output work; GPU has a much larger output-path share.

## Stage Detail

The Python stage timers are nested: for example, `native_engine_delivery` includes lower-level
per-chunk work. Treat them as hotspot context, not as additive totals.

### GPU, 8-Trait Parquet

Best output case: `parquet_final_8_phenotypes_large_bsize_8192_writer1_queue1_chunks16_zstd`.

| Stage | Time | Wall share |
| --- | ---: | ---: |
| `native_engine_delivery` | 18.898 s | 64.9% |
| `host_to_device_transfer` | 14.216 s | 48.8% |
| `writer_finish_and_parquet_finalization` | 5.601 s | 19.2% |
| `preflight_validation` | 1.762 s | 6.1% |
| `bgen_engine_open_index_setup` | 1.110 s | 3.8% |
| `device_to_host_materialization` | 0.655 s | 2.2% |
| `jax_compute` | 0.500 s | 1.7% |
| `output_write` | 0.024 s | 0.1% |

Output-path breakdown:

| Output substage | Time | Output share | Wall share |
| --- | ---: | ---: | ---: |
| Rust Arrow writer total | 6.067 s | 53.0% | 20.8% |
| Parquet finalization total | 4.673 s | 40.8% | 16.1% |
| Device-to-host materialization | 0.655 s | 5.7% | 2.2% |
| Python output write | 0.024 s | 0.2% | 0.1% |

Important writer/finalization internals:

| Internal timer | Time |
| --- | ---: |
| `rust_output_writer_arrow_file_write` | 3.988 s |
| `rust_output_finalization_write_parquet` | 2.377 s |
| `rust_output_writer_record_batch_build` | 2.079 s |
| `rust_output_finalization_read_arrow` | 0.061 s |

### CPU, 8-Trait Parquet

Best output case: `parquet_final_8_phenotypes_large_bsize_8192_writer1_queue1_chunks16_zstd`.

| Stage | Time | Wall share |
| --- | ---: | ---: |
| `native_engine_delivery` | 62.915 s | 80.5% |
| `jax_compute` | 29.770 s | 38.1% |
| `host_to_device_transfer` | 24.915 s | 31.9% |
| `writer_finish_and_parquet_finalization` | 7.688 s | 9.8% |
| `preflight_validation` | 1.742 s | 2.2% |
| `callback_drain` | 1.446 s | 1.9% |
| `bgen_engine_open_index_setup` | 1.390 s | 1.8% |
| `device_to_host_materialization` | 0.148 s | 0.2% |
| `output_write` | 0.034 s | 0.0% |

Output-path breakdown:

| Output substage | Time | Output share | Wall share |
| --- | ---: | ---: | ---: |
| Parquet finalization total | 6.913 s | 52.7% | 8.8% |
| Rust Arrow writer total | 5.990 s | 45.7% | 7.7% |
| Device-to-host/materialization | 0.148 s | 1.1% | 0.2% |
| Python output write | 0.034 s | 0.3% | 0.0% |

Important writer/finalization internals:

| Internal timer | Time |
| --- | ---: |
| `rust_output_writer_arrow_file_write` | 4.289 s |
| `rust_output_writer_arrow_batch_write` | 3.741 s |
| `rust_output_finalization_write_parquet` | 3.190 s |
| `rust_output_writer_record_batch_build` | 1.701 s |
| `rust_output_writer_metadata_arrays` | 1.218 s |
| `rust_output_finalization_read_arrow` | 0.249 s |

## Parameter Effects

Mean full output time by benchmark knob:

| Device | Workload | `bsize=1024` | `bsize=8192` | Change |
| --- | --- | ---: | ---: | ---: |
| GPU | 8 traits Arrow | 12.586 s | 8.458 s | 32.8% faster |
| GPU | 8 traits Parquet | 20.184 s | 13.240 s | 34.4% faster |
| CPU | 8 traits Arrow | 19.088 s | 7.843 s | 58.9% faster |
| CPU | 8 traits Parquet | 33.554 s | 17.489 s | 47.9% faster |

| Device | Workload | `chunks_per_arrow_file=4` | `chunks_per_arrow_file=16` | Change |
| --- | --- | ---: | ---: | ---: |
| GPU | 8 traits Arrow | 12.127 s | 8.917 s | 26.5% faster |
| GPU | 8 traits Parquet | 18.539 s | 14.885 s | 19.7% faster |
| CPU | 8 traits Arrow | 16.807 s | 10.124 s | 39.8% faster |
| CPU | 8 traits Parquet | 31.133 s | 19.909 s | 36.1% faster |

Compression was not a clear output-speed win:

| Device | Workload | `none` | `zstd` | Read |
| --- | --- | ---: | ---: | --- |
| GPU | 8 traits Arrow | 10.144 s | 10.900 s | `none` faster |
| GPU | 8 traits Parquet | 17.049 s | 16.375 s | `zstd` slightly faster |
| CPU | 8 traits Arrow | 13.068 s | 13.863 s | `none` faster |
| CPU | 8 traits Parquet | 25.456 s | 25.586 s | effectively tied |

Writer threads were mixed:

| Device | Workload | 1 thread | 4 threads | Read |
| --- | --- | ---: | ---: | --- |
| GPU | 8 traits Arrow | 10.587 s | 10.457 s | tied |
| GPU | 8 traits Parquet | 17.385 s | 16.039 s | 4 threads helped |
| CPU | 8 traits Arrow | 13.830 s | 13.101 s | 4 threads helped slightly |
| CPU | 8 traits Parquet | 24.815 s | 26.227 s | 1 thread was better |

## Biggest Potential Wins

1. Direct Parquet output is the highest-confidence structural win for final Parquet mode. The
   current path writes Arrow chunks, reopens them, reads them back, and then writes Parquet.
   In the best 8-trait Parquet case, finalization alone costs 4.673 s on GPU and 6.913 s on
   CPU. If a direct writer also avoids much of the intermediate Arrow file write, the upper
   bound is roughly finalization plus Arrow file write: 8.661 s on GPU and 11.202 s on CPU.
   The realistic win is lower, but this is the clearest target.

2. Wide multi-trait output should be prototyped. The 8-trait output path scales much more than
   8x in some CPU cases and repeats metadata/files per trait. Best CPU full output is 0.661 s
   for 1-trait Arrow versus 6.332 s for 8-trait Arrow, and 1.234 s for 1-trait Parquet versus
   13.121 s for 8-trait Parquet. GPU shows the same pattern, though less severely.

3. Larger chunks and fewer files are already proven wins. `bsize=8192` is consistently better
   than `1024`, and `chunks_per_arrow_file=16` is consistently better than `4`. Profile a larger
   candidate such as `16384` or `32768`, and include `chunks_per_arrow_file=64`.

4. Intermediate Arrow compression should not be the default speed path. `zstd` is slower for
   Arrow-only output on both CPU and GPU, and CPU Parquet output is effectively tied. If final
   Parquet stays compressed, either leave intermediate Arrow uncompressed or remove it with
   direct Parquet.

5. DLPack and Python handoff are not the limiting path. In the best 8-trait Parquet case,
   device-to-host/materialization plus Python output write is 0.679 s on GPU and 0.182 s on CPU.
   That is much smaller than writer/finalization time.

## Recommendation

Keep production defaults unchanged on this profiling branch. The next implementation prototype
should be direct Parquet for final output, with an experimental wide multi-trait schema behind an
opt-in benchmark flag. Use `bsize=8192` and `chunks_per_arrow_file=16` as the current baseline,
then test larger values before changing defaults.

Promote the optimization only if it clears the earlier threshold: at least 20% representative
8-trait wall-time improvement, or at least 10% single-trait wall-time improvement.
