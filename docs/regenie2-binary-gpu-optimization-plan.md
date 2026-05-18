# REGENIE2 Binary GPU Optimization Plan

## Goal

Reduce full-chromosome `g regenie2 --trait-type binary --device gpu` wall time for the CLI workflow without changing the public output schema or statistical defaults.

## Current Profile Basis

Recent local landau artifacts:

- `data/profiles/optimized_current_20260516T164459Z/binary_stage_timings.json`
- `data/profiles/optimized_current_20260516T164459Z/binary_api_timed_fresh_cache.stdout`
- `data/profiles/landau_gpu_fixed_20260516T113349Z/tuning_binary_gpu.json`
- `data/profiles/optimized_current_20260517T_landau_current/current_binary_gpu_summary.json`
- `data/profiles/variant_major_binary_20260517T_parity/variant_major_binary_summary.json`

Key observations from the current profiled full chr22 binary GPU run:

| Signal | Value |
|---|---:|
| Output rows | 418,943 |
| Firth candidates | 17,938 |
| Max Firth candidates in one chunk | 531 |
| Full chunk variant count | 8,192 |
| `native_engine_delivery` | 14.94s |
| `jax_compute` | 6.26s |
| `host_to_device_transfer` | 1.77s |
| Firth failures | 2,295 |

The current binary path spends time in two different places:

- Native BGEN decode and row-major chunk staging.
- JAX Firth correction for score-test candidate variants.

## Strategy Ranking

### 1. Bound Padded Firth Candidate Work

Status: implemented.

The Firth correction path currently sizes `jnp.nonzero(..., size=...)` and downstream gather buffers to the full chunk width. For the standard `chunk_size=8192`, this builds candidate buffers sized for 8,192 variants even though the observed maximum candidate count was 531.

Plan:

1. Add `G_REGENIE2_BINARY_FIRTH_CANDIDATE_CAPACITY`, default `1024`.
2. Build Firth batch plans against that capacity when `fallback_count <= capacity`.
3. Keep an automatic full-chunk fallback when any chunk exceeds the configured capacity.
4. Preserve output parity by applying the same Firth kernel and merge logic after candidate selection.
5. Add tests for:
   - bounded plan shape and active mask construction;
   - overflow fallback parity against the bounded/full path;
   - existing Firth candidate behavior.

Expected benefit:

- Reduces candidate gather, heuristic pre-dispatch, initial coefficient construction, and scan padding from full chunk width to a bounded capacity.
- For the current chr22 profile, default capacity 1024 covers all observed chunks and reduces candidate-lane padding from 8,192 to 1,024 per full chunk.

Risk:

- Too-low capacity silently truncating candidates would be unacceptable. The implementation must branch to the existing full-width path when the dynamic candidate count exceeds capacity.

Implementation notes:

- Environment variable: `G_REGENIE2_BINARY_FIRTH_CANDIDATE_CAPACITY`
- Default: `1024`
- Overflow behavior: if `fallback_count > capacity`, the kernel branches to the existing full chunk-width plan.
- Validation: `tests/test_regenie2_binary.py` covers capacity parsing, fixed-shape batch planning, and overflow fallback parity.

Initial landau measurement:

| Scenario | Candidate capacity | Wall time |
|---|---:|---:|
| Hot in-process binary GPU, no XLA autotune cache | 1,024 | 15.55s |
| Hot in-process binary GPU, no XLA autotune cache | 8,192 | 15.74s |

Interpretation:

- The bounded candidate path is correct and slightly faster in hot execution.
- Cold timings are still dominated by compilation and cache policy, so the next optimization should address XLA autotune cache reliability and same-process warm/run behavior before reading too much into cold wall-time deltas.

### 2. Fuse Trusted No-Missing BGEN Stats

Status: implemented.

Rust currently decodes a full row-major dosage matrix and then scans it again to compute allele frequencies, observation counts, and missing-value imputation. For validated no-missing diploid input, the decode loop can compute allele sums directly and set `observation_count = sample_count`.

Plan:

1. Add a REGENIE2-only CLI/API flag for trusted no-missing diploid BGENs.
2. Reuse the existing native validation function before enabling the trusted fast path.
3. Accumulate allele sums during decode.
4. Return `ChunkStats` without a second full matrix scan.

Detailed plan:

1. Expose `trusted_no_missing_diploid` on `api.ComputeConfig` and as `--trusted-no-missing-diploid` on `g regenie2` and `g regenie2-linear`.
2. Pass the flag to `_core.Regenie2RunEngine`.
3. When the flag is enabled, call `validate_trusted_no_missing_diploid()` immediately after engine open and before chunk delivery.
4. Split the native BGEN decode result into:
   - per-tile profiling counters;
   - optional per-variant selected-sample dosage totals.
5. Keep optional totals disabled for ordinary reads.
6. In `read_preprocessed_dosage_f32_into_address_prepared`, use the optional totals to build:
   - `allele_one_frequency = dosage_total / (2 * selected_sample_count)`;
   - `observation_count = selected_sample_count`;
   - `has_missing_values = false`.
7. Preserve the safe path for untrusted reads, including missing-value imputation and `NaN` scanning.

Implementation notes:

- Python/API flag: `ComputeConfig.trusted_no_missing_diploid`
- CLI flag: `--trusted-no-missing-diploid`
- Native validation remains explicit and rejects unsupported phased, non-diploid, missing, non-biallelic, or non-8-bit BGEN variants before the pipeline uses the trusted path.
- The trusted fast path affects only native preprocessed dosage reads. JAX kernels and output schema are unchanged.

Initial local measurement:

| Scenario | Mean full chr22 native preprocessed read | Notes |
|---|---:|---|
| Safe stats scan | 11.31s | 3 repeats, 418,943 variants |
| Trusted fused stats | 8.81s | 3 repeats, identical checksum |
| Trusted validation | 4.00s | One-time validation cost, measured separately |

Interpretation:

- The native decode/stats portion is 22% faster after validation is excluded.
- A CLI run that validates every time may not win end-to-end unless the saved stats scan exceeds validation cost. This flag is most useful when the same validated BGEN is reused across runs, or if a future mode records trusted validation out of band.

Follow-up: trusted validation amortization.

Status: implemented behind an internal switch.

Repeated landau jobs on the same chr22 BGEN pay trusted validation every process even when the file has already been validated in the same workflow. Hot trusted runs currently spend about `2.55s` in `bgen_engine_open_index_setup`, mostly validation. For a CLI-first workflow, this is a large fraction of the remaining wall time.

Detailed plan:

1. Keep the default behavior unchanged: `trusted_no_missing_diploid=True` still validates before using trusted decode paths.
2. Add an explicit internal switch for workflows that have already validated the exact input BGEN:
   - `G_REGENIE2_ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED=1`
3. When the switch is set, construct the trusted engine but skip `validate_trusted_no_missing_diploid()`.
4. Keep the switch out of normal CLI help for now because it bypasses a safety check.
5. Add tests that the default trusted path validates and the explicit assumed-validated path does not.

Risk:

- If the switch is used on a BGEN that is not actually biallelic, diploid, no-missing, unphased, and 8-bit, trusted decode results are not guaranteed. This is an expert-only escape hatch for repeated runs after prior validation.

Measurement target:

- Reduce trusted hot full chr22 wall by approximately the validation stage cost, currently around `2.55s`.

Landau measurement:

| Scenario | Wall | Engine open | Native delivery | JAX compute | Notes |
|---|---:|---:|---:|---:|---|
| Trusted variant-major, validation enabled | 9.828s | 2.551s | 6.955s | 4.967s | Exact parity against sample-major |
| Trusted variant-major, validation assumed complete | 7.379s | 0.159s | 6.892s | 4.943s | Same process hot run |

Interpretation:

- Skipping repeated trusted validation after an explicit prior validation saves about `2.39s` in the hot full-chr22 run.
- This makes the trusted variant-major path slightly faster than the latest safe hot baseline (`7.676s`) while retaining exact parity against the current sample-major JAX kernel for the same decoded data path.

### 3. Variant-Major Binary Flow

Status: implemented behind an internal switch, with parity-preserving JAX fallback.

Binary score and Firth naturally consume candidate genotypes by variant. The current path decodes sample-major and then gathers candidate columns with a transpose. A variant-major decode and JAX path would reduce strided writes and candidate transpose work.

Detailed plan:

1. Add a native BGEN decode API that fills a C-contiguous `(variant_count, sample_count)` float32 buffer. Done.
2. Keep the current sample-major API for untrusted BGENs, where the trusted variant-major decoder is not available. Done.
3. Add a Python callback path for variant-major preprocessed chunks with the same `_core.VariantMetadata` and `_core.ChunkStats` objects. Done.
4. Add binary-only JAX kernels that accept `genotype_matrix_by_variant` directly:
   - score test uses variant-major matrix-vector and matrix-matrix products;
   - Firth candidate selection uses direct row gathers instead of column gathers and transposes;
   - Firth batch planning keeps the bounded candidate capacity introduced above.
5. Add a temporary internal switch for A/B testing, then remove it once the trusted path is accepted. Done.
6. Test against the sample-major path for:
   - row counts;
   - output schema;
   - beta, standard error, chi-squared, log10p, extra-code parity within existing tolerances.
7. Benchmark with the same full chr22 binary GPU workload, recording native BGEN profile counters and JAX stage timings separately.

Correctness result:

- Direct variant-major JAX score/Firth kernels passed toy tests but did not preserve full chr22 Firth parity. The full run had the same 17,938 Firth candidates but different Firth failure counts and 1,568 `EXTRA` mismatches.
- Production variant-major mode therefore uses the native `(variant, sample)` decoder, then transposes on device into the existing sample-major JAX binary kernel. This keeps the lower Rust write volume while preserving exact current statistical output.
- Full chr22 parity after the fallback:
  - row count equal: `418,943`;
  - `A1FREQ`, `BETA`, `SE`, `CHISQ`, `LOG10P` max absolute difference: `0.0`;
  - `N` mismatch count: `0`;
  - `EXTRA` mismatch count: `0`.

Implementation notes:

- Trusted binary BGEN runs now use native variant-major delivery automatically.
- Untrusted binary BGEN runs continue to use sample-major delivery.
- The temporary `G_REGENIE2_BINARY_VARIANT_MAJOR` comparison switch was removed after parity and performance were measured.
- Native API: `run_bgen_variant_major_dosage_buffered_chunks`
- Python callback: `compute_preprocessed_variant_major_dosage_chunk`
- Current JAX production behavior: `jnp.transpose(genotype_matrix_by_variant)` followed by `compute_regenie2_binary_chunk_from_chromosome_state`.
- Direct variant-major JAX kernels remain in code for focused experiments but are not used by the production callback until full-data Firth parity is solved.

Landau measurement:

| Scenario | Wall | Native delivery | JAX compute | Host to device | Engine open/validation | Notes |
|---|---:|---:|---:|---:|---:|---|
| Trusted sample-major hot | 9.929s | 7.048s | 4.904s | 1.713s | 2.553s | From `variant_major_binary_20260517T_current` |
| Variant-major direct JAX hot | 9.880s | 6.959s | 4.945s | 1.592s | 2.573s | Rejected: full-data Firth parity mismatch |
| Variant-major native + sample-major JAX hot | 9.828s | 6.955s | 4.967s | 1.589s | 2.551s | Accepted: exact full-data parity |

Interpretation:

- Variant-major native decode halves profiled native output bytes (`8.39GB` to `4.20GB`) and reduces native delivery slightly, but the full CLI path is still dominated by trusted validation and Firth/JAX compute.
- The accepted path is about flat end-to-end versus trusted sample-major (`~1%` faster in the hot run). It is not a major optimization by itself, but it creates the native delivery shape needed for later JAX/Firth work.
- The next performance item should target Firth compute or trusted-validation amortization rather than more native layout work.

### 4. Safer XLA Autotune Cache Policy

Status: implemented.

Observed failure:

`FAILED_PRECONDITION ... xla_gpu_per_fusion_autotune_cache_dir ... Device or resource busy`

Plan:

- Make XLA autotune cache opt-in.
- Keep normal JAX persistent compilation cache enabled.

Detailed plan:

1. Keep `jax_compilation_cache_dir`, `jax_persistent_cache_min_entry_size_bytes=-1`, and `jax_persistent_cache_min_compile_time_secs=0` unchanged.
2. Add `G_ENABLE_JAX_XLA_AUTOTUNE_CACHE`.
3. Default XLA auxiliary caches to `none`.
4. Enable `xla_gpu_per_fusion_autotune_cache_dir` only when:
   - `G_ENABLE_JAX_XLA_AUTOTUNE_CACHE=1`;
   - the JAX compilation cache is node-local;
   - the cache path is not on BeeGFS.
5. Add tests for default-disabled, node-local opt-in, and BeeGFS rejection.

Implementation notes:

- Environment variable: `G_ENABLE_JAX_XLA_AUTOTUNE_CACHE`
- Default: disabled
- Persistent JAX compilation caching remains enabled by default.

Interpretation:

- This avoids the observed autotune-cache `Device or resource busy` failure by default.
- Users can still opt in on node-local storage after confirming that their CUDA/JAX/XLA stack handles the auxiliary cache reliably.

### 5. Same-Process Warm And Run

Status: implemented.

Separate `regenie2-warm-cache` avoids compile cost only if the warm step is excluded from timing. A `--warm-cache-first` option on `g regenie2` would warm exact shapes and run in one Python/JAX process.

Detailed plan:

1. Add `warm_cache_first` to `api.ComputeConfig`.
2. Add `--warm-cache-first` to `g regenie2` and `g regenie2-linear`.
3. In `api.regenie2`, after JAX device configuration and input normalization, call the same warm-cache engine used by `g regenie2-warm-cache`.
4. Warm the same full and tail chunk shapes that the real run will use.
5. Keep output-run preparation after warmup, so failed warmup does not create partially initialized output directories.
6. Record warmup time as `jax_cache_warmup` in stage timings.
7. Pass `trusted_no_missing_diploid` into warmup so validation behavior matches the subsequent run.

Implementation notes:

- CLI flag: `--warm-cache-first`
- Python flag: `ComputeConfig.warm_cache_first`
- The existing `g regenie2-warm-cache` command remains available for SLURM workflows that want warmup as a separate job step.

Measurement status:

- On `landau`, same-process warm/run works and separates warmup from the measured run in stage timings.
- Example full chr22 binary trusted sample-major run: wall `46.501s`, `jax_cache_warmup` `22.719s`, hot JAX compute `4.922s`.
- This option is useful for avoiding a second Python/JAX startup when the workflow wants cache warmup and the real run in one command.

### 6. Firth Failure Shortcuts

Status: in progress.

The current binary run had 2,295 failed Firth lanes. Many failed lanes likely run to 50 iterations. Add failure-reason diagnostics first, then short-circuit obvious separation/non-convergence cases if parity remains acceptable.

Diagnostic plan:

1. Add an internal Firth failure-reason code alongside `firth_iteration_count`.
2. Keep public output schema unchanged; failure codes are only emitted through stage diagnostics.
3. Classify failed Firth lanes as:
   - numerical failure;
   - maximum-iteration non-convergence;
   - converged-but-invalid final statistic.
4. Preserve existing merge behavior: failed Firth lanes keep score-test statistics and write `TEST_FAIL`.
5. Benchmark on landau with trusted variant-major hot mode and already-validated BGEN input.

Diagnostic result:

| Scenario | Firth candidates | Failed | Numerical | Max iteration | Invalid statistic | Hot wall |
|---|---:|---:|---:|---:|---:|---:|
| Trusted variant-major hot, validation skipped | 17,938 | 2,201 | 0 | 2,201 | 0 | 7.426s |

Interpretation:

- All observed failed lanes are maximum-iteration non-convergence. There is no evidence that numerical failure handling is a meaningful optimization target for this dataset.
- The safe optimization target is reducing batch-level waiting caused by max-iteration lanes.

Candidate-grouping plan:

1. Use the existing case-control allele-count separation heuristic as a predictor for long-running lanes.
2. Reorder fixed-shape Firth candidate lanes before batching:
   - active ordinary lanes first;
   - active heuristic/separation lanes second;
   - padded inactive lanes last.
3. Keep scatter indices with each lane so output order remains unchanged.
4. Validate exact parity on the toy tests and full chr22 output before accepting the optimization.
5. Benchmark hot landau runs against grouping disabled.

Risk:

- Reordering is statistically neutral because each Firth lane is independent and results are scattered back by original variant index.

Implementation notes:

- The temporary `G_REGENIE2_BINARY_GROUP_FIRTH_CANDIDATES` comparison switch was removed after parity and performance were measured.

Landau measurement:

| Scenario | Wall | JAX compute | Firth candidates | Max-iteration failures | Full output parity |
|---|---:|---:|---:|---:|---|
| Grouping disabled hot | 7.479s | 4.959s | 17,938 | 2,305 | reference |
| Grouping enabled hot | 7.460s | 4.935s | 17,938 | 2,305 | exact |

Interpretation:

- Grouping is exact-output compatible, but only marginally faster on the measured chr22 run (`~0.02s` hot wall, `~0.024s` JAX compute).
- This suggests max-iteration lanes are not the only cause of batch-level waiting, or the heuristic does not strongly isolate the slow lanes.
- The next Firth optimization should tune batch size with the new diagnostics rather than invest more in lane ordering.

Batch-size grid result:

Status: measured, not accepted as a default change.

The original plan called for a focused Firth batch-size grid over `32,64,128,256,512`. The grid was run on landau with trusted variant-major mode, validation skipped, and candidate grouping enabled.

| Firth batch size | Hot wall | JAX compute | Firth failures | Output parity vs batch 64 |
|---:|---:|---:|---:|---|
| 32 | 9.767s | 7.228s | 2,131 | rejected |
| 64 | 7.471s | 4.938s | 2,201 | reference |
| 128 | 6.510s | 3.983s | 2,281 | rejected |
| 256 | 6.293s | 3.776s | 2,281 | rejected |
| 512 | 5.960s | 3.431s | 2,177 | rejected |

Parity notes:

- All batch sizes preserved row count, allele frequency, and `N`.
- Batch sizes other than `64` changed thousands of Firth beta/SE/chi-square/log10p values and thousands of `EXTRA` labels.
- Therefore `DEFAULT_FIRTH_BATCH_SIZE` remains `64`.

Interpretation:

- Larger batches are faster, but batch size currently affects the Firth numerical trajectory enough to change output classifications. This is not acceptable for the default workflow.
- Any future larger-batch optimization must first make Firth numerics batch-size invariant, likely by refactoring the batched IRLS solve and convergence logic rather than only changing the environment default.

Block-math refactor result:

Status: implemented as an experimental switch, rejected as default.

The first Firth math refactor removed repeated `full_design_matrix` materialization from the IRLS loop. It computes leverage, adjusted scores, and Hessian blocks directly from covariate and genotype parts. The default path remains the previous full-design-matrix math because full-data parity was not preserved.

Implementation notes:

- Environment variable: `G_REGENIE2_BINARY_USE_BLOCK_FIRTH_MATH=1`
- Default: disabled
- Public output schema unchanged.

Landau measurement:

| Scenario | Hot wall | JAX compute | Firth failures | Output parity vs accepted grouping-enabled run |
|---|---:|---:|---:|---|
| Full-design-matrix math | 7.460s | 4.935s | 2,305 | reference |
| Experimental block math | 6.655s | 4.123s | 2,227 | rejected |

Parity notes:

- Row count, allele frequency, and `N` matched.
- Experimental block math changed thousands of Firth beta/SE/chi-square/log10p values and `3,860` `EXTRA` labels.

Interpretation:

- The block formulation is faster, but the current implementation changes enough floating-point behavior to alter Firth convergence and output classification.
- It is useful as a profiling probe, but not acceptable as a default optimization until the Firth solver is made numerically invariant across equivalent formulations.

### 7. Rust-Backed Sample Alignment

Status: implemented for the native BGEN path.

The native BGEN path still enters Python/Polars for phenotype and covariate alignment before chunk delivery. Earlier timings showed several seconds of setup around alignment, prediction loading, and output preparation. The first alignment slice keeps sample identifier resolution in Python, then moves the TSV join/filter/recode/design-matrix construction to Rust.

Detailed plan:

1. Keep `.sample` and embedded BGEN sample identifier resolution unchanged in Python.
2. Add `_core.align_sample_data(...)` that accepts already-resolved sample indices, family identifiers, and individual identifiers.
3. Parse phenotype and covariate TSV files in Rust.
4. Match the existing Python semantics:
   - join by `IID` only;
   - ignore `FID` for matching;
   - infer covariates from non-`FID`/`IID` columns when names are not supplied;
   - drop rows with null phenotype or selected covariates;
   - sort aligned rows by `sample_index`;
   - add an intercept column;
   - recode binary phenotypes from `1/2` to `0/1`;
   - preserve duplicate join expansion where duplicate `IID` rows exist.
5. Convert the native result into the existing `AlignedSampleData` dataclass so downstream prediction loading, JAX state construction, and output schema stay unchanged.
6. Enable the path by default after payload parity is proven and remove the temporary comparison switch.

Implementation notes:

- The temporary `G_REGENIE2_RUST_SAMPLE_ALIGNMENT` comparison switch was removed after the Rust path became the default.
- Public CLI/API unchanged.
- External Oxford `.sample` parsing now also uses Rust.
- Embedded BGEN samples still enter through Python as string identifiers before calling the Rust alignment core.
- Prediction-source alignment remains a separate optimization candidate after this path is measured.

Validation plan:

- Unit parity against Python alignment for continuous and binary traits.
- Pipeline dispatch test proving the native alignment path is used only when the internal switch is enabled.
- Landau timing with stage diagnostics to compare `sample_phenotype_covariate_alignment` before and after enabling the switch.

Initial landau smoke measurement:

| Scenario | Variant limit | Alignment stage | Python API entry | Output note |
|---|---:|---:|---:|---|
| Python/Polars alignment | 1,000 | 1.936s | 28.262s | reference process |
| Rust TSV alignment, Python `.sample` parsing | 1,000 | 0.415s | 23.040s | alignment payload exact |
| Rust TSV alignment and Rust `.sample` parsing | 1,000 | 0.189s | 22.359s | current default path |

Interpretation:

- The measured alignment stage is `1.75s` faster on the smoke run after moving external `.sample` parsing too, a `~90%` reduction for this setup slice.
- Direct payload comparison on the real chr22 sample, phenotype, and covariate inputs matched exactly for sample indices, family identifiers, individual identifiers, binary phenotype vector, covariate names, and covariate matrix.
- Separate GPU process outputs still showed small Firth-level differences on the smoke run. That is consistent with the already-observed cross-process Firth variability; the alignment payload itself matched exactly, so the Rust path is now the default with an environment fallback.
