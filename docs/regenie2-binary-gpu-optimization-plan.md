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
2. Keep the current sample-major API as the default until parity and performance are proven. Done.
3. Add a Python callback path for variant-major preprocessed chunks with the same `_core.VariantMetadata` and `_core.ChunkStats` objects. Done.
4. Add binary-only JAX kernels that accept `genotype_matrix_by_variant` directly:
   - score test uses variant-major matrix-vector and matrix-matrix products;
   - Firth candidate selection uses direct row gathers instead of column gathers and transposes;
   - Firth batch planning keeps the bounded candidate capacity introduced above.
5. Add a `G_REGENIE2_BINARY_VARIANT_MAJOR=1` internal switch for A/B testing. Done.
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

- Environment variable: `G_REGENIE2_BINARY_VARIANT_MAJOR=1`
- Required with current decoder: `trusted_no_missing_diploid=True` / `--trusted-no-missing-diploid`
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

Status: diagnostics-driven follow-up.

The current binary run had 2,295 failed Firth lanes. Many failed lanes likely run to 50 iterations. Add failure-reason diagnostics first, then short-circuit obvious separation/non-convergence cases if parity remains acceptable.
