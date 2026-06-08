# Binary REGENIE Step 2 Learning Notes

This document is for engineers who need to maintain or extend binary REGENIE
step 2 in `g`. It explains the statistical model, the original REGENIE
algorithm, the computational choices in this repository, and the parity traps
that came up while matching REGENIE approximate Firth behavior.

The most important mental model is this:

1. Score-test candidate selection is based on the ordinary covariate-only
   logistic null model.
2. Approximate Firth correction is based on a separate covariate-only Firth
   null model, used as an offset.
3. The default `--bt --firth --approx` path is scalar approximate Firth, not
   full covariate-plus-genotype Firth.
4. The implementation is performance-oriented: Rust streams and summarizes
   genotype chunks, Python orchestrates, JAX runs fixed-shape kernels, and Rust
   writes Arrow/Parquet output.
5. Successful Firth or SPA corrections are an internal event, not an output
   label. REGENIE-compatible final `EXTRA` is null for successful rows and
   `TEST_FAIL` only for failed correction rows.

## Scope

This document focuses on additive binary trait association in step 2:

- Ordinary score test for all variants.
- Approximate Firth correction for variants whose score-test p-value crosses
  the configured threshold.
- REGENIE-compatible output semantics for `BETA`, `SE`, `CHISQ`, `LOG10P`, and
  `EXTRA`.

It does not cover quantitative step 2, exact binary Firth without `--approx`,
or SPA beyond how the code reserves output labels for it. Exact Firth and SPA
are not implemented as public binary fallback paths in `g` at the time of this
writing.

## Original REGENIE

In this repository, "original REGENIE" means the patched C++ reference under
`reference/regenie-patched`. The key files and functions are:

- `reference/regenie-patched/src/Geno.cpp`
  - `flip_geno`: recodes high-frequency tested alleles to the minor allele
    before association testing.
  - `check_sparse_G`: chooses sparse genotype storage from the count of zero
    entries after REGENIE-style coding.
- `reference/regenie-patched/src/Step2_Models.cpp`
  - `compute_score_bt`: binary trait score-test loop.
  - `check_pval_snp`: decides whether a variant needs Firth or SPA correction.
  - `run_firth_correction_snp`: dispatches exact or approximate Firth.
  - `fit_approx_firth_null`: fits the covariate-only null Firth model used by
    approximate Firth.
  - `fit_firth_logistic_snp_fast`: scalar approximate Firth for one variant.
  - `fit_firth_pseudo`: first scalar pseudo-Firth attempt.
  - `fit_firth`: scalar Newton-Raphson fallback for approximate Firth.
- `reference/regenie-patched/src/Step1_Models.cpp`
  - `get_pvec`: logistic probability clipping. REGENIE clips eta at
    `ETAMINTHR=-30` and `ETAMAXTHR=30`, with probability endpoints controlled
    by `numtol_eps`.
- `reference/regenie-patched/src/Regenie.hpp`
  - Default tolerances and iteration counts, including
    `numtol_eps = 10 * double_epsilon`, `niter_max_firth_null = 1000`, and
    `maxstep_null = 25`.

Original REGENIE is a C++/Eigen/OpenMP streaming CPU implementation. It holds
mutable per-thread scratch state in objects such as `data_thread`, reads a
variant block, tests each phenotype, and appends text summary-statistic lines.
`g` instead separates those concerns into native genotype streaming, JAX compute
kernels, structured result objects, and a native output writer.

## Code Map In `g`

The main implementation is split by responsibility:

- `src/g/execution_plan.py`
  - Normalizes CLI flags. `--firth --approx` maps to
    `BinaryFallbackMethod.FIRTH_APPROXIMATE`.
  - `--firth` without `--approx` and `--spa` currently raise
    `NotImplementedError`.
- `src/g/interface/config.py`
  - Public defaults and compute settings.
- `src/g/compute/regenie2_binary/config.py`
  - Owns `BinaryKernelConfig`, the nested binary kernel policy used by the
    execution plan:
    `numerical`, `null_logistic`, `firth_candidate`, `approximate_firth`, and
    `null_firth`.
  - `GComputeConfig.use_block_firth_math` is normalized into
    `BinaryKernelConfig.approximate_firth.use_block_math`, which keeps the
    older full-model JAX Firth path behind an explicit internal switch.
- `src/g/types.py`
  - Public internal enums:
    `BinaryFallbackMethod`, `BinaryExtraCode`, `FirthFailureCode`, and
    `FirthCorrectionCode`.
- `src/g/engine/callbacks.py`
  - Bridges the native BGEN pipeline to JAX.
  - `BinaryRegenie2PipelineCallback` and `MultiBinaryRegenie2PipelineCallback`
    prepare one chromosome state, compute chunks, record diagnostics, and
    enqueue output.
  - Passes `chunk_stats.is_rare_sparse_firth_candidate` into the Firth kernel.
- `src/g/compute/regenie2_binary/state.py`
  - JAX pytree dataclasses for reusable binary state.
- `src/g/compute/regenie2_binary/result.py`
  - JAX pytree dataclasses and builders for score and corrected chunk results.
- `src/g/compute/regenie2_binary/api.py`
  - Public compute entry points for sample-major, multi-trait, and
    variant-major binary chunks.
- `src/g/compute/regenie2_binary/score.py`
  - Ordinary binary score-test kernels, genotype flipping, score variance, and
    candidate marking.
- `src/g/compute/regenie2_binary/candidates.py`
  - Fixed-shape candidate selection and batching for device-resident Firth.
- `src/g/compute/regenie2_binary/correction.py`
  - Applies device candidate corrections and merges corrected statistics back
    into score-test result arrays.
- `src/g/compute/regenie2_binary/variant_major_correction.py`
  - Direct variant-major score path and variant-major Firth helper.
- `src/g/compute/regenie2_binary/diagnostics.py`
  - Aggregates Firth candidate, convergence, failure, branch, and sparse counts.
- `src/g/compute/regenie2_binary/logistic.py`
  - Shared logistic probability helpers.
- `src/g/compute/regenie2_binary/null_logistic.py`
  - Ordinary covariate-only null logistic IRLS.
- `src/g/compute/regenie2_binary/firth/null.py`
  - Covariate-only null Firth model used by approximate Firth.
- `src/g/compute/regenie2_binary/firth/scalar_approx.py`
  - REGENIE-compatible scalar approximate Firth correction.
- `src/g/compute/regenie2_binary/firth/full_model.py`
  - Legacy full covariate-plus-genotype Firth implementation used only behind
    the internal block-math switch.
- `src/g/compute/regenie2_binary/firth/batch.py`
  - Fixed-lane batching for Firth candidates.
- `src/g/compute/regenie2_binary/firth/line_search.py`
  - Shared Firth line-search and step-halving helpers.
- `src/g/compute/regenie2_binary/firth/types.py`
  - Firth solver state and result containers.
- `src/genotype/preprocess.rs`
  - Native chunk summaries, including allele frequency, minor allele count,
    sparse flags, and rare-sparse Firth eligibility.
- `src/output/schema.rs`, `src/output/writer.rs`, `src/output/finalization.rs`
  - Arrow IPC and final Parquet schemas, `EXTRA` rendering, chunk metadata, and
    final materialization.

## End-To-End Data Flow

The binary step 2 data flow is:

1. CLI and config parsing produce a `BinaryCorrectionPlan` and
   `BinaryKernelConfig`.
2. The native BGEN engine decodes variants into chunks and computes per-variant
   summaries.
3. The callback builds a reusable binary state from covariates and phenotype.
4. When chromosome changes, the callback fetches the LOCO prediction vector and
   calls `prepare_regenie2_binary_chromosome_state`.
5. The chromosome state stores:
   - Ordinary null logistic coefficients.
   - Ordinary fitted probabilities and score residuals.
   - Weighted covariate projection state for score tests.
   - Null Firth coefficients and `null_firth_offset` when correction is enabled.
6. Each genotype chunk is sent to JAX.
7. JAX computes the score test for every variant.
8. `build_extra_code` marks variants requiring correction as internal
   `BinaryExtraCode.FIRTH`.
9. `apply_device_candidate_corrections_firth` batches candidate lanes and runs
   scalar approximate Firth on device.
10. Corrected statistics replace score-test statistics for successful
    candidates. Failed candidates become internal `TEST_FAIL`.
11. The callback transfers arrays back to host and sends them to the Rust writer.
12. The writer maps internal `EXTRA` codes to REGENIE-compatible strings:
    score, Firth, and SPA successes become null; only `TEST_FAIL` becomes the
    string `"TEST_FAIL"`.

For multi-trait binary runs, the same model is vectorized trait-major. The
chromosome state has one row per trait for probabilities, residuals, null
Firth offsets, and diagnostics.

## Mathematical Model

Let:

- `y_i` be a binary phenotype in `{0, 1}`.
- `X_i` be the covariate row, including intercept.
- `o_i` be the LOCO offset from step 1 predictions.
- `g_i` be the genotype dosage for the tested allele, after any REGENIE-style
  recoding when applicable.
- `p_i = logistic(eta_i)`.

The ordinary null logistic model is:

```text
eta0_i = X_i * alpha + o_i
p0_i = logistic(eta0_i)
W0_i = p0_i * (1 - p0_i)
r_i = y_i - p0_i
```

`fit_null_logistic_coefficients` fits `alpha` by IRLS. This state is used for
the score test and candidate selection.

### Score Test

The score test residualizes genotype against covariates under the ordinary
null logistic model. Define:

```text
Xw = sqrt(W0) * X
gw = sqrt(W0) * g
P_Xw = Xw * (Xw' Xw)^(-1) * Xw'
gres_w = (I - P_Xw) * gw
V = gres_w' * gres_w
U = g' * r
```

The score statistics are:

```text
beta_score = U / V
se_score = sqrt(1 / V)
chisq_score = U^2 / V
LOG10P = -log10(P(ChiSq_1 >= chisq_score))
```

`g` stores the reusable projection as a Cholesky-whitened
`weighted_genotype_projection_matrix` in `Regenie2BinaryChromosomeState`.
This avoids re-solving the covariate projection from scratch for every variant.

### REGENIE-Style Genotype Flip

Original REGENIE tests the minor-allele coding when the tested allele has mean
dosage greater than 1:

```text
if sum(g) > sample_count:
    g_test = 2 - g
    flip beta sign back before output
else:
    g_test = g
```

This happens before the score test and before scalar approximate Firth. The
final `BETA` is returned on the original A1 allele scale. This is easy to miss:
candidate selection can look correct while high-frequency A1 variants have
wrong `BETA` signs or small but unacceptable numeric drift.

The relevant helper is `build_regenie_flipped_genotypes`.

### Null Firth Model

Approximate Firth does not reuse the ordinary logistic null offset. It first
fits a covariate-only Firth logistic model:

```text
etaF0_i = X_i * alpha_firth + o_i
pF0_i = logistic(etaF0_i)
```

Firth logistic regression maximizes the penalized log-likelihood:

```text
loglik_penalized(beta) = loglik(beta) + 0.5 * log |I(beta)|
```

Equivalently, in deviance form:

```text
deviance_penalized(beta) = logistic_deviance(beta) - log |I(beta)|
```

The adjusted score uses the diagonal leverage values `h_i`:

```text
U_firth = X' * (y - p + h * (0.5 - p))
```

`fit_covariate_only_firth_null_model` follows REGENIE's retry structure:

1. Start from ordinary null logistic coefficients.
2. Retry from a zero-like start with intercept adjusted for mean LOCO offset.
3. Retry from that zero-like start with smaller maximum step and more
   iterations.
4. Retry from original coefficients with score-increase checking disabled.

Current constants match REGENIE defaults:

```text
FIRTH_NULL_MAXIMUM_ITERATIONS = 1000
FIRTH_NULL_GRADIENT_TOLERANCE = 50e-6
FIRTH_NULL_MAXIMUM_STEP_SIZE = 25
fallback iterations = 5 * 1000
fallback maximum step = 25 / 5
```

The output of this fit is stored in the chromosome state as:

```text
null_firth_coefficients
null_firth_offset = X * null_firth_coefficients + LOCO offset
null_firth_penalized_log_likelihood
```

The score-test residuals still use the ordinary null logistic model. Only the
scalar approximate Firth correction uses `null_firth_offset`.

### Scalar Approximate Firth

Original REGENIE's `--firth --approx` does not fit a full model with covariates
and genotype for every corrected variant. It residualizes/scales the genotype
once and then fits a scalar one-parameter Firth model:

```text
eta_i(beta) = null_firth_offset_i + beta * g_resid_i
```

The residualized genotype used here is:

```text
G_w = sqrt(W_score) * G
G_resid_w = (I - P_Xw) * G_w
g_resid = G_resid_w / sqrt(W_score)
```

This is implemented in
`residualize_and_scale_genotypes_for_approximate_firth`.

At a scalar beta, the core quantities are:

```text
p_i = clipped_logistic(offset_i + beta * g_i)
w_i = p_i * (1 - p_i)
I_beta = sum(active_i * g_i^2 * w_i)
h_i = active_i * g_i^2 * w_i / I_beta
deviance_penalized = non_active_deviance
                      + logistic_deviance(active samples)
                      - log(I_beta)
y_star_i = y_i + h_i * (0.5 - p_i)
score_beta = sum(active_i * g_i * (y_star_i - p_i))
```

`compute_scalar_firth_components` computes these values.

The correction flow in `fit_single_variant_regenie_approximate_firth` is:

1. Start with `beta = 0`. This matches REGENIE's default `bstart=0`.
2. Run scalar pseudo-Firth first.
3. If pseudo-Firth fails and the rare-sparse path used a nonzero warm start,
   try Newton-Raphson from zero. In normal current usage the warm start is zero,
   so this branch is mostly a compatibility guard.
4. If still needed, run scalar Newton-Raphson from the warm start.
5. Report `BETA`, `SE`, LRT-based `CHISQ`, and `LOG10P` from the selected
   successful branch.

The pseudo-Firth branch mirrors REGENIE's `fit_firth_pseudo`:

- It uses a pseudo-response `y_star`.
- The inner scalar logistic step is capped at maximum absolute step 5.
- It fails if the step size increases.
- It checks beta movement between iterations 14 and 15.
- It computes `SE = sqrt(1 / I_beta)` at the final beta.

The Newton-Raphson branch mirrors REGENIE's scalar `fit_firth`:

- It uses the modified scalar score.
- It caps the absolute beta step by `maxstep`.
- It uses line search against decreasing penalized deviance.
- It computes the likelihood-ratio statistic as:

```text
CHISQ = deviance_null - deviance_at_beta_hat
```

### Probability Clipping

There are two probability helpers:

- `compute_logistic_probability` uses a simple internal floor for ordinary
  logistic null fitting.
- `compute_regenie_logistic_probability` uses REGENIE-style clipping:

```text
eta < -30 -> eps / (1 + eps)
eta >  30 -> 1 / (1 + eps)
eps = 10 * double_epsilon
```

Approximate Firth uses the REGENIE clipping path. Using a float32-oriented
floor here is enough to perturb corrected statistics.

## Sparse Rare-Variant Correction

Original REGENIE has a carrier-only fast path for rare sparse variants:

```text
if dt_thr->is_sparse && mac < 50:
    use carrier indices only
```

`g` mirrors that through native metadata:

- `src/genotype/preprocess.rs` computes `is_sparse_candidate` from zero density
  after REGENIE-style allele flipping.
- It computes `minor_allele_count = min(allele_count, reference_allele_count)`.
- It sets `is_rare_sparse_firth_candidate` when the sparse flag is true and
  minor allele count is below the rare-sparse threshold.
- The callback passes that boolean vector to the JAX correction kernel.

Inside scalar Firth, `sparse_correction=True` means:

- Active samples are carriers only, using `raw_genotype > 1e-4`.
- The null deviance is still the full-sample null deviance.
- Non-carrier null deviance is carried as `non_active_deviance`.
- The scalar genotype information and score are computed over active samples.

Do not use a case-control separation heuristic to enable carrier-only Firth.
The separation heuristic may be useful for lane ordering or initializers, but
it is not equivalent to REGENIE's native sparse rare-variant condition.

## Computational Choices

The implementation is designed to keep the heavy loops device-friendly and to
avoid per-variant Python work.

### Chunked Native Decode

Rust decodes BGEN chunks and computes summary metadata before JAX sees the
data. This keeps I/O, validation, missing-value handling, allele frequency,
INFO, sparse flags, and chunk metadata close to the native data path.

### Reused Chromosome State

The LOCO offset changes by chromosome, not by variant. `g` prepares a cached
chromosome state once per chromosome:

- Ordinary null logistic fit.
- Score residual and Bernoulli weights.
- Weighted covariate projection.
- Null Firth fit and null Firth offset when needed.

Every chunk on that chromosome reuses this state.

### Fixed-Shape JAX Candidate Batching

JAX compilation works best with stable shapes. Firth candidates are therefore
converted to padded fixed-size batches:

- `build_device_firth_batch_plan` finds candidate indices with fixed capacity.
- The code uses `firth_batch_size` lanes per batch.
- If candidate count exceeds `firth_candidate_capacity`, the kernel falls back
  to full chunk capacity.
- Inactive padded lanes are skipped with masks and empty result placeholders.

This avoids host callbacks and per-candidate Python loops.

### Scalar Approximate Firth Instead Of Full-Model Firth

Full Firth over `covariates + genotype` needs repeated factorizations of a
`(covariate_count + 1) x (covariate_count + 1)` information matrix per variant
and per iteration. That is expensive and, more importantly, it is not
REGENIE's `--firth --approx` algorithm.

The scalar approximate path instead:

- Reuses the covariate projection from the score-test null.
- Fits only one scalar beta per candidate.
- Computes scalar information as a sum.
- Uses the null Firth model as an offset.

That is both closer to REGENIE and much cheaper.

### Float64 Where It Matters

The public output schema stores `BETA`, `SE`, `CHISQ`, and `LOG10P` as
float32, matching the existing writer schema and REGENIE-compatible output
target. `write_regenie2_native_chunk_with_optional_timing()` narrows any
higher-precision materialized arrays to float32 immediately before the Rust
writer call. Internally, scalar Firth and null Firth use float64. This matters
for:

- REGENIE endpoint probability clipping.
- Penalized deviance differences.
- `log(I_beta)` and Cholesky log determinants.
- Rare-variant corrected statistics.

The runtime always configures JAX with `jax_enable_x64=True`.

### Output Writer

Rust writes chunk Arrow IPC files, then finalizes to Parquet. String columns in
the IPC chunk schema are UTF-8 rather than dictionary arrays. Parquet still
enables dictionary encoding for selected low-cardinality columns, including
`EXTRA`.

The writer maps internal binary `extra_code` values as:

```text
0 SCORE       -> null
1 FIRTH       -> null
2 SPA         -> null
3 TEST_FAIL   -> "TEST_FAIL"
```

This is deliberate REGENIE output compatibility, not a loss of diagnostic
information. Internal diagnostics still retain Firth branch and failure counts.

## Differences From Original REGENIE

The goal is output parity for supported modes, not identical implementation
structure.

Important differences:

- Original REGENIE is C++/Eigen/OpenMP and writes text summary statistics.
  `g` uses Rust for genotype streaming/output and JAX for compute kernels.
- Original REGENIE uses mutable per-thread scratch objects. `g` uses immutable
  JAX dataclasses and returns arrays for each chunk.
- Original REGENIE processes candidate corrections inside the per-variant loop.
  `g` batches candidates into padded device arrays.
- Original REGENIE stores sparse genotypes as Eigen sparse vectors. `g` keeps
  dense JAX arrays but switches scalar Firth active samples to carriers only
  when native metadata says the variant is rare sparse.
- Original REGENIE's approximate Firth naturally runs in double precision.
  `g` keeps output float32 but uses float64 inside Firth math.
- Original REGENIE text output has null `EXTRA` for successful corrected rows.
  `g` retains internal success codes until rendering, then maps them to null.
- `g` still contains a full-model JAX Firth implementation for experiments
  behind `BinaryKernelConfig.approximate_firth.use_block_math=True`, currently
  sourced from `GComputeConfig.use_block_firth_math`. The public
  REGENIE-equivalent `--bt --firth --approx` path uses scalar approximate
  Firth.

## Mistakes And Lessons From The Parity Work

These are the traps that caused real mismatches.

### Candidate Selection Can Be Right While Corrected Stats Are Wrong

The score-test path and correction path share inputs but use different null
states. Candidate counts matched before corrected `BETA` and `LOG10P` matched.
Always inspect corrected variants separately from score-only variants.

### Ordinary Null Offset Is Not The Approximate-Firth Offset

The first scalar Firth implementation used:

```text
ordinary offset = X * alpha_logistic + LOCO
```

REGENIE uses:

```text
null Firth offset = X * alpha_firth + LOCO
```

That difference was enough to leave visible drift in corrected statistics.
The fix was to carry `null_firth_coefficients` and `null_firth_offset` in the
chromosome state.

### Starting Scalar Firth From Score Beta Was Wrong

It is tempting to warm-start scalar Firth from the score-test beta. REGENIE's
default `fit_firth_logistic_snp_fast` starts at `bstart=0` except for special
HTP output cases. Starting from score beta changed branch behavior and final
statistics. The default path now passes a zero warm start.

### Null Firth Retry Order Matters

REGENIE's `fit_approx_firth_null` retry sequence is not just "run Firth until
it converges." It changes the start, maximum step, iteration count, and
score-increase checking in a specific order. Matching that order removed null
offset differences that propagated into every corrected candidate.

### Float32 Floors Were Too Coarse

Using `1e-6` clipping or float32-oriented probability floors inside Firth
perturbed rare-variant deviance and LRT values. REGENIE uses eta clipping at
`[-30, 30]` and `numtol_eps = 10 * double_epsilon`. Approximate Firth now uses
that logic.

### Sparse Fast Path Was Over-Applied

A separation heuristic can identify hard variants, but it is not REGENIE's
rare-sparse condition. Carrier-only scalar Firth should be used only when
native metadata says the variant is sparse after REGENIE-style coding and
minor allele count is below the threshold.

### Genotype Flip Has To Happen Before Residualization

For high-frequency A1 variants, REGENIE flips genotype coding before the score
test and before approximate Firth, then flips beta back for output. Applying
the flip too late or only to the final beta leaves numeric mismatches because
the residualized genotype itself changes.

### `EXTRA=FIRTH` Is Not REGENIE-Compatible Final Output

Internal `BinaryExtraCode.FIRTH` means "this row was successfully corrected."
Original REGENIE final output leaves `EXTRA` empty for successful corrections.
Only failed correction rows print `TEST_FAIL`.

### Arrow Dictionary Arrays Were A Bad Intermediate Format

Using dictionary arrays directly in chunk IPC made full-run schema replacement
fragile. The stable intermediate schema uses UTF-8 string columns. Parquet
writer properties still enable dictionary encoding for final storage.

### JAX Branch Dtypes Need Discipline

Once x64 Firth math is enabled, empty and padded branches must return the same
dtypes as active branches. Otherwise JAX tracing can fail in paths that only
show up for chunks with no candidates or padded inactive lanes.

## Validation History

The parity endpoint used during this work was:

```bash
REGENIE_BIN=/mnt/beegfs/kirill/Projects/g/.tools/bin/regenie \
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
uv run python scripts/benchmark_regenie_comparison.py \
  --only-binary-step2 \
  --cpu-only \
  --variant-limit 1000 \
  --output-dir /mnt/beegfs/kirill/Projects/g/data/benchmarks/firth_equivalence_validation_v2_1k
```

Acceptance for the 1k run was:

```text
merged_variants = 1000
extra_match_rate = 1.0
BETA max error = 6.77e-6
LOG10P max error = 1.43e-5
```

The full chromosome 22 run used the same command without `--variant-limit`.
Acceptance at the end of the parity work was:

```text
merged_variants = 418943
extra_match_rate = 1.0
BETA max error = 9.21e-5
LOG10P max error = 1.60e-4
REGENIE corrections = 17,938
REGENIE correction failures = 0 / 17,938
```

These numbers are useful as historical anchors. Future changes should rerun
validation rather than assuming the numbers remain true.

## Recommended Test And Validation Checklist

For focused Python parity tests:

```bash
uv run pytest \
  tests/test_regenie2_binary.py \
  tests/test_regenie2_binary_diagnostics.py \
  tests/test_regenie_binary_correction_contract.py \
  tests/test_regenie2_binary_scalar_firth.py \
  -q
```

For API, CLI, pipeline, and comparison-script coverage:

```bash
uv run pytest \
  tests/test_api.py \
  tests/test_cli.py \
  tests/test_regenie2_pipeline.py \
  tests/test_regenie_comparison_scripts.py \
  -q
```

For output and setup regressions that affected the parity work:

```bash
uv run pytest tests/test_io_output.py tests/test_jax_setup.py -q
```

For local quality gates:

```bash
just lint-local
just typecheck-local
cargo fmt --check
cargo clippy --lib -- -D warnings -W clippy::pedantic
git diff --check
```

For runtime parity against REGENIE:

```bash
REGENIE_BIN=/mnt/beegfs/kirill/Projects/g/.tools/bin/regenie \
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
uv run python scripts/benchmark_regenie_comparison.py \
  --only-binary-step2 \
  --cpu-only \
  --variant-limit 1000 \
  --output-dir /mnt/beegfs/kirill/Projects/g/data/benchmarks/firth_equivalence_validation_next_1k
```

Then run the full chr22 validation:

```bash
REGENIE_BIN=/mnt/beegfs/kirill/Projects/g/.tools/bin/regenie \
GWAS_ENGINE_DATA_DIR=/mnt/beegfs/kirill/Projects/g/data \
uv run python scripts/benchmark_regenie_comparison.py \
  --only-binary-step2 \
  --cpu-only \
  --output-dir /mnt/beegfs/kirill/Projects/g/data/benchmarks/firth_equivalence_validation_next_full
```

Expected high-level criteria:

- Candidate count matches REGENIE.
- `extra_match_rate` is `1.0`.
- Corrected `BETA`, `SE`, `CHISQ`, and `LOG10P` are within float32 output
  tolerance.
- REGENIE reports `17,938` Firth corrections and `0` failed corrections on the
  current chr22 benchmark data.

## Debugging Recipes

### Find Whether The Error Is Score Or Correction

Split mismatches by output `EXTRA` and p-value threshold:

- Score-only rows test ordinary null logistic, genotype flipping, projection,
  and chi-square conversion.
- Corrected rows test null Firth, scalar approximate Firth, rare-sparse logic,
  branch selection, and output rendering.

If candidate counts match but corrected stats do not, start with
`null_firth_offset`, scalar start beta, probability clipping, and sparse flags.

### Inspect A Single Variant

Useful source entry points:

- `prepare_regenie2_binary_chromosome_state`
- `compute_regenie2_binary_score_test_chunk_from_chromosome_state`
- `build_regenie_flipped_genotypes`
- `residualize_and_scale_genotypes_for_approximate_firth`
- `fit_single_variant_regenie_approximate_firth`
- `compute_scalar_firth_components`

For a candidate mismatch, compare these values against instrumented REGENIE:

```text
variant ID
raw allele count
flip mask
minor allele count
sparse flag
carrier count
score beta
score variance
null logistic offset checksum
null Firth offset checksum
scalar bstart
pseudo-Firth branch result
Newton-Raphson branch result
final beta, SE, LRT, LOG10P
failure state
```

The repository also includes a small harness for capturing these values from
the active `g` implementation:

```bash
uv run python scripts/debug_binary_regenie_parity.py \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --pheno-file data/pheno_bin.txt \
  --pheno-col phenotype_binary \
  --covar-file data/covariates.txt \
  --covar-col-list age,sex \
  --pred data/baselines/regenie_step1_pred.list \
  --variant-id rs545553139 \
  --output-json data/benchmarks/binary_debug_rs545553139.json
```

Pass `--regenie-debug-jsonl` with JSONL emitted by an instrumented REGENIE
build to get numeric field-level diffs for common keys. The script does not
modify public `g regenie` behavior; it streams selected variants from BGEN and
emits debug JSON for score-test, sparse, null-model, and scalar Firth internals.

### Instrument Original REGENIE Temporarily

If local code inspection is not enough, patch the reference REGENIE source in a
disposable worktree and print diagnostics from:

- `fit_approx_firth_null`
- `fit_firth_logistic_snp_fast`
- `fit_firth_pseudo`
- scalar `fit_firth`
- `check_sparse_G`
- `flip_geno`

Remove all instrumentation before committing `g` changes. The reference source
is not part of the product code path.

### Read Stage Timing Diagnostics

The callback records null logistic and null Firth iteration diagnostics when a
stage timing recorder is enabled. Binary chunk diagnostics include:

- Candidate count.
- Converged Firth count.
- Failed Firth count.
- Pseudo-Firth count.
- Newton-Raphson fallback counts.
- Sparse carrier-only count.

These counters are often faster to inspect than scanning the full association
output.

## Practical Rules For Future Changes

- Keep score-test candidate selection tied to the ordinary null logistic model.
- Keep scalar approximate Firth tied to the null Firth offset.
- Do not change scalar Firth start values unless matching a specific REGENIE
  mode that also changes them.
- Treat carrier-only correction as a native rare-sparse metadata condition.
- Preserve genotype flipping before residualization and flip beta back only at
  output merge time.
- Use float64 for null Firth and scalar Firth internals.
- Keep public output rendering REGENIE-compatible: successful correction is
  null `EXTRA`, failed correction is `TEST_FAIL`.
- When adding a new branch to a JAX `cond` or `scan`, check dtype and shape
  equality on active, empty, and padded-lane paths.
- If a performance optimization changes batching, first prove score-only
  parity, then corrected-row parity, then full output parity.

## Follow-Up Tracking

Open binary/Firth implementation work is tracked in Linear instead of this
learning note. The current follow-ups from the docs-task audit are
[GLA-23](https://linear.app/glaphyra/issue/GLA-23/profile-packed8-score-custom-kernel-gate)
for packed8 custom-kernel profiling and
[GLA-26](https://linear.app/glaphyra/issue/GLA-26/persist-binary-benchmark-diagnostics-in-summaries)
for richer benchmark diagnostics. Durable binary/Firth lessons from historical
task notes are consolidated in [Agent Learning](agent-learning.md).
