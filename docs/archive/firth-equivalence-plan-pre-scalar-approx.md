# Binary Firth Equivalence Plan

## Goal

Make `g --bt --firth --approx` numerically and behaviorally equivalent to REGENIE's binary approximate Firth step 2 path before optimizing the implementation.

The current profiling result shows candidate selection is already aligned, but correction behavior is not:

| Metric | REGENIE | `g` CPU | `g` GPU |
| --- | ---: | ---: | ---: |
| Firth candidates | 17,938 | 17,938 | 17,938 |
| Converged corrections | 17,938 | 12,962 | 14,959 |
| Failed corrections | 0 | 4,976 | 2,979 |

The first implementation target is correctness parity, not raw speed.

## Contract

The equivalence target is REGENIE binary step 2 with:

- `--bt`
- `--firth`
- `--approx`
- Same phenotype, covariates, predictions, BGEN filters, and correction threshold.
- Same sparse rare-variant behavior.
- Same correction failure semantics.

Acceptance requires per-variant agreement for:

- Score-test candidate selection.
- Corrected `BETA`.
- Corrected `SE`.
- Corrected `CHISQ`.
- Corrected `LOG10P`.
- `EXTRA`/failure labels.

## Source Audit Points

`g` score and candidate path:

- `src/g/compute/regenie2_binary.py`
- `src/g/compute/regenie2_binary_candidate_planning.py`

`g` current Firth path:

- `src/g/compute/regenie2_binary.py`
- `src/g/compute/regenie2_binary_variant_major.py`

REGENIE score and correction path:

- `archive/direct_association/regenie/src/Step2_Models.cpp`
- `archive/direct_association/regenie/src/SKAT.cpp`
- `archive/direct_association/regenie/src/Regenie.hpp`

Important REGENIE functions and branches:

- `compute_score_bt`
- `check_pval_snp`
- `run_firth_correction_snp`
- `fit_firth_logistic_snp_fast`
- scalar `fit_firth_pseudo`
- scalar `fit_firth`
- `apply_firth_snp`

## Main Difference

REGENIE approximate Firth does not refit the full covariate plus genotype model in the same way as the current `g` implementation.

REGENIE uses:

- Covariate effects as an offset.
- Residualized and scaled genotype for the tested variant.
- Scalar SNP-effect-only approximate Firth.
- Carrier-only fast path for sparse rare variants.
- Pseudo-Firth first.
- Scalar Newton-Raphson fallback if pseudo-Firth fails.

`g` currently uses:

- Full covariate plus genotype design in the Firth solve.
- Repeated full information matrix and Cholesky work.
- Fixed-shape JAX batched full-model correction.
- Different convergence and failure behavior.
- Float32-oriented JAX math by default.

The implementer should add a REGENIE-equivalent approximate Firth kernel rather than tune the current full-model kernel first.

## Implementation Plan

### 1. Preserve Score-Test Candidate Parity

Keep the existing score-test threshold behavior intact unless a parity test proves a mismatch.

`g` currently selects candidates with:

```text
log10_p_value > -log10(p_threshold)
```

REGENIE selects candidates with:

```text
abs(score_statistic) > z_threshold
```

For a one-degree-of-freedom score test these should be equivalent. The profiled chr22 run confirms both selected `17,938` candidates.

### 2. Implement REGENIE-Equivalent Approximate Firth

Implement a scalar approximate Firth correction matching REGENIE's `fit_firth_logistic_snp_fast` and scalar `fit_firth_pseudo` path.

The core state is:

```text
offset = covariate/null Firth offset
p = logistic(offset + beta * genotype)
W = p * (1 - p)
XtWX = sum(genotype^2 * W)
h = genotype^2 * W / XtWX
ystar = phenotype + h * (0.5 - p)
score = sum(genotype * (ystar - p))
step = score / XtWX
```

The null deviance must match REGENIE:

```text
dev0 = logistic_deviance(phenotype, logistic(offset)) - log(sum(genotype^2 * W_null))
```

The corrected statistic must match REGENIE:

```text
lrt = dev0 - dev_new
chisq = lrt
se = sqrt(1 / XtWX)
```

Return failure if:

- Iteration limit is exceeded.
- Fitted probability or weight becomes invalid.
- Beta step behavior matches REGENIE's pseudo-Firth failure checks.
- `lrt < 0`.

### 3. Match REGENIE Fallback Order

For `--firth --approx`, REGENIE uses a specific fallback order:

1. Try scalar pseudo-Firth.
2. If pseudo-Firth fails and a sparse/warm-start condition applies, retry Newton-Raphson from zero.
3. If still failing, retry Newton-Raphson from the original warm start.
4. Mark the correction as failed only if all REGENIE-equivalent attempts fail.

`g` should expose diagnostics for each branch:

- `pseudo_firth_attempt_count`
- `pseudo_firth_success_count`
- `nr_zero_start_attempt_count`
- `nr_zero_start_success_count`
- `nr_warm_start_attempt_count`
- `nr_warm_start_success_count`
- `negative_lrt_failure_count`
- `probability_failure_count`
- `max_iteration_failure_count`
- `step_failure_count`

### 4. Implement Sparse Carrier Fast Path

REGENIE has a carrier-only path for sparse rare variants. This is likely important for both parity and speed.

Implementer tasks:

- Verify native BGEN output provides all required per-variant metadata.
- Ensure `g` can identify sparse candidate variants and low-MAC variants the same way REGENIE does.
- Expose carrier indices or an equivalent compact representation from the native engine.
- Run approximate Firth on carrier-only arrays where REGENIE would do so.
- Preserve dense path behavior for variants outside the sparse rare path.

Do not assume dense full-sample Firth will match REGENIE for sparse rare variants.

### 5. Match Numeric Precision And Tolerances

REGENIE uses Eigen/C++ double precision. For parity work, use float64 in the Firth correction path.

Match these REGENIE values and semantics:

- `numtol_firth`
- `numtol_eps`
- `maxstep`
- `niter_max_firth`
- `niter_max_line_search`
- probability clipping behavior
- weight-zero failure behavior
- negative-LRT failure behavior

Only consider float32 or mixed precision after the float64 implementation passes parity.

### 6. Preserve Existing Full-Model Path Behind A Separate Option

The current full-model Firth path may remain useful as an experimental or exact-ish implementation, but it should not be the default for REGENIE `--firth --approx` equivalence.

Suggested split:

- `FIRTH_APPROXIMATE_REGENIE`: default for `--firth --approx`.
- `FIRTH_APPROXIMATE_FULL_MODEL`: current experimental full-model JAX path.

Avoid changing public CLI semantics unless needed. The external behavior of `--bt --firth --approx` should match REGENIE.

## Diagnostics And Profiling

The current `jax_compute` timing is too coarse. Add timing around:

- Binary score test.
- Candidate planning.
- Pseudo-Firth correction.
- Newton-Raphson fallback.
- Sparse carrier extraction.
- Dense correction path.
- Result merge.

Add per-run counters:

- Number of score-test candidates.
- Number of Firth candidates.
- Number of sparse-carrier corrections.
- Number of dense corrections.
- Number of pseudo-Firth successes.
- Number of NR fallback successes.
- Number of failures by reason.
- Iteration min, median, p90, p99, and max.

The chr22 run should make it obvious whether time is in pseudo-Firth, NR fallback, sparse extraction, dense correction, or result merging.

## Test Plan

### Unit Tests

Add focused tests for:

- Candidate threshold equivalence.
- Logistic deviance against REGENIE-style formulas.
- Pseudo-Firth scalar update on hand-computed toy inputs.
- Negative-LRT failure.
- Probability/weight failure.
- Max-iteration failure.
- Fallback order.
- Sparse carrier-only path.

### Golden Tests Against REGENIE

Create small deterministic fixtures:

- 10-100 samples.
- One binary phenotype.
- Intercept-only and intercept-plus-covariate cases.
- Dense common variant.
- Sparse rare variant.
- Variant that converges through pseudo-Firth.
- Variant that requires NR fallback.
- Variant that should fail.

For each fixture:

- Run archived REGENIE.
- Run `g`.
- Compare per-variant output columns.
- Compare correction labels and failure labels.
- Store expected values or generate them through a controlled REGENIE fixture runner.

### Integration Tests

Run a small BGEN integration case through both tools and compare:

- Candidate count.
- Corrected count.
- Failed count.
- Top corrected variants.
- Full per-variant statistics within tolerance.

### Full chr22 Acceptance

On the current chr22 binary profile:

- Firth candidates remain `17,938`.
- Failed corrections match REGENIE, expected `0` for this run.
- Corrected statistics match REGENIE within agreed tolerance.
- CPU binary `jax_compute` drops substantially from the current `460.42s`.
- Any remaining mismatch is reported by variant ID with branch-level diagnostics.

## Performance Work After Parity

Optimize only after equivalence is passing.

Recommended order:

1. Formula parity.
2. Failure parity.
3. Sparse carrier path.
4. Float64 vs float32 decision.
5. Batching and vectorization.
6. CPU performance tuning.
7. GPU performance tuning.

Likely performance opportunities after parity:

- Use the scalar approximate Firth formula instead of full-model Cholesky work.
- Group sparse-carrier variants separately from dense variants.
- Batch dense scalar corrections in JAX.
- Use compact carrier buffers for sparse corrections.
- Avoid fixed-shape work over inactive padded lanes where practical.
- Reuse null offsets and per-variant score-test quantities.

## Current Profiling Artifacts

Latest rebased profiling output:

- `data/profiles/regenie_profiled_full_gpu_rebased/summary.md`
- `data/profiles/regenie_profiled_full_gpu_rebased/summary.json`

Important observed timings:

- `g` binary CPU `jax_compute`: `460.42s`
- `g` binary GPU `jax_compute`: `23.71s`
- REGENIE binary wall time: about `14.84s`

The performance gap is not caused by candidate volume. It is caused by doing different and heavier correction work per candidate.
