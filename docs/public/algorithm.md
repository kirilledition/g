# Algorithm

`g` runs BGEN-backed REGENIE Step 2 association scans. It tests one marker at a
time while using chromosome-specific leave-one-chromosome-out (LOCO) predictions
from REGENIE Step 1 as fixed adjustment terms. It does not fit REGENIE Step 1;
`--pred` must point to a prediction list produced by upstream `regenie`.

The implemented statistical surface is:

| Mode | User options | What is tested |
| --- | --- | --- |
| Quantitative | `--step 2 --qt` | Additive dosage effect in a linear model after covariate and LOCO adjustment. |
| Binary score test | `--step 2 --bt` | Additive dosage effect in a logistic model, evaluated by a score test at the null model. |
| Binary approximate Firth fallback | `--step 2 --bt --firth --approx` | Score test for all variants, then approximate Firth logistic correction for score-test candidates selected by `--pThresh`. |

Recognized REGENIE options outside this surface, such as `--bed`, `--pgen`,
`--spa`, categorical covariates, and exact Firth without `--approx`, fail
clearly instead of being ignored.

## High-Level Flow

Every run follows the same execution shape:

1. Merge defaults, TOML config, and CLI flags into one execution plan.
2. Open the BGEN 1.2 source and resolve sample identifiers from the `.sample`
   file or embedded BGEN samples.
3. Align genotype samples, phenotype rows, covariate rows, and Step 1 LOCO
   predictions using `IID` or `(FID, IID)` according to `--g-sample-key-mode`.
4. Drop rows that are incomplete for the requested phenotype and covariates.
   Require Step 1 LOCO predictions for the remaining aligned samples. Binary
   phenotypes are recoded from REGENIE coding `1 = control`, `2 = case` to
   internal `0/1`.
5. Build reusable null-model state for the aligned samples.
6. Stream BGEN variants in chunks of `--bsize` variants.
7. Decode each chunk to allele-one dosages in `[0, 2]`, mean-impute missing
   genotypes for the compute matrix, and retain observed genotype counts for
   output and numerical statistics.
8. Dispatch the chunk to JAX on `--g-device cpu` or `--g-device gpu`.
9. Write Arrow, Parquet, or REGENIE-text result chunks with a run manifest and
   effective config.

Step 2 is therefore conditional on the Step 1 prediction file, the aligned
sample set, covariates, trait mode, and binary correction plan. Changing any of
those can change the statistics, not just the runtime.

## Notation

The formulas below use this notation:

| Symbol | Meaning |
| --- | --- |
| `n` | Number of aligned, complete samples for this phenotype or phenotype group. |
| `X` | `n x p` covariate design matrix. It includes the intercept that `g` adds internally. |
| `y` | Quantitative phenotype vector. |
| `z` | Binary phenotype vector after `1/2` is recoded to `0/1`. |
| `g_j` | Dosage vector for variant `j`, coded as allele-one expected allele count. |
| `l_c` | Quantitative LOCO prediction vector for chromosome `c`. |
| `o_c` | Binary LOCO offset vector for chromosome `c`. |
| `P_X` | Projection onto the column space of `X`: `X (X'X)^-1 X'`. |
| `R_X` | Residual projection: `I - P_X`. |
| `p_i` | Fitted binary null probability for sample `i`. |
| `W` | Diagonal Bernoulli variance matrix with entries `p_i (1 - p_i)`. |

The public result fields are written as `float32` in the current output schema,
even when an internal kernel uses a wider dtype for parity-sensitive work.

## Quantitative Step 2

For `--qt`, `g` uses a covariate-adjusted linear association test. The model for
one variant on chromosome `c` is:

```text
y = X alpha + l_c + g_j beta_j + error
```

The implementation avoids refitting this model from scratch for every variant.
It precomputes the covariate projection once, then reuses it across BGEN chunks.

For each phenotype:

```text
phenotype_residual = y - P_X y
```

For each chromosome:

```text
adjusted_residual = phenotype_residual - l_c
residual = R_X adjusted_residual
sigma2 = (residual' residual) / (n - p)
```

For each variant:

```text
genotype_residual = R_X g_j
genotype_variance = genotype_residual' genotype_residual
covariance = genotype_residual' residual

BETA = covariance / genotype_variance
SE = sqrt(sigma2 / genotype_variance)
CHISQ = BETA^2 / SE^2
LOG10P = -log10(Pr[ChiSquared(df=1) >= CHISQ])
```

This is the same one-degree-of-freedom additive association test that can be
viewed as a score, Wald, or single-variant least-squares update after the null
model has been projected out.

Numerical safeguards:

- If the allele-one mean dosage is greater than `1`, `g` subtracts `2` from the
  genotype vector before projection. Because the model contains an intercept,
  this constant shift does not change the residualized genotype or statistic;
  it reduces float32 cancellation for high-frequency allele-one variants.
- A variant is marked invalid when residualized genotype variance is not larger
  than `max(--g-linear-minimum-variance, raw_sum_squares *
  --g-linear-relative-variance-tolerance)`.
- A phenotype/chromosome state is invalid if the adjusted residual variance is
  not positive.

Changing `--covarCol`, `--covarColList`, `--pred`, `--phenoCol`, sample-key mode,
or multi-phenotype sample mode changes the projection or residual and therefore
can change `BETA`, `SE`, `CHISQ`, and `LOG10P`.

## Binary Score Test

For `--bt`, `g` fits a chromosome-specific logistic null model with the LOCO
offset fixed:

```text
logit(p_i) = X_i alpha + o_c,i
z_i ~ Bernoulli(p_i)
```

The null coefficients `alpha` are estimated by iteratively reweighted least
squares (IRLS). The maximum iteration count and coefficient tolerance are
controlled by `--g-binary-null-maximum-iterations` and
`--g-binary-null-coefficient-tolerance`.

After the null model is fitted:

```text
score_residual_i = z_i - p_i
W_i = max(p_i * (1 - p_i), --g-binary-minimum-variance)
```

For each variant, `g` computes a weighted genotype residual variance without
materializing a full residualized matrix:

```text
U_j = g_j' score_residual
V_j = g_j' W g_j - projection_of_g_j_onto_weighted_covariates

BETA = U_j / V_j
SE = sqrt(1 / V_j)
CHISQ = U_j^2 / V_j
LOG10P = -log10(Pr[ChiSquared(df=1) >= CHISQ])
```

Here `BETA` is the score-test effect estimate for allele-one dosage under the
null model. It is not a full per-variant logistic maximum-likelihood fit unless
the approximate Firth fallback replaces that row.

Numerical safeguards:

- Fitted probabilities are clipped by `--g-binary-minimum-probability`.
- Binary variance and information matrices use `--g-binary-minimum-variance` as
  an absolute floor.
- The score statistic is invalid when `V_j` is not larger than
  `max(--g-binary-minimum-variance, weighted_sum_squares *
  --g-binary-relative-variance-tolerance)`.
- By default, `--g-null-logistic-nonconvergence fail` aborts when a chromosome's
  null logistic model does not converge. With `warn`, the run continues, but
  nonconverged chromosome/trait rows fail the statistic mask.
- High-frequency allele-one variants may be tested internally after flipping
  genotype coding to keep rare carriers near zero; successful `BETA` values are
  restored to the public `ALLELE1` orientation.

## Binary Approximate Firth Fallback

For `--bt --firth --approx`, all variants first receive the binary score test.
Then `g` selects Firth candidates with:

```text
score_p_value < --pThresh
```

Equivalently, because output stores `-log10(p)`, candidates satisfy:

```text
LOG10P > -log10(--pThresh)
```

Lowering `--pThresh` reduces the number of Firth candidates and makes the run
closer to score-only behavior. Raising it sends more variants into the slower
correction path and can change more binary rows.

Firth correction uses the bias-reduced logistic objective:

```text
penalized_log_likelihood(theta)
  = logistic_log_likelihood(theta) + 0.5 * log(|I(theta)|)
```

where `I(theta)` is the model information matrix. This is the Jeffreys-prior
penalty used by Firth's bias-reduction method. It is useful for rare variants,
unbalanced case-control traits, and separation-prone binary models.

`g` implements the REGENIE-style approximate Firth fallback path:

1. Fit or prepare a covariate-only Firth null model for the chromosome.
2. For candidate variants, recode high-frequency variants if needed and restore
   public allele-one sign after correction.
3. Try the scalar pseudo-Firth approximation and sparse-carrier compaction paths
   when applicable.
4. Fall back through Newton-Raphson zero-start and warm-start attempts according
   to the configured iteration and line-search limits.
5. Use the penalized likelihood-ratio statistic for the corrected row:

```text
CHISQ = max(2 * (full_penalized_log_likelihood
                 - null_penalized_log_likelihood), 0)
LOG10P = -log10(Pr[ChiSquared(df=1) >= CHISQ])
```

`--firth-se` changes only the reported `SE` for successful Firth rows. When it
is enabled, `g` reports:

```text
SE = abs(BETA) / sqrt(CHISQ)
```

for corrected rows with positive `CHISQ`. This ties the standard error to the
likelihood-ratio statistic. It does not change candidate selection, the Firth
fit, `BETA`, `CHISQ`, or `LOG10P`.

Firth-specific tuning options such as `--g-firth-batch-size`,
`--g-firth-candidate-capacity`, `--g-firth-maximum-iterations`,
`--g-firth-gradient-tolerance`, `--g-firth-coefficient-tolerance`,
`--g-firth-likelihood-tolerance`, line-search limits, and sparse-carrier
thresholds should normally stay at their defaults. They affect performance,
convergence, or numerical acceptance of candidate rows, not the intended
scientific model.

## Genotype Handling

`g` reads BGEN 1.2 genotype probability records and converts them to allele-one
dosage:

```text
dosage = Pr(heterozygous) + 2 * Pr(homozygous_allele_one)
```

For trusted unphased 8-bit BGEN records, the packed probability-pair path uses:

```text
dosage = (510 - 2 * P_homozygous_reference_byte - P_heterozygous_byte) / 255
```

Missing genotype dosages are represented as `NaN` during decode. Before the
statistical kernel runs, each missing dosage is replaced by that variant's
observed mean dosage among aligned samples. The output `N`, `A1FREQ`, and
`INFO` are based on observed genotype calls:

| Field | Meaning |
| --- | --- |
| `N` | Observed genotype count for the variant after sample alignment. |
| `A1FREQ` | Observed mean allele-one dosage divided by `2`. |
| `INFO` | Observed dosage variance divided by the expected Hardy-Weinberg dosage variance, clamped to `[0, 1]`; missing calls are not counted in the expected denominator. |

The trusted fast path controlled by `--g-trusted-no-missing-diploid` assumes the
BGEN records are diploid, unphased, no-missing records that match the optimized
decoder's constraints. `--g-trusted-bgen-validation-mode` controls whether that
assumption is validated on cache miss, always validated, or assumed by the user.

## Multi-Phenotype Behavior

Multiple phenotypes can be requested with repeated `--phenoCol` flags or
`--phenoColList`.

`--g-multi-phenotype-sample-mode per-phenotype` is the default. Each phenotype
keeps its own complete-case sample set. `g` may group phenotypes that happen to
share compatible aligned samples, but the statistical semantics match separate
single-phenotype runs.

`--g-multi-phenotype-sample-mode complete-case` builds one shared complete-case
intersection across all requested phenotypes. This can reuse genotype decode and
device transfer work across traits, but it is not equivalent to per-phenotype
analysis when missingness differs across phenotypes. It changes `n`, the
covariate projection, LOCO alignment, and all downstream statistics.

## Parameter Effects

Statistical parameters:

| Option | Changes statistics? | Effect |
| --- | --- | --- |
| `--qt` / `--bt` | Yes | Selects linear quantitative or logistic binary model. |
| `--phenoCol`, `--phenoColList` | Yes | Selects the trait vector or trait matrix. |
| `--covarCol`, `--covarColList` | Yes | Changes `X`, residualization, degrees of freedom, and null model fits. |
| `--pred` | Yes | Supplies chromosome-specific LOCO predictions or offsets. Different Step 1 models produce different Step 2 adjustments. |
| `--sample`, `--g-sample-key-mode` | Yes | Changes sample identity resolution and row alignment. |
| `--g-multi-phenotype-sample-mode` | Yes | Controls whether phenotypes keep independent complete-case samples or share one intersection. |
| `--firth --approx` | Yes, for binary | Replaces selected score-test rows with approximate Firth-corrected rows. |
| `--pThresh` | Yes, for binary Firth | Sets the score-test p-value threshold for Firth candidates. |
| `--firth-se` | Yes, for reported `SE` only | Recomputes successful Firth `SE` as `abs(BETA) / sqrt(CHISQ)`. |
| `--g-score-dtype` | Can | Changes score-test compute precision. Public statistics still write as `float32`. |

Numerical policy parameters:

| Option | Effect |
| --- | --- |
| `--g-linear-minimum-variance`, `--g-linear-relative-variance-tolerance` | Decide when a residualized quantitative genotype is too close to zero variance to test safely. |
| `--g-binary-minimum-probability` | Clips binary fitted probabilities away from exact `0` and `1`. |
| `--g-binary-minimum-variance`, `--g-binary-relative-variance-tolerance` | Stabilize binary score variances and information matrices. |
| `--g-binary-null-maximum-iterations`, `--g-binary-null-coefficient-tolerance` | Control binary null logistic IRLS termination. |
| `--g-null-logistic-nonconvergence` | Chooses whether nonconverged binary null fits abort the run or warn and continue with invalid statistic rows. |
| `--g-firth-*`, `--g-null-firth-*` | Control approximate-Firth iteration limits, tolerances, line search, step halving, and sparse-carrier behavior. |

Runtime and output parameters:

| Option | Changes statistics? | Effect |
| --- | --- | --- |
| `--bsize` | No intended change | Number of variants decoded and dispatched per chunk. It affects memory, compile shape, and throughput. |
| `--threads` | No intended change | Requested native CPU thread count for Rust-owned work. |
| `--g-device` | No intended change | Selects JAX CPU or GPU execution. |
| `--g-staging-depth` | No intended change | Controls how far native chunk delivery can stage callback work. |
| `--g-bgen-decode-tile-variant-count` | No intended change | Native BGEN decode tile size. |
| `--g-gpu-genotype-format` | No intended change | Selects dosage transfer or packed8 transfer/decode for GPU-compatible paths. |
| `--g-output-format` | No | Chooses Arrow, Parquet, or REGENIE-text materialization. |
| `--g-resume`, `--g-resume-mode` | No | Reuses previously committed chunks when the manifest accepts the execution plan. |
| `--g-telemetry`, `--g-log-*` | No | Controls diagnostics and logging. Profile and trace modes can add synchronization overhead. |

Runtime parameters should not be used to change scientific conclusions. If a
runtime-only parameter changes results beyond normal floating-point tolerance,
treat it as a bug or a reproducibility finding.

## Reading Output Rows

`g` writes REGENIE Step 2-style additive test rows:

| Field | Interpretation |
| --- | --- |
| `CHROM`, `GENPOS`, `ID` | Variant identity from BGEN metadata. |
| `ALLELE0`, `ALLELE1` | Reported alleles. Effects are for `ALLELE1` dosage. |
| `A1FREQ` | Observed allele-one frequency after sample alignment. |
| `INFO` | Observed dosage INFO score as described above. |
| `N` | Observed genotype count after sample alignment. |
| `TEST` | Currently `ADD` for the additive dosage test. |
| `BETA` | Additive allele-one effect estimate for the selected mode. |
| `SE` | Standard error for the reported effect estimate. |
| `CHISQ` | One-degree-of-freedom chi-squared statistic. |
| `LOG10P` | `-log10(p)` for the chi-squared tail probability. Larger means stronger evidence. |
| `EXTRA` | Null/`NA` for ordinary successful rows; `TEST_FAIL` when the statistic or correction failed. |

Binary successful Firth rows are not currently labeled separately in the public
`EXTRA` field; the run manifest, telemetry, and binary diagnostics are the
places to inspect correction-plan behavior.

## What to Expect Operationally

- The first chunk shape for a mode may trigger JAX compilation. CPU and GPU
  runs can therefore have a warmup cost before steady-state throughput.
- GPU acceleration depends on enough work per transfer. Single-trait scans can
  be limited by BGEN decode, host-device transfer, or output writing.
- Increasing `--bsize` usually reduces per-chunk overhead but raises memory use
  and can change JAX compile shapes.
- Approximate Firth is intentionally slower than score-only binary testing
  because candidate variants require iterative correction.
- Resume checks compare the manifest against execution-plan-affecting inputs
  and settings. Use `--g-resume-mode strict` when restart correctness matters
  more than startup speed.
- Result p-values are not multiple-testing corrected. Genome-wide significance
  thresholds and downstream quality control remain the user's responsibility.

## References

- Mbatchou, J., Barnard, L., Backman, J. et al. [Computationally efficient whole-genome regression for quantitative and binary traits](https://www.nature.com/articles/s41588-021-00870-7). Nature Genetics 53, 1097-1103 (2021). DOI: [10.1038/s41588-021-00870-7](https://doi.org/10.1038/s41588-021-00870-7).
- Firth, D. [Bias reduction of maximum likelihood estimates](https://doi.org/10.1093/biomet/80.1.27). Biometrika 80(1), 27-38 (1993). DOI: [10.1093/biomet/80.1.27](https://doi.org/10.1093/biomet/80.1.27).
- Rao, C. R. [Large sample tests of statistical hypotheses concerning several parameters with applications to problems of estimation](https://cir.nii.ac.jp/crid/1361418518596307328). Mathematical Proceedings of the Cambridge Philosophical Society 44(1), 50-57 (1948). General background: [Score test](https://en.wikipedia.org/wiki/Score_test).
- Wilks, S. S. [The large-sample distribution of the likelihood ratio for testing composite hypotheses](https://doi.org/10.1214/aoms/1177732360). The Annals of Mathematical Statistics 9(1), 60-62 (1938). General background: [Likelihood-ratio test](https://en.wikipedia.org/wiki/Likelihood-ratio_test).
- BGEN Working Group. [The BGEN format, version 1.2](https://www.chg.ox.ac.uk/~gav/bgen_format/spec/v1.2.html).
- General background: [Genome-wide association study](https://en.wikipedia.org/wiki/Genome-wide_association_study), [linear least squares](https://en.wikipedia.org/wiki/Linear_least_squares), and [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression).
