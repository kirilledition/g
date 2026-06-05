# Linear REGENIE Step 2 Learning

This note records the quantitative-trait step 2 parity model used while hardening
`g` as a replacement for original REGENIE.

## Reference Algorithm

Original REGENIE runs quantitative step 2 through `compute_score_qt()` in
`archive/direct_association/regenie/src/Step2_Models.cpp`.

For the default dense score-test path, REGENIE:

1. Residualizes each genotype against `pheno_data.new_cov`.
2. Scales the residualized genotype by
   `norm(g_residual) / sqrt(n_analyzed - ncov_analyzed)`.
3. Uses the LOCO-adjusted phenotype residual matrix `yres`.
4. Computes the score numerator and denominator.
5. Reports:
   - `BETA = score / denominator * pheno_data.scf_sv`
   - `SE = BETA / score_statistic`
   - `CHISQ = score_statistic^2`
   - `LOG10P = get_logp(CHISQ)`

For strict quantitative runs, the denominator simplifies to
`scale_fac^2 * (n_analyzed - ncov_analyzed)` after genotype residualization and
scaling. In non-strict mode REGENIE uses phenotype-specific missingness masks
and a per-phenotype denominator.

## Current `g` Model

`g` computes the same score-test algebra without explicitly materializing
REGENIE's scaled residual genotype:

```text
genotype_residual = g - X @ inv(X'X) @ X' @ g
covariance = genotype_residual' @ adjusted_residual
genotype_ss = genotype_residual' @ genotype_residual
null_mse = adjusted_residual_ss / (n - covariate_count)

BETA = covariance / genotype_ss
SE = sqrt(null_mse / genotype_ss)
CHISQ = BETA^2 / SE^2
LOG10P = -log10(P(ChiSq_1 >= CHISQ))
```

This is algebraically equivalent to REGENIE's dense strict score path when the
same sample set, LOCO residual, genotype coding, and output precision are used.

## Dtype Policy

Original REGENIE uses Eigen `double` for this path, but current full-chromosome
CPU and GPU parity checks show that linear float32 compute is not the limiting
factor for replacement-grade agreement. `g` therefore keeps quantitative step 2
linear compute and output fixed to float32. There is no runtime dtype switch for
this path. The output writer persists `BETA`, `SE`, `CHISQ`, and `LOG10P` as
float32 and records that dtype in the run manifest.

## Debug Workflow

Use `scripts/debug_linear_regenie_parity.py` to capture `g` internals for one or
more variants. Use `REGENIE_QT_DEBUG_JSONL=/path/to/reference.jsonl` with the
patched local REGENIE copy to emit reference records from `compute_score_qt()`.

The debug records focus on the quantities that decide parity:

- allele count, frequency, MAC, INFO, sparse flag
- genotype normalization offset and projected genotype sum of squares
- covariance with the adjusted phenotype residual
- null MSE and adjusted residual summaries
- `BETA`, `SE`, `CHISQ`, `LOG10P`, and validity

## Acceptance Target

Quantitative parity must be judged on all reported statistics, not just `BETA`
and `LOG10P`. The comparison benchmark now records `BETA`, `SE`, `CHISQ`,
`LOG10P`, `EXTRA`, and top variant-level discrepancies.

Remaining deltas should be classified as one of:

- algorithm or branch mismatch
- float32 compute precision
- float32 output truncation
- original REGENIE output formatting
