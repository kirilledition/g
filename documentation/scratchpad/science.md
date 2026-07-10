# Science Notes

> Internal scratchpad. Not user docs. May stale.

## Binary REGENIE Step 2

Binary Step 2 has 2 null models:

1. Ordinary covariate-only logistic null: score tests + candidate selection.
2. Covariate-only Firth null: approximate Firth offsets.

Default `--bt --firth --approx` = scalar approximate Firth, not full covariate-plus-genotype Firth.

Internal correction codes distinguish score success/failure from approximate-Firth success/failure.

### Binary Score Test

For phenotype `y`, covariates `X`, LOCO offset `o`, genotype `g`:

```text
eta0 = X * alpha + o
p0 = logistic(eta0)
W0 = p0 * (1 - p0)
r = y - p0
```

Residualize genotype under ordinary null:

```text
Xw = sqrt(W0) * X
gw = sqrt(W0) * g
gres_w = (I - projection(Xw)) * gw
V = gres_w' * gres_w
U = g' * r
```

Reported score stats:

```text
BETA = U / V
SE = sqrt(1 / V)
CHISQ = U^2 / V
LOG10P = -log10(P(ChiSq_1 >= CHISQ))
```

### Genotype Flip

REGENIE flips high-frequency tested alleles before association:

```text
if sum(g) > sample_count:
    g_test = 2 - g
    flip final beta sign back
```

Must happen before score + approximate Firth.

### Approximate Firth

Approximate Firth first fits covariate-only Firth null:

```text
etaF0 = X * alpha_firth + o
pF0 = logistic(etaF0)
null_firth_offset = etaF0
```

Each candidate runs scalar approximate Firth with this offset. Never reuse ordinary logistic offset.

### Binary Parity Pitfalls

- Candidate selection can be right while corrected stats wrong.
- Ordinary null offset != approximate-Firth null offset.
- Starting scalar Firth from score beta broke convergence.
- Null Firth retry order matters.
- Coarse float32 floors can change correction results.
- Sparse fast paths apply only to already-selected Firth lanes.
- Genotype flipping must precede residualization/correction.
- `EXTRA=FIRTH` is not REGENIE-compatible final output for success rows.
- JAX branch dtypes need discipline.

### Binary Implementation Rules

- Keep full-chunk overflow correction in separate executable if common path slows.
- Keep fixed-shape tiny/small/bounded candidate tiers.
- Multi-trait candidate capacity comes from flattened trait-variant lanes.
- Preserve overflow semantics + failure labels across single/multi-trait paths.
- Invalid binary score stats should emit NaN public stats.
- Exact Firth without `--approx` and SPA = separate science/parity projects.

## Linear REGENIE Step 2

Original REGENIE quantitative Step 2 dense score tests residualize genotype against covariates, use LOCO-adjusted residuals, report `BETA`, `SE`, `CHISQ`, `LOG10P`.

Strict quantitative `g` algebra:

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

Matches REGENIE when sample set, LOCO residual, genotype coding, output precision match.

Current quantitative compute/output = float32. No runtime dtype switch. Future float64 output needs schema, manifest, writer, resume, parity.

## Debug Anchors

Binary:

- allele count, frequency, MAC, INFO, sparse flag;
- genotype flip decision;
- ordinary null probabilities and residuals;
- score `U`, `V`, `BETA`, `SE`, `CHISQ`, `LOG10P`;
- Firth null coefficients and offset;
- scalar Firth iteration count, convergence, and failure label;
- final output `EXTRA` rendering.

Linear:

- allele count, frequency, MAC, INFO;
- genotype normalization and projected genotype sum of squares;
- covariance with adjusted phenotype residual;
- null MSE and residual summaries;
- `BETA`, `SE`, `CHISQ`, `LOG10P`, and validity.

Classify deltas: branch, sample/mask, LOCO alignment, float32 compute, float32 output truncation, original REGENIE text formatting.
