# Quickstart

These examples use the local fixture-data conventions from the repository. Paths under `data/` are local and git-ignored.

## Prepare Example Data

```bash
just setup-data
```

For binary examples:

```bash
just setup-binary-baseline
```

## Quantitative Step 2

```bash
uv run g regenie \
  --step 2 \
  --qt \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --phenoFile data/pheno_cont.txt \
  --phenoCol phenotype_continuous \
  --covarFile data/covariates.txt \
  --covarColList age,sex \
  --pred data/baselines/regenie_step1_qt_pred.list \
  --out data/example_regenie2 \
  --g-output-format parquet
```

## Binary Score Test

```bash
uv run g regenie \
  --step 2 \
  --bt \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --phenoFile data/pheno_bin.txt \
  --phenoCol phenotype_binary \
  --covarFile data/covariates.txt \
  --covarColList age,sex \
  --pred data/baselines/regenie_step1_pred.list \
  --out data/example_regenie2_binary_score \
  --g-output-format parquet
```

## Binary Approximate Firth Fallback

```bash
uv run g regenie \
  --step 2 \
  --bt \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --phenoFile data/pheno_bin.txt \
  --phenoCol phenotype_binary \
  --covarFile data/covariates.txt \
  --covarColList age,sex \
  --pred data/baselines/regenie_step1_pred.list \
  --firth \
  --approx \
  --pThresh 0.01 \
  --out data/example_regenie2_binary_firth \
  --g-output-format parquet
```

Approximate Firth is implemented but numerically sensitive. Use equivalent statistical modes when comparing results against upstream REGENIE.

See [Algorithm](algorithm.md) for the quantitative, binary score-test, and
approximate-Firth formulas behind these commands.

## REGENIE Text Output

Use `--g-output-format regenie` to write a REGENIE Step 2-compatible
tab-separated `final.regenie` file for workflow compatibility:

```bash
uv run g regenie \
  --step 2 \
  --qt \
  --bgen data/1kg_chr22_full.bgen \
  --sample data/1kg_chr22_full.sample \
  --phenoFile data/pheno_cont.txt \
  --phenoCol phenotype_continuous \
  --pred data/baselines/regenie_step1_qt_pred.list \
  --out data/example_regenie2_text \
  --g-output-format regenie
```

## Direct Executable

The direct console script is also available:

```bash
g-regenie --step 2 --qt --bgen ... --phenoFile ... --phenoCol ... --pred ... --out ...
```

## Output

Successful runs print the run directory. See [Input and Output](input-output.md) for the generated layout and schema.
