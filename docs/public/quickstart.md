# Quickstart

These examples assume you installed `g` with [Installation](installation.md) and are running commands
from the repository checkout. Replace the paths with your own BGEN, phenotype, covariate, and
upstream REGENIE Step 1 prediction files.

## Inputs You Need

`g` runs REGENIE Step 2. It does not run REGENIE Step 1.

- `--bgen`: BGEN genotype file.
- `--sample`: Oxford sample file when samples are not embedded or when you need explicit sample IDs.
- `--phenoFile`: Phenotype table.
- `--phenoCol` or `--phenoColList`: Phenotype column names.
- `--covarFile` and `--covarColList`: Covariates when your model uses them.
- `--pred`: Step 1 prediction list produced by upstream `regenie`.
- `--out`: Output prefix. `g` writes a run directory next to this prefix.

## Quantitative Step 2

```bash
uv run g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --covarFile /path/to/covariates.tsv \
  --covarColList age,sex \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_quantitative_regenie2 \
  --g-device cpu \
  --g-output-format parquet
```

## Binary Score Test

```bash
uv run g regenie \
  --step 2 \
  --bt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_binary \
  --covarFile /path/to/covariates.tsv \
  --covarColList age,sex \
  --pred /path/to/regenie_step1_pred.list \
  --out /path/to/output/g_binary_score_regenie2 \
  --g-device cpu \
  --g-output-format parquet
```

## Binary Approximate Firth Fallback

```bash
uv run g regenie \
  --step 2 \
  --bt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_binary \
  --covarFile /path/to/covariates.tsv \
  --covarColList age,sex \
  --pred /path/to/regenie_step1_pred.list \
  --firth \
  --approx \
  --pThresh 0.01 \
  --out /path/to/output/g_binary_firth_regenie2 \
  --g-device cpu \
  --g-output-format parquet
```

Approximate Firth is implemented but numerically sensitive. Use equivalent statistical modes when comparing results against upstream REGENIE.

## GPU Execution

Install the GPU dependency group first, then change the device:

```bash
uv run g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_gpu_regenie2 \
  --g-device gpu
```

Submit GPU commands on a GPU node or through your scheduler. See [GPU and SLURM](gpu-and-slurm.md)
for cluster notes.

## REGENIE Text Output

Use `--g-output-format regenie` to write a REGENIE Step 2-compatible
tab-separated `final.regenie` file for workflow compatibility:

```bash
uv run g regenie \
  --step 2 \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_regenie_text \
  --g-output-format regenie
```

## Repository Fixture Data

Developers and evaluators can generate local 1000 Genomes chromosome 22 fixture data with repository
recipes:

```bash
just setup-data
just setup-binary-baseline
```

These commands require the development tooling described in
[Development Installation](installation.md#development-installation). Fixture paths under `data/`
are local and git-ignored.

## Output

Successful runs print the run directory. See [Input and Output](input-output.md) for the generated layout and schema.
