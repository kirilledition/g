# Quickstart

| Status | Applies to | Owner |
| --- | --- | --- |
| Pre-release draft | main branch as of 2026-06-30 `g regenie` examples | Public user docs |

These examples assume you installed `g` with [Installation](installation.md) and are running commands
from the repository checkout. Replace the paths with your own BGEN, phenotype, covariate, and
upstream REGENIE Step 1 prediction files.

## Inputs You Need

`g` runs REGENIE Step 2. It does not run REGENIE Step 1.

- `--bgen`: BGEN genotype file.
- `--sample`: Required Oxford sample file providing BGEN row identities.
- `--phenoFile`: Phenotype table.
- Repeated `--phenoCol`: Phenotype column names.
- `--covarFile` and repeated `--covarCol`: Covariates when your model uses them.
- `--pred`: Step 1 prediction list produced by upstream `regenie`.
- `--out`: Output prefix. `g` writes a run directory next to this prefix.

Before running, check:

```bash
uv run g regenie --help
test -s /path/to/genotypes.bgen
test -s /path/to/regenie_step1_pred.list
```

For exact file contracts, see [Input Files](input-files.md).

## Quantitative Step 2

```bash
uv run g regenie \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --covarFile /path/to/covariates.tsv \
  --covarCol age --covarCol sex \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_quantitative_regenie2
```

## Binary Score Test

```bash
uv run g regenie \
  --bt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_binary \
  --covarFile /path/to/covariates.tsv \
  --covarCol age --covarCol sex \
  --pred /path/to/regenie_step1_pred.list \
  --out /path/to/output/g_binary_score_regenie2
```

## Binary Approximate Firth Fallback

```bash
uv run g regenie \
  --bt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_binary \
  --covarFile /path/to/covariates.tsv \
  --covarCol age --covarCol sex \
  --pred /path/to/regenie_step1_pred.list \
  --binary-fallback firth_approximate \
  --pThresh 0.01 \
  --out /path/to/output/g_binary_firth_regenie2
```

Approximate Firth is implemented but numerically sensitive. Use equivalent statistical modes when comparing results against upstream REGENIE.

See [Algorithm](algorithm.md) for the quantitative, binary score-test, and
approximate-Firth formulas behind these commands.

## Multi-Phenotype Sample Modes

You can request multiple traits in one run with repeated `--phenoCol` flags.

```bash
uv run g regenie \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous_a \
  --phenoCol phenotype_continuous_b \
  --covarFile /path/to/covariates.tsv \
  --covarCol age --covarCol sex --covarCol pc1 --covarCol pc2 \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_multi_per_phenotype
```

That command is equivalent to running two separate commands with the same flags
except one `--phenoCol` at a time (subject to random differences in I/O timing
and scheduling).

Use shared-sample mode when you explicitly want all traits on the same
intersection. Put the mode in `complete-case.toml`:

```toml
[compute]
multi_phenotype_sample_mode = "complete-case"
```

```bash
uv run g regenie \
  --config complete-case.toml \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous_a \
  --phenoCol phenotype_continuous_b \
  --covarFile /path/to/covariates.tsv \
  --covarCol age --covarCol sex --covarCol pc1 --covarCol pc2 \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_multi_complete_case
```

To see runtime knobs for this setting in config, use
`[compute] multi_phenotype_sample_mode` via
[Configuration](configuration.md#cli-to-toml-mapping).

When validating this choice, compare `sample_count` and run manifest metadata
under `attempts/<attempt>/<phenotype-output-name>/run_manifest.json` alongside
the statistical results.

## GPU Execution

Sync the normal runtime environment, which already includes CUDA-enabled JAX,
verify GPU visibility, then create `gpu.toml`:

```toml
[compute]
device = "gpu"
```

```bash
uv run g regenie \
  --config gpu.toml \
  --qt \
  --bgen /path/to/genotypes.bgen \
  --sample /path/to/genotypes.sample \
  --phenoFile /path/to/phenotypes.tsv \
  --phenoCol phenotype_continuous \
  --pred /path/to/regenie_step1_qt_pred.list \
  --out /path/to/output/g_gpu_regenie2
```

Submit GPU commands on a GPU node or through your scheduler. See [GPU and Clusters](gpu-and-clusters.md)
for cluster notes.

## Repository Fixture Data

Developers and evaluators can generate local 1000 Genomes chromosome 22 fixture data with repository
recipes:

```bash
just data-prepare
just data-baseline-qt
just data-baseline-binary
```

These commands require the development tooling described in
[Development Installation](installation.md#development-installation). Fixture paths under `data/`
are local and git-ignored.

## Output

Successful runs print the run directory and its `parts/` Parquet dataset
directory. See [Output Files](output-files.md) for the layout and schema, and
[Resume and Manifest](resume-and-manifest.md) for restart behavior.
