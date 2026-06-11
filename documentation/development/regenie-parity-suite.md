# REGENIE Parity Suite

The pre-release parity suite is the release gate for Step 2 behavior that claims
REGENIE compatibility. It combines external upstream REGENIE golden outputs with
small internal contracts for sample semantics, resume behavior, and native BGEN
missingness handling.

The machine-readable coverage record is checked in at
`tests/parity/golden_metadata.json`. It records the upstream REGENIE version,
command lines, expected output paths, tolerance columns, contract tests, and
known gaps for every required workflow.

## Commands

The lightweight metadata and harness checks are login-node-safe:

```bash
uv run pytest tests/parity -q
```

The external golden checks should run on a CPU SLURM node because they can build
the native extension and scan the phase-0 1KG fixture:

```bash
JUST_TEMPDIR=/tmp \
GWAS_ENGINE_SLURM_EXCLUSIVE=0 \
GWAS_ENGINE_CPU_TIME=00:30:00 \
GWAS_ENGINE_CPU_CPUS_PER_TASK=8 \
GWAS_ENGINE_CPU_MEMORY=32G \
just slurm-cpu-run 'uv run pytest tests/parity tests/test_regenie2_parity.py -q'
```

Set `GWAS_ENGINE_DATA_DIR` when the phase-0 data lives outside `data/`.
`tests/test_regenie2_parity.py` skips external cases whose BGEN, sample,
phenotype, covariate, prediction list, or golden output files are absent.

The full pre-release contract gate should include the external golden checks and
the representative focused tests referenced by the metadata:

```bash
JUST_TEMPDIR=/tmp \
GWAS_ENGINE_SLURM_EXCLUSIVE=0 \
GWAS_ENGINE_CPU_TIME=00:30:00 \
GWAS_ENGINE_CPU_CPUS_PER_TASK=8 \
GWAS_ENGINE_CPU_MEMORY=32G \
just slurm-cpu-run 'uv run pytest \
  tests/parity \
  tests/test_regenie2_parity.py \
  tests/test_tabular.py::test_native_multi_alignment_is_explicit_complete_case_trait_major_matrix \
  tests/test_tabular.py::test_native_grouped_alignment_batches_traits_with_identical_sample_sets \
  tests/test_tabular.py::test_native_alignment_matches_iid_only_and_sorts_by_sample_order \
  tests/test_regenie2_binary.py::test_score_only_plan_produces_no_fallback_candidates \
  tests/test_regenie2_binary.py::test_variant_major_score_only_bt_matches_sample_major_with_covariates_loco_and_edge_genotypes \
  tests/test_regenie2_binary.py::test_variant_major_approximate_firth_matches_sample_major_with_covariates_loco_and_edge_genotypes \
  tests/test_regenie2_binary_diagnostics.py::test_binary_chunk_diagnostics_count_all_failure_categories \
  tests/test_regenie2_pipeline.py::test_build_phenotype_compute_groups_distinguishes_sample_modes \
  tests/test_regenie2_pipeline.py::test_complete_case_compute_group_resolution_adds_alignment_fingerprints \
  tests/test_regenie2_pipeline.py::test_multi_linear_resume_recomputes_partial_chunks_without_duplicate_writes \
  tests/test_io_output.py::test_initialize_output_run_compatible_resume_preserves_committed_chunks \
  tests/test_io_output.py::test_prepare_output_run_strict_resume_validates_manifest_chunks \
  -q'
```

Run the native BGEN imputation contract on CPU SLURM when native decode behavior
changes:

```bash
JUST_TEMPDIR=/tmp \
GWAS_ENGINE_SLURM_EXCLUSIVE=0 \
GWAS_ENGINE_CPU_TIME=00:20:00 \
GWAS_ENGINE_CPU_CPUS_PER_TASK=8 \
GWAS_ENGINE_CPU_MEMORY=32G \
just slurm-cpu-run 'cargo test genotype::bgen::decode::tests::variant_major_decode_covers_eight_bit_identity_subset_and_imputation_paths -- --nocapture'
```

## Golden Inputs

The upstream reference is REGENIE v4.1. The project server tooling downloads the
same release in `tooling/server/bootstrap_tools.py`.

Quantitative single-phenotype golden outputs are generated with:

```bash
regenie --step 1 --bed data/1kg_chr22_full --phenoFile data/pheno_cont.txt --covarFile data/covariates.txt --qt --force-step1 --bsize 1000 --out data/baselines/regenie_step1_qt
regenie --step 2 --bgen data/1kg_chr22_full.bgen --sample data/1kg_chr22_full.sample --ref-first --phenoFile data/pheno_cont.txt --covarFile data/covariates.txt --qt --bsize 400 --pred data/baselines/regenie_step1_qt_pred.list --out data/baselines/regenie_step2_qt
```

The expected output is
`data/baselines/regenie_step2_qt_phenotype_continuous.regenie`.

Binary score-only golden outputs are generated with:

```bash
regenie --step 1 --bed data/1kg_chr22_full --phenoFile data/pheno_bin.txt --covarFile data/covariates.txt --bt --cc12 --force-step1 --bsize 1000 --out data/baselines/regenie_step1
regenie --step 2 --bgen data/1kg_chr22_full.bgen --sample data/1kg_chr22_full.sample --ref-first --phenoFile data/pheno_bin.txt --covarFile data/covariates.txt --bt --cc12 --bsize 400 --pred data/baselines/regenie_step1_pred.list --out data/baselines/regenie_step2_score_only
```

The expected output is
`data/baselines/regenie_step2_score_only_phenotype_binary.regenie`.

## Coverage

| Workflow | Status | Gate |
| --- | --- | --- |
| Quantitative single phenotype, BGEN, covariates, LOCO predictions | External golden | `tests/test_regenie2_parity.py` compares `BETA`, `SE`, `CHISQ`, and `LOG10P`. |
| Quantitative multiple phenotypes, per-phenotype sample mode | Contract | Pipeline and tabular tests assert independent sample sets and grouped compatible alignments. |
| Quantitative multiple phenotypes, complete-case sample mode | Contract | Pipeline and tabular tests assert shared complete-case intersections and manifest fingerprints. |
| Binary score-only | External golden | `tests/test_regenie2_parity.py` compares `BETA`, `SE`, `CHISQ`, and `LOG10P` when the score-only golden exists. |
| Binary `--firth --approx` | Experimental | Internal diagnostics and kernel contracts exist, but upstream golden parity is not a production gate yet. |
| Missing phenotype/covariate filtering | Contract | Tabular alignment tests assert explicit missing-value exclusion. |
| Variant missingness/imputation | Contract | Native BGEN decode tests and variant-major kernel tests assert imputation/stat summary behavior. |
| Resume after interruption | Contract | Output and pipeline tests assert committed chunks are reused and partial chunks are rewritten. |

## Tolerances

Golden comparisons use statistic-specific absolute tolerances from
`tests/parity/golden_metadata.json`. The phase-0 defaults are:

| Statistic | Quantitative tolerance | Binary score-only tolerance |
| --- | ---: | ---: |
| `BETA` | `1.0e-3` | `1.0e-3` |
| `SE` | `1.0e-3` | `1.0e-3` |
| `CHISQ` | `1.5e-2` | `2.0e-2` |
| `LOG10P` | `1.5e-2` | `2.0e-2` |

Keep tolerances absolute and statistic-specific. If a tolerance changes, update
the metadata, this page, and the review note explaining the numerical cause.

## Known Gaps

Approximate Firth is accepted by the CLI/API but remains experimental for
production compatibility until a v4.1 upstream golden output is added to the
suite. The internal tests verify candidate selection, correction diagnostics,
failure labeling, and sample-major versus variant-major equivalence, but they do
not replace an upstream REGENIE comparison.

Large-fixture parity and performance comparisons remain outside this gate. Use
the benchmark tooling when changing performance assumptions.
