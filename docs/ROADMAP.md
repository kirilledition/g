# Roadmap

## Active Direction

The active product direction is a biobank-scale **REGENIE-first** engine:

1. Harden and optimize REGENIE step 2 (`regenie2-linear`) in Python/JAX.
2. Improve BGEN ingestion and dosage chunk throughput for large cohorts.
3. Keep Arrow/Parquet chunk persistence robust for long-running resumable jobs.
4. Expand profiling and performance work around REGENIE execution.

## Deferred Direction

Direct PLINK-style linear/logistic association in `g` is intentionally deferred. The historical implementation has been moved to `archive/direct_association/` for future reuse if priorities change.

## Near-Term Milestones

1. REGENIE step 2 profiling-driven performance improvements.
2. CI hardening for active REGENIE and shared I/O paths.
3. Architecture cleanup to keep active modules free of direct-regression branches.
4. Keep `data/benchmarks/regenie_comparison/` and `data/profiles/regenie_comparison/` up to date with:
   - original regenie step 1/2 binary + quantitative benchmarks
   - g quantitative step 2 CPU/GPU comparisons
   - explicit `not_implemented` placeholders for unsupported g REGENIE programs
