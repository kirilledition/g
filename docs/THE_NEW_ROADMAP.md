# Vision

## Core Target

Build a high-performance GWAS engine centered on **REGENIE workflows** for biobank-scale data.

## Active Scope

1. REGENIE step 2 linear association as the primary public CLI/API workflow.
2. BGEN ingestion + dosage chunk processing optimized for accelerator execution.
3. Reliable resumable Arrow chunk output with final Parquet compaction.
4. Profiling-first performance iteration on GPU/CPU JAX paths.

## Deferred Scope

Direct PLINK-style linear/logistic regression is deferred and archived in `archive/direct_association/`. It is not part of active API/CLI or CI.

## Engineering Priorities

1. Keep public surface small and explicit around supported REGENIE paths.
2. Avoid architecture branches for unsupported workflows.
3. Preserve archived direct-association artifacts as reusable references, not active dependencies.
