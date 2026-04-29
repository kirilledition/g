#!/usr/bin/env python3
"""Simple benchmark: TSV vs Arrow chunks I/O speed."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from g import api, engine, jax_setup, types
from g.io import output, source

BGEN = Path("data/1kg_chr22_full.bgen")
SAMPLE = Path("data/1kg_chr22_full.sample")
PHENO = Path("data/pheno_cont.txt")
COVAR = Path("data/covariates.txt")
PRED = Path("data/baselines/regenie_step1_qt_pred.list")
CHUNK_SIZE = 2048


def main() -> None:
    """Run simple I/O benchmark."""
    print("=" * 60)
    print("Simple I/O Benchmark: TSV vs Arrow Chunks")
    print("=" * 60)

    # Setup GPU
    jax_setup.configure_jax_device(types.Device.GPU)

    # Collect all results in memory (one compute pass)
    print("\n1. Running REGENIE step 2 linear on chr22 (GPU)...")
    start = time.perf_counter()

    genotype_config = source.build_bgen_source_config(BGEN, SAMPLE)
    frame_iterator = engine.iter_regenie2_linear_output_frames(
        genotype_source_config=genotype_config,
        phenotype_path=PHENO,
        phenotype_name="phenotype_continuous",
        prediction_list_path=PRED,
        covariate_path=COVAR,
        covariate_names=("age", "sex"),
        chunk_size=CHUNK_SIZE,
        variant_limit=None,
        prefetch_chunks=1,
        committed_chunk_identifiers=set(),
    )
    accumulators = list(frame_iterator)

    compute_time = time.perf_counter() - start
    total_variants = sum(len(acc.regenie2_linear_result.beta) for acc in accumulators)
    print(f"   Compute: {compute_time:.2f}s ({total_variants:,} variants, {len(accumulators)} chunks)")

    # Benchmark TSV writing
    print("\n2. Benchmarking TSV write...")
    with tempfile.TemporaryDirectory() as tmp:
        tsv_path = Path(tmp) / "results.tsv"
        start = time.perf_counter()
        engine.write_frame_iterator_to_tsv(iter(accumulators), tsv_path)
        tsv_time = time.perf_counter() - start
        tsv_size = tsv_path.stat().st_size / 1024 / 1024
    print(f"   TSV: {tsv_time:.2f}s ({tsv_size:.1f} MB)")

    # Benchmark Arrow chunks writing (no finalization)
    print("\n3. Benchmarking Arrow chunks write...")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        chunks_dir = run_dir / "chunks"
        chunks_dir.mkdir(parents=True)
        output_paths = output.OutputRunPaths(run_directory=run_dir, chunks_directory=chunks_dir)

        start = time.perf_counter()
        output.persist_chunked_results(
            frame_iterator=iter(accumulators),
            output_run_paths=output_paths,
            association_mode=types.AssociationMode.REGENIE2_LINEAR,
        )
        arrow_time = time.perf_counter() - start
        arrow_size = sum(f.stat().st_size for f in chunks_dir.glob("*.arrow")) / 1024 / 1024
    print(f"   Arrow: {arrow_time:.2f}s ({arrow_size:.1f} MB)")

    # Benchmark Arrow + Parquet finalization
    print("\n4. Benchmarking Arrow + Parquet finalization...")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        chunks_dir = run_dir / "chunks"
        chunks_dir.mkdir(parents=True)
        output_paths = output.OutputRunPaths(run_directory=run_dir, chunks_directory=chunks_dir)

        start = time.perf_counter()
        output.persist_chunked_results(
            frame_iterator=iter(accumulators),
            output_run_paths=output_paths,
            association_mode=types.AssociationMode.REGENIE2_LINEAR,
        )
        parquet_path = output.finalize_chunks_to_parquet(
            output_paths, types.AssociationMode.REGENIE2_LINEAR
        )
        parquet_time = time.perf_counter() - start
        parquet_size = parquet_path.stat().st_size / 1024 / 1024
    print(f"   Arrow+Parquet: {parquet_time:.2f}s ({parquet_size:.1f} MB)")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Mode':<25} {'Time (s)':<12} {'Size (MB)':<12} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'TSV':<25} {tsv_time:<12.2f} {tsv_size:<12.1f} {'baseline':<10}")
    print(f"{'Arrow chunks':<25} {arrow_time:<12.2f} {arrow_size:<12.1f} {tsv_time/arrow_time:<10.2f}x")
    print(f"{'Arrow + Parquet':<25} {parquet_time:<12.2f} {parquet_size:<12.1f} {tsv_time/parquet_time:<10.2f}x")
    print("-" * 60)

    if arrow_time < tsv_time:
        print(f"\n>>> Arrow chunks is {(tsv_time/arrow_time - 1)*100:.0f}% FASTER than TSV")
    else:
        print(f"\n>>> TSV is {(arrow_time/tsv_time - 1)*100:.0f}% FASTER than Arrow chunks")

    print(f">>> Arrow is {tsv_size/arrow_size:.1f}x smaller on disk")


if __name__ == "__main__":
    main()
