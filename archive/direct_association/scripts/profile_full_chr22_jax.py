#!/usr/bin/env python3
"""Capture a full-chromosome JAX profiler trace and memory profile."""

from __future__ import annotations

import argparse
import typing
from pathlib import Path

import jax
import jax.profiler

from g.engine import iter_linear_output_frames, iter_logistic_output_frames

if typing.TYPE_CHECKING:
    import collections.abc

    from g.engine import LinearChunkAccumulator, LogisticChunkAccumulator


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for full-chromosome profiling."""
    parser = argparse.ArgumentParser(description="Profile a full chr22 GWAS run with JAX tracing.")
    parser.add_argument("--bfile", required=True, type=Path, help="PLINK dataset prefix.")
    parser.add_argument("--pheno", required=True, type=Path, help="Phenotype table path.")
    parser.add_argument("--pheno-name", required=True, help="Phenotype column name to analyze.")
    parser.add_argument("--covar", required=True, type=Path, help="Covariate table path.")
    parser.add_argument("--covar-names", default="age,sex", help="Comma-separated covariate names.")
    parser.add_argument("--glm", required=True, choices=("linear", "logistic"), help="Association model.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Variants per chunk.")
    parser.add_argument("--variant-limit", type=int, help="Optional variant cap.")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum logistic IRLS iterations.")
    parser.add_argument("--tolerance", type=float, default=1.0e-8, help="Logistic convergence tolerance.")
    parser.add_argument("--trace-dir", required=True, type=Path, help="Directory for JAX trace output.")
    parser.add_argument(
        "--memory-profile",
        type=Path,
        help="Optional output path for a JAX device memory profile captured after the run.",
    )
    return parser


def parse_covariate_names(raw_covariate_names: str) -> tuple[str, ...] | None:
    """Parse a comma-separated covariate name list."""
    covariate_names = tuple(name.strip() for name in raw_covariate_names.split(",") if name.strip())
    return covariate_names or None


def run_and_materialize_frames(
    frame_iterator: collections.abc.Iterator[LinearChunkAccumulator]
    | collections.abc.Iterator[LogisticChunkAccumulator],
) -> None:
    """Force a full iterator run so the trace captures compute and formatting work."""
    for accumulator in frame_iterator:
        _ = len(accumulator.metadata.variant_identifiers)


def main() -> None:
    """Capture a trace for a full chr22 association run."""
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    covariate_names = parse_covariate_names(arguments.covar_names)
    arguments.trace_dir.mkdir(parents=True, exist_ok=True)

    with jax.profiler.trace(arguments.trace_dir, create_perfetto_trace=True):
        if arguments.glm == "linear":
            run_and_materialize_frames(
                iter_linear_output_frames(
                    bed_prefix=arguments.bfile,
                    phenotype_path=arguments.pheno,
                    phenotype_name=arguments.pheno_name,
                    covariate_path=arguments.covar,
                    covariate_names=covariate_names,
                    chunk_size=arguments.chunk_size,
                    variant_limit=arguments.variant_limit,
                )
            )
        else:
            run_and_materialize_frames(
                iter_logistic_output_frames(
                    bed_prefix=arguments.bfile,
                    phenotype_path=arguments.pheno,
                    phenotype_name=arguments.pheno_name,
                    covariate_path=arguments.covar,
                    covariate_names=covariate_names,
                    chunk_size=arguments.chunk_size,
                    variant_limit=arguments.variant_limit,
                    max_iterations=arguments.max_iterations,
                    tolerance=arguments.tolerance,
                )
            )

    if arguments.memory_profile is not None:
        jax.profiler.save_device_memory_profile(arguments.memory_profile)


if __name__ == "__main__":
    main()
