"""Command-line interface for the GWAS engine."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_LINEAR_CHUNK_SIZE = 2048
DEFAULT_LOGISTIC_CHUNK_SIZE = 512


def parse_covariate_names(raw_covariate_names: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated covariate name list."""
    if raw_covariate_names is None:
        return None
    covariate_names = tuple(name.strip() for name in raw_covariate_names.split(",") if name.strip())
    return covariate_names or None


def resolve_chunk_size(requested_chunk_size: int | None, glm_mode: str) -> int:
    """Resolve the effective chunk size for the selected model."""
    if requested_chunk_size is not None:
        return requested_chunk_size
    if glm_mode == "linear":
        return DEFAULT_LINEAR_CHUNK_SIZE
    return DEFAULT_LOGISTIC_CHUNK_SIZE


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Run GWAS association testing with JAX and Polars.")
    parser.add_argument("--bfile", required=True, type=Path, help="PLINK dataset prefix.")
    parser.add_argument("--pheno", required=True, type=Path, help="Phenotype table path.")
    parser.add_argument("--pheno-name", required=True, help="Phenotype column name to analyze.")
    parser.add_argument("--covar", required=True, type=Path, help="Covariate table path.")
    parser.add_argument("--covar-names", help="Optional comma-separated covariate column names.")
    parser.add_argument("--glm", required=True, choices=("linear", "logistic"), help="Association model.")
    parser.add_argument("--out", required=True, type=Path, help="Output prefix.")
    parser.add_argument("--chunk-size", type=int, help="Variants per BED chunk.")
    parser.add_argument("--variant-limit", type=int, help="Optional variant cap for debugging or tests.")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum logistic IRLS iterations.")
    parser.add_argument("--tolerance", type=float, default=1.0e-8, help="Logistic convergence tolerance.")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX execution device. Use 'gpu' to enable GPU acceleration.",
    )
    return parser


def main() -> None:
    """Run the GWAS CLI."""
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()

    from g.engine import iter_linear_output_frames, iter_logistic_output_frames, write_frame_iterator_to_tsv
    from g.jax_setup import configure_jax_device

    configure_jax_device(arguments.device)

    covariate_names = parse_covariate_names(arguments.covar_names)
    chunk_size = resolve_chunk_size(arguments.chunk_size, arguments.glm)

    if arguments.glm == "linear":
        frame_iterator = iter_linear_output_frames(
            bed_prefix=arguments.bfile,
            phenotype_path=arguments.pheno,
            phenotype_name=arguments.pheno_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=chunk_size,
            variant_limit=arguments.variant_limit,
        )
        output_path = arguments.out.with_suffix(".linear.tsv")
    else:
        frame_iterator = iter_logistic_output_frames(
            bed_prefix=arguments.bfile,
            phenotype_path=arguments.pheno,
            phenotype_name=arguments.pheno_name,
            covariate_path=arguments.covar,
            covariate_names=covariate_names,
            chunk_size=chunk_size,
            variant_limit=arguments.variant_limit,
            max_iterations=arguments.max_iterations,
            tolerance=arguments.tolerance,
        )
        output_path = arguments.out.with_suffix(".logistic.tsv")

    write_frame_iterator_to_tsv(frame_iterator, output_path)
