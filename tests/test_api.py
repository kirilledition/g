from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from g.api import (
    RunArtifacts,
    linear,
    logistic,
    parse_covariate_name_list,
    resolve_output_path,
)
from g.config import ComputeConfig, LogisticConfig


def test_parse_covariate_name_list_handles_string_input() -> None:
    """Ensure covariate names are normalized from a comma-separated string."""
    assert parse_covariate_name_list(" age, sex ,, bmi ") == ("age", "sex", "bmi")


def test_parse_covariate_name_list_handles_iterable_input() -> None:
    """Ensure covariate names are normalized from an iterable."""
    assert parse_covariate_name_list(["age", " sex ", ""]) == ("age", "sex")


def test_resolve_output_path_appends_mode_suffix_for_prefix() -> None:
    """Ensure output prefixes receive the expected mode-specific suffix."""
    assert resolve_output_path("results/output", "linear") == Path("results/output.linear.tsv")


def test_resolve_output_path_preserves_tsv_suffix() -> None:
    """Ensure explicit TSV paths are preserved."""
    assert resolve_output_path("results/output.tsv", "logistic") == Path("results/output.tsv")


def test_linear_uses_public_api_defaults() -> None:
    """Ensure the linear API configures JAX and writes the expected output path."""
    with (
        patch("g.api.configure_jax_device") as mock_configure_jax_device,
        patch("g.api.iter_linear_output_frames", return_value=iter(())) as mock_iter_linear_output_frames,
        patch("g.api.write_frame_iterator_to_tsv") as mock_write_frame_iterator_to_tsv,
    ):
        artifacts = linear(
            bfile="dataset",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
            covar_names="age,sex",
        )

    assert artifacts == RunArtifacts(sumstats_tsv=Path("results/output.linear.tsv"))
    mock_configure_jax_device.assert_called_once_with("cpu")
    assert mock_iter_linear_output_frames.call_args.kwargs["covariate_path"] is None
    assert mock_iter_linear_output_frames.call_args.kwargs["covariate_names"] == ("age", "sex")
    assert mock_iter_linear_output_frames.call_args.kwargs["chunk_size"] == 2048
    assert mock_write_frame_iterator_to_tsv.call_args.args[1] == Path("results/output.linear.tsv")


def test_logistic_uses_mode_specific_defaults() -> None:
    """Ensure the logistic API uses the tuned logistic chunk size."""
    with (
        patch("g.api.configure_jax_device") as mock_configure_jax_device,
        patch("g.api.iter_logistic_output_frames", return_value=iter(())) as mock_iter_logistic_output_frames,
        patch("g.api.write_frame_iterator_to_tsv") as mock_write_frame_iterator_to_tsv,
    ):
        artifacts = logistic(
            bfile="dataset",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
        )

    assert artifacts == RunArtifacts(sumstats_tsv=Path("results/output.logistic.tsv"))
    mock_configure_jax_device.assert_called_once_with("cpu")
    assert mock_iter_logistic_output_frames.call_args.kwargs["chunk_size"] == 1024
    assert mock_iter_logistic_output_frames.call_args.kwargs["max_iterations"] == 50
    assert mock_iter_logistic_output_frames.call_args.kwargs["tolerance"] == 1.0e-8
    assert mock_write_frame_iterator_to_tsv.call_args.args[1] == Path("results/output.logistic.tsv")


def test_logistic_rejects_invalid_solver_configuration() -> None:
    """Ensure invalid solver values raise a clear error."""
    with pytest.raises(ValueError, match="Tolerance must be positive"):
        logistic(
            bfile="dataset",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
            compute=ComputeConfig(),
            solver=LogisticConfig(tolerance=0.0),
        )
