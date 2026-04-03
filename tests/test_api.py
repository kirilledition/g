from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from g.api import (
    ComputeConfig,
    LogisticConfig,
    RunArtifacts,
    linear,
    logistic,
    parse_covariate_name_list,
    resolve_output_path,
    validate_compute_config,
    validate_logistic_config,
)


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
    assert mock_iter_linear_output_frames.call_args.kwargs["genotype_source_config"].source_format == "plink"
    assert mock_iter_linear_output_frames.call_args.kwargs["prefetch_chunks"] == 1
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
    assert mock_iter_logistic_output_frames.call_args.kwargs["genotype_source_config"].source_format == "plink"
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


def test_linear_supports_bgen_input() -> None:
    """Ensure the public API can dispatch a BGEN-backed run."""
    with (
        patch("g.api.configure_jax_device"),
        patch("g.api.iter_linear_output_frames", return_value=iter(())) as mock_iter_linear_output_frames,
        patch("g.api.write_frame_iterator_to_tsv"),
    ):
        linear(
            bfile=None,
            bgen="dataset.bgen",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
        )

    genotype_source_config = mock_iter_linear_output_frames.call_args.kwargs["genotype_source_config"]
    assert genotype_source_config.source_format == "bgen"
    assert genotype_source_config.source_path == Path("dataset.bgen")


def test_linear_supports_explicit_bgen_sample_file() -> None:
    """Ensure the public API preserves an explicit BGEN sample path."""
    with (
        patch("g.api.configure_jax_device"),
        patch("g.api.iter_linear_output_frames", return_value=iter(())) as mock_iter_linear_output_frames,
        patch("g.api.write_frame_iterator_to_tsv"),
    ):
        linear(
            bfile=None,
            bgen="dataset.bgen",
            sample="dataset.sample",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
        )

    genotype_source_config = mock_iter_linear_output_frames.call_args.kwargs["genotype_source_config"]
    assert genotype_source_config.sample_path == Path("dataset.sample")


@pytest.mark.parametrize(
    ("compute_config", "expected_message"),
    [
        (ComputeConfig(chunk_size=0), "Chunk size must be positive"),
        (ComputeConfig(variant_limit=0), "Variant limit must be positive"),
        (ComputeConfig(prefetch_chunks=-1), "Prefetch chunk count must be zero or positive"),
        (ComputeConfig(device="tpu"), "Unsupported device 'tpu'"),
        (ComputeConfig(output_mode="parquet"), "Unsupported output mode 'parquet'"),
    ],
)
def test_validate_compute_config_rejects_invalid_values(
    compute_config: ComputeConfig,
    expected_message: str,
) -> None:
    """Ensure public compute validation rejects unsupported values."""
    with pytest.raises(ValueError, match=expected_message):
        validate_compute_config(compute_config)


@pytest.mark.parametrize(
    ("solver_config", "expected_message"),
    [
        (LogisticConfig(max_iterations=0), "Maximum iterations must be positive"),
        (LogisticConfig(tolerance=0.0), "Tolerance must be positive"),
    ],
)
def test_validate_logistic_config_rejects_invalid_values(
    solver_config: LogisticConfig,
    expected_message: str,
) -> None:
    """Ensure logistic solver validation rejects invalid numeric parameters."""
    with pytest.raises(ValueError, match=expected_message):
        validate_logistic_config(solver_config)


def test_linear_chunked_output_returns_run_artifacts_and_finalizes_parquet() -> None:
    """Ensure chunked linear runs prepare, persist, and optionally finalize outputs."""
    prepared_output_run = SimpleNamespace(committed_chunk_identifiers=frozenset({2, 4}))
    persisted_output_run_paths = SimpleNamespace(run_directory=Path("results/output.linear.run"))

    with (
        patch("g.api.configure_jax_device") as mock_configure_jax_device,
        patch("g.api.build_output_run_configuration", return_value=object()),
        patch("g.api.prepare_output_run", return_value=prepared_output_run) as mock_prepare_output_run,
        patch("g.api.iter_linear_output_frames", return_value=iter(())) as mock_iter_linear_output_frames,
        patch("g.api.persist_chunked_results", return_value=persisted_output_run_paths) as mock_persist_chunked_results,
        patch(
            "g.api.finalize_chunks_to_parquet", return_value=Path("results/output.linear.run/final.parquet")
        ) as mock_finalize,
    ):
        artifacts = linear(
            bfile="dataset",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
            compute=ComputeConfig(
                output_mode="arrow_chunks",
                output_run_directory=Path("results/output"),
                resume=True,
                finalize_parquet=True,
            ),
        )

    assert artifacts == RunArtifacts(
        output_run_directory=Path("results/output.linear.run"),
        final_parquet=Path("results/output.linear.run/final.parquet"),
    )
    mock_configure_jax_device.assert_called_once_with("cpu")
    assert mock_iter_linear_output_frames.call_args.kwargs["committed_chunk_identifiers"] == frozenset({2, 4})
    assert mock_persist_chunked_results.call_args.kwargs["resume"] is True
    mock_prepare_output_run.assert_called_once()
    mock_finalize.assert_called_once_with(persisted_output_run_paths)


def test_logistic_chunked_output_returns_run_artifacts_without_finalization() -> None:
    """Ensure chunked logistic runs persist artifacts even without final Parquet compaction."""
    prepared_output_run = SimpleNamespace(committed_chunk_identifiers=frozenset({1}))
    persisted_output_run_paths = SimpleNamespace(run_directory=Path("results/output.logistic.run"))

    with (
        patch("g.api.configure_jax_device") as mock_configure_jax_device,
        patch("g.api.build_output_run_configuration", return_value=object()),
        patch("g.api.prepare_output_run", return_value=prepared_output_run) as mock_prepare_output_run,
        patch("g.api.iter_logistic_output_frames", return_value=iter(())) as mock_iter_logistic_output_frames,
        patch("g.api.persist_chunked_results", return_value=persisted_output_run_paths) as mock_persist_chunked_results,
        patch("g.api.finalize_chunks_to_parquet") as mock_finalize,
    ):
        artifacts = logistic(
            bfile="dataset",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
            compute=ComputeConfig(
                output_mode="arrow_chunks",
                output_run_directory=Path("results/output"),
                resume=True,
            ),
        )

    assert artifacts == RunArtifacts(output_run_directory=Path("results/output.logistic.run"), final_parquet=None)
    mock_configure_jax_device.assert_called_once_with("cpu")
    assert mock_iter_logistic_output_frames.call_args.kwargs["committed_chunk_identifiers"] == frozenset({1})
    assert mock_persist_chunked_results.call_args.kwargs["resume"] is True
    mock_prepare_output_run.assert_called_once()
    mock_finalize.assert_not_called()


def test_linear_chunked_output_raises_when_prepare_output_run_returns_none() -> None:
    """Ensure linear chunked runs fail loudly if preparation metadata is unexpectedly absent."""
    with (
        patch("g.api.configure_jax_device"),
        patch("g.api.build_output_run_configuration", return_value=object()),
        patch("g.api.prepare_output_run", return_value=None),
        patch("g.api.iter_linear_output_frames", return_value=iter(())),
        pytest.raises(RuntimeError, match="prepared output metadata"),
    ):
        linear(
            bfile="dataset",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
            compute=ComputeConfig(output_mode="arrow_chunks"),
        )


def test_logistic_chunked_output_raises_when_prepare_output_run_returns_none() -> None:
    """Ensure logistic chunked runs fail loudly if preparation metadata is unexpectedly absent."""
    with (
        patch("g.api.configure_jax_device"),
        patch("g.api.build_output_run_configuration", return_value=object()),
        patch("g.api.prepare_output_run", return_value=None),
        patch("g.api.iter_logistic_output_frames", return_value=iter(())),
        pytest.raises(RuntimeError, match="prepared output metadata"),
    ):
        logistic(
            bfile="dataset",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
            compute=ComputeConfig(output_mode="arrow_chunks"),
        )
