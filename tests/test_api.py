from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from g.api import (
    ComputeConfig,
    LogisticConfig,
    RunArtifacts,
    linear,
    logistic,
    parse_covariate_name_list,
    regenie2_linear,
    resolve_output_path,
    validate_compute_config,
    validate_logistic_config,
)
from g.io.output import OutputRunPaths, PreparedOutputRun
from g.types import AssociationMode, Device, GenotypeSourceFormat, OutputMode


def test_parse_covariate_name_list_handles_string_input() -> None:
    """Ensure covariate names are normalized from a comma-separated string."""
    assert parse_covariate_name_list(" age, sex ,, bmi ") == ("age", "sex", "bmi")


def test_parse_covariate_name_list_handles_iterable_input() -> None:
    """Ensure covariate names are normalized from an iterable."""
    assert parse_covariate_name_list(["age", " sex ", ""]) == ("age", "sex")


def test_resolve_output_path_appends_mode_suffix_for_prefix() -> None:
    """Ensure output prefixes receive the expected mode-specific suffix."""
    assert resolve_output_path("results/output", AssociationMode.LINEAR) == Path("results/output.linear.tsv")


def test_resolve_output_path_preserves_tsv_suffix() -> None:
    """Ensure explicit TSV paths are preserved."""
    assert resolve_output_path("results/output.tsv", AssociationMode.LOGISTIC) == Path("results/output.tsv")


def test_linear_uses_public_api_defaults() -> None:
    """Ensure the linear API configures JAX and writes the expected output path."""
    with (
        patch("g.jax_setup.configure_jax_device") as mock_configure_jax_device,
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
    mock_configure_jax_device.assert_called_once_with(Device.CPU)
    assert mock_iter_linear_output_frames.call_args.kwargs["covariate_path"] is None
    assert mock_iter_linear_output_frames.call_args.kwargs["covariate_names"] == ("age", "sex")
    assert mock_iter_linear_output_frames.call_args.kwargs["chunk_size"] == 2048
    assert (
        mock_iter_linear_output_frames.call_args.kwargs["genotype_source_config"].source_format
        == GenotypeSourceFormat.PLINK
    )
    assert mock_iter_linear_output_frames.call_args.kwargs["prefetch_chunks"] == 1
    assert mock_write_frame_iterator_to_tsv.call_args.args[1] == Path("results/output.linear.tsv")


def test_logistic_uses_mode_specific_defaults() -> None:
    """Ensure the logistic API uses the tuned logistic chunk size."""
    with (
        patch("g.jax_setup.configure_jax_device") as mock_configure_jax_device,
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
    mock_configure_jax_device.assert_called_once_with(Device.CPU)
    assert mock_iter_logistic_output_frames.call_args.kwargs["chunk_size"] == 1024
    assert mock_iter_logistic_output_frames.call_args.kwargs["max_iterations"] == 50
    assert mock_iter_logistic_output_frames.call_args.kwargs["tolerance"] == 1.0e-8
    assert (
        mock_iter_logistic_output_frames.call_args.kwargs["genotype_source_config"].source_format
        == GenotypeSourceFormat.PLINK
    )
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
        patch("g.jax_setup.configure_jax_device"),
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
    assert genotype_source_config.source_format == GenotypeSourceFormat.BGEN
    assert genotype_source_config.source_path == Path("dataset.bgen")


def test_linear_supports_explicit_bgen_sample_file() -> None:
    """Ensure the public API preserves an explicit BGEN sample path."""
    with (
        patch("g.jax_setup.configure_jax_device"),
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


def test_regenie2_linear_uses_bgen_input_and_prediction_list() -> None:
    """Ensure the REGENIE step 2 API dispatches through the BGEN-backed iterator."""
    with (
        patch("g.jax_setup.configure_jax_device") as mock_configure_jax_device,
        patch("g.api.iter_regenie2_linear_output_frames", return_value=iter(())) as mock_iterator,
        patch("g.api.write_frame_iterator_to_tsv") as mock_write_frame_iterator_to_tsv,
    ):
        artifacts = regenie2_linear(
            bgen="dataset.bgen",
            sample="dataset.sample",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
            covar_names="age,sex",
            pred="predictions.list",
        )

    assert artifacts == RunArtifacts(sumstats_tsv=Path("results/output.regenie2_linear.tsv"))
    mock_configure_jax_device.assert_called_once_with(Device.CPU)
    assert mock_iterator.call_args.kwargs["covariate_names"] == ("age", "sex")
    assert mock_iterator.call_args.kwargs["prediction_list_path"] == Path("predictions.list")
    assert mock_iterator.call_args.kwargs["chunk_size"] == 2048
    genotype_source_config = mock_iterator.call_args.kwargs["genotype_source_config"]
    assert genotype_source_config.source_format == GenotypeSourceFormat.BGEN
    assert genotype_source_config.source_path == Path("dataset.bgen")
    assert genotype_source_config.sample_path == Path("dataset.sample")
    assert mock_write_frame_iterator_to_tsv.call_args.args[1] == Path("results/output.regenie2_linear.tsv")


@pytest.mark.parametrize(
    ("compute_config", "expected_message"),
    [
        (ComputeConfig(chunk_size=0), "Chunk size must be positive"),
        (ComputeConfig(variant_limit=0), "Variant limit must be positive"),
        (ComputeConfig(prefetch_chunks=-1), "Prefetch chunk count must be zero or positive"),
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
    mock_output_run_paths = OutputRunPaths(
        run_directory=Path("results/output.linear.run"),
        chunks_directory=Path("results/output.linear.run/chunks"),
    )

    with (
        patch("g.jax_setup.configure_jax_device") as mock_configure_jax_device,
        patch(
            "g.api.prepare_output_run",
            return_value=PreparedOutputRun(
                output_run_paths=mock_output_run_paths,
                committed_chunk_identifiers=frozenset({2, 4}),
            ),
        ) as mock_prepare_output_run,
        patch("g.api.iter_linear_output_frames", return_value=iter(())) as mock_iter_linear_output_frames,
        patch("g.api.persist_chunked_results") as mock_persist_chunked_results,
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
                output_mode=OutputMode.ARROW_CHUNKS,
                output_run_directory=Path("results/output"),
                resume=True,
                finalize_parquet=True,
            ),
        )

    assert artifacts == RunArtifacts(
        output_run_directory=Path("results/output.linear.run"),
        final_parquet=Path("results/output.linear.run/final.parquet"),
    )
    mock_configure_jax_device.assert_called_once_with(Device.CPU)
    assert mock_iter_linear_output_frames.call_args.kwargs["committed_chunk_identifiers"] == {2, 4}
    mock_persist_chunked_results.assert_called_once()
    mock_prepare_output_run.assert_called_once()
    mock_finalize.assert_called_once_with(mock_output_run_paths, AssociationMode.LINEAR)


def test_logistic_chunked_output_returns_run_artifacts_without_finalization() -> None:
    """Ensure chunked logistic runs persist artifacts even without final Parquet compaction."""
    mock_output_run_paths = OutputRunPaths(
        run_directory=Path("results/output.logistic.run"),
        chunks_directory=Path("results/output.logistic.run/chunks"),
    )

    with (
        patch("g.jax_setup.configure_jax_device") as mock_configure_jax_device,
        patch(
            "g.api.prepare_output_run",
            return_value=PreparedOutputRun(
                output_run_paths=mock_output_run_paths,
                committed_chunk_identifiers=frozenset({1}),
            ),
        ) as mock_prepare_output_run,
        patch("g.api.iter_logistic_output_frames", return_value=iter(())) as mock_iter_logistic_output_frames,
        patch("g.api.persist_chunked_results") as mock_persist_chunked_results,
        patch("g.api.finalize_chunks_to_parquet") as mock_finalize,
    ):
        artifacts = logistic(
            bfile="dataset",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
            compute=ComputeConfig(
                output_mode=OutputMode.ARROW_CHUNKS,
                output_run_directory=Path("results/output"),
                resume=True,
            ),
        )

    assert artifacts == RunArtifacts(output_run_directory=Path("results/output.logistic.run"), final_parquet=None)
    mock_configure_jax_device.assert_called_once_with(Device.CPU)
    assert mock_iter_logistic_output_frames.call_args.kwargs["committed_chunk_identifiers"] == {1}
    mock_persist_chunked_results.assert_called_once()
    mock_prepare_output_run.assert_called_once()
    mock_finalize.assert_not_called()


def test_regenie2_linear_chunked_output_returns_run_artifacts_without_finalization() -> None:
    """Ensure chunked REGENIE step 2 runs persist artifacts with the correct mode."""
    mock_output_run_paths = OutputRunPaths(
        run_directory=Path("results/output.regenie2_linear.run"),
        chunks_directory=Path("results/output.regenie2_linear.run/chunks"),
    )

    with (
        patch("g.jax_setup.configure_jax_device") as mock_configure_jax_device,
        patch(
            "g.api.prepare_output_run",
            return_value=PreparedOutputRun(
                output_run_paths=mock_output_run_paths,
                committed_chunk_identifiers=frozenset({3}),
            ),
        ) as mock_prepare_output_run,
        patch("g.api.iter_regenie2_linear_output_frames", return_value=iter(())) as mock_iterator,
        patch("g.api.persist_chunked_results") as mock_persist_chunked_results,
        patch("g.api.finalize_chunks_to_parquet") as mock_finalize,
    ):
        artifacts = regenie2_linear(
            bgen="dataset.bgen",
            pheno="phenotype.tsv",
            pheno_name="trait",
            out="results/output",
            pred="predictions.list",
            compute=ComputeConfig(
                output_mode=OutputMode.ARROW_CHUNKS,
                output_run_directory=Path("results/output"),
                resume=True,
            ),
        )

    assert artifacts == RunArtifacts(
        output_run_directory=Path("results/output.regenie2_linear.run"),
        final_parquet=None,
    )
    mock_configure_jax_device.assert_called_once_with(Device.CPU)
    assert mock_iterator.call_args.kwargs["committed_chunk_identifiers"] == {3}
    mock_persist_chunked_results.assert_called_once()
    mock_prepare_output_run.assert_called_once()
    mock_finalize.assert_not_called()
