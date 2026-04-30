from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import g
from g import api
from g.api import (
    ComputeConfig,
    RunArtifacts,
    parse_covariate_name_list,
    regenie2_linear,
    validate_compute_config,
)
from g.io.output import OutputRunPaths, PreparedOutputRun
from g.types import Device, GenotypeSourceFormat


def test_public_package_no_longer_exposes_direct_linear_or_logistic() -> None:
    assert not hasattr(g, "linear")
    assert not hasattr(g, "logistic")


def test_parse_covariate_name_list_handles_string_input() -> None:
    assert parse_covariate_name_list(" age, sex ,, bmi ") == ("age", "sex", "bmi")


def test_parse_covariate_name_list_handles_iterable_input() -> None:
    assert parse_covariate_name_list(["age", " sex ", ""]) == ("age", "sex")


def test_regenie2_linear_uses_bgen_input_and_prediction_list() -> None:
    with (
        patch("g.api.configure_jax_device") as mock_configure_jax_device,
        patch("g.api.iter_regenie2_linear_output_frames", return_value=iter(())) as mock_iterator,
        patch(
            "g.api.prepare_output_run",
            return_value=PreparedOutputRun(
                output_run_paths=OutputRunPaths(
                    run_directory=Path("results/output.regenie2_linear.run"),
                    chunks_directory=Path("results/output.regenie2_linear.run/chunks"),
                ),
                committed_chunk_identifiers=frozenset(),
            ),
        ),
        patch("g.api.persist_chunked_results") as mock_persist_chunked_results,
        patch(
            "g.api.finalize_chunks_to_parquet",
            return_value=Path("results/output.regenie2_linear.run/final.parquet"),
        ),
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

    assert artifacts == RunArtifacts(
        output_run_directory=Path("results/output.regenie2_linear.run"),
        final_parquet=Path("results/output.regenie2_linear.run/final.parquet"),
    )
    mock_configure_jax_device.assert_called_once_with(Device.CPU)
    assert mock_iterator.call_args.kwargs["covariate_names"] == ("age", "sex")
    assert mock_iterator.call_args.kwargs["prediction_list_path"] == Path("predictions.list")
    genotype_source_config = mock_iterator.call_args.kwargs["genotype_source_config"]
    assert genotype_source_config.source_format == GenotypeSourceFormat.BGEN
    mock_persist_chunked_results.assert_called_once()


@pytest.mark.parametrize(
    ("compute_config", "expected_message"),
    [
        (ComputeConfig(chunk_size=0), "Chunk size must be positive"),
        (ComputeConfig(variant_limit=0), "Variant limit must be positive"),
        (ComputeConfig(prefetch_chunks=-1), "Prefetch chunk count must be zero or positive"),
        (ComputeConfig(arrow_payload_batch_size=0), "Arrow payload batch size must be positive"),
    ],
)
def test_validate_compute_config_rejects_invalid_values(
    compute_config: ComputeConfig,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        validate_compute_config(compute_config)


def test_regenie2_linear_chunked_output_returns_run_artifacts_without_finalization() -> None:
    mock_output_run_paths = OutputRunPaths(
        run_directory=Path("results/output.regenie2_linear.run"),
        chunks_directory=Path("results/output.regenie2_linear.run/chunks"),
    )

    with (
        patch("g.api.configure_jax_device") as mock_configure_jax_device,
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
                output_run_directory=Path("results/output"),
                resume=True,
                finalize_parquet=False,
            ),
        )

    assert artifacts == RunArtifacts(
        output_run_directory=Path("results/output.regenie2_linear.run"),
        final_parquet=None,
    )
    mock_configure_jax_device.assert_called_once_with(Device.CPU)
    assert mock_iterator.call_args.kwargs["committed_chunk_identifiers"] == {3}
    mock_persist_chunked_results.assert_called_once()
    assert mock_persist_chunked_results.call_args.kwargs["payload_batch_size"] == api.DEFAULT_ARROW_PAYLOAD_BATCH_SIZE
    mock_prepare_output_run.assert_called_once()
    mock_finalize.assert_not_called()
