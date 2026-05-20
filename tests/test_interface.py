from __future__ import annotations

from pathlib import Path

import pytest

from g import types
from g.interface import config, options


def test_all_option_specs_are_accepted_by_python_options() -> None:
    raw_options = {
        "step": 2,
        "qt": True,
        "bgen": "dataset.bgen",
        "sample": "dataset.sample",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "covarFile": "covariates.tsv",
        "covarCol": "age",
        "pred": "predictions.list",
        "bsize": 4096,
        "threads": 2,
        "out": "results/output",
        "g-device": "cpu",
        "g-staging-depth": 2,
        "g-variant-limit": 100,
        "g-trusted-no-missing-diploid": True,
        "g-trusted-bgen-validation-mode": "assume_validated",
        "g-sample-key-mode": "iid",
        "g-multi-phenotype-sample-mode": "complete-case",
        "g-output-format": "arrow",
        "g-writer-threads": 2,
        "g-writer-queue-depth": 3,
        "g-output-chunks-per-arrow-file": 2,
        "g-output-arrow-compression": "none",
        "g-firth-batch-size": 8,
        "g-firth-candidate-capacity": 16,
        "g-binary-null-maximum-iterations": 25,
        "g-binary-null-coefficient-tolerance": 1.0e-5,
        "g-firth-maximum-iterations": 30,
        "g-firth-gradient-tolerance": 1.0e-5,
        "g-firth-coefficient-tolerance": 1.0e-5,
        "g-firth-likelihood-tolerance": 1.0e-5,
        "g-firth-maximum-step-size": 4.0,
        "g-use-block-firth-math": True,
        "g-bgen-decode-tile-variant-count": 32,
        "g-jax-cache-dir": "cache/jax",
        "g-jax-matmul-precision": "highest",
        "g-jax-persistent-cache": False,
        "g-jax-persistent-cache-min-entry-size-bytes": 1024,
        "g-jax-persistent-cache-min-compile-time-seconds": 1,
        "g-jax-xla-autotune-cache": True,
        "g-jax-transfer-guard": True,
        "g-stage-timings-json": "timings.json",
    }

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.g_compute.trusted_bgen_validation_mode == types.TrustedBgenValidationMode.ASSUME_VALIDATED
    assert regenie_config.g_compute.multi_phenotype_sample_mode == types.MultiPhenotypeSampleMode.COMPLETE_CASE
    assert regenie_config.g_compute.firth_batch_size == 8
    assert regenie_config.g_compute.firth_candidate_capacity == 16
    assert regenie_config.g_compute.binary_null_maximum_iterations == 25
    assert regenie_config.g_compute.use_block_firth_math is True
    assert regenie_config.g_compute.bgen_decode_tile_variant_count == 32
    assert regenie_config.g_compute.jax_matmul_precision == types.JaxMatmulPrecision.HIGHEST
    assert regenie_config.g_compute.jax_persistent_cache is False
    assert regenie_config.g_output.format == types.OutputFormat.ARROW
    assert regenie_config.g_output.chunks_per_arrow_file == 2
    assert regenie_config.g_output.arrow_compression == types.ArrowCompression.NONE
    assert regenie_config.g_diagnostics.stage_timings_json == Path("timings.json")


def test_every_supported_option_has_explain_metadata() -> None:
    for option_name in options.supported_option_names() | options.unsupported_option_names():
        explanation = options.explain_option(option_name)
        assert option_name in explanation


def test_toml_round_trip_preserves_runtime_knobs(tmp_path: Path) -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "bt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "firth": True,
            "approx": True,
            "g-output-format": "arrow",
            "g-output-arrow-compression": "none",
            "g-firth-batch-size": 8,
            "g-jax-persistent-cache": False,
            "g-stage-timings-json": "timings.json",
        }
    )
    config_path = tmp_path / "effective_config.toml"

    config.write_toml(regenie_config, config_path)
    loaded_config = config.RegenieConfig.from_toml(config_path)

    assert loaded_config == regenie_config


def test_unknown_and_unsupported_options_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="Unknown g regenie option"):
        config.RegenieConfig.from_options({"not_a_real_option": True})

    with pytest.raises(ValueError, match="Unknown g regenie option: g-allow-duplicate-iid-alignment"):
        config.RegenieConfig.from_options({"g-allow-duplicate-iid-alignment": True})

    with pytest.raises(ValueError, match="Unknown g regenie option: g-allow-duplicate-iid-alignment"):
        config.RegenieConfig.from_options({"g": {"compute": {"allow-duplicate-iid-alignment": True}}})

    with pytest.raises(ValueError, match="valid REGENIE option"):
        config.RegenieConfig.from_options({"pgen": "dataset", "phenoFile": "phenotype.tsv"})


def test_staging_depth_must_be_positive() -> None:
    raw_options: dict[str, object] = {
        "step": 2,
        "qt": True,
        "bgen": "dataset.bgen",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "pred": "predictions.list",
        "out": "results/output",
        "g-staging-depth": 0,
    }

    with pytest.raises(ValueError, match="--g-staging-depth must be positive"):
        config.RegenieConfig.from_options(raw_options)


def test_duplicate_phenotype_names_are_rejected() -> None:
    raw_options: dict[str, object] = {
        "step": 2,
        "qt": True,
        "bgen": "dataset.bgen",
        "phenoFile": "phenotype.tsv",
        "phenoColList": "trait,other,trait",
        "pred": "predictions.list",
        "out": "results/output",
    }

    with pytest.raises(ValueError, match="Duplicate phenotype names are not allowed: trait"):
        config.RegenieConfig.from_options(raw_options)
