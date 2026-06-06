from __future__ import annotations

from pathlib import Path

import pytest

from g import execution_plan, types
from g.interface import config, config_layers, defaults, options, toml_schema


def build_valid_quantitative_options() -> dict[str, object]:
    """Build minimal valid quantitative REGENIE options."""
    return {
        "step": 2,
        "qt": True,
        "bgen": "dataset.bgen",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "pred": "predictions.list",
        "out": "results/output",
    }


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
        "g-null-logistic-nonconvergence": "warn",
        "g-binary-minimum-probability": 1.0e-7,
        "g-binary-minimum-variance": 1.0e-9,
        "g-binary-relative-variance-tolerance": 2.0e-6,
        "g-firth-maximum-iterations": 30,
        "g-firth-gradient-tolerance": 1.0e-5,
        "g-firth-coefficient-tolerance": 1.0e-5,
        "g-firth-likelihood-tolerance": 1.0e-5,
        "g-firth-maximum-step-size": 4.0,
        "g-use-block-firth-math": True,
        "g-bgen-decode-tile-variant-count": 32,
        "g-gpu-genotype-format": "dosage",
        "g-score-dtype": "float64",
        "g-firth-dtype": "float64",
        "g-jax-cache-dir": "cache/jax",
        "g-jax-matmul-precision": "highest",
        "g-jax-persistent-cache": False,
        "g-jax-persistent-cache-min-entry-size-bytes": 1024,
        "g-jax-persistent-cache-min-compile-time-seconds": 1,
        "g-jax-xla-autotune-cache": True,
        "g-jax-transfer-guard": True,
        "g-telemetry": "trace",
        "g-log-dir": "logs",
        "g-stage-timings-json": "timings.json",
        "g-log-filter": "g=debug",
        "g-log-file": "logs/g.jsonl",
        "g-log-stderr": False,
        "g-progress-interval-seconds": 1.5,
        "g-progress-interval-chunks": 4,
        "g-profile-summary-json": "logs/profile.summary.json",
        "g-trace-file": "logs/trace.jsonl",
        "g-trace-filter": "g=trace",
        "g-log-queue-size": 1024,
        "g-log-lossy": False,
        "g-include-source-location": True,
        "g-include-span-events": True,
    }

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.g_compute.trusted_bgen_validation_mode == types.TrustedBgenValidationMode.ASSUME_VALIDATED
    assert regenie_config.g_compute.multi_phenotype_sample_mode == types.MultiPhenotypeSampleMode.COMPLETE_CASE
    assert regenie_config.g_compute.firth_batch_size == 8
    assert regenie_config.g_compute.firth_candidate_capacity == 16
    assert regenie_config.g_compute.binary_null_maximum_iterations == 25
    assert regenie_config.g_compute.binary_null_coefficient_tolerance == 1.0e-5
    assert regenie_config.g_compute.null_logistic_nonconvergence_policy == (types.NullLogisticNonconvergencePolicy.WARN)
    assert regenie_config.g_compute.binary_minimum_probability == 1.0e-7
    assert regenie_config.g_compute.binary_minimum_variance == 1.0e-9
    assert regenie_config.g_compute.binary_relative_variance_tolerance == 2.0e-6
    assert regenie_config.g_compute.use_block_firth_math is True
    assert regenie_config.g_compute.bgen_decode_tile_variant_count == 32
    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.DOSAGE
    assert regenie_config.g_compute.score_dtype == types.FloatingPointDtype.FLOAT64
    assert regenie_config.g_compute.firth_dtype == types.FloatingPointDtype.FLOAT64
    assert regenie_config.g_compute.jax_matmul_precision == types.JaxMatmulPrecision.HIGHEST
    assert regenie_config.g_compute.jax_persistent_cache is False
    assert regenie_config.g_output.format == types.OutputFormat.ARROW
    assert regenie_config.g_output.chunks_per_arrow_file == 2
    assert regenie_config.g_output.arrow_compression == types.ArrowCompression.NONE
    assert regenie_config.g_diagnostics.telemetry == types.TelemetryMode.TRACE
    assert regenie_config.g_diagnostics.log_dir == Path("logs")
    assert regenie_config.g_diagnostics.stage_timings_json == Path("timings.json")
    assert regenie_config.g_diagnostics.log_filter == "g=debug"
    assert regenie_config.g_diagnostics.log_file == Path("logs/g.jsonl")
    assert regenie_config.g_diagnostics.log_stderr is False
    assert regenie_config.g_diagnostics.progress_interval_seconds == 1.5
    assert regenie_config.g_diagnostics.progress_interval_chunks == 4
    assert regenie_config.g_diagnostics.profile_summary_json == Path("logs/profile.summary.json")
    assert regenie_config.g_diagnostics.trace_file == Path("logs/trace.jsonl")
    assert regenie_config.g_diagnostics.trace_filter == "g=trace"
    assert regenie_config.g_diagnostics.log_queue_size == 1024
    assert regenie_config.g_diagnostics.log_lossy is False
    assert regenie_config.g_diagnostics.include_source_location is True
    assert regenie_config.g_diagnostics.include_span_events is True


def test_every_supported_option_has_explain_metadata() -> None:
    for option_name in options.supported_option_names() | options.unsupported_option_names():
        explanation = options.explain_option(option_name)
        assert option_name in explanation


def test_packaged_default_catalog_matches_option_policies() -> None:
    default_catalog = defaults.load_default_option_catalog()
    defaulted_option_names = {
        option_spec.name
        for option_spec in options.OPTION_SPECS
        if option_spec.default_policy == options.DefaultPolicy.VALUE
    }
    non_defaultable_option_names = {
        option_spec.name
        for option_spec in options.OPTION_SPECS
        if option_spec.default_policy
        in {
            options.DefaultPolicy.ABSENT_IS_NONE,
            options.DefaultPolicy.REQUIRED_AT_RUNTIME,
            options.DefaultPolicy.UNSUPPORTED,
            options.DefaultPolicy.DERIVED,
        }
    }

    assert set(default_catalog.normalized_options) == defaulted_option_names
    assert not set(default_catalog.normalized_options) & non_defaultable_option_names
    assert len(default_catalog.default_config_hash) == 64


def test_packaged_default_hash_uses_raw_toml_payload() -> None:
    raw_toml = config_layers.decode_toml_builtin_mapping(
        defaults.load_default_toml_bytes(),
        source="config.default.toml",
    )
    default_catalog = defaults.load_default_option_catalog()

    assert default_catalog.raw_toml == raw_toml
    assert default_catalog.default_config_hash == defaults.build_default_config_hash(raw_toml)
    assert isinstance(raw_toml["g"]["diagnostics"]["progress-interval-seconds"], int)


def test_typed_toml_schema_matches_option_registry() -> None:
    assert toml_schema.schema_toml_paths() == frozenset(options.OPTION_SPEC_BY_TOML_PATH)


def test_packaged_default_toml_decodes_to_typed_config() -> None:
    default_catalog = defaults.load_default_option_catalog()

    assert isinstance(default_catalog.toml_config, toml_schema.TomlConfig)
    assert default_catalog.raw_toml["trait"]["step"] == 2
    assert default_catalog.normalized_options["g-device"] == "cpu"


def test_msgspec_toml_schema_rejects_unknown_keys_and_wrong_types() -> None:
    with pytest.raises(ValueError, match="unknown field `not-a-real-key`"):
        config_layers.decode_toml_bytes(
            "[g.compute]\nnot-a-real-key = true\n",
            source="inline",
        )

    with pytest.raises(ValueError, match="Expected `int`"):
        config_layers.decode_toml_bytes(
            '[trait]\nstep = "2"\n',
            source="inline",
        )


def test_msgspec_toml_schema_rejects_removed_jax_x64_option() -> None:
    with pytest.raises(ValueError, match="jax-enable-x64"):
        config_layers.decode_toml_bytes(
            "[g.compute]\njax-enable-x64 = false\n",
            source="inline",
        )


def test_toml_metadata_is_accepted_but_not_an_option() -> None:
    toml_config = config_layers.decode_toml_bytes(
        '[metadata]\ncustom = "ignored"\n[trait]\nstep = 2\n',
        source="inline",
    )

    assert config_layers.toml_config_to_option_dictionary(toml_config) == {"step": 2}


def test_no_configurable_default_constants_reappear_in_source() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "g"
    forbidden_fragments = (
        "DEFAULT_FIRTH",
        "DEFAULT_BINARY_NULL",
        "DEFAULT_BGEN_DECODE",
        "DEFAULT_JAX",
        "DEFAULT_OUTPUT_WRITER",
        "DEFAULT_LOG",
        "DEFAULT_PROGRESS",
    )
    checked_suffixes = {".py", ".pyi", ".rs"}
    offenders: list[str] = []

    for source_path in source_root.rglob("*"):
        if source_path.suffix not in checked_suffixes:
            continue
        source_text = source_path.read_text(encoding="utf-8")
        for forbidden_fragment in forbidden_fragments:
            if forbidden_fragment in source_text:
                relative_path = source_path.relative_to(source_root.parent.parent)
                offenders.append(f"{relative_path}: {forbidden_fragment}")

    assert offenders == []


def test_logging_diagnostics_default_to_info_stderr() -> None:
    diagnostics_config = config.GDiagnosticsConfig()

    assert diagnostics_config.telemetry == types.TelemetryMode.PROGRESS
    assert diagnostics_config.log_dir is None
    assert diagnostics_config.log_filter == "info"
    assert diagnostics_config.log_file is None
    assert diagnostics_config.log_stderr is True
    assert diagnostics_config.progress_interval_seconds == 5
    assert diagnostics_config.progress_interval_chunks == 10
    assert diagnostics_config.profile_summary_json is None
    assert diagnostics_config.trace_file is None
    assert diagnostics_config.log_queue_size == 65536
    assert diagnostics_config.log_lossy is True
    assert diagnostics_config.include_source_location is False
    assert diagnostics_config.include_span_events is False


def test_packaged_default_toml_is_loaded_for_python_options() -> None:
    regenie_config = config.RegenieConfig.from_options(build_valid_quantitative_options())

    assert config.load_default_option_dictionary()["trait"]["bsize"] == config.default_int_option("bsize")
    assert regenie_config.trait.bsize == config.default_int_option("bsize")
    assert regenie_config.g_compute.device == types.Device.CPU
    assert regenie_config.g_compute.null_logistic_nonconvergence_policy == types.NullLogisticNonconvergencePolicy.FAIL
    assert regenie_config.g_compute.score_dtype == types.FloatingPointDtype.FLOAT32
    assert regenie_config.g_compute.firth_dtype == types.FloatingPointDtype.FLOAT64
    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.DOSAGE
    assert regenie_config.g_compute.jax_persistent_cache is True
    assert regenie_config.g_output.format == types.OutputFormat.PARQUET
    assert regenie_config.g_diagnostics.log_filter == config.default_string_option("g-log-filter")
    assert "pThresh" not in regenie_config.explicit_options
    assert "firth" not in regenie_config.explicit_options


def test_user_toml_overrides_packaged_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[input]",
                'bgen = "dataset.bgen"',
                'phenoFile = "phenotype.tsv"',
                'phenoCol = "trait"',
                'pred = "predictions.list"',
                "[trait]",
                "bsize = 2048",
                "[output]",
                'out = "results/output"',
                "[g.compute]",
                'device = "gpu"',
                "[g.output]",
                'format = "arrow"',
                "[g.diagnostics]",
                'log-filter = "g=debug"',
            ]
        ),
        encoding="utf-8",
    )

    regenie_config = config.RegenieConfig.from_toml(config_path)

    assert regenie_config.trait.bsize == 2048
    assert regenie_config.g_compute.device == types.Device.GPU
    assert regenie_config.g_output.format == types.OutputFormat.ARROW
    assert regenie_config.g_diagnostics.log_filter == "g=debug"


def test_user_toml_binary_trait_overrides_default_quantitative_trait(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[input]",
                'bgen = "dataset.bgen"',
                'phenoFile = "phenotype.tsv"',
                'phenoCol = "trait"',
                'pred = "predictions.list"',
                "[trait]",
                "bt = true",
                "[output]",
                'out = "results/output"',
            ]
        ),
        encoding="utf-8",
    )

    regenie_config = config.RegenieConfig.from_toml(config_path)

    assert regenie_config.trait.trait_type == types.RegenieTraitType.BINARY


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
            "g-null-logistic-nonconvergence": "warn",
            "g-score-dtype": "float64",
            "g-firth-dtype": "float64",
            "g-jax-persistent-cache": False,
            "g-stage-timings-json": "timings.json",
            "g-log-filter": "g=trace",
            "g-log-file": "logs/g.jsonl",
            "g-log-stderr": False,
        }
    )
    config_path = tmp_path / "effective_config.toml"

    config.write_toml(regenie_config, config_path)
    loaded_config = config.RegenieConfig.from_toml(config_path)

    assert loaded_config == regenie_config


def test_logging_options_ignore_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G_LOG_FILTER", "g=info")
    monkeypatch.setenv("G_LOG_FILE", "logs/environment.jsonl")
    monkeypatch.setenv("G_LOG_STDERR", "false")
    monkeypatch.setenv("G_JAX_CACHE_DIR", "/ignored/g/cache")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/ignored/jax/cache")

    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "g-log-filter": "g=debug",
        }
    )

    assert regenie_config.g_diagnostics.log_filter == "g=debug"
    assert regenie_config.g_diagnostics.log_file is None
    assert regenie_config.g_diagnostics.log_stderr is True
    assert regenie_config.g_compute.jax_cache_dir is None


def test_unknown_and_unsupported_options_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="Unknown g regenie option"):
        config.RegenieConfig.from_options({"not_a_real_option": True})

    with pytest.raises(ValueError, match="Unknown g regenie option: g-allow-duplicate-iid-alignment"):
        config.RegenieConfig.from_options({"g-allow-duplicate-iid-alignment": True})

    with pytest.raises(ValueError, match="Unknown g regenie option: g-jax-enable-x64"):
        config.RegenieConfig.from_options({"g-jax-enable-x64": False})

    with pytest.raises(ValueError, match=r"Unknown g regenie option: g\.compute\.allow-duplicate-iid-alignment"):
        config.RegenieConfig.from_options({"g": {"compute": {"allow-duplicate-iid-alignment": True}}})

    with pytest.raises(ValueError, match="valid REGENIE option"):
        config.RegenieConfig.from_options({"pgen": "dataset", "phenoFile": "phenotype.tsv"})


@pytest.mark.parametrize(
    ("option_name", "error_match"),
    [
        ("bed", "--bed is a valid REGENIE option"),
        ("spa", "--spa is a valid REGENIE option"),
        ("keep", "--keep is a valid REGENIE option"),
    ],
)
def test_recognized_unsupported_options_use_specific_errors(option_name: str, error_match: str) -> None:
    raw_options = (
        {
            "step": 2,
            "bt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
        }
        if option_name == "spa"
        else build_valid_quantitative_options()
    )
    raw_options[option_name] = "unsupported.txt"

    with pytest.raises(ValueError, match=error_match):
        config.RegenieConfig.from_options(raw_options)


@pytest.mark.parametrize(
    ("mutated_options", "error_match"),
    [
        ({"step": 1}, "--step 1 is recognized"),
        ({"step": 3}, "requires --step 2"),
        ({"bgen": None}, "Exactly one genotype source"),
        ({"phenoFile": None}, "--phenoFile is required"),
        ({"phenoCol": None}, "At least one --phenoCol"),
        ({"pred": None}, "--pred is required"),
        ({"out": None}, "--out is required"),
        ({"bsize": 0}, "--bsize must be positive"),
        ({"threads": 0}, "--threads must be positive"),
        ({"g-variant-limit": 0}, "--g-variant-limit must be positive"),
        ({"g-writer-threads": 0}, "--g-writer-threads must be positive"),
        ({"g-writer-queue-depth": 0}, "--g-writer-queue-depth must be positive"),
        ({"g-output-chunks-per-arrow-file": 0}, "--g-output-chunks-per-arrow-file must be positive"),
    ],
)
def test_config_validation_rejects_required_and_positive_option_errors(
    mutated_options: dict[str, object],
    error_match: str,
) -> None:
    raw_options = build_valid_quantitative_options()
    raw_options.update(mutated_options)

    with pytest.raises(ValueError, match=error_match):
        config.RegenieConfig.from_options(raw_options)


@pytest.mark.parametrize(
    ("mutated_options", "error_match"),
    [
        ({"pThresh": 0.0}, "--pThresh must be in"),
        ({"pThresh": 1.0}, "--pThresh must be in"),
        ({"firth": True, "approx": False}, "Exact --firth is not implemented"),
        ({"firth": False, "approx": True}, "--approx requires --firth"),
    ],
)
def test_binary_config_validation_rejects_invalid_fallback_combinations(
    mutated_options: dict[str, object],
    error_match: str,
) -> None:
    raw_options: dict[str, object] = {
        "step": 2,
        "bt": True,
        "bgen": "dataset.bgen",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "pred": "predictions.list",
        "out": "results/output",
        "firth": True,
        "approx": True,
    }
    raw_options.update(mutated_options)

    with pytest.raises(ValueError, match=error_match):
        config.RegenieConfig.from_options(raw_options)


@pytest.mark.parametrize(
    ("mutated_options", "error_match"),
    [
        ({"g-firth-dtype": "float32"}, "--g-firth-dtype currently supports float64 only"),
    ],
)
def test_config_validation_rejects_invalid_dtype_policy(
    mutated_options: dict[str, object],
    error_match: str,
) -> None:
    raw_options = build_valid_quantitative_options()
    raw_options.update(mutated_options)

    with pytest.raises(ValueError, match=error_match):
        config.RegenieConfig.from_options(raw_options)


@pytest.mark.parametrize(
    ("mutated_options", "error_match"),
    [
        ({"g-gpu-genotype-format": "packed8", "g-device": "cpu"}, "--g-gpu-genotype-format=packed8 requires"),
        (
            {
                "g-gpu-genotype-format": "packed8",
                "g-device": "gpu",
                "phenoCol": ("first", "second"),
            },
            "packed8 currently supports one phenotype",
        ),
        (
            {
                "g-gpu-genotype-format": "packed8",
                "g-device": "gpu",
                "phenoCol": None,
                "phenoColList": "first,second",
            },
            "packed8 currently supports one phenotype",
        ),
    ],
)
def test_config_validation_rejects_unsupported_packed8_uses(
    mutated_options: dict[str, object],
    error_match: str,
) -> None:
    raw_options: dict[str, object] = {
        "step": 2,
        "bt": True,
        "bgen": "dataset.bgen",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "pred": "predictions.list",
        "out": "results/output",
        "g-device": "gpu",
    }
    raw_options.update(mutated_options)

    with pytest.raises(ValueError, match=error_match):
        config.RegenieConfig.from_options(raw_options)


def test_config_validation_accepts_quantitative_single_phenotype_packed8_gpu() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options.update({"g-gpu-genotype-format": "packed8", "g-device": "gpu"})

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE
    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.PACKED8


def test_repeated_and_list_columns_are_mutually_exclusive() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options["phenoCol"] = ("trait",)
    raw_options["phenoColList"] = "trait"

    with pytest.raises(ValueError, match="Use either --phenoCol or --phenoColList"):
        config.RegenieConfig.from_options(raw_options)


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("firth", True),
        ("approx", True),
        ("firth-se", True),
        ("spa", True),
        ("pThresh", config.default_float_option("pThresh")),
    ],
)
def test_quantitative_trait_rejects_explicit_binary_only_options(option_name: str, option_value: object) -> None:
    raw_options = build_valid_quantitative_options()
    raw_options[option_name] = option_value

    with pytest.raises(ValueError, match=f"--{option_name} can only be used with --bt"):
        config.RegenieConfig.from_options(raw_options)


def test_quantitative_trait_ignores_none_binary_only_python_options() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options.update(
        {
            "firth": None,
            "approx": None,
            "firth_se": None,
            "spa": None,
            "pThresh": None,
        }
    )

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE
    assert "firth" not in regenie_config.explicit_options
    assert "approx" not in regenie_config.explicit_options
    assert "firth-se" not in regenie_config.explicit_options
    assert "spa" not in regenie_config.explicit_options
    assert "pThresh" not in regenie_config.explicit_options


def test_quantitative_trait_accepts_defaulted_binary_threshold() -> None:
    regenie_config = config.RegenieConfig.from_options(build_valid_quantitative_options())

    assert regenie_config.binary.p_threshold == config.default_float_option("pThresh")


def test_output_tuning_defaults_come_from_packaged_default_config() -> None:
    default_options = config.load_default_option_dictionary()
    default_output_options = default_options["g"]["output"]
    regenie_config = config.RegenieConfig.from_options(build_valid_quantitative_options())

    assert regenie_config.g_output.writer_threads == default_output_options["writer-threads"]
    assert regenie_config.g_output.writer_queue_depth == default_output_options["writer-queue-depth"]
    assert regenie_config.g_output.chunks_per_arrow_file == default_output_options["chunks-per-arrow-file"]


def test_quantitative_execution_plan_rejects_direct_binary_only_config() -> None:
    regenie_config = config.RegenieConfig(
        input=config.InputConfig(
            bgen=Path("dataset.bgen"),
            pheno_file=Path("phenotype.tsv"),
            pheno_columns=("trait",),
            pred=Path("predictions.list"),
        ),
        trait=config.TraitConfig(trait_type=types.RegenieTraitType.QUANTITATIVE),
        binary=config.BinaryConfig(firth=True, approx=True, p_threshold=0.01),
        g_output=config.GOutputConfig(out=Path("results/output")),
    )

    with pytest.raises(ValueError, match="--firth, --approx, --pThresh can only be used with --bt"):
        execution_plan.build_regenie_execution_plan(regenie_config)


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


def test_config_helper_normalizers_cover_optional_and_trait_alias_paths() -> None:
    assert config.split_name_list(None) == ()
    assert config.split_name_list(" age, sex ,,") == ("age", "sex")
    assert config.optional_string(123) == "123"
    assert config.optional_string(None) is None
    assert config.normalize_option_name("trait_type") == "trait_type"
    assert config.normalize_option_name("g_null_logistic_nonconvergence_policy") == "g-null-logistic-nonconvergence"
    with pytest.raises(ValueError, match="--qt and --bt are mutually exclusive"):
        config.normalize_trait_type(qt=True, bt=True)


def test_flatten_option_dictionary_preserves_unknown_sections_and_g_scalars() -> None:
    flattened_options = config.flatten_option_dictionary(
        {
            "unknown": {"nested": "value"},
            "g": {
                "compute": {"device": "gpu"},
                "output": {"format": "arrow"},
                "diagnostics": {"log-file": "logs/g.jsonl"},
                "scalar": True,
            },
        }
    )

    assert flattened_options["unknown.nested"] == "value"
    assert flattened_options["g-device"] == "gpu"
    assert flattened_options["g-output-format"] == "arrow"
    assert flattened_options["g-log-file"] == "logs/g.jsonl"
    assert flattened_options["g.scalar"] is True


def test_config_positive_validation_helpers_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="--count must be positive"):
        config.validate_positive_integer("--count", 0)
    with pytest.raises(ValueError, match="--scale must be positive"):
        config.validate_positive_float("--scale", 0.0)
    with pytest.raises(ValueError, match=r"--probability must be less than 0\.5"):
        config.validate_probability_floor("--probability", 0.5)


def test_format_toml_value_serializes_lists_as_toml_arrays() -> None:
    serialized_value = config.format_toml_value(["trait_a", "trait_b"])

    assert serialized_value == '["trait_a", "trait_b"]'


def test_toml_serialization_emits_multi_column_and_binary_sections() -> None:
    regenie_config = config.RegenieConfig(
        input=config.InputConfig(
            bgen=Path("dataset.bgen"),
            sample=Path("dataset.sample"),
            pheno_file=Path("phenotype.tsv"),
            pheno_columns=("trait_a", "trait_b"),
            covar_file=Path("covariates.tsv"),
            covar_columns=("age", "sex"),
            pred=Path("predictions.list"),
        ),
        trait=config.TraitConfig(trait_type=types.RegenieTraitType.BINARY),
        binary=config.BinaryConfig(firth=True, approx=True, firth_se=True, spa=False, p_threshold=0.01),
        g_output=config.GOutputConfig(out=Path("results/output")),
    )

    config_text = regenie_config.to_toml()

    assert 'sample = "dataset.sample"' in config_text
    assert 'phenoColList = "trait_a,trait_b"' in config_text
    assert 'covarColList = "age,sex"' in config_text
    assert 'pred = "predictions.list"' in config_text
    assert "[binary]" in config_text
    assert "firth = true" in config_text
