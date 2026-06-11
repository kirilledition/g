from __future__ import annotations

import re
from pathlib import Path

import pytest

import g._core
from g import types
from g.interface import config


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


def build_input_config(**overrides: object) -> config.InputConfig:
    """Build packaged input config with test overrides."""
    pytest.skip("Outdated dataclass config helper; rebuild after Rust config API settles.")


def build_trait_config(**overrides: object) -> config.TraitConfig:
    """Build packaged trait config with test overrides."""
    pytest.skip("Outdated dataclass config helper; rebuild after Rust config API settles.")


def build_binary_config(**overrides: object) -> config.BinaryConfig:
    """Build packaged binary config with test overrides."""
    pytest.skip("Outdated dataclass config helper; rebuild after Rust config API settles.")


def build_output_config(**overrides: object) -> config.GOutputConfig:
    """Build packaged output config with test overrides."""
    pytest.skip("Outdated dataclass config helper; rebuild after Rust config API settles.")


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
        "device": "cpu",
        "staging_depth": 2,
        "result_in_flight_limit": 5,
        "dosage_buffer_limit": 6,
        "variant_limit": 100,
        "trusted_no_missing_diploid": True,
        "trusted_bgen_validation_mode": "assume_validated",
        "sample_key_mode": "iid",
        "multi_phenotype_sample_mode": "complete-case",
        "format": "arrow",
        "writer_threads": 2,
        "writer_queue_depth": 3,
        "chunks_per_arrow_file": 2,
        "arrow_compression": "none",
        "parquet_compression": "zstd",
        "firth_batch_size": 8,
        "firth_candidate_capacity": 16,
        "binary_null_maximum_iterations": 25,
        "binary_null_coefficient_tolerance": 1.0e-5,
        "null_logistic_nonconvergence_policy": "warn",
        "binary_minimum_probability": 1.0e-7,
        "binary_minimum_variance": 1.0e-9,
        "binary_relative_variance_tolerance": 2.0e-6,
        "linear_minimum_variance": 3.0e-9,
        "linear_relative_variance_tolerance": 4.0e-6,
        "firth_maximum_iterations": 30,
        "firth_gradient_tolerance": 1.0e-5,
        "firth_coefficient_tolerance": 1.0e-5,
        "firth_likelihood_tolerance": 1.0e-5,
        "firth_maximum_step_size": 4.0,
        "use_block_firth_math": True,
        "bgen_decode_tile_variant_count": 32,
        "gpu_genotype_format": "dosage",
        "score_dtype": "float64",
        "firth_dtype": "float64",
        "jax_cache_dir": "cache/jax",
        "jax_matmul_precision": "highest",
        "jax_persistent_cache": False,
        "jax_persistent_cache_min_entry_size_bytes": 1024,
        "jax_persistent_cache_min_compile_time_seconds": 1,
        "jax_xla_autotune_cache": True,
        "jax_transfer_guard": True,
        "telemetry": "trace",
        "log_dir": "logs",
        "stage_timings_json": "timings.json",
        "log_filter": "g=debug",
        "log_file": "logs/g.jsonl",
        "log_stderr": False,
        "progress_interval_seconds": 1.5,
        "progress_interval_chunks": 4,
        "profile_summary_json": "logs/profile.summary.json",
        "trace_file": "logs/trace.jsonl",
        "trace_filter": "g=trace",
        "trace_event_cap": 2048,
        "log_queue_size": 1024,
        "log_lossy": False,
        "include_source_location": True,
        "include_span_events": True,
    }

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.g_compute.trusted_bgen_validation_mode == types.TrustedBgenValidationMode.ASSUME_VALIDATED
    assert regenie_config.g_compute.result_in_flight_limit == 5
    assert regenie_config.g_compute.dosage_buffer_limit == 6
    assert regenie_config.g_compute.multi_phenotype_sample_mode == types.MultiPhenotypeSampleMode.COMPLETE_CASE
    assert regenie_config.g_compute.firth_batch_size == 8
    assert regenie_config.g_compute.firth_candidate_capacity == 16
    assert regenie_config.g_compute.binary_null_maximum_iterations == 25
    assert regenie_config.g_compute.binary_null_coefficient_tolerance == pytest.approx(1.0e-5)
    assert regenie_config.g_compute.null_logistic_nonconvergence_policy == (types.NullLogisticNonconvergencePolicy.WARN)
    assert regenie_config.g_compute.binary_minimum_probability == pytest.approx(1.0e-7)
    assert regenie_config.g_compute.binary_minimum_variance == pytest.approx(1.0e-9)
    assert regenie_config.g_compute.binary_relative_variance_tolerance == pytest.approx(2.0e-6)
    assert regenie_config.g_compute.linear_minimum_variance == pytest.approx(3.0e-9)
    assert regenie_config.g_compute.linear_relative_variance_tolerance == pytest.approx(4.0e-6)
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
    assert regenie_config.g_output.parquet_compression == types.ParquetCompression.ZSTD
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
    assert regenie_config.g_diagnostics.trace_event_cap == 2048
    assert regenie_config.g_diagnostics.log_queue_size == 1024
    assert regenie_config.g_diagnostics.log_lossy is False
    assert regenie_config.g_diagnostics.include_source_location is True
    assert regenie_config.g_diagnostics.include_span_events is True


def test_python_options_merge_flat_options_with_native_sections() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options.update(
        {
            "compute": {
                "device": "cpu",
                "variant_limit": 100,
            },
            "output": {
                "format": "parquet",
                "writer_threads": 1,
            },
        }
    )

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.g_output.out == Path("results/output")
    assert regenie_config.g_output.format == types.OutputFormat.PARQUET
    assert regenie_config.g_output.writer_threads == 1
    assert regenie_config.g_compute.device == types.Device.CPU
    assert regenie_config.g_compute.variant_limit == 100


def test_python_flat_option_schema_is_owned_by_native_metadata() -> None:
    expected_flat_option_sections: dict[str, tuple[str, str]] = {}
    expected_boolean_option_names: set[str] = set()
    for option_metadata in g._core.config_option_schema():
        for python_name in option_metadata["flat_python_names"]:
            assert not python_name.startswith(("g-", "g_"))
            expected_flat_option_sections[python_name] = (option_metadata["section"], option_metadata["toml_name"])
            if option_metadata["value_kind"] == "boolean":
                expected_boolean_option_names.add(python_name)

    assert expected_flat_option_sections == config.FLAT_OPTION_SECTIONS
    assert frozenset(expected_boolean_option_names) == config.BOOLEAN_PYTHON_OPTIONS
    assert config.FLAT_OPTION_SECTIONS["device"] == ("compute", "device")
    assert config.FLAT_OPTION_SECTIONS["phenoCol"] == ("input", "pheno_col")
    assert "g-device" not in config.FLAT_OPTION_SECTIONS


@pytest.mark.parametrize("option_name", ["g-device", "g_device", "g-output-format", "g_output_format", "pheno_file"])
def test_python_options_reject_undocumented_flat_aliases(option_name: str) -> None:
    raw_options = build_valid_quantitative_options()
    raw_options[option_name] = "ignored"

    with pytest.raises(ValueError, match=f"Unknown g regenie option: {re.escape(option_name)}"):
        config.RegenieConfig.from_options(raw_options)


def test_public_docs_do_not_reference_legacy_g_dash_flags() -> None:
    documentation_root = Path(__file__).resolve().parents[1] / "documentation" / "public"
    offenders: list[str] = []
    for documentation_path in documentation_root.rglob("*.md"):
        documentation_text = documentation_path.read_text(encoding="utf-8")
        for match in re.finditer(r"--g-[A-Za-z0-9_-]+", documentation_text):
            relative_path = documentation_path.relative_to(documentation_root.parent.parent)
            offenders.append(f"{relative_path}:{match.group(0)}")

    assert offenders == []


@pytest.mark.skip(reason="Outdated Python option metadata test; Rust config API is not settled.")
def test_every_supported_option_has_explain_metadata() -> None:
    pass


@pytest.mark.skip(reason="Outdated Python default catalog test; Rust config API is not settled.")
def test_packaged_default_catalog_matches_option_policies() -> None:
    pass


@pytest.mark.skip(reason="Outdated Python default catalog test; Rust config API is not settled.")
def test_packaged_default_hash_uses_raw_toml_payload() -> None:
    pass


@pytest.mark.skip(reason="Outdated Python TOML schema test; Rust config API is not settled.")
def test_typed_toml_schema_matches_option_registry() -> None:
    pass


@pytest.mark.skip(reason="Outdated Python TOML schema test; Rust config API is not settled.")
def test_packaged_default_toml_decodes_to_typed_config() -> None:
    pass


@pytest.mark.skip(reason="Outdated Python msgspec TOML schema test; Rust config API is not settled.")
def test_msgspec_toml_schema_rejects_unknown_keys_and_wrong_types() -> None:
    pass


@pytest.mark.skip(reason="Outdated Python msgspec TOML schema test; Rust config API is not settled.")
def test_msgspec_toml_schema_rejects_removed_jax_x64_option() -> None:
    pass


@pytest.mark.skip(reason="Outdated Python TOML schema test; Rust config API is not settled.")
def test_toml_metadata_is_accepted_but_not_an_option() -> None:
    pass


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
    diagnostics_config = config.load_packaged_config().g_diagnostics

    assert diagnostics_config.telemetry == types.TelemetryMode.PROGRESS
    assert diagnostics_config.log_dir is None
    assert diagnostics_config.log_filter == "info"
    assert diagnostics_config.log_file is None
    assert diagnostics_config.log_stderr is True
    assert diagnostics_config.progress_interval_seconds == 5
    assert diagnostics_config.progress_interval_chunks == 10
    assert diagnostics_config.profile_summary_json is None
    assert diagnostics_config.trace_file is None
    assert diagnostics_config.trace_event_cap == 1_000_000
    assert diagnostics_config.log_queue_size == 65536
    assert diagnostics_config.log_lossy is True
    assert diagnostics_config.include_source_location is False
    assert diagnostics_config.include_span_events is False


@pytest.mark.parametrize(
    "config_type",
    [
        config.InputConfig,
        config.TraitConfig,
        config.BinaryConfig,
        config.GComputeConfig,
        config.GOutputConfig,
        config.GDiagnosticsConfig,
        config.RegenieConfig,
    ],
)
def test_runtime_config_dataclasses_require_resolved_values(config_type: type[object]) -> None:
    with pytest.raises(TypeError):
        config_type()


def test_packaged_default_toml_is_loaded_for_python_options() -> None:
    regenie_config = config.RegenieConfig.from_options(build_valid_quantitative_options())
    packaged_config = config.load_packaged_config()

    assert regenie_config.trait.bsize == packaged_config.trait.bsize
    assert regenie_config.g_compute.device == types.Device.CPU
    assert regenie_config.g_compute.null_logistic_nonconvergence_policy == types.NullLogisticNonconvergencePolicy.FAIL
    assert regenie_config.g_compute.linear_minimum_variance == packaged_config.g_compute.linear_minimum_variance
    assert (
        regenie_config.g_compute.linear_relative_variance_tolerance
        == packaged_config.g_compute.linear_relative_variance_tolerance
    )
    assert regenie_config.g_compute.score_dtype == types.FloatingPointDtype.FLOAT32
    assert regenie_config.g_compute.firth_dtype == types.FloatingPointDtype.FLOAT64
    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.DOSAGE
    assert regenie_config.g_compute.jax_persistent_cache is True
    assert regenie_config.g_output.format == types.OutputFormat.PARQUET
    assert regenie_config.g_diagnostics.log_filter == packaged_config.g_diagnostics.log_filter
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
                'format = "arrow"',
                "[compute]",
                'device = "gpu"',
                "[diagnostics]",
                'log_filter = "g=debug"',
            ]
        ),
        encoding="utf-8",
    )

    regenie_config = config.RegenieConfig.from_toml(config_path)

    assert regenie_config.trait.bsize == 2048
    assert regenie_config.g_compute.device == types.Device.GPU
    assert regenie_config.g_output.format == types.OutputFormat.ARROW
    assert regenie_config.g_diagnostics.log_filter == "g=debug"


def test_python_options_accept_regenie_text_output_format() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options["format"] = "regenie"

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.g_output.format == types.OutputFormat.REGENIE


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
            "format": "arrow",
            "arrow_compression": "none",
            "parquet_compression": "zstd",
            "firth_batch_size": 8,
            "null_logistic_nonconvergence_policy": "warn",
            "score_dtype": "float64",
            "firth_dtype": "float64",
            "jax_persistent_cache": False,
            "stage_timings_json": "timings.json",
            "log_filter": "g=trace",
            "log_file": "logs/g.jsonl",
            "log_stderr": False,
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
            "log_filter": "g=debug",
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

    with pytest.raises(ValueError, match="Unknown g regenie option: pgen"):
        config.RegenieConfig.from_options({"pgen": "dataset", "phenoFile": "phenotype.tsv"})


@pytest.mark.parametrize(
    "option_name",
    [
        "bed",
        "pgen",
        "keep",
        "remove",
        "extract",
        "exclude",
        "catCovarList",
        "test",
        "t2e",
        "spa",
    ],
)
def test_unsupported_regenie_options_are_unknown(option_name: str) -> None:
    raw_options = build_valid_quantitative_options()
    raw_options[option_name] = "unsupported.txt"

    with pytest.raises(ValueError, match=f"Unknown g regenie option: {option_name}"):
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
        ({"bsize": 0}, "trait.bsize"),
        ({"threads": 0}, "trait.threads"),
        ({"result_in_flight_limit": 0}, "compute.result_in_flight_limit"),
        ({"dosage_buffer_limit": 0}, "compute.dosage_buffer_limit"),
        ({"variant_limit": 0}, "compute.variant_limit"),
        ({"linear_minimum_variance": 0.0}, "compute.linear_minimum_variance"),
        (
            {"linear_relative_variance_tolerance": 0.0},
            "compute.linear_relative_variance_tolerance",
        ),
        ({"writer_threads": 0}, "output.writer_threads"),
        ({"writer_queue_depth": 0}, "output.writer_queue_depth"),
        ({"chunks_per_arrow_file": 0}, "output.chunks_per_arrow_file"),
        ({"trace_event_cap": -1}, "diagnostics.trace_event_cap"),
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


def test_trace_event_cap_zero_disables_cap_in_config() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options["trace_event_cap"] = 0

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.g_diagnostics.trace_event_cap == 0


@pytest.mark.parametrize(
    ("mutated_options", "error_match"),
    [
        ({"pThresh": 1.0}, "binary.p_threshold"),
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
        ({"firth_dtype": "float32"}, "--firth_dtype currently supports float64 only"),
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
        ({"gpu_genotype_format": "packed8", "device": "cpu"}, "--gpu_genotype_format=packed8 requires"),
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
        "device": "gpu",
    }
    raw_options.update(mutated_options)

    with pytest.raises(ValueError, match=error_match):
        config.RegenieConfig.from_options(raw_options)


def test_config_validation_accepts_quantitative_single_phenotype_packed8_gpu() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options.update({"gpu_genotype_format": "packed8", "device": "gpu"})

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE
    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.PACKED8


def test_config_validation_accepts_quantitative_multi_phenotype_packed8_gpu() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options.update(
        {
            "gpu_genotype_format": "packed8",
            "device": "gpu",
            "phenoCol": ("first", "second"),
        }
    )

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.input.pheno_columns == ("first", "second")
    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.PACKED8


def test_repeated_and_list_columns_are_mutually_exclusive() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options["phenoCol"] = ("trait",)
    raw_options["phenoColList"] = "trait"

    with pytest.raises(ValueError, match="Use only one of pheno_columns, pheno_col, or pheno_col_list"):
        config.RegenieConfig.from_options(raw_options)


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("firth", True),
        ("approx", True),
        ("firth-se", True),
        ("pThresh", config.load_packaged_config().binary.p_threshold),
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
            "pThresh": None,
        }
    )

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE
    assert "firth" not in regenie_config.explicit_options
    assert "approx" not in regenie_config.explicit_options
    assert "firth-se" not in regenie_config.explicit_options
    assert "pThresh" not in regenie_config.explicit_options


def test_trait_flags_are_mutually_exclusive_within_one_layer() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options["bt"] = True

    with pytest.raises(ValueError, match="--qt and --bt are mutually exclusive"):
        config.RegenieConfig.from_options(raw_options)


def test_python_trait_type_alias_selects_binary_trait() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options.pop("qt")
    raw_options.update({"trait_type": "binary", "firth": True, "approx": True})

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.trait.trait_type == types.RegenieTraitType.BINARY


def test_python_boolean_string_options_are_parsed_strictly() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options.update({"jax_persistent_cache": "false", "jax_transfer_guard": "on"})

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.g_compute.jax_persistent_cache is False
    assert regenie_config.g_compute.jax_transfer_guard is True


def test_python_boolean_string_options_reject_ambiguous_values() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options["jax_persistent_cache"] = "maybe"

    with pytest.raises(ValueError, match="Boolean option value must be a bool"):
        config.RegenieConfig.from_options(raw_options)


def test_quantitative_trait_accepts_defaulted_binary_threshold() -> None:
    regenie_config = config.RegenieConfig.from_options(build_valid_quantitative_options())

    assert regenie_config.binary.p_threshold == config.load_packaged_config().binary.p_threshold


def test_output_tuning_defaults_come_from_packaged_default_config() -> None:
    packaged_config = config.load_packaged_config()
    regenie_config = config.RegenieConfig.from_options(build_valid_quantitative_options())

    assert regenie_config.g_output.writer_threads == packaged_config.g_output.writer_threads
    assert regenie_config.g_output.writer_queue_depth == packaged_config.g_output.writer_queue_depth
    assert regenie_config.g_output.chunks_per_arrow_file == packaged_config.g_output.chunks_per_arrow_file
    assert regenie_config.g_output.parquet_compression == packaged_config.g_output.parquet_compression
    assert regenie_config.g_output.finalize_parquet is False


@pytest.mark.skip(reason="Rust-owned config objects are no longer dataclasses; rebuild this test with native helpers.")
def test_quantitative_execution_plan_rejects_direct_binary_only_config() -> None:
    pass


def test_staging_depth_must_be_positive() -> None:
    raw_options: dict[str, object] = {
        "step": 2,
        "qt": True,
        "bgen": "dataset.bgen",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "pred": "predictions.list",
        "out": "results/output",
        "staging_depth": 0,
    }

    with pytest.raises(ValueError, match=r"compute\.staging_depth"):
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


@pytest.mark.skip(reason="Outdated Python alias normalization test; Rust frontend accepts canonical CLI names only.")
def test_config_helper_normalizers_cover_optional_and_trait_alias_paths() -> None:
    assert config.split_name_list(None) == ()
    assert config.split_name_list(" age, sex ,,") == ("age", "sex")
    assert config.optional_string(123) == "123"
    assert config.optional_string(None) is None
    assert config.normalize_option_name("trait_type") == "trait_type"
    assert config.normalize_option_name("g_null_logistic_nonconvergence_policy") == (
        "null_logistic_nonconvergence_policy"
    )
    with pytest.raises(ValueError, match="--qt and --bt are mutually exclusive"):
        config.normalize_trait_type(qt=True, bt=True)


def test_flatten_toml_mapping_preserves_unknown_nested_sections() -> None:
    flattened_options = config.flatten_toml_mapping(
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
    assert flattened_options["g.compute.device"] == "gpu"
    assert flattened_options["g.output.format"] == "arrow"
    assert flattened_options["g.diagnostics.log-file"] == "logs/g.jsonl"
    assert flattened_options["g.scalar"] is True


@pytest.mark.skip(reason="Rust-owned config objects are no longer dataclasses; rebuild this test with native helpers.")
def test_toml_serialization_emits_multi_column_and_binary_sections() -> None:
    pass
