from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import pytest

import g._core
from g import types
from g.interface import config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "crates" / "interface" / "src" / "config.default.toml"


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
        "device": "cpu",
        "staging_depth": 2,
        "native_callback_batch_size": 3,
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
        "output_statistic_dtype": "float64",
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
    assert regenie_config.g_compute.native_callback_batch_size == 3
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
    assert regenie_config.g_output.output_statistic_dtype == types.FloatingPointDtype.FLOAT64
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
    expected_flat_option_sections: dict[str, config.FlatOptionTarget] = {}
    expected_boolean_option_names: set[str] = set()
    for option_metadata in g._core.config_option_schema():
        for python_name in option_metadata["flat_python_names"]:
            assert not python_name.startswith(("g-", "g_"))
            expected_flat_option_sections[python_name] = config.FlatOptionTarget(
                section_name=option_metadata["section"],
                option_name=option_metadata["toml_name"],
            )
            if option_metadata["value_kind"] == "boolean":
                expected_boolean_option_names.add(python_name)

    assert expected_flat_option_sections == config.FLAT_OPTION_SECTIONS
    assert frozenset(expected_boolean_option_names) == config.BOOLEAN_PYTHON_OPTIONS
    assert config.FLAT_OPTION_SECTIONS["device"] == config.FlatOptionTarget(
        section_name="compute",
        option_name="device",
    )
    assert config.FLAT_OPTION_SECTIONS["phenoCol"] == config.FlatOptionTarget(
        section_name="input",
        option_name="pheno_col",
    )
    assert "g-device" not in config.FLAT_OPTION_SECTIONS


@pytest.mark.parametrize("option_name", ["g-device", "g_device", "g-output-format", "g_output_format", "pheno_file"])
def test_python_options_reject_undocumented_flat_aliases(option_name: str) -> None:
    raw_options = build_valid_quantitative_options()
    raw_options[option_name] = "ignored"

    with pytest.raises(ValueError, match=f"Unknown g regenie option: {re.escape(option_name)}"):
        config.RegenieConfig.from_options(raw_options)


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("phenoCol", ""),
        ("phenoColList", ""),
        ("covarCol", ""),
        ("covarColList", ""),
        ("phenoColList", "trait_a,,trait_b"),
        ("covarColList", "age,,sex"),
        ("phenoColList", ["trait_a", "", "trait_b"]),
        ("covarCol", ["age", " ", "sex"]),
    ],
)
def test_python_options_reject_empty_selected_column_values(option_name: str, option_value: object) -> None:
    raw_options = build_valid_quantitative_options()
    raw_options[option_name] = option_value

    with pytest.raises(ValueError, match="empty entry"):
        config.RegenieConfig.from_options(raw_options)


def test_python_options_reject_nested_empty_selected_column_values() -> None:
    raw_options: dict[str, object] = {
        "input": {
            "bgen": "dataset.bgen",
            "pheno_file": "phenotype.tsv",
            "pheno_columns": ["trait_a", "", "trait_b"],
            "pred": "predictions.list",
        },
        "trait": {"step": 2, "qt": True},
        "output": {"out": "results/output"},
    }

    with pytest.raises(ValueError, match="empty entry"):
        config.RegenieConfig.from_options(raw_options)


def test_python_options_reject_empty_selected_column_lists() -> None:
    raw_options: dict[str, object] = {
        "input": {
            "bgen": "dataset.bgen",
            "pheno_file": "phenotype.tsv",
            "pheno_columns": [],
            "pred": "predictions.list",
        },
        "trait": {"step": 2, "qt": True},
        "output": {"out": "results/output"},
    }

    with pytest.raises(ValueError, match="at least one name"):
        config.RegenieConfig.from_options(raw_options)


def test_python_options_reject_nested_none_values() -> None:
    raw_options: dict[str, object] = {
        "input": {
            "bgen": "dataset.bgen",
            "pheno_file": None,
            "pheno_col": "trait",
            "pred": "predictions.list",
        },
        "trait": {"step": 2, "qt": True},
        "output": {"out": "results/output"},
    }

    with pytest.raises(ValueError, match="do not accept None"):
        config.RegenieConfig.from_options(raw_options)


def test_python_options_accept_pathlike_values() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options["bgen"] = Path("dataset.bgen")
    raw_options["phenoFile"] = Path("phenotype.tsv")
    raw_options["pred"] = Path("predictions.list")
    raw_options["out"] = Path("results/output")

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.input.bgen == Path("dataset.bgen")
    assert regenie_config.input.pheno_file == Path("phenotype.tsv")
    assert regenie_config.input.pred == Path("predictions.list")
    assert regenie_config.g_output.out == Path("results/output")


def test_python_options_reject_unsupported_object_values() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options["out"] = object()

    with pytest.raises(TypeError, match="Unsupported Python option value type"):
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


def test_every_supported_option_has_native_surface_metadata() -> None:
    supported_value_kinds = {
        "boolean",
        "float",
        "integer",
        "name-list",
        "path",
        "string",
        "string-enum",
    }
    for option_metadata in g._core.config_option_schema():
        assert set(option_metadata) == {
            "section",
            "toml_name",
            "accepted_toml_names",
            "cli_long_name",
            "negative_cli_long_name",
            "flat_python_names",
            "value_kind",
        }
        assert option_metadata["section"] in config.NATIVE_CONFIG_SECTION_NAMES | {"config"}
        assert isinstance(option_metadata["toml_name"], str)
        assert option_metadata["toml_name"]
        assert option_metadata["value_kind"] in supported_value_kinds
        assert isinstance(option_metadata["accepted_toml_names"], list)
        assert isinstance(option_metadata["flat_python_names"], list)
        if option_metadata["negative_cli_long_name"] is not None:
            assert option_metadata["value_kind"] == "boolean"


def test_packaged_default_catalog_matches_option_policies() -> None:
    default_payload = tomllib.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    known_options_by_section: dict[str, set[str]] = {}
    for option_metadata in g._core.config_option_schema():
        section_name = option_metadata["section"]
        if section_name == "config":
            continue
        known_options_by_section.setdefault(section_name, set()).add(option_metadata["toml_name"])

    assert set(default_payload) <= set(known_options_by_section)
    assert {"trait", "binary", "compute", "output", "diagnostics"} <= set(default_payload)
    for section_name, section_payload in default_payload.items():
        assert isinstance(section_payload, dict)
        assert set(section_payload) <= known_options_by_section[section_name]


def test_packaged_default_hash_uses_raw_toml_payload() -> None:
    expected_hash = hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()
    effective_default_payload = tomllib.loads(config.dumps_toml(config.load_packaged_config()))

    assert effective_default_payload["metadata"]["default-config-hash"] == expected_hash
    assert effective_default_payload["metadata"]["option-schema-version"] == 2


def test_typed_toml_schema_matches_option_registry() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "bt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoColList": "trait_a,trait_b",
            "covarCol": "age",
            "pred": "predictions.list",
            "out": "results/output",
            "firth": True,
            "approx": True,
            "firth-se": True,
        }
    )
    effective_payload = tomllib.loads(config.dumps_toml(regenie_config))
    known_options_by_section: dict[str, set[str]] = {}
    for option_metadata in g._core.config_option_schema():
        section_name = option_metadata["section"]
        if section_name == "config":
            continue
        known_options_by_section.setdefault(section_name, set()).add(option_metadata["toml_name"])
    known_options_by_section["input"].update({"pheno_columns", "covar_columns"})

    for section_name, section_payload in effective_payload.items():
        if section_name == "metadata":
            continue
        assert isinstance(section_payload, dict)
        assert set(section_payload) <= known_options_by_section[section_name]


def test_packaged_default_toml_decodes_to_typed_config() -> None:
    packaged_config = config.load_packaged_config()
    packaged_payload = tomllib.loads(config.dumps_toml(packaged_config))

    assert packaged_payload["trait"]["step"] == packaged_config.trait.step
    assert packaged_payload["trait"]["trait_type"] == packaged_config.trait.trait_type.value
    assert packaged_payload["binary"]["p_threshold"] == pytest.approx(packaged_config.binary.p_threshold)
    assert packaged_payload["compute"]["device"] == packaged_config.g_compute.device.value
    assert packaged_payload["output"]["format"] == packaged_config.g_output.format.value
    assert packaged_payload["diagnostics"]["telemetry"] == packaged_config.g_diagnostics.telemetry.value


@pytest.mark.parametrize(
    ("toml_text", "error_match"),
    [
        ("[unknown]\nvalue = true\n", "unknown field"),
        ('[trait]\nstep = "two"\n', "invalid type"),
    ],
)
def test_native_toml_schema_rejects_unknown_keys_and_wrong_types(
    tmp_path: Path,
    toml_text: str,
    error_match: str,
) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(toml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        config.RegenieConfig.from_toml(config_path)


def test_native_toml_schema_rejects_removed_jax_x64_option(tmp_path: Path) -> None:
    config_path = tmp_path / "removed.toml"
    config_path.write_text("[compute]\njax_x64 = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unknown field.*jax_x64"):
        config.RegenieConfig.from_toml(config_path)


def test_toml_metadata_is_accepted_but_not_an_option(tmp_path: Path) -> None:
    config_path = tmp_path / "effective.toml"
    effective_toml = config.dumps_toml(config.RegenieConfig.from_options(build_valid_quantitative_options()))
    config_path.write_text(f'{effective_toml}ignored-user-field = "ignored"\n', encoding="utf-8")

    regenie_config = config.RegenieConfig.from_toml(config_path)
    emitted_payload = tomllib.loads(config.dumps_toml(regenie_config))

    assert "metadata" not in config.FLAT_OPTION_SECTIONS
    assert "ignored-user-field" not in emitted_payload["metadata"]


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
    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.AUTO
    assert regenie_config.g_compute.jax_persistent_cache is True
    assert regenie_config.g_output.format == types.OutputFormat.PARQUET
    assert regenie_config.g_output.output_statistic_dtype == types.FloatingPointDtype.FLOAT32
    assert regenie_config.g_diagnostics.log_filter == packaged_config.g_diagnostics.log_filter


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


def test_gpu_genotype_format_auto_is_accepted_from_python_options() -> None:
    raw_options = build_valid_quantitative_options()
    raw_options["gpu_genotype_format"] = "auto"

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.AUTO


def test_gpu_genotype_format_auto_is_accepted_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[input]",
                'bgen = "dataset.bgen"',
                'phenoFile = "phenotype.tsv"',
                'phenoCol = "trait"',
                'pred = "predictions.list"',
                "[compute]",
                'gpu_genotype_format = "auto"',
                "[output]",
                'out = "results/output"',
            ]
        ),
        encoding="utf-8",
    )

    regenie_config = config.RegenieConfig.from_toml(config_path)

    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.AUTO


def test_gpu_genotype_format_auto_is_accepted_from_cli(tmp_path: Path) -> None:
    bgen_path = tmp_path / "dataset.bgen"
    phenotype_path = tmp_path / "phenotype.tsv"
    prediction_path = tmp_path / "predictions.list"
    bgen_path.write_bytes(b"")
    phenotype_path.write_text("FID IID trait\n", encoding="utf-8")
    prediction_path.write_text("", encoding="utf-8")

    cli_outcome = g._core.dispatch_cli(
        [
            "regenie",
            "--step",
            "2",
            "--qt",
            "--bgen",
            str(bgen_path),
            "--phenoFile",
            str(phenotype_path),
            "--phenoCol",
            "trait",
            "--pred",
            str(prediction_path),
            "--out",
            str(tmp_path / "results" / "output"),
            "--gpu_genotype_format",
            "auto",
        ]
    )

    assert cli_outcome.exit_code == 0
    assert cli_outcome.config is not None
    assert cli_outcome.config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.AUTO


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
            "output_statistic_dtype": "float64",
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
        ({"bgen": None}, "Option bgen does not accept None"),
        ({"phenoFile": None}, "Option phenoFile does not accept None"),
        ({"phenoCol": None}, "Option phenoCol does not accept None"),
        ({"pred": None}, "Option pred does not accept None"),
        ({"out": None}, "Option out does not accept None"),
        ({"bsize": 0}, "trait.bsize"),
        ({"threads": 0}, "trait.threads"),
        ({"native_callback_batch_size": 0}, "compute.native_callback_batch_size"),
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


def test_config_validation_accepts_auto_on_cpu() -> None:
    raw_options: dict[str, object] = {
        "step": 2,
        "bt": True,
        "bgen": "dataset.bgen",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "pred": "predictions.list",
        "out": "results/output",
        "device": "cpu",
        "gpu_genotype_format": "auto",
    }

    regenie_config = config.RegenieConfig.from_options(raw_options)

    assert regenie_config.g_compute.device == types.Device.CPU
    assert regenie_config.g_compute.gpu_genotype_format == types.GpuGenotypeFormat.AUTO


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


@pytest.mark.parametrize("option_name", ["firth", "approx", "firth_se", "pThresh"])
def test_python_options_reject_none_binary_only_options(option_name: str) -> None:
    raw_options = build_valid_quantitative_options()
    raw_options[option_name] = None

    with pytest.raises(ValueError, match=f"Option {option_name} does not accept None"):
        config.RegenieConfig.from_options(raw_options)


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
    assert regenie_config.g_output.output_statistic_dtype == packaged_config.g_output.output_statistic_dtype
    assert regenie_config.g_output.finalize_parquet is False


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


def test_config_helper_normalizers_cover_optional_and_trait_validation() -> None:
    assert config.split_name_list(None) == ()
    assert config.split_name_list(" age, sex ") == ("age", "sex")
    assert config.optional_string(123) == "123"
    assert config.optional_string(None) is None
    with pytest.raises(ValueError, match="--qt and --bt are mutually exclusive"):
        config.normalize_trait_type(qt=True, bt=True, trait_type=None)


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


def test_toml_serialization_emits_multi_column_and_binary_sections() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "bt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoColList": "trait_a,trait_b",
            "covarColList": "age,sex",
            "pred": "predictions.list",
            "out": "results/output",
            "firth": True,
            "approx": True,
            "pThresh": 0.01,
            "firth-se": True,
        }
    )

    toml_payload = tomllib.loads(config.dumps_toml(regenie_config))

    assert toml_payload["input"]["pheno_columns"] == ["trait_a", "trait_b"]
    assert toml_payload["input"]["covar_columns"] == ["age", "sex"]
    assert toml_payload["binary"] == {
        "firth": True,
        "approx": True,
        "p_threshold": 0.01,
        "firth_se": True,
    }
