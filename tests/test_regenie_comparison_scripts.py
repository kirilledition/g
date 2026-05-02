from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))


def load_script_module(module_name: str, relative_path: str):
    module_path = REPOSITORY_ROOT / relative_path
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


baseline_benchmark = load_script_module("baseline_benchmark_script", "scripts/benchmark.py")
bgen_reader_benchmark = load_script_module("bgen_reader_benchmark_script", "scripts/benchmark_bgen_reader.py")
comparison_benchmark = load_script_module("comparison_benchmark_script", "scripts/benchmark_regenie_comparison.py")
comparison_profile = load_script_module("comparison_profile_script", "scripts/profile_regenie_comparison.py")
fresh_process_benchmark = load_script_module(
    "fresh_process_benchmark_script",
    "scripts/benchmark_regenie2_linear_fresh_process.py",
)


def test_regenie_command_builders_shape() -> None:
    baseline_paths = baseline_benchmark.build_baseline_paths()
    command_specs = comparison_benchmark.build_regenie_program_specs("regenie", baseline_paths)
    assert len(command_specs) == 4
    assert command_specs[0][0] == "regenie_step1_binary"
    assert "--step" in command_specs[0][3]
    assert "--bt" in command_specs[0][3]
    assert command_specs[1][0] == "regenie_step2_binary"
    assert "--bgen" in command_specs[1][3]
    assert command_specs[2][0] == "regenie_step1_quantitative"
    assert "--qt" in command_specs[2][3]
    assert command_specs[3][0] == "regenie_step2_quantitative"
    assert "--pred" in command_specs[3][3]


def test_bgen_reader_benchmark_parses_sweep_lists() -> None:
    assert bgen_reader_benchmark.parse_optional_int_list("8192,16384") == [8192, 16384]
    assert bgen_reader_benchmark.parse_optional_int_list("default,4") == [None, 4]


def test_bgen_reader_benchmark_parses_path_modes() -> None:
    path_modes = bgen_reader_benchmark.parse_path_modes("read_float32,read_float32_prepared,read_float32_into_prepared")
    assert [path_mode.value for path_mode in path_modes] == [
        "read_float32",
        "read_float32_prepared",
        "read_float32_into_prepared",
    ]


def test_bgen_reader_benchmark_parses_boolean_modes() -> None:
    assert bgen_reader_benchmark.parse_boolean_mode_list("trusted,safe") == [True, False]


def test_regenie_command_builders_can_focus_quantitative_step2() -> None:
    baseline_paths = baseline_benchmark.build_baseline_paths()
    command_specs = comparison_benchmark.build_regenie_program_specs(
        "regenie",
        baseline_paths,
        only_quantitative_step2=True,
    )
    assert len(command_specs) == 1
    assert command_specs[0][0] == "regenie_step2_quantitative"
    assert "--step" in command_specs[0][3]
    assert command_specs[0][3][command_specs[0][3].index("--step") + 1] == "2"
    assert "--qt" in command_specs[0][3]


def test_g_comparison_runner_builds_cpu_and_gpu_commands() -> None:
    baseline_paths = baseline_benchmark.build_baseline_paths()
    cpu_command = comparison_benchmark.build_g_step2_command(
        uv_executable="uv",
        baseline_paths=baseline_paths,
        output_prefix=Path("data/benchmarks/out_cpu"),
        device="cpu",
        chunk_size=512,
        variant_limit=1024,
        output_writer_backend="python",
    )
    gpu_command = comparison_benchmark.build_g_step2_command(
        uv_executable="uv",
        baseline_paths=baseline_paths,
        output_prefix=Path("data/benchmarks/out_gpu"),
        device="gpu",
        chunk_size=2048,
        variant_limit=None,
        output_writer_backend="python",
    )
    binary_command = comparison_benchmark.build_g_step2_command(
        uv_executable="uv",
        baseline_paths=baseline_paths,
        output_prefix=Path("data/benchmarks/out_bin"),
        device="cpu",
        chunk_size=8192,
        variant_limit=None,
        output_writer_backend="rust",
        trait_type="binary",
    )
    assert cpu_command[:4] == ["uv", "run", "g", "regenie2"]
    assert "--trait-type" in cpu_command
    assert cpu_command[cpu_command.index("--trait-type") + 1] == "quantitative"
    assert "--device" in cpu_command
    assert cpu_command[cpu_command.index("--device") + 1] == "cpu"
    assert cpu_command[cpu_command.index("--output-writer-backend") + 1] == "python"
    assert "--finalize-parquet" in cpu_command
    assert "--variant-limit" in cpu_command
    assert gpu_command[gpu_command.index("--device") + 1] == "gpu"
    assert "--variant-limit" not in gpu_command
    assert binary_command[binary_command.index("--trait-type") + 1] == "binary"
    assert binary_command[binary_command.index("--output-writer-backend") + 1] == "rust"
    assert "phenotype_binary" in binary_command


def test_unsupported_g_program_result_marked_not_implemented() -> None:
    result = comparison_benchmark.build_not_implemented_result(
        program_name="g_regenie2_binary_step1",
        trait_type="binary",
        step=1,
        device="cpu",
    )
    assert result.status == "not_implemented"
    assert result.implementation == "g"
    assert result.notes is not None


def test_profiled_subprocess_wrapper_metadata(tmp_path: Path) -> None:
    stdout_log_path = tmp_path / "stdout.log"
    stderr_log_path = tmp_path / "stderr.log"
    success, wall_time_seconds, peak_rss_megabytes, cpu_user_seconds, cpu_system_seconds, error_message = (
        comparison_profile.run_profiled_subprocess(
            command_arguments=[sys.executable, "-c", "import time; print('ok'); time.sleep(0.05)"],
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            sample_interval_seconds=0.01,
        )
    )
    assert success
    assert wall_time_seconds > 0.0
    assert peak_rss_megabytes is not None
    assert peak_rss_megabytes >= 0.0
    assert cpu_user_seconds >= 0.0
    assert cpu_system_seconds >= 0.0
    assert error_message is None
    assert "ok" in stdout_log_path.read_text()


def test_summary_serializer_json_shape() -> None:
    result = comparison_benchmark.ComparisonProgramResult(
        program_name="regenie_step2_quantitative",
        implementation="regenie",
        trait_type="quantitative",
        step=2,
        device="external_cpu",
        status="success",
        wall_time_seconds=12.3,
        variants_per_second=1000.0,
        peak_memory_megabytes=None,
        stdout_log_path="stdout.log",
        stderr_log_path="stderr.log",
        output_paths=["out.regenie"],
        output_row_count=100,
        prediction_list_present=None,
    )
    payload = {"results": [result.__dict__]}
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert isinstance(decoded["results"], list)
    assert decoded["results"][0]["program_name"] == "regenie_step2_quantitative"
    assert decoded["results"][0]["status"] == "success"


def test_text_summary_includes_required_sections(tmp_path: Path) -> None:
    results = [
        comparison_benchmark.ComparisonProgramResult(
            program_name="regenie_step2_quantitative",
            implementation="regenie",
            trait_type="quantitative",
            step=2,
            device="external_cpu",
            status="success",
            wall_time_seconds=20.0,
            variants_per_second=100.0,
            peak_memory_megabytes=None,
            stdout_log_path=None,
            stderr_log_path=None,
            output_paths=[],
            output_row_count=1000,
            prediction_list_present=None,
        ),
        comparison_benchmark.ComparisonProgramResult(
            program_name="g_regenie2_quantitative_step2_cpu",
            implementation="g",
            trait_type="quantitative",
            step=2,
            device="cpu",
            status="success",
            wall_time_seconds=10.0,
            variants_per_second=200.0,
            peak_memory_megabytes=None,
            stdout_log_path=None,
            stderr_log_path=None,
            output_paths=[],
            output_row_count=1000,
            prediction_list_present=None,
        ),
        comparison_benchmark.ComparisonProgramResult(
            program_name="g_regenie2_quantitative_step2_gpu",
            implementation="g",
            trait_type="quantitative",
            step=2,
            device="gpu",
            status="not_implemented",
            wall_time_seconds=None,
            variants_per_second=None,
            peak_memory_megabytes=None,
            stdout_log_path=None,
            stderr_log_path=None,
            output_paths=[],
            output_row_count=None,
            prediction_list_present=None,
            notes="not_implemented",
        ),
    ]
    agreement = comparison_benchmark.QuantitativeStep2Agreement(
        comparable=True,
        merged_variant_count=1000,
        beta_max_abs_error=1.0e-4,
        beta_mean_abs_error=1.0e-5,
        beta_allclose_within_tolerance=True,
        log10p_max_abs_error=1.0e-4,
        log10p_mean_abs_error=1.0e-5,
        log10p_allclose_within_tolerance=True,
    )
    report_path = tmp_path / "summary.txt"
    comparison_benchmark.write_text_summary(
        report_path=report_path,
        results=results,
        agreement_cpu=agreement,
        agreement_gpu=None,
    )
    summary = report_path.read_text()
    assert "regenie_step2_quantitative" in summary
    assert "g_regenie2_quantitative_step2_cpu" in summary
    assert "Direct Runtime Comparisons" in summary
    assert "Numeric Agreement" in summary


def test_quantitative_step2_comparison_wires_parity_logic(tmp_path: Path) -> None:
    regenie_output = tmp_path / "regenie.regenie"
    g_output = tmp_path / "g.parquet"
    regenie_output.write_text("CHROM GENPOS ID BETA LOG10P\n1 100 rs1 0.1 1.0\n1 200 rs2 0.2 2.0\n")
    pd.DataFrame(
        {
            "variant_identifier": ["rs1", "rs2"],
            "beta": [0.1, 0.2],
            "log10_p_value": [1.0, 2.0],
        }
    ).to_parquet(g_output, index=False)
    agreement = comparison_benchmark.summarize_quantitative_step2_agreement(
        regenie_output_path=regenie_output,
        g_output_path=g_output,
    )
    assert agreement.comparable
    assert agreement.merged_variant_count == 2
    assert agreement.beta_allclose_within_tolerance is True
    assert agreement.log10p_allclose_within_tolerance is True


def test_fresh_process_benchmark_parser_accepts_output_writer_options() -> None:
    arguments = fresh_process_benchmark.build_argument_parser().parse_args(
        [
            "--output-writer-thread-count",
            "2",
        ]
    )
    assert arguments.output_writer_thread_count == 2


def test_fresh_process_benchmark_summary_tracks_output_metrics() -> None:
    trial_results = [
        fresh_process_benchmark.TrialResult(
            trial_index=0,
            wall_time_seconds=2.0,
            output_path="out0",
            output_row_count=100,
            chunk_file_count=2,
            chunk_bytes=1024,
            final_parquet_bytes=512,
        ),
        fresh_process_benchmark.TrialResult(
            trial_index=1,
            wall_time_seconds=1.0,
            output_path="out1",
            output_row_count=100,
            chunk_file_count=2,
            chunk_bytes=2048,
            final_parquet_bytes=1024,
        ),
    ]
    summary = fresh_process_benchmark.build_summary(
        device="gpu",
        chunk_size=8192,
        finalize_parquet=True,
        arrow_payload_batch_size=1,
        output_writer_thread_count=2,
        warmup_count=1,
        trial_results=trial_results,
    )
    assert summary.mean_rows_per_second == 75.0
    assert summary.mean_chunk_bytes == 1536.0
    assert summary.mean_final_parquet_bytes == 768.0


def test_quantitative_step2_comparison_uses_full_variant_identity_when_available(tmp_path: Path) -> None:
    regenie_output = tmp_path / "regenie.regenie"
    g_output = tmp_path / "g.parquet"
    regenie_output.write_text(
        "\n".join(
            [
                "CHROM GENPOS ID ALLELE0 ALLELE1 BETA LOG10P",
                "1 100 rs1 A G 0.1 1.0",
                "1 101 rs1 A T 0.9 9.0",
            ]
        )
        + "\n"
    )
    pd.DataFrame(
        {
            "chromosome": [1],
            "position": [100],
            "variant_identifier": ["rs1"],
            "allele_one": ["G"],
            "allele_two": ["A"],
            "beta": [0.1],
            "log10_p_value": [1.0],
        }
    ).to_parquet(g_output, index=False)
    agreement = comparison_benchmark.summarize_quantitative_step2_agreement(
        regenie_output_path=regenie_output,
        g_output_path=g_output,
    )
    assert agreement.comparable
    assert agreement.merged_variant_count == 1
    assert agreement.beta_allclose_within_tolerance is True
    assert agreement.log10p_allclose_within_tolerance is True


def test_quantitative_step2_comparison_coerces_merge_key_types(tmp_path: Path) -> None:
    regenie_output = tmp_path / "regenie.regenie"
    g_output = tmp_path / "g.parquet"
    regenie_output.write_text(
        "\n".join(
            [
                "CHROM GENPOS ID ALLELE0 ALLELE1 BETA LOG10P",
                "22 100 rs1 A G 0.1 1.0",
            ]
        )
        + "\n"
    )
    pd.DataFrame(
        {
            "chromosome": ["22"],
            "position": [100],
            "variant_identifier": ["rs1"],
            "allele_one": ["G"],
            "allele_two": ["A"],
            "beta": [0.1],
            "log10_p_value": [1.0],
        }
    ).to_parquet(g_output, index=False)
    agreement = comparison_benchmark.summarize_quantitative_step2_agreement(
        regenie_output_path=regenie_output,
        g_output_path=g_output,
    )
    assert agreement.comparable
    assert agreement.merged_variant_count == 1


def test_quantitative_step2_comparison_reads_parquet_outputs(tmp_path: Path) -> None:
    regenie_output = tmp_path / "regenie.regenie"
    g_output = tmp_path / "g.parquet"
    regenie_output.write_text("CHROM GENPOS ID BETA LOG10P\n1 100 rs1 0.1 1.0\n1 200 rs2 0.2 2.0\n")
    pd.DataFrame(
        {
            "variant_identifier": ["rs1", "rs2"],
            "beta": [0.1, 0.2],
            "log10_p_value": [1.0, 2.0],
        }
    ).to_parquet(g_output, index=False)
    agreement = comparison_benchmark.summarize_quantitative_step2_agreement(
        regenie_output_path=regenie_output,
        g_output_path=g_output,
    )
    assert agreement.comparable
    assert agreement.merged_variant_count == 2
    assert agreement.beta_allclose_within_tolerance is True
    assert agreement.log10p_allclose_within_tolerance is True
