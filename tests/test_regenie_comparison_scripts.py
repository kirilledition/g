from __future__ import annotations

import importlib.util
import json
import sys
import typing
from pathlib import Path

import polars as pl

if typing.TYPE_CHECKING:
    import pytest

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
deep_profile = load_script_module("deep_profile_script", "scripts/profile_regenie2_deep.py")
fresh_process_benchmark = load_script_module(
    "fresh_process_benchmark_script",
    "scripts/benchmark_regenie2_linear_fresh_process.py",
)
binary_hot_benchmark = load_script_module(
    "binary_hot_benchmark_script",
    "scripts/benchmark_regenie2_binary_hot.py",
)
binary_firth_parity = load_script_module(
    "binary_firth_parity_script",
    "scripts/compare_binary_firth_paths.py",
)
tuning_benchmark = load_script_module(
    "tuning_benchmark_script",
    "scripts/tune_regenie2_gpu.py",
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
    path_modes = bgen_reader_benchmark.parse_path_modes("sample_major_buffered,variant_major_buffered")
    assert [path_mode.value for path_mode in path_modes] == [
        "sample_major_buffered",
        "variant_major_buffered",
    ]


def test_bgen_reader_benchmark_parses_boolean_modes() -> None:
    assert bgen_reader_benchmark.parse_boolean_mode_list("trusted,safe") == [True, False]


def test_tuning_benchmark_builds_queue_depth_values() -> None:
    assert tuning_benchmark.build_queue_depth_values(4, (1, 2)) == (4, 8)


def test_binary_firth_parity_harness_synthetic_fixture_passes() -> None:
    comparison = binary_firth_parity.compare_binary_paths(
        inputs=binary_firth_parity.build_synthetic_inputs(),
        correction_plan=binary_firth_parity.types.BinaryCorrectionPlan(
            method=binary_firth_parity.types.BinaryFallbackMethod.FIRTH_APPROXIMATE
        ),
    )

    assert comparison.passed is True
    assert comparison.production_metrics == comparison.variant_major_metrics
    assert comparison.production_metrics.firth_candidate_count >= 0


def test_binary_firth_parity_harness_loads_npz_fixture(tmp_path: Path) -> None:
    inputs = binary_firth_parity.build_synthetic_inputs()
    fixture_path = tmp_path / "binary_fixture.npz"
    binary_firth_parity.np.savez(
        fixture_path,
        covariate_matrix=binary_firth_parity.np.asarray(inputs.covariate_matrix),
        phenotype_vector=binary_firth_parity.np.asarray(inputs.phenotype_vector),
        genotype_matrix=binary_firth_parity.np.asarray(inputs.genotype_matrix),
        loco_offset=binary_firth_parity.np.asarray(inputs.loco_offset),
    )

    loaded_inputs = binary_firth_parity.load_npz_inputs(fixture_path)

    assert loaded_inputs.genotype_matrix.shape == inputs.genotype_matrix.shape


def test_tuning_benchmark_builds_trial_environment_from_low_level_knobs() -> None:
    candidate = tuning_benchmark.Step2TuningCandidate(
        trait_type=tuning_benchmark.types.RegenieTraitType.BINARY,
        chunk_size=8192,
        staging_depth=1,
        output_writer_thread_count=8,
        output_writer_queue_depth=16,
        bgen_decode_tile_variant_count=128,
        rayon_thread_count=4,
        firth_batch_size=64,
    )
    environment = tuning_benchmark.build_step2_trial_environment(candidate)
    assert "G_BGEN_DECODE_TILE_VARIANT_COUNT" not in environment
    assert "RAYON_NUM_THREADS" not in environment
    assert "G_REGENIE2_BINARY_FIRTH_BATCH_SIZE" not in environment


def test_tuning_benchmark_builds_shared_compute_candidates() -> None:
    bgen_candidate_summary = tuning_benchmark.BgenCandidateSummary(
        candidate=tuning_benchmark.BgenCandidate(
            decode_tile_variant_count=64,
            rayon_thread_count=2,
            benchmark_chunk_size=8192,
        ),
        median_seconds=0.1,
        mean_seconds=0.1,
        repeat_count=3,
    )
    candidates = tuning_benchmark.build_compute_stage_candidates(
        trait_type=tuning_benchmark.types.RegenieTraitType.QUANTITATIVE,
        chunk_sizes=(4096, 8192),
        staging_depth_values=(1, 2),
        bgen_candidates=(bgen_candidate_summary,),
        firth_batch_sizes=(32, 64),
    )
    assert len(candidates) == 4
    assert all(candidate.firth_batch_size is None for candidate in candidates)


def test_tuning_benchmark_builds_binary_compute_candidates_with_firth_sizes() -> None:
    bgen_candidate_summary = tuning_benchmark.BgenCandidateSummary(
        candidate=tuning_benchmark.BgenCandidate(
            decode_tile_variant_count=64,
            rayon_thread_count=2,
            benchmark_chunk_size=8192,
        ),
        median_seconds=0.1,
        mean_seconds=0.1,
        repeat_count=3,
    )
    candidates = tuning_benchmark.build_compute_stage_candidates(
        trait_type=tuning_benchmark.types.RegenieTraitType.BINARY,
        chunk_sizes=(8192,),
        staging_depth_values=(1,),
        bgen_candidates=(bgen_candidate_summary,),
        firth_batch_sizes=(32, 64),
    )
    assert [candidate.firth_batch_size for candidate in candidates] == [32, 64]


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
    )
    gpu_command = comparison_benchmark.build_g_step2_command(
        uv_executable="uv",
        baseline_paths=baseline_paths,
        output_prefix=Path("data/benchmarks/out_gpu"),
        device="gpu",
        chunk_size=2048,
        variant_limit=None,
    )
    binary_command = comparison_benchmark.build_g_step2_command(
        uv_executable="uv",
        baseline_paths=baseline_paths,
        output_prefix=Path("data/benchmarks/out_bin"),
        device="cpu",
        chunk_size=8192,
        variant_limit=None,
        trait_type="binary",
    )
    assert cpu_command[:4] == ["uv", "run", "g", "regenie"]
    assert "--step" in cpu_command
    assert cpu_command[cpu_command.index("--step") + 1] == "2"
    assert "--qt" in cpu_command
    assert "--g-device" in cpu_command
    assert cpu_command[cpu_command.index("--g-device") + 1] == "cpu"
    assert "--g-output-format" in cpu_command
    assert cpu_command[cpu_command.index("--g-output-format") + 1] == "parquet"
    assert "--g-variant-limit" in cpu_command
    assert "--variant-limit" not in cpu_command
    assert gpu_command[gpu_command.index("--g-device") + 1] == "gpu"
    assert "--g-variant-limit" not in gpu_command
    assert "--bt" in binary_command
    assert "--firth" in binary_command
    assert "--approx" in binary_command
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
    pl.DataFrame(
        {
            "ID": ["rs1", "rs2"],
            "BETA": [0.1, 0.2],
            "LOG10P": [1.0, 2.0],
        }
    ).write_parquet(g_output)
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
        output_writer_thread_count=2,
        warmup_count=1,
        trial_results=trial_results,
    )
    assert summary.mean_rows_per_second == 75.0
    assert summary.mean_chunk_bytes == 1536.0
    assert summary.mean_final_parquet_bytes == 768.0


def test_binary_hot_benchmark_defaults_to_comparable_modes() -> None:
    arguments = binary_hot_benchmark.build_argument_parser().parse_args([])
    assert arguments.device == "gpu"
    assert arguments.chunk_size == 8192
    assert arguments.output_writer_thread_count == 8
    assert arguments.trusted_no_missing_diploid is True
    trial_specs = binary_hot_benchmark.build_trial_specs(
        include_cold_process=arguments.include_cold_process,
        include_no_final_hot=arguments.include_no_final_hot,
        include_finalized_hot=arguments.include_finalized_hot,
    )
    assert [trial_spec.mode.value for trial_spec in trial_specs] == [
        "cold_process_finalized",
        "warm_same_process_no_final",
        "hot_same_process_no_final",
        "warm_same_process_finalized",
        "hot_same_process_finalized",
    ]


def test_binary_hot_child_process_command_contains_binary_controls(tmp_path: Path) -> None:
    configuration = binary_hot_benchmark.BenchmarkConfiguration(
        data_directory=Path("data"),
        output_directory=tmp_path / "profile",
        device=binary_hot_benchmark.types.Device.CPU,
        chunk_size=4096,
        staging_depth=2,
        output_writer_thread_count=4,
        output_writer_queue_depth=8,
        trusted_no_missing_diploid=True,
        assume_trusted_validated=True,
        firth_batch_size=64,
        variant_limit=1000,
        python_executable=sys.executable,
        jax_cache_directory=tmp_path / "jax-cache",
    )
    trial_spec = binary_hot_benchmark.TrialSpec(
        name="cold_process_finalized",
        mode=binary_hot_benchmark.BenchmarkMode.COLD_PROCESS_FINALIZED,
        finalize_parquet=True,
        fresh_process=True,
        same_process_group=None,
    )
    child_command = binary_hot_benchmark.build_fresh_process_command(
        configuration=configuration,
        trial_spec=trial_spec,
        stage_timing_path=tmp_path / "stages.json",
    )
    command_text = child_command.command_arguments[2]
    assert child_command.command_arguments[:2] == [sys.executable, "-c"]
    assert "benchmark_regenie2_binary_hot" in command_text
    assert "trusted_no_missing_diploid" in command_text
    assert "variant_limit" in command_text
    assert "G_REGENIE2_BINARY_FIRTH_BATCH_SIZE" not in child_command.environment_overrides
    assert "G_REGENIE2_ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED" not in child_command.environment_overrides
    assert "JAX_PLATFORMS" not in child_command.environment_overrides


def test_binary_hot_summary_records_headline_modes(tmp_path: Path) -> None:
    configuration = binary_hot_benchmark.BenchmarkConfiguration(
        data_directory=Path("data"),
        output_directory=tmp_path / "profile",
        device=binary_hot_benchmark.types.Device.GPU,
        chunk_size=8192,
        staging_depth=1,
        output_writer_thread_count=8,
        output_writer_queue_depth=8,
        trusted_no_missing_diploid=True,
        assume_trusted_validated=True,
        firth_batch_size=64,
        variant_limit=None,
        python_executable=sys.executable,
        jax_cache_directory=tmp_path / "jax-cache",
    )
    output_metrics = binary_hot_benchmark.OutputMetrics(
        output_run_directory="run",
        final_parquet=None,
        output_row_count=100,
        info_non_null_count=100,
        chunk_file_count=2,
        chunk_bytes=1024,
        final_parquet_bytes=None,
    )
    trial_results = [
        binary_hot_benchmark.TrialResult(
            name="hot_same_process_no_final",
            mode=binary_hot_benchmark.BenchmarkMode.HOT_SAME_PROCESS_NO_FINAL,
            fresh_process=False,
            finalize_parquet=False,
            same_process_group="no_final",
            wall_time_seconds=7.25,
            stage_timing_path="hot_no_final.json",
            output_metrics=output_metrics,
        ),
        binary_hot_benchmark.TrialResult(
            name="hot_same_process_finalized",
            mode=binary_hot_benchmark.BenchmarkMode.HOT_SAME_PROCESS_FINALIZED,
            fresh_process=False,
            finalize_parquet=True,
            same_process_group="finalized",
            wall_time_seconds=7.85,
            stage_timing_path="hot_finalized.json",
            output_metrics=output_metrics,
        ),
    ]
    summary = binary_hot_benchmark.build_summary(configuration=configuration, trial_results=trial_results)
    assert summary["headline"]["hot_same_process_no_final_seconds"] == 7.25
    assert summary["headline"]["hot_same_process_finalized_seconds"] == 7.85
    assert summary["metadata"]["configuration"]["trusted_no_missing_diploid"] is True


def test_deep_profile_builds_cache_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G_PROFILE_GPU_JAX_CACHE_PARENT", str(tmp_path / "gpu_cache"))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    candidate = deep_profile.Step2Candidate(
        trait_type="binary",
        device="gpu",
        chunk_size=8192,
        staging_depth=1,
        output_writer_thread_count=4,
        output_writer_queue_depth=8,
        bgen_decode_tile_variant_count=128,
        rayon_thread_count=2,
        firth_batch_size=64,
    )
    environment = deep_profile.build_g_trial_environment(
        candidate=candidate,
        cache_directory=tmp_path / "jax_cache",
        stage_timing_path=tmp_path / "stages.json",
    )
    assert "JAX_COMPILATION_CACHE_DIR" not in environment
    assert "G_REGENIE2_STAGE_TIMINGS_JSON" not in environment
    assert "G_BGEN_DECODE_TILE_VARIANT_COUNT" not in environment
    assert "RAYON_NUM_THREADS" not in environment
    assert "G_REGENIE2_BINARY_FIRTH_BATCH_SIZE" not in environment
    assert "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES" not in environment


def test_deep_profile_child_command_contains_binary_controls() -> None:
    baseline_paths = baseline_benchmark.build_baseline_paths()
    candidate = deep_profile.Step2Candidate(
        trait_type="binary",
        device="cpu",
        chunk_size=4096,
        staging_depth=2,
        output_writer_thread_count=1,
        output_writer_queue_depth=2,
        bgen_decode_tile_variant_count=None,
        rayon_thread_count=None,
        firth_batch_size=32,
    )
    command = deep_profile.build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=Path("data/profiles/out"),
        variant_limit=1000,
    )
    command_text = command[2]
    assert command[:2] == [sys.executable, "-c"]
    assert "phenotype_binary" in command_text
    assert "\"g-device\": 'cpu'" in command_text
    assert '"bsize": 4096' in command_text
    assert '"g-variant-limit": 1000' in command_text
    assert '"firth": True' in command_text
    assert "jax_probe_device_platform" in command_text


def test_deep_profile_aggregates_trial_results() -> None:
    trial_results = [
        deep_profile.TrialResult(
            name="trial0",
            implementation="g",
            trait_type="quantitative",
            device="gpu",
            status="success",
            wall_time_seconds=2.0,
            output_row_count=100,
            stdout_log_path="stdout0",
            stderr_log_path="stderr0",
            command_arguments=["python"],
            environment_overrides={},
        ),
        deep_profile.TrialResult(
            name="trial1",
            implementation="g",
            trait_type="quantitative",
            device="gpu",
            status="success",
            wall_time_seconds=1.0,
            output_row_count=100,
            stdout_log_path="stdout1",
            stderr_log_path="stderr1",
            command_arguments=["python"],
            environment_overrides={},
        ),
    ]
    aggregate = deep_profile.aggregate_trial_results(
        name="headline_g_quantitative_gpu",
        implementation="g",
        trait_type="quantitative",
        device="gpu",
        warmup_count=1,
        trial_results=trial_results,
    )
    assert aggregate.status == "success"
    assert aggregate.median_wall_time_seconds == 1.5
    assert aggregate.rows_per_second == 100 / 1.5


def test_deep_profile_runtime_comparison_uses_regenie_baseline() -> None:
    regenie_result = deep_profile.AggregateResult(
        name="headline_regenie_quantitative",
        implementation="regenie",
        trait_type="quantitative",
        device="external_cpu",
        status="success",
        trial_count=1,
        warmup_count=0,
        median_wall_time_seconds=10.0,
        mean_wall_time_seconds=10.0,
        min_wall_time_seconds=10.0,
        max_wall_time_seconds=10.0,
        standard_deviation_seconds=0.0,
        rows_per_second=10.0,
        trials=[],
    )
    g_result = deep_profile.AggregateResult(
        name="headline_g_quantitative_gpu",
        implementation="g",
        trait_type="quantitative",
        device="gpu",
        status="success",
        trial_count=1,
        warmup_count=0,
        median_wall_time_seconds=2.5,
        mean_wall_time_seconds=2.5,
        min_wall_time_seconds=2.5,
        max_wall_time_seconds=2.5,
        standard_deviation_seconds=0.0,
        rows_per_second=40.0,
        trials=[],
    )
    comparisons = deep_profile.build_runtime_comparisons([regenie_result, g_result])
    comparison = comparisons["headline_g_quantitative_gpu_vs_regenie_quantitative"]
    assert comparison["speedup_ratio"] == 4.0
    assert comparison["absolute_delta_seconds"] == -7.5


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
    pl.DataFrame(
        {
            "CHROM": [1],
            "GENPOS": [100],
            "ID": ["rs1"],
            "ALLELE0": ["A"],
            "ALLELE1": ["G"],
            "BETA": [0.1],
            "LOG10P": [1.0],
        }
    ).write_parquet(g_output)
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
    pl.DataFrame(
        {
            "CHROM": ["22"],
            "GENPOS": [100],
            "ID": ["rs1"],
            "ALLELE0": ["A"],
            "ALLELE1": ["G"],
            "BETA": [0.1],
            "LOG10P": [1.0],
        }
    ).write_parquet(g_output)
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
    pl.DataFrame(
        {
            "ID": ["rs1", "rs2"],
            "BETA": [0.1, 0.2],
            "LOG10P": [1.0, 2.0],
        }
    ).write_parquet(g_output)
    agreement = comparison_benchmark.summarize_quantitative_step2_agreement(
        regenie_output_path=regenie_output,
        g_output_path=g_output,
    )
    assert agreement.comparable
    assert agreement.merged_variant_count == 2
    assert agreement.beta_allclose_within_tolerance is True
    assert agreement.log10p_allclose_within_tolerance is True
