from __future__ import annotations

import json
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import pytest

from g import _core

TEST_DATA_DIRECTORY = Path(__file__).parent / "data" / "bgen"
HAPLOTYPES_BGEN_PATH = TEST_DATA_DIRECTORY / "haplotypes.bgen"


def run_logging_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def build_native_runtime_compatibility_token() -> _core.NativeRuntimeCompatibilityToken:
    runtime_state = _core.NativeRuntimeState()
    logging_policy_payload = _core.build_logging_runtime_policy_payload(
        log_filter="info",
        log_file=None,
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter="info",
        trace_event_cap=None,
        telemetry_mode="off",
        telemetry_stream_file=None,
    )
    jax_policy_payload: dict[str, object] = {
        "device": "cpu",
        "cache_directory": None,
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }
    return runtime_state.require_compatible_runtime_policy(
        logging_policy_payload,
        None,
        jax_policy_payload,
    )


def test_initialize_logging_is_idempotent_and_writes_python_and_rust_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "g.jsonl"

    completed_process = run_logging_subprocess(
        "\n".join(
            [
                "import logging",
                "from g import _core",
                f"log_path = {str(log_path)!r}",
                'first_result = _core.initialize_logging(log_filter="info", log_file=log_path, log_stderr=False)',
                'second_result = _core.initialize_logging(log_filter="debug", log_file=log_path, log_stderr=False)',
                'logging.warning("python warning reaches tracing")',
                "_core.shutdown_logging()",
                "print(first_result, second_result)",
            ]
        )
    )

    log_text = log_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in log_text.splitlines() if line]

    assert completed_process.stdout.strip() == "True False"
    assert records
    assert "python warning reaches tracing" in log_text
    assert "logging initialized" in log_text


def test_initialize_logging_defaults_to_info_filter(tmp_path: Path) -> None:
    log_path = tmp_path / "g-default.jsonl"

    run_logging_subprocess(
        "\n".join(
            [
                "import logging",
                "from g import _core",
                f"log_path = {str(log_path)!r}",
                "_core.initialize_logging(log_file=log_path, log_stderr=False)",
                'logging.warning("default warning is visible")',
                "_core.shutdown_logging()",
            ]
        )
    )

    log_text = log_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in log_text.splitlines() if line]

    assert records
    assert "default warning is visible" in log_text
    assert "logging initialized" in log_text


def test_plan_genotype_chunks_splits_by_boundaries_and_resume_state() -> None:
    """Ensure the native chunk planner returns chromosome-homogeneous work units."""
    chunks = _core.plan_genotype_chunks(
        variant_count=12,
        chunk_size=5,
        chromosome_boundary_indices=[0, 3, 9, 12],
        committed_chunk_identifiers=[5],
    )

    assert [(chunk.variant_start_index, chunk.variant_stop_index) for chunk in chunks] == [
        (0, 3),
        (3, 5),
        (9, 10),
        (10, 12),
    ]


def test_intersect_committed_chunk_identifier_sets_returns_sorted_shared_identifiers() -> None:
    shared_chunk_identifiers = _core.intersect_committed_chunk_identifier_sets(((64, 0, 32), (32, 64, 96), (32, 128)))

    assert shared_chunk_identifiers == [32]
    assert _core.intersect_committed_chunk_identifier_sets(()) == []


def test_plan_multi_trait_chunk_write_uses_native_committed_chunk_policy() -> None:
    write_plan = _core.plan_multi_trait_chunk_write(
        writer_session_count=3,
        chunk_identifier=32,
        committed_chunk_identifier_sets=((0,), (32,), (64,)),
    )
    assert write_plan.active_trait_indices == [0, 2]
    assert write_plan.total_trait_count == 3
    assert write_plan.active_trait_count == 2
    assert write_plan.all_traits_committed is False

    committed_write_plan = _core.plan_multi_trait_chunk_write(
        writer_session_count=2,
        chunk_identifier=32,
        committed_chunk_identifier_sets=((32,), (0, 32)),
    )
    assert committed_write_plan.active_trait_indices == []
    assert committed_write_plan.active_trait_count == 0
    assert committed_write_plan.all_traits_committed is True

    with pytest.raises(ValueError, match="Committed chunk identifier set count"):
        _core.plan_multi_trait_chunk_write(
            writer_session_count=2,
            chunk_identifier=32,
            committed_chunk_identifier_sets=((32,),),
        )


def test_native_preflight_shape_payloads_validate_deterministic_policy() -> None:
    single_payload = _core.validate_single_trait_preflight_shape_payload(
        phenotype_sample_count=3,
        covariate_dimension_count=2,
        covariate_sample_count=3,
        covariate_count=2,
    )
    assert single_payload == {"sample_count": 3, "covariate_count": 2}

    multi_payload = _core.validate_multi_trait_preflight_shape_payload(
        phenotype_dimension_count=2,
        phenotype_trait_count=2,
        phenotype_sample_count=3,
        covariate_dimension_count=2,
        covariate_sample_count=3,
        covariate_count=2,
    )
    assert multi_payload == {"trait_count": 2, "sample_count": 3, "covariate_count": 2}

    with pytest.raises(ValueError, match="Covariate matrix must be two-dimensional"):
        _core.validate_single_trait_preflight_shape_payload(
            phenotype_sample_count=3,
            covariate_dimension_count=1,
            covariate_sample_count=3,
            covariate_count=0,
        )

    with pytest.raises(ValueError, match="Phenotype matrix must contain at least one trait"):
        _core.validate_multi_trait_preflight_shape_payload(
            phenotype_dimension_count=2,
            phenotype_trait_count=0,
            phenotype_sample_count=3,
            covariate_dimension_count=2,
            covariate_sample_count=3,
            covariate_count=2,
        )


def test_native_preflight_binary_and_prediction_shape_policy() -> None:
    _core.validate_binary_phenotype_case_control_counts(case_count=1, control_count=2)
    _core.validate_finite_array("Phenotype", all_values_finite=True)
    _core.validate_covariate_matrix_rank(covariate_rank=2, covariate_count=2)
    _core.validate_binary_phenotype_coding(is_binary_coded=True)
    _core.validate_single_prediction_preflight_shape("1", (3,), sample_count=3)
    _core.validate_multi_prediction_preflight_shape("2", (2, 3), trait_count=2, sample_count=3)

    with pytest.raises(ValueError, match="Binary phenotype must contain at least one case and one control"):
        _core.validate_binary_phenotype_case_control_counts(case_count=0, control_count=2)

    with pytest.raises(ValueError, match="Phenotype contains non-finite values"):
        _core.validate_finite_array("Phenotype", all_values_finite=False)

    with pytest.raises(ValueError, match="Covariate matrix is rank deficient"):
        _core.validate_covariate_matrix_rank(covariate_rank=1, covariate_count=2)

    with pytest.raises(ValueError, match="Binary phenotype must be coded as 0/1 after alignment"):
        _core.validate_binary_phenotype_coding(is_binary_coded=False)

    with pytest.raises(ValueError, match="Prediction sample count for chromosome 1 is 2, expected 3"):
        _core.validate_single_prediction_preflight_shape("1", (2,), sample_count=3)

    with pytest.raises(
        ValueError,
        match=r"Prediction matrix shape for chromosome 2 is \(2, 2\), expected \(2, 3\)",
    ):
        _core.validate_multi_prediction_preflight_shape("2", (2, 2), trait_count=2, sample_count=3)


def test_native_pipeline_resume_compatibility_validates_all_manifests(tmp_path: Path) -> None:
    chunks_directory = tmp_path / "chunks"
    chunks_directory.mkdir()
    manifest_json = json.dumps(
        {"schema_version": 7, "chunk_size": 32, "committed_chunks": []},
        sort_keys=True,
    )
    current_header_json = json.dumps({"schema_version": 7, "chunk_size": 32}, sort_keys=True)

    _core.validate_pipeline_resume_compatibility(
        chunks_directories=(str(chunks_directory),),
        existing_manifest_json_values=(manifest_json,),
        current_header_json_values=(current_header_json,),
        resume_mode="fast",
    )
    _core.validate_pipeline_resume_compatibility(
        chunks_directories=(str(chunks_directory),),
        existing_manifest_json_values=(manifest_json,),
        current_header_json_values=(current_header_json,),
        resume_mode="strict",
    )

    with pytest.raises(ValueError, match=r"Resume requires run_manifest\.json"):
        _core.validate_pipeline_resume_compatibility(
            chunks_directories=(str(chunks_directory),),
            existing_manifest_json_values=(None,),
            current_header_json_values=(current_header_json,),
            resume_mode="fast",
        )

    with pytest.raises(ValueError, match="input counts must match"):
        _core.validate_pipeline_resume_compatibility(
            chunks_directories=(str(chunks_directory),),
            existing_manifest_json_values=(),
            current_header_json_values=(current_header_json,),
            resume_mode="fast",
        )

    incompatible_header_json = json.dumps({"schema_version": 7, "chunk_size": 64}, sort_keys=True)
    with pytest.raises(ValueError, match="chunk_size"):
        _core.validate_pipeline_resume_compatibility(
            chunks_directories=(str(chunks_directory),),
            existing_manifest_json_values=(manifest_json,),
            current_header_json_values=(incompatible_header_json,),
            resume_mode="fast",
        )


def test_native_pipeline_output_initialization_returns_committed_sets(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)
    existing_manifest_json = json.dumps(
        {
            "schema_version": 7,
            "chunk_size": 32,
            "committed_chunks": [
                {
                    "chunk_identifier": 2,
                    "variant_start_index": 2,
                    "variant_stop_index": 4,
                    "row_count": 2,
                    "chunk_file_name": "chunk_2.arrow",
                }
            ],
        },
        sort_keys=True,
    )
    current_header_json = json.dumps({"schema_version": 7, "chunk_size": 32}, sort_keys=True)

    committed_chunk_identifier_sets = _core.initialize_pipeline_output_runs(
        run_directories=(str(run_directory),),
        chunks_directories=(str(chunks_directory),),
        existing_manifest_json_values=(existing_manifest_json,),
        current_header_json_values=(current_header_json,),
        resume=True,
        resume_mode="fast",
        runtime_compatibility_token=build_native_runtime_compatibility_token(),
    )

    assert committed_chunk_identifier_sets == [[2]]
    written_manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert written_manifest["committed_chunks"][0]["chunk_identifier"] == 2


def test_native_pipeline_output_initialization_handle_returns_committed_sets(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)
    existing_manifest_json = json.dumps(
        {
            "schema_version": 7,
            "chunk_size": 32,
            "committed_chunks": [
                {
                    "chunk_identifier": 2,
                    "variant_start_index": 2,
                    "variant_stop_index": 4,
                    "row_count": 2,
                    "chunk_file_name": "chunk_2.arrow",
                }
            ],
        },
        sort_keys=True,
    )
    current_header_json = json.dumps({"schema_version": 7, "chunk_size": 32}, sort_keys=True)

    native_initialization = _core.initialize_pipeline_output_run_batch(
        run_directories=(str(run_directory),),
        chunks_directories=(str(chunks_directory),),
        existing_manifest_json_values=(existing_manifest_json,),
        current_header_json_values=(current_header_json,),
        resume=True,
        resume_mode="fast",
        runtime_compatibility_token=build_native_runtime_compatibility_token(),
    )

    assert isinstance(native_initialization, _core.NativePipelineOutputInitialization)
    assert native_initialization.output_count == 1
    assert native_initialization.committed_chunk_identifier_sets() == [[2]]
    assert native_initialization.committed_chunk_identifiers(0) == [2]
    with pytest.raises(ValueError, match="Output index 1 is out of range"):
        native_initialization.committed_chunk_identifiers(1)


def test_native_pipeline_output_preparation_batch_initializes_outputs(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)
    existing_manifest_json = json.dumps(
        {
            "schema_version": 7,
            "chunk_size": 32,
            "committed_chunks": [
                {
                    "chunk_identifier": 2,
                    "variant_start_index": 2,
                    "variant_stop_index": 4,
                    "row_count": 2,
                    "chunk_file_name": "chunk_2.arrow",
                }
            ],
        },
        sort_keys=True,
    )
    current_header_json = json.dumps({"schema_version": 7, "chunk_size": 32}, sort_keys=True)
    native_preparation_batch = _core.NativePipelineOutputPreparationBatch(
        run_directories=(str(run_directory),),
        chunks_directories=(str(chunks_directory),),
        existing_manifest_json_values=(existing_manifest_json,),
        current_header_json_values=(current_header_json,),
        resume=True,
        resume_mode="fast",
    )

    native_preparation_batch.validate_resume_compatibility()
    native_initialization = native_preparation_batch.initialize(build_native_runtime_compatibility_token())

    assert native_preparation_batch.output_count == 1
    assert native_preparation_batch.resume is True
    assert isinstance(native_initialization, _core.NativePipelineOutputInitialization)
    assert native_initialization.committed_chunk_identifiers(0) == [2]


def test_native_effective_trusted_no_missing_diploid_policy() -> None:
    assert not _core.resolve_effective_trusted_no_missing_diploid(
        requested_trusted_no_missing_diploid=False,
        variant_major_packed8_probability_pairs=False,
    )
    assert _core.resolve_effective_trusted_no_missing_diploid(
        requested_trusted_no_missing_diploid=True,
        variant_major_packed8_probability_pairs=False,
    )
    assert _core.resolve_effective_trusted_no_missing_diploid(
        requested_trusted_no_missing_diploid=False,
        variant_major_packed8_probability_pairs=True,
    )


def test_native_null_logistic_nonconvergence_policy() -> None:
    continue_plan = _core.plan_null_logistic_nonconvergence(
        chromosome="22",
        convergence_flags=(True,),
        scalar_convergence=True,
        phenotype_names=None,
        policy="fail",
    )
    assert continue_plan.action == "continue"
    assert continue_plan.failed_trait_indices == []
    assert continue_plan.message is None
    assert continue_plan.warning_message is None

    fail_plan = _core.plan_null_logistic_nonconvergence(
        chromosome="22",
        convergence_flags=(False,),
        scalar_convergence=True,
        phenotype_names=None,
        policy="fail",
    )
    assert fail_plan.action == "fail"
    assert fail_plan.failed_trait_indices == [0]
    assert fail_plan.message == "Binary null logistic model did not converge for chromosome 22."
    assert fail_plan.warning_message is None

    warn_plan = _core.plan_null_logistic_nonconvergence(
        chromosome="22",
        convergence_flags=(True, False),
        scalar_convergence=False,
        phenotype_names=("trait_a", "trait_b"),
        policy="warn",
    )
    assert warn_plan.action == "warn"
    assert warn_plan.failed_trait_indices == [1]
    assert warn_plan.message == "Binary null logistic model did not converge for chromosome 22: trait_b."
    assert warn_plan.warning_message == (
        "Binary null logistic model did not converge for chromosome 22: trait_b. "
        "Continuing because --null_logistic_nonconvergence_policy=warn."
    )

    with pytest.raises(ValueError, match="Unsupported null logistic nonconvergence policy"):
        _core.plan_null_logistic_nonconvergence(
            chromosome="22",
            convergence_flags=(False,),
            scalar_convergence=True,
            phenotype_names=None,
            policy="ignore",
        )


def test_native_runtime_state_issues_compatibility_token() -> None:
    runtime_state = _core.NativeRuntimeState()
    logging_policy_payload = _core.build_logging_runtime_policy_payload(
        log_filter="info",
        log_file=None,
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter="info",
        trace_event_cap=None,
        telemetry_mode="off",
        telemetry_stream_file=None,
    )
    jax_policy_payload: dict[str, object] = {
        "device": "cpu",
        "cache_directory": None,
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }

    runtime_token = runtime_state.require_compatible_runtime_policy(
        logging_policy_payload,
        None,
        jax_policy_payload,
    )
    runtime_policy = _core.build_runtime_policy_handle(logging_policy_payload, None, jax_policy_payload)
    runtime_token_from_policy_handle = runtime_state.require_compatible_runtime_policy_handle(runtime_policy)
    run_runtime = runtime_state.build_run_runtime(runtime_policy)

    assert isinstance(runtime_token, _core.NativeRuntimeCompatibilityToken)
    assert isinstance(runtime_policy, _core.NativeRuntimePolicy)
    assert isinstance(runtime_token_from_policy_handle, _core.NativeRuntimeCompatibilityToken)
    assert isinstance(run_runtime, _core.NativeRunRuntime)
    assert isinstance(run_runtime.runtime_compatibility_token(), _core.NativeRuntimeCompatibilityToken)
    assert runtime_policy.rayon_thread_count is None
    assert runtime_policy.logging_runtime_policy_payload() == logging_policy_payload
    assert runtime_policy.jax_runtime_policy_payload() == jax_policy_payload
    assert run_runtime.rayon_thread_count is None
    assert run_runtime.logging_runtime_policy_payload() == logging_policy_payload
    assert run_runtime.jax_runtime_policy_payload() == jax_policy_payload

    runtime_state.record_jax_runtime_policy({**jax_policy_payload, "cache_directory": "/tmp/first-cache"})
    with pytest.raises(RuntimeError, match="JAX runtime is already configured"):
        runtime_state.require_compatible_runtime_policy(
            logging_policy_payload,
            None,
            {**jax_policy_payload, "cache_directory": "/tmp/second-cache"},
        )


def test_native_runtime_state_plans_rayon_thread_pool_configuration() -> None:
    runtime_state = _core.NativeRuntimeState()

    configure_plan = runtime_state.plan_rayon_thread_pool_configuration(4)
    runtime_state.record_rayon_thread_count(4)
    skip_plan = runtime_state.plan_rayon_thread_pool_configuration(4)

    assert isinstance(configure_plan, _core.NativeRayonThreadPoolConfigurationPlan)
    assert configure_plan.should_configure is True
    assert configure_plan.thread_count == 4
    assert skip_plan.should_configure is False
    assert skip_plan.thread_count is None
    with pytest.raises(RuntimeError, match="Rayon --threads is process-global"):
        runtime_state.plan_rayon_thread_pool_configuration(8)


def test_native_runtime_state_plans_jax_runtime_setup_lifecycle() -> None:
    runtime_state = _core.NativeRuntimeState()
    jax_policy_payload: dict[str, object] = {
        "device": "cpu",
        "cache_directory": "/tmp/g-jax-cache",
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }

    configure_plan = runtime_state.plan_jax_runtime_setup_lifecycle(jax_policy_payload)
    configure_session = runtime_state.build_jax_runtime_setup_session(jax_policy_payload, "/tmp/g-jax-cache")
    runtime_state.record_jax_runtime_policy(jax_policy_payload)
    skip_plan = runtime_state.plan_jax_runtime_setup_lifecycle(jax_policy_payload)
    skip_session = runtime_state.build_jax_runtime_setup_session(jax_policy_payload, "/tmp/g-jax-cache")

    assert isinstance(configure_plan, _core.NativeJaxRuntimeSetupLifecyclePlan)
    assert configure_plan.should_configure is True
    assert isinstance(configure_session, _core.NativeJaxRuntimeSetupSession)
    assert configure_session.should_configure is True
    assert configure_session.setup_payload()["cache_directory"] == "/tmp/g-jax-cache"
    assert configure_session.side_effect_plan_payload() == {
        "should_create_cache_directory": True,
        "should_validate_gpu": False,
    }
    assert configure_session.config_update_payloads()[0]["setting_name"] == "jax_platforms"
    assert configure_session.diagnostic_event_payloads()[0]["event_name"] == "jax_platform_selected"
    assert skip_plan.should_configure is False
    assert skip_session.should_configure is False
    with pytest.raises(RuntimeError, match="JAX runtime is already configured"):
        runtime_state.plan_jax_runtime_setup_lifecycle({**jax_policy_payload, "cache_directory": "/tmp/other-cache"})
    with pytest.raises(RuntimeError, match="JAX runtime is already configured"):
        runtime_state.build_jax_runtime_setup_session(
            {**jax_policy_payload, "cache_directory": "/tmp/other-cache"},
            "/tmp/other-cache",
        )


def test_native_jax_runtime_setup_session_completes_validation() -> None:
    setup_payload = _core.resolve_jax_runtime_setup_payload(
        requested_device="gpu",
        cache_directory="/tmp/g-jax-cache",
        matmul_precision=None,
        persistent_cache=True,
        persistent_cache_min_entry_size_bytes=0,
        persistent_cache_min_compile_time_seconds=0,
        xla_autotune_cache=False,
        transfer_guard=False,
    )
    native_setup_session = _core.NativeJaxRuntimeSetupSession(setup_payload, should_configure=True)

    completed_payload = native_setup_session.complete_validation_payload("succeeded", "gpu ready")
    diagnostic_payloads = native_setup_session.diagnostic_event_payloads()
    gpu_validation_fields = typing.cast("tuple[dict[str, object], ...]", diagnostic_payloads[-1]["fields"])

    assert native_setup_session.should_configure is True
    assert completed_payload["gpu_validation_status"] == "succeeded"
    assert native_setup_session.setup_payload()["gpu_validation_message"] == "gpu ready"
    assert gpu_validation_fields[0]["value"] == "succeeded"


def test_native_jax_runtime_policy_payload() -> None:
    jax_policy_payload = _core.build_jax_runtime_policy_payload(
        device="gpu",
        cache_directory="/tmp/g-jax-cache",
        matmul_precision="highest",
        persistent_cache=False,
        persistent_cache_min_entry_size_bytes=1024,
        persistent_cache_min_compile_time_seconds=5,
        xla_autotune_cache=True,
        transfer_guard=True,
    )

    assert jax_policy_payload == {
        "device": "gpu",
        "cache_directory": "/tmp/g-jax-cache",
        "matmul_precision": "highest",
        "persistent_cache": False,
        "persistent_cache_min_entry_size_bytes": 1024,
        "persistent_cache_min_compile_time_seconds": 5,
        "xla_autotune_cache": True,
        "transfer_guard": True,
    }


def test_native_rayon_thread_pool_rejects_zero_thread_count() -> None:
    with pytest.raises(ValueError, match="Rayon thread count must be positive"):
        _core.configure_rayon_global_thread_pool(0)


def test_native_rayon_thread_pool_configuration_error_message() -> None:
    message = _core.format_rayon_thread_pool_configuration_error_value(
        thread_count=4,
        source_error="global pool already initialized",
    )

    assert message == (
        "Unable to configure Rayon global thread pool for --threads=4; "
        "existing Rayon settings are unknown: global pool already initialized"
    )


def test_native_jax_runtime_setup_diagnostic_payloads() -> None:
    diagnostic_payloads = _core.build_jax_runtime_setup_diagnostic_payloads(
        requested_device="gpu",
        platform_name="cuda",
        cache_directory="/tmp/g-cache",
        matmul_precision="float32",
        persistent_cache_enabled=True,
        persistent_cache_min_entry_size_bytes=1024,
        persistent_cache_min_compile_time_seconds=5,
        xla_auxiliary_cache_mode="xla_gpu_per_fusion_autotune_cache_dir",
        xla_auxiliary_cache_reason="XLA auxiliary cache was requested",
        transfer_guard_enabled=True,
        gpu_validation_status="failed",
        gpu_validation_message="no gpu",
    )

    assert [payload["event_name"] for payload in diagnostic_payloads] == [
        "jax_platform_selected",
        "jax_persistent_cache_configured",
        "jax_xla_auxiliary_cache_configured",
        "jax_transfer_guard_configured",
        "jax_gpu_validation",
    ]
    persistent_cache_fields = typing.cast("tuple[dict[str, object], ...]", diagnostic_payloads[1]["fields"])
    auxiliary_cache_fields = typing.cast("tuple[dict[str, object], ...]", diagnostic_payloads[2]["fields"])
    gpu_validation_fields = typing.cast("tuple[dict[str, object], ...]", diagnostic_payloads[4]["fields"])
    assert diagnostic_payloads[0]["message"] == "Selected JAX platform cuda."
    assert persistent_cache_fields[0] == {"name": "enabled", "value": True}
    assert auxiliary_cache_fields[0] == {"name": "enabled", "value": True}
    assert diagnostic_payloads[4]["level"] == "error"
    assert list(gpu_validation_fields) == [
        {"name": "status", "value": "failed"},
        {"name": "message", "value": "no gpu"},
    ]


def test_native_jax_runtime_config_update_payloads() -> None:
    update_payloads = _core.plan_jax_runtime_config_update_payloads(
        platform_name="cuda",
        cache_directory="/tmp/g-cache",
        matmul_precision="float32",
        persistent_cache_enabled=True,
        persistent_cache_min_entry_size_bytes=1024,
        persistent_cache_min_compile_time_seconds=5,
        xla_auxiliary_cache_mode="xla_gpu_per_fusion_autotune_cache_dir",
        transfer_guard_enabled=True,
    )

    assert list(update_payloads) == [
        {"setting_name": "jax_platforms", "value": "cuda"},
        {"setting_name": "jax_enable_x64", "value": True},
        {"setting_name": "jax_default_matmul_precision", "value": "float32"},
        {"setting_name": "jax_compilation_cache_dir", "value": "/tmp/g-cache"},
        {"setting_name": "jax_persistent_cache_min_entry_size_bytes", "value": 1024},
        {"setting_name": "jax_persistent_cache_min_compile_time_secs", "value": 5},
        {
            "setting_name": "jax_persistent_cache_enable_xla_caches",
            "value": "xla_gpu_per_fusion_autotune_cache_dir",
        },
        {"setting_name": "jax_transfer_guard", "value": "disallow"},
    ]

    minimal_update_payloads = _core.plan_jax_runtime_config_update_payloads(
        platform_name="cpu",
        cache_directory="/tmp/g-cache",
        matmul_precision="highest",
        persistent_cache_enabled=False,
        persistent_cache_min_entry_size_bytes=0,
        persistent_cache_min_compile_time_seconds=0,
        xla_auxiliary_cache_mode="none",
        transfer_guard_enabled=False,
    )

    assert list(minimal_update_payloads) == [
        {"setting_name": "jax_platforms", "value": "cpu"},
        {"setting_name": "jax_enable_x64", "value": True},
        {"setting_name": "jax_default_matmul_precision", "value": "highest"},
    ]


def test_native_jax_runtime_setup_side_effect_plan() -> None:
    cpu_plan = _core.plan_jax_runtime_setup_side_effects_payload(
        requested_device="cpu",
        persistent_cache_enabled=True,
    )
    gpu_plan = _core.plan_jax_runtime_setup_side_effects_payload(
        requested_device="gpu",
        persistent_cache_enabled=False,
    )

    assert cpu_plan == {
        "should_create_cache_directory": True,
        "should_validate_gpu": False,
    }
    assert gpu_plan == {
        "should_create_cache_directory": False,
        "should_validate_gpu": True,
    }


def test_native_jax_runtime_setup_validation_completion() -> None:
    completed_setup = _core.complete_jax_runtime_setup_validation_payload(
        requested_device="gpu",
        platform_name="cuda",
        cache_directory="cache",
        matmul_precision="float32",
        persistent_cache_enabled=True,
        persistent_cache_min_entry_size_bytes=0,
        persistent_cache_min_compile_time_seconds=0,
        xla_auxiliary_cache_mode="none",
        xla_auxiliary_cache_reason="XLA auxiliary cache was not requested",
        transfer_guard_enabled=False,
        gpu_validation_status="succeeded",
        gpu_validation_message="gpu ready",
    )

    assert completed_setup["requested_device"] == "gpu"
    assert completed_setup["cache_directory"] == "cache"
    assert completed_setup["gpu_validation_status"] == "succeeded"
    assert completed_setup["gpu_validation_message"] == "gpu ready"


def test_native_jax_runtime_diagnostic_record_plan() -> None:
    info_plan = _core.plan_jax_runtime_diagnostic_record_payload(
        diagnostic_level="info",
        has_telemetry_session=True,
    )
    error_plan = _core.plan_jax_runtime_diagnostic_record_payload(
        diagnostic_level="error",
        has_telemetry_session=False,
    )

    assert info_plan == {
        "logging_level_name": "INFO",
        "should_emit_telemetry": True,
        "telemetry_level": "info",
    }
    assert error_plan == {
        "logging_level_name": "ERROR",
        "should_emit_telemetry": False,
        "telemetry_level": "error",
    }


def test_native_nvidia_driver_visibility_uses_any_driver_path(tmp_path: Path) -> None:
    control_device_path = tmp_path / "nvidiactl"
    uvm_device_path = tmp_path / "nvidia-uvm"
    driver_directory_path = tmp_path / "driver"

    assert not _core.nvidia_driver_files_are_visible_value(
        control_device_path=str(control_device_path),
        uvm_device_path=str(uvm_device_path),
        driver_directory_path=str(driver_directory_path),
    )

    driver_directory_path.mkdir()

    assert _core.nvidia_driver_files_are_visible_value(
        control_device_path=str(control_device_path),
        uvm_device_path=str(uvm_device_path),
        driver_directory_path=str(driver_directory_path),
    )


def test_native_default_local_cache_directory_value() -> None:
    assert (
        _core.build_default_local_cache_directory_value(
            temporary_root="/tmp",
            user_name="alice",
            directory_name="g-jax-cache",
        )
        == "/tmp/alice/g-jax-cache"
    )
    assert (
        _core.build_default_local_cache_directory_value(
            temporary_root="/tmp",
            user_name="",
            directory_name="g-jax-cache",
        )
        == "/tmp/unknown/g-jax-cache"
    )


def test_native_jax_gpu_validation_plan() -> None:
    missing_driver_plan = _core.plan_jax_gpu_validation_payload(
        nvidia_driver_visible=False,
        backend_initialization_failed=False,
        device_platforms=(),
        device_descriptions=(),
    )
    assert missing_driver_plan["status"] == "failed"
    assert missing_driver_plan["should_raise"] is True
    assert "cannot see the NVIDIA driver" in typing.cast("str", missing_driver_plan["message"])

    backend_failure_plan = _core.plan_jax_gpu_validation_payload(
        nvidia_driver_visible=True,
        backend_initialization_failed=True,
        device_platforms=(),
        device_descriptions=(),
    )
    assert backend_failure_plan["status"] == "failed"
    assert backend_failure_plan["should_raise"] is True
    assert "no CUDA-enabled JAX backend" in typing.cast("str", backend_failure_plan["message"])

    cpu_only_plan = _core.plan_jax_gpu_validation_payload(
        nvidia_driver_visible=True,
        backend_initialization_failed=False,
        device_platforms=("cpu",),
        device_descriptions=("CpuDevice(id=0)",),
    )
    assert cpu_only_plan["status"] == "failed"
    assert cpu_only_plan["should_raise"] is True
    assert "Observed devices: CpuDevice(id=0)." in typing.cast("str", cpu_only_plan["message"])

    gpu_plan = _core.plan_jax_gpu_validation_payload(
        nvidia_driver_visible=True,
        backend_initialization_failed=False,
        device_platforms=("gpu",),
        device_descriptions=("GpuDevice(id=0)",),
    )
    assert gpu_plan == {
        "status": "succeeded",
        "message": "JAX reported at least one GPU device.",
        "should_raise": False,
    }

    with pytest.raises(ValueError, match="device platform and description counts must match"):
        _core.plan_jax_gpu_validation_payload(
            nvidia_driver_visible=True,
            backend_initialization_failed=False,
            device_platforms=("cpu", "gpu"),
            device_descriptions=("CpuDevice(id=0)",),
        )


def test_native_gpu_genotype_format_resolution_policy() -> None:
    assert (
        _core.resolve_manifest_gpu_genotype_format(
            resume=True,
            manifest_gpu_genotype_format="packed8",
            association_backend_genotype_format="dosage",
        )
        == "packed8"
    )
    assert (
        _core.resolve_manifest_gpu_genotype_format(
            resume=True,
            manifest_gpu_genotype_format=None,
            association_backend_genotype_format="dosage",
        )
        == "dosage"
    )
    assert (
        _core.resolve_manifest_gpu_genotype_format(
            resume=False,
            manifest_gpu_genotype_format="packed8",
            association_backend_genotype_format=None,
        )
        is None
    )

    auto_to_dosage_plan = _core.plan_gpu_genotype_format_auto_to_dosage(
        requested_gpu_genotype_format="auto",
        resolution_reason="multi_trait_or_linear_pipeline",
    )
    assert auto_to_dosage_plan.requested_gpu_genotype_format == "auto"
    assert auto_to_dosage_plan.resolved_gpu_genotype_format == "dosage"
    assert auto_to_dosage_plan.resolution_reason == "multi_trait_or_linear_pipeline"
    assert auto_to_dosage_plan.fallback_error is None
    assert auto_to_dosage_plan.requires_trusted_validation is False
    assert auto_to_dosage_plan.is_resolved is True
    assert auto_to_dosage_plan.should_log_auto_resolution is True

    explicit_plan = _core.plan_single_trait_binary_gpu_genotype_format_resolution(
        requested_gpu_genotype_format="packed8",
        manifest_gpu_genotype_format=None,
        association_backend_genotype_format=None,
        resume=False,
        jax_device="gpu",
    )
    assert explicit_plan.resolved_gpu_genotype_format == "packed8"
    assert explicit_plan.resolution_reason == "explicit"
    assert explicit_plan.should_log_auto_resolution is False

    manifest_plan = _core.plan_single_trait_binary_gpu_genotype_format_resolution(
        requested_gpu_genotype_format="auto",
        manifest_gpu_genotype_format=None,
        association_backend_genotype_format="dosage",
        resume=True,
        jax_device="gpu",
    )
    assert manifest_plan.resolved_gpu_genotype_format == "dosage"
    assert manifest_plan.resolution_reason == "resume_manifest"
    assert manifest_plan.requires_trusted_validation is False

    validation_plan = _core.plan_single_trait_binary_gpu_genotype_format_resolution(
        requested_gpu_genotype_format="auto",
        manifest_gpu_genotype_format=None,
        association_backend_genotype_format=None,
        resume=False,
        jax_device="gpu",
    )
    assert validation_plan.resolved_gpu_genotype_format is None
    assert validation_plan.resolution_reason is None
    assert validation_plan.requires_trusted_validation is True

    passed_plan = _core.plan_auto_gpu_genotype_format_after_trusted_validation(fallback_error=None)
    assert passed_plan.resolved_gpu_genotype_format == "packed8"
    assert passed_plan.resolution_reason == "trusted_validation_passed"

    failed_plan = _core.plan_auto_gpu_genotype_format_after_trusted_validation(
        fallback_error="packed8 incompatible",
    )
    assert failed_plan.resolved_gpu_genotype_format == "dosage"
    assert failed_plan.resolution_reason == "trusted_validation_failed"
    assert failed_plan.fallback_error == "packed8 incompatible"

    with pytest.raises(ValueError, match="Unsupported GPU genotype format"):
        _core.plan_gpu_genotype_format_auto_to_dosage(
            requested_gpu_genotype_format="unknown",
            resolution_reason="unused",
        )


def test_resolve_delivery_callback_batch_size_enforces_native_delivery_policy() -> None:
    assert (
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=None,
            variant_major_packed8_probability_pairs=False,
        )
        == 1
    )
    assert (
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=2,
            variant_major_packed8_probability_pairs=False,
        )
        == 2
    )
    assert (
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=1,
            variant_major_packed8_probability_pairs=True,
        )
        == 1
    )
    with pytest.raises(ValueError, match="native_callback_batch_size must be positive"):
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=0,
            variant_major_packed8_probability_pairs=False,
        )
    with pytest.raises(ValueError, match="packed8 BGEN delivery"):
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=2,
            variant_major_packed8_probability_pairs=True,
        )


def test_plan_bgen_delivery_invocation_uses_native_delivery_policy() -> None:
    dosage_plan = _core.plan_bgen_delivery_invocation(
        callback_batch_size=2,
        variant_major_packed8_probability_pairs=False,
        has_native_multi_aligned_sample_data=True,
        has_native_aligned_sample_data=True,
    )
    assert dosage_plan.delivery_method == "dosage_native_multi_aligned_samples"
    assert dosage_plan.callback_batch_size == 2

    fallback_dosage_plan = _core.plan_bgen_delivery_invocation(
        callback_batch_size=None,
        variant_major_packed8_probability_pairs=False,
        has_native_multi_aligned_sample_data=False,
        has_native_aligned_sample_data=False,
    )
    assert fallback_dosage_plan.delivery_method == "dosage_sample_indices"
    assert fallback_dosage_plan.callback_batch_size == 1

    packed8_plan = _core.plan_bgen_delivery_invocation(
        callback_batch_size=1,
        variant_major_packed8_probability_pairs=True,
        has_native_multi_aligned_sample_data=False,
        has_native_aligned_sample_data=True,
    )
    assert packed8_plan.delivery_method == "packed8_native_aligned_samples"
    assert packed8_plan.callback_batch_size == 1

    with pytest.raises(ValueError, match="packed8 BGEN delivery"):
        _core.plan_bgen_delivery_invocation(
            callback_batch_size=2,
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=False,
        )


def test_resolve_grouped_union_callback_batch_size_enforces_native_delivery_policy() -> None:
    assert _core.resolve_grouped_union_callback_batch_size(native_callback_batch_size=1) == 1
    with pytest.raises(ValueError, match="native_callback_batch_size must be positive"):
        _core.resolve_grouped_union_callback_batch_size(native_callback_batch_size=0)
    with pytest.raises(ValueError, match="grouped union BGEN delivery"):
        _core.resolve_grouped_union_callback_batch_size(native_callback_batch_size=2)


def test_native_callback_worker_lifecycle_state_tracks_start() -> None:
    lifecycle_state = _core.NativeCallbackWorkerLifecycleState()

    assert lifecycle_state.has_started is False
    assert lifecycle_state.mark_started() is True
    assert lifecycle_state.has_started is True
    assert lifecycle_state.mark_started() is False


def test_plan_callback_worker_start_uses_native_start_policy() -> None:
    start_plan = _core.plan_callback_worker_start(has_started=False)

    assert start_plan.should_start is True
    assert start_plan.start_result_worker is True
    assert start_plan.start_dosage_worker is True
    assert start_plan.start_actions == ["start_result_worker", "start_dosage_worker"]

    already_started_plan = _core.plan_callback_worker_start(has_started=True)
    assert already_started_plan.should_start is False
    assert already_started_plan.start_result_worker is False
    assert already_started_plan.start_dosage_worker is False
    assert already_started_plan.start_actions == []


def test_native_callback_scheduler_state_owns_callback_resource_state() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=3,
        native_callback_batch_size=2,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
    )

    assert scheduler_state.native_callback_batch_size == 2
    assert scheduler_state.dosage_queue_depth == 3
    assert scheduler_state.dosage_queue_capacity == 3
    assert scheduler_state.dosage_queue_occupied_count == 0
    assert scheduler_state.has_available_dosage_queue_slot() is True
    assert scheduler_state.result_queue_depth == 3
    assert scheduler_state.result_queue_capacity == 3
    assert scheduler_state.result_queue_occupied_count == 0
    assert scheduler_state.has_available_result_queue_slot() is True
    assert scheduler_state.result_in_flight_limit == 7
    assert scheduler_state.result_in_flight_slot_limit == 7
    assert scheduler_state.dosage_buffer_limit == 8
    assert scheduler_state.dosage_buffer_pool_limit == 8
    assert scheduler_state.has_started is False
    start_plan = scheduler_state.plan_worker_start()
    assert start_plan.should_start is True
    assert start_plan.start_actions == ["start_result_worker", "start_dosage_worker"]
    assert scheduler_state.mark_started() is True
    assert scheduler_state.has_started is True
    assert scheduler_state.plan_worker_start().should_start is False
    assert scheduler_state.mark_started() is False

    assert scheduler_state.acquire_dosage_queue_slot() is True
    assert scheduler_state.dosage_queue_occupied_count == 1
    assert scheduler_state.has_available_dosage_queue_slot() is True
    assert scheduler_state.release_dosage_queue_slot() is True
    assert scheduler_state.dosage_queue_occupied_count == 0

    assert scheduler_state.acquire_result_queue_slot() is True
    assert scheduler_state.result_queue_occupied_count == 1
    assert scheduler_state.has_available_result_queue_slot() is True
    assert scheduler_state.release_result_queue_slot() is True
    assert scheduler_state.result_queue_occupied_count == 0

    assert scheduler_state.acquire_result_in_flight_slot() is True
    assert scheduler_state.result_in_flight_occupied_count == 1
    assert scheduler_state.has_available_result_in_flight_slot() is True
    assert scheduler_state.release_result_in_flight_slot() is True
    assert scheduler_state.result_in_flight_occupied_count == 0

    assert scheduler_state.register_dosage_buffer(11) is True
    assert scheduler_state.owns_dosage_buffer(11) is True
    assert scheduler_state.dosage_buffer_allocated_count == 1
    assert scheduler_state.dosage_buffer_identifiers == [11]
    assert scheduler_state.has_available_dosage_buffer_slot() is True
    assert scheduler_state.discard_dosage_buffer(11) is True
    assert scheduler_state.dosage_buffer_allocated_count == 0
    assert scheduler_state.has_dosage_worker_error is False
    assert scheduler_state.has_result_worker_error is False
    assert scheduler_state.dosage_worker_error_message is None
    assert scheduler_state.result_worker_error_message is None

    scheduler_state.record_dosage_worker_error("dosage failed")
    scheduler_state.record_result_worker_error("writer failed")

    assert scheduler_state.has_dosage_worker_error is True
    assert scheduler_state.has_result_worker_error is True
    assert scheduler_state.dosage_worker_error_message == "native pipeline callback worker failed: dosage failed"
    assert scheduler_state.result_worker_error_message == "native pipeline result writer worker failed: writer failed"
    assert scheduler_state.clear_dosage_worker_error() is True
    assert scheduler_state.clear_result_worker_error() is True
    assert scheduler_state.dosage_worker_error_message is None
    assert scheduler_state.result_worker_error_message is None

    assert scheduler_state.backpressure_poll_timeout_seconds == 0.1
    finish_plan = scheduler_state.plan_worker_finish()
    assert finish_plan.finish_actions == [
        "stop_dosage_worker",
        "join_dosage_worker",
        "stop_result_worker",
        "join_result_worker",
        "raise_worker_error",
        "complete_progress",
        "emit_binary_correction_summary",
    ]
    assert finish_plan.stop_dosage_worker is True
    assert finish_plan.join_dosage_worker is True
    assert finish_plan.stop_result_worker is True
    assert finish_plan.join_result_worker is True
    assert finish_plan.raise_worker_error is True
    assert finish_plan.complete_progress is True
    assert finish_plan.emit_binary_correction_summary is True
    assert finish_plan.dosage_stop_timeout_seconds == 60.0
    assert finish_plan.dosage_join_timeout_seconds == 300.0
    assert finish_plan.result_stop_timeout_seconds == 60.0
    assert finish_plan.result_join_timeout_seconds == 300.0
    abort_plan = scheduler_state.plan_worker_abort()
    assert abort_plan.abort_actions == ["stop_dosage_worker", "stop_result_worker"]
    assert abort_plan.stop_dosage_worker is True
    assert abort_plan.stop_result_worker is True
    assert abort_plan.dosage_stop_timeout_seconds == 1.0
    assert abort_plan.result_stop_timeout_seconds == 1.0

    queue_operation_plan = scheduler_state.plan_queue_operation_observation(
        queue_name="dosage_buffer_pool",
        operation_name="return",
        elapsed_seconds=0.25,
        blocked=True,
    )
    assert queue_operation_plan.queue_name == "dosage_buffer_pool"
    assert queue_operation_plan.operation_name == "return"
    assert queue_operation_plan.blocked_seconds == 0.25

    queue_backpressure_observation = scheduler_state.plan_queue_backpressure_observation(
        queue_name="dosage_buffer_pool",
        operation_name="return",
        queue_depth=1,
        queue_capacity=2,
        elapsed_seconds=0.25,
        blocked=True,
    )
    assert queue_backpressure_observation.queue_name == "dosage_buffer_pool"
    assert queue_backpressure_observation.operation_name == "return"
    assert queue_backpressure_observation.queue_depth == 1
    assert queue_backpressure_observation.queue_capacity == 2
    assert queue_backpressure_observation.elapsed_seconds == 0.25
    assert queue_backpressure_observation.blocked_seconds == 0.25

    queue_stage_plan = scheduler_state.plan_queue_stage_observation(
        queue_name="dosage_queue",
        operation_name="producer_blocking",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert queue_stage_plan.queue_name == "dosage_queue"
    assert queue_stage_plan.operation_name == "producer_blocking"
    assert queue_stage_plan.stage_name == "callback_queue_producer_blocking"
    assert queue_stage_plan.blocked_seconds == 0.5

    queue_stage_backpressure_observation = scheduler_state.plan_queue_stage_backpressure_observation(
        queue_name="dosage_queue",
        operation_name="producer_blocking",
        queue_depth=3,
        queue_capacity=3,
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert queue_stage_backpressure_observation.queue_name == "dosage_queue"
    assert queue_stage_backpressure_observation.operation_name == "producer_blocking"
    assert queue_stage_backpressure_observation.stage_name == "callback_queue_producer_blocking"
    assert queue_stage_backpressure_observation.queue_depth == 3
    assert queue_stage_backpressure_observation.queue_capacity == 3
    assert queue_stage_backpressure_observation.elapsed_seconds == 0.5
    assert queue_stage_backpressure_observation.blocked_seconds == 0.5

    reuse_plan = scheduler_state.plan_dosage_buffer_reuse(
        buffered_shape=(4, 5),
        expected_shape=(2, 3),
    )
    assert reuse_plan is not None
    assert reuse_plan.requires_slice is True
    assert reuse_plan.slice_dimensions == [2, 3]
    assert (
        scheduler_state.plan_dosage_buffer_reuse(
            buffered_shape=(2, 3),
            expected_shape=(3, 2),
        )
        is None
    )

    batch_handoff_plan = scheduler_state.plan_variant_major_dosage_batch_handoff(
        metadata_count=2,
        genotype_matrix_by_variant_count=2,
        chunk_stats_count=2,
    )
    assert batch_handoff_plan.chunk_count == 2
    with pytest.raises(ValueError, match="identical lengths"):
        scheduler_state.plan_variant_major_dosage_batch_handoff(
            metadata_count=2,
            genotype_matrix_by_variant_count=1,
            chunk_stats_count=2,
        )

    dosage_join_plan = scheduler_state.plan_dosage_worker_join(timeout_seconds=None)
    assert dosage_join_plan.should_join is True
    assert dosage_join_plan.timeout_seconds == 60.0
    dosage_stop_plan = scheduler_state.plan_dosage_worker_stop(timeout_seconds=None, is_worker_alive=True)
    assert dosage_stop_plan.should_stop is True
    assert dosage_stop_plan.timeout_seconds == 60.0
    dosage_poll_plan = scheduler_state.plan_dosage_worker_stop_poll(
        remaining_timeout_seconds=1.0,
        is_worker_alive=True,
    )
    assert dosage_poll_plan.should_stop is True
    assert dosage_poll_plan.poll_timeout_seconds == 0.1

    scheduler_state.record_result_worker_error("writer failed")
    result_stop_plan = scheduler_state.plan_result_worker_stop(timeout_seconds=None, is_worker_alive=True)
    assert result_stop_plan.should_stop is False
    assert result_stop_plan.timeout_seconds == 60.0
    result_poll_plan = scheduler_state.plan_result_worker_stop_poll(
        remaining_timeout_seconds=1.0,
        is_worker_alive=True,
    )
    assert result_poll_plan.should_stop is False
    assert result_poll_plan.poll_timeout_seconds == 0.1
    result_join_plan = scheduler_state.plan_result_worker_join(timeout_seconds=0.25)
    assert result_join_plan.should_join is True
    assert result_join_plan.timeout_seconds == 0.25

    with pytest.raises(ValueError, match="effective dosage_buffer_limit"):
        _core.NativeCallbackSchedulerState(
            staging_depth=1,
            native_callback_batch_size=3,
            result_in_flight_limit=None,
            dosage_buffer_limit=2,
        )


def test_native_callback_progress_state_tracks_chromosome_transitions() -> None:
    progress_state = _core.NativeCallbackProgressState()

    first_identity = _core.build_callback_chunk_identity("chr1", 0, 8)
    assert first_identity.chunk_identifier == 0
    assert first_identity.chromosome == "chr1"
    assert first_identity.variant_start_index == 0
    assert first_identity.variant_stop_index == 8
    assert first_identity.variant_count == 8

    first_update = progress_state.record_processed_chunk(first_identity)
    assert first_update.processed_chunk_count == 1
    assert first_update.completed_chromosome is None
    assert first_update.completed_processed_chunk_count is None
    assert first_update.started_chromosome == "chr1"
    assert first_update.chunk_identity.variant_count == 8
    assert progress_state.current_progress_chromosome == "chr1"
    first_telemetry_plan = first_update.telemetry_plan
    assert [
        (event.event_name, event.level, event.chromosome, event.processed_chunk_count)
        for event in first_telemetry_plan.events
    ] == [
        ("chromosome_started", "info", "chr1", 1),
    ]
    assert first_telemetry_plan.progress.processed_chunk_count == 1
    assert first_telemetry_plan.progress.chromosome == "chr1"
    assert first_telemetry_plan.progress.chunk_identifier == 0
    assert first_telemetry_plan.progress.variant_start_index == 0
    assert first_telemetry_plan.progress.variant_stop_index == 8
    assert first_telemetry_plan.progress.variant_count == 8

    second_update = progress_state.record_processed_chunk(_core.build_callback_chunk_identity("chr2", 8, 10))
    assert second_update.processed_chunk_count == 2
    assert second_update.completed_chromosome == "chr1"
    assert second_update.completed_processed_chunk_count == 1
    assert second_update.started_chromosome == "chr2"
    assert progress_state.current_progress_chromosome == "chr2"
    assert [
        (event.event_name, event.level, event.chromosome, event.processed_chunk_count)
        for event in second_update.telemetry_plan.events
    ] == [
        ("chromosome_completed", "info", "chr1", 1),
        ("chromosome_started", "info", "chr2", 2),
    ]

    progress_completion = progress_state.finish_progress()
    assert progress_completion is not None
    assert progress_completion.chromosome == "chr2"
    assert progress_completion.processed_chunk_count == 2
    progress_completion_telemetry_event = progress_completion.telemetry_event
    assert progress_completion_telemetry_event.event_name == "chromosome_completed"
    assert progress_completion_telemetry_event.level == "info"
    assert progress_completion_telemetry_event.chromosome == "chr2"
    assert progress_completion_telemetry_event.processed_chunk_count == 2
    assert progress_state.finish_progress() is None

    progress_state.record_processed_chunk_without_progress()
    assert progress_state.processed_chunk_count == 3
    assert progress_state.current_progress_chromosome is None
    assert progress_state.finish_progress() is None


def test_resolve_native_callback_worker_shutdown_timeouts_returns_native_defaults() -> None:
    worker_shutdown_timeouts = _core.resolve_native_callback_worker_shutdown_timeouts()

    assert worker_shutdown_timeouts.dosage_worker_join_timeout_seconds == 60.0
    assert worker_shutdown_timeouts.result_worker_join_timeout_seconds == 60.0
    assert worker_shutdown_timeouts.graceful_dosage_worker_join_timeout_seconds == 300.0
    assert worker_shutdown_timeouts.graceful_result_worker_join_timeout_seconds == 300.0
    assert worker_shutdown_timeouts.worker_abort_stop_timeout_seconds == 1.0


def test_resolve_callback_worker_backpressure_poll_timeout_seconds_returns_native_default() -> None:
    assert _core.resolve_callback_worker_backpressure_poll_timeout_seconds() == 0.1


def test_resolve_callback_worker_stop_poll_timeout_seconds_caps_deadline_remaining_time() -> None:
    assert _core.resolve_callback_worker_stop_poll_timeout_seconds(1.0) == 0.1
    assert _core.resolve_callback_worker_stop_poll_timeout_seconds(0.05) == 0.05
    assert _core.resolve_callback_worker_stop_poll_timeout_seconds(0.0) == 0.0
    assert _core.resolve_callback_worker_stop_poll_timeout_seconds(-1.0) == 0.0


def test_should_attempt_callback_worker_stop_uses_native_lifecycle_policy() -> None:
    assert _core.should_attempt_callback_worker_stop(
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert not _core.should_attempt_callback_worker_stop(
        has_started=False,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert not _core.should_attempt_callback_worker_stop(
        has_started=True,
        has_worker_error=True,
        is_worker_alive=True,
    )
    assert not _core.should_attempt_callback_worker_stop(
        has_started=True,
        has_worker_error=False,
        is_worker_alive=False,
    )


def test_plan_callback_worker_join_uses_native_timeout_policy() -> None:
    dosage_join_plan = _core.plan_dosage_callback_worker_join(
        timeout_seconds=None,
        has_started=True,
    )
    assert dosage_join_plan.should_join is True
    assert dosage_join_plan.timeout_seconds == 60.0

    result_join_plan = _core.plan_result_callback_worker_join(
        timeout_seconds=0.25,
        has_started=True,
    )
    assert result_join_plan.should_join is True
    assert result_join_plan.timeout_seconds == 0.25

    unstarted_join_plan = _core.plan_result_callback_worker_join(
        timeout_seconds=None,
        has_started=False,
    )
    assert unstarted_join_plan.should_join is False
    assert unstarted_join_plan.timeout_seconds == 60.0


def test_plan_callback_worker_stop_uses_native_timeout_policy() -> None:
    dosage_stop_plan = _core.plan_dosage_callback_worker_stop(
        timeout_seconds=None,
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert dosage_stop_plan.should_stop is True
    assert dosage_stop_plan.timeout_seconds == 60.0

    result_stop_plan = _core.plan_result_callback_worker_stop(
        timeout_seconds=0.25,
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert result_stop_plan.should_stop is True
    assert result_stop_plan.timeout_seconds == 0.25

    failed_worker_stop_plan = _core.plan_result_callback_worker_stop(
        timeout_seconds=None,
        has_started=True,
        has_worker_error=True,
        is_worker_alive=True,
    )
    assert failed_worker_stop_plan.should_stop is False
    assert failed_worker_stop_plan.timeout_seconds == 60.0


def test_plan_callback_worker_finish_and_abort_use_native_timeout_policy() -> None:
    finish_plan = _core.plan_callback_worker_finish()
    assert finish_plan.finish_actions == [
        "stop_dosage_worker",
        "join_dosage_worker",
        "stop_result_worker",
        "join_result_worker",
        "raise_worker_error",
        "complete_progress",
        "emit_binary_correction_summary",
    ]
    assert finish_plan.stop_dosage_worker is True
    assert finish_plan.join_dosage_worker is True
    assert finish_plan.stop_result_worker is True
    assert finish_plan.join_result_worker is True
    assert finish_plan.raise_worker_error is True
    assert finish_plan.complete_progress is True
    assert finish_plan.emit_binary_correction_summary is True
    assert finish_plan.dosage_stop_timeout_seconds == 60.0
    assert finish_plan.dosage_join_timeout_seconds == 300.0
    assert finish_plan.result_stop_timeout_seconds == 60.0
    assert finish_plan.result_join_timeout_seconds == 300.0

    abort_plan = _core.plan_callback_worker_abort()
    assert abort_plan.abort_actions == ["stop_dosage_worker", "stop_result_worker"]
    assert abort_plan.stop_dosage_worker is True
    assert abort_plan.stop_result_worker is True
    assert abort_plan.dosage_stop_timeout_seconds == 1.0
    assert abort_plan.result_stop_timeout_seconds == 1.0


def test_plan_callback_worker_stop_poll_uses_native_loop_policy() -> None:
    active_poll_plan = _core.plan_callback_worker_stop_poll(
        remaining_timeout_seconds=1.0,
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert active_poll_plan.should_stop is True
    assert active_poll_plan.poll_timeout_seconds == 0.1

    failed_poll_plan = _core.plan_callback_worker_stop_poll(
        remaining_timeout_seconds=0.05,
        has_started=True,
        has_worker_error=True,
        is_worker_alive=True,
    )
    assert failed_poll_plan.should_stop is False
    assert failed_poll_plan.poll_timeout_seconds == 0.05

    expired_poll_plan = _core.plan_callback_worker_stop_poll(
        remaining_timeout_seconds=-1.0,
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert expired_poll_plan.should_stop is True
    assert expired_poll_plan.poll_timeout_seconds == 0.0


def test_format_callback_worker_error_messages_uses_native_policy() -> None:
    assert (
        _core.format_dosage_callback_worker_error_message("dosage failed")
        == "native pipeline callback worker failed: dosage failed"
    )
    assert (
        _core.format_result_callback_worker_error_message("writer failed")
        == "native pipeline result writer worker failed: writer failed"
    )


def test_resolve_native_callback_queue_limits_uses_native_capacity_policy() -> None:
    queue_limits = _core.resolve_native_callback_queue_limits(
        staging_depth=3,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert queue_limits.dosage_queue_depth == 3
    assert queue_limits.result_queue_depth == 3
    assert queue_limits.result_in_flight_limit == 4
    assert queue_limits.dosage_buffer_limit == 4

    explicit_queue_limits = _core.resolve_native_callback_queue_limits(
        staging_depth=3,
        native_callback_batch_size=2,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
    )
    assert explicit_queue_limits.result_in_flight_limit == 7
    assert explicit_queue_limits.dosage_buffer_limit == 8

    with pytest.raises(ValueError, match="staging_depth must be positive"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=0,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
        )
    with pytest.raises(ValueError, match="native_callback_batch_size must be positive"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=1,
            native_callback_batch_size=0,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
        )
    with pytest.raises(ValueError, match="result_in_flight_limit must be positive"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=0,
            dosage_buffer_limit=None,
        )
    with pytest.raises(ValueError, match="dosage_buffer_limit must be positive"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=0,
        )
    with pytest.raises(ValueError, match="effective dosage_buffer_limit"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=1,
            native_callback_batch_size=3,
            result_in_flight_limit=None,
            dosage_buffer_limit=2,
        )


def test_plan_callback_queue_stage_observation_uses_native_timing_policy() -> None:
    queue_observation_plan = _core.plan_callback_queue_stage_observation(
        queue_name="dosage_queue",
        operation_name="put",
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert queue_observation_plan.queue_name == "dosage_queue"
    assert queue_observation_plan.operation_name == "put"
    assert queue_observation_plan.stage_name == "callback_queue_put"
    assert queue_observation_plan.blocked_seconds == 0.0

    queue_backpressure_observation = _core.plan_callback_queue_stage_backpressure_observation(
        queue_name="dosage_queue",
        operation_name="put",
        queue_depth=2,
        queue_capacity=3,
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert queue_backpressure_observation.queue_name == "dosage_queue"
    assert queue_backpressure_observation.operation_name == "put"
    assert queue_backpressure_observation.stage_name == "callback_queue_put"
    assert queue_backpressure_observation.queue_depth == 2
    assert queue_backpressure_observation.queue_capacity == 3
    assert queue_backpressure_observation.elapsed_seconds == 0.25
    assert queue_backpressure_observation.blocked_seconds == 0.0

    blocked_observation_plan = _core.plan_callback_queue_stage_observation(
        queue_name="result_in_flight_slots",
        operation_name="producer_blocking",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert blocked_observation_plan.stage_name == "result_in_flight_producer_blocking"
    assert blocked_observation_plan.blocked_seconds == 0.5

    with pytest.raises(ValueError, match="Unsupported callback queue stage operation"):
        _core.plan_callback_queue_stage_observation(
            queue_name="unknown_queue",
            operation_name="put",
            elapsed_seconds=0.25,
            blocked=False,
        )


def test_plan_callback_queue_operation_observation_uses_native_timing_policy() -> None:
    pool_observation_plan = _core.plan_callback_queue_operation_observation(
        queue_name="dosage_buffer_pool",
        operation_name="reuse",
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert pool_observation_plan.queue_name == "dosage_buffer_pool"
    assert pool_observation_plan.operation_name == "reuse"
    assert pool_observation_plan.blocked_seconds == 0.0

    pool_backpressure_observation = _core.plan_callback_queue_backpressure_observation(
        queue_name="dosage_buffer_pool",
        operation_name="reuse",
        queue_depth=1,
        queue_capacity=2,
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert pool_backpressure_observation.queue_name == "dosage_buffer_pool"
    assert pool_backpressure_observation.operation_name == "reuse"
    assert pool_backpressure_observation.queue_depth == 1
    assert pool_backpressure_observation.queue_capacity == 2
    assert pool_backpressure_observation.elapsed_seconds == 0.25
    assert pool_backpressure_observation.blocked_seconds == 0.0

    blocked_observation_plan = _core.plan_callback_queue_operation_observation(
        queue_name="result_in_flight_slots",
        operation_name="release",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert blocked_observation_plan.queue_name == "result_in_flight_slots"
    assert blocked_observation_plan.operation_name == "release"
    assert blocked_observation_plan.blocked_seconds == 0.5

    with pytest.raises(ValueError, match="Unsupported callback queue operation"):
        _core.plan_callback_queue_operation_observation(
            queue_name="dosage_buffer_pool",
            operation_name="unknown_operation",
            elapsed_seconds=0.25,
            blocked=False,
        )


def test_plan_dosage_buffer_reuse_uses_native_shape_policy() -> None:
    exact_reuse_plan = _core.plan_dosage_buffer_reuse(
        buffered_shape=(2, 3),
        expected_shape=(2, 3),
    )
    assert exact_reuse_plan is not None
    assert exact_reuse_plan.requires_slice is False
    assert exact_reuse_plan.slice_dimensions == [2, 3]

    sliced_reuse_plan = _core.plan_dosage_buffer_reuse(
        buffered_shape=(4, 5),
        expected_shape=(2, 3),
    )
    assert sliced_reuse_plan is not None
    assert sliced_reuse_plan.requires_slice is True
    assert sliced_reuse_plan.slice_dimensions == [2, 3]

    assert _core.plan_dosage_buffer_reuse(buffered_shape=(2, 3), expected_shape=(2, 3, 1)) is None
    assert _core.plan_dosage_buffer_reuse(buffered_shape=(2, 3), expected_shape=(3, 2)) is None


def test_plan_variant_major_dosage_batch_handoff_uses_native_batch_policy() -> None:
    batch_handoff_plan = _core.plan_variant_major_dosage_batch_handoff(
        metadata_count=2,
        genotype_matrix_by_variant_count=2,
        chunk_stats_count=2,
    )
    assert batch_handoff_plan.chunk_count == 2

    with pytest.raises(ValueError, match="identical lengths"):
        _core.plan_variant_major_dosage_batch_handoff(
            metadata_count=2,
            genotype_matrix_by_variant_count=1,
            chunk_stats_count=2,
        )
    with pytest.raises(ValueError, match="at least one chunk"):
        _core.plan_variant_major_dosage_batch_handoff(
            metadata_count=0,
            genotype_matrix_by_variant_count=0,
            chunk_stats_count=0,
        )


def test_native_dosage_buffer_pool_state_tracks_capacity_and_ownership() -> None:
    buffer_pool_state = _core.NativeDosageBufferPoolState(buffer_limit=2)

    assert buffer_pool_state.buffer_limit == 2
    assert buffer_pool_state.allocated_count == 0
    assert buffer_pool_state.buffer_identifiers == []
    assert buffer_pool_state.has_available_slot() is True
    assert buffer_pool_state.register_buffer(11) is True
    assert buffer_pool_state.owns_buffer(11) is True
    assert buffer_pool_state.register_buffer(11) is False
    assert buffer_pool_state.register_buffer(7) is True
    assert buffer_pool_state.allocated_count == 2
    assert buffer_pool_state.buffer_identifiers == [7, 11]
    assert buffer_pool_state.has_available_slot() is False
    assert buffer_pool_state.register_buffer(13) is False
    assert buffer_pool_state.discard_buffer(11) is True
    assert buffer_pool_state.owns_buffer(11) is False
    assert buffer_pool_state.has_available_slot() is True
    assert buffer_pool_state.discard_buffer(99) is False


def test_native_result_in_flight_slot_state_tracks_capacity() -> None:
    slot_state = _core.NativeResultInFlightSlotState(slot_limit=2)

    assert slot_state.slot_limit == 2
    assert slot_state.occupied_count == 0
    assert slot_state.has_available_slot() is True
    assert slot_state.acquire_slot() is True
    assert slot_state.occupied_count == 1
    assert slot_state.acquire_slot() is True
    assert slot_state.occupied_count == 2
    assert slot_state.has_available_slot() is False
    assert slot_state.acquire_slot() is False
    assert slot_state.release_slot() is True
    assert slot_state.occupied_count == 1
    assert slot_state.release_slot() is True
    assert slot_state.occupied_count == 0
    assert slot_state.release_slot() is False


def test_resolve_bgen_delivery_method_uses_native_alignment_precedence() -> None:
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=False,
            has_native_multi_aligned_sample_data=True,
            has_native_aligned_sample_data=True,
        )
        == "dosage_native_multi_aligned_samples"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=False,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=True,
        )
        == "dosage_native_aligned_samples"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=False,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=False,
        )
        == "dosage_sample_indices"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=True,
            has_native_aligned_sample_data=True,
        )
        == "packed8_native_multi_aligned_samples"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=True,
        )
        == "packed8_native_aligned_samples"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=False,
        )
        == "packed8_sample_indices"
    )


def test_resolve_writer_finish_thread_count_enforces_native_cleanup_policy() -> None:
    assert _core.resolve_writer_finish_thread_count(0, 0) == 0
    assert _core.resolve_writer_finish_thread_count(3, 2) == 2
    assert _core.resolve_writer_finish_thread_count(3, 5) == 3
    with pytest.raises(ValueError, match="Writer finish thread count must be positive"):
        _core.resolve_writer_finish_thread_count(1, 0)


def test_plan_writer_finish_execution_uses_native_cleanup_policy() -> None:
    empty_finish_plan = _core.plan_writer_finish_execution(writer_session_count=0, requested_thread_count=0)
    assert empty_finish_plan.writer_session_count == 0
    assert empty_finish_plan.thread_count == 0
    assert empty_finish_plan.has_writer_sessions is False
    assert empty_finish_plan.uses_parallel_finish is False

    serial_finish_plan = _core.plan_writer_finish_execution(writer_session_count=1, requested_thread_count=1)
    assert serial_finish_plan.writer_session_count == 1
    assert serial_finish_plan.thread_count == 1
    assert serial_finish_plan.has_writer_sessions is True
    assert serial_finish_plan.uses_parallel_finish is False

    parallel_finish_plan = _core.plan_writer_finish_execution(writer_session_count=3, requested_thread_count=2)
    assert parallel_finish_plan.writer_session_count == 3
    assert parallel_finish_plan.thread_count == 2
    assert parallel_finish_plan.has_writer_sessions is True
    assert parallel_finish_plan.uses_parallel_finish is True

    with pytest.raises(ValueError, match="Writer finish thread count must be positive"):
        _core.plan_writer_finish_execution(writer_session_count=1, requested_thread_count=0)


def test_plan_bgen_delivery_cleanup_uses_native_lifecycle_policy() -> None:
    success_plan = _core.plan_bgen_delivery_cleanup(cleanup_outcome="success", callback_finished=False)
    assert success_plan.cleanup_actions == [
        "drain_callback",
        "finish_writer_sessions",
        "write_stage_timing_snapshot",
    ]
    assert success_plan.drain_callback is True
    assert success_plan.finish_writer_sessions is True
    assert success_plan.finish_interrupted_writer_sessions is False
    assert success_plan.abort_callback is False
    assert success_plan.abort_writer_sessions is False
    assert success_plan.write_stage_timing_snapshot is True

    interrupted_pending_callback_plan = _core.plan_bgen_delivery_cleanup(
        cleanup_outcome="interrupted",
        callback_finished=False,
    )
    assert interrupted_pending_callback_plan.drain_callback is True
    assert interrupted_pending_callback_plan.finish_writer_sessions is False
    assert interrupted_pending_callback_plan.finish_interrupted_writer_sessions is True
    assert interrupted_pending_callback_plan.abort_callback is False
    assert interrupted_pending_callback_plan.abort_writer_sessions is False
    assert interrupted_pending_callback_plan.cleanup_actions == [
        "drain_callback",
        "finish_interrupted_writer_sessions",
        "write_stage_timing_snapshot",
    ]

    interrupted_finished_callback_plan = _core.plan_bgen_delivery_cleanup(
        cleanup_outcome="interrupted",
        callback_finished=True,
    )
    assert interrupted_finished_callback_plan.drain_callback is False
    assert interrupted_finished_callback_plan.finish_interrupted_writer_sessions is True
    assert interrupted_finished_callback_plan.cleanup_actions == [
        "finish_interrupted_writer_sessions",
        "write_stage_timing_snapshot",
    ]

    failure_plan = _core.plan_bgen_delivery_cleanup(cleanup_outcome="failure", callback_finished=False)
    assert failure_plan.drain_callback is False
    assert failure_plan.finish_writer_sessions is False
    assert failure_plan.finish_interrupted_writer_sessions is False
    assert failure_plan.abort_callback is True
    assert failure_plan.abort_writer_sessions is True
    assert failure_plan.write_stage_timing_snapshot is True
    assert failure_plan.cleanup_actions == [
        "abort_callback",
        "abort_writer_sessions",
        "write_stage_timing_snapshot",
    ]

    cleanup_failure_plan = _core.plan_bgen_delivery_cleanup(
        cleanup_outcome="interrupted_cleanup_failure",
        callback_finished=False,
    )
    assert cleanup_failure_plan.abort_callback is True
    assert cleanup_failure_plan.abort_writer_sessions is True

    with pytest.raises(ValueError, match="Unsupported BGEN delivery cleanup outcome"):
        _core.plan_bgen_delivery_cleanup(cleanup_outcome="unknown", callback_finished=False)


def test_plan_output_write_methods_use_native_dtype_policy() -> None:
    native_float64_write_plan = _core.plan_single_trait_output_write(
        is_native_writer_session=True,
        output_statistic_dtype="float64",
    )
    assert native_float64_write_plan.method_name == "write_regenie2_native_chunk_f64"
    assert native_float64_write_plan.uses_float64_native_writer is True

    fallback_float64_write_plan = _core.plan_single_trait_output_write(
        is_native_writer_session=False,
        output_statistic_dtype="float64",
    )
    assert fallback_float64_write_plan.method_name == "write_regenie2_native_chunk"
    assert fallback_float64_write_plan.uses_float64_native_writer is False

    native_multi_write_plan = _core.plan_multi_trait_output_write(
        active_trait_count=2,
        all_writer_sessions_native=True,
        output_statistic_dtype="float64",
    )
    assert native_multi_write_plan.active_trait_count == 2
    assert native_multi_write_plan.use_native_multi_writer is True
    assert native_multi_write_plan.uses_float64_native_writer is True

    fallback_multi_write_plan = _core.plan_multi_trait_output_write(
        active_trait_count=2,
        all_writer_sessions_native=False,
        output_statistic_dtype="float64",
    )
    assert fallback_multi_write_plan.use_native_multi_writer is False
    assert fallback_multi_write_plan.uses_float64_native_writer is False

    with pytest.raises(ValueError, match="Unsupported public statistic output dtype"):
        _core.plan_single_trait_output_write(is_native_writer_session=True, output_statistic_dtype="float16")
    with pytest.raises(ValueError, match="Unsupported public statistic output dtype"):
        _core.plan_multi_trait_output_write(
            active_trait_count=1,
            all_writer_sessions_native=True,
            output_statistic_dtype="float16",
        )


def test_regenie2_run_engine_required_chromosomes_returns_boundary_labels() -> None:
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)

    assert engine.required_chromosomes() == ["1"]
    assert engine.required_chromosomes(variant_limit=1) == ["1"]
    assert engine.required_chromosomes(variant_limit=0) == []


def test_regenie2_run_engine_buffered_chunks_deliver_preprocessed_variant_major_dosage_chunks() -> None:
    class RecordingCallback:
        def __init__(self) -> None:
            self.chunk_shapes: list[tuple[int, int, int]] = []
            self.free_buffers: list[np.ndarray] = []

        def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> np.ndarray:
            if self.free_buffers:
                return self.free_buffers.pop()
            return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

        def compute_preprocessed_variant_major_dosage_chunk(
            self,
            metadata: _core.VariantMetadata,
            genotype_matrix: np.ndarray,
            chunk_stats: _core.ChunkStats,
        ) -> None:
            self.chunk_shapes.append(
                (
                    metadata.variant_start_index,
                    genotype_matrix.shape[0],
                    genotype_matrix.shape[1],
                )
            )
            assert metadata.chromosome_label == "1"
            assert not np.isnan(genotype_matrix).any()
            np.testing.assert_allclose(chunk_stats.allele_one_frequency, genotype_matrix.mean(axis=1) / 2.0)
            np.testing.assert_array_equal(chunk_stats.observation_count, np.full(genotype_matrix.shape[0], 4))
            self.free_buffers.append(genotype_matrix)

    callback = RecordingCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)

    processed_chunk_count = engine.run_bgen_variant_major_dosage_buffered_chunks(
        np.arange(4, dtype=np.int64),
        callback,
    )

    assert processed_chunk_count == 2
    assert callback.chunk_shapes == [(0, 2, 4), (2, 2, 4)]


def test_regenie2_run_engine_variant_major_chunks_support_untrusted_bgen() -> None:
    class RecordingCallback:
        def __init__(self) -> None:
            self.chunk_shapes: list[tuple[int, int, int]] = []
            self.free_buffers: list[np.ndarray] = []

        def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> np.ndarray:
            if self.free_buffers:
                return self.free_buffers.pop()
            return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

        def compute_preprocessed_variant_major_dosage_chunk(
            self,
            metadata: _core.VariantMetadata,
            genotype_matrix_by_variant: np.ndarray,
            chunk_stats: _core.ChunkStats,
        ) -> None:
            self.chunk_shapes.append(
                (
                    metadata.variant_start_index,
                    genotype_matrix_by_variant.shape[0],
                    genotype_matrix_by_variant.shape[1],
                )
            )
            assert not np.isnan(genotype_matrix_by_variant).any()
            np.testing.assert_allclose(chunk_stats.allele_one_frequency, genotype_matrix_by_variant.mean(axis=1) / 2.0)
            np.testing.assert_array_equal(
                chunk_stats.observation_count,
                np.full(genotype_matrix_by_variant.shape[0], 4),
            )
            np.testing.assert_allclose(chunk_stats.dosage_sum, genotype_matrix_by_variant.sum(axis=1))
            self.free_buffers.append(genotype_matrix_by_variant)

    callback = RecordingCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2, trusted_no_missing_diploid=False)

    processed_chunk_count = engine.run_bgen_variant_major_dosage_buffered_chunks(
        np.arange(4, dtype=np.int64),
        callback,
    )

    assert processed_chunk_count == 2
    assert callback.chunk_shapes == [(0, 2, 4), (2, 2, 4)]


def test_regenie_prediction_source_loads_aligned_loco_predictions(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 0.1 0.2 0.3\n01 1.0 2.0 3.0\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    prediction_source = _core.RegeniePredictionSource(
        str(prediction_list_path),
        "trait",
        ["0", "0"],
        ["C", "A"],
    )

    assert prediction_source.get_chromosome_predictions("22").dtype == np.float32
    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("chr22"), [0.3, 0.1], atol=1e-6)
    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("1"), [3.0, 1.0], atol=1e-6)


def test_regenie_prediction_source_loads_from_native_aligned_sample_data(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 0.1 0.2 0.3\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_path.write_text("IID\ttrait\nC\t1.0\nA\t2.0\n")
    native_aligned_sample_data = _core.align_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["0", "0"],
        ["C", "A"],
        str(phenotype_path),
        "trait",
    )

    prediction_source = _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(prediction_list_path),
        "trait",
        native_aligned_sample_data,
    )

    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("chr22"), [0.3, 0.1], atol=1e-6)


def test_multi_regenie_prediction_source_returns_trait_major_loco_matrix(tmp_path: Path) -> None:
    trait_a_loco_path = tmp_path / "trait_a.loco"
    trait_a_loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 0.1 0.2 0.3\n")
    trait_b_loco_path = tmp_path / "trait_b.loco"
    trait_b_loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 1.1 1.2 1.3\n")
    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(f"trait_a {trait_a_loco_path}\ntrait_b {trait_b_loco_path}\n")
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_path.write_text("IID\ttrait_a\ttrait_b\nC\t1.0\t2.0\nA\t3.0\t4.0\n")
    native_multi_aligned_sample_data = _core.align_multi_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["0", "0"],
        ["C", "A"],
        str(phenotype_path),
        ["trait_a", "trait_b"],
    )

    prediction_source = _core.MultiRegeniePredictionSource.from_native_multi_aligned_sample_data(
        str(prediction_list_path),
        native_multi_aligned_sample_data,
    )

    np.testing.assert_allclose(
        prediction_source.get_chromosome_predictions("chr22"),
        np.asarray([[0.3, 0.1], [1.3, 1.1]], dtype=np.float32),
        atol=1e-6,
    )


def test_multi_regenie_prediction_source_reports_missing_phenotype(tmp_path: Path) -> None:
    trait_a_loco_path = tmp_path / "trait_a.loco"
    trait_a_loco_path.write_text("FID_IID 0_A\n22 0.1\n")
    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(f"trait_a {trait_a_loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Phenotype 'trait_b' not found"):
        _core.MultiRegeniePredictionSource(
            str(prediction_list_path),
            ["trait_a", "trait_b"],
            ["0"],
            ["A"],
        )


def test_multi_regenie_prediction_source_reports_missing_chromosome(tmp_path: Path) -> None:
    trait_a_loco_path = tmp_path / "trait_a.loco"
    trait_a_loco_path.write_text("FID_IID 0_A\n22 0.1\n")
    trait_b_loco_path = tmp_path / "trait_b.loco"
    trait_b_loco_path.write_text("FID_IID 0_A\n22 1.1\n")
    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(f"trait_a {trait_a_loco_path}\ntrait_b {trait_b_loco_path}\n")
    prediction_source = _core.MultiRegeniePredictionSource(
        str(prediction_list_path),
        ["trait_a", "trait_b"],
        ["0"],
        ["A"],
    )

    with np.testing.assert_raises_regex(ValueError, "Chromosome '1'"):
        prediction_source.get_chromosome_predictions("1")


def test_regenie_prediction_source_reports_missing_samples(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID 0_A\n22 0.1\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Target samples not found in LOCO file"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["0"],
            ["missing"],
        )


def test_regenie_prediction_source_rejects_duplicate_loco_iid_by_default(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f2_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Duplicate LOCO IID 's1'"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["f1"],
            ["s1"],
        )


def test_regenie_prediction_source_rejects_duplicate_target_iid_by_default(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1\n22 0.1\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Duplicate target IID 's1'"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["f1", "f2"],
            ["s1", "s1"],
        )


def test_regenie_prediction_source_rejects_duplicate_exact_loco_key(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f1_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Duplicate LOCO sample key: f1_s1"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["f1"],
            ["s1"],
        )


def test_regenie_prediction_source_fid_iid_mode_aligns_repeated_iid(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f2_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    prediction_source = _core.RegeniePredictionSource(
        str(prediction_list_path),
        "trait",
        ["f2", "f1"],
        ["s1", "s1"],
        sample_key_mode="fid_iid",
    )

    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("22"), [0.2, 0.1], atol=1e-6)
