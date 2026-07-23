from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import types
import typing
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import tests.numerical
import tests.parity.harness
import tooling.science_gate


def variant_frame(*, beta_values: list[float], log10p_values: list[float]) -> pl.DataFrame:
    """Build two rows whose repeated ID still has unique full variant keys."""
    return pl.DataFrame(
        {
            "CHROM": ["22", "22"],
            "GENPOS": [100, 200],
            "ID": ["repeated_identifier", "repeated_identifier"],
            "ALLELE0": ["A", "C"],
            "ALLELE1": ["G", "T"],
            "BETA": beta_values,
            "LOG10P": log10p_values,
        }
    )


def test_golden_metadata_covers_only_required_external_workflows() -> None:
    metadata = tests.parity.harness.load_golden_metadata()

    tests.parity.harness.assert_metadata_covers_required_workflows(metadata)
    assert metadata.schema_version == tests.parity.harness.PARITY_METADATA_SCHEMA_VERSION
    assert metadata.regenie_reference["version"] == "v4.1"
    assert metadata.workflow_identifiers == tests.parity.harness.REQUIRED_WORKFLOW_IDENTIFIERS
    assert all(
        workflow.gate_status == tests.parity.harness.ParityGateStatus.REQUIRED for workflow in metadata.workflows
    )
    assert all(workflow.qualification is None for workflow in metadata.workflows)
    assert all(
        workflow.qualification_hosts == tests.parity.harness.REQUIRED_QUALIFICATION_HOSTS
        for workflow in metadata.workflows
    )


def qualification_evidence(
    workflow: tests.parity.harness.GoldenWorkflow,
    *,
    science_source_sha256: str,
) -> tests.parity.harness.QualificationEvidence:
    """Build complete synthetic evidence for focused metadata tests."""
    git_commit = "a" * 40
    return tests.parity.harness.QualificationEvidence(
        passed=True,
        qualified_git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        working_tree_clean=True,
        qualification_generated_at_utc="2026-07-23T00:00:00+00:00",
        qualification_node="landau",
        cargo_lock_sha256=tests.parity.harness.sha256_file(tests.parity.harness.REPOSITORY_ROOT / "Cargo.lock"),
        uv_lock_sha256=tests.parity.harness.sha256_file(tests.parity.harness.REPOSITORY_ROOT / "uv.lock"),
        jax_version=tests.parity.harness.REQUIRED_JAX_VERSION,
        jaxlib_version=tests.parity.harness.REQUIRED_JAX_VERSION,
        configured_device=str(workflow.g_cli_options["device"]),
        actual_device=tests.parity.harness.QualificationDeviceEvidence(
            platform="gpu",
            device_kind="NVIDIA GPU",
            device_count=1,
            backend_platform_version="CUDA 12.9",
            nvidia_driver_version="575.57.08",
            cuda_runtime_version="12.9.79",
        ),
        native_build=tests.parity.harness.QualificationNativeBuild(
            git_commit=git_commit,
            science_source_sha256=science_source_sha256,
            source_clean=True,
            profile=tests.parity.harness.NativeBuildProfile.RELEASE,
            library_sha256="b" * 64,
            library_size_bytes=1,
        ),
        observed_row_count=workflow.expected_row_count,
        output_fields=tests.parity.harness.PRODUCTION_OUTPUT_FIELDS,
        statistics=tuple(
            tests.parity.harness.QualificationStatisticEvidence(
                observed_column=tolerance.observed_column,
                baseline_column=tolerance.baseline_column,
                maximum_absolute_difference=0.0,
                absolute_tolerance=tolerance.absolute_tolerance,
            )
            for tolerance in workflow.tolerances
        ),
        observed_correction_count=workflow.expected_correction_count or 0,
        observed_correction_failure_count=workflow.expected_correction_failure_count or 0,
        exact_variant_keys=True,
        exact_sample_counts=True,
        exact_nonfinite_classes=True,
        exact_significance_classifications=True,
    )


def test_checked_in_required_workflows_use_external_evidence() -> None:
    metadata = tests.parity.harness.load_golden_metadata()

    tests.parity.harness.assert_metadata_covers_required_workflows(metadata)
    assert all(workflow.qualification is None for workflow in metadata.workflows)


def test_science_source_fingerprint_is_ordered_and_excludes_promotion_claims() -> None:
    metadata_content = (
        b'{"schema_version":3,"workflows":[{"identifier":"workflow",'
        b'"gate_status":"diagnostic","qualification":null,"tolerances":[1]}]}'
    )
    entries = (
        tooling.science_gate.ScienceSourceEntry(
            relative_path="tests/parity/golden_metadata.json",
            git_mode="100644",
            content=metadata_content,
        ),
        tooling.science_gate.ScienceSourceEntry(
            relative_path="src/science.py",
            git_mode="100644",
            content=b"result = 1\n",
        ),
    )

    observed = tooling.science_gate.science_source_fingerprint(entries)
    assert tooling.science_gate.science_source_fingerprint(reversed(entries)) == observed

    promoted_metadata = metadata_content.replace(
        b'"gate_status":"diagnostic","qualification":null',
        b'"gate_status":"required","qualification":{"passed":true}',
    )
    promoted_entries = (
        dataclasses.replace(entries[0], content=promoted_metadata),
        entries[1],
    )
    assert tooling.science_gate.science_source_fingerprint(promoted_entries) == observed

    changed_source_entries = (
        entries[0],
        dataclasses.replace(entries[1], content=b"result = 2\n"),
    )
    assert tooling.science_gate.science_source_fingerprint(changed_source_entries) != observed

    changed_contract_entries = (
        dataclasses.replace(entries[0], content=metadata_content.replace(b'"tolerances":[1]', b'"tolerances":[2]')),
        entries[1],
    )
    assert tooling.science_gate.science_source_fingerprint(changed_contract_entries) != observed


def test_qualification_evidence_round_trips_through_strict_payload() -> None:
    workflow = tests.parity.harness.load_golden_metadata().workflow_by_identifier("binary_score_only")
    evidence = qualification_evidence(workflow, science_source_sha256="c" * 64)
    payload = tests.parity.harness.qualification_evidence_payload(evidence)

    assert tests.parity.harness.parse_qualification_evidence(payload) == evidence

    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        tests.parity.harness.parse_qualification_evidence(payload)


def test_required_workflow_rejects_checked_in_or_stale_evidence() -> None:
    metadata = tests.parity.harness.load_golden_metadata()
    workflow = metadata.workflow_by_identifier("binary_score_only")
    current_evidence = qualification_evidence(workflow, science_source_sha256="d" * 64)
    metadata_with_checked_evidence = dataclasses.replace(
        metadata,
        workflows=tuple(
            dataclasses.replace(candidate, qualification=current_evidence)
            if candidate.identifier == workflow.identifier
            else candidate
            for candidate in metadata.workflows
        ),
    )
    with pytest.raises(AssertionError, match="must remain external"):
        tests.parity.harness.assert_metadata_covers_required_workflows(metadata_with_checked_evidence)

    stale_evidence = qualification_evidence(workflow, science_source_sha256="c" * 64)
    with pytest.raises(AssertionError, match="Stale science-source qualification"):
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            stale_evidence,
            git_commit="a" * 40,
            science_source_sha256="d" * 64,
        )

    with pytest.raises(AssertionError, match="Stale Git commit qualification"):
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            current_evidence,
            git_commit="b" * 40,
            science_source_sha256="d" * 64,
        )


def test_qualification_rejects_wrong_host_or_non_cuda_device() -> None:
    workflow = tests.parity.harness.load_golden_metadata().workflow_by_identifier("binary_score_only")
    science_source_sha256 = "c" * 64
    evidence = qualification_evidence(workflow, science_source_sha256=science_source_sha256)
    wrong_host_evidence = dataclasses.replace(evidence, qualification_node="other-gpu")
    with pytest.raises(AssertionError, match="host is not allowed"):
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            wrong_host_evidence,
            git_commit=evidence.qualified_git_commit,
            science_source_sha256=science_source_sha256,
        )

    cpu_evidence = dataclasses.replace(
        evidence,
        actual_device=dataclasses.replace(
            evidence.actual_device,
            platform="cpu",
            backend_platform_version="cpu",
        ),
    )
    with pytest.raises(AssertionError, match="JAX GPU platform"):
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            cpu_evidence,
            git_commit=evidence.qualified_git_commit,
            science_source_sha256=science_source_sha256,
        )


def test_qualification_rejects_reordered_or_retyped_output_schema() -> None:
    workflow = tests.parity.harness.load_golden_metadata().workflow_by_identifier("binary_score_only")
    science_source_sha256 = "c" * 64
    evidence = qualification_evidence(workflow, science_source_sha256=science_source_sha256)
    reordered_evidence = dataclasses.replace(
        evidence,
        output_fields=tuple(reversed(evidence.output_fields)),
    )
    with pytest.raises(AssertionError, match="schema/order/dtypes"):
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            reordered_evidence,
            git_commit=evidence.qualified_git_commit,
            science_source_sha256=science_source_sha256,
        )

    retyped_fields = list(evidence.output_fields)
    retyped_fields[1] = dataclasses.replace(
        retyped_fields[1],
        data_type=tests.parity.harness.QualificationOutputDataType.INT32,
    )
    retyped_evidence = dataclasses.replace(evidence, output_fields=tuple(retyped_fields))
    with pytest.raises(AssertionError, match="schema/order/dtypes"):
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            retyped_evidence,
            git_commit=evidence.qualified_git_commit,
            science_source_sha256=science_source_sha256,
        )


def initialize_test_repository(repository_root: Path) -> str:
    """Create one committed science source and return its full commit."""
    repository_root.mkdir()
    subprocess.run(["git", "-C", str(repository_root), "init", "--quiet"], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "config", "user.email", "parity@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository_root), "config", "user.name", "Parity Test"],
        check=True,
    )
    source_path = repository_root / "src" / "science.py"
    source_path.parent.mkdir()
    source_path.write_text("result = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository_root), "add", "src/science.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "commit", "--quiet", "-m", "test source"],
        check=True,
    )
    return tooling.science_gate.repository_git_commit(repository_root)


def test_exact_source_rejects_wrong_commit_and_dirty_checkout(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    git_commit = initialize_test_repository(repository_root)

    source_state = tooling.science_gate.assert_clean_exact_source(repository_root, git_commit)
    assert source_state.git_commit == git_commit

    with pytest.raises(AssertionError, match="wrong Git commit"):
        tooling.science_gate.assert_clean_exact_source(repository_root, "f" * 40)

    (repository_root / "src" / "science.py").write_text("result = 2\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="checkout is dirty"):
        tooling.science_gate.assert_clean_exact_source(repository_root, git_commit)


def test_exact_qualification_rejects_stale_native_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    git_commit = "a" * 40
    science_source_sha256 = "b" * 64
    source_state = tooling.science_gate.ScienceSourceState(
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
    )
    native_library_path = tmp_path / "_core.so"
    native_library_path.write_bytes(b"native")
    native_core = types.SimpleNamespace(
        __file__=str(native_library_path),
        __build_git_commit__=git_commit,
        __build_science_source_sha256__="c" * 64,
        __build_source_clean__=True,
        __build_profile__="release",
    )
    monkeypatch.setenv(tests.test_regenie2_parity.EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE, git_commit)
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE,
        science_source_sha256,
    )
    monkeypatch.setattr(
        tooling.science_gate,
        "assert_clean_exact_source",
        lambda _repository_root, _expected_git_commit: source_state,
    )

    with pytest.raises(AssertionError, match="stale science source"):
        tests.test_regenie2_parity.assert_exact_qualification_source(
            typing.cast("tests.test_regenie2_parity.NativeCoreProtocol", native_core)
        )


def test_observed_qualification_device_requires_actual_jax_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    cuda_client = types.SimpleNamespace(platform_version="CUDA 12.9")
    cuda_device = types.SimpleNamespace(
        platform="gpu",
        device_kind="NVIDIA H100",
        client=cuda_client,
    )
    fake_jax_module = types.SimpleNamespace(devices=lambda: [cuda_device])
    monkeypatch.setattr(
        tests.test_regenie2_parity.importlib,
        "import_module",
        lambda module_name: fake_jax_module,
    )
    monkeypatch.setattr(
        tests.test_regenie2_parity,
        "nvidia_driver_version",
        lambda: "575.57.08",
    )
    monkeypatch.setattr(
        tests.test_regenie2_parity.importlib.metadata,
        "version",
        lambda distribution_name: "12.9.79",
    )

    assert (
        tests.test_regenie2_parity.observe_qualification_device()
        == tests.parity.harness.QualificationDeviceEvidence(
            platform="gpu",
            device_kind="NVIDIA H100",
            device_count=1,
            backend_platform_version="CUDA 12.9",
            nvidia_driver_version="575.57.08",
            cuda_runtime_version="12.9.79",
        )
    )

    fake_jax_module.devices = lambda: [
        types.SimpleNamespace(
            platform="cpu",
            device_kind="cpu",
            client=types.SimpleNamespace(platform_version="cpu"),
        )
    ]
    with pytest.raises(AssertionError, match="must use JAX CUDA"):
        tests.test_regenie2_parity.observe_qualification_device()


def test_sanitized_bundle_requires_all_workflows_and_omits_protected_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    science_source_sha256 = "c" * 64
    reports: list[tests.test_regenie2_parity.WorkflowQualificationReport] = []
    for workflow in tests.parity.harness.load_golden_metadata().workflows:
        evidence = qualification_evidence(
            workflow,
            science_source_sha256=science_source_sha256,
        )
        report_path = tmp_path / f"{workflow.identifier}.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "workflow": {"identifier": workflow.identifier},
                    "qualification": {"passed": True, "failure": None},
                    "qualification_evidence": tests.parity.harness.qualification_evidence_payload(evidence),
                    "protected_path_that_must_not_propagate": str(tmp_path / "protected"),
                }
            ),
            encoding="utf-8",
        )
        reports.append(
            tests.test_regenie2_parity.WorkflowQualificationReport(
                workflow_identifier=workflow.identifier,
                report_path=report_path,
            )
        )
    monkeypatch.setenv(
        tests.test_regenie2_parity.REPORT_DIRECTORY_ENVIRONMENT_VARIABLE,
        str(tmp_path / "bundle"),
    )
    monkeypatch.setattr(
        tooling.science_gate,
        "assert_clean_exact_source",
        lambda repository_root, expected_git_commit: tooling.science_gate.ScienceSourceState(
            git_commit=expected_git_commit,
            science_source_sha256=science_source_sha256,
        ),
    )

    with pytest.raises(AssertionError, match="workflow mismatch"):
        tests.test_regenie2_parity.write_qualification_bundle(tuple(reports[:-1]))

    bundle_path = tests.test_regenie2_parity.write_qualification_bundle(tuple(reports))
    bundle_text = bundle_path.read_text(encoding="utf-8")
    assert str(tmp_path / "protected") not in bundle_text
    assert {
        workflow_payload["identifier"] for workflow_payload in json.loads(bundle_text)["workflows"]
    } == tests.parity.harness.REQUIRED_WORKFLOW_IDENTIFIERS

    inconsistent_payload = typing.cast(
        "dict[str, object]",
        json.loads(reports[1].report_path.read_text(encoding="utf-8")),
    )
    inconsistent_evidence = typing.cast(
        "dict[str, object]",
        inconsistent_payload["qualification_evidence"],
    )
    inconsistent_native_build = typing.cast(
        "dict[str, object]",
        inconsistent_evidence["native_build"],
    )
    inconsistent_native_build["library_sha256"] = "d" * 64
    reports[1].report_path.write_text(json.dumps(inconsistent_payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="one native/runtime build"):
        tests.test_regenie2_parity.write_qualification_bundle(tuple(reports))


def test_upstream_golden_workflows_record_commands_hashes_rows_and_tolerances() -> None:
    metadata = tests.parity.harness.load_golden_metadata()
    expected_statistic_columns = {"BETA", "SE", "CHISQ", "LOG10P"}

    for workflow in metadata.workflows:
        assert workflow.status == tests.parity.harness.ParityWorkflowStatus.EXTERNAL_GOLDEN
        assert workflow.regenie_version == "v4.1"
        assert workflow.regenie_commands
        assert len(workflow.expected_output_sha256) == 64
        assert len(workflow.expected_log_sha256) == 64
        assert workflow.expected_row_count == 418_943
        assert set(workflow.input_sha256) == set(tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES)
        assert all(len(input_sha256) == 64 for input_sha256 in workflow.input_sha256.values())
        assert all(int(input_sha256, 16) >= 0 for input_sha256 in workflow.input_sha256.values())
        assert workflow.prediction_file_sha256
        assert all(len(file_sha256) == 64 for file_sha256 in workflow.prediction_file_sha256.values())
        assert all(int(file_sha256, 16) >= 0 for file_sha256 in workflow.prediction_file_sha256.values())
        assert {tolerance.observed_column for tolerance in workflow.tolerances} == expected_statistic_columns
        assert all(tolerance.absolute_tolerance > 0.0 for tolerance in workflow.tolerances)

    assert (
        metadata.workflow_by_identifier("quantitative_single_bgen_loco").gate_status
        == tests.parity.harness.ParityGateStatus.REQUIRED
    )
    assert (
        metadata.workflow_by_identifier("binary_score_only").gate_status
        == tests.parity.harness.ParityGateStatus.REQUIRED
    )
    assert (
        metadata.workflow_by_identifier("binary_approximate_firth").gate_status
        == tests.parity.harness.ParityGateStatus.REQUIRED
    )


def test_native_cli_config_emits_binary_section_only_for_binary_traits(tmp_path: Path) -> None:
    import tests.test_regenie2_parity

    metadata = tests.parity.harness.load_golden_metadata()
    quantitative_config_path = tests.test_regenie2_parity.write_native_cli_config(
        metadata.workflow_by_identifier("quantitative_single_bgen_loco"),
        output_root=tmp_path / "quantitative-output",
        jax_cache_directory=tmp_path / "jax-cache",
    )
    binary_config_path = tests.test_regenie2_parity.write_native_cli_config(
        metadata.workflow_by_identifier("binary_score_only"),
        output_root=tmp_path / "binary-output",
        jax_cache_directory=tmp_path / "jax-cache",
    )

    quantitative_config = quantitative_config_path.read_text(encoding="utf-8")
    binary_config = binary_config_path.read_text(encoding="utf-8")
    assert "\n[binary]\n" not in quantitative_config
    assert "\n[binary]\n" in binary_config
    assert 'fallback_method = "score_only"' in binary_config
    assert "p_threshold = 0.05" in binary_config
    assert "firth_se = false" in binary_config


def test_validation_node_files_and_documentation_exist() -> None:
    metadata = tests.parity.harness.load_golden_metadata()

    for workflow in metadata.workflows:
        for validation_node in workflow.validation_nodes:
            validation_path_text, test_name = validation_node.split("::", maxsplit=1)
            validation_path = tests.parity.harness.REPOSITORY_ROOT / validation_path_text
            assert validation_path.exists(), validation_node
            validation_module = ast.parse(validation_path.read_text(encoding="utf-8"))
            matching_tests = [
                node for node in validation_module.body if isinstance(node, ast.FunctionDef) and node.name == test_name
            ]
            assert len(matching_tests) == 1, validation_node
            decorator_names = {ast.unparse(decorator) for decorator in matching_tests[0].decorator_list}
            assert "pytest.mark.parity_required" in decorator_names, validation_node
        for documentation_path in workflow.documentation_paths:
            assert documentation_path.exists(), documentation_path

    parity_test_path = tests.parity.harness.REPOSITORY_ROOT / "tests/test_regenie2_parity.py"
    parity_test_module = ast.parse(parity_test_path.read_text(encoding="utf-8"))
    required_marker_tests = {
        node.name
        for node in parity_test_module.body
        if isinstance(node, ast.FunctionDef)
        and "pytest.mark.parity_required" in {ast.unparse(decorator) for decorator in node.decorator_list}
    }
    expected_required_tests = {
        *(
            validation_node.split("::", maxsplit=1)[1]
            for workflow in metadata.workflows
            for validation_node in workflow.validation_nodes
        ),
        "test_exact_head_qualification_bundle_covers_every_workflow",
    }
    assert required_marker_tests == expected_required_tests


def test_approximate_firth_preserves_existing_full_upstream_golden() -> None:
    metadata = tests.parity.harness.load_golden_metadata()
    workflow = metadata.workflow_by_identifier("binary_approximate_firth")

    assert workflow.status == tests.parity.harness.ParityWorkflowStatus.EXTERNAL_GOLDEN
    assert workflow.expected_output_relative_path == Path("baselines/regenie_step2_phenotype_binary.regenie")
    assert workflow.expected_output_sha256 == "0b9dc124525b6fec63e1b0d3f446263c05f690862235bd84f51b1b3c77b6ed72"
    assert workflow.expected_correction_count == 17_938
    assert workflow.expected_correction_failure_count == 0
    assert workflow.gate_status == tests.parity.harness.ParityGateStatus.REQUIRED
    assert workflow.qualification is None
    assert tuple(tolerance.absolute_tolerance for tolerance in workflow.tolerances) == (
        0.002,
        0.001,
        0.003,
        0.001,
    )


def test_score_only_preserves_existing_full_upstream_golden() -> None:
    metadata = tests.parity.harness.load_golden_metadata()
    workflow = metadata.workflow_by_identifier("binary_score_only")

    assert workflow.gate_status == tests.parity.harness.ParityGateStatus.REQUIRED
    assert workflow.expected_output_relative_path == Path("baselines/regenie_step2_score_only_phenotype_binary.regenie")
    assert workflow.expected_output_sha256 == "ba7278541d211a8ca446f5af3d45beba06030ad40f8124651db3038c196dac33"
    assert workflow.expected_log_sha256 == "c4002866c86dd67ebe23fcb563f17488635b59547cc30baa3a8566730e2e0e5b"
    assert workflow.expected_correction_count is None
    assert workflow.expected_correction_failure_count is None
    assert workflow.qualification is None
    assert tuple(tolerance.absolute_tolerance for tolerance in workflow.tolerances) == (
        0.001,
        0.001,
        0.02,
        0.02,
    )


def test_statistic_comparison_uses_full_variant_key_and_strict_tolerance() -> None:
    observed_results = variant_frame(beta_values=[1.25, 2.25], log10p_values=[2.0, 8.0])
    baseline_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[2.0, 8.0])
    tolerance = tests.parity.harness.StatisticTolerance(
        observed_column="BETA",
        baseline_column="BETA",
        absolute_tolerance=0.5,
    )

    comparison = tests.parity.harness.assert_statistic_columns_match(
        observed_results,
        baseline_results,
        tolerance=tolerance,
        expected_row_count=2,
    )

    assert comparison.row_count == 2
    assert comparison.maximum_absolute_difference < tolerance.absolute_tolerance

    boundary_results = variant_frame(beta_values=[1.5, 2.0], log10p_values=[2.0, 8.0])
    with pytest.raises(AssertionError):
        tests.parity.harness.assert_statistic_columns_match(
            boundary_results,
            baseline_results,
            tolerance=tolerance,
            expected_row_count=2,
        )


def test_negative_differences_use_the_same_strict_absolute_boundary() -> None:
    baseline_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[2.0, 8.0])
    tolerance = tests.parity.harness.StatisticTolerance(
        observed_column="BETA",
        baseline_column="BETA",
        absolute_tolerance=0.5,
    )
    within_results = variant_frame(beta_values=[0.75, 1.75], log10p_values=[2.0, 8.0])
    tests.parity.harness.assert_statistic_columns_match(
        within_results,
        baseline_results,
        tolerance=tolerance,
        expected_row_count=2,
    )

    for beta_value in (0.5, 0.49):
        outside_results = variant_frame(beta_values=[beta_value, 2.0], log10p_values=[2.0, 8.0])
        with pytest.raises(AssertionError):
            tests.parity.harness.assert_statistic_columns_match(
                outside_results,
                baseline_results,
                tolerance=tolerance,
                expected_row_count=2,
            )


def test_nonfinite_masks_match_exactly() -> None:
    reference_values = np.asarray([np.nan, np.inf, -np.inf, 1.0])
    tests.numerical.assert_absolute_difference_less_than(
        reference_values.copy(),
        reference_values,
        0.5,
    )

    mismatches = (
        np.asarray([0.0, np.inf, -np.inf, 1.0]),
        np.asarray([np.nan, -np.inf, -np.inf, 1.0]),
        np.asarray([np.nan, np.inf, np.inf, 1.0]),
    )
    for actual_values in mismatches:
        with pytest.raises(AssertionError):
            tests.numerical.assert_absolute_difference_less_than(
                actual_values,
                reference_values,
                0.5,
            )


def test_regenie_correction_summary_is_parsed_from_log(tmp_path: Path) -> None:
    log_path = tmp_path / "regenie.log"
    log_path.write_text(
        "Number of tests with Firth correction : 17938\nNumber of failed tests : (0/17938)\n",
        encoding="utf-8",
    )

    summary = tests.parity.harness.parse_regenie_correction_summary(log_path)

    assert summary == tests.parity.harness.RegenieCorrectionSummary(
        correction_count=17_938,
        correction_failure_count=0,
    )


def test_regenie_correction_summary_rejects_inconsistent_denominator(tmp_path: Path) -> None:
    log_path = tmp_path / "regenie.log"
    log_path.write_text(
        "Number of tests with Firth correction : 10\nNumber of failed tests : (1/9)\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="denominator mismatch"):
        tests.parity.harness.parse_regenie_correction_summary(log_path)


def test_regenie_score_only_log_has_no_correction_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "regenie.log"
    log_path.write_text("Number of ignored tests due to low MAC : 0\n", encoding="utf-8")

    assert tests.parity.harness.parse_regenie_correction_summary(log_path) is None


def test_prediction_list_members_are_resolved_inside_data_root(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    prediction_directory = data_directory / "baselines"
    prediction_directory.mkdir(parents=True)
    prediction_path = prediction_directory / "step1_1.loco"
    prediction_path.write_text("prediction data\n", encoding="utf-8")
    prediction_list_path = prediction_directory / "step1_pred.list"
    prediction_list_path.write_text(f"phenotype {prediction_path}\n", encoding="utf-8")

    resolved_paths = tests.parity.harness.resolve_prediction_files(
        prediction_list_path,
        data_directory=data_directory,
    )

    assert resolved_paths == {"baselines/step1_1.loco": prediction_path.resolve()}


def test_prediction_list_members_cannot_escape_data_root(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    outside_path = tmp_path / "outside.loco"
    outside_path.write_text("prediction data\n", encoding="utf-8")
    prediction_list_path = data_directory / "step1_pred.list"
    prediction_list_path.write_text(f"phenotype {outside_path}\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="escapes configured data directory"):
        tests.parity.harness.resolve_prediction_files(
            prediction_list_path,
            data_directory=data_directory,
        )


def test_missing_prediction_member_skips_optional_run_and_fails_required_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    workflow = dataclasses.replace(
        tests.parity.harness.load_golden_metadata().workflow_by_identifier("quantitative_single_bgen_loco"),
        expected_output_relative_path=Path("baselines/reference.regenie"),
        expected_log_relative_path=Path("baselines/reference.log"),
    )
    data_directory = tmp_path / "data"
    for option_name in tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES:
        input_path = data_directory / str(workflow.g_cli_options[option_name])
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.touch()
    prediction_list_path = data_directory / str(workflow.g_cli_options["prediction_list"])
    prediction_list_path.write_text("phenotype missing_1.loco\n", encoding="utf-8")
    for relative_path in (workflow.expected_output_relative_path, workflow.expected_log_relative_path):
        artifact_path = data_directory / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.touch()

    monkeypatch.setattr(tests.test_regenie2_parity, "DATA_DIRECTORY", data_directory)
    monkeypatch.delenv(tests.test_regenie2_parity.REQUIRE_DATA_ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(pytest.skip.Exception, match="Missing prediction file"):
        tests.test_regenie2_parity.require_or_skip_workflow_data(workflow)

    monkeypatch.setenv(tests.test_regenie2_parity.REQUIRE_DATA_ENVIRONMENT_VARIABLE, "1")
    with pytest.raises(pytest.fail.Exception, match="Missing prediction file"):
        tests.test_regenie2_parity.require_or_skip_workflow_data(workflow)


def test_failure_reporting_never_masks_original_contract_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    workflow = dataclasses.replace(
        tests.parity.harness.load_golden_metadata().workflow_by_identifier("quantitative_single_bgen_loco"),
        expected_row_count=2,
    )
    baseline_results = pl.DataFrame(
        {
            "CHROM": ["22", "22"],
            "GENPOS": [100, 200],
            "ID": ["first", "second"],
            "ALLELE0": ["A", "C"],
            "ALLELE1": ["G", "T"],
            "BETA": [1.0, 2.0],
            "SE": [0.1, 0.2],
            "CHISQ": [10.0, 20.0],
            "LOG10P": [2.0, 8.0],
            "N": [100, 100],
        }
    )
    observed_results = baseline_results.head(1).with_columns(
        pl.lit("score").alias("CORRECTION_METHOD"),
        pl.lit("success").alias("CORRECTION_STATUS"),
    )
    parity_results = tests.test_regenie2_parity.RegenieParityResults(
        observed_results=observed_results,
        baseline_results=baseline_results,
        output_root=tmp_path / "output",
        config_path=tmp_path / "config.toml",
        observed_input_sha256={},
        observed_prediction_file_sha256={},
        reference_correction_summary=None,
        exact_qualification_source=None,
    )

    def fail_report(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("report writer failed")

    monkeypatch.setattr(tests.test_regenie2_parity, "write_qualification_report", fail_report)
    with pytest.raises(AssertionError, match="g result schema mismatch"):
        tests.test_regenie2_parity.assert_and_record_workflow_qualification(workflow, parity_results)


def test_variant_key_order_must_match_exactly() -> None:
    observed_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[2.0, 8.0])
    baseline_results = observed_results.reverse()

    with pytest.raises(AssertionError, match="row order"):
        tests.parity.harness.assert_variant_key_order_match(
            observed_results,
            baseline_results,
            expected_row_count=2,
        )


def test_significance_classifications_are_exact() -> None:
    baseline_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[2.0, 8.0])
    matching_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[3.0, 9.0])

    tests.parity.harness.assert_significance_classifications_match(
        matching_results,
        baseline_results,
        expected_row_count=2,
    )

    mismatching_results = variant_frame(beta_values=[1.0, 2.0], log10p_values=[1.0, 9.0])
    with pytest.raises(AssertionError):
        tests.parity.harness.assert_significance_classifications_match(
            mismatching_results,
            baseline_results,
            expected_row_count=2,
        )
