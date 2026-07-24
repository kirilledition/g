from __future__ import annotations

import ast
import dataclasses
import datetime
import hashlib
import importlib.machinery
import json
import struct
import subprocess
import sys
import types
import typing
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import tests.numerical
import tests.parity.harness
import tooling.science_gate
import tooling.server.exact_parity_slurm


def write_synthetic_native_elf(
    file_path: Path,
    *,
    git_commit: str,
    science_source_sha256: str,
    run_nonce: str,
) -> None:
    """Write the minimal ELF and embedded identity accepted by focused tests."""
    elf_header = bytearray(64)
    elf_header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", elf_header, 16, 3, 62, 1)
    embedded_values = (
        b"PyInit__core",
        b"__build_git_commit__",
        b"__build_science_source_sha256__",
        b"__build_source_clean__",
        b"__build_profile__",
        b"__build_run_nonce__",
        git_commit.encode(),
        science_source_sha256.encode(),
        run_nonce.encode(),
    )
    file_path.write_bytes(bytes(elf_header) + b"\0".join(embedded_values) + b"\0")


def write_synthetic_slurm_attestation(
    file_path: Path,
    *,
    evidence: tests.parity.harness.QualificationEvidence,
) -> str:
    """Write a canonical private Slurm attestation for bundle tests."""
    bootstrap_path = "/tmp/exact_parity_bootstrap.sh"
    attestation = tooling.server.exact_parity_slurm.SlurmProcessAttestation(
        schema_version=tooling.server.exact_parity_slurm.SCHEMA_VERSION,
        cluster_name="abraxas",
        node_name=evidence.qualification_node,
        job_id=evidence.slurm_job_id,
        step_id=evidence.slurm_step_id,
        user_name="parity-user",
        user_id=1017,
        host_boot_id="12345678-1234-5678-9234-567812345678",
        host_process_id=4242,
        host_process_start_time_ticks=9001,
        host_process_pid_namespace_inode=4026533662,
        host_process_cgroup_namespace_inode=4026531835,
        cgroup_v2_path=f"/system.slice/slurmstepd.scope/job_{evidence.slurm_job_id}/step_{evidence.slurm_step_id}/user/task_0",
        job_node_count=1,
        job_cpu_count=8,
        job_memory_bytes=64 * 1024**3,
        job_gpu_count=1,
        job_task_count=1,
        step_node_count=1,
        step_cpu_count=8,
        step_memory_bytes=64 * 1024**3,
        step_gpu_count=1,
        step_task_count=1,
        first_job_record_sha256="1" * 64,
        second_job_record_sha256="2" * 64,
        first_step_record_sha256="3" * 64,
        second_step_record_sha256="4" * 64,
        listpids_sha256="5" * 64,
        bootstrap_path=bootstrap_path,
        bootstrap_sha256=evidence.bootstrap_sha256,
        job_command=bootstrap_path,
        bootstrap_process_command_sha256="6" * 64,
        constrain_cores=False,
        constrain_ram_space=False,
        constrain_devices=False,
        scheduler_entitlement_proven=True,
        kernel_enforcement_proven=False,
    )
    file_path.write_bytes(tooling.server.exact_parity_slurm.canonical_slurm_process_attestation(attestation))
    return tests.parity.harness.sha256_file(file_path)


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


def test_unreleased_qualification_schemas_remain_zero() -> None:
    """Keep every private pre-release evidence contract at schema zero."""
    import tests.test_regenie2_parity

    assert tests.parity.harness.PARITY_METADATA_SCHEMA_VERSION == 0
    assert tooling.server.exact_parity_slurm.SCHEMA_VERSION == 0
    assert tests.test_regenie2_parity.QUALIFICATION_REPORT_SCHEMA_VERSION == 0
    assert tests.test_regenie2_parity.QUALIFICATION_BUNDLE_SCHEMA_VERSION == 0


def test_exact_qualification_gpu_count_matches_slurm_attestation() -> None:
    """Keep runtime-device evidence aligned with the exact scheduler lane."""
    assert (
        tests.parity.harness.REQUIRED_QUALIFICATION_GPU_COUNT
        == tooling.server.exact_parity_slurm.QUALIFICATION_GPU_COUNT
    )


def qualification_evidence(
    workflow: tests.parity.harness.GoldenWorkflow,
    *,
    science_source_sha256: str,
) -> tests.parity.harness.QualificationEvidence:
    """Build complete synthetic evidence for focused metadata tests."""
    git_commit = "a" * 40
    toolchain = tests.parity.harness.QualificationToolchainEvidence(
        bash=tests.parity.harness.QualificationToolEvidence(
            path="/usr/bin/bash",
            sha256="2" * 64,
            version="GNU bash 5",
        ),
        ar=tests.parity.harness.QualificationToolEvidence(
            path="/usr/bin/ar",
            sha256="0" * 64,
            version="GNU ar 2",
        ),
        assembler=tests.parity.harness.QualificationToolEvidence(
            path="/usr/bin/as",
            sha256="1" * 64,
            version="GNU assembler 2",
        ),
        cc=tests.parity.harness.QualificationToolEvidence(
            path="/usr/bin/cc",
            sha256="d" * 64,
            version="cc 14",
        ),
        cc1=tests.parity.harness.QualificationToolEvidence(
            path="/usr/libexec/gcc/cc1",
            sha256="a" * 64,
            version="GNU C17 14",
        ),
        cc1plus=tests.parity.harness.QualificationToolEvidence(
            path="/usr/libexec/gcc/cc1plus",
            sha256="b" * 64,
            version="GNU C++17 14",
        ),
        cargo=tests.parity.harness.QualificationToolEvidence(
            path="/opt/tools/cargo",
            sha256="7" * 64,
            version="cargo 1",
        ),
        collect2=tests.parity.harness.QualificationToolEvidence(
            path="/usr/libexec/gcc/collect2",
            sha256="c" * 64,
            version="collect2 14",
        ),
        cxx=tests.parity.harness.QualificationToolEvidence(
            path="/usr/bin/c++",
            sha256="f" * 64,
            version="c++ 14",
        ),
        environment=tests.parity.harness.QualificationToolEvidence(
            path="/usr/bin/env",
            sha256="d" * 64,
            version="env 9",
        ),
        git=tests.parity.harness.QualificationToolEvidence(
            path="/usr/bin/git",
            sha256="3" * 64,
            version="git version 2",
        ),
        just=tests.parity.harness.QualificationToolEvidence(
            path="/opt/tools/just",
            sha256="4" * 64,
            version="just 1",
        ),
        maturin=tests.parity.harness.QualificationToolEvidence(
            path="/checkout/.venv/bin/maturin",
            sha256="e" * 64,
            version="maturin 1",
        ),
        mold=tests.parity.harness.QualificationToolEvidence(
            path="/opt/tools/mold",
            sha256="9" * 64,
            version="mold 2",
        ),
        python=tests.parity.harness.QualificationToolEvidence(
            path="/opt/tools/python3.14",
            sha256="b" * 64,
            version="Python 3.14",
        ),
        ranlib=tests.parity.harness.QualificationToolEvidence(
            path="/usr/bin/ranlib",
            sha256="c" * 64,
            version="GNU ranlib 2",
        ),
        rustc=tests.parity.harness.QualificationToolEvidence(
            path="/opt/tools/rustc",
            sha256="8" * 64,
            version="rustc 1",
        ),
        scontrol=tests.parity.harness.QualificationToolEvidence(
            path="/usr/bin/scontrol",
            sha256="5" * 64,
            version="slurm-wlm 23",
        ),
        uv=tests.parity.harness.QualificationToolEvidence(
            path="/opt/tools/uv",
            sha256="6" * 64,
            version="uv 0",
        ),
        venv_python=tests.parity.harness.QualificationToolEvidence(
            path="/checkout/.venv/bin/python",
            sha256="f" * 64,
            version="Python 3.14",
        ),
    )
    return tests.parity.harness.QualificationEvidence(
        passed=True,
        qualified_git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        working_tree_clean=True,
        qualification_generated_at_utc="2026-07-23T00:00:00+00:00",
        qualification_node="landau",
        slurm_job_id="12345",
        slurm_step_id="0",
        run_nonce="1" * 32,
        run_started_at_utc="2026-07-23T00:00:00+00:00",
        bootstrap_relative_path=tests.parity.harness.QUALIFICATION_BOOTSTRAP_RELATIVE_PATH,
        bootstrap_sha256="9" * 64,
        toolchain=toolchain,
        cargo_lock_sha256=tests.parity.harness.sha256_file(tests.parity.harness.REPOSITORY_ROOT / "Cargo.lock"),
        cargo_configuration_sha256=tests.parity.harness.sha256_file(
            tests.parity.harness.REPOSITORY_ROOT / ".cargo" / "config.toml"
        ),
        rust_toolchain_sha256=tests.parity.harness.sha256_file(
            tests.parity.harness.REPOSITORY_ROOT / "rust-toolchain.toml"
        ),
        rustflags_environment="",
        cargo_encoded_rustflags_environment="",
        rustc_wrapper_environment="",
        cargo_build_rustc_wrapper_environment="",
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
            run_nonce="1" * 32,
            library_sha256="b" * 64,
            library_size_bytes=1,
        ),
        observed_row_count=workflow.expected_row_count,
        observed_output_sha256="e" * 64,
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
        b'{"schema_version":0,"workflows":[{"identifier":"workflow",'
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


def test_qualification_rejects_future_generated_timestamp() -> None:
    workflow = tests.parity.harness.load_golden_metadata().workflow_by_identifier("binary_score_only")
    science_source_sha256 = "c" * 64
    evidence = qualification_evidence(workflow, science_source_sha256=science_source_sha256)
    future_evidence = dataclasses.replace(
        evidence,
        qualification_generated_at_utc=(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1)).isoformat(),
    )

    with pytest.raises(AssertionError, match="implausibly in the future"):
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            future_evidence,
            git_commit=evidence.qualified_git_commit,
            science_source_sha256=science_source_sha256,
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

    multiple_device_evidence = dataclasses.replace(
        evidence,
        actual_device=dataclasses.replace(evidence.actual_device, device_count=2),
    )
    with pytest.raises(AssertionError, match="exactly one JAX CUDA device"):
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            multiple_device_evidence,
            git_commit=evidence.qualified_git_commit,
            science_source_sha256=science_source_sha256,
        )

    stale_run_evidence = dataclasses.replace(evidence, run_nonce="2" * 32)
    with pytest.raises(AssertionError, match="run nonce differs"):
        tests.parity.harness.assert_workflow_qualification_is_current(
            workflow,
            stale_run_evidence,
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


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_exact_source_rejects_git_index_flags_hidden_from_status(
    tmp_path: Path,
    index_flag: str,
) -> None:
    repository_root = tmp_path / "repository"
    git_commit = initialize_test_repository(repository_root)
    subprocess.run(
        ["git", "-C", str(repository_root), "update-index", index_flag, "src/science.py"],
        check=True,
    )
    (repository_root / "src" / "science.py").write_text("result = 2\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
    ).stdout
    assert status == b""

    with pytest.raises(AssertionError, match="forbidden"):
        tooling.science_gate.assert_clean_exact_source(repository_root, git_commit)


def test_exact_source_byte_compares_head_index_and_disk_when_status_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_repository = tmp_path / "working"
    working_commit = initialize_test_repository(working_repository)
    (working_repository / "src" / "science.py").write_text("result = 2\n", encoding="utf-8")
    monkeypatch.setattr(tooling.science_gate, "repository_working_tree_status", lambda _root: b"")
    with pytest.raises(AssertionError, match="working-tree bytes differ"):
        tooling.science_gate.assert_clean_exact_source(working_repository, working_commit)

    index_repository = tmp_path / "index"
    index_commit = initialize_test_repository(index_repository)
    (index_repository / "src" / "science.py").write_text("result = 3\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(index_repository), "add", "src/science.py"], check=True)
    with pytest.raises(AssertionError, match="index differs from HEAD"):
        tooling.science_gate.assert_clean_exact_source(index_repository, index_commit)


def test_exact_source_rejects_hidden_imported_package_initializer(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    initialize_test_repository(repository_root)
    initializer_path = repository_root / "tooling" / "__init__.py"
    initializer_path.parent.mkdir()
    initializer_path.write_text('"""Trusted package."""\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repository_root), "add", "tooling/__init__.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "commit", "--quiet", "-m", "add package initializer"],
        check=True,
    )
    git_commit = tooling.science_gate.repository_git_commit(repository_root)
    subprocess.run(
        ["git", "-C", str(repository_root), "update-index", "--skip-worktree", "tooling/__init__.py"],
        check=True,
    )
    initializer_path.write_text("raise RuntimeError('untrusted')\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="forbidden"):
        tooling.science_gate.assert_clean_exact_source(repository_root, git_commit)


def test_exact_source_rejects_committed_science_symlink(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    initialize_test_repository(repository_root)
    source_path = repository_root / "src" / "science.py"
    source_path.unlink()
    source_path.symlink_to("missing.py")
    subprocess.run(["git", "-C", str(repository_root), "add", "src/science.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "commit", "--quiet", "-m", "replace source with symlink"],
        check=True,
    )
    git_commit = tooling.science_gate.repository_git_commit(repository_root)

    with pytest.raises(AssertionError, match="not a committed regular file"):
        tooling.science_gate.assert_clean_exact_source(repository_root, git_commit)


@pytest.mark.parametrize(
    "forbidden_environment_name",
    [
        "BASH_ENV",
        "G_REGENIE_PARITY_LD_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "SLURM_CLUSTERS",
        "SLURM_CONF",
        "SLURM_CONF_SERVER",
    ],
)
def test_exact_bootstrap_rejects_pre_shell_injection_environment(
    forbidden_environment_name: str,
) -> None:
    bootstrap_path = tests.parity.harness.REPOSITORY_ROOT / tests.parity.harness.QUALIFICATION_BOOTSTRAP_RELATIVE_PATH
    launch_environment = {
        "HOME": str(Path.home()),
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        forbidden_environment_name: "",
    }

    completed_process = subprocess.run(
        [
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            str(bootstrap_path),
            str(tests.parity.harness.REPOSITORY_ROOT),
            "a" * 40,
        ],
        env=launch_environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 1
    assert f"forbidden environment variable: {forbidden_environment_name}" in completed_process.stderr


def test_exact_inner_maturin_never_selects_managed_base_interpreter() -> None:
    justfile_text = (tests.parity.harness.REPOSITORY_ROOT / "Justfile").read_text(encoding="utf-8")
    synchronization_offset = justfile_text.index('"${uv_project[@]}" sync')
    virtual_environment_binding_offset = justfile_text.index(
        'export VIRTUAL_ENV="${qualification_checkout}/.venv"',
        synchronization_offset,
    )
    uv_python_binding_offset = justfile_text.index(
        'export UV_PYTHON="${virtual_environment_python_path}"',
        virtual_environment_binding_offset,
    )
    patchelf_rejection_offset = justfile_text.index(
        "if command -v patchelf >/dev/null 2>&1",
        uv_python_binding_offset,
    )
    maturin_offset = justfile_text.index(
        '"${trusted_maturin_path}" develop --profile release --uv',
        patchelf_rejection_offset,
    )
    build_environment_text = justfile_text[virtual_environment_binding_offset:maturin_offset]

    assert (
        synchronization_offset
        < virtual_environment_binding_offset
        < uv_python_binding_offset
        < patchelf_rejection_offset
        < maturin_offset
    )
    assert 'UV_PYTHON="${trusted_python_interpreter_path}"' not in build_environment_text
    assert 'VIRTUAL_ENV}" != "${qualification_checkout}/.venv"' in build_environment_text
    assert 'UV_PYTHON}" != "${virtual_environment_python_path}"' in build_environment_text


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
    write_synthetic_native_elf(
        native_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce="1" * 32,
    )
    native_core = types.SimpleNamespace(
        __file__=str(native_library_path),
        __build_git_commit__=git_commit,
        __build_science_source_sha256__="c" * 64,
        __build_source_clean__=True,
        __build_profile__="release",
        __build_run_nonce__="1" * 32,
    )
    monkeypatch.setenv(tests.test_regenie2_parity.EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE, git_commit)
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE,
        science_source_sha256,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE,
        str(native_library_path),
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE,
        tests.parity.harness.sha256_file(native_library_path),
    )
    monkeypatch.setenv(tests.test_regenie2_parity.RUN_NONCE_ENVIRONMENT_VARIABLE, "1" * 32)
    monkeypatch.setattr(
        tooling.science_gate,
        "assert_clean_exact_source",
        lambda _repository_root, _expected_git_commit: source_state,
    )

    with pytest.raises(AssertionError, match="stale science source"):
        tests.test_regenie2_parity.assert_exact_qualification_source(
            typing.cast("tests.test_regenie2_parity.NativeCoreProtocol", native_core)
        )


def test_exact_qualification_binds_loaded_native_path_and_run_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    git_commit = "a" * 40
    science_source_sha256 = "b" * 64
    loaded_library_path = tmp_path / "loaded_core.so"
    expected_library_path = tmp_path / "expected_core.so"
    write_synthetic_native_elf(
        loaded_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce="1" * 32,
    )
    write_synthetic_native_elf(
        expected_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce="1" * 32,
    )
    native_core = types.SimpleNamespace(
        __file__=str(loaded_library_path),
        __build_git_commit__=git_commit,
        __build_science_source_sha256__=science_source_sha256,
        __build_source_clean__=True,
        __build_profile__="release",
        __build_run_nonce__="1" * 32,
    )
    source_state = tooling.science_gate.ScienceSourceState(
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
    )
    monkeypatch.setenv(tests.test_regenie2_parity.EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE, git_commit)
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE,
        science_source_sha256,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE,
        str(expected_library_path),
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE,
        tests.parity.harness.sha256_file(expected_library_path),
    )
    monkeypatch.setenv(tests.test_regenie2_parity.RUN_NONCE_ENVIRONMENT_VARIABLE, "1" * 32)
    monkeypatch.setattr(
        tooling.science_gate,
        "assert_clean_exact_source",
        lambda _repository_root, _expected_git_commit: source_state,
    )

    with pytest.raises(AssertionError, match="path differs"):
        tests.test_regenie2_parity.assert_exact_qualification_source(
            typing.cast("tests.test_regenie2_parity.NativeCoreProtocol", native_core)
        )

    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE,
        str(loaded_library_path),
    )
    monkeypatch.setenv(tests.test_regenie2_parity.RUN_NONCE_ENVIRONMENT_VARIABLE, "2" * 32)
    write_synthetic_native_elf(
        loaded_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce="2" * 32,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE,
        tests.parity.harness.sha256_file(loaded_library_path),
    )
    with pytest.raises(AssertionError, match="wrong qualification nonce"):
        tests.test_regenie2_parity.assert_exact_qualification_source(
            typing.cast("tests.test_regenie2_parity.NativeCoreProtocol", native_core)
        )


def test_required_native_import_rejects_wrong_spec_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    expected_library_path = tmp_path / "expected_core.so"
    unexpected_library_path = tmp_path / "unexpected_core.so"
    git_commit = "a" * 40
    science_source_sha256 = "b" * 64
    run_nonce = "1" * 32
    write_synthetic_native_elf(
        expected_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce=run_nonce,
    )
    write_synthetic_native_elf(
        unexpected_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce=run_nonce,
    )
    monkeypatch.setenv(tests.test_regenie2_parity.REQUIRE_DATA_ENVIRONMENT_VARIABLE, "1")
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE,
        git_commit,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE,
        science_source_sha256,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.RUN_NONCE_ENVIRONMENT_VARIABLE,
        run_nonce,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE,
        str(expected_library_path),
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE,
        tests.parity.harness.sha256_file(expected_library_path),
    )
    unexpected_loader = importlib.machinery.ExtensionFileLoader(
        "g._core",
        str(unexpected_library_path),
    )
    unexpected_specification = importlib.machinery.ModuleSpec(
        "g._core",
        unexpected_loader,
        origin=str(unexpected_library_path),
        is_package=False,
    )
    unexpected_specification.has_location = True
    monkeypatch.setattr(
        tests.test_regenie2_parity.importlib.util,
        "find_spec",
        lambda _module_name: unexpected_specification,
    )
    imported_modules: list[str] = []
    monkeypatch.setattr(
        tests.test_regenie2_parity.importlib,
        "import_module",
        lambda module_name: imported_modules.append(module_name),
    )

    with pytest.raises(AssertionError, match="import path differs"):
        tests.test_regenie2_parity.load_native_core()
    assert imported_modules == []


def test_required_native_import_rejects_noncanonical_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    git_commit = "a" * 40
    science_source_sha256 = "b" * 64
    run_nonce = "1" * 32
    native_library_path = tmp_path / "_core.so"
    write_synthetic_native_elf(
        native_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce=run_nonce,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE,
        git_commit,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE,
        science_source_sha256,
    )
    monkeypatch.setenv(tests.test_regenie2_parity.RUN_NONCE_ENVIRONMENT_VARIABLE, run_nonce)
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE,
        str(native_library_path),
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE,
        tests.parity.harness.sha256_file(native_library_path),
    )
    expected_artifact = tests.test_regenie2_parity.expected_native_artifact()
    source_loader = importlib.machinery.SourceFileLoader("g._core", str(native_library_path))
    source_specification = importlib.machinery.ModuleSpec(
        "g._core",
        source_loader,
        origin=str(native_library_path),
        is_package=False,
    )

    with pytest.raises(AssertionError, match="did not use ExtensionFileLoader"):
        tests.test_regenie2_parity.validate_native_module_specification(
            source_specification,
            expected_artifact,
            label="synthetic",
        )

    class ForgedExtensionFileLoader(importlib.machinery.ExtensionFileLoader):
        """Loader subclass that could override extension execution."""

    forged_loader = ForgedExtensionFileLoader("g._core", str(native_library_path))
    forged_specification = importlib.machinery.ModuleSpec(
        "g._core",
        forged_loader,
        origin=str(native_library_path),
        is_package=False,
    )
    with pytest.raises(AssertionError, match="did not use ExtensionFileLoader"):
        tests.test_regenie2_parity.validate_native_module_specification(
            forged_specification,
            expected_artifact,
            label="synthetic",
        )

    overridden_loader = importlib.machinery.ExtensionFileLoader("g._core", str(native_library_path))
    setattr(overridden_loader, "exec_module", lambda _module: None)
    overridden_specification = importlib.machinery.ModuleSpec(
        "g._core",
        overridden_loader,
        origin=str(native_library_path),
        is_package=False,
    )
    with pytest.raises(AssertionError, match="noncanonical executable state"):
        tests.test_regenie2_parity.validate_native_module_specification(
            overridden_specification,
            expected_artifact,
            label="synthetic",
        )


def test_required_native_import_uses_one_validated_finder_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    git_commit = "a" * 40
    science_source_sha256 = "b" * 64
    run_nonce = "1" * 32
    native_library_path = tmp_path / "_core.so"
    write_synthetic_native_elf(
        native_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce=run_nonce,
    )
    monkeypatch.setenv(tests.test_regenie2_parity.REQUIRE_DATA_ENVIRONMENT_VARIABLE, "1")
    monkeypatch.setenv(tests.test_regenie2_parity.EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE, git_commit)
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE,
        science_source_sha256,
    )
    monkeypatch.setenv(tests.test_regenie2_parity.RUN_NONCE_ENVIRONMENT_VARIABLE, run_nonce)
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE,
        str(native_library_path),
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE,
        tests.parity.harness.sha256_file(native_library_path),
    )
    native_loader = importlib.machinery.ExtensionFileLoader("g._core", str(native_library_path))
    native_specification = importlib.machinery.ModuleSpec(
        "g._core",
        native_loader,
        origin=str(native_library_path),
        is_package=False,
    )
    native_specification.has_location = True
    finder_call_count = 0

    def find_spec_once(_module_name: str) -> importlib.machinery.ModuleSpec:
        nonlocal finder_call_count
        finder_call_count += 1
        if finder_call_count > 1:
            raise AssertionError("Required parity performed a second finder resolution")
        return native_specification

    monkeypatch.setattr(tests.test_regenie2_parity.importlib.util, "find_spec", find_spec_once)
    monkeypatch.setattr(tests.test_regenie2_parity, "_QUALIFICATION_NATIVE_MODULE", None)
    monkeypatch.delitem(sys.modules, "g._core", raising=False)
    parent_module = importlib.import_module("g")
    monkeypatch.delattr(parent_module, "_core", raising=False)

    with pytest.raises(ImportError):
        tests.test_regenie2_parity.load_native_core()

    assert finder_call_count == 1
    assert "g._core" not in sys.modules
    assert not hasattr(parent_module, "_core")


def test_required_native_import_rejects_preloaded_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    git_commit = "a" * 40
    science_source_sha256 = "b" * 64
    run_nonce = "1" * 32
    native_library_path = tmp_path / "_core.so"
    write_synthetic_native_elf(
        native_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce=run_nonce,
    )
    monkeypatch.setenv(tests.test_regenie2_parity.REQUIRE_DATA_ENVIRONMENT_VARIABLE, "1")
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE,
        git_commit,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE,
        science_source_sha256,
    )
    monkeypatch.setenv(tests.test_regenie2_parity.RUN_NONCE_ENVIRONMENT_VARIABLE, run_nonce)
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE,
        str(native_library_path),
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE,
        tests.parity.harness.sha256_file(native_library_path),
    )
    native_loader = importlib.machinery.ExtensionFileLoader("g._core", str(native_library_path))
    native_specification = importlib.machinery.ModuleSpec(
        "g._core",
        native_loader,
        origin=str(native_library_path),
        is_package=False,
    )
    native_specification.has_location = True
    monkeypatch.setattr(
        tests.test_regenie2_parity.importlib.util,
        "find_spec",
        lambda _module_name: native_specification,
    )
    monkeypatch.setattr(tests.test_regenie2_parity, "_QUALIFICATION_NATIVE_MODULE", None)
    monkeypatch.setitem(sys.modules, "g._core", types.ModuleType("g._core"))

    with pytest.raises(AssertionError, match="loaded before validation"):
        tests.test_regenie2_parity.load_native_core()


def test_required_native_artifact_rejects_wrong_elf_machine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    git_commit = "a" * 40
    science_source_sha256 = "b" * 64
    run_nonce = "1" * 32
    native_library_path = tmp_path / "_core.so"
    write_synthetic_native_elf(
        native_library_path,
        git_commit=git_commit,
        science_source_sha256=science_source_sha256,
        run_nonce=run_nonce,
    )
    native_library_bytes = bytearray(native_library_path.read_bytes())
    struct.pack_into("<H", native_library_bytes, 18, 183)
    native_library_path.write_bytes(native_library_bytes)
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE,
        git_commit,
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE,
        science_source_sha256,
    )
    monkeypatch.setenv(tests.test_regenie2_parity.RUN_NONCE_ENVIRONMENT_VARIABLE, run_nonce)
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE,
        str(native_library_path),
    )
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE,
        tests.parity.harness.sha256_file(native_library_path),
    )

    with pytest.raises(AssertionError, match="wrong ELF type, machine, or version"):
        tests.test_regenie2_parity.expected_native_artifact()


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

    fake_jax_module.devices = lambda: [cuda_device, cuda_device]
    with pytest.raises(AssertionError, match="exactly one JAX device"):
        tests.test_regenie2_parity.observe_qualification_device()

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

    def synthetic_input_paths(
        workflow: tests.parity.harness.GoldenWorkflow,
    ) -> dict[str, Path]:
        return {
            option_name: tmp_path / "protected" / workflow.identifier / "inputs" / option_name
            for option_name in tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES
        }

    def synthetic_prediction_paths(
        workflow: tests.parity.harness.GoldenWorkflow,
    ) -> dict[str, Path]:
        return {
            relative_path: tmp_path / "protected" / workflow.identifier / "predictions" / relative_path
            for relative_path in workflow.prediction_file_sha256
        }

    monkeypatch.setattr(
        tests.test_regenie2_parity,
        "workflow_input_paths",
        synthetic_input_paths,
    )
    monkeypatch.setattr(
        tests.test_regenie2_parity,
        "workflow_prediction_file_paths",
        synthetic_prediction_paths,
    )
    science_source_sha256 = "c" * 64
    report_directory = tmp_path / "bundle"
    report_directory.mkdir()
    native_library_path = tmp_path / "protected" / "_core.so"
    native_library_path.parent.mkdir()
    native_library_path.write_bytes(b"x")
    native_library_sha256 = tests.parity.harness.sha256_file(native_library_path)
    workflows = tests.parity.harness.load_golden_metadata().workflows
    slurm_attestation_relative_path = "slurm_process_attestation.json"
    slurm_attestation_sha256 = write_synthetic_slurm_attestation(
        report_directory / slurm_attestation_relative_path,
        evidence=qualification_evidence(
            workflows[0],
            science_source_sha256=science_source_sha256,
        ),
    )
    reports: list[tests.test_regenie2_parity.WorkflowQualificationReport] = []
    for workflow in workflows:
        evidence = qualification_evidence(
            workflow,
            science_source_sha256=science_source_sha256,
        )
        protected_directory = tmp_path / "protected" / workflow.identifier
        output_root = protected_directory / "output"
        output_dataset_directory = output_root / "run" / "parts"
        output_dataset_directory.mkdir(parents=True)
        parquet_path = output_dataset_directory / "part-0.parquet"
        parquet_path.write_bytes(workflow.identifier.encode())
        run_manifest_path = output_root / "run" / "run_manifest.json"
        effective_config_path = output_root / "run" / "effective_config.toml"
        run_manifest_path.write_text("{}\n", encoding="utf-8")
        effective_config_path.write_text("[output]\n", encoding="utf-8")
        configuration_path = protected_directory / "config.toml"
        configuration_path.write_text("[parity]\n", encoding="utf-8")
        observed_output_sha256 = tests.parity.harness.sha256_file_set(
            (parquet_path,),
            root=output_root,
        )
        evidence = dataclasses.replace(
            evidence,
            observed_output_sha256=observed_output_sha256,
            native_build=dataclasses.replace(
                evidence.native_build,
                library_sha256=native_library_sha256,
                library_size_bytes=native_library_path.stat().st_size,
            ),
        )
        workflow_report_directory = report_directory / workflow.identifier
        workflow_report_directory.mkdir()
        report_path = workflow_report_directory / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": tests.test_regenie2_parity.QUALIFICATION_REPORT_SCHEMA_VERSION,
                    "generated_at_utc": evidence.qualification_generated_at_utc,
                    "run": {
                        "qualification_node": evidence.qualification_node,
                        "slurm_job_id": evidence.slurm_job_id,
                        "slurm_step_id": evidence.slurm_step_id,
                        "run_nonce": evidence.run_nonce,
                        "run_started_at_utc": evidence.run_started_at_utc,
                        "bootstrap_relative_path": evidence.bootstrap_relative_path,
                        "bootstrap_sha256": evidence.bootstrap_sha256,
                        "slurm_attestation_relative_path": slurm_attestation_relative_path,
                        "slurm_attestation_sha256": slurm_attestation_sha256,
                        "toolchain": tests.parity.harness.qualification_toolchain_evidence_payload(evidence.toolchain),
                    },
                    "workflow": {
                        "identifier": workflow.identifier,
                        "gate_status": workflow.gate_status.value,
                        "regenie_version": workflow.regenie_version,
                    },
                    "qualification": {"passed": True, "failure": None},
                    "qualification_evidence": tests.parity.harness.qualification_evidence_payload(evidence),
                    "source": {
                        "git_commit": evidence.qualified_git_commit,
                        "working_tree_dirty": False,
                        "git_status_sha256": hashlib.sha256(b"").hexdigest(),
                        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
                        "science_source_sha256": evidence.science_source_sha256,
                        "native_library_path": str(native_library_path),
                        "native_library_sha256": evidence.native_build.library_sha256,
                        "native_build_git_commit": evidence.native_build.git_commit,
                        "native_build_science_source_sha256": evidence.native_build.science_source_sha256,
                        "native_build_source_clean": evidence.native_build.source_clean,
                        "native_build_profile": evidence.native_build.profile.value,
                        "native_build_run_nonce": evidence.native_build.run_nonce,
                        "cargo_lock_sha256": evidence.cargo_lock_sha256,
                        "cargo_configuration_sha256": evidence.cargo_configuration_sha256,
                        "rust_toolchain_sha256": evidence.rust_toolchain_sha256,
                        "rustflags_environment": evidence.rustflags_environment,
                        "cargo_encoded_rustflags_environment": evidence.cargo_encoded_rustflags_environment,
                        "rustc_wrapper_environment": evidence.rustc_wrapper_environment,
                        "cargo_build_rustc_wrapper_environment": (evidence.cargo_build_rustc_wrapper_environment),
                        "uv_lock_sha256": evidence.uv_lock_sha256,
                    },
                    "runtime": {
                        "jax_version": evidence.jax_version,
                        "jaxlib_version": evidence.jaxlib_version,
                        "configured_device": evidence.configured_device,
                        "jax_platforms_environment": "cuda",
                    },
                    "configuration": {
                        "metadata_options": workflow.g_cli_options,
                        "toml_path": str(configuration_path),
                        "toml_sha256": tests.parity.harness.sha256_file(configuration_path),
                    },
                    "inputs": {
                        option_name: {
                            "path": str(tests.test_regenie2_parity.workflow_input_paths(workflow)[option_name]),
                            "sha256": workflow.input_sha256[option_name],
                        }
                        for option_name in tests.parity.harness.REQUIRED_INPUT_OPTION_NAMES
                    },
                    "prediction_files": {
                        relative_path: {
                            "path": str(
                                tests.test_regenie2_parity.workflow_prediction_file_paths(workflow)[relative_path]
                            ),
                            "sha256": prediction_sha256,
                        }
                        for relative_path, prediction_sha256 in workflow.prediction_file_sha256.items()
                    },
                    "reference": {
                        "output_path": str(
                            tests.test_regenie2_parity.DATA_DIRECTORY / workflow.expected_output_relative_path
                        ),
                        "output_sha256": workflow.expected_output_sha256,
                        "log_path": str(
                            tests.test_regenie2_parity.DATA_DIRECTORY / workflow.expected_log_relative_path
                        ),
                        "log_sha256": workflow.expected_log_sha256,
                        "corrections": tests.test_regenie2_parity.correction_summary_payload(
                            None
                            if workflow.expected_correction_count is None
                            else tests.parity.harness.RegenieCorrectionSummary(
                                correction_count=workflow.expected_correction_count,
                                correction_failure_count=typing.cast(
                                    "int",
                                    workflow.expected_correction_failure_count,
                                ),
                            )
                        ),
                    },
                    "output": {
                        "root": str(output_root),
                        "dataset_directory": str(output_dataset_directory),
                        "completion_line": f"Parquet dataset saved to {output_dataset_directory}",
                        "row_count": workflow.expected_row_count,
                        "column_order": [field.name for field in evidence.output_fields],
                        "schema": [
                            {
                                "name": field.name,
                                "data_type": field.data_type.value,
                                "nullable": field.nullable,
                            }
                            for field in evidence.output_fields
                        ],
                        "parquet_dataset_sha256": evidence.observed_output_sha256,
                        "parquet_files": [
                            {
                                "relative_path": parquet_path.relative_to(output_root).as_posix(),
                                "sha256": tests.parity.harness.sha256_file(parquet_path),
                            }
                        ],
                        "run_metadata_files": [
                            {
                                "relative_path": metadata_path.relative_to(output_root).as_posix(),
                                "sha256": tests.parity.harness.sha256_file(metadata_path),
                            }
                            for metadata_path in (effective_config_path, run_manifest_path)
                        ],
                        "corrections": tests.test_regenie2_parity.correction_summary_payload(
                            tests.parity.harness.RegenieCorrectionSummary(
                                correction_count=evidence.observed_correction_count,
                                correction_failure_count=evidence.observed_correction_failure_count,
                            )
                        ),
                    },
                    "statistics": [
                        {
                            "observed_column": statistic.observed_column,
                            "reference_column": statistic.baseline_column,
                            "row_count": evidence.observed_row_count,
                            "maximum_absolute_difference": statistic.maximum_absolute_difference,
                            "absolute_tolerance": statistic.absolute_tolerance,
                        }
                        for statistic in evidence.statistics
                    ],
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
        str(report_directory),
    )
    expected_bundle_path = report_directory / (f"qualification_bundle_{'a' * 40}_{'12345'}_0_{'1' * 32}.json")
    monkeypatch.setenv(
        tests.test_regenie2_parity.EXPECTED_BUNDLE_PATH_ENVIRONMENT_VARIABLE,
        str(expected_bundle_path),
    )
    monkeypatch.setattr(
        tooling.science_gate,
        "assert_clean_exact_source",
        lambda repository_root, expected_git_commit: tooling.science_gate.ScienceSourceState(
            git_commit=expected_git_commit,
            science_source_sha256=science_source_sha256,
        ),
    )
    monkeypatch.setattr(
        tests.test_regenie2_parity,
        "snapshot_workflow_artifacts",
        lambda workflow: tests.test_regenie2_parity.WorkflowArtifactSnapshot(
            input_sha256=workflow.input_sha256,
            prediction_file_sha256=workflow.prediction_file_sha256,
            reference_output_sha256=workflow.expected_output_sha256,
            reference_log_sha256=workflow.expected_log_sha256,
        ),
    )

    with pytest.raises(AssertionError, match="workflow mismatch"):
        tests.test_regenie2_parity.write_qualification_bundle(tuple(reports[:-1]))

    first_report_path = reports[0].report_path
    first_report_text = first_report_path.read_text(encoding="utf-8")
    wrong_report_schema = json.loads(first_report_text)
    wrong_report_schema["schema_version"] = 1
    first_report_path.write_text(json.dumps(wrong_report_schema), encoding="utf-8")
    with pytest.raises(AssertionError, match="Unsupported qualification report schema"):
        tests.test_regenie2_parity.write_qualification_bundle(tuple(reports))
    first_report_path.write_text(first_report_text, encoding="utf-8")
    boolean_report_schema = json.loads(first_report_text)
    boolean_report_schema["schema_version"] = False
    first_report_path.write_text(json.dumps(boolean_report_schema), encoding="utf-8")
    with pytest.raises(AssertionError, match="Unsupported qualification report schema"):
        tests.test_regenie2_parity.write_qualification_bundle(tuple(reports))
    first_report_path.write_text(first_report_text, encoding="utf-8")

    slurm_attestation_path = report_directory / slurm_attestation_relative_path
    slurm_attestation_bytes = slurm_attestation_path.read_bytes()
    slurm_attestation_path.write_bytes(slurm_attestation_bytes + b" ")
    with pytest.raises(AssertionError, match="Slurm attestation ancestor mismatch"):
        tests.test_regenie2_parity.write_qualification_bundle(tuple(reports))
    slurm_attestation_path.write_bytes(slurm_attestation_bytes)

    bundle_path = tests.test_regenie2_parity.write_qualification_bundle(tuple(reports))
    bundle_text = bundle_path.read_text(encoding="utf-8")
    assert str(tmp_path / "protected") not in bundle_text
    assert {
        workflow_payload["identifier"] for workflow_payload in json.loads(bundle_text)["workflows"]
    } == tests.parity.harness.REQUIRED_WORKFLOW_IDENTIFIERS

    bundle_alias_path = report_directory / "qualification_bundle_alias.json"
    bundle_alias_path.symlink_to(bundle_path.name)
    with pytest.raises(AssertionError, match="canonical nonsymbolic file"):
        tests.test_regenie2_parity.validate_published_qualification_bundle(
            bundle_alias_path,
            expected_git_commit="a" * 40,
            expected_science_source_sha256=science_source_sha256,
            expected_slurm_job_id="12345",
            expected_slurm_step_id="0",
            expected_run_nonce="1" * 32,
            expected_run_started_at_utc="2026-07-23T00:00:00+00:00",
            expected_bootstrap_sha256="9" * 64,
        )
    bundle_alias_path.unlink()

    with pytest.raises(FileExistsError):
        tests.test_regenie2_parity.write_qualification_bundle(tuple(reports))

    original_bundle_text = bundle_path.read_text(encoding="utf-8")
    wrong_bundle_schema = json.loads(original_bundle_text)
    wrong_bundle_schema["schema_version"] = 1
    bundle_path.write_text(json.dumps(wrong_bundle_schema), encoding="utf-8")
    with pytest.raises(AssertionError, match="Unsupported qualification bundle schema"):
        tests.test_regenie2_parity.validate_published_qualification_bundle(
            bundle_path,
            expected_git_commit="a" * 40,
            expected_science_source_sha256=science_source_sha256,
            expected_slurm_job_id="12345",
            expected_slurm_step_id="0",
            expected_run_nonce="1" * 32,
            expected_run_started_at_utc="2026-07-23T00:00:00+00:00",
            expected_bootstrap_sha256="9" * 64,
        )
    bundle_path.write_text(original_bundle_text, encoding="utf-8")
    boolean_bundle_schema = json.loads(original_bundle_text)
    boolean_bundle_schema["schema_version"] = False
    bundle_path.write_text(json.dumps(boolean_bundle_schema), encoding="utf-8")
    with pytest.raises(AssertionError, match="Unsupported qualification bundle schema"):
        tests.test_regenie2_parity.validate_published_qualification_bundle(
            bundle_path,
            expected_git_commit="a" * 40,
            expected_science_source_sha256=science_source_sha256,
            expected_slurm_job_id="12345",
            expected_slurm_step_id="0",
            expected_run_nonce="1" * 32,
            expected_run_started_at_utc="2026-07-23T00:00:00+00:00",
            expected_bootstrap_sha256="9" * 64,
        )
    bundle_path.write_text(original_bundle_text, encoding="utf-8")

    original_bundle_payload = json.loads(original_bundle_text)
    original_report_relative_path = original_bundle_payload["workflows"][0]["qualification_report_relative_path"]
    for noncanonical_report_path in (
        str(report_directory / original_report_relative_path),
        f"{Path(original_report_relative_path).parent}/../{original_report_relative_path}",
    ):
        noncanonical_bundle_payload = json.loads(original_bundle_text)
        noncanonical_bundle_payload["workflows"][0]["qualification_report_relative_path"] = noncanonical_report_path
        bundle_path.write_text(json.dumps(noncanonical_bundle_payload), encoding="utf-8")
        with pytest.raises(AssertionError, match="not canonical and relative"):
            tests.test_regenie2_parity.validate_published_qualification_bundle(
                bundle_path,
                expected_git_commit="a" * 40,
                expected_science_source_sha256=science_source_sha256,
                expected_slurm_job_id="12345",
                expected_slurm_step_id="0",
                expected_run_nonce="1" * 32,
                expected_run_started_at_utc="2026-07-23T00:00:00+00:00",
                expected_bootstrap_sha256="9" * 64,
            )
    bundle_path.write_text(original_bundle_text, encoding="utf-8")

    future_bundle_payload = json.loads(original_bundle_text)
    future_bundle_payload["generated_at_utc"] = (
        datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1)
    ).isoformat()
    bundle_path.write_text(json.dumps(future_bundle_payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="implausibly in the future"):
        tests.test_regenie2_parity.validate_published_qualification_bundle(
            bundle_path,
            expected_git_commit="a" * 40,
            expected_science_source_sha256=science_source_sha256,
            expected_slurm_job_id="12345",
            expected_slurm_step_id="0",
            expected_run_nonce="1" * 32,
            expected_run_started_at_utc="2026-07-23T00:00:00+00:00",
            expected_bootstrap_sha256="9" * 64,
        )
    bundle_path.write_text(original_bundle_text, encoding="utf-8")

    reports[1].report_path.write_text(
        f"{reports[1].report_path.read_text(encoding='utf-8')}\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="report digest mismatch"):
        tests.test_regenie2_parity.validate_published_qualification_bundle(
            bundle_path,
            expected_git_commit="a" * 40,
            expected_science_source_sha256=science_source_sha256,
            expected_slurm_job_id="12345",
            expected_slurm_step_id="0",
            expected_run_nonce="1" * 32,
            expected_run_started_at_utc="2026-07-23T00:00:00+00:00",
            expected_bootstrap_sha256="9" * 64,
        )


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


def test_completed_parquet_dataset_uses_exact_native_completion_path(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    selected_directory = output_root / "attempts" / "attempt-1" / "phenotype" / "parts"
    selected_directory.mkdir(parents=True)
    selected_part = selected_directory / "part-0000.parquet"
    selected_part.touch()
    decoy_directory = output_root / "decoy.run" / "parts"
    decoy_directory.mkdir(parents=True)
    (decoy_directory / "decoy.parquet").touch()
    completion_line = f"{tests.parity.harness.PARQUET_DATASET_COMPLETION_PREFIX}{selected_directory}"

    dataset = tests.parity.harness.completed_parquet_dataset(
        output_root,
        ("unrelated output\n", f"{completion_line}\n"),
    )

    assert dataset.directory == selected_directory
    assert dataset.completion_line == completion_line
    assert dataset.parquet_paths == (selected_part,)


def test_completed_parquet_dataset_rejects_missing_duplicate_or_escaping_paths(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    dataset_directory = output_root / "run" / "parts"
    dataset_directory.mkdir(parents=True)
    (dataset_directory / "part.parquet").touch()
    completion_line = f"{tests.parity.harness.PARQUET_DATASET_COMPLETION_PREFIX}{dataset_directory}"
    with pytest.raises(AssertionError, match="found 0"):
        tests.parity.harness.completed_parquet_dataset(output_root, ("other output\n",))
    with pytest.raises(AssertionError, match="found 2"):
        tests.parity.harness.completed_parquet_dataset(
            output_root,
            (f"{completion_line}\n{completion_line}\n",),
        )
    with pytest.raises(AssertionError, match="must be absolute"):
        tests.parity.harness.completed_parquet_dataset(
            output_root,
            (f"{tests.parity.harness.PARQUET_DATASET_COMPLETION_PREFIX}relative/parts\n",),
        )

    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    (outside_directory / "part.parquet").touch()
    with pytest.raises(AssertionError, match="escapes"):
        tests.parity.harness.completed_parquet_dataset(
            output_root,
            (f"{tests.parity.harness.PARQUET_DATASET_COMPLETION_PREFIX}{outside_directory}\n",),
        )
    escaping_link = output_root / "escaping-parts"
    escaping_link.symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(AssertionError, match="escapes"):
        tests.parity.harness.completed_parquet_dataset(
            output_root,
            (f"{tests.parity.harness.PARQUET_DATASET_COMPLETION_PREFIX}{escaping_link}\n",),
        )


def test_output_artifact_snapshot_rejects_late_byte_or_file_set_changes(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    dataset_directory = output_root / "run" / "parts"
    dataset_directory.mkdir(parents=True)
    first_part = dataset_directory / "part-0000.parquet"
    first_part.write_bytes(b"initial")
    completion_line = f"{tests.parity.harness.PARQUET_DATASET_COMPLETION_PREFIX}{dataset_directory}"
    dataset = tests.parity.harness.completed_parquet_dataset(output_root, (f"{completion_line}\n",))
    snapshot = tests.test_regenie2_parity.snapshot_output_artifacts(dataset)

    first_part.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="Production output changed"):
        tests.test_regenie2_parity.assert_output_artifact_snapshot_unchanged(dataset, snapshot)

    first_part.write_bytes(b"initial")
    tests.test_regenie2_parity.assert_output_artifact_snapshot_unchanged(dataset, snapshot)
    (dataset_directory / "part-0001.parquet").write_bytes(b"late")
    with pytest.raises(AssertionError, match="Production output changed"):
        tests.test_regenie2_parity.assert_output_artifact_snapshot_unchanged(dataset, snapshot)


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


def test_workflow_artifact_snapshot_rejects_mid_run_fixture_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    data_directory = tmp_path / "data"
    data_directory.mkdir()
    options = dict(
        tests.parity.harness.load_golden_metadata()
        .workflow_by_identifier("quantitative_single_bgen_loco")
        .g_cli_options
    )
    relative_input_paths = {
        "bgen": Path("input.bgen"),
        "sample": Path("input.sample"),
        "phenotype_file": Path("phenotype.txt"),
        "covariate_file": Path("covariate.txt"),
        "prediction_list": Path("prediction.list"),
    }
    for option_name, relative_path in relative_input_paths.items():
        options[option_name] = relative_path.as_posix()
        if option_name != "prediction_list":
            (data_directory / relative_path).write_text(f"{option_name}\n", encoding="utf-8")
    prediction_path = data_directory / "prediction.loco"
    prediction_path.write_text("prediction\n", encoding="utf-8")
    prediction_list_path = data_directory / relative_input_paths["prediction_list"]
    prediction_list_path.write_text("phenotype prediction.loco\n", encoding="utf-8")
    reference_output_path = data_directory / "reference.regenie"
    reference_log_path = data_directory / "reference.log"
    reference_output_path.write_text("reference\n", encoding="utf-8")
    reference_log_path.write_text("log\n", encoding="utf-8")
    workflow = dataclasses.replace(
        tests.parity.harness.load_golden_metadata().workflow_by_identifier("quantitative_single_bgen_loco"),
        g_cli_options=options,
        expected_output_relative_path=Path("reference.regenie"),
        expected_output_sha256=tests.parity.harness.sha256_file(reference_output_path),
        expected_log_relative_path=Path("reference.log"),
        expected_log_sha256=tests.parity.harness.sha256_file(reference_log_path),
        input_sha256={
            option_name: tests.parity.harness.sha256_file(data_directory / relative_path)
            for option_name, relative_path in relative_input_paths.items()
        },
        prediction_file_sha256={
            "prediction.loco": tests.parity.harness.sha256_file(prediction_path),
        },
    )
    monkeypatch.setattr(tests.test_regenie2_parity, "DATA_DIRECTORY", data_directory)

    snapshot = tests.test_regenie2_parity.snapshot_workflow_artifacts(workflow)
    prediction_path.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="changed during qualification"):
        tests.test_regenie2_parity.assert_workflow_artifact_snapshot_unchanged(workflow, snapshot)


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


def test_data_present_optional_mode_skips_exact_bundle_without_source_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_regenie2_parity

    workflow = tests.parity.harness.load_golden_metadata().workflow_by_identifier("quantitative_single_bgen_loco")
    fixture_path = tmp_path / "fixture"
    fixture_path.write_bytes(b"present")
    monkeypatch.setattr(
        tests.test_regenie2_parity,
        "required_workflow_paths",
        lambda _workflow: (fixture_path,),
    )
    monkeypatch.setattr(
        tests.test_regenie2_parity,
        "workflow_prediction_file_paths",
        lambda _workflow: {"prediction.loco": fixture_path},
    )
    for variable_name in (
        tests.test_regenie2_parity.REQUIRE_DATA_ENVIRONMENT_VARIABLE,
        tests.test_regenie2_parity.EXPECTED_GIT_COMMIT_ENVIRONMENT_VARIABLE,
        tests.test_regenie2_parity.EXPECTED_SCIENCE_SOURCE_ENVIRONMENT_VARIABLE,
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_PATH_ENVIRONMENT_VARIABLE,
        tests.test_regenie2_parity.EXPECTED_NATIVE_LIBRARY_SHA256_ENVIRONMENT_VARIABLE,
    ):
        monkeypatch.delenv(variable_name, raising=False)

    tests.test_regenie2_parity.require_or_skip_workflow_data(workflow)
    with pytest.raises(pytest.skip.Exception, match="required parity recipe"):
        tests.test_regenie2_parity.require_exact_bundle_mode()


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
        output_dataset=tests.parity.harness.CompletedParquetDataset(
            output_root=tmp_path / "output",
            directory=tmp_path / "output" / "run" / "parts",
            completion_line=f"Parquet dataset saved to {tmp_path / 'output' / 'run' / 'parts'}",
            parquet_paths=(),
        ),
        config_path=tmp_path / "config.toml",
        artifact_snapshot=tests.test_regenie2_parity.WorkflowArtifactSnapshot(
            input_sha256={},
            prediction_file_sha256={},
            reference_output_sha256="a" * 64,
            reference_log_sha256="b" * 64,
        ),
        output_artifact_snapshot=tests.test_regenie2_parity.OutputArtifactSnapshot(
            parquet_file_sha256={},
            parquet_dataset_sha256="c" * 64,
        ),
        reference_correction_summary=None,
        exact_qualification_source=None,
    )

    def fail_report(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("report writer failed")

    monkeypatch.setattr(tests.test_regenie2_parity, "write_qualification_report", fail_report)
    monkeypatch.setattr(
        tests.test_regenie2_parity,
        "assert_workflow_artifact_snapshot_unchanged",
        lambda *_arguments: None,
    )
    monkeypatch.setattr(
        tests.test_regenie2_parity,
        "assert_output_artifact_snapshot_unchanged",
        lambda *_arguments: None,
    )
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
