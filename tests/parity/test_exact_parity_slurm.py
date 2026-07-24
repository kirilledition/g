"""Adversarial tests for exact Slurm process attestation."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import typing

import pytest

import tooling.server.exact_parity_slurm

if typing.TYPE_CHECKING:
    from pathlib import Path

JOB_ID = "12345"
STEP_ID = "0"
USER_NAME = "parity-user"
USER_ID = 1017
PROCESS_ID = 4242
CPU_COUNT = 8
MEMORY_BYTES = 64 * 1024**3
GPU_COUNT = 1
HOST_BOOT_ID = "12345678-1234-5678-9234-567812345678"
PID_NAMESPACE_INODE = 4026533662
CGROUP_NAMESPACE_INODE = 4026531835


def validation_request(tmp_path: Path) -> tooling.server.exact_parity_slurm.SlurmValidationRequest:
    bootstrap_path = tmp_path / "exact-parity-bootstrap"
    bootstrap_path.write_bytes(b"#!/usr/bin/bash\nexit 0\n")
    bootstrap_path.chmod(0o500)
    source_repository = tmp_path / "source"
    source_repository.mkdir()
    return tooling.server.exact_parity_slurm.SlurmValidationRequest(
        schema_version=tooling.server.exact_parity_slurm.SCHEMA_VERSION,
        cluster_name="abraxas",
        node_name="landau",
        job_id=JOB_ID,
        step_id=STEP_ID,
        user_name=USER_NAME,
        user_id=USER_ID,
        process_id=PROCESS_ID,
        bootstrap_path=bootstrap_path,
        bootstrap_sha256=tooling.server.exact_parity_slurm.sha256_file(bootstrap_path),
        source_repository=source_repository,
        expected_git_commit="a" * 40,
        expected_cpu_count=CPU_COUNT,
        expected_memory_bytes=MEMORY_BYTES,
        expected_gpu_count=GPU_COUNT,
        require_kernel_enforcement=False,
    )


def job_record(
    request: tooling.server.exact_parity_slurm.SlurmValidationRequest,
    *,
    runtime: str,
    allocated_cpu_count: int | None = None,
) -> str:
    job_cpu_count = allocated_cpu_count or request.expected_cpu_count
    return (
        f"JobId={request.job_id} JobName=exact-parity "
        f"UserId={request.user_name}({request.user_id}) JobState=RUNNING "
        f"RunTime={runtime} NodeList={request.node_name} NumNodes=1 "
        f"NumCPUs={job_cpu_count} NumTasks=1 "
        f"CPUs/Task={request.expected_cpu_count} "
        f"AllocTRES=cpu={job_cpu_count},mem=64G,node=1,billing={job_cpu_count},gres/gpu=1 "
        f"MinMemoryNode=64G Command={request.bootstrap_path} WorkDir=/tmp "
        "TresPerNode=gres/gpu:1\n"
    )


def step_record(
    request: tooling.server.exact_parity_slurm.SlurmValidationRequest,
    *,
    runtime: str,
) -> str:
    return (
        f"StepId={request.job_id}.{request.step_id} UserId={request.user_id} "
        f"StartTime=2026-07-24T00:00:00 RunTime={runtime} State=RUNNING "
        f"NodeList={request.node_name} Nodes=1 CPUs={request.expected_cpu_count} "
        f"Tasks=1 Name=exact-parity "
        f"TRES=cpu={request.expected_cpu_count},mem=64G,node=1,gres/gpu=1 "
        "TresPerNode=gres/gpu:1\n"
    )


def observed_state(
    request: tooling.server.exact_parity_slurm.SlurmValidationRequest,
) -> tooling.server.exact_parity_slurm.SlurmObservedState:
    bootstrap_identity = tooling.server.exact_parity_slurm.bootstrap_file_identity(request.bootstrap_path)
    process_command_line = tooling.server.exact_parity_slurm.expected_bootstrap_command_line(request)
    return tooling.server.exact_parity_slurm.SlurmObservedState(
        first_job_record=job_record(request, runtime="00:00:01"),
        first_step_record=step_record(request, runtime="00:00:01"),
        listpids_record=(
            "PID      JOBID    STEPID   LOCALID GLOBALID\n"
            f"4000     {request.job_id}    {request.step_id}        -       -\n"
            f"{request.process_id}     {request.job_id}    {request.step_id}        0       0\n"
        ),
        second_job_record=job_record(request, runtime="00:00:02"),
        second_step_record=step_record(request, runtime="00:00:02"),
        slurm_configuration=(
            "ClusterName = abraxas\nProctrackType = proctrack/cgroup\nTaskPlugin = task/cgroup,task/affinity\n"
        ),
        cgroup_configuration=(
            "CgroupMountpoint=/sys/fs/cgroup\nConstrainCores=no\nConstrainRAMSpace=no\nConstrainDevices=no\n"
        ),
        environment_values={
            "SLURM_CPUS_PER_TASK": str(request.expected_cpu_count),
            "SLURM_JOB_ID": request.job_id,
            "SLURM_JOB_NODELIST": request.node_name,
            "SLURM_JOB_UID": str(request.user_id),
            "SLURM_JOB_USER": request.user_name,
            "SLURM_STEP_ID": request.step_id,
            "SLURM_STEP_NODELIST": request.node_name,
            "SLURMD_NODENAME": request.node_name,
        },
        observer_process_id=5000,
        observer_parent_process_id=request.process_id,
        host_pid_namespace_inode=PID_NAMESPACE_INODE,
        host_cgroup_namespace_inode=CGROUP_NAMESPACE_INODE,
        observer_pid_namespace_inode=PID_NAMESPACE_INODE,
        observer_cgroup_namespace_inode=CGROUP_NAMESPACE_INODE,
        process_pid_namespace_inode=PID_NAMESPACE_INODE,
        process_cgroup_namespace_inode=CGROUP_NAMESPACE_INODE,
        process_namespace_pid_values=(request.process_id,),
        host_boot_id=HOST_BOOT_ID,
        process_start_time_ticks_before=9001,
        process_start_time_ticks_after=9001,
        process_cgroup_record=(
            f"0::/system.slice/slurmstepd.scope/job_{request.job_id}/step_{request.step_id}/user/task_0\n"
        ),
        process_mountinfo_record=(
            "8448 8446 0:30 / /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime "
            "master:9 - cgroup2 cgroup2 rw,nsdelegate,memory_recursiveprot\n"
        ),
        process_command_line_before=process_command_line,
        process_command_line_after=process_command_line,
        process_executable_path_before="/usr/bin/bash",
        process_executable_path_after="/usr/bin/bash",
        bootstrap_sha256_before=request.bootstrap_sha256,
        bootstrap_sha256_after=request.bootstrap_sha256,
        bootstrap_file_identity_before=bootstrap_identity,
        bootstrap_file_identity_after=bootstrap_identity,
        cgroup_directory_exists=True,
    )


def test_slurm_attestation_proves_entitlement_without_claiming_kernel_enforcement(
    tmp_path: Path,
) -> None:
    request = validation_request(tmp_path)
    observations = observed_state(request)

    attestation = tooling.server.exact_parity_slurm.validate_slurm_observations(request, observations)
    canonical_json = tooling.server.exact_parity_slurm.canonical_slurm_process_attestation(attestation)
    parsed_payload = json.loads(canonical_json)

    assert attestation.scheduler_entitlement_proven is True
    assert attestation.kernel_enforcement_proven is False
    assert attestation.constrain_cores is False
    assert attestation.constrain_ram_space is False
    assert attestation.constrain_devices is False
    assert attestation.host_process_id == request.process_id
    assert attestation.host_process_start_time_ticks == observations.process_start_time_ticks_before
    assert attestation.job_cpu_count == request.expected_cpu_count
    assert attestation.job_task_count == 1
    assert attestation.step_gpu_count == request.expected_gpu_count
    assert canonical_json.endswith(b"\n")
    assert canonical_json.count(b"\n") == 1
    assert canonical_json == (json.dumps(parsed_payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    assert tooling.server.exact_parity_slurm.parse_slurm_process_attestation(parsed_payload) == attestation
    assert attestation.first_job_record_sha256 != attestation.second_job_record_sha256
    assert attestation.first_step_record_sha256 != attestation.second_step_record_sha256


def test_slurm_attestation_accepts_scheduler_rounded_job_cpu_allocation(
    tmp_path: Path,
) -> None:
    request = validation_request(tmp_path)
    observations = dataclasses.replace(
        observed_state(request),
        first_job_record=job_record(request, runtime="00:00:01", allocated_cpu_count=16),
        second_job_record=job_record(request, runtime="00:00:02", allocated_cpu_count=16),
    )

    attestation = tooling.server.exact_parity_slurm.validate_slurm_observations(request, observations)

    assert attestation.job_cpu_count == 16
    assert attestation.step_cpu_count == request.expected_cpu_count


def test_slurm_attestation_accepts_site_tres_records_without_gpu_field(
    tmp_path: Path,
) -> None:
    request = validation_request(tmp_path)
    observations = observed_state(request)
    observations = dataclasses.replace(
        observations,
        first_job_record=observations.first_job_record.replace(",gres/gpu=1", ""),
        second_job_record=observations.second_job_record.replace(",gres/gpu=1", ""),
        first_step_record=observations.first_step_record.replace(",gres/gpu=1", ""),
        second_step_record=observations.second_step_record.replace(",gres/gpu=1", ""),
    )

    attestation = tooling.server.exact_parity_slurm.validate_slurm_observations(request, observations)

    assert attestation.job_gpu_count == GPU_COUNT
    assert attestation.step_gpu_count == GPU_COUNT


@pytest.mark.parametrize(
    "replacement_fields",
    [
        {"expected_cpu_count": CPU_COUNT - 1},
        {"expected_memory_bytes": MEMORY_BYTES // 2},
        {"expected_gpu_count": GPU_COUNT + 1},
    ],
)
def test_slurm_request_rejects_noncanonical_resource_expectations(
    tmp_path: Path,
    replacement_fields: dict[str, int],
) -> None:
    request = dataclasses.replace(validation_request(tmp_path), **replacement_fields)

    with pytest.raises(
        tooling.server.exact_parity_slurm.SlurmValidationError,
        match="exact qualification resource entitlement",
    ):
        tooling.server.exact_parity_slurm.validate_slurm_request(request)


def test_slurm_attestation_rejects_forged_environment(tmp_path: Path) -> None:
    request = validation_request(tmp_path)
    observations = observed_state(request)
    forged_environment = dict(observations.environment_values)
    forged_environment["SLURM_JOB_ID"] = "99999"

    with pytest.raises(
        tooling.server.exact_parity_slurm.SlurmValidationError,
        match="environment disagrees",
    ):
        tooling.server.exact_parity_slurm.validate_slurm_observations(
            request,
            dataclasses.replace(observations, environment_values=forged_environment),
        )


@pytest.mark.parametrize(
    ("listpids_record", "expected_message"),
    [
        (
            f"PID JOBID STEPID LOCALID GLOBALID\n4000 {JOB_ID} {STEP_ID} 0 0\n",
            "exactly once",
        ),
        (
            f"PID JOBID STEPID LOCALID GLOBALID\n{PROCESS_ID} 99999 {STEP_ID} 0 0\n",
            "another job or step",
        ),
        (
            f"PID JOBID STEPID LOCALID GLOBALID\n{PROCESS_ID} - {STEP_ID} 0 0\n",
            "malformed",
        ),
        (
            (
                "PID JOBID STEPID LOCALID GLOBALID\n"
                f"{PROCESS_ID} {JOB_ID} {STEP_ID} 0 0\n"
                f"{PROCESS_ID} {JOB_ID} {STEP_ID} 0 0\n"
            ),
            "duplicates PID",
        ),
    ],
)
def test_slurm_attestation_rejects_missing_wrong_or_duplicate_pid(
    tmp_path: Path,
    listpids_record: str,
    expected_message: str,
) -> None:
    request = validation_request(tmp_path)
    observations = dataclasses.replace(
        observed_state(request),
        listpids_record=listpids_record,
    )

    with pytest.raises(
        tooling.server.exact_parity_slurm.SlurmValidationError,
        match=expected_message,
    ):
        tooling.server.exact_parity_slurm.validate_slurm_observations(request, observations)


@pytest.mark.parametrize(
    "cgroup_record",
    [
        f"0::/slurm/job_{JOB_ID}suffix/step_{STEP_ID}/task_0\n",
        f"0::/slurm/job_{JOB_ID}/step_{STEP_ID}suffix/task_0\n",
        f"0::/slurm/job_9{JOB_ID}/step_{STEP_ID}/task_0\n",
        f"0::/slurm/step_{STEP_ID}/job_{JOB_ID}/task_0\n",
        f"0::/slurm/job_{JOB_ID}/job_{JOB_ID}/step_{STEP_ID}/task_0\n",
    ],
)
def test_slurm_attestation_rejects_cgroup_substrings_and_wrong_structure(
    tmp_path: Path,
    cgroup_record: str,
) -> None:
    request = validation_request(tmp_path)
    observations = dataclasses.replace(
        observed_state(request),
        process_cgroup_record=cgroup_record,
    )

    with pytest.raises(tooling.server.exact_parity_slurm.SlurmValidationError, match="cgroup"):
        tooling.server.exact_parity_slurm.validate_slurm_observations(request, observations)


@pytest.mark.parametrize(
    "replacement_fields",
    [
        {"process_cgroup_namespace_inode": CGROUP_NAMESPACE_INODE + 1},
        {"process_pid_namespace_inode": PID_NAMESPACE_INODE + 1},
        {"host_pid_namespace_inode": PID_NAMESPACE_INODE + 1},
        {"process_namespace_pid_values": (99, PROCESS_ID)},
    ],
)
def test_slurm_attestation_rejects_namespace_mismatch_or_hidden_host_pid(
    tmp_path: Path,
    replacement_fields: dict[str, object],
) -> None:
    request = validation_request(tmp_path)
    observations = dataclasses.replace(observed_state(request), **replacement_fields)

    with pytest.raises(
        tooling.server.exact_parity_slurm.SlurmValidationError,
        match=r"namespace|namespace-hidden",
    ):
        tooling.server.exact_parity_slurm.validate_slurm_observations(request, observations)


@pytest.mark.parametrize(
    "first_job_record",
    [
        "",
        "JobId=12345\nJobId=12345\n",
        "JobId=12345 JobId=12345 JobState=RUNNING\n",
    ],
)
def test_slurm_attestation_rejects_empty_or_ambiguous_scheduler_records(
    tmp_path: Path,
    first_job_record: str,
) -> None:
    request = validation_request(tmp_path)
    observations = dataclasses.replace(
        observed_state(request),
        first_job_record=first_job_record,
    )

    with pytest.raises(tooling.server.exact_parity_slurm.SlurmValidationError):
        tooling.server.exact_parity_slurm.validate_slurm_observations(request, observations)


@pytest.mark.parametrize(
    ("record_name", "old_text", "new_text", "expected_message"),
    [
        (
            "first_job_record",
            "Command=",
            "Command=/tmp/wrong-bootstrap ",
            "command",
        ),
        (
            "first_job_record",
            "NumCPUs=8",
            "NumCPUs=7",
            "resource allocation|per-node",
        ),
        (
            "first_step_record",
            "mem=64G",
            "mem=32G",
            "resource entitlement",
        ),
        (
            "first_step_record",
            "TresPerNode=gres/gpu:1",
            "TresPerNode=gres/gpu:2",
            "resource entitlement",
        ),
        (
            "first_job_record",
            "billing=8,gres/gpu=1",
            "billing=8,gres/gpu=0",
            "per-node fields disagree",
        ),
        (
            "first_step_record",
            "node=1,gres/gpu=1",
            "node=1,gres/gpu=0",
            "fields disagree",
        ),
    ],
)
def test_slurm_attestation_rejects_command_and_resource_mismatch(
    tmp_path: Path,
    record_name: str,
    old_text: str,
    new_text: str,
    expected_message: str,
) -> None:
    request = validation_request(tmp_path)
    observations = observed_state(request)
    original_record = typing.cast("str", getattr(observations, record_name))
    if old_text == "Command=":
        original_command = f"Command={request.bootstrap_path}"
        changed_record = original_record.replace(original_command, new_text.rstrip())
    else:
        changed_record = original_record.replace(old_text, new_text)
    observations = dataclasses.replace(observations, **{record_name: changed_record})

    with pytest.raises(
        tooling.server.exact_parity_slurm.SlurmValidationError,
        match=expected_message,
    ):
        tooling.server.exact_parity_slurm.validate_slurm_observations(request, observations)


@pytest.mark.parametrize(
    "replacement_fields",
    [
        {"process_start_time_ticks_after": 9002},
        {"process_command_line_after": b"/usr/bin/bash\0/tmp/replaced\0"},
        {"bootstrap_sha256_after": "f" * 64},
        {"second_job_record": ""},
        {"second_step_record": ""},
    ],
)
def test_slurm_attestation_rejects_toctou_changes(
    tmp_path: Path,
    replacement_fields: dict[str, object],
) -> None:
    request = validation_request(tmp_path)
    observations = dataclasses.replace(observed_state(request), **replacement_fields)

    with pytest.raises(tooling.server.exact_parity_slurm.SlurmValidationError):
        tooling.server.exact_parity_slurm.validate_slurm_observations(request, observations)


def test_slurm_attestation_rejects_unsupported_strict_enforcement(tmp_path: Path) -> None:
    request = dataclasses.replace(
        validation_request(tmp_path),
        require_kernel_enforcement=True,
    )

    with pytest.raises(
        tooling.server.exact_parity_slurm.SlurmValidationError,
        match="Strict kernel enforcement is unsupported",
    ):
        tooling.server.exact_parity_slurm.validate_slurm_observations(
            request,
            observed_state(request),
        )


@pytest.mark.parametrize("field_name", ["unknown", "proc-root", "scontrol-path", "slurm-conf"])
def test_slurm_cli_rejects_routing_overrides(
    tmp_path: Path,
    field_name: str,
) -> None:
    request = validation_request(tmp_path)
    arguments = [
        "--cluster-name",
        request.cluster_name,
        "--node-name",
        request.node_name,
        "--job-id",
        request.job_id,
        "--step-id",
        request.step_id,
        "--user-name",
        request.user_name,
        "--user-id",
        str(request.user_id),
        "--process-id",
        str(request.process_id),
        "--bootstrap-path",
        str(request.bootstrap_path),
        "--bootstrap-sha256",
        request.bootstrap_sha256,
        "--source-repository",
        str(request.source_repository),
        "--expected-git-commit",
        request.expected_git_commit,
        "--expected-cpu-count",
        str(request.expected_cpu_count),
        "--expected-memory-bytes",
        str(request.expected_memory_bytes),
        "--expected-gpu-count",
        str(request.expected_gpu_count),
        f"--{field_name}",
        "/tmp/untrusted",
    ]

    with pytest.raises(SystemExit):
        tooling.server.exact_parity_slurm.build_argument_parser().parse_args(arguments)


def test_scontrol_queries_drop_environment_routing_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_environment: dict[str, str] = {}

    def fake_run(
        arguments: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, timeout
        assert arguments[0] == "/usr/bin/scontrol"
        observed_environment.update(env)
        return subprocess.CompletedProcess(arguments, 0, stdout="record\n", stderr="")

    monkeypatch.setenv("SLURM_CONF", "/tmp/untrusted")
    monkeypatch.setenv("SLURM_CLUSTERS", "untrusted")
    monkeypatch.setenv("SCONTROL_PATH", "/tmp/untrusted")
    monkeypatch.setattr(tooling.server.exact_parity_slurm.subprocess, "run", fake_run)

    assert tooling.server.exact_parity_slurm.run_scontrol("show", "config") == "record\n"
    assert observed_environment == {
        "HOME": "/tmp",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_slurm_attestation_schema_rejects_missing_and_unknown_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    request = validation_request(tmp_path)
    attestation = tooling.server.exact_parity_slurm.validate_slurm_observations(
        request,
        observed_state(request),
    )
    payload = tooling.server.exact_parity_slurm.slurm_process_attestation_payload(attestation)
    if mutation == "missing":
        payload.pop("job_id")
    else:
        payload["unexpected"] = "value"

    with pytest.raises(
        tooling.server.exact_parity_slurm.SlurmValidationError,
        match="field mismatch",
    ):
        tooling.server.exact_parity_slurm.parse_slurm_process_attestation(payload)


@pytest.mark.parametrize(
    ("field_name", "field_value", "expected_message"),
    [
        ("cgroup_v2_path", f"/slurm/job_{JOB_ID}/step_999/task_0", "cgroup"),
        ("host_boot_id", "not-a-uuid", "boot ID"),
        ("job_cpu_count", CPU_COUNT - 1, "resource"),
        ("job_task_count", 2, "resource"),
        ("scheduler_entitlement_proven", False, "does not prove"),
    ],
)
def test_slurm_attestation_schema_rejects_inconsistent_claims(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    expected_message: str,
) -> None:
    request = validation_request(tmp_path)
    attestation = tooling.server.exact_parity_slurm.validate_slurm_observations(
        request,
        observed_state(request),
    )
    payload = tooling.server.exact_parity_slurm.slurm_process_attestation_payload(attestation)
    payload[field_name] = field_value

    with pytest.raises(
        tooling.server.exact_parity_slurm.SlurmValidationError,
        match=expected_message,
    ):
        tooling.server.exact_parity_slurm.parse_slurm_process_attestation(payload)


@pytest.mark.parametrize(
    "replacement_fields",
    [
        {"job_cpu_count": CPU_COUNT - 1, "step_cpu_count": CPU_COUNT - 1},
        {"job_memory_bytes": MEMORY_BYTES // 2, "step_memory_bytes": MEMORY_BYTES // 2},
        {"job_gpu_count": GPU_COUNT + 1, "step_gpu_count": GPU_COUNT + 1},
    ],
)
def test_slurm_attestation_schema_rejects_self_consistent_noncanonical_resources(
    tmp_path: Path,
    replacement_fields: dict[str, int],
) -> None:
    request = validation_request(tmp_path)
    attestation = tooling.server.exact_parity_slurm.validate_slurm_observations(
        request,
        observed_state(request),
    )
    payload = tooling.server.exact_parity_slurm.slurm_process_attestation_payload(attestation)
    payload.update(replacement_fields)

    with pytest.raises(
        tooling.server.exact_parity_slurm.SlurmValidationError,
        match="exact qualification resource entitlement",
    ):
        tooling.server.exact_parity_slurm.parse_slurm_process_attestation(payload)
