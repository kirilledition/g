"""Fail-closed Slurm process attestation for exact parity qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import typing
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 0
QUALIFICATION_CPU_COUNT = 8
QUALIFICATION_MEMORY_BYTES = 64 * 1024**3
QUALIFICATION_GPU_COUNT = 1
SYSTEM_BASH_PATH = Path("/usr/bin/bash")
SYSTEM_SCONTROL_PATH = Path("/usr/bin/scontrol")
PROC_ROOT = Path("/proc")
CGROUP_MOUNT_POINT = Path("/sys/fs/cgroup")
CGROUP_CONFIGURATION_PATH = Path("/etc/slurm/cgroup.conf")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
NUMERIC_IDENTIFIER_PATTERN = re.compile(r"^[0-9]+$")
POSITIVE_IDENTIFIER_PATTERN = re.compile(r"^[1-9][0-9]*$")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
SCHEDULER_FIELD_PATTERN = re.compile(r"(?:^| )([A-Za-z][A-Za-z0-9_/:]*)=")
CGROUP_JOB_COMPONENT_PATTERN = re.compile(r"^job_([0-9]+)$")
CGROUP_STEP_COMPONENT_PATTERN = re.compile(r"^step_([0-9]+)$")
MEMORY_SIZE_PATTERN = re.compile(r"^([0-9]+)([KMGTP]?)$")
SLURM_ENVIRONMENT_NAMES = (
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_UID",
    "SLURM_JOB_USER",
    "SLURM_STEP_ID",
    "SLURM_STEP_NODELIST",
    "SLURMD_NODENAME",
)
ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "cluster_name",
        "node_name",
        "job_id",
        "step_id",
        "user_name",
        "user_id",
        "host_boot_id",
        "host_process_id",
        "host_process_start_time_ticks",
        "host_process_pid_namespace_inode",
        "host_process_cgroup_namespace_inode",
        "cgroup_v2_path",
        "job_node_count",
        "job_cpu_count",
        "job_memory_bytes",
        "job_gpu_count",
        "job_task_count",
        "step_node_count",
        "step_cpu_count",
        "step_memory_bytes",
        "step_gpu_count",
        "step_task_count",
        "first_job_record_sha256",
        "second_job_record_sha256",
        "first_step_record_sha256",
        "second_step_record_sha256",
        "listpids_sha256",
        "bootstrap_path",
        "bootstrap_sha256",
        "job_command",
        "bootstrap_process_command_sha256",
        "constrain_cores",
        "constrain_ram_space",
        "constrain_devices",
        "scheduler_entitlement_proven",
        "kernel_enforcement_proven",
    }
)


class SlurmValidationError(ValueError):
    """Raised when live Slurm observations do not prove the requested step."""


@dataclass(frozen=True)
class SlurmValidationRequest:
    """Trusted expectations that live scheduler observations must prove."""

    schema_version: int
    cluster_name: str
    node_name: str
    job_id: str
    step_id: str
    user_name: str
    user_id: int
    process_id: int
    bootstrap_path: Path
    bootstrap_sha256: str
    source_repository: Path
    expected_git_commit: str
    expected_cpu_count: int
    expected_memory_bytes: int
    expected_gpu_count: int
    require_kernel_enforcement: bool


@dataclass(frozen=True)
class BootstrapFileIdentity:
    """Stable filesystem identity of the executing bootstrap."""

    device: int
    inode: int
    size_bytes: int
    modification_time_nanoseconds: int


@dataclass(frozen=True)
class SlurmObservedState:
    """Raw controller, local process, namespace, and cgroup observations."""

    first_job_record: str
    first_step_record: str
    listpids_record: str
    second_job_record: str
    second_step_record: str
    slurm_configuration: str
    cgroup_configuration: str
    environment_values: dict[str, str]
    observer_process_id: int
    observer_parent_process_id: int
    host_pid_namespace_inode: int
    host_cgroup_namespace_inode: int
    observer_pid_namespace_inode: int
    observer_cgroup_namespace_inode: int
    process_pid_namespace_inode: int
    process_cgroup_namespace_inode: int
    process_namespace_pid_values: tuple[int, ...]
    host_boot_id: str
    process_start_time_ticks_before: int
    process_start_time_ticks_after: int
    process_cgroup_record: str
    process_mountinfo_record: str
    process_command_line_before: bytes
    process_command_line_after: bytes
    process_executable_path_before: str
    process_executable_path_after: str
    bootstrap_sha256_before: str
    bootstrap_sha256_after: str
    bootstrap_file_identity_before: BootstrapFileIdentity
    bootstrap_file_identity_after: BootstrapFileIdentity
    cgroup_directory_exists: bool


@dataclass(frozen=True)
class SchedulerEntitlement:
    """Stable resource fields parsed from one scheduler record."""

    node_count: int
    cpu_count: int
    memory_bytes: int
    gpu_count: int
    task_count: int


@dataclass(frozen=True)
class SlurmProcessAttestation:
    """Canonical proof that one host process belongs to one numeric Slurm step."""

    schema_version: int
    cluster_name: str
    node_name: str
    job_id: str
    step_id: str
    user_name: str
    user_id: int
    host_boot_id: str
    host_process_id: int
    host_process_start_time_ticks: int
    host_process_pid_namespace_inode: int
    host_process_cgroup_namespace_inode: int
    cgroup_v2_path: str
    job_node_count: int
    job_cpu_count: int
    job_memory_bytes: int
    job_gpu_count: int
    job_task_count: int
    step_node_count: int
    step_cpu_count: int
    step_memory_bytes: int
    step_gpu_count: int
    step_task_count: int
    first_job_record_sha256: str
    second_job_record_sha256: str
    first_step_record_sha256: str
    second_step_record_sha256: str
    listpids_sha256: str
    bootstrap_path: str
    bootstrap_sha256: str
    job_command: str
    bootstrap_process_command_sha256: str
    constrain_cores: bool
    constrain_ram_space: bool
    constrain_devices: bool
    scheduler_entitlement_proven: bool
    kernel_enforcement_proven: bool


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 of bytes."""
    return hashlib.sha256(content).hexdigest()


def sha256_file(file_path: Path) -> str:
    """Return the lowercase SHA-256 of one regular file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_file_identity(file_path: Path) -> BootstrapFileIdentity:
    """Return stable metadata for one canonical regular bootstrap."""
    path_status = file_path.lstat()
    if not stat.S_ISREG(path_status.st_mode) or file_path.is_symlink():
        raise SlurmValidationError(f"Bootstrap is not a nonsymlink regular file: {file_path}")
    if file_path.resolve(strict=True) != file_path:
        raise SlurmValidationError(f"Bootstrap path is not canonical: {file_path}")
    return BootstrapFileIdentity(
        device=path_status.st_dev,
        inode=path_status.st_ino,
        size_bytes=path_status.st_size,
        modification_time_nanoseconds=path_status.st_mtime_ns,
    )


def run_scontrol(*arguments: str) -> str:
    """Run one fixed-path Slurm control query in a routing-free environment."""
    completed_process = subprocess.run(
        [str(SYSTEM_SCONTROL_PATH), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            "HOME": "/tmp",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    if completed_process.returncode != 0:
        standard_error = completed_process.stderr.strip()
        raise SlurmValidationError(
            f"scontrol {' '.join(arguments)} failed with status "
            f"{completed_process.returncode}: {standard_error or 'no diagnostic'}"
        )
    return completed_process.stdout


def parse_process_start_time_ticks(process_statistic: str) -> int:
    """Parse Linux field 22 without trusting spaces or parentheses in comm."""
    command_end = process_statistic.rfind(")")
    if command_end < 0:
        raise SlurmValidationError("Process stat record has no closing command delimiter")
    remaining_fields = process_statistic[command_end + 1 :].split()
    start_time_index = 22 - 3
    if len(remaining_fields) <= start_time_index:
        raise SlurmValidationError("Process stat record is truncated before starttime")
    start_time_text = remaining_fields[start_time_index]
    if NUMERIC_IDENTIFIER_PATTERN.fullmatch(start_time_text) is None:
        raise SlurmValidationError("Process stat starttime is not numeric")
    start_time_ticks = int(start_time_text)
    if start_time_ticks <= 0:
        raise SlurmValidationError("Process stat starttime must be positive")
    return start_time_ticks


def parse_process_namespace_pid_values(process_status: str) -> tuple[int, ...]:
    """Parse the sole Linux NSpid status row."""
    namespace_rows = [
        line.split(":", maxsplit=1)[1].split() for line in process_status.splitlines() if line.startswith("NSpid:")
    ]
    if len(namespace_rows) != 1 or not namespace_rows[0]:
        raise SlurmValidationError("Process status must contain exactly one nonempty NSpid row")
    namespace_values = namespace_rows[0]
    if any(NUMERIC_IDENTIFIER_PATTERN.fullmatch(value) is None for value in namespace_values):
        raise SlurmValidationError("Process NSpid row contains a nonnumeric identifier")
    return tuple(int(value) for value in namespace_values)


def parse_process_environment(process_environment: bytes) -> dict[str, str]:
    """Parse a NUL-delimited process environment without duplicate names."""
    if not process_environment.endswith(b"\0"):
        raise SlurmValidationError("Process environment is not NUL terminated")
    values: dict[str, str] = {}
    for raw_entry in process_environment[:-1].split(b"\0"):
        if b"=" not in raw_entry:
            raise SlurmValidationError("Process environment contains an entry without '='")
        raw_name, raw_value = raw_entry.split(b"=", maxsplit=1)
        name = raw_name.decode("utf-8")
        value = raw_value.decode("utf-8")
        if name in values:
            raise SlurmValidationError(f"Process environment duplicates {name}")
        values[name] = value
    return values


def parse_cgroup_v2_path(cgroup_record: str) -> PurePosixPath:
    """Parse one unified cgroup-v2 membership record."""
    lines = [line for line in cgroup_record.splitlines() if line]
    if len(lines) != 1:
        raise SlurmValidationError("Process cgroup record must contain exactly one hierarchy")
    hierarchy_fields = lines[0].split(":", maxsplit=2)
    if len(hierarchy_fields) != 3 or hierarchy_fields[:2] != ["0", ""]:
        raise SlurmValidationError("Process cgroup record is not a unified cgroup-v2 hierarchy")
    path_text = hierarchy_fields[2]
    if not path_text.startswith("/") or "//" in path_text or "/./" in f"{path_text}/" or "/../" in f"{path_text}/":
        raise SlurmValidationError("Process cgroup-v2 path is not structurally canonical")
    cgroup_path = PurePosixPath(path_text)
    if cgroup_path.as_posix() != path_text:
        raise SlurmValidationError("Process cgroup-v2 path is not lexically canonical")
    return cgroup_path


def parse_namespace_inode(namespace_path: Path) -> int:
    """Return the inode that identifies one Linux namespace."""
    namespace_inode = namespace_path.stat().st_ino
    if namespace_inode <= 0:
        raise SlurmValidationError(f"Namespace inode is invalid: {namespace_path}")
    return namespace_inode


def collect_live_observations(request: SlurmValidationRequest) -> SlurmObservedState:
    """Collect controller snapshots around local PID and cgroup evidence."""
    validate_slurm_request(request)
    process_root = PROC_ROOT / str(request.process_id)
    process_status_path = process_root / "status"
    process_statistic_path = process_root / "stat"
    process_environment_path = process_root / "environ"
    process_cgroup_path = process_root / "cgroup"
    process_mountinfo_path = process_root / "mountinfo"
    process_command_line_path = process_root / "cmdline"
    process_executable_link = process_root / "exe"

    first_job_record = run_scontrol("show", "job", request.job_id, "--oneliner")
    first_step_record = run_scontrol("show", "step", f"{request.job_id}.{request.step_id}", "--oneliner")
    listpids_record = run_scontrol("listpids", f"{request.job_id}.{request.step_id}")
    slurm_configuration = run_scontrol("show", "config")

    process_status = process_status_path.read_text(encoding="utf-8")
    process_environment = parse_process_environment(process_environment_path.read_bytes())
    environment_values = {
        environment_name: process_environment.get(environment_name, "") for environment_name in SLURM_ENVIRONMENT_NAMES
    }
    process_start_time_ticks_before = parse_process_start_time_ticks(process_statistic_path.read_text(encoding="utf-8"))
    process_command_line_before = process_command_line_path.read_bytes()
    process_executable_path_before = str(process_executable_link.resolve(strict=True))
    process_cgroup_record = process_cgroup_path.read_text(encoding="utf-8")
    process_mountinfo_record = process_mountinfo_path.read_text(encoding="utf-8")
    cgroup_v2_path = parse_cgroup_v2_path(process_cgroup_record)
    resolved_cgroup_directory = (CGROUP_MOUNT_POINT / cgroup_v2_path.relative_to("/")).resolve(strict=False)
    cgroup_directory_exists = (
        resolved_cgroup_directory.is_dir()
        and not resolved_cgroup_directory.is_symlink()
        and resolved_cgroup_directory.is_relative_to(CGROUP_MOUNT_POINT)
    )
    bootstrap_sha256_before = sha256_file(request.bootstrap_path)
    bootstrap_file_identity_before = bootstrap_file_identity(request.bootstrap_path)

    second_job_record = run_scontrol("show", "job", request.job_id, "--oneliner")
    second_step_record = run_scontrol("show", "step", f"{request.job_id}.{request.step_id}", "--oneliner")
    process_start_time_ticks_after = parse_process_start_time_ticks(process_statistic_path.read_text(encoding="utf-8"))
    process_command_line_after = process_command_line_path.read_bytes()
    process_executable_path_after = str(process_executable_link.resolve(strict=True))
    bootstrap_sha256_after = sha256_file(request.bootstrap_path)
    bootstrap_file_identity_after = bootstrap_file_identity(request.bootstrap_path)

    return SlurmObservedState(
        first_job_record=first_job_record,
        first_step_record=first_step_record,
        listpids_record=listpids_record,
        second_job_record=second_job_record,
        second_step_record=second_step_record,
        slurm_configuration=slurm_configuration,
        cgroup_configuration=CGROUP_CONFIGURATION_PATH.read_text(encoding="utf-8"),
        environment_values=environment_values,
        observer_process_id=os.getpid(),
        observer_parent_process_id=os.getppid(),
        host_pid_namespace_inode=parse_namespace_inode(PROC_ROOT / "1" / "ns" / "pid"),
        host_cgroup_namespace_inode=parse_namespace_inode(PROC_ROOT / "1" / "ns" / "cgroup"),
        observer_pid_namespace_inode=parse_namespace_inode(PROC_ROOT / "self" / "ns" / "pid"),
        observer_cgroup_namespace_inode=parse_namespace_inode(PROC_ROOT / "self" / "ns" / "cgroup"),
        process_pid_namespace_inode=parse_namespace_inode(process_root / "ns" / "pid"),
        process_cgroup_namespace_inode=parse_namespace_inode(process_root / "ns" / "cgroup"),
        process_namespace_pid_values=parse_process_namespace_pid_values(process_status),
        host_boot_id=(PROC_ROOT / "sys" / "kernel" / "random" / "boot_id").read_text(encoding="utf-8").strip(),
        process_start_time_ticks_before=process_start_time_ticks_before,
        process_start_time_ticks_after=process_start_time_ticks_after,
        process_cgroup_record=process_cgroup_record,
        process_mountinfo_record=process_mountinfo_record,
        process_command_line_before=process_command_line_before,
        process_command_line_after=process_command_line_after,
        process_executable_path_before=process_executable_path_before,
        process_executable_path_after=process_executable_path_after,
        bootstrap_sha256_before=bootstrap_sha256_before,
        bootstrap_sha256_after=bootstrap_sha256_after,
        bootstrap_file_identity_before=bootstrap_file_identity_before,
        bootstrap_file_identity_after=bootstrap_file_identity_after,
        cgroup_directory_exists=cgroup_directory_exists,
    )


def validate_slurm_request(request: SlurmValidationRequest) -> None:
    """Reject malformed or noncanonical trusted expectations."""
    if request.schema_version != SCHEMA_VERSION:
        raise SlurmValidationError(f"Unsupported Slurm attestation schema: {request.schema_version}")
    if NAME_PATTERN.fullmatch(request.cluster_name) is None:
        raise SlurmValidationError("Expected Slurm cluster name is malformed")
    if NAME_PATTERN.fullmatch(request.node_name) is None:
        raise SlurmValidationError("Expected Slurm node name is malformed")
    if NUMERIC_IDENTIFIER_PATTERN.fullmatch(request.job_id) is None:
        raise SlurmValidationError("Expected Slurm job ID is not numeric")
    if NUMERIC_IDENTIFIER_PATTERN.fullmatch(request.step_id) is None:
        raise SlurmValidationError("Expected Slurm step ID is not numeric")
    if NAME_PATTERN.fullmatch(request.user_name) is None:
        raise SlurmValidationError("Expected Slurm user name is malformed")
    if request.user_id < 0 or request.process_id <= 1:
        raise SlurmValidationError("Expected user and process identifiers must be host numeric values")
    if not request.bootstrap_path.is_absolute() or not request.source_repository.is_absolute():
        raise SlurmValidationError("Bootstrap and source repository paths must be absolute")
    if request.bootstrap_path != request.bootstrap_path.resolve(strict=True):
        raise SlurmValidationError("Bootstrap path must be canonical")
    if request.source_repository != request.source_repository.resolve(strict=True):
        raise SlurmValidationError("Source repository path must be canonical")
    if SHA256_PATTERN.fullmatch(request.bootstrap_sha256) is None:
        raise SlurmValidationError("Bootstrap SHA-256 is malformed")
    if GIT_COMMIT_PATTERN.fullmatch(request.expected_git_commit) is None:
        raise SlurmValidationError("Expected Git commit is malformed")
    if request.expected_cpu_count <= 0 or request.expected_memory_bytes <= 0 or request.expected_gpu_count <= 0:
        raise SlurmValidationError("Expected CPU, memory, and GPU entitlements must be positive")
    if (
        request.expected_cpu_count != QUALIFICATION_CPU_COUNT
        or request.expected_memory_bytes != QUALIFICATION_MEMORY_BYTES
        or request.expected_gpu_count != QUALIFICATION_GPU_COUNT
    ):
        raise SlurmValidationError("Slurm request differs from the exact qualification resource entitlement")


def parse_scheduler_record(record: str, *, label: str) -> dict[str, str]:
    """Parse exactly one oneliner scheduler record with unique fields."""
    lines = [line for line in record.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SlurmValidationError(f"{label} must contain exactly one nonempty record")
    line = lines[0].strip()
    matches = list(SCHEDULER_FIELD_PATTERN.finditer(line))
    if not matches or matches[0].start() != 0:
        raise SlurmValidationError(f"{label} does not start with a scheduler field")
    fields: dict[str, str] = {}
    for match_index, match in enumerate(matches):
        field_name = match.group(1)
        value_start = match.end()
        value_end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(line)
        field_value = line[value_start:value_end].strip()
        if field_name in fields:
            raise SlurmValidationError(f"{label} duplicates field {field_name}")
        fields[field_name] = field_value
    return fields


def required_scheduler_field(fields: dict[str, str], field_name: str, *, label: str) -> str:
    """Return one required scheduler field."""
    try:
        field_value = fields[field_name]
    except KeyError as error:
        raise SlurmValidationError(f"{label} is missing field {field_name}") from error
    if not field_value:
        raise SlurmValidationError(f"{label} has an empty required field {field_name}")
    return field_value


def parse_positive_integer(value: str, *, label: str) -> int:
    """Parse one strictly positive decimal integer."""
    if POSITIVE_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise SlurmValidationError(f"{label} must be a positive decimal integer")
    return int(value)


def parse_nonnegative_integer(value: str, *, label: str) -> int:
    """Parse one nonnegative decimal integer."""
    if NUMERIC_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise SlurmValidationError(f"{label} must be a nonnegative decimal integer")
    return int(value)


def parse_memory_bytes(value: str, *, label: str) -> int:
    """Parse one integral Slurm binary memory quantity."""
    match = MEMORY_SIZE_PATTERN.fullmatch(value)
    if match is None:
        raise SlurmValidationError(f"{label} has an unsupported memory quantity: {value}")
    magnitude = int(match.group(1))
    suffix = match.group(2)
    multiplier = 1 if not suffix else 1024 ** ("KMGTP".index(suffix) + 1)
    memory_bytes = magnitude * multiplier
    if memory_bytes <= 0:
        raise SlurmValidationError(f"{label} must be positive")
    return memory_bytes


def parse_tres_fields(value: str, *, label: str) -> dict[str, str]:
    """Parse a comma-separated Slurm TRES assignment list."""
    if value == "(null)":
        raise SlurmValidationError(f"{label} has no resource entitlement")
    fields: dict[str, str] = {}
    for assignment in value.split(","):
        if assignment.count("=") != 1:
            raise SlurmValidationError(f"{label} contains a malformed assignment: {assignment}")
        field_name, field_value = assignment.split("=", maxsplit=1)
        if not field_name or not field_value or field_name in fields:
            raise SlurmValidationError(f"{label} contains an empty or duplicate resource")
        fields[field_name] = field_value
    return fields


def parse_tres_gpu_count(fields: dict[str, str], *, label: str) -> int:
    """Return the generic or typed GPU entitlement without double counting."""
    typed_gpu_fields = {name: value for name, value in fields.items() if name.startswith("gres/gpu:")}
    if "gres/gpu" in fields and typed_gpu_fields:
        raise SlurmValidationError(f"{label} mixes generic and typed GPU entitlements")
    if "gres/gpu" in fields:
        return parse_nonnegative_integer(fields["gres/gpu"], label=f"{label}.gres/gpu")
    typed_gpu_values = [
        parse_nonnegative_integer(value, label=f"{label}.{name}") for name, value in typed_gpu_fields.items()
    ]
    return sum(typed_gpu_values)


def parse_optional_tres_gpu_count(fields: dict[str, str], *, label: str) -> int | None:
    """Return a GPU entitlement when an allocated TRES record exposes one."""
    has_gpu_field = "gres/gpu" in fields or any(name.startswith("gres/gpu:") for name in fields)
    if not has_gpu_field:
        return None
    return parse_tres_gpu_count(fields, label=label)


def parse_tres_per_node_gpu_count(value: str, *, label: str) -> int:
    """Return generic or typed GPU count from Slurm's colon-delimited request form."""
    if value == "(null)":
        raise SlurmValidationError(f"{label} has no per-node resource entitlement")
    fields: dict[str, str] = {}
    for assignment in value.split(","):
        field_name, separator, field_value = assignment.rpartition(":")
        if (
            separator != ":"
            or not field_name
            or not field_value
            or field_name in fields
            or NUMERIC_IDENTIFIER_PATTERN.fullmatch(field_value) is None
        ):
            raise SlurmValidationError(f"{label} contains a malformed or duplicate resource: {assignment}")
        fields[field_name] = field_value
    return parse_tres_gpu_count(fields, label=label)


def parse_user_field(value: str, *, expected_name: str, expected_identifier: int, label: str) -> None:
    """Require either the numeric or canonical name-and-numeric Slurm user form."""
    expected_values = {str(expected_identifier), f"{expected_name}({expected_identifier})"}
    if value not in expected_values:
        raise SlurmValidationError(
            f"{label} identifies another user: expected {sorted(expected_values)}, observed {value}"
        )


def parse_job_entitlement(
    fields: dict[str, str],
    request: SlurmValidationRequest,
    *,
    label: str,
) -> SchedulerEntitlement:
    """Parse and validate one running one-node job allocation."""
    if required_scheduler_field(fields, "JobId", label=label) != request.job_id:
        raise SlurmValidationError(f"{label} has the wrong JobId")
    if required_scheduler_field(fields, "JobState", label=label) != "RUNNING":
        raise SlurmValidationError(f"{label} is not RUNNING")
    if required_scheduler_field(fields, "NodeList", label=label) != request.node_name:
        raise SlurmValidationError(f"{label} has the wrong NodeList")
    parse_user_field(
        required_scheduler_field(fields, "UserId", label=label),
        expected_name=request.user_name,
        expected_identifier=request.user_id,
        label=f"{label}.UserId",
    )
    if required_scheduler_field(fields, "Command", label=label) != str(request.bootstrap_path):
        raise SlurmValidationError(f"{label} command is not the exact bootstrap path")
    node_count = parse_positive_integer(
        required_scheduler_field(fields, "NumNodes", label=label),
        label=f"{label}.NumNodes",
    )
    cpu_count = parse_positive_integer(
        required_scheduler_field(fields, "NumCPUs", label=label),
        label=f"{label}.NumCPUs",
    )
    task_count = parse_positive_integer(
        required_scheduler_field(fields, "NumTasks", label=label),
        label=f"{label}.NumTasks",
    )
    cpu_count_per_task = parse_positive_integer(
        required_scheduler_field(fields, "CPUs/Task", label=label),
        label=f"{label}.CPUs/Task",
    )
    memory_bytes = parse_memory_bytes(
        required_scheduler_field(fields, "MinMemoryNode", label=label),
        label=f"{label}.MinMemoryNode",
    )
    tres_fields = parse_tres_fields(
        required_scheduler_field(fields, "AllocTRES", label=label),
        label=f"{label}.AllocTRES",
    )
    tres_node_count = parse_positive_integer(
        required_scheduler_field(tres_fields, "node", label=f"{label}.AllocTRES"),
        label=f"{label}.AllocTRES.node",
    )
    tres_cpu_count = parse_positive_integer(
        required_scheduler_field(tres_fields, "cpu", label=f"{label}.AllocTRES"),
        label=f"{label}.AllocTRES.cpu",
    )
    tres_memory_bytes = parse_memory_bytes(
        required_scheduler_field(tres_fields, "mem", label=f"{label}.AllocTRES"),
        label=f"{label}.AllocTRES.mem",
    )
    gpu_count = parse_tres_per_node_gpu_count(
        required_scheduler_field(fields, "TresPerNode", label=label),
        label=f"{label}.TresPerNode",
    )
    allocated_gpu_count = parse_optional_tres_gpu_count(
        tres_fields,
        label=f"{label}.AllocTRES",
    )
    entitlement = SchedulerEntitlement(
        node_count=node_count,
        cpu_count=cpu_count,
        memory_bytes=memory_bytes,
        gpu_count=gpu_count,
        task_count=task_count,
    )
    expected_entitlement = SchedulerEntitlement(
        node_count=1,
        cpu_count=cpu_count,
        memory_bytes=request.expected_memory_bytes,
        gpu_count=request.expected_gpu_count,
        task_count=1,
    )
    if entitlement != expected_entitlement or cpu_count < request.expected_cpu_count:
        raise SlurmValidationError(f"{label} resource allocation differs from the trusted request")
    if (
        tres_node_count != node_count
        or tres_cpu_count != cpu_count
        or tres_memory_bytes != memory_bytes
        or (allocated_gpu_count is not None and allocated_gpu_count != gpu_count)
        or cpu_count_per_task != request.expected_cpu_count
    ):
        raise SlurmValidationError(f"{label} per-node fields disagree with AllocTRES")
    return entitlement


def parse_step_entitlement(
    fields: dict[str, str],
    request: SlurmValidationRequest,
    *,
    label: str,
) -> SchedulerEntitlement:
    """Parse and validate one running numeric step entitlement."""
    expected_step_identifier = f"{request.job_id}.{request.step_id}"
    if required_scheduler_field(fields, "StepId", label=label) != expected_step_identifier:
        raise SlurmValidationError(f"{label} has the wrong StepId")
    if required_scheduler_field(fields, "State", label=label) != "RUNNING":
        raise SlurmValidationError(f"{label} is not RUNNING")
    if required_scheduler_field(fields, "NodeList", label=label) != request.node_name:
        raise SlurmValidationError(f"{label} has the wrong NodeList")
    parse_user_field(
        required_scheduler_field(fields, "UserId", label=label),
        expected_name=request.user_name,
        expected_identifier=request.user_id,
        label=f"{label}.UserId",
    )
    node_count = parse_positive_integer(
        required_scheduler_field(fields, "Nodes", label=label),
        label=f"{label}.Nodes",
    )
    cpu_count = parse_positive_integer(
        required_scheduler_field(fields, "CPUs", label=label),
        label=f"{label}.CPUs",
    )
    task_count = parse_positive_integer(
        required_scheduler_field(fields, "Tasks", label=label),
        label=f"{label}.Tasks",
    )
    tres_fields = parse_tres_fields(
        required_scheduler_field(fields, "TRES", label=label),
        label=f"{label}.TRES",
    )
    tres_node_count = parse_positive_integer(
        required_scheduler_field(tres_fields, "node", label=f"{label}.TRES"),
        label=f"{label}.TRES.node",
    )
    tres_cpu_count = parse_positive_integer(
        required_scheduler_field(tres_fields, "cpu", label=f"{label}.TRES"),
        label=f"{label}.TRES.cpu",
    )
    memory_bytes = parse_memory_bytes(
        required_scheduler_field(tres_fields, "mem", label=f"{label}.TRES"),
        label=f"{label}.TRES.mem",
    )
    gpu_count = parse_tres_per_node_gpu_count(
        required_scheduler_field(fields, "TresPerNode", label=label),
        label=f"{label}.TresPerNode",
    )
    allocated_gpu_count = parse_optional_tres_gpu_count(
        tres_fields,
        label=f"{label}.TRES",
    )
    entitlement = SchedulerEntitlement(
        node_count=node_count,
        cpu_count=cpu_count,
        memory_bytes=memory_bytes,
        gpu_count=gpu_count,
        task_count=task_count,
    )
    expected_entitlement = SchedulerEntitlement(
        node_count=1,
        cpu_count=request.expected_cpu_count,
        memory_bytes=request.expected_memory_bytes,
        gpu_count=request.expected_gpu_count,
        task_count=1,
    )
    if entitlement != expected_entitlement:
        raise SlurmValidationError(f"{label} resource entitlement differs from the trusted request")
    if (
        tres_node_count != node_count
        or tres_cpu_count != cpu_count
        or (allocated_gpu_count is not None and allocated_gpu_count != gpu_count)
    ):
        raise SlurmValidationError(f"{label} fields disagree with TRES")
    return entitlement


def validate_listpids(record: str, request: SlurmValidationRequest) -> None:
    """Require the bootstrap PID exactly once in the requested local step."""
    lines = [line for line in record.splitlines() if line.strip()]
    if not lines:
        raise SlurmValidationError("scontrol listpids returned no records")
    if tuple(lines[0].split()) != ("PID", "JOBID", "STEPID", "LOCALID", "GLOBALID"):
        raise SlurmValidationError("scontrol listpids returned an unexpected header")
    observed_process_identifiers: set[int] = set()
    matching_process_count = 0
    for row_index, line in enumerate(lines[1:], start=1):
        fields = line.split()
        if (
            len(fields) != 5
            or any(NUMERIC_IDENTIFIER_PATTERN.fullmatch(field) is None for field in fields[:3])
            or any(field != "-" and NUMERIC_IDENTIFIER_PATTERN.fullmatch(field) is None for field in fields[3:])
        ):
            raise SlurmValidationError(f"scontrol listpids row {row_index} is malformed")
        process_id, job_id, step_id, _local_id, _global_id = fields
        numeric_process_id = int(process_id)
        if numeric_process_id in observed_process_identifiers:
            raise SlurmValidationError(f"scontrol listpids duplicates PID {numeric_process_id}")
        observed_process_identifiers.add(numeric_process_id)
        if job_id != request.job_id or step_id != request.step_id:
            raise SlurmValidationError("scontrol listpids returned a row from another job or step")
        if numeric_process_id == request.process_id:
            matching_process_count += 1
    if matching_process_count != 1:
        raise SlurmValidationError(f"scontrol listpids must contain bootstrap PID {request.process_id} exactly once")


def validate_cgroup_mount(process_mountinfo_record: str) -> None:
    """Require exactly one cgroup2 mount at the fixed host path."""
    matching_mount_count = 0
    for line in process_mountinfo_record.splitlines():
        fields = line.split()
        if "-" not in fields:
            continue
        separator_index = fields.index("-")
        if separator_index + 2 >= len(fields):
            raise SlurmValidationError("Process mountinfo contains a truncated filesystem record")
        if fields[separator_index + 1] == "cgroup2" and fields[4] == str(CGROUP_MOUNT_POINT):
            matching_mount_count += 1
    if matching_mount_count != 1:
        raise SlurmValidationError("Process mount namespace lacks one fixed cgroup-v2 mount")


def validate_cgroup_membership(
    cgroup_path: PurePosixPath,
    *,
    job_id: str,
    step_id: str,
) -> None:
    """Require exact ordered Slurm job and numeric-step path components."""
    components = cgroup_path.parts[1:]
    job_components: list[tuple[int, str]] = []
    step_components: list[tuple[int, str]] = []
    for component_index, component in enumerate(components):
        job_match = CGROUP_JOB_COMPONENT_PATTERN.fullmatch(component)
        step_match = CGROUP_STEP_COMPONENT_PATTERN.fullmatch(component)
        if "job_" in component and job_match is None:
            raise SlurmValidationError("Process cgroup contains a nonstructural job substring")
        if "step_" in component and step_match is None:
            raise SlurmValidationError("Process cgroup contains a nonstructural step substring")
        if job_match is not None:
            job_components.append((component_index, job_match.group(1)))
        if step_match is not None:
            step_components.append((component_index, step_match.group(1)))
    if len(job_components) != 1 or len(step_components) != 1:
        raise SlurmValidationError("Process cgroup must contain exactly one Slurm job and step component")
    job_component_index, job_identifier = job_components[0]
    step_component_index, step_identifier = step_components[0]
    if job_identifier != job_id or step_identifier != step_id or job_component_index >= step_component_index:
        raise SlurmValidationError("Process cgroup does not identify the requested ordered job and step")


def parse_configuration_fields(configuration: str, *, label: str) -> dict[str, str]:
    """Parse unique or identical repeated key/value configuration lines."""
    fields: dict[str, str] = {}
    for line in configuration.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        if "=" not in stripped_line:
            continue
        field_name, field_value = (field.strip() for field in stripped_line.split("=", maxsplit=1))
        if not field_name:
            raise SlurmValidationError(f"{label} contains an assignment without a field name")
        existing_value = fields.get(field_name)
        if existing_value is not None and existing_value != field_value:
            raise SlurmValidationError(f"{label} contains conflicting values for {field_name}")
        fields[field_name] = field_value
    return fields


def parse_configuration_boolean(fields: dict[str, str], field_name: str) -> bool:
    """Parse one Slurm boolean whose documented default is disabled."""
    value = fields.get(field_name, "no").lower()
    if value in {"yes", "true", "1"}:
        return True
    if value in {"no", "false", "0"}:
        return False
    raise SlurmValidationError(f"Slurm cgroup configuration has an invalid {field_name}: {value}")


def validate_slurm_configuration(configuration: str, request: SlurmValidationRequest) -> None:
    """Bind controller routing and cgroup tracking to the expected cluster."""
    fields = parse_configuration_fields(configuration, label="Slurm controller configuration")
    if fields.get("ClusterName") != request.cluster_name:
        raise SlurmValidationError("Slurm controller configuration identifies another cluster")
    if fields.get("ProctrackType") != "proctrack/cgroup":
        raise SlurmValidationError("Slurm controller does not track processes through cgroups")
    task_plugins = fields.get("TaskPlugin", "").split(",")
    if "task/cgroup" not in task_plugins:
        raise SlurmValidationError("Slurm controller does not expose the task/cgroup plugin")


def expected_bootstrap_command_line(request: SlurmValidationRequest) -> bytes:
    """Return the exact direct-shebang bootstrap command line."""
    arguments = (
        str(SYSTEM_BASH_PATH),
        str(request.bootstrap_path),
        str(request.source_repository),
        request.expected_git_commit,
    )
    return b"\0".join(argument.encode("utf-8") for argument in arguments) + b"\0"


def validate_environment_consistency(
    environment_values: dict[str, str],
    request: SlurmValidationRequest,
) -> None:
    """Treat the bootstrap environment only as a consistency assertion."""
    expected_values = {
        "SLURM_CPUS_PER_TASK": str(request.expected_cpu_count),
        "SLURM_JOB_ID": request.job_id,
        "SLURM_JOB_NODELIST": request.node_name,
        "SLURM_JOB_UID": str(request.user_id),
        "SLURM_JOB_USER": request.user_name,
        "SLURM_STEP_ID": request.step_id,
        "SLURM_STEP_NODELIST": request.node_name,
        "SLURMD_NODENAME": request.node_name,
    }
    if environment_values != expected_values:
        raise SlurmValidationError("Slurm process environment disagrees with trusted live observations")


def validate_slurm_observations(
    request: SlurmValidationRequest,
    observations: SlurmObservedState,
) -> SlurmProcessAttestation:
    """Validate two live controller snapshots and local process membership."""
    validate_slurm_request(request)
    validate_environment_consistency(observations.environment_values, request)
    if observations.observer_parent_process_id != request.process_id:
        raise SlurmValidationError("Slurm attestation helper is not a direct child of the bootstrap process")
    if observations.observer_process_id == request.process_id:
        raise SlurmValidationError("Slurm attestation helper and bootstrap unexpectedly share a PID")
    if observations.process_namespace_pid_values != (request.process_id,):
        raise SlurmValidationError("Bootstrap PID is namespace-hidden or differs from its host PID")
    if (
        observations.process_pid_namespace_inode != observations.host_pid_namespace_inode
        or observations.observer_pid_namespace_inode != observations.host_pid_namespace_inode
        or observations.process_cgroup_namespace_inode != observations.host_cgroup_namespace_inode
        or observations.observer_cgroup_namespace_inode != observations.host_cgroup_namespace_inode
    ):
        raise SlurmValidationError("Bootstrap and observer are not in the host PID and cgroup namespaces")
    if observations.process_start_time_ticks_before != observations.process_start_time_ticks_after:
        raise SlurmValidationError("Bootstrap PID start identity changed during Slurm observation")
    try:
        uuid.UUID(observations.host_boot_id)
    except ValueError as error:
        raise SlurmValidationError("Host boot ID is malformed") from error

    expected_command_line = expected_bootstrap_command_line(request)
    if (
        observations.process_command_line_before != expected_command_line
        or observations.process_command_line_after != expected_command_line
    ):
        raise SlurmValidationError("Bootstrap process command line does not match the direct trusted launch")
    if observations.process_executable_path_before != str(
        SYSTEM_BASH_PATH
    ) or observations.process_executable_path_after != str(SYSTEM_BASH_PATH):
        raise SlurmValidationError("Bootstrap process executable is not the fixed system Bash")
    if (
        observations.bootstrap_sha256_before != request.bootstrap_sha256
        or observations.bootstrap_sha256_after != request.bootstrap_sha256
        or observations.bootstrap_file_identity_before != observations.bootstrap_file_identity_after
    ):
        raise SlurmValidationError("Bootstrap file identity changed during Slurm observation")

    first_job_fields = parse_scheduler_record(observations.first_job_record, label="first Slurm job record")
    second_job_fields = parse_scheduler_record(observations.second_job_record, label="second Slurm job record")
    first_step_fields = parse_scheduler_record(observations.first_step_record, label="first Slurm step record")
    second_step_fields = parse_scheduler_record(observations.second_step_record, label="second Slurm step record")
    first_job_entitlement = parse_job_entitlement(first_job_fields, request, label="first Slurm job record")
    second_job_entitlement = parse_job_entitlement(second_job_fields, request, label="second Slurm job record")
    first_step_entitlement = parse_step_entitlement(first_step_fields, request, label="first Slurm step record")
    second_step_entitlement = parse_step_entitlement(second_step_fields, request, label="second Slurm step record")
    if first_job_entitlement != second_job_entitlement:
        raise SlurmValidationError("Slurm job entitlement changed between controller snapshots")
    if first_step_entitlement != second_step_entitlement:
        raise SlurmValidationError("Slurm step entitlement changed between controller snapshots")
    if (
        first_job_entitlement.node_count != first_step_entitlement.node_count
        or first_job_entitlement.cpu_count < first_step_entitlement.cpu_count
        or first_job_entitlement.memory_bytes != first_step_entitlement.memory_bytes
        or first_job_entitlement.gpu_count != first_step_entitlement.gpu_count
        or first_job_entitlement.task_count != first_step_entitlement.task_count
    ):
        raise SlurmValidationError("Slurm job allocation and numeric-step entitlement disagree")

    validate_listpids(observations.listpids_record, request)
    validate_cgroup_mount(observations.process_mountinfo_record)
    cgroup_v2_path = parse_cgroup_v2_path(observations.process_cgroup_record)
    validate_cgroup_membership(
        cgroup_v2_path,
        job_id=request.job_id,
        step_id=request.step_id,
    )
    if not observations.cgroup_directory_exists:
        raise SlurmValidationError("Bootstrap cgroup-v2 membership directory is not live")
    validate_slurm_configuration(observations.slurm_configuration, request)
    cgroup_configuration_fields = parse_configuration_fields(
        observations.cgroup_configuration,
        label="Slurm cgroup configuration",
    )
    constrain_cores = parse_configuration_boolean(cgroup_configuration_fields, "ConstrainCores")
    constrain_ram_space = parse_configuration_boolean(cgroup_configuration_fields, "ConstrainRAMSpace")
    constrain_devices = parse_configuration_boolean(cgroup_configuration_fields, "ConstrainDevices")
    kernel_enforcement_proven = constrain_cores and constrain_ram_space and constrain_devices
    if request.require_kernel_enforcement and not kernel_enforcement_proven:
        raise SlurmValidationError(
            "Strict kernel enforcement is unsupported: "
            "ConstrainCores, ConstrainRAMSpace, and ConstrainDevices are not all enabled"
        )

    return SlurmProcessAttestation(
        schema_version=SCHEMA_VERSION,
        cluster_name=request.cluster_name,
        node_name=request.node_name,
        job_id=request.job_id,
        step_id=request.step_id,
        user_name=request.user_name,
        user_id=request.user_id,
        host_boot_id=observations.host_boot_id,
        host_process_id=request.process_id,
        host_process_start_time_ticks=observations.process_start_time_ticks_before,
        host_process_pid_namespace_inode=observations.process_pid_namespace_inode,
        host_process_cgroup_namespace_inode=observations.process_cgroup_namespace_inode,
        cgroup_v2_path=cgroup_v2_path.as_posix(),
        job_node_count=first_job_entitlement.node_count,
        job_cpu_count=first_job_entitlement.cpu_count,
        job_memory_bytes=first_job_entitlement.memory_bytes,
        job_gpu_count=first_job_entitlement.gpu_count,
        job_task_count=first_job_entitlement.task_count,
        step_node_count=first_step_entitlement.node_count,
        step_cpu_count=first_step_entitlement.cpu_count,
        step_memory_bytes=first_step_entitlement.memory_bytes,
        step_gpu_count=first_step_entitlement.gpu_count,
        step_task_count=first_step_entitlement.task_count,
        first_job_record_sha256=sha256_bytes(observations.first_job_record.encode("utf-8")),
        second_job_record_sha256=sha256_bytes(observations.second_job_record.encode("utf-8")),
        first_step_record_sha256=sha256_bytes(observations.first_step_record.encode("utf-8")),
        second_step_record_sha256=sha256_bytes(observations.second_step_record.encode("utf-8")),
        listpids_sha256=sha256_bytes(observations.listpids_record.encode("utf-8")),
        bootstrap_path=str(request.bootstrap_path),
        bootstrap_sha256=request.bootstrap_sha256,
        job_command=required_scheduler_field(first_job_fields, "Command", label="first Slurm job record"),
        bootstrap_process_command_sha256=sha256_bytes(expected_command_line),
        constrain_cores=constrain_cores,
        constrain_ram_space=constrain_ram_space,
        constrain_devices=constrain_devices,
        scheduler_entitlement_proven=True,
        kernel_enforcement_proven=kernel_enforcement_proven,
    )


def slurm_process_attestation_payload(attestation: SlurmProcessAttestation) -> dict[str, object]:
    """Serialize one Slurm process attestation."""
    return {field_name: getattr(attestation, field_name) for field_name in sorted(ATTESTATION_FIELDS)}


def parse_payload_mapping(value: object, *, label: str, expected_fields: frozenset[str]) -> dict[str, object]:
    """Parse a mapping with an exact field set."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SlurmValidationError(f"{label} must be a string-keyed object")
    payload = typing.cast("dict[str, object]", value)
    observed_fields = frozenset(payload)
    if observed_fields != expected_fields:
        missing_fields = sorted(expected_fields - observed_fields)
        unexpected_fields = sorted(observed_fields - expected_fields)
        raise SlurmValidationError(f"{label} field mismatch: missing={missing_fields}, unexpected={unexpected_fields}")
    return payload


def parse_payload_string(payload: dict[str, object], field_name: str) -> str:
    """Parse one nonempty string payload field."""
    value = payload[field_name]
    if not isinstance(value, str) or not value:
        raise SlurmValidationError(f"Slurm attestation.{field_name} must be a nonempty string")
    return value


def parse_payload_integer(payload: dict[str, object], field_name: str, *, positive: bool) -> int:
    """Parse one exact integer payload field."""
    value = payload[field_name]
    if type(value) is not int:
        raise SlurmValidationError(f"Slurm attestation.{field_name} must be an integer")
    integer_value = value
    if (positive and integer_value <= 0) or (not positive and integer_value < 0):
        raise SlurmValidationError(f"Slurm attestation.{field_name} has an invalid sign")
    return integer_value


def parse_payload_boolean(payload: dict[str, object], field_name: str) -> bool:
    """Parse one exact Boolean payload field."""
    value = payload[field_name]
    if type(value) is not bool:
        raise SlurmValidationError(f"Slurm attestation.{field_name} must be a Boolean")
    return value


def parse_slurm_process_attestation(value: object) -> SlurmProcessAttestation:
    """Parse one strict schema-0 Slurm process attestation."""
    payload = parse_payload_mapping(
        value,
        label="Slurm attestation",
        expected_fields=ATTESTATION_FIELDS,
    )
    schema_version = parse_payload_integer(payload, "schema_version", positive=False)
    if schema_version != SCHEMA_VERSION:
        raise SlurmValidationError(f"Unsupported Slurm attestation schema: {schema_version}")
    attestation = SlurmProcessAttestation(
        schema_version=schema_version,
        cluster_name=parse_payload_string(payload, "cluster_name"),
        node_name=parse_payload_string(payload, "node_name"),
        job_id=parse_payload_string(payload, "job_id"),
        step_id=parse_payload_string(payload, "step_id"),
        user_name=parse_payload_string(payload, "user_name"),
        user_id=parse_payload_integer(payload, "user_id", positive=False),
        host_boot_id=parse_payload_string(payload, "host_boot_id"),
        host_process_id=parse_payload_integer(payload, "host_process_id", positive=True),
        host_process_start_time_ticks=parse_payload_integer(
            payload,
            "host_process_start_time_ticks",
            positive=True,
        ),
        host_process_pid_namespace_inode=parse_payload_integer(
            payload,
            "host_process_pid_namespace_inode",
            positive=True,
        ),
        host_process_cgroup_namespace_inode=parse_payload_integer(
            payload,
            "host_process_cgroup_namespace_inode",
            positive=True,
        ),
        cgroup_v2_path=parse_payload_string(payload, "cgroup_v2_path"),
        job_node_count=parse_payload_integer(payload, "job_node_count", positive=True),
        job_cpu_count=parse_payload_integer(payload, "job_cpu_count", positive=True),
        job_memory_bytes=parse_payload_integer(payload, "job_memory_bytes", positive=True),
        job_gpu_count=parse_payload_integer(payload, "job_gpu_count", positive=True),
        job_task_count=parse_payload_integer(payload, "job_task_count", positive=True),
        step_node_count=parse_payload_integer(payload, "step_node_count", positive=True),
        step_cpu_count=parse_payload_integer(payload, "step_cpu_count", positive=True),
        step_memory_bytes=parse_payload_integer(payload, "step_memory_bytes", positive=True),
        step_gpu_count=parse_payload_integer(payload, "step_gpu_count", positive=True),
        step_task_count=parse_payload_integer(payload, "step_task_count", positive=True),
        first_job_record_sha256=parse_payload_string(payload, "first_job_record_sha256"),
        second_job_record_sha256=parse_payload_string(payload, "second_job_record_sha256"),
        first_step_record_sha256=parse_payload_string(payload, "first_step_record_sha256"),
        second_step_record_sha256=parse_payload_string(payload, "second_step_record_sha256"),
        listpids_sha256=parse_payload_string(payload, "listpids_sha256"),
        bootstrap_path=parse_payload_string(payload, "bootstrap_path"),
        bootstrap_sha256=parse_payload_string(payload, "bootstrap_sha256"),
        job_command=parse_payload_string(payload, "job_command"),
        bootstrap_process_command_sha256=parse_payload_string(
            payload,
            "bootstrap_process_command_sha256",
        ),
        constrain_cores=parse_payload_boolean(payload, "constrain_cores"),
        constrain_ram_space=parse_payload_boolean(payload, "constrain_ram_space"),
        constrain_devices=parse_payload_boolean(payload, "constrain_devices"),
        scheduler_entitlement_proven=parse_payload_boolean(payload, "scheduler_entitlement_proven"),
        kernel_enforcement_proven=parse_payload_boolean(payload, "kernel_enforcement_proven"),
    )
    for digest_field_name in (
        "first_job_record_sha256",
        "second_job_record_sha256",
        "first_step_record_sha256",
        "second_step_record_sha256",
        "listpids_sha256",
        "bootstrap_sha256",
        "bootstrap_process_command_sha256",
    ):
        if SHA256_PATTERN.fullmatch(typing.cast("str", getattr(attestation, digest_field_name))) is None:
            raise SlurmValidationError(f"Slurm attestation.{digest_field_name} is malformed")
    bootstrap_path = PurePosixPath(attestation.bootstrap_path)
    if (
        not bootstrap_path.is_absolute()
        or bootstrap_path.as_posix() != attestation.bootstrap_path
        or "//" in attestation.bootstrap_path
        or "/./" in f"{attestation.bootstrap_path}/"
        or "/../" in f"{attestation.bootstrap_path}/"
        or "\0" in attestation.bootstrap_path
        or attestation.job_command != attestation.bootstrap_path
    ):
        raise SlurmValidationError("Slurm attestation bootstrap command identity is inconsistent")
    if (
        NAME_PATTERN.fullmatch(attestation.cluster_name) is None
        or NAME_PATTERN.fullmatch(attestation.node_name) is None
        or NAME_PATTERN.fullmatch(attestation.user_name) is None
    ):
        raise SlurmValidationError("Slurm attestation cluster, node, or user name is malformed")
    if NUMERIC_IDENTIFIER_PATTERN.fullmatch(attestation.job_id) is None:
        raise SlurmValidationError("Slurm attestation job ID is not numeric")
    if NUMERIC_IDENTIFIER_PATTERN.fullmatch(attestation.step_id) is None:
        raise SlurmValidationError("Slurm attestation step ID is not numeric")
    try:
        uuid.UUID(attestation.host_boot_id)
    except ValueError as error:
        raise SlurmValidationError("Slurm attestation host boot ID is malformed") from error
    cgroup_v2_path = parse_cgroup_v2_path(f"0::{attestation.cgroup_v2_path}\n")
    validate_cgroup_membership(
        cgroup_v2_path,
        job_id=attestation.job_id,
        step_id=attestation.step_id,
    )
    if (
        attestation.job_node_count != 1
        or attestation.step_node_count != 1
        or attestation.step_cpu_count != QUALIFICATION_CPU_COUNT
        or attestation.job_cpu_count < attestation.step_cpu_count
        or attestation.step_memory_bytes != QUALIFICATION_MEMORY_BYTES
        or attestation.job_memory_bytes != attestation.step_memory_bytes
        or attestation.step_gpu_count != QUALIFICATION_GPU_COUNT
        or attestation.job_gpu_count != attestation.step_gpu_count
        or attestation.job_task_count != 1
        or attestation.step_task_count != 1
    ):
        raise SlurmValidationError("Slurm attestation differs from the exact qualification resource entitlement")
    if not attestation.scheduler_entitlement_proven:
        raise SlurmValidationError("Slurm attestation does not prove scheduler entitlement")
    if attestation.kernel_enforcement_proven != (
        attestation.constrain_cores and attestation.constrain_ram_space and attestation.constrain_devices
    ):
        raise SlurmValidationError("Slurm attestation kernel-enforcement claim is inconsistent")
    return attestation


def canonical_slurm_process_attestation(attestation: SlurmProcessAttestation) -> bytes:
    """Return canonical one-line JSON for one validated attestation."""
    parsed_attestation = parse_slurm_process_attestation(slurm_process_attestation_payload(attestation))
    payload = slurm_process_attestation_payload(parsed_attestation)
    return f"{json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)}\n".encode()


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the fixed-route command-line parser."""
    argument_parser = argparse.ArgumentParser(
        description="Prove exact Slurm job-step membership for parity qualification.",
    )
    argument_parser.add_argument("--cluster-name", required=True)
    argument_parser.add_argument("--node-name", required=True)
    argument_parser.add_argument("--job-id", required=True)
    argument_parser.add_argument("--step-id", required=True)
    argument_parser.add_argument("--user-name", required=True)
    argument_parser.add_argument("--user-id", required=True, type=int)
    argument_parser.add_argument("--process-id", required=True, type=int)
    argument_parser.add_argument("--bootstrap-path", required=True, type=Path)
    argument_parser.add_argument("--bootstrap-sha256", required=True)
    argument_parser.add_argument("--source-repository", required=True, type=Path)
    argument_parser.add_argument("--expected-git-commit", required=True)
    argument_parser.add_argument("--expected-cpu-count", required=True, type=int)
    argument_parser.add_argument("--expected-memory-bytes", required=True, type=int)
    argument_parser.add_argument("--expected-gpu-count", required=True, type=int)
    argument_parser.add_argument("--require-kernel-enforcement", action="store_true")
    return argument_parser


def main(arguments: typing.Sequence[str] | None = None) -> int:
    """Collect, validate, and emit canonical schema-0 JSON."""
    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(arguments)
    request = SlurmValidationRequest(
        schema_version=SCHEMA_VERSION,
        cluster_name=parsed_arguments.cluster_name,
        node_name=parsed_arguments.node_name,
        job_id=parsed_arguments.job_id,
        step_id=parsed_arguments.step_id,
        user_name=parsed_arguments.user_name,
        user_id=parsed_arguments.user_id,
        process_id=parsed_arguments.process_id,
        bootstrap_path=parsed_arguments.bootstrap_path,
        bootstrap_sha256=parsed_arguments.bootstrap_sha256,
        source_repository=parsed_arguments.source_repository,
        expected_git_commit=parsed_arguments.expected_git_commit,
        expected_cpu_count=parsed_arguments.expected_cpu_count,
        expected_memory_bytes=parsed_arguments.expected_memory_bytes,
        expected_gpu_count=parsed_arguments.expected_gpu_count,
        require_kernel_enforcement=parsed_arguments.require_kernel_enforcement,
    )
    try:
        observations = collect_live_observations(request)
        attestation = validate_slurm_observations(request, observations)
    except (OSError, subprocess.SubprocessError, UnicodeError, SlurmValidationError) as error:
        sys.stderr.write(f"Exact Slurm attestation failed: {error}\n")
        return 1
    sys.stdout.buffer.write(canonical_slurm_process_attestation(attestation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
