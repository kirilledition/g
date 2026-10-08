"""Isolated, auditable Workbench attempts with fresh native output identities."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import typing
import uuid
from pathlib import Path

from tooling.common import g_regenie
from tooling.workbench import manifest, outputs, preflight, profile, storage

CANCELLATION_GRACE_SECONDS = 30.0


class Action(enum.StrEnum):
    """Supported Workbench CLI actions."""

    PREFLIGHT = "preflight"
    PROFILE = "profile"
    RUN = "run"


@dataclasses.dataclass(frozen=True)
class RunnerArguments:
    """CLI settings independent of scientific input manifests."""

    action: Action
    manifest_path: Path
    work_directory: Path
    dry_run: bool
    billing_project: str | None
    runner_prefix: tuple[str, ...]
    profile_max_variants: int


@dataclasses.dataclass(frozen=True)
class AttemptResult:
    """The final local evidence path and optional committed publication."""

    attempt_directory: Path
    publication_uri: str | None
    dry_run: bool


def utc_timestamp() -> str:
    """Return an unambiguous UTC timestamp for diagnostic artifacts."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def validate_private_directory(path: Path) -> None:
    """Require current-user ownership and exclude untrusted cache writers."""
    observed = path.stat()
    if path.is_symlink() or observed.st_uid != os.getuid() or observed.st_mode & 0o022:
        raise ValueError(
            "Workbench work/cache directory must be owned by this user and not writable by group or others."
        )


def new_attempt_directory(work_directory: Path) -> Path:
    """Create one unpredictable, never-reused attempt directory."""
    work_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    validate_private_directory(work_directory)
    attempt_id = datetime.datetime.now(datetime.UTC).strftime("attempt-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
    attempt = work_directory / attempt_id
    attempt.mkdir(mode=0o700)
    return attempt


def localize_inputs(
    job: manifest.JobManifest, attempt: Path, billing_project: str | None
) -> dict[str, storage.LocalizedFile]:
    """Materialize trusted input names independent of source URI basenames."""
    localized = {
        name: storage.localize_file(source, attempt / "inputs" / name, billing_project)
        for name, source in job.inputs.items()
    }
    for prediction in job.predictions:
        key = f"prediction:{prediction.phenotype}"
        localized[key] = storage.localize_file(
            prediction.file, attempt / "inputs" / f"prediction-{prediction.phenotype}.loco", billing_project
        )
    return localized


def build_engine_spec(
    job: manifest.JobManifest,
    paths: dict[str, Path],
    attempt: Path,
    cache_directory: Path,
    capacity: preflight.CapacityPlan,
    command_prefix: tuple[str, ...],
) -> g_regenie.RegenieRunSpec:
    """Render the maintained native TOML contract through its shared owner."""
    prediction_list = attempt / "predictions.list"
    prediction_list.write_text(
        "".join(
            f"{entry.phenotype} {paths[f'prediction:{entry.phenotype}'].relative_to(attempt)}\n"
            for entry in job.predictions
        ),
        encoding="utf-8",
    )
    binary = (
        g_regenie.RegenieBinaryOptions(
            fallback_method=g_regenie.RegenieBinaryFallback(job.analysis.fallback_method.value),
            p_threshold=job.analysis.p_threshold,
            firth_se=None,
        )
        if job.analysis.fallback_method is not None
        else None
    )
    return g_regenie.RegenieRunSpec(
        trait_kind=g_regenie.RegenieTraitKind(job.analysis.trait_type.value),
        command_prefix=command_prefix,
        inputs=g_regenie.RegenieInputSpec(
            bgen_path=paths["bgen"],
            sample_path=paths["sample"],
            phenotype_path=paths["pheno_file"],
            phenotype_columns=job.analysis.phenotype_columns,
            covariate_path=paths.get("covar_file"),
            covariate_columns=job.analysis.covariate_columns,
            prediction_list_path=prediction_list,
            output_prefix=attempt / "association",
        ),
        compute=g_regenie.RegenieComputeOptions(
            device=g_regenie.RegenieDevice(job.resources.device.value),
            bsize=capacity.chunk_variants,
            cpu_threads=job.resources.cpu_threads,
            multi_phenotype_sample_mode=g_regenie.RegenieMultiPhenotypeSampleMode.PER_PHENOTYPE,
            firth_batch_size=capacity.firth_batch_size,
            firth_candidate_capacity=(
                preflight.FIRTH_CANDIDATE_CAPACITY
                if job.analysis.fallback_method == manifest.FallbackMethod.FIRTH_APPROXIMATE
                else None
            ),
            jax_cache_dir=cache_directory,
        ),
        output=g_regenie.RegenieOutputOptions(
            output_run_directory=attempt / "engine", writer_threads=job.resources.writer_threads, resume=False
        ),
        diagnostics=g_regenie.RegenieDiagnosticsOptions(telemetry=g_regenie.RegenieTelemetry.PROGRESS),
        binary=binary,
    )


def engine_environment(job: manifest.JobManifest, cache_directory: Path) -> dict[str, str]:
    """Specify bounded parallelism, explicit device and a trusted local cache."""
    threads = str(job.resources.cpu_threads)
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "JAX_PLATFORMS": "cuda" if job.resources.device == manifest.Device.GPU else "cpu",
            "JAX_ENABLE_X64": "true",
            "JAX_COMPILATION_CACHE_DIR": str(cache_directory),
        }
    )
    if job.resources.device == manifest.Device.GPU:
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(job.resources.memory_fraction)
    return environment


def signal_process_group(process_group: int, signal_number: int) -> bool:
    """Signal the isolated engine session, including launcher descendants."""
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return False
    return True


def terminate_child(child: subprocess.Popen[bytes]) -> None:
    """Allow all engine descendants to preserve commits, then kill the group."""
    try:
        if not signal_process_group(child.pid, signal.SIGINT):
            child.wait()
            return
        deadline = time.monotonic() + CANCELLATION_GRACE_SECONDS
        while time.monotonic() < deadline:
            child.poll()
            if not signal_process_group(child.pid, 0):
                child.wait()
                return
            time.sleep(0.1)
        signal_process_group(child.pid, signal.SIGKILL)
        child.wait()
    except KeyboardInterrupt:
        # Repeated cancellation requests force shutdown immediately. Returning
        # lets execute_engine re-raise the original cancellation afterwards.
        signal_process_group(child.pid, signal.SIGKILL)
        child.wait()


def execute_engine(command: list[str], attempt: Path, environment: dict[str, str]) -> None:
    """Launch one fresh subprocess and retain both success and failure logs."""
    with (
        (attempt / "engine.stdout.log").open("xb") as stdout,
        (attempt / "engine.stderr.log").open("xb") as stderr,
        subprocess.Popen(
            command, cwd=attempt, env=environment, stdout=stdout, stderr=stderr, start_new_session=True
        ) as child,
    ):
        try:
            return_code = child.wait()
        except KeyboardInterrupt:
            terminate_child(child)
            raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def verify_localized_inputs(localized: dict[str, storage.LocalizedFile]) -> None:
    """Reject inputs changed by a process before outputs can be published."""
    for source in localized.values():
        storage.verify_file(source.path, source.source)


def create_bundle(
    attempt: Path,
    job: manifest.JobManifest,
    validated_files: tuple[outputs.OutputFile, ...],
) -> Path:
    """Stage validated results and restricted provenance without copying inputs."""
    bundle = attempt / "publication"
    bundle.mkdir()
    for output in validated_files:
        source = attempt / "engine" / output.relative_path
        destination = bundle / "outputs" / output.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        observed = storage.file_digest(destination)
        if observed.sha256 != output.sha256 or observed.size_bytes != output.size_bytes:
            raise ValueError("Validated engine output changed while staging publication.")
    for name in (
        "job.json",
        "engine.toml",
        "preflight.json",
        "alignment.json",
        "capacity.json",
        "provenance.json",
        "engine.stdout.log",
        "engine.stderr.log",
    ):
        shutil.copyfile(attempt / name, bundle / name)
    inventory = [
        {
            "relative_path": path.relative_to(bundle).as_posix(),
            **dataclasses.asdict(storage.file_digest(path)),
        }
        for path in storage.regular_files(bundle)
    ]
    storage.write_json_atomic(
        bundle / "COMMITTED.json",
        {
            "schema_version": 1,
            "attempt_id": attempt.name,
            "status": "completed",
            "created_at_utc": utc_timestamp(),
            "dataset": dataclasses.asdict(job.dataset),
            "files": inventory,
            "restricted_workspace_artifact": True,
            "safe_for_external_export": False,
        },
    )
    return bundle


def installed_versions() -> dict[str, str | None]:
    """Inspect installed package metadata without initializing numerical runtimes."""
    result: dict[str, str | None] = {}
    for name in ("g", "jax", "jaxlib", "numpy", "pyarrow", "nvidia-libnvcomp-cu12"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def launcher_native_identity() -> dict[str, object]:
    """Locate native bytes through the package path without loading the extension."""
    specification = importlib.util.find_spec("g")
    locations = specification.submodule_search_locations if specification is not None else None
    identities: dict[str, object] = {}
    for location in locations or ():
        for path in sorted(Path(location).glob("_core*.so")):
            identities[str(path)] = dataclasses.asdict(storage.file_digest(path))
    return identities


def provenance_document(
    job: manifest.JobManifest,
    localized: dict[str, storage.LocalizedFile],
    command: list[str],
    environment: dict[str, str],
    manifest_path: Path,
) -> dict[str, object]:
    """Record reproducible content identity and scoped runtime settings."""
    executable = shutil.which(command[0])
    executable_digest = storage.file_digest(Path(executable)) if executable is not None else None
    return {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "dataset": dataclasses.asdict(job.dataset),
        "inputs": {
            name: {
                "source": dataclasses.asdict(source.source),
                "localized_path": str(source.path),
                "resolved_generation": source.generation,
            }
            for name, source in localized.items()
        },
        "command": command,
        "executable": executable,
        "executable_digest": dataclasses.asdict(executable_digest) if executable_digest is not None else None,
        "python_version": sys.version,
        "launcher_package_versions": installed_versions(),
        "launcher_native_extensions": launcher_native_identity(),
        "source_revision": environment.get("GWAS_ENGINE_SOURCE_REVISION"),
        "job_manifest_digest": dataclasses.asdict(storage.file_digest(manifest_path)),
        "localized_input_content_verified": True,
        "platform": platform.platform(),
        "environment": {
            key: value
            for key, value in environment.items()
            if key.startswith(("JAX_", "XLA_")) or key in {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
        },
        "original_native_resume_policy_changed": False,
        "all_of_us_qualified": False,
    }


def run_attempt(arguments: RunnerArguments) -> AttemptResult:
    """Verify, localize and execute an isolated job, committing only valid outputs."""
    if not arguments.runner_prefix or any(not value or "\x00" in value for value in arguments.runner_prefix):
        raise ValueError("runner_prefix must contain a nonempty, shell-free command argument list.")
    if arguments.profile_max_variants <= 0:
        raise ValueError("profile_max_variants must be positive.")
    attempt = new_attempt_directory(arguments.work_directory.resolve())
    shutil.copyfile(arguments.manifest_path, attempt / "job.json")
    status_path = attempt / "status.json"
    storage.write_json_atomic(
        status_path, {"status": "preparing", "attempt_id": attempt.name, "started_at": utc_timestamp()}
    )
    publishing_started = False
    try:
        job = manifest.load_manifest(attempt / "job.json")
        report = preflight.run_preflight(
            job,
            arguments.work_directory,
            dry_run=arguments.dry_run,
            billing_project=arguments.billing_project,
        )
        storage.write_json_atomic(attempt / "preflight.json", dataclasses.asdict(report))
        if arguments.action == Action.PREFLIGHT or arguments.dry_run:
            storage.write_json_atomic(
                status_path,
                {
                    "status": "dry_run" if arguments.dry_run else "preflight_completed",
                    "all_input_content_verified": report.all_input_content_verified,
                    "attempt_id": attempt.name,
                    "completed_at": utc_timestamp(),
                    "engine_executed": False,
                },
            )
            return AttemptResult(attempt, None, arguments.dry_run)
        localized = localize_inputs(job, attempt, arguments.billing_project)
        paths = {name: source.path for name, source in localized.items()}
        header = profile.read_header(paths["bgen"])
        if header.layout != 2 or header.sample_count <= 0 or header.variant_count <= 0:
            raise ValueError("Workbench jobs require a nonempty BGEN Layout 2 file.")
        alignment = preflight.validate_alignment(job, paths, header)
        capacity = preflight.plan_capacity(job, header.sample_count, header.variant_count)
        if preflight.available_disk_bytes(attempt) < preflight.estimated_disk_bytes(job, header.variant_count) - sum(
            source.size_bytes for source in manifest.iter_input_files(job)
        ):
            raise ValueError("Insufficient free local disk after localization for estimated output copies.")
        storage.write_json_atomic(attempt / "alignment.json", dataclasses.asdict(alignment))
        storage.write_json_atomic(attempt / "capacity.json", dataclasses.asdict(capacity))
        if arguments.action == Action.PROFILE:
            inventory = profile.profile_bgen(paths["bgen"], arguments.profile_max_variants, sample_path=paths["sample"])
            storage.write_json_atomic(attempt / "bgen-profile.json", dataclasses.asdict(inventory))
            verify_localized_inputs(localized)
            storage.write_json_atomic(status_path, {"status": "profile_completed", "completed_at": utc_timestamp()})
            return AttemptResult(attempt, None, dry_run=False)
        cache_directory = arguments.work_directory.resolve() / "trusted-jax-cache" / job.resources.device.value
        cache_root = arguments.work_directory.resolve() / "trusted-jax-cache"
        cache_root.mkdir(mode=0o700, exist_ok=True)
        validate_private_directory(cache_root)
        cache_directory.mkdir(mode=0o700, exist_ok=True)
        validate_private_directory(cache_directory)
        spec = build_engine_spec(job, paths, attempt, cache_directory, capacity, arguments.runner_prefix)
        config_path = g_regenie.write_regenie_toml(spec, attempt / "engine.toml")
        command = g_regenie.render_g_regenie_command(spec, config_path)
        environment = engine_environment(job, cache_directory)
        storage.write_json_atomic(
            attempt / "provenance.json", provenance_document(job, localized, command, environment, attempt / "job.json")
        )
        storage.write_json_atomic(
            status_path, {"status": "running", "attempt_id": attempt.name, "started_at": utc_timestamp()}
        )
        execute_engine(command, attempt, environment)
        verify_localized_inputs(localized)
        validated_files = outputs.validate_completed_outputs(
            attempt / "engine", job.analysis.phenotype_columns, header.variant_count
        )
        bundle = create_bundle(attempt, job, validated_files)
        publishing_started = True
        storage.write_json_atomic(status_path, {"status": "publishing", "attempt_id": attempt.name})
        publication_uri = storage.publish_bundle(bundle, job.output_uri, attempt.name, arguments.billing_project)
        storage.write_json_atomic(
            status_path,
            {
                "status": "completed",
                "attempt_id": attempt.name,
                "publication_uri": publication_uri,
                "completed_at": utc_timestamp(),
            },
        )
        return AttemptResult(attempt, publication_uri, dry_run=False)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        storage.write_json_atomic(
            status_path,
            {
                "status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                "attempt_id": attempt.name,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_at": utc_timestamp(),
                "publication_committed": None if publishing_started else False,
            },
        )
        raise


def interrupt_on_termination(signal_number: int, frame: object) -> typing.NoReturn:
    """Route process termination through the same artifact-preserving cancellation."""
    del frame
    raise KeyboardInterrupt(f"Received signal {signal_number}.")
