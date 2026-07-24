"""Tests for the CUDA-aware Clang-Tidy runner."""

from __future__ import annotations

import importlib.metadata
import subprocess
import unittest.mock
from pathlib import Path

import pytest

from tooling.debug import check_cuda_native

EXPECTED_NATIVE_PATHS = (
    Path("crates/compute-cuda/native/firth_components_kernel.cu"),
    Path("crates/genotype-cuda/native/packed8_kernel.cu"),
    Path("crates/compute-cuda/native/firth_components_ffi.cc"),
    Path("crates/genotype-cuda/native/packed8_deflate_ffi.cc"),
    Path("native/cuda-driver/cuda_driver.h"),
    Path("crates/genotype-cuda/native/nvcomp_abi.h"),
)


def create_repository(repository_root: Path) -> None:
    """Create a minimal repository with the maintained native inventory."""
    repository_root.mkdir(parents=True)
    (repository_root / ".clang-tidy").write_text("Checks: '-*'\n", encoding="utf-8")
    (repository_root / "Justfile").write_text(
        "cuda_native_sources := '" + " ".join(str(path) for path in EXPECTED_NATIVE_PATHS) + "'\n",
        encoding="utf-8",
    )
    (repository_root / "vendor/openxla").mkdir(parents=True)
    for relative_path in EXPECTED_NATIVE_PATHS:
        source_path = repository_root / relative_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("// test\n", encoding="utf-8")
    (repository_root / EXPECTED_NATIVE_PATHS[2]).write_text('#include "cuda_driver.h"\n', encoding="utf-8")
    (repository_root / EXPECTED_NATIVE_PATHS[3]).write_text(
        '#include "cuda_driver.h"\n#include "nvcomp_abi.h"\n',
        encoding="utf-8",
    )
    for relative_directory in check_cuda_native.NATIVE_DIRECTORIES:
        (repository_root / relative_directory / "README.md").write_text("ignored\n", encoding="utf-8")
        (repository_root / relative_directory / "generated.ptx").write_text("ignored\n", encoding="utf-8")


def create_toolkit_sources(source_root: Path) -> check_cuda_native.CudaToolkitSources:
    """Create minimal CUDA wheel directory trees for toolkit assembly tests."""
    runtime_include_directory = source_root / "runtime"
    compiler_include_directory = source_root / "compiler"
    core_compute_libraries_include_directory = source_root / "core_compute_libraries"
    random_include_directory = source_root / "random"
    libdevice_directory = source_root / "libdevice"
    for directory in (
        runtime_include_directory,
        compiler_include_directory,
        core_compute_libraries_include_directory,
        random_include_directory,
        libdevice_directory,
    ):
        directory.mkdir(parents=True)

    (runtime_include_directory / "cuda.h").write_text("// runtime\n", encoding="utf-8")
    (runtime_include_directory / "__init__.py").write_text("", encoding="utf-8")
    (compiler_include_directory / "crt").mkdir()
    (compiler_include_directory / "crt/host_config.h").write_text("// compiler\n", encoding="utf-8")
    (core_compute_libraries_include_directory / "thrust").mkdir()
    (core_compute_libraries_include_directory / "thrust/version.h").write_text("// cccl\n", encoding="utf-8")
    (random_include_directory / "curand_kernel.h").write_text("// random\n", encoding="utf-8")
    (libdevice_directory / "libdevice.10.bc").write_bytes(b"bitcode")
    return check_cuda_native.CudaToolkitSources(
        runtime_include_directory=runtime_include_directory,
        compiler_include_directory=compiler_include_directory,
        core_compute_libraries_include_directory=core_compute_libraries_include_directory,
        random_include_directory=random_include_directory,
        libdevice_directory=libdevice_directory,
    )


def create_distribution(distribution_root: Path, version: str) -> importlib.metadata.Distribution:
    """Create import metadata rooted beside fake installed package files."""
    metadata_directory = distribution_root / f"fake-{version}.dist-info"
    metadata_directory.mkdir(parents=True)
    (metadata_directory / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: fake\nVersion: {version}\n",
        encoding="utf-8",
    )
    return importlib.metadata.Distribution.at(metadata_directory)


def test_discover_repository_inventory_matches_current_scope(tmp_path: Path) -> None:
    """The runner finds the four translation units and two transitive headers."""
    repository_root = tmp_path / "repository"
    create_repository(repository_root)

    inventory = check_cuda_native.discover_repository_inventory(repository_root)

    assert tuple(unit.relative_path for unit in inventory.translation_units) == EXPECTED_NATIVE_PATHS[:4]
    assert inventory.headers == EXPECTED_NATIVE_PATHS[4:]


def test_discover_repository_inventory_rejects_nested_native_source(tmp_path: Path) -> None:
    """A nested native file cannot silently bypass the exact lint allowlist."""
    repository_root = tmp_path / "repository"
    create_repository(repository_root)
    nested_source_path = repository_root / "crates/compute-cuda/native/nested/new_kernel.cu"
    nested_source_path.parent.mkdir()
    nested_source_path.write_text("// unexpected\n", encoding="utf-8")

    with pytest.raises(check_cuda_native.CudaNativeCheckError, match=r"unexpected: .*nested/new_kernel\.cu"):
        check_cuda_native.discover_repository_inventory(repository_root)


def test_discover_repository_inventory_rejects_formatter_allowlist_drift(tmp_path: Path) -> None:
    """Formatter and Clang-Tidy source allowlists cannot silently diverge."""
    repository_root = tmp_path / "repository"
    create_repository(repository_root)
    (repository_root / "Justfile").write_text("cuda_native_sources := 'missing.cu'\n", encoding="utf-8")

    with pytest.raises(check_cuda_native.CudaNativeCheckError, match="does not match"):
        check_cuda_native.discover_repository_inventory(repository_root)


def test_discover_repository_inventory_requires_transitive_header_include(tmp_path: Path) -> None:
    """A maintained header cannot be counted unless a lint translation unit includes it."""
    repository_root = tmp_path / "repository"
    create_repository(repository_root)
    (repository_root / EXPECTED_NATIVE_PATHS[3]).write_text('#include "cuda_driver.h"\n', encoding="utf-8")

    with pytest.raises(check_cuda_native.CudaNativeCheckError, match=r"nvcomp_abi\.h"):
        check_cuda_native.discover_repository_inventory(repository_root)


def test_resolve_distribution_directory_uses_locked_version_and_path(tmp_path: Path) -> None:
    """CUDA wheel paths are resolved through import metadata at an exact version."""
    relative_directory = Path("nvidia/cuda_runtime/include")
    expected_directory = tmp_path / relative_directory
    expected_directory.mkdir(parents=True)
    distribution = create_distribution(tmp_path, "12.9.79")
    distribution_spec = check_cuda_native.CudaDistributionSpec(
        name="nvidia-cuda-runtime-cu12",
        expected_version="12.9.79",
    )

    with unittest.mock.patch.object(check_cuda_native.importlib.metadata, "distribution", return_value=distribution):
        observed_directory = check_cuda_native.resolve_distribution_directory(
            distribution_spec,
            relative_directory,
        )

    assert observed_directory == expected_directory


def test_resolve_distribution_directory_rejects_version_drift(tmp_path: Path) -> None:
    """A CUDA wheel version different from the lock fails before Clang runs."""
    relative_directory = Path("nvidia/cuda_runtime/include")
    (tmp_path / relative_directory).mkdir(parents=True)
    distribution = create_distribution(tmp_path, "12.9.80")
    distribution_spec = check_cuda_native.CudaDistributionSpec(
        name="nvidia-cuda-runtime-cu12",
        expected_version="12.9.79",
    )

    with (
        unittest.mock.patch.object(check_cuda_native.importlib.metadata, "distribution", return_value=distribution),
        pytest.raises(check_cuda_native.CudaNativeCheckError, match=r"expected 12\.9\.79"),
    ):
        check_cuda_native.resolve_distribution_directory(distribution_spec, relative_directory)


def test_create_cuda_toolkit_layout_links_only_clang_requirements(tmp_path: Path) -> None:
    """The toolkit contains Clang's directory markers, headers, and libdevice without ptxas."""
    sources = create_toolkit_sources(tmp_path / "sources")
    toolkit_root = tmp_path / "toolkit"

    layout = check_cuda_native.create_cuda_toolkit_layout(toolkit_root, sources)

    assert layout.root_directory == toolkit_root
    assert (layout.include_directory / "cuda.h").is_symlink()
    assert (layout.include_directory / "crt").is_symlink()
    assert (layout.include_directory / "thrust").is_symlink()
    assert (layout.include_directory / "curand_kernel.h").is_symlink()
    assert not (layout.include_directory / "__init__.py").exists()
    assert layout.libdevice_directory.is_symlink()
    assert (toolkit_root / "bin").is_dir()
    assert tuple((toolkit_root / "bin").iterdir()) == ()


def test_build_clang_tidy_commands_cover_host_and_device_parsing(tmp_path: Path) -> None:
    """Command construction uses the required language modes, includes, and absolute paths."""
    repository_root = tmp_path / "repository"
    create_repository(repository_root)
    inventory = check_cuda_native.discover_repository_inventory(repository_root)
    toolkit_root = tmp_path / "toolkit"
    toolkit_root.mkdir()
    toolkit_layout = check_cuda_native.CudaToolkitLayout(
        root_directory=toolkit_root,
        include_directory=toolkit_root / "include",
        libdevice_directory=toolkit_root / "nvvm/libdevice",
    )
    generated_include_directory = tmp_path / "generated"
    arguments = check_cuda_native.CudaNativeCheckArguments(
        repository_root=repository_root,
        clang_tidy_executable="clang-tidy",
        gpu_architecture=check_cuda_native.CudaGpuArchitecture.VOLTA,
    )

    commands = check_cuda_native.build_clang_tidy_commands(
        arguments,
        toolkit_layout,
        generated_include_directory,
        inventory,
    )

    assert len(commands) == 4
    for command in commands:
        assert command.source_path.is_absolute()
        assert command.arguments[1] == str(command.source_path)
        assert f"--config-file={repository_root / '.clang-tidy'}" in command.arguments
        assert "--warnings-as-errors=*" in command.arguments
        if command.source_path.suffix == ".cu":
            assert "-std=c++17" in command.arguments
            assert f"--cuda-path={toolkit_root}" in command.arguments
            assert "--cuda-gpu-arch=sm_70" in command.arguments
            assert "--cuda-device-only" not in command.arguments
            assert "--cuda-host-only" not in command.arguments
        else:
            assert "-std=c++20" in command.arguments
            assert f"-I{repository_root / 'native/cuda-driver'}" in command.arguments
            assert f"-I{generated_include_directory}" in command.arguments
            assert str(repository_root / "vendor/openxla") in command.arguments


def test_generated_include_placeholders_cover_native_build_outputs(tmp_path: Path) -> None:
    """Native lint supplies every generated PTX and artifact-identity include."""
    generated_include_directory = tmp_path / "generated"

    check_cuda_native.create_placeholder_includes(generated_include_directory)

    expected_file_names = {placeholder.file_name for placeholder in check_cuda_native.GENERATED_INCLUDE_PLACEHOLDERS}
    observed_file_names = {path.name for path in generated_include_directory.iterdir()}
    assert observed_file_names == expected_file_names
    assert "kMinimumComputeCapabilityMinor" in (
        generated_include_directory / "firth_components_artifact_identity.inc"
    ).read_text(encoding="utf-8")
    assert "kMinimumComputeCapabilityMinor" in (
        generated_include_directory / "packed8_artifact_identity.inc"
    ).read_text(encoding="utf-8")


def test_run_tool_propagates_clang_tidy_failure(tmp_path: Path) -> None:
    """Any failing translation unit makes the static-analysis command fail."""
    repository_root = tmp_path / "repository"
    create_repository(repository_root)
    sources = create_toolkit_sources(tmp_path / "sources")
    arguments = check_cuda_native.CudaNativeCheckArguments(
        repository_root=repository_root,
        clang_tidy_executable="clang-tidy",
        gpu_architecture=check_cuda_native.CudaGpuArchitecture.VOLTA,
    )
    completed_processes = (
        subprocess.CompletedProcess(args=(), returncode=0),
        subprocess.CompletedProcess(args=(), returncode=0),
        subprocess.CompletedProcess(args=(), returncode=1),
        subprocess.CompletedProcess(args=(), returncode=0),
        subprocess.CompletedProcess(args=(), returncode=0),
    )

    with (
        unittest.mock.patch.object(check_cuda_native, "resolve_cuda_toolkit_sources", return_value=sources),
        unittest.mock.patch.object(
            check_cuda_native.subprocess,
            "run",
            side_effect=completed_processes,
        ) as subprocess_run,
    ):
        exit_code = check_cuda_native.run_tool(arguments)

    assert exit_code == 1
    assert subprocess_run.call_count == 5


def test_run_tool_stops_when_clang_tidy_configuration_is_invalid(tmp_path: Path) -> None:
    """Invalid Clang-Tidy configuration prevents source analysis."""
    repository_root = tmp_path / "repository"
    create_repository(repository_root)
    sources = create_toolkit_sources(tmp_path / "sources")
    arguments = check_cuda_native.CudaNativeCheckArguments(
        repository_root=repository_root,
        clang_tidy_executable="clang-tidy",
        gpu_architecture=check_cuda_native.CudaGpuArchitecture.VOLTA,
    )

    with (
        unittest.mock.patch.object(check_cuda_native, "resolve_cuda_toolkit_sources", return_value=sources),
        unittest.mock.patch.object(
            check_cuda_native.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(args=(), returncode=1),
        ) as subprocess_run,
    ):
        exit_code = check_cuda_native.run_tool(arguments)

    assert exit_code == 1
    assert subprocess_run.call_count == 1
