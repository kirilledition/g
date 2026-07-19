#!/usr/bin/env python3
"""Run Clang-Tidy over the maintained CUDA and C++ native sources."""

from __future__ import annotations

import enum
import importlib.metadata
import re
import subprocess
import sys
import tempfile
import typing
from dataclasses import dataclass
from pathlib import Path

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf


class CudaGpuArchitecture(enum.StrEnum):
    """Supported CUDA GPU architectures for static analysis."""

    VOLTA = "sm_70"


class NativeSourceLanguage(enum.StrEnum):
    """Native translation-unit languages checked by Clang-Tidy."""

    CUDA = "cuda"
    CPLUSPLUS = "c++"


@dataclass(frozen=True)
class CudaDistributionSpec:
    """One locked Python distribution supplying CUDA toolkit files.

    Attributes:
        name: Python distribution name.
        expected_version: Exact version required by the lint environment.

    """

    name: str
    expected_version: str


@dataclass(frozen=True)
class CudaToolkitSources:
    """Installed wheel paths used to assemble a CUDA toolkit view.

    Attributes:
        runtime_include_directory: CUDA runtime header directory.
        compiler_include_directory: CUDA compiler header directory.
        core_compute_libraries_include_directory: CCCL header directory.
        random_include_directory: cuRAND header directory.
        libdevice_directory: NVVM libdevice bitcode directory.

    """

    runtime_include_directory: Path
    compiler_include_directory: Path
    core_compute_libraries_include_directory: Path
    random_include_directory: Path
    libdevice_directory: Path


@dataclass(frozen=True)
class CudaToolkitLayout:
    """Temporary CUDA toolkit layout consumed by Clang.

    Attributes:
        root_directory: Toolkit root passed through ``--cuda-path``.
        include_directory: Merged CUDA include directory.
        libdevice_directory: Linked NVVM libdevice directory.

    """

    root_directory: Path
    include_directory: Path
    libdevice_directory: Path


@dataclass(frozen=True)
class NativeTranslationUnit:
    """One maintained native translation unit.

    Attributes:
        relative_path: Repository-relative source path.
        language: Source language parsed by Clang.
        language_standard: C++ language standard argument.

    """

    relative_path: Path
    language: NativeSourceLanguage
    language_standard: str


@dataclass(frozen=True)
class NativeSourceInventory:
    """Discovered maintained native source inventory.

    Attributes:
        translation_units: CUDA and C++ translation units to analyze.
        headers: Headers analyzed transitively through those translation units.

    """

    translation_units: tuple[NativeTranslationUnit, ...]
    headers: tuple[Path, ...]


@dataclass(frozen=True)
class ClangTidyCommand:
    """One deterministic Clang-Tidy invocation.

    Attributes:
        source_path: Absolute translation-unit path.
        arguments: Complete subprocess argument vector.

    """

    source_path: Path
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class CudaNativeCheckArguments:
    """Resolved arguments for CUDA native static analysis.

    Attributes:
        repository_root: Repository root containing native sources and configuration.
        clang_tidy_executable: Clang-Tidy executable name or path.
        gpu_architecture: CUDA GPU architecture parsed by Clang.

    """

    repository_root: Path
    clang_tidy_executable: str
    gpu_architecture: CudaGpuArchitecture


class CudaNativeCheckError(RuntimeError):
    """Raised when the deterministic CUDA lint environment cannot be built."""


CUDA_RUNTIME_DISTRIBUTION = CudaDistributionSpec(
    name="nvidia-cuda-runtime-cu12",
    expected_version="12.9.79",
)
CUDA_COMPILER_DISTRIBUTION = CudaDistributionSpec(
    name="nvidia-cuda-nvcc-cu12",
    expected_version="12.9.86",
)
CUDA_CORE_COMPUTE_LIBRARIES_DISTRIBUTION = CudaDistributionSpec(
    name="nvidia-cuda-cccl-cu12",
    expected_version="12.9.27",
)
CUDA_RANDOM_DISTRIBUTION = CudaDistributionSpec(
    name="nvidia-curand-cu12",
    expected_version="10.3.10.19",
)

CUDA_RUNTIME_INCLUDE_PATH = Path("nvidia/cuda_runtime/include")
CUDA_COMPILER_INCLUDE_PATH = Path("nvidia/cuda_nvcc/include")
CUDA_CORE_COMPUTE_LIBRARIES_INCLUDE_PATH = Path("nvidia/cuda_cccl/include")
CUDA_RANDOM_INCLUDE_PATH = Path("nvidia/curand/include")
CUDA_LIBDEVICE_PATH = Path("nvidia/cuda_nvcc/nvvm/libdevice")

NATIVE_DIRECTORIES = (
    Path("crates/compute-cuda/native"),
    Path("crates/genotype-cuda/native"),
)
MAINTAINED_NATIVE_PATHS = (
    Path("crates/compute-cuda/native/firth_components_kernel.cu"),
    Path("crates/genotype-cuda/native/packed8_kernel.cu"),
    Path("crates/compute-cuda/native/firth_components_ffi.cc"),
    Path("crates/genotype-cuda/native/packed8_deflate_ffi.cc"),
    Path("crates/compute-cuda/native/cuda_driver_abi.h"),
    Path("crates/genotype-cuda/native/cuda_driver_abi.h"),
    Path("crates/genotype-cuda/native/nvcomp_abi.h"),
)
REQUIRED_TRANSITIVE_HEADERS = {
    Path("crates/compute-cuda/native/firth_components_ffi.cc"): ("cuda_driver_abi.h",),
    Path("crates/genotype-cuda/native/packed8_deflate_ffi.cc"): ("cuda_driver_abi.h", "nvcomp_abi.h"),
}
NATIVE_SOURCE_SUFFIXES = frozenset({".cc", ".cu", ".h"})
PLACEHOLDER_INCLUDE_NAMES = (
    "firth_components_kernel_ptx.inc",
    "packed8_kernel_ptx.inc",
)
JUSTFILE_NATIVE_SOURCES_PATTERN = re.compile(r"^cuda_native_sources := '([^']+)'$", re.MULTILINE)


def build_arguments_from_config(config: omegaconf.DictConfig) -> CudaNativeCheckArguments:
    """Build resolved CUDA native check arguments from Hydra configuration."""
    return CudaNativeCheckArguments(
        repository_root=Path(str(config.tool.repository_root)).resolve(),
        clang_tidy_executable=str(config.tool.clang_tidy_executable),
        gpu_architecture=CudaGpuArchitecture(str(config.tool.gpu_architecture)),
    )


def resolve_distribution_directory(
    distribution_spec: CudaDistributionSpec,
    relative_path: Path,
) -> Path:
    """Resolve and validate one locked CUDA wheel directory.

    Args:
        distribution_spec: Locked Python distribution requirement.
        relative_path: Distribution-relative directory path.

    Returns:
        Absolute artifact path.

    Raises:
        CudaNativeCheckError: If the distribution, version, or directory is invalid.

    """
    try:
        distribution = importlib.metadata.distribution(distribution_spec.name)
    except importlib.metadata.PackageNotFoundError as error:
        message = f"required CUDA lint distribution is not installed: {distribution_spec.name}"
        raise CudaNativeCheckError(message) from error

    if distribution.version != distribution_spec.expected_version:
        message = (
            f"CUDA lint distribution {distribution_spec.name} has version {distribution.version}; "
            f"expected {distribution_spec.expected_version}"
        )
        raise CudaNativeCheckError(message)

    directory_path = Path(str(distribution.locate_file(relative_path))).resolve()
    if not directory_path.is_dir():
        message = (
            f"CUDA lint distribution {distribution_spec.name} does not provide required directory: {relative_path}"
        )
        raise CudaNativeCheckError(message)
    return directory_path


def resolve_cuda_toolkit_sources() -> CudaToolkitSources:
    """Resolve all locked CUDA wheel paths needed by Clang."""
    return CudaToolkitSources(
        runtime_include_directory=resolve_distribution_directory(
            CUDA_RUNTIME_DISTRIBUTION,
            CUDA_RUNTIME_INCLUDE_PATH,
        ),
        compiler_include_directory=resolve_distribution_directory(
            CUDA_COMPILER_DISTRIBUTION,
            CUDA_COMPILER_INCLUDE_PATH,
        ),
        core_compute_libraries_include_directory=resolve_distribution_directory(
            CUDA_CORE_COMPUTE_LIBRARIES_DISTRIBUTION,
            CUDA_CORE_COMPUTE_LIBRARIES_INCLUDE_PATH,
        ),
        random_include_directory=resolve_distribution_directory(
            CUDA_RANDOM_DISTRIBUTION,
            CUDA_RANDOM_INCLUDE_PATH,
        ),
        libdevice_directory=resolve_distribution_directory(
            CUDA_COMPILER_DISTRIBUTION,
            CUDA_LIBDEVICE_PATH,
        ),
    )


def link_directory_entries(source_directory: Path, destination_directory: Path) -> None:
    """Link non-Python entries from one wheel directory into a merged directory."""
    for source_path in sorted(source_directory.iterdir(), key=lambda path: path.name):
        if source_path.name in {"__init__.py", "__pycache__"}:
            continue
        destination_path = destination_directory / source_path.name
        if destination_path.exists() or destination_path.is_symlink():
            message = f"CUDA toolkit include collision for {destination_path.name}"
            raise CudaNativeCheckError(message)
        destination_path.symlink_to(source_path.resolve(), target_is_directory=source_path.is_dir())


def create_cuda_toolkit_layout(root_directory: Path, sources: CudaToolkitSources) -> CudaToolkitLayout:
    """Assemble the temporary CUDA toolkit filesystem expected by Clang."""
    # Clang's CUDA installation detector requires the conventional bin directory
    # to exist even for syntax-only analysis. It does not require an executable.
    (root_directory / "bin").mkdir(parents=True)
    include_directory = root_directory / "include"
    include_directory.mkdir()
    for source_directory in (
        sources.runtime_include_directory,
        sources.compiler_include_directory,
        sources.core_compute_libraries_include_directory,
        sources.random_include_directory,
    ):
        link_directory_entries(source_directory, include_directory)

    nvvm_directory = root_directory / "nvvm"
    nvvm_directory.mkdir()
    libdevice_directory = nvvm_directory / "libdevice"
    libdevice_directory.symlink_to(sources.libdevice_directory.resolve(), target_is_directory=True)

    return CudaToolkitLayout(
        root_directory=root_directory,
        include_directory=include_directory,
        libdevice_directory=libdevice_directory,
    )


def create_placeholder_includes(directory: Path) -> None:
    """Write parse-only string literals for build-script-generated PTX includes."""
    directory.mkdir(parents=True)
    for include_name in PLACEHOLDER_INCLUDE_NAMES:
        (directory / include_name).write_text('""\n', encoding="utf-8")


def build_native_translation_unit(relative_path: Path) -> NativeTranslationUnit:
    """Build the language-specific configuration for one maintained translation unit."""
    if relative_path.suffix == ".cu":
        return NativeTranslationUnit(
            relative_path=relative_path,
            language=NativeSourceLanguage.CUDA,
            language_standard="c++17",
        )
    if relative_path.suffix == ".cc":
        return NativeTranslationUnit(
            relative_path=relative_path,
            language=NativeSourceLanguage.CPLUSPLUS,
            language_standard="c++20",
        )
    message = f"unsupported native translation unit: {relative_path}"
    raise CudaNativeCheckError(message)


def discover_justfile_native_paths(repository_root: Path) -> frozenset[Path]:
    """Read the formatter's maintained native allowlist from the Justfile."""
    justfile_path = repository_root / "Justfile"
    try:
        justfile_text = justfile_path.read_text(encoding="utf-8")
    except OSError as error:
        message = f"could not read native source allowlist from {justfile_path}: {error}"
        raise CudaNativeCheckError(message) from error
    assignment_match = JUSTFILE_NATIVE_SOURCES_PATTERN.search(justfile_text)
    if assignment_match is None:
        message = f"Justfile does not define the maintained cuda_native_sources allowlist: {justfile_path}"
        raise CudaNativeCheckError(message)
    return frozenset(Path(value) for value in assignment_match.group(1).split())


def validate_transitive_header_includes(repository_root: Path) -> None:
    """Require every maintained header to be parsed through a translation unit."""
    missing_includes: list[str] = []
    for relative_source_path, header_names in REQUIRED_TRANSITIVE_HEADERS.items():
        source_text = (repository_root / relative_source_path).read_text(encoding="utf-8")
        for header_name in header_names:
            if f'#include "{header_name}"' not in source_text:
                missing_includes.append(f"  {relative_source_path}: {header_name}")
    if missing_includes:
        diagnostics = ["maintained native headers are not included by their lint translation units:"]
        diagnostics.extend(missing_includes)
        raise CudaNativeCheckError("\n".join(diagnostics))


def discover_repository_inventory(repository_root: Path) -> NativeSourceInventory:
    """Discover native sources recursively and require the exact maintained allowlist."""
    if not repository_root.is_dir():
        message = f"repository root does not exist: {repository_root}"
        raise CudaNativeCheckError(message)
    clang_tidy_configuration_path = repository_root / ".clang-tidy"
    if not clang_tidy_configuration_path.is_file():
        message = f"Clang-Tidy configuration does not exist: {clang_tidy_configuration_path}"
        raise CudaNativeCheckError(message)

    discovered_paths: set[Path] = set()
    for relative_directory in NATIVE_DIRECTORIES:
        native_directory = repository_root / relative_directory
        if not native_directory.is_dir():
            message = f"maintained native directory does not exist: {native_directory}"
            raise CudaNativeCheckError(message)
        for native_path in native_directory.rglob("*"):
            if native_path.is_file() and native_path.suffix in NATIVE_SOURCE_SUFFIXES:
                discovered_paths.add(native_path.relative_to(repository_root))

    expected_paths = frozenset(MAINTAINED_NATIVE_PATHS)
    formatter_paths = discover_justfile_native_paths(repository_root)
    if formatter_paths != expected_paths:
        message = "Justfile cuda_native_sources does not match the Clang-Tidy maintained source allowlist"
        raise CudaNativeCheckError(message)
    missing_paths = sorted(expected_paths.difference(discovered_paths))
    unexpected_paths = sorted(discovered_paths.difference(expected_paths))
    if missing_paths or unexpected_paths:
        diagnostics = ["maintained CUDA native inventory does not match the lint allowlist:"]
        diagnostics.extend(f"  missing: {path}" for path in missing_paths)
        diagnostics.extend(f"  unexpected: {path}" for path in unexpected_paths)
        raise CudaNativeCheckError("\n".join(diagnostics))
    validate_transitive_header_includes(repository_root)

    translation_units = tuple(
        build_native_translation_unit(path) for path in MAINTAINED_NATIVE_PATHS if path.suffix in {".cc", ".cu"}
    )
    headers = tuple(path for path in MAINTAINED_NATIVE_PATHS if path.suffix == ".h")
    return NativeSourceInventory(
        translation_units=translation_units,
        headers=headers,
    )


def build_header_filter(repository_root: Path) -> str:
    """Build a regex matching only maintained native header directories."""
    escaped_directories = (
        re.escape(str((repository_root / relative_directory).resolve())) for relative_directory in NATIVE_DIRECTORIES
    )
    return "(?:" + "|".join(f"{directory}/.*" for directory in escaped_directories) + ")"


def build_clang_tidy_commands(
    arguments: CudaNativeCheckArguments,
    toolkit_layout: CudaToolkitLayout,
    generated_include_directory: Path,
    inventory: NativeSourceInventory,
) -> tuple[ClangTidyCommand, ...]:
    """Build deterministic Clang-Tidy commands for all native translation units."""
    repository_root = arguments.repository_root.resolve()
    configuration_path = repository_root / ".clang-tidy"
    header_filter = build_header_filter(repository_root)
    vendor_include_directory = repository_root / "vendor/openxla"
    commands: list[ClangTidyCommand] = []
    for translation_unit in inventory.translation_units:
        source_path = (repository_root / translation_unit.relative_path).resolve()
        common_arguments = (
            arguments.clang_tidy_executable,
            str(source_path),
            f"--config-file={configuration_path}",
            f"--header-filter={header_filter}",
            "--warnings-as-errors=*",
            "--quiet",
            "--",
        )
        if translation_unit.language is NativeSourceLanguage.CUDA:
            compiler_arguments = (
                "-x",
                NativeSourceLanguage.CUDA.value,
                f"-std={translation_unit.language_standard}",
                f"--cuda-path={toolkit_layout.root_directory}",
                f"--cuda-gpu-arch={arguments.gpu_architecture.value}",
                "-Wno-unknown-cuda-version",
                "-Wall",
                "-Wextra",
            )
        else:
            native_include_directory = source_path.parent
            compiler_arguments = (
                "-x",
                NativeSourceLanguage.CPLUSPLUS.value,
                f"-std={translation_unit.language_standard}",
                "-DNDEBUG",
                f"-I{native_include_directory}",
                f"-I{generated_include_directory}",
                "-isystem",
                str(vendor_include_directory),
                "-Wall",
                "-Wextra",
            )
        commands.append(
            ClangTidyCommand(
                source_path=source_path,
                arguments=(*common_arguments, *compiler_arguments),
            )
        )
    return tuple(commands)


def run_clang_tidy_commands(commands: tuple[ClangTidyCommand, ...]) -> int:
    """Run every Clang-Tidy command and return a failing status if any command fails."""
    failed_source_paths: list[Path] = []
    for command in commands:
        print(f"Clang-Tidy: {command.source_path}")
        completed_process = subprocess.run(command.arguments, check=False)
        if completed_process.returncode != 0:
            failed_source_paths.append(command.source_path)
    if failed_source_paths:
        print("CUDA native Clang-Tidy failures:", file=sys.stderr)
        for source_path in failed_source_paths:
            print(f"  {source_path}", file=sys.stderr)
        return 1
    return 0


def verify_clang_tidy_configuration(arguments: CudaNativeCheckArguments) -> int:
    """Verify the checked-in Clang-Tidy configuration before analyzing source."""
    configuration_path = arguments.repository_root.resolve() / ".clang-tidy"
    completed_process = subprocess.run(
        (
            arguments.clang_tidy_executable,
            f"--config-file={configuration_path}",
            "--verify-config",
        ),
        check=False,
    )
    return completed_process.returncode


def run_tool(arguments: CudaNativeCheckArguments) -> int:
    """Run CUDA-aware Clang-Tidy without requiring a system CUDA toolkit or GPU."""
    try:
        inventory = discover_repository_inventory(arguments.repository_root)
        if verify_clang_tidy_configuration(arguments):
            print("CUDA native Clang-Tidy configuration verification failed.", file=sys.stderr)
            return 1
        toolkit_sources = resolve_cuda_toolkit_sources()
        with tempfile.TemporaryDirectory(prefix="g-cuda-native-lint-") as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            toolkit_layout = create_cuda_toolkit_layout(temporary_directory / "cuda", toolkit_sources)
            generated_include_directory = temporary_directory / "generated"
            create_placeholder_includes(generated_include_directory)
            commands = build_clang_tidy_commands(
                arguments,
                toolkit_layout,
                generated_include_directory,
                inventory,
            )
            exit_code = run_clang_tidy_commands(commands)
    except (CudaNativeCheckError, OSError) as error:
        print(f"CUDA native Clang-Tidy setup failed: {error}", file=sys.stderr)
        return 1

    if exit_code:
        return exit_code
    print(
        "CUDA native Clang-Tidy passed for "
        f"{len(inventory.translation_units)} translation units and {len(inventory.headers)} transitive headers."
    )
    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="debug_check_cuda_native")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run CUDA native static analysis from Hydra configuration."""
    exit_code = run_tool(build_arguments_from_config(config))
    if exit_code:
        raise SystemExit(exit_code)


def main() -> int:
    """Run CUDA native static analysis from its default configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
