#!/usr/bin/env python3
"""Download and prepare 1000 Genomes chromosome 22 benchmark data."""

from __future__ import annotations

import shutil
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path

import hydra

from tooling.common import downloads as tooling_downloads
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.data import registry as data_registry

if typing.TYPE_CHECKING:
    import omegaconf

TOY_VARIANT_COUNT = 5_000
DEFAULT_DATA_DIRECTORY = Path("data")


@dataclass(frozen=True)
class DatasetPaths:
    """Filesystem paths used for Phase 0 dataset preparation."""

    data_directory: Path
    full_dataset_prefix: Path
    toy_dataset_prefix: Path


def ensure_command_available(command_name: str) -> None:
    """Raise an error when a required external command is unavailable.

    Args:
        command_name: Executable name that must be present on PATH.

    Raises:
        RuntimeError: The executable is not available.

    """
    if shutil.which(command_name) is None:
        raise RuntimeError(f"Required command '{command_name}' is not available on PATH.")


def download_registry_entry(
    *,
    registry_entry: tooling_downloads.DownloadRegistryEntry,
    destination_path: Path,
) -> None:
    """Retrieve one registered source file through Pooch.

    Args:
        registry_entry: Named source file registry entry.
        destination_path: Local path where the file will be written.

    """
    if destination_path.name != registry_entry.file_name:
        message = (
            f"Registry entry {registry_entry.name} expects {registry_entry.file_name}, got {destination_path.name}."
        )
        raise ValueError(message)
    tooling_downloads.retrieve_registry_entry(
        registry_entry=registry_entry,
        destination_directory=destination_path.parent,
    )


def run_command(command_arguments: list[str]) -> None:
    """Run an external command and raise a descriptive error on failure.

    Args:
        command_arguments: Command line arguments.

    Raises:
        RuntimeError: The command fails.

    """
    print(f"Running: {' '.join(command_arguments)}")
    completed_process = subprocess.run(
        command_arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode == 0:
        return

    raise RuntimeError(
        "Command failed\n"
        f"Command: {' '.join(command_arguments)}\n"
        f"Exit code: {completed_process.returncode}\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


def decompress_zstandard_file(compressed_path: Path) -> Path:
    """Decompress a `.zst` file into the same directory.

    Args:
        compressed_path: Compressed input path.

    Returns:
        The decompressed output path.

    """
    if compressed_path.suffix != ".zst":
        return compressed_path

    output_path = compressed_path.with_suffix("")
    if output_path.exists():
        print(f"Reusing {output_path}")
        return output_path

    run_command(
        [
            "zstd",
            "--decompress",
            "--keep",
            "--force",
            str(compressed_path),
            "-o",
            str(output_path),
        ]
    )
    return output_path


def download_source_files(dataset_paths: DatasetPaths) -> None:
    """Download the raw chromosome 22 source files.

    Args:
        dataset_paths: Dataset paths configuration.

    """
    dataset_paths.data_directory.mkdir(exist_ok=True)
    for suffix, registry_entry in data_registry.CHROMOSOME_22_SOURCE_REGISTRY.items():
        destination_path = dataset_paths.full_dataset_prefix.with_suffix(f".{suffix}")
        download_registry_entry(registry_entry=registry_entry, destination_path=destination_path)


def create_plink_binary_files(dataset_paths: DatasetPaths) -> None:
    """Create PLINK BED/BIM/FAM files from downloaded PGEN sources.

    Args:
        dataset_paths: Dataset paths configuration.

    """
    bed_path = dataset_paths.full_dataset_prefix.with_suffix(".bed")
    if bed_path.exists():
        print(f"Reusing {bed_path}")
        return

    decompress_zstandard_file(dataset_paths.full_dataset_prefix.with_suffix(".pgen.zst"))
    decompress_zstandard_file(dataset_paths.full_dataset_prefix.with_suffix(".pvar.zst"))

    run_command(
        [
            "plink2",
            "--pfile",
            str(dataset_paths.full_dataset_prefix),
            "--max-alleles",
            "2",
            "--nonfounders",
            "--mac",
            "5",
            "--make-bed",
            "--out",
            str(dataset_paths.full_dataset_prefix),
        ]
    )


def create_bgen_files(dataset_paths: DatasetPaths) -> None:
    """Export BGEN and sample files for Regenie step 2.

    Args:
        dataset_paths: Dataset paths configuration.

    """
    bgen_path = dataset_paths.full_dataset_prefix.with_suffix(".bgen")
    sample_path = dataset_paths.full_dataset_prefix.with_suffix(".sample")
    if bgen_path.exists() and sample_path.exists():
        print(f"Reusing {bgen_path} and {sample_path}")
        return

    run_command(
        [
            "plink2",
            "--bfile",
            str(dataset_paths.full_dataset_prefix),
            "--export",
            "bgen-1.2",
            "ref-first",
            "bits=8",
            "--out",
            str(dataset_paths.full_dataset_prefix),
        ]
    )


def create_toy_slice(dataset_paths: DatasetPaths) -> None:
    """Create a smaller BED dataset for fast tests.

    Args:
        dataset_paths: Dataset paths configuration.

    """
    toy_bed_path = dataset_paths.toy_dataset_prefix.with_suffix(".bed")
    if toy_bed_path.exists():
        print(f"Reusing {toy_bed_path}")
        return

    run_command(
        [
            "plink2",
            "--bfile",
            str(dataset_paths.full_dataset_prefix),
            "--thin-count",
            str(TOY_VARIANT_COUNT),
            "--make-bed",
            "--out",
            str(dataset_paths.toy_dataset_prefix),
        ]
    )


def build_dataset_paths() -> DatasetPaths:
    """Construct the standard Phase 0 dataset paths."""
    data_directory = DEFAULT_DATA_DIRECTORY
    return DatasetPaths(
        data_directory=data_directory,
        full_dataset_prefix=data_directory / "1kg_chr22_full",
        toy_dataset_prefix=data_directory / "1kg_chr22_toy",
    )


@dataclass(frozen=True)
class FetchArguments:
    """Resolved parameters for dataset fetching."""

    data_directory: Path


def build_dataset_paths_with_data_directory(data_directory: Path) -> DatasetPaths:
    """Construct dataset paths under a non-default data root."""
    return DatasetPaths(
        data_directory=data_directory,
        full_dataset_prefix=data_directory / "1kg_chr22_full",
        toy_dataset_prefix=data_directory / "1kg_chr22_toy",
    )


def run_tool(arguments: FetchArguments) -> None:
    """Fetch and prepare benchmark data under the configured directory."""
    ensure_command_available("plink2")
    ensure_command_available("zstd")

    dataset_paths = build_dataset_paths_with_data_directory(arguments.data_directory)
    download_source_files(dataset_paths)
    create_plink_binary_files(dataset_paths)
    create_bgen_files(dataset_paths)
    create_toy_slice(dataset_paths)
    print("Data fetching and preparation complete.")


def build_arguments_from_config(config: omegaconf.DictConfig) -> FetchArguments:
    """Resolve fetch parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return FetchArguments(
        data_directory=Path(str(tool_values["data_directory"])),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="data_fetch")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run dataset fetching from Hydra configuration."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Download and prepare benchmark data from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


def main_hydra() -> None:
    """Hydra entrypoint for direct module execution."""
    hydra_main()


if __name__ == "__main__":
    main()
