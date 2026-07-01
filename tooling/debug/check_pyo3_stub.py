#!/usr/bin/env python3
"""Verify that the Python stub mirrors `_core` registrations."""

from __future__ import annotations

import re
import sys
import typing
from pathlib import Path

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf

RUST_EXPORT_FILES = (
    Path("src/python/association_backend.rs"),
    Path("src/python/callback_diagnostics.rs"),
    Path("src/python/callback_progress.rs"),
    Path("src/python/callback_queue.rs"),
    Path("src/python/callback_runtime_resources.rs"),
    Path("src/python/callback_summary.rs"),
    Path("src/python/jax_runtime.rs"),
    Path("src/python/logging.rs"),
    Path("src/python/mod.rs"),
    Path("src/python/config/mod.rs"),
    Path("src/python/genotype.rs"),
    Path("src/python/host_policy.rs"),
    Path("src/python/output.rs"),
    Path("src/python/preflight.rs"),
    Path("src/python/preparation.rs"),
    Path("src/python/prediction_sources.rs"),
    Path("src/python/run_events.rs"),
    Path("src/python/run_engine.rs"),
    Path("src/python/run_metadata.rs"),
    Path("src/python/runtime.rs"),
    Path("src/python/runtime_paths.rs"),
    Path("src/python/runtime_policy.rs"),
    Path("src/python/runtime_state.rs"),
    Path("src/python/sample_alignment.rs"),
    Path("src/python/schedule.rs"),
    Path("src/python/shutdown.rs"),
    Path("src/python/telemetry_policy.rs"),
    Path("src/python/timing.rs"),
    Path("src/python/trusted_validation.rs"),
)
STUB_FILE = Path("src/g/_core.pyi")

CLASS_PATTERN = re.compile(r"add_class::<([A-Za-z_][A-Za-z0-9_]*)>")
FUNC_PATTERN = re.compile(r"add_function\(wrap_pyfunction!\(\s*([A-Za-z_][A-Za-z0-9_]*)")
ALLOWED_STUB_ONLY_CLASSES = {"ChunkStatsComputeArrays"}


def read_rust_exports(paths: tuple[Path, ...]) -> tuple[set[str], set[str]]:
    """Read exported PyO3 classes and functions from Rust registration files."""
    rust_classes: set[str] = set()
    rust_functions: set[str] = set()

    for path in paths:
        text = path.read_text()
        rust_classes.update(CLASS_PATTERN.findall(text))
        rust_functions.update(FUNC_PATTERN.findall(text))

    return rust_classes, rust_functions


def read_stub_exports(stub_path: Path) -> tuple[set[str], set[str]]:
    """Read declared classes and functions from the `_core` Python stub."""
    stub_classes: set[str] = set()
    stub_functions: set[str] = set()

    for line in stub_path.read_text().splitlines():
        if line.startswith("class "):
            stub_classes.add(line.removeprefix("class ").split("(", maxsplit=1)[0].removesuffix(":"))
        if line.startswith("def "):
            stub_functions.add(line.removeprefix("def ").split("(", maxsplit=1)[0])

    return stub_classes, stub_functions


def format_list(values: set[str]) -> str:
    """Format a set of names as a human-readable indented list."""
    if not values:
        return "none"
    return "\n".join(f"  - {value}" for value in sorted(values))


def run_tool() -> int:
    """Verify that Rust `_core` registrations match the Python stub."""
    rust_classes, rust_functions = read_rust_exports(RUST_EXPORT_FILES)
    stub_classes, stub_functions = read_stub_exports(STUB_FILE)

    missing_classes = sorted(rust_classes.difference(stub_classes))
    extra_classes = sorted(stub_classes.difference(rust_classes).difference(ALLOWED_STUB_ONLY_CLASSES))
    missing_functions = sorted(rust_functions.difference(stub_functions))
    extra_functions = sorted(stub_functions.difference(rust_functions))

    if missing_classes or extra_classes or missing_functions or extra_functions:
        print("Mismatch between Rust `_core` registrations and `src/g/_core.pyi`:")
        print(f"  Rust-only classes ({len(missing_classes)}):")
        print(format_list(set(missing_classes)))
        print(f"  Stub-only classes ({len(extra_classes)}):")
        print(format_list(set(extra_classes)))
        print(f"  Rust-only functions ({len(missing_functions)}):")
        print(format_list(set(missing_functions)))
        print(f"  Stub-only functions ({len(extra_functions)}):")
        print(format_list(set(extra_functions)))
        return 1

    print("`src/g/_core.pyi` matches Rust `_core` registrations.")
    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="debug_check_pyo3_stub")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the stub checker from Hydra configuration."""
    del config
    exit_code = run_tool()
    if exit_code:
        raise SystemExit(exit_code)


def main() -> int:
    """Run the stub checker from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
