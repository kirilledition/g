"""Cross-language invariants for CUDA typed-XLA target names."""

from __future__ import annotations

import re
from pathlib import Path

from g.compute import cuda_ffi

REPOSITORY_ROOT = Path(__file__).parents[2]


def read_rust_string_constant(path: Path, name: str) -> str:
    """Read one crate-owned public Rust string constant."""
    source = path.read_text(encoding="utf-8")
    match = re.search(rf'pub const {re.escape(name)}: &str = "([^"]+)";', source)
    if match is None:
        raise AssertionError(f"Missing Rust string constant {name} in {path}.")
    return match.group(1)


def test_python_cuda_ffi_targets_match_crate_owned_rust_contracts() -> None:
    """Keep call sites and native registration on the exact same target names."""
    assert (
        read_rust_string_constant(
            REPOSITORY_ROOT / "crates/compute-cuda/src/api.rs",
            "FIRTH_COMPONENTS_FFI_TARGET",
        )
        == cuda_ffi.FIRTH_COMPONENTS_FFI_TARGET
    )
    assert (
        read_rust_string_constant(
            REPOSITORY_ROOT / "crates/genotype-cuda/src/api.rs",
            "PACKED8_DEFLATE_FFI_TARGET",
        )
        == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    )
