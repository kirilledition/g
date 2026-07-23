"""Tests for the PyO3 registration and stub consistency checker."""

from __future__ import annotations

import tomllib
from pathlib import Path

from tooling.debug import check_pyo3_stub

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_nested_cli_stub_matches_registered_rust_submodule(tmp_path: Path) -> None:
    """A stub namespace is matched against the Rust submodule that registers it."""
    binding_directory = tmp_path / "binding"
    binding_directory.mkdir()
    registration_path = binding_directory / "mod.rs"
    registration_path.write_text(
        """
pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let module_name = module.name()?;
    let full_name = format!("{}.cli", module_name.to_str()?);
    let submodule = PyModule::new(module.py(), &full_name)?;
    cli::register_module(&submodule)?;
    module.add_submodule(&submodule)?;
    Ok(())
}
""",
        encoding="utf-8",
    )
    (binding_directory / "cli.rs").write_text(
        """
pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunResult>()?;
    module.add_function(wrap_pyfunction!(run, module)?)?;
    Ok(())
}
""",
        encoding="utf-8",
    )
    stub_path = tmp_path / "_core.pyi"
    stub_path.write_text(
        """
class cli:
    class NativeCliRunResult: ...

    @staticmethod
    def run(arguments: list[str]) -> NativeCliRunResult: ...
""",
        encoding="utf-8",
    )

    rust_exports = check_pyo3_stub.read_rust_exports(registration_path)
    stub_exports = check_pyo3_stub.read_stub_exports(stub_path, rust_exports.module_names)

    assert rust_exports == check_pyo3_stub.ExportSurface(
        classes=frozenset({"cli.NativeCliRunResult"}),
        functions=frozenset({"cli.run"}),
        module_names=frozenset({"cli"}),
    )
    assert stub_exports == rust_exports


def test_stub_parser_retains_real_top_level_exports(tmp_path: Path) -> None:
    """Only registered namespace wrappers are excluded from class exports."""
    stub_path = tmp_path / "_core.pyi"
    stub_path.write_text(
        """
class cli:
    class NativeCliRunResult: ...

class UnexpectedRootClass: ...

def unexpected_root_function() -> None: ...
""",
        encoding="utf-8",
    )

    stub_exports = check_pyo3_stub.read_stub_exports(stub_path, frozenset({"cli"}))

    assert stub_exports.classes == frozenset({"cli.NativeCliRunResult", "UnexpectedRootClass"})
    assert stub_exports.functions == frozenset({"unexpected_root_function"})


def test_repository_production_binding_surface_is_exact_and_private_support_is_opt_in() -> None:
    rust_exports = check_pyo3_stub.read_rust_exports(REPOSITORY_ROOT / "src/binding/mod.rs")

    assert rust_exports == check_pyo3_stub.ExportSurface(
        classes=frozenset({"cli.NativeCliRunResult"}),
        functions=frozenset({"cli.run"}),
        module_names=frozenset({"cli"}),
    )
    cargo_manifest = tomllib.loads((REPOSITORY_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    genotype_cuda_manifest = tomllib.loads(
        (REPOSITORY_ROOT / "crates/genotype-cuda/Cargo.toml").read_text(encoding="utf-8")
    )
    project_configuration = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert cargo_manifest["features"]["default"] == []
    assert cargo_manifest["features"]["private-test-support"] == ["g-genotype-cuda/private-test-support"]
    assert genotype_cuda_manifest["features"] == {
        "default": [],
        "private-test-support": [],
    }
    assert project_configuration["tool"]["maturin"]["features"] == ["extension-module"]
    justfile_source = (REPOSITORY_ROOT / "Justfile").read_text(encoding="utf-8")
    assert "--features extension-module,private-test-support" in justfile_source

    binding_source = (REPOSITORY_ROOT / "src/binding/mod.rs").read_text(encoding="utf-8")
    assert '#[cfg(feature = "private-test-support")]\nmod test_support;' in binding_source
    assert '#[cfg(feature = "private-test-support")]\n    test_support::register_module(module)?;' in binding_source
    assert "_testing" not in (REPOSITORY_ROOT / "src/g/_core.pyi").read_text(encoding="utf-8")
