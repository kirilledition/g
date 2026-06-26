from __future__ import annotations

import textwrap
import typing

from tooling.debug import check_justfile

if typing.TYPE_CHECKING:
    from pathlib import Path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def violation_kinds(
    violations: tuple[check_justfile.JustfileViolation, ...],
) -> set[check_justfile.JustfileViolationKind]:
    return {violation.kind for violation in violations}


def test_justfile_check_accepts_config_backed_recipes_and_documentation(tmp_path: Path) -> None:
    justfile_path = tmp_path / "Justfile"
    documentation_path = tmp_path / "documentation" / "development" / "justfile.md"
    write_text(
        justfile_path,
        """
        set shell := ["bash", "-cu"]

        default: help

        help:
            @just --list --unsorted

        bench-bgen-reader *overrides:
            uv run --no-sync python -m tooling.cli.benchmark_bgen_reader --config-name bench_bgen_reader {{ overrides }}
        """,
    )
    write_text(
        documentation_path,
        """
        # Commands

        Run `just bench-bgen-reader`.
        """,
    )

    violations = check_justfile.collect_justfile_violations(
        check_justfile.JustfileCheckArguments(
            justfile_path=justfile_path,
            documentation_paths=(documentation_path,),
        )
    )

    assert violations == ()


def test_justfile_check_rejects_inline_hydra_override_sprawl(tmp_path: Path) -> None:
    justfile_path = tmp_path / "Justfile"
    write_text(
        justfile_path,
        """
        profile-chr10-binary-gpu-full:
            uv run --no-sync python -m tooling.cli.profile_regenie2_deep \
              machine=landau_gpu dataset=chr10_local tool.workload_keys=[binary_gpu] tool.enable_memray=true
        """,
    )

    violations = check_justfile.collect_justfile_violations(
        check_justfile.JustfileCheckArguments(justfile_path=justfile_path, documentation_paths=())
    )

    assert check_justfile.JustfileViolationKind.INLINE_HYDRA_OVERRIDES in violation_kinds(violations)
    assert check_justfile.JustfileViolationKind.MISSING_CONFIG_NAME in violation_kinds(violations)


def test_justfile_check_rejects_legacy_names_stale_flags_and_bad_prefixes(tmp_path: Path) -> None:
    justfile_path = tmp_path / "Justfile"
    write_text(
        justfile_path,
        """
        regenie2-binary-gpu:
            uv run g regenie --g-device gpu

        misc-run:
            echo ok
        """,
    )

    violations = check_justfile.collect_justfile_violations(
        check_justfile.JustfileCheckArguments(justfile_path=justfile_path, documentation_paths=())
    )
    kinds = violation_kinds(violations)

    assert check_justfile.JustfileViolationKind.DEPRECATED_RECIPE in kinds
    assert check_justfile.JustfileViolationKind.INVALID_RECIPE_PREFIX in kinds
    assert check_justfile.JustfileViolationKind.STALE_G_FLAG in kinds


def test_justfile_check_rejects_stale_documented_recipe_references(tmp_path: Path) -> None:
    justfile_path = tmp_path / "Justfile"
    documentation_path = tmp_path / "documentation" / "development" / "justfile.md"
    write_text(
        justfile_path,
        """
        default: help

        help:
            @just --list --unsorted
        """,
    )
    write_text(
        documentation_path,
        """
        Run `just benchmark-regenie2-binary-hot-gpu`.
        Run `just missing-new-recipe`.
        """,
    )

    violations = check_justfile.collect_justfile_violations(
        check_justfile.JustfileCheckArguments(
            justfile_path=justfile_path,
            documentation_paths=(documentation_path,),
        )
    )
    kinds = violation_kinds(violations)

    assert check_justfile.JustfileViolationKind.DEPRECATED_REFERENCE in kinds
    assert check_justfile.JustfileViolationKind.DOCUMENTED_RECIPE_MISSING in kinds
