from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from g.jax_setup import resolve_jax_compilation_cache_directory


def test_resolve_jax_cache_uses_jax_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure JAX_COMPILATION_CACHE_DIR takes highest precedence."""
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/custom/jax/cache")
    monkeypatch.setenv("XDG_CACHE_HOME", "/custom/xdg/cache")

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/custom/jax/cache")


def test_resolve_jax_cache_expands_jax_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure JAX_COMPILATION_CACHE_DIR is expanded."""
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "~/custom/jax/cache")
    monkeypatch.setenv("HOME", "/mock/home")

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/mock/home/custom/jax/cache")


def test_resolve_jax_cache_uses_xdg_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure XDG_CACHE_HOME is used when JAX_COMPILATION_CACHE_DIR is not set."""
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/custom/xdg/cache")

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/custom/xdg/cache/g/jax")


def test_resolve_jax_cache_expands_xdg_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure XDG_CACHE_HOME is expanded."""
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "~/custom/xdg/cache")
    monkeypatch.setenv("HOME", "/mock/home")

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/mock/home/custom/xdg/cache/g/jax")


def test_resolve_jax_cache_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure fallback to ~/.cache/g/jax is used when no env vars are set."""
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", "/mock/home")
    monkeypatch.setattr(Path, "home", lambda: Path("/mock/home"))

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/mock/home/.cache/g/jax")
