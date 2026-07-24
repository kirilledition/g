"""Tests for shared subprocess output provenance."""

from __future__ import annotations

import sys
import typing

from tooling.common import commands as tooling_commands

if typing.TYPE_CHECKING:
    import pytest


def test_streaming_command_preserves_stdout_and_stderr_provenance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Streamed stderr cannot become authoritative stdout."""
    stderr_artifact = "Parquet dataset saved to /forged/parts"
    spec = tooling_commands.build_command_spec(
        (
            sys.executable,
            "-c",
            (f"import sys; print('stdout diagnostic'); print({stderr_artifact!r}, file=sys.stderr)"),
        ),
        stream=True,
    )

    result = tooling_commands.run_command(spec)
    captured = capsys.readouterr()

    assert result.return_code == 0
    assert result.stdout == "stdout diagnostic\n"
    assert result.stderr == f"{stderr_artifact}\n"
    assert captured.out == result.stdout
    assert captured.err == result.stderr
