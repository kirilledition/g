"""Console forwarding for the native GWAS CLI."""

from __future__ import annotations

import sys
import typing

import g._core


def run(arguments: typing.Sequence[str]) -> int:
    """Run CLI arguments through the native CLI driver."""
    result = g._core.cli.run(list(arguments))
    for output_text in result.stdout_chunks:
        print(output_text, end="")
    for output_text in result.stderr_chunks:
        print(output_text, end="", file=sys.stderr)
    return result.exit_code


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run(sys.argv[1:]))
