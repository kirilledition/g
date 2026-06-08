#!/usr/bin/env python3
"""CLI wrapper for benchmark JSON summary comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tooling.performance_compare as performance_compare


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser.

    """
    parser = argparse.ArgumentParser(description="Compare two benchmark JSON summary files.")
    parser.add_argument("baseline_json", type=Path, help="Baseline benchmark JSON summary.")
    parser.add_argument("new_json", type=Path, help="New benchmark JSON summary.")
    return parser


def main() -> None:
    """Run the benchmark summary comparison CLI."""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        report = performance_compare.compare_summary_paths(arguments.baseline_json, arguments.new_json)
    except performance_compare.PerformanceComparisonError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(performance_compare.render_comparison_report(report), end="")


if __name__ == "__main__":
    main()
