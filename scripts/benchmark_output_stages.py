#!/usr/bin/env python3
"""Compatibility wrapper for the development tooling output-stage benchmark."""

from __future__ import annotations

import tooling.cli.benchmark_output_stages as benchmark_output_stages
from tooling.cli.benchmark_output_stages import *  # noqa: F403

if __name__ == "__main__":
    benchmark_output_stages.main()
