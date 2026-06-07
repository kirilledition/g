#!/usr/bin/env python3
"""Compatibility wrapper for the development tooling BGEN reader benchmark."""

from __future__ import annotations

import tooling.cli.benchmark_bgen_reader as benchmark_bgen_reader
from tooling.cli.benchmark_bgen_reader import *  # noqa: F403

if __name__ == "__main__":
    benchmark_bgen_reader.main()
