#!/usr/bin/env python3
"""Compatibility wrapper for the development tooling binary-hot benchmark."""

from __future__ import annotations

import tooling.cli.benchmark_regenie2_binary_hot as benchmark_regenie2_binary_hot
from tooling.cli.benchmark_regenie2_binary_hot import *  # noqa: F403

if __name__ == "__main__":
    benchmark_regenie2_binary_hot.main()
