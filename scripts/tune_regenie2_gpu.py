#!/usr/bin/env python3
"""Compatibility wrapper for the development tooling GPU tuner."""

from __future__ import annotations

import tooling.cli.tune_regenie2_gpu as tune_regenie2_gpu
from tooling.cli.tune_regenie2_gpu import *  # noqa: F403

if __name__ == "__main__":
    tune_regenie2_gpu.main()
