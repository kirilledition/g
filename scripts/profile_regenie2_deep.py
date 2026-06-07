#!/usr/bin/env python3
"""Compatibility wrapper for the development tooling deep profiler."""

from __future__ import annotations

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep
from tooling.cli.profile_regenie2_deep import *  # noqa: F403

if __name__ == "__main__":
    profile_regenie2_deep.main()
