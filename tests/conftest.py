"""Process-wide JAX policy for CPU-safe mathematical tests."""

from __future__ import annotations

import os

# Pytest imports this file before test modules. Keep runtime selection here so
# direct compute-module imports cannot initialize a CUDA backend on login nodes.
# GPU jobs can opt in explicitly by exporting JAX_PLATFORMS before pytest starts.
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
