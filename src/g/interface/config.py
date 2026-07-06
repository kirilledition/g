"""Thin Python boundary for the Rust-owned REGENIE configuration frontend."""

from __future__ import annotations

import g._core

InputConfig = g._core.InputConfig
TraitConfig = g._core.TraitConfig
BinaryConfig = g._core.BinaryConfig
GComputeConfig = g._core.GComputeConfig
GOutputConfig = g._core.GOutputConfig
GDiagnosticsConfig = g._core.GDiagnosticsConfig
RegenieConfig = g._core.RegenieConfig
load_packaged_config = g._core.load_packaged_config
