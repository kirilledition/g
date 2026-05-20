"""Public Python API for GWAS execution."""

from __future__ import annotations

import typing

from g import runner
from g.interface import config

RunArtifacts = runner.RunArtifacts


class RegenieApi:
    """Callable public REGENIE-compatible API."""

    def __call__(self, regenie_config: config.RegenieConfig) -> RunArtifacts:
        """Run from a normalized config."""
        return runner.regenie(regenie_config)

    def from_options(self, raw_options: typing.Mapping[str, typing.Any]) -> RunArtifacts:
        """Build a config from Python options and run it."""
        return runner.regenie(config.RegenieConfig.from_options(raw_options))


regenie = RegenieApi()
