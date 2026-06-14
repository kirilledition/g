"""BGEN genotype source configuration."""

from __future__ import annotations

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path


@dataclasses.dataclass(frozen=True)
class GenotypeSourceConfig:
    """Configuration describing one resolved BGEN input source.

    Attributes:
        source_path: BGEN genotype file path.
        sample_path: Explicit Oxford sample file path, or None to use embedded BGEN sample identifiers.

    """

    source_path: Path
    sample_path: Path | None

    def __post_init__(self) -> None:
        """Validate the configured BGEN source path."""
        if self.source_path.suffix != ".bgen":
            message = f"Expected a .bgen source path, found '{self.source_path}'."
            raise ValueError(message)
