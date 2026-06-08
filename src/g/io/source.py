"""BGEN genotype source configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class GenotypeSourceConfig:
    """Configuration describing one resolved BGEN input source.

    Attributes:
        source_path: BGEN genotype file path.
        sample_path: Explicit Oxford sample file path, if configured.
        resolved_sample_path: Explicit or adjacent Oxford sample file path, if available.

    """

    source_path: Path
    sample_path: Path | None = None
    resolved_sample_path: Path | None = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validate and resolve derived BGEN source paths."""
        source_path = Path(self.source_path)
        sample_path = Path(self.sample_path) if self.sample_path is not None else None
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "sample_path", sample_path)
        if source_path.suffix != ".bgen":
            message = f"Expected a .bgen source path, found '{source_path}'."
            raise ValueError(message)
        object.__setattr__(
            self,
            "resolved_sample_path",
            resolve_bgen_sample_path(source_path, sample_path),
        )


def build_bgen_source_config(bgen_path: Path | str, sample_path: Path | str | None = None) -> GenotypeSourceConfig:
    """Build and validate a genotype source config for a BGEN file."""
    return GenotypeSourceConfig(
        source_path=Path(bgen_path),
        sample_path=Path(sample_path) if sample_path is not None else None,
    )


def resolve_bgen_sample_path(bgen_path: Path, sample_path: Path | None = None) -> Path | None:
    """Resolve an explicit or adjacent Oxford sample file for one BGEN file."""
    if sample_path is not None:
        return sample_path
    adjacent_sample_path = bgen_path.with_suffix(".sample")
    return adjacent_sample_path if adjacent_sample_path.exists() else None
