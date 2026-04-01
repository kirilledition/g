"""Format-agnostic genotype source orchestration."""

from __future__ import annotations

import queue as queue_module
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from g.io.bgen import (
    iter_genotype_chunks as iter_bgen_genotype_chunks,
)
from g.io.bgen import (
    iter_linear_genotype_chunks as iter_bgen_linear_genotype_chunks,
)
from g.io.bgen import (
    load_bgen_sample_table,
)
from g.io.plink import (
    iter_genotype_chunks as iter_plink_genotype_chunks,
)
from g.io.plink import (
    iter_linear_genotype_chunks as iter_plink_linear_genotype_chunks,
)
from g.io.tabular import load_aligned_sample_data, load_aligned_sample_data_from_sample_table

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from g.models import AlignedSampleData, GenotypeChunk, LinearGenotypeChunk


@dataclass(frozen=True)
class GenotypeSourceConfig:
    """Configuration describing one genotype input source."""

    source_format: str
    source_path: Path


@dataclass(frozen=True)
class PrefetchEnvelope:
    """One prefetched iterator item or terminal signal."""

    value: object | None = None
    error: Exception | None = None
    traceback_text: str | None = None
    is_finished: bool = False


SUPPORTED_SOURCE_FORMATS = frozenset({"plink", "bgen"})


def build_plink_source_config(bed_prefix: Path | str) -> GenotypeSourceConfig:
    """Build a genotype source config for a PLINK dataset prefix."""
    return GenotypeSourceConfig(source_format="plink", source_path=Path(bed_prefix))


def build_bgen_source_config(bgen_path: Path | str) -> GenotypeSourceConfig:
    """Build a genotype source config for a BGEN file."""
    return GenotypeSourceConfig(source_format="bgen", source_path=Path(bgen_path))


def resolve_genotype_source_config(
    bfile: Path | str | None,
    bgen: Path | str | None,
) -> GenotypeSourceConfig:
    """Resolve the requested genotype source from public API arguments."""
    if (bfile is None) == (bgen is None):
        message = "Exactly one genotype source must be provided via bfile or bgen."
        raise ValueError(message)
    if bfile is not None:
        return build_plink_source_config(bfile)
    return build_bgen_source_config(bgen)  # type: ignore[arg-type]


def validate_genotype_source_config(genotype_source_config: GenotypeSourceConfig) -> None:
    """Validate that a genotype source config uses a supported format."""
    if genotype_source_config.source_format not in SUPPORTED_SOURCE_FORMATS:
        message = (
            f"Unsupported genotype source format '{genotype_source_config.source_format}'. "
            f"Expected one of {sorted(SUPPORTED_SOURCE_FORMATS)}."
        )
        raise ValueError(message)


def build_genotype_source_signature_paths(genotype_source_config: GenotypeSourceConfig) -> tuple[Path, ...]:
    """Return the input files that define reproducibility for one source."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == "plink":
        return (
            genotype_source_config.source_path.with_suffix(".bed"),
            genotype_source_config.source_path.with_suffix(".bim"),
            genotype_source_config.source_path.with_suffix(".fam"),
        )
    return (genotype_source_config.source_path,)


def load_aligned_sample_data_from_source(
    genotype_source_config: GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    *,
    is_binary_trait: bool,
) -> AlignedSampleData:
    """Load aligned sample data for any supported genotype source."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == "plink":
        return load_aligned_sample_data(
            bed_prefix=genotype_source_config.source_path,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            is_binary_trait=is_binary_trait,
        )
    return load_aligned_sample_data_from_sample_table(
        sample_table=load_bgen_sample_table(genotype_source_config.source_path),
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        match_family_and_individual_identifiers=False,
    )


def prefetch_iterator_values(
    value_iterator: Iterator[GenotypeChunk] | Iterator[LinearGenotypeChunk],
    prefetch_chunks: int,
) -> Iterator[GenotypeChunk] | Iterator[LinearGenotypeChunk]:
    """Prefetch a bounded number of iterator values on a background thread."""
    if prefetch_chunks <= 0:
        return value_iterator

    delivery_queue: queue_module.Queue[PrefetchEnvelope] = queue_module.Queue(maxsize=prefetch_chunks)
    stop_event = threading.Event()

    def worker() -> None:
        try:
            for value in value_iterator:
                if stop_event.is_set():
                    return
                while True:
                    try:
                        delivery_queue.put(PrefetchEnvelope(value=value), timeout=0.1)
                        break
                    except queue_module.Full:
                        if stop_event.is_set():
                            return
            delivery_queue.put(PrefetchEnvelope(is_finished=True))
        except Exception as error:  # noqa: BLE001
            delivery_queue.put(
                PrefetchEnvelope(
                    error=error,
                    traceback_text=traceback.format_exc(),
                )
            )

    prefetch_thread = threading.Thread(target=worker, name="genotype-source-prefetch", daemon=True)
    prefetch_thread.start()

    def consume_prefetched_values() -> Iterator[GenotypeChunk] | Iterator[LinearGenotypeChunk]:
        try:
            while True:
                envelope = delivery_queue.get()
                if envelope.error is not None:
                    message = f"Genotype prefetch failed: {envelope.error}"
                    if envelope.traceback_text is not None:
                        message = f"{message}\n{envelope.traceback_text}"
                    raise RuntimeError(message) from envelope.error
                if envelope.is_finished:
                    return
                if envelope.value is None:
                    message = "Genotype prefetch returned an empty envelope without finishing."
                    raise RuntimeError(message)
                yield envelope.value  # type: ignore[misc]
        finally:
            stop_event.set()
            prefetch_thread.join(timeout=0.5)

    return consume_prefetched_values()


def iter_genotype_chunks_from_source(
    genotype_source_config: GenotypeSourceConfig,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    include_missing_value_flag: bool = True,
    prefetch_chunks: int = 0,
) -> Iterator[GenotypeChunk]:
    """Yield genotype chunks for any supported source format."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == "plink":
        base_iterator = iter_plink_genotype_chunks(
            bed_prefix=genotype_source_config.source_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            include_missing_value_flag=include_missing_value_flag,
        )
    else:
        base_iterator = iter_bgen_genotype_chunks(
            bgen_path=genotype_source_config.source_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            include_missing_value_flag=include_missing_value_flag,
        )
    return prefetch_iterator_values(base_iterator, prefetch_chunks)


def iter_linear_genotype_chunks_from_source(
    genotype_source_config: GenotypeSourceConfig,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    prefetch_chunks: int = 0,
) -> Iterator[LinearGenotypeChunk]:
    """Yield linear-regression genotype chunks for any supported source format."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == "plink":
        base_iterator = iter_plink_linear_genotype_chunks(
            bed_prefix=genotype_source_config.source_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
    else:
        base_iterator = iter_bgen_linear_genotype_chunks(
            bgen_path=genotype_source_config.source_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
    return prefetch_iterator_values(base_iterator, prefetch_chunks)
