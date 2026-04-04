"""REGENIE step 1 output file parsers for LOCO predictions."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class LocoSampleIndex:
    """Sample alignment index for LOCO predictions.

    Attributes:
        family_identifiers: Family identifiers from LOCO header.
        individual_identifiers: Individual identifiers from LOCO header.
        loco_sample_count: Number of samples in the LOCO file.

    """

    family_identifiers: npt.NDArray[np.str_]
    individual_identifiers: npt.NDArray[np.str_]
    loco_sample_count: int


@dataclass(frozen=True)
class LocoPredictions:
    """LOCO predictions for all chromosomes.

    Attributes:
        sample_index: Sample alignment information.
        chromosome_predictions: Dictionary mapping chromosome string to prediction array.

    """

    sample_index: LocoSampleIndex
    chromosome_predictions: dict[str, npt.NDArray[np.float64]]


@dataclass(frozen=True)
class PredictionListEntry:
    """Single entry from a _pred.list file.

    Attributes:
        phenotype_name: Name of the phenotype.
        loco_file_path: Path to the .loco file.

    """

    phenotype_name: str
    loco_file_path: Path


class Step1PredictionSource(typing.Protocol):
    """Protocol for step 1 prediction sources.

    Allows future implementations (e.g., native step 1) to share the same interface.
    """

    def get_chromosome_predictions(
        self,
        chromosome: str,
        sample_family_identifiers: npt.NDArray[np.str_],
        sample_individual_identifiers: npt.NDArray[np.str_],
    ) -> jax.Array:
        """Return LOCO predictions for a chromosome aligned to the sample order."""
        ...


def parse_prediction_list_file(prediction_list_path: Path) -> list[PredictionListEntry]:
    """Parse a REGENIE _pred.list file.

    Args:
        prediction_list_path: Path to the _pred.list file.

    Returns:
        List of prediction list entries with phenotype names and LOCO file paths.

    Raises:
        ValueError: If any line does not contain exactly two space-delimited fields.
        FileNotFoundError: If the prediction list file does not exist.

    """
    if not prediction_list_path.exists():
        message = f"Prediction list file not found: {prediction_list_path}"
        raise FileNotFoundError(message)

    entries: list[PredictionListEntry] = []
    with prediction_list_path.open() as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            fields = stripped_line.split()
            if len(fields) != 2:
                message = (
                    f"Invalid _pred.list line {line_number}: expected 2 space-delimited fields, found {len(fields)}."
                )
                raise ValueError(message)
            phenotype_name, loco_path_string = fields
            entries.append(
                PredictionListEntry(
                    phenotype_name=phenotype_name,
                    loco_file_path=Path(loco_path_string),
                )
            )

    if not entries:
        message = f"Prediction list file is empty: {prediction_list_path}"
        raise ValueError(message)

    return entries


def parse_loco_sample_identifiers(header_line: str) -> LocoSampleIndex:
    """Parse the FID_IID header line from a .loco file.

    The header line format is: `FID_IID sample1_fid_iid sample2_fid_iid ...`
    Each sample identifier is in the format `FID_IID` where FID and IID are
    separated by underscore.

    Args:
        header_line: The first line of a .loco file.

    Returns:
        Sample index with parsed family and individual identifiers.

    Raises:
        ValueError: If the header format is invalid or contains no sample identifiers.

    """
    fields = header_line.strip().split()

    if len(fields) < 2:
        message = "LOCO header must contain at least the FID_IID marker and one sample identifier."
        raise ValueError(message)

    if fields[0] != "FID_IID":
        message = f"LOCO header must start with 'FID_IID', found '{fields[0]}'."
        raise ValueError(message)

    sample_identifiers = fields[1:]
    family_identifiers: list[str] = []
    individual_identifiers: list[str] = []

    for sample_index, sample_identifier in enumerate(sample_identifiers):
        parts = sample_identifier.split("_", maxsplit=1)
        if len(parts) != 2:
            message = (
                f"Sample identifier at position {sample_index} ('{sample_identifier}') "
                f"does not contain underscore separator for FID_IID format."
            )
            raise ValueError(message)
        family_identifier, individual_identifier = parts
        family_identifiers.append(family_identifier)
        individual_identifiers.append(individual_identifier)

    return LocoSampleIndex(
        family_identifiers=np.array(family_identifiers, dtype=np.str_),
        individual_identifiers=np.array(individual_identifiers, dtype=np.str_),
        loco_sample_count=len(sample_identifiers),
    )


def normalize_chromosome(chromosome: str) -> str:
    """Normalize chromosome string for matching.

    Strips 'chr' prefix and leading zeros for numeric chromosomes.

    Args:
        chromosome: Raw chromosome string.

    Returns:
        Normalized chromosome string.

    """
    normalized = chromosome.lower()
    if normalized.startswith("chr"):
        normalized = normalized[3:]
    if normalized.isdigit():
        normalized = str(int(normalized))
    return normalized


def parse_loco_file(loco_file_path: Path) -> LocoPredictions:
    """Parse a REGENIE .loco file into sample index and predictions.

    The .loco file format:
    - Line 1: Header with FID_IID marker and sample identifiers
    - Lines 2+: Chromosome number followed by space-separated predictions

    Args:
        loco_file_path: Path to the .loco file.

    Returns:
        LOCO predictions with sample index and per-chromosome prediction arrays.

    Raises:
        ValueError: If the file format is invalid.
        FileNotFoundError: If the .loco file does not exist.

    """
    if not loco_file_path.exists():
        message = f"LOCO file not found: {loco_file_path}"
        raise FileNotFoundError(message)

    chromosome_predictions: dict[str, npt.NDArray[np.float64]] = {}
    sample_index: LocoSampleIndex | None = None

    with loco_file_path.open() as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if line_number == 1:
                sample_index = parse_loco_sample_identifiers(stripped_line)
                continue

            fields = stripped_line.split()
            if len(fields) < 2:
                message = (
                    f"LOCO data line {line_number}: expected chromosome and predictions, found {len(fields)} fields."
                )
                raise ValueError(message)

            chromosome = normalize_chromosome(fields[0])
            prediction_strings = fields[1:]

            if sample_index is not None and len(prediction_strings) != sample_index.loco_sample_count:
                message = (
                    f"LOCO data line {line_number}: expected {sample_index.loco_sample_count} predictions, "
                    f"found {len(prediction_strings)}."
                )
                raise ValueError(message)

            if chromosome in chromosome_predictions:
                message = f"LOCO file contains duplicate chromosome: {chromosome}"
                raise ValueError(message)

            chromosome_predictions[chromosome] = np.array(
                [float(value) for value in prediction_strings],
                dtype=np.float64,
            )

    if sample_index is None:
        message = f"LOCO file is empty or missing header: {loco_file_path}"
        raise ValueError(message)

    if not chromosome_predictions:
        message = f"LOCO file contains no chromosome predictions: {loco_file_path}"
        raise ValueError(message)

    return LocoPredictions(
        sample_index=sample_index,
        chromosome_predictions=chromosome_predictions,
    )


def build_sample_alignment_indices(
    loco_sample_index: LocoSampleIndex,
    target_family_identifiers: npt.NDArray[np.str_],
    target_individual_identifiers: npt.NDArray[np.str_],
) -> npt.NDArray[np.int64]:
    """Build indices to align LOCO samples to target sample order.

    Args:
        loco_sample_index: Sample index from the LOCO file.
        target_family_identifiers: Target family identifiers to align to.
        target_individual_identifiers: Target individual identifiers to align to.

    Returns:
        Array of indices where result[i] is the position in LOCO of target sample i.

    Raises:
        ValueError: If any target sample is not found in the LOCO file.

    """
    if len(target_family_identifiers) != len(target_individual_identifiers):
        message = "Target family and individual identifier arrays must have the same length."
        raise ValueError(message)

    loco_lookup: dict[tuple[str, str], int] = {}
    for index in range(loco_sample_index.loco_sample_count):
        family_identifier = str(loco_sample_index.family_identifiers[index])
        individual_identifier = str(loco_sample_index.individual_identifiers[index])
        loco_lookup[(family_identifier, individual_identifier)] = index

    alignment_indices: list[int] = []
    missing_samples: list[str] = []

    for target_index in range(len(target_family_identifiers)):
        family_identifier = str(target_family_identifiers[target_index])
        individual_identifier = str(target_individual_identifiers[target_index])
        key = (family_identifier, individual_identifier)

        if key not in loco_lookup:
            missing_samples.append(f"{family_identifier}_{individual_identifier}")
        else:
            alignment_indices.append(loco_lookup[key])

    if missing_samples:
        sample_examples = missing_samples[:5]
        sample_list = ", ".join(sample_examples)
        if len(missing_samples) > 5:
            sample_list += f", ... ({len(missing_samples)} total)"
        message = f"Target samples not found in LOCO file: {sample_list}"
        raise ValueError(message)

    return np.array(alignment_indices, dtype=np.int64)


def load_aligned_chromosome_predictions(
    loco_predictions: LocoPredictions,
    chromosome: str,
    target_family_identifiers: npt.NDArray[np.str_],
    target_individual_identifiers: npt.NDArray[np.str_],
) -> jax.Array:
    """Load and align LOCO predictions for a specific chromosome.

    Args:
        loco_predictions: Parsed LOCO predictions.
        chromosome: Chromosome to load predictions for.
        target_family_identifiers: Target family identifiers to align to.
        target_individual_identifiers: Target individual identifiers to align to.

    Returns:
        JAX array of predictions aligned to target sample order.

    Raises:
        KeyError: If the chromosome is not found in the LOCO file.
        ValueError: If any target sample is not found in the LOCO file.

    """
    normalized_chromosome = normalize_chromosome(chromosome)

    if normalized_chromosome not in loco_predictions.chromosome_predictions:
        available_chromosomes = sorted(loco_predictions.chromosome_predictions.keys())
        message = (
            f"Chromosome '{chromosome}' (normalized: '{normalized_chromosome}') "
            f"not found in LOCO file. Available chromosomes: {available_chromosomes}"
        )
        raise KeyError(message)

    alignment_indices = build_sample_alignment_indices(
        loco_sample_index=loco_predictions.sample_index,
        target_family_identifiers=target_family_identifiers,
        target_individual_identifiers=target_individual_identifiers,
    )

    chromosome_predictions = loco_predictions.chromosome_predictions[normalized_chromosome]
    aligned_predictions = chromosome_predictions[alignment_indices]

    return jnp.asarray(aligned_predictions, dtype=jnp.float32)


@dataclass(frozen=True)
class RegeniePredictionSource:
    """REGENIE step 1 prediction source implementation.

    Wraps parsed LOCO predictions to implement the Step1PredictionSource protocol.

    Attributes:
        loco_predictions: Parsed LOCO predictions.

    """

    loco_predictions: LocoPredictions

    def get_chromosome_predictions(
        self,
        chromosome: str,
        sample_family_identifiers: npt.NDArray[np.str_],
        sample_individual_identifiers: npt.NDArray[np.str_],
    ) -> jax.Array:
        """Return LOCO predictions for a chromosome aligned to the sample order.

        Args:
            chromosome: Chromosome to load predictions for.
            sample_family_identifiers: Target family identifiers to align to.
            sample_individual_identifiers: Target individual identifiers to align to.

        Returns:
            JAX array of predictions aligned to target sample order.

        """
        return load_aligned_chromosome_predictions(
            loco_predictions=self.loco_predictions,
            chromosome=chromosome,
            target_family_identifiers=sample_family_identifiers,
            target_individual_identifiers=sample_individual_identifiers,
        )


def load_prediction_source(
    prediction_list_path: Path,
    phenotype_name: str,
) -> RegeniePredictionSource:
    """Load a REGENIE prediction source for a specific phenotype.

    Args:
        prediction_list_path: Path to the _pred.list file.
        phenotype_name: Name of the phenotype to load predictions for.

    Returns:
        Prediction source ready for use in REGENIE step 2.

    Raises:
        ValueError: If the phenotype is not found in the prediction list.
        FileNotFoundError: If the prediction list or LOCO file does not exist.

    """
    entries = parse_prediction_list_file(prediction_list_path)

    matching_entry: PredictionListEntry | None = None
    for entry in entries:
        if entry.phenotype_name == phenotype_name:
            matching_entry = entry
            break

    if matching_entry is None:
        available_phenotypes = [entry.phenotype_name for entry in entries]
        message = (
            f"Phenotype '{phenotype_name}' not found in prediction list. Available phenotypes: {available_phenotypes}"
        )
        raise ValueError(message)

    loco_predictions = parse_loco_file(matching_entry.loco_file_path)

    return RegeniePredictionSource(loco_predictions=loco_predictions)
