from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
from dataclasses import dataclass

from g.io import reader

OpenBgenHandle = object
CoreBgenHandle = object


@dataclass(frozen=True)
class CoreVariantMetadata:
    chromosome_values: list[str]
    variant_identifier_values: list[str]
    position_values: list[int]
    allele_one_values: list[str]
    allele_two_values: list[str]


def resolve_variant_identifier_values(variant_identifier_values: npt.NDArray[np.str_], rsid_values: npt.NDArray[np.str_]) -> npt.NDArray[np.str_]:
    return np.asarray(np.where(rsid_values != "", rsid_values, variant_identifier_values), dtype=np.str_)


def build_bgen_variant_table_arrays(bgen_handle: OpenBgenHandle) -> reader.VariantTableArrays:
    allele_identifier_values = np.asarray(bgen_handle.allele_ids, dtype=np.str_)
    allele_pairs = [allele_identifier_value.split(",") for allele_identifier_value in allele_identifier_values]
    counted_allele_values = [allele_pair[-1] if allele_pair else "" for allele_pair in allele_pairs]
    reference_allele_values = [allele_pair[0] if allele_pair else "" for allele_pair in allele_pairs]
    variant_identifier_values = np.asarray(bgen_handle.ids, dtype=np.str_)
    rsid_values = np.asarray(bgen_handle.rsids, dtype=np.str_)
    return reader.VariantTableArrays(
        chromosome_values=np.asarray(bgen_handle.chromosomes, dtype=np.str_),
        variant_identifier_values=resolve_variant_identifier_values(variant_identifier_values, rsid_values),
        position_values=np.asarray(bgen_handle.positions, dtype=np.int64),
        allele_one_values=np.asarray(counted_allele_values, dtype=np.str_),
        allele_two_values=np.asarray(reference_allele_values, dtype=np.str_),
    )


def build_bgen_variant_table(bgen_handle: OpenBgenHandle) -> pl.DataFrame:
    variant_table_arrays = build_bgen_variant_table_arrays(bgen_handle)
    return build_variant_table_from_arrays(variant_table_arrays)


def build_variant_table_arrays_from_core_metadata(variant_metadata: CoreVariantMetadata) -> reader.VariantTableArrays:
    return reader.VariantTableArrays(
        chromosome_values=np.asarray(variant_metadata.chromosome_values, dtype=np.str_),
        variant_identifier_values=np.asarray(variant_metadata.variant_identifier_values, dtype=np.str_),
        position_values=np.asarray(variant_metadata.position_values, dtype=np.int64),
        allele_one_values=np.asarray(variant_metadata.allele_one_values, dtype=np.str_),
        allele_two_values=np.asarray(variant_metadata.allele_two_values, dtype=np.str_),
    )


def build_variant_table_from_arrays(variant_table_arrays: reader.VariantTableArrays) -> pl.DataFrame:
    return pl.DataFrame({
        "chromosome": variant_table_arrays.chromosome_values,
        "variant_identifier": variant_table_arrays.variant_identifier_values,
        "genetic_distance": np.zeros(len(variant_table_arrays.position_values), dtype=np.float32),
        "position": variant_table_arrays.position_values,
        "allele_one": variant_table_arrays.allele_one_values,
        "allele_two": variant_table_arrays.allele_two_values,
    })


def build_variant_table_from_core_metadata(variant_metadata: CoreVariantMetadata) -> pl.DataFrame:
    return build_variant_table_from_arrays(build_variant_table_arrays_from_core_metadata(variant_metadata))


def build_core_variant_metadata(core_reader: CoreBgenHandle, variant_start: int, variant_stop: int) -> CoreVariantMetadata:
    chromosome_values, variant_identifier_values, position_values, allele_one_values, allele_two_values = core_reader.variant_metadata_slice(variant_start, variant_stop)
    return CoreVariantMetadata(list(chromosome_values), list(variant_identifier_values), list(position_values), list(allele_one_values), list(allele_two_values))
