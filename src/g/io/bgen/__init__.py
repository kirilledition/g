"""BGEN input and Oxford sample-file helpers."""

from g.io.bgen.chunks import (
    iter_dosage_genotype_chunks,
    iter_genotype_chunks,
    load_bgen_sample_table,
    read_bgen_chunk,
    read_bgen_chunk_host,
    validate_bgen_sample_order,
)
from g.io.bgen.dosage import (
    convert_probability_matrix_to_dosage,
    convert_probability_tensor_to_dosage,
)
from g.io.bgen.metadata import (
    CoreVariantMetadata,
    build_bgen_variant_table,
    build_bgen_variant_table_arrays,
    build_core_variant_metadata,
    build_variant_table_arrays_from_core_metadata,
    build_variant_table_from_arrays,
    build_variant_table_from_core_metadata,
    resolve_variant_identifier_values,
)
from g.io.bgen.reader import BgenReader, open_bgen
from g.io.bgen.sample import (
    build_generated_sample_identifier_array,
    build_sample_identifier_table,
    load_sample_identifier_table,
    resolve_bgen_sample_path,
    resolve_sample_identifier_source,
    split_sample_file_line,
)
from g.io.bgen.selectors import (
    ContiguousVariantSlice,
    ReadSelection,
    VariantReadRun,
    build_variant_read_runs,
    normalize_axis_index,
    normalize_axis_selector,
    normalize_read_selection,
    resolve_contiguous_variant_slice,
)

__all__ = [
    "BgenReader",
    "ContiguousVariantSlice",
    "CoreVariantMetadata",
    "ReadSelection",
    "VariantReadRun",
    "build_bgen_variant_table",
    "build_bgen_variant_table_arrays",
    "build_core_variant_metadata",
    "build_generated_sample_identifier_array",
    "build_sample_identifier_table",
    "build_variant_read_runs",
    "build_variant_table_arrays_from_core_metadata",
    "build_variant_table_from_arrays",
    "build_variant_table_from_core_metadata",
    "convert_probability_matrix_to_dosage",
    "convert_probability_tensor_to_dosage",
    "iter_dosage_genotype_chunks",
    "iter_genotype_chunks",
    "load_bgen_sample_table",
    "load_sample_identifier_table",
    "normalize_axis_index",
    "normalize_axis_selector",
    "normalize_read_selection",
    "open_bgen",
    "read_bgen_chunk",
    "read_bgen_chunk_host",
    "resolve_bgen_sample_path",
    "resolve_contiguous_variant_slice",
    "resolve_sample_identifier_source",
    "resolve_variant_identifier_values",
    "split_sample_file_line",
    "validate_bgen_sample_order",
]
