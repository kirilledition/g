"""BGEN input and Oxford sample-file helpers."""

from g.io.bgen.chunks import iter_dosage_genotype_chunks, iter_genotype_chunks, load_bgen_sample_table, read_bgen_chunk, read_bgen_chunk_host, validate_bgen_sample_order
from g.io.bgen.dosage import convert_probability_tensor_to_dosage, load_backend_core
from g.io.bgen.metadata import CoreVariantMetadata, build_bgen_variant_table, build_bgen_variant_table_arrays, build_core_variant_metadata, build_variant_table_arrays_from_core_metadata, build_variant_table_from_core_metadata, resolve_variant_identifier_values
from g.io.bgen.reader import BgenReader, open_bgen, resolve_sample_identifier_source
from g.io.bgen.sample import build_sample_identifier_table, load_sample_identifier_table, resolve_bgen_sample_path, split_sample_file_line
from g.io.bgen.selectors import ContiguousVariantSlice, ReadSelection, VariantReadRun, build_variant_read_runs, normalize_axis_index, normalize_axis_selector, normalize_read_selection, resolve_contiguous_variant_slice
