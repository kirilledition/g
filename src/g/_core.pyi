from pathlib import Path

class NativeBedChunkReadResult:
    sample_count: int
    variant_count: int
    genotype_values_le: bytes

class NativePreprocessedGenotypeChunkResult:
    sample_count: int
    variant_count: int
    imputed_genotype_values_le: bytes
    missing_mask_values: bytes
    allele_one_frequency_le: bytes
    observation_count_le: bytes

def hello_from_bin() -> str: ...
def preprocess_genotype_matrix_f64(
    genotype_matrix: object,
) -> NativePreprocessedGenotypeChunkResult: ...
def read_bed_chunk_f64(
    bed_path: Path,
    sample_indices: list[int],
    variant_start: int,
    variant_stop: int,
) -> NativeBedChunkReadResult: ...
