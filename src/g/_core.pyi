from pathlib import Path

class NativeBedChunkReadResult:
    sample_count: int
    variant_count: int
    genotype_values_le: bytes


class NativeFloat64Buffer:
    def __buffer__(self, flags: int, /) -> memoryview: ...


class NativeInt64Buffer:
    def __buffer__(self, flags: int, /) -> memoryview: ...


class NativeUInt8Buffer:
    def __buffer__(self, flags: int, /) -> memoryview: ...


class NativePreprocessedGenotypeChunkResult:
    sample_count: int
    variant_count: int
    imputed_genotype_values: NativeFloat64Buffer
    missing_mask_values: NativeUInt8Buffer
    allele_one_frequency_values: NativeFloat64Buffer
    observation_count_values: NativeInt64Buffer

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
