from pathlib import Path

class NativeBedChunkReadResult:
    sample_count: int
    variant_count: int
    genotype_values_le: bytes

def hello_from_bin() -> str: ...
def read_bed_chunk_f64(
    bed_path: Path,
    sample_indices: list[int],
    variant_start: int,
    variant_stop: int,
) -> NativeBedChunkReadResult: ...
