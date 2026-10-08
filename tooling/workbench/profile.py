"""Bounded, identifier-free inspection of local BGEN Layout 2 inputs.

This is an encoding inventory, not a genotype decoder or the native reader's
whole-file compatibility gate. No JAX or native extension is imported.
"""

from __future__ import annotations

import enum
import math
import os
import re
import struct
import typing
import zlib
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path


class BgenProfileError(ValueError):
    """An inspected input is malformed or exceeds a configured safety bound."""


class BgenCompression(enum.StrEnum):
    """Compression choices in the BGEN header."""

    NONE = "none"
    ZLIB = "zlib"
    ZSTANDARD = "zstandard"


class InspectionScope(enum.StrEnum):
    """Extent of the sequential encoding inspection."""

    PREFIX = "prefix_only"
    ALL_DECLARED_RECORDS = "all_declared_records"


class DecodeRoute(enum.StrEnum):
    """Observed route candidates, subject to native validation at execution."""

    PACKED8_CANDIDATE = "packed8_candidate_requires_native_preflight"
    GENERIC_DOSAGE = "generic_dosage_then_gpu_association"
    UNSUPPORTED = "unsupported_records_observed"
    NO_RECORDS = "no_records_inspected"


@dataclass(frozen=True)
class BgenHeader:
    """Counts and encoding declarations read without scanning variant records."""

    sample_count: int
    variant_count: int
    compression: BgenCompression
    layout: int
    embedded_samples: bool
    first_variant_offset: int
    file_size_bytes: int
    header_block_length: int


@dataclass(frozen=True)
class SampleFileProfile:
    """Identifier-free result of validating an Oxford sample file."""

    sample_count: int
    unique_sample_keys: bool


@dataclass(frozen=True)
class SampleIdentifier:
    """One sample key used internally for analysis alignment, never reporting."""

    family: str
    individual: str

    def loco_key(self) -> str:
        """Return REGENIE's serialized prediction key for alignment checks."""
        return f"{self.family}_{self.individual}"


@dataclass(frozen=True)
class VariantEncoding:
    """Structural observations for one record, without identifiers or values."""

    allele_count: int
    phased: bool
    probability_bits: int
    minimum_ploidy: int
    maximum_ploidy: int
    missing_sample_count: int
    all_samples_diploid: bool


@dataclass(frozen=True)
class VariantBlockExtent:
    """One structural genotype extent, without metadata or decompressed bytes."""

    allele_count: int
    stored_length: int
    payload_length: int
    expanded_length: int
    end_offset: int


@dataclass(frozen=True)
class BgenProfile:
    """Bounded encoding observations that do not qualify whole-file execution.

    Attributes:
        header: File declarations and size.
        inspected_variant_count: Number of initial records inspected.
        inspection_scope: Whether every declared record was inspected.
        observed_allele_counts: Distinct allele counts in inspected records.
        observed_probability_bits: Distinct probability precisions observed.
        observed_phasing: Distinct phase flags observed.
        minimum_observed_ploidy: Smallest declared ploidy in inspected records.
        maximum_observed_ploidy: Largest declared ploidy in inspected records.
        missing_sample_calls: Missing calls across inspected records.
        inspected_sample_calls: Number of sample-record observations.
        unsupported_variant_count: Records outside native biallelic diploid support.
        sampled_packed8_compatible: Whether inspected encodings fit the fast route.
        decode_route: Route suggested by encoding observations alone.
        sample_file: Optional Oxford sample validation result.
        probability_values_validated: Always false; probabilities are not decoded.
        native_preflight_required: Always true, including an entire encoding scan.
        whole_file_fastpath_qualified: Always false; only native validation qualifies it.

    """

    header: BgenHeader
    inspected_variant_count: int
    inspection_scope: InspectionScope
    observed_allele_counts: tuple[int, ...]
    observed_probability_bits: tuple[int, ...]
    observed_phasing: tuple[bool, ...]
    minimum_observed_ploidy: int | None
    maximum_observed_ploidy: int | None
    missing_sample_calls: int
    inspected_sample_calls: int
    unsupported_variant_count: int
    sampled_packed8_compatible: bool
    decode_route: DecodeRoute
    sample_file: SampleFileProfile | None
    probability_values_validated: bool
    native_preflight_required: bool
    whole_file_fastpath_qualified: bool


def read_exact(stream: typing.BinaryIO, byte_count: int) -> bytes:
    """Read an already bounded number of bytes or report truncation."""
    result = stream.read(byte_count)
    if len(result) != byte_count:
        raise BgenProfileError("BGEN is truncated inside an inspected field or block.")
    return result


def read_unsigned(stream: typing.BinaryIO, byte_count: int) -> int:
    """Read a small little-endian integer."""
    return int.from_bytes(read_exact(stream, byte_count), "little")


def parse_header(stream: typing.BinaryIO, file_size_bytes: int) -> BgenHeader:
    """Read one header and validate its offset arithmetic without variant scans."""
    stream.seek(0)
    prefix = read_exact(stream, 20)
    offset, header_length, variant_count, sample_count = struct.unpack("<IIII", prefix[:16])
    if prefix[16:] not in (b"bgen", b"\x00\x00\x00\x00"):
        raise BgenProfileError("BGEN magic must be 'bgen' or four zero bytes.")
    if header_length < 20:
        raise BgenProfileError("BGEN header length must be at least 20 bytes.")
    first_variant_offset = offset + 4
    header_stop = header_length + 4
    if not header_stop <= first_variant_offset <= file_size_bytes:
        raise BgenProfileError("BGEN header or first-variant offset lies outside the file or overlaps the header.")
    stream.seek(header_stop - 4)
    flags = read_unsigned(stream, 4)
    if flags & 0x7FFFFFC0:
        raise BgenProfileError("BGEN header contains reserved flag bits.")
    compression_flag = flags & 3
    if compression_flag == 3:
        raise BgenProfileError("BGEN compression flag 3 is reserved.")
    compression = (BgenCompression.NONE, BgenCompression.ZLIB, BgenCompression.ZSTANDARD)[compression_flag]
    layout = (flags >> 2) & 15
    if layout != 2:
        raise BgenProfileError("This bounded profiler supports BGEN Layout 2 only.")
    embedded_samples = bool(flags >> 31)
    if embedded_samples:
        stream.seek(header_stop)
        sample_block_length = read_unsigned(stream, 4)
        stored_sample_count = read_unsigned(stream, 4)
        if sample_block_length < 8 or header_stop + sample_block_length > first_variant_offset:
            raise BgenProfileError("Embedded BGEN sample block overlaps variants or has an invalid length.")
        if stored_sample_count != sample_count:
            raise BgenProfileError("Embedded BGEN sample count differs from the header.")
    return BgenHeader(
        sample_count=sample_count,
        variant_count=variant_count,
        compression=compression,
        layout=layout,
        embedded_samples=embedded_samples,
        first_variant_offset=first_variant_offset,
        file_size_bytes=file_size_bytes,
        header_block_length=header_length,
    )


def read_header(path: Path) -> BgenHeader:
    """Read BGEN declarations cheaply without loading sample identifiers or JAX.

    Args:
        path: Local BGEN file.

    Returns:
        Validated Layout 2 header declarations.

    Raises:
        BgenProfileError: An inspected header field or offset is invalid.
        OSError: The local input cannot be opened or read.

    """
    with path.open("rb") as stream:
        return parse_header(stream, os.fstat(stream.fileno()).st_size)


def skip_string(stream: typing.BinaryIO, length_bytes: int, file_size_bytes: int) -> None:
    """Skip metadata without allocating or disclosing its contents."""
    length = read_unsigned(stream, length_bytes)
    end_offset = stream.tell() + length
    if end_offset > file_size_bytes:
        raise BgenProfileError("BGEN metadata length extends beyond the file.")
    stream.seek(end_offset)


def decompress_block(
    payload: bytes,
    compression_type: BgenCompression,
    expected_length: int,
    max_window_bytes: int = 256 * 1024 * 1024,
) -> bytes:
    """Bound decompression to one byte beyond the declared output length."""
    if compression_type is BgenCompression.NONE:
        return payload
    if compression_type is BgenCompression.ZLIB:
        decompressor = zlib.decompressobj()
        try:
            result = decompressor.decompress(payload, expected_length + 1)
        except zlib.error as error:
            raise BgenProfileError("Inspected BGEN block contains invalid zlib data.") from error
    else:
        try:
            import compression.zstd
        except ImportError as error:
            raise BgenProfileError(
                "This Python build lacks optional stdlib compression.zstd; inspect zstandard inputs in a "
                "Python 3.14 build with Zstandard support. The native engine supports Zstandard independently."
            ) from error
        decompressor = compression.zstd.ZstdDecompressor(
            options={
                compression.zstd.DecompressionParameter.window_log_max: max(10, (max_window_bytes - 1).bit_length())
            }
        )
        try:
            result = decompressor.decompress(payload, expected_length + 1)
        except compression.zstd.ZstdError as error:
            raise BgenProfileError(
                "Inspected BGEN block contains invalid Zstandard data or exceeds the configured decoder window bound."
            ) from error
    if len(result) != expected_length:
        raise BgenProfileError("Inspected BGEN block does not match its declared decompressed length.")
    if not decompressor.eof or decompressor.unused_data:
        raise BgenProfileError("Inspected BGEN block is incomplete or has trailing compressed data.")
    return result


def inspect_probability_block(payload: bytes, header: BgenHeader, allele_count: int) -> VariantEncoding:
    """Inspect encoding flags and probability sizing without decoding values."""
    minimum_length = header.sample_count + 10
    if len(payload) < minimum_length:
        raise BgenProfileError("Inspected BGEN probability block is shorter than its sample descriptors.")
    stored_sample_count, stored_allele_count, minimum_ploidy, maximum_ploidy = struct.unpack("<IHBB", payload[:8])
    if stored_sample_count != header.sample_count or stored_allele_count != allele_count:
        raise BgenProfileError("Inspected BGEN probability counts disagree with the file or record header.")
    if not 0 <= minimum_ploidy <= maximum_ploidy <= 63:
        raise BgenProfileError("Inspected BGEN ploidy bounds are invalid.")
    phase_flag, probability_bits = payload[8 + header.sample_count : minimum_length]
    if phase_flag not in (0, 1) or not 1 <= probability_bits <= 32:
        raise BgenProfileError("Inspected BGEN phase flag or probability precision is invalid.")
    missing_count = 0
    probability_count = 0
    all_samples_diploid = minimum_ploidy == maximum_ploidy == 2
    for descriptor in memoryview(payload)[8 : 8 + header.sample_count]:
        ploidy = descriptor & 63
        if descriptor & 64 or not minimum_ploidy <= ploidy <= maximum_ploidy:
            raise BgenProfileError("Inspected BGEN sample ploidy flags are reserved or outside declared bounds.")
        missing_count += bool(descriptor & 128)
        all_samples_diploid &= ploidy == 2
        probability_count += (
            ploidy * (allele_count - 1) if phase_flag else math.comb(ploidy + allele_count - 1, allele_count - 1) - 1
        )
    expected_probability_bytes = (probability_count * probability_bits + 7) // 8
    if len(payload) - minimum_length != expected_probability_bytes:
        raise BgenProfileError("Inspected BGEN probability payload length disagrees with its encoding flags.")
    return VariantEncoding(
        allele_count=allele_count,
        phased=bool(phase_flag),
        probability_bits=probability_bits,
        minimum_ploidy=minimum_ploidy,
        maximum_ploidy=maximum_ploidy,
        missing_sample_count=missing_count,
        all_samples_diploid=all_samples_diploid,
    )


def read_variant_extent(stream: typing.BinaryIO, header: BgenHeader) -> VariantBlockExtent:
    """Parse metadata lengths and leave the cursor at the genotype payload."""
    for _ in range(3):
        skip_string(stream, 2, header.file_size_bytes)
    read_unsigned(stream, 4)  # Genomic position is intentionally not retained.
    allele_count = read_unsigned(stream, 2)
    if allele_count < 1:
        raise BgenProfileError("BGEN records must contain at least one allele.")
    for _ in range(allele_count):
        skip_string(stream, 4, header.file_size_bytes)
    block_length = read_unsigned(stream, 4)
    end_offset = stream.tell() + block_length
    if not block_length or end_offset > header.file_size_bytes:
        raise BgenProfileError("BGEN genotype block is empty, truncated, or extends beyond the file.")
    if header.compression is BgenCompression.NONE:
        expected_length = block_length
        payload_length = block_length
    else:
        if block_length <= 4:
            raise BgenProfileError("Compressed BGEN block must include its output length and a nonempty payload.")
        expected_length = read_unsigned(stream, 4)
        payload_length = block_length - 4
    if expected_length < header.sample_count + 10:
        raise BgenProfileError("BGEN expanded genotype block is shorter than its sample descriptors.")
    return VariantBlockExtent(allele_count, block_length, payload_length, expected_length, end_offset)


def validate_record_boundaries(path: Path) -> BgenHeader:
    """Traverse every declared variant extent without reading genotype payloads.

    This completion check rejects truncated exports, empty compressed members,
    and trailing bytes. It does not decompress blocks, validate probability
    values, inspect sample ploidy, or establish scientific compatibility.
    """
    with path.open("rb") as stream:
        identity = os.fstat(stream.fileno())
        header = parse_header(stream, identity.st_size)
        stream.seek(header.first_variant_offset)
        for _ in range(header.variant_count):
            extent = read_variant_extent(stream, header)
            stream.seek(extent.end_offset)
        if stream.tell() != header.file_size_bytes:
            raise BgenProfileError("BGEN has trailing bytes after all declared records.")
        final_identity = os.fstat(stream.fileno())
        if (identity.st_size, identity.st_mtime_ns, identity.st_ctime_ns) != (
            final_identity.st_size,
            final_identity.st_mtime_ns,
            final_identity.st_ctime_ns,
        ):
            raise BgenProfileError("BGEN changed while its record boundaries were checked.")
    return header


def inspect_variant(
    stream: typing.BinaryIO,
    header: BgenHeader,
    max_compressed_bytes: int,
    max_decompressed_bytes: int,
) -> VariantEncoding:
    """Inspect one bounded record and advance to the next record."""
    extent = read_variant_extent(stream, header)
    if extent.allele_count < 2:
        raise BgenProfileError("BGEN Layout 2 profiling requires at least two alleles.")
    if extent.stored_length > max_compressed_bytes:
        raise BgenProfileError(
            "Inspected BGEN block exceeds max_compressed_bytes; raise the explicit bound if intended."
        )
    if extent.expanded_length > max_decompressed_bytes:
        raise BgenProfileError(
            "Inspected BGEN block exceeds max_decompressed_bytes; raise the explicit bound if intended."
        )
    compressed_payload = read_exact(stream, extent.payload_length)
    payload = decompress_block(compressed_payload, header.compression, extent.expanded_length, max_decompressed_bytes)
    return inspect_probability_block(payload, header, extent.allele_count)


def nonempty_sample_lines(stream: typing.TextIO) -> collections.abc.Iterator[list[str]]:
    """Read Oxford rows with a finite line limit and omit empty lines."""
    while line := stream.readline(65_537):
        if len(line) > 65_536:
            raise BgenProfileError("Oxford sample line exceeds the 65,536-character inspection bound.")
        content = line.strip(" \t\n\r\f")
        if content:
            yield re.split(r"[ \t\n\r\f]+", content)


def iter_sample_identifiers(path: Path) -> collections.abc.Iterator[SampleIdentifier]:
    """Stream validated Oxford sample keys for internal alignment only.

    Repeated family identifiers are valid. Uniqueness applies to the (FID, IID)
    pair, matching the engine's alignment contract. Identifiers must not be
    serialized into profiling or diagnostic reports.
    """
    with path.open(encoding="utf-8") as stream:
        lines = nonempty_sample_lines(stream)
        column_names = next(lines, [])
        column_types = next(lines, [])
        if not column_names or len(column_names) != len(column_types):
            raise BgenProfileError("Oxford sample header and type rows must be present with equal column counts.")
        for identifier in ("ID_1", "ID_2"):
            if column_names.count(identifier) != 1 or column_types[column_names.index(identifier)] != "0":
                raise BgenProfileError("Oxford sample requires exactly one ID_1 and ID_2 column, each with type 0.")
        family_column = column_names.index("ID_1")
        individual_column = column_names.index("ID_2")
        sample_keys: set[tuple[str, str]] = set()
        for columns in lines:
            if len(columns) != len(column_names):
                raise BgenProfileError("Oxford sample row has missing identifiers or a different column count.")
            key = (columns[family_column], columns[individual_column])
            if key in sample_keys:
                raise BgenProfileError("Oxford sample contains a duplicate (FID, IID) key; identifiers are omitted.")
            sample_keys.add(key)
            yield SampleIdentifier(family=key[0], individual=key[1])


def validate_sample_file(path: Path, expected_sample_count: int) -> SampleFileProfile:
    """Validate Oxford keys and BGEN sample count without printing identifiers."""
    count = 0
    for _ in iter_sample_identifiers(path):
        count += 1
        if count > expected_sample_count:
            raise BgenProfileError("Oxford sample contains more samples than the BGEN header.")
    if count != expected_sample_count:
        raise BgenProfileError("Oxford sample count differs from the BGEN header.")
    return SampleFileProfile(sample_count=count, unique_sample_keys=True)


def profile_bgen(
    path: Path,
    max_variants: int,
    sample_path: Path | None = None,
    max_compressed_bytes: int = 64 * 1024 * 1024,
    max_decompressed_bytes: int = 256 * 1024 * 1024,
) -> BgenProfile:
    """Inspect initial BGEN records with bounded memory and no accelerator imports.

    Args:
        path: Local BGEN input, already localized from approved cloud storage.
        max_variants: Positive maximum number of initial records to inspect.
        sample_path: Optional Oxford file to validate against the header count.
        max_compressed_bytes: Maximum stored genotype block size permitted.
        max_decompressed_bytes: Maximum expanded genotype block size permitted.

    Returns:
        Identifier-free declarations, observed encodings and route candidates.

    Raises:
        BgenProfileError: An inspected structure is invalid or exceeds a bound.
        OSError: An input cannot be opened or read.

    """
    if min(max_variants, max_compressed_bytes, max_decompressed_bytes) < 1:
        raise BgenProfileError("Inspection count and compressed/decompressed byte limits must be positive.")
    encodings: list[VariantEncoding] = []
    with path.open("rb") as stream:
        identity = os.fstat(stream.fileno())
        header = parse_header(stream, identity.st_size)
        stream.seek(header.first_variant_offset)
        for _ in range(min(max_variants, header.variant_count)):
            encodings.append(inspect_variant(stream, header, max_compressed_bytes, max_decompressed_bytes))
        inspected_all = len(encodings) == header.variant_count
        if inspected_all and stream.tell() != header.file_size_bytes:
            raise BgenProfileError("BGEN has trailing bytes after all declared records.")
        final_identity = os.fstat(stream.fileno())
        if (identity.st_size, identity.st_mtime_ns, identity.st_ctime_ns) != (
            final_identity.st_size,
            final_identity.st_mtime_ns,
            final_identity.st_ctime_ns,
        ):
            raise BgenProfileError("BGEN changed while the inspected records were read.")
    unsupported_count = sum(encoding.allele_count != 2 or not encoding.all_samples_diploid for encoding in encodings)
    packed_candidate = (
        bool(encodings)
        and header.compression is BgenCompression.ZLIB
        and all(
            encoding.allele_count == 2
            and encoding.all_samples_diploid
            and not encoding.phased
            and encoding.probability_bits == 8
            and not encoding.missing_sample_count
            for encoding in encodings
        )
    )
    if not encodings:
        route = DecodeRoute.NO_RECORDS
    elif unsupported_count:
        route = DecodeRoute.UNSUPPORTED
    elif packed_candidate:
        route = DecodeRoute.PACKED8_CANDIDATE
    else:
        route = DecodeRoute.GENERIC_DOSAGE
    return BgenProfile(
        header=header,
        inspected_variant_count=len(encodings),
        inspection_scope=InspectionScope.ALL_DECLARED_RECORDS if inspected_all else InspectionScope.PREFIX,
        observed_allele_counts=tuple(sorted({encoding.allele_count for encoding in encodings})),
        observed_probability_bits=tuple(sorted({encoding.probability_bits for encoding in encodings})),
        observed_phasing=tuple(sorted({encoding.phased for encoding in encodings})),
        minimum_observed_ploidy=min((encoding.minimum_ploidy for encoding in encodings), default=None),
        maximum_observed_ploidy=max((encoding.maximum_ploidy for encoding in encodings), default=None),
        missing_sample_calls=sum(encoding.missing_sample_count for encoding in encodings),
        inspected_sample_calls=len(encodings) * header.sample_count,
        unsupported_variant_count=unsupported_count,
        sampled_packed8_compatible=packed_candidate,
        decode_route=route,
        sample_file=validate_sample_file(sample_path, header.sample_count) if sample_path is not None else None,
        probability_values_validated=False,
        native_preflight_required=True,
        whole_file_fastpath_qualified=False,
    )
