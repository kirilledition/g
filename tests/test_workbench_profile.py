from __future__ import annotations

import dataclasses
import json
import math
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from tooling.workbench import profile


@dataclass(frozen=True)
class VariantSpec:
    allele_count: int = 2
    descriptors: bytes = bytes([2, 2, 2])
    minimum_ploidy: int = 2
    maximum_ploidy: int = 2
    phase_flag: int = 0
    probability_bits: int = 8
    stored_sample_count: int | None = None
    stored_allele_count: int | None = None


def probability_block(specification: VariantSpec) -> bytes:
    count = sum(
        (descriptor & 63) * (specification.allele_count - 1)
        if specification.phase_flag
        else math.comb((descriptor & 63) + specification.allele_count - 1, specification.allele_count - 1) - 1
        for descriptor in specification.descriptors
    )
    return (
        struct.pack(
            "<IHBB",
            specification.stored_sample_count
            if specification.stored_sample_count is not None
            else len(specification.descriptors),
            specification.stored_allele_count
            if specification.stored_allele_count is not None
            else specification.allele_count,
            specification.minimum_ploidy,
            specification.maximum_ploidy,
        )
        + specification.descriptors
        + bytes([specification.phase_flag, specification.probability_bits])
        + bytes((count * specification.probability_bits + 7) // 8)
    )


def encoded_string(value: bytes, byte_count: int) -> bytes:
    return len(value).to_bytes(byte_count, "little") + value


def variant_record(specification: VariantSpec, compression_type: profile.BgenCompression) -> bytes:
    metadata = (
        encoded_string(b"private_variant", 2)
        + encoded_string(b"private_rsid", 2)
        + encoded_string(b"22", 2)
        + struct.pack("<IH", 12_345, specification.allele_count)
        + b"".join(encoded_string(b"A", 4) for _ in range(specification.allele_count))
    )
    block = probability_block(specification)
    if compression_type is profile.BgenCompression.ZLIB:
        block = len(block).to_bytes(4, "little") + zlib.compress(block)
    elif compression_type is profile.BgenCompression.ZSTANDARD:
        import compression.zstd

        block = len(block).to_bytes(4, "little") + compression.zstd.compress(block)
    return metadata + len(block).to_bytes(4, "little") + block


def write_bgen(
    path: Path,
    specifications: list[VariantSpec],
    compression: profile.BgenCompression = profile.BgenCompression.ZLIB,
    embedded_identifiers: list[bytes] | None = None,
) -> Path:
    sample_count = len(specifications[0].descriptors) if specifications else 3
    sample_block = b""
    flags = 8 | list(profile.BgenCompression).index(compression)
    if embedded_identifiers is not None:
        identifiers = b"".join(encoded_string(identifier, 2) for identifier in embedded_identifiers)
        sample_block = struct.pack("<II", len(identifiers) + 8, len(embedded_identifiers)) + identifiers
        flags |= 1 << 31
    header = struct.pack("<IIII4sI", 20 + len(sample_block), 20, len(specifications), sample_count, b"bgen", flags)
    path.write_bytes(header + sample_block + b"".join(variant_record(item, compression) for item in specifications))
    return path


def patch_integer(path: Path, offset: int, value: int, byte_count: int = 4) -> None:
    content = bytearray(path.read_bytes())
    content[offset : offset + byte_count] = value.to_bytes(byte_count, "little")
    path.write_bytes(content)


@pytest.mark.parametrize("compression", list(profile.BgenCompression))
def test_compressions_report_format_without_qualifying_fastpath(
    tmp_path: Path, compression: profile.BgenCompression
) -> None:
    if compression is profile.BgenCompression.ZSTANDARD:
        pytest.importorskip("compression.zstd")
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec(), VariantSpec()], compression)
    result = profile.profile_bgen(path, 2)
    assert result.header.sample_count == 3
    assert result.header.variant_count == 2
    assert result.header.compression is compression
    assert result.inspection_scope is profile.InspectionScope.ALL_DECLARED_RECORDS
    assert result.inspected_sample_calls == 6
    assert result.observed_probability_bits == (8,)
    assert result.observed_allele_counts == (2,)
    assert result.observed_phasing == (False,)
    assert result.unsupported_variant_count == 0
    assert result.sampled_packed8_compatible == (compression is profile.BgenCompression.ZLIB)
    assert result.native_preflight_required
    assert not result.whole_file_fastpath_qualified
    assert not result.probability_values_validated
    serialized = json.dumps(dataclasses.asdict(result))
    assert "private_variant" not in serialized
    assert "private_rsid" not in serialized


def test_bounded_prefix_does_not_read_later_corruption(tmp_path: Path) -> None:
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec(), VariantSpec()])
    path.write_bytes(path.read_bytes()[:-5])
    result = profile.profile_bgen(path, 1)
    assert result.inspected_variant_count == 1
    assert result.inspection_scope is profile.InspectionScope.PREFIX
    assert result.decode_route is profile.DecodeRoute.PACKED8_CANDIDATE
    assert not result.whole_file_fastpath_qualified
    with pytest.raises(profile.BgenProfileError, match="truncated"):
        profile.profile_bgen(path, 2)


def test_encoding_inspection_does_not_claim_probability_validation(tmp_path: Path) -> None:
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()], profile.BgenCompression.NONE)
    content = bytearray(path.read_bytes())
    content[-6:-4] = b"\xff\xff"  # Structurally sized, mathematically invalid unphased pair.
    path.write_bytes(content)
    result = profile.profile_bgen(path, 1)
    assert not result.probability_values_validated
    assert result.native_preflight_required
    assert not result.whole_file_fastpath_qualified


@pytest.mark.parametrize(
    "specification",
    [VariantSpec(descriptors=bytes([2, 130, 2])), VariantSpec(phase_flag=1), VariantSpec(probability_bits=12)],
)
def test_valid_generic_encodings_remain_usable(tmp_path: Path, specification: VariantSpec) -> None:
    path = write_bgen(tmp_path / "input.bgen", [specification])
    result = profile.profile_bgen(path, 1)
    assert result.decode_route is profile.DecodeRoute.GENERIC_DOSAGE
    assert result.unsupported_variant_count == 0
    assert not result.sampled_packed8_compatible
    assert result.missing_sample_calls == specification.descriptors.count(130)


@pytest.mark.parametrize(
    "specification",
    [VariantSpec(allele_count=3), VariantSpec(descriptors=bytes([1, 2, 2]), minimum_ploidy=1)],
)
def test_unsupported_biallelic_diploid_inputs_are_reported(tmp_path: Path, specification: VariantSpec) -> None:
    path = write_bgen(tmp_path / "input.bgen", [specification])
    result = profile.profile_bgen(path, 1)
    assert result.unsupported_variant_count == 1
    assert result.decode_route is profile.DecodeRoute.UNSUPPORTED


@pytest.mark.parametrize(
    ("specification", "message"),
    [
        (VariantSpec(phase_flag=2), "phase flag"),
        (VariantSpec(probability_bits=0), "precision"),
        (VariantSpec(probability_bits=33), "precision"),
        (VariantSpec(descriptors=bytes([2, 66, 2])), "ploidy flags"),
        (VariantSpec(descriptors=bytes([1, 2, 2])), "ploidy flags"),
        (VariantSpec(minimum_ploidy=3, maximum_ploidy=2), "ploidy bounds"),
        (VariantSpec(stored_sample_count=4), "counts disagree"),
        (VariantSpec(stored_allele_count=3), "counts disagree"),
    ],
)
def test_malformed_encoding_is_rejected(tmp_path: Path, specification: VariantSpec, message: str) -> None:
    path = write_bgen(tmp_path / "input.bgen", [specification])
    with pytest.raises(profile.BgenProfileError, match=message):
        profile.profile_bgen(path, 1)


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (0, 0, "overlaps"),
        (0, 2**32 - 1, "outside"),
        (4, 19, "at least"),
        (20, 7, "reserved"),
        (20, 8 | 64, "reserved"),
        (20, 4, "Layout 2"),
    ],
)
def test_header_bounds_and_flags(tmp_path: Path, offset: int, value: int, message: str) -> None:
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()])
    patch_integer(path, offset, value)
    with pytest.raises(profile.BgenProfileError, match=message):
        profile.read_header(path)


def test_header_accepts_zero_magic_and_skips_variant_scan(tmp_path: Path) -> None:
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()])
    patch_integer(path, 16, 0)
    content = path.read_bytes()
    path.write_bytes(content[:24] + b"bad")
    header = profile.read_header(path)
    assert header.variant_count == 1
    with pytest.raises(profile.BgenProfileError):
        profile.profile_bgen(path, 1)


def test_invalid_magic_and_short_header(tmp_path: Path) -> None:
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()])
    patch_integer(path, 16, 1)
    with pytest.raises(profile.BgenProfileError, match="magic"):
        profile.read_header(path)
    path.write_bytes(b"short")
    with pytest.raises(profile.BgenProfileError, match="truncated"):
        profile.read_header(path)


def test_extended_header_uses_declared_flags_and_variant_offsets(tmp_path: Path) -> None:
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()])
    content = path.read_bytes()
    free_data = b"ignored free data"
    path.write_bytes(content[:20] + free_data + content[20:])
    patch_integer(path, 0, 20 + len(free_data))
    patch_integer(path, 4, 20 + len(free_data))
    result = profile.profile_bgen(path, 1)
    assert result.header.header_block_length == 20 + len(free_data)
    assert result.header.first_variant_offset == 24 + len(free_data)
    assert result.inspected_variant_count == 1


def test_embedded_samples_counts_are_checked_without_identifier_output(tmp_path: Path) -> None:
    path = write_bgen(
        tmp_path / "input.bgen", [VariantSpec()], embedded_identifiers=[b"secret-a", b"secret-b", b"secret-c"]
    )
    result = profile.profile_bgen(path, 1)
    assert result.header.embedded_samples
    assert "secret" not in json.dumps(dataclasses.asdict(result))
    patch_integer(path, 28, 4)
    with pytest.raises(profile.BgenProfileError, match="sample count"):
        profile.read_header(path)


def test_embedded_block_must_not_overlap_variant(tmp_path: Path) -> None:
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()], embedded_identifiers=[b"a", b"b", b"c"])
    patch_integer(path, 24, 2**32 - 1)
    with pytest.raises(profile.BgenProfileError, match="overlaps"):
        profile.read_header(path)


@pytest.mark.parametrize("compression", list(profile.BgenCompression))
def test_block_limits_precede_allocation(tmp_path: Path, compression: profile.BgenCompression) -> None:
    if compression is profile.BgenCompression.ZSTANDARD:
        pytest.importorskip("compression.zstd")
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()], compression)
    with pytest.raises(profile.BgenProfileError, match="max_compressed_bytes"):
        profile.profile_bgen(path, 1, max_compressed_bytes=1)
    with pytest.raises(profile.BgenProfileError, match="max_decompressed_bytes"):
        profile.profile_bgen(path, 1, max_decompressed_bytes=1)


@pytest.mark.parametrize("compression", [profile.BgenCompression.ZLIB, profile.BgenCompression.ZSTANDARD])
def test_decompression_rejects_wrong_length_truncation_and_trailing_data(compression: profile.BgenCompression) -> None:
    if compression is profile.BgenCompression.ZLIB:
        encoded = zlib.compress(b"payload")
    else:
        compression_module = pytest.importorskip("compression.zstd")
        encoded = compression_module.compress(b"payload")
    assert profile.decompress_block(encoded, compression, 7) == b"payload"
    for content, expected_length in [(encoded, 6), (encoded, 8), (encoded[:-1], 7), (encoded + b"trailing", 7)]:
        with pytest.raises(profile.BgenProfileError):
            profile.decompress_block(content, compression, expected_length)


def test_empty_file_inventory_is_not_a_fastpath_candidate(tmp_path: Path) -> None:
    path = write_bgen(tmp_path / "input.bgen", [])
    result = profile.profile_bgen(path, 1)
    assert result.inspected_variant_count == 0
    assert result.decode_route is profile.DecodeRoute.NO_RECORDS
    assert not result.sampled_packed8_compatible


def test_zstandard_unavailable_is_explicit_and_does_not_install_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "compression.zstd", None)
    with pytest.raises(profile.BgenProfileError, match=r"Python 3\.14 build with Zstandard support"):
        profile.decompress_block(b"placeholder", profile.BgenCompression.ZSTANDARD, 1)


def test_invalid_limits_and_trailing_file_content(tmp_path: Path) -> None:
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()])
    with pytest.raises(profile.BgenProfileError, match="positive"):
        profile.profile_bgen(path, 0)
    path.write_bytes(path.read_bytes() + b"trailing")
    with pytest.raises(profile.BgenProfileError, match="trailing"):
        profile.profile_bgen(path, 1)


@pytest.mark.parametrize("compression", list(profile.BgenCompression))
def test_record_boundary_validation_traverses_every_record_without_decompression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: profile.BgenCompression,
) -> None:
    if compression is profile.BgenCompression.ZSTANDARD:
        pytest.importorskip("compression.zstd")
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec(), VariantSpec()], compression)

    def fail_decompression(*arguments: object) -> bytes:
        del arguments
        raise AssertionError("Boundary validation must not decompress genotype blocks.")

    monkeypatch.setattr(profile, "decompress_block", fail_decompression)
    header = profile.validate_record_boundaries(path)
    assert header.sample_count == 3
    assert header.variant_count == 2
    assert header.compression is compression
    content = path.read_bytes()
    path.write_bytes(content[:-1])
    with pytest.raises(profile.BgenProfileError, match="truncated"):
        profile.validate_record_boundaries(path)
    path.write_bytes(content + b"trailing")
    with pytest.raises(profile.BgenProfileError, match="trailing"):
        profile.validate_record_boundaries(path)


def test_record_boundary_validation_rejects_empty_compressed_export(tmp_path: Path) -> None:
    specification = VariantSpec()
    path = write_bgen(tmp_path / "input.bgen", [specification])
    content = path.read_bytes()
    payload_start = len(content) - len(zlib.compress(probability_block(specification)))
    path.write_bytes(content[:payload_start])
    patch_integer(path, payload_start - 8, 4)
    # Some PLINK builds return success for this small-cohort empty-member export.
    with pytest.raises(profile.BgenProfileError, match="nonempty payload"):
        profile.validate_record_boundaries(path)


def test_record_boundary_validation_rejects_too_short_expanded_length(tmp_path: Path) -> None:
    specification = VariantSpec()
    path = write_bgen(tmp_path / "input.bgen", [specification])
    payload_start = len(path.read_bytes()) - len(zlib.compress(probability_block(specification)))
    patch_integer(path, payload_start - 4, 1)
    with pytest.raises(profile.BgenProfileError, match="sample descriptors"):
        profile.validate_record_boundaries(path)


def test_record_boundary_validation_does_not_require_zstandard_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()])
    patch_integer(path, 20, 10)  # Layout 2, Zstandard, with an opaque stored payload.
    monkeypatch.setitem(sys.modules, "compression.zstd", None)
    header = profile.validate_record_boundaries(path)
    assert header.compression is profile.BgenCompression.ZSTANDARD


def test_sample_file_matches_engine_pair_identity_contract(tmp_path: Path) -> None:
    sample_path = tmp_path / "input.sample"
    sample_path.write_text(
        "\nID_2 extra ID_1\n0 C 0\nsample-a unused family\nsample-b unused family\nsample-a unused other\n"
    )
    result = profile.validate_sample_file(sample_path, 3)
    assert result.sample_count == 3
    assert result.unique_sample_keys
    path = write_bgen(tmp_path / "input.bgen", [VariantSpec()])
    assert profile.profile_bgen(path, 1, sample_path).sample_file == result


def test_shared_identifier_reader_preserves_native_ascii_whitespace_contract(tmp_path: Path) -> None:
    path = tmp_path / "input.sample"
    path.write_text("ID_1 ID_2\n0 0\nfamily\tprivate\u00a0sample\n")
    identifiers = tuple(profile.iter_sample_identifiers(path))
    assert identifiers == (profile.SampleIdentifier(family="family", individual="private\u00a0sample"),)
    assert identifiers[0].loco_key() == "family_private\u00a0sample"


@pytest.mark.parametrize(
    "content",
    [
        "",
        "ID_1 ID_2\n0\n",
        "ID_1 ID_1 ID_2\n0 0 0\n",
        "ID_1 ID_2\n0 C\n",
        "ID_1 ID_2\n0 0\nfamily private-a\nfamily private-a\n",
        "ID_1 ID_2\n0 0\nfamily\n",
        "ID_1 ID_2\n0 0\nfamily private-a\n",
        "ID_1 ID_2\n0 0\n" + "a" * 65_537 + " private-a\n",
    ],
)
def test_sample_errors_do_not_disclose_identifiers(tmp_path: Path, content: str) -> None:
    path = tmp_path / "input.sample"
    path.write_text(content)
    with pytest.raises(profile.BgenProfileError) as exception:
        profile.validate_sample_file(path, 2)
    assert "private-a" not in str(exception.value)


def test_import_does_not_load_accelerator_or_native_modules() -> None:
    script = (
        "import sys; from tooling.workbench import profile; "
        "assert not any(name == 'jax' or name.startswith('jax.') or name == 'g._core' for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.phase0_data
def test_public_fixture_profile() -> None:
    path = Path("data/1kg_chr22_full.bgen")
    if not path.exists():
        pytest.skip("Public 1000 Genomes fixture is not localized in this checkout.")
    result = profile.profile_bgen(path, 8, Path("data/1kg_chr22_full.sample"))
    assert result.header.sample_count == 2504
    assert result.header.variant_count == 418_943
    assert result.inspected_variant_count == 8
    assert result.inspection_scope is profile.InspectionScope.PREFIX
    assert result.unsupported_variant_count == 0
    assert result.sampled_packed8_compatible
    assert not result.whole_file_fastpath_qualified
