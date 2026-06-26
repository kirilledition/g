from __future__ import annotations

import json
import typing
from pathlib import Path

from tooling.common import downloads as tooling_downloads
from tooling.data import fetch as data_fetch
from tooling.data import registry as data_registry
from tooling.server import bootstrap_tools, nsight_tools

if typing.TYPE_CHECKING:
    import pytest


def test_retrieve_file_uses_pooch_hash_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_bytes = b"registered fixture\n"
    expected_sha256 = "a" * 64
    recorded_call: dict[str, object] = {}

    def fake_retrieve(**keyword_arguments: object) -> str:
        recorded_call.update(keyword_arguments)
        destination_path = Path(str(keyword_arguments["path"])) / str(keyword_arguments["fname"])
        destination_path.write_bytes(file_bytes)
        return str(destination_path)

    monkeypatch.setattr(tooling_downloads.pooch, "retrieve", fake_retrieve)

    destination_path = tmp_path / "downloads" / "fixture.txt"
    downloaded_file = tooling_downloads.retrieve_file(
        download_url="https://example.test/fixture.txt",
        destination_path=destination_path,
        expected_sha256=expected_sha256,
    )

    assert recorded_call["known_hash"] == f"sha256:{expected_sha256}"
    assert recorded_call["fname"] == "fixture.txt"
    assert downloaded_file.path == destination_path.resolve()
    manifest = json.loads(downloaded_file.manifest_path.read_text(encoding="utf-8"))
    assert manifest["managed_by"] == "pooch"
    assert manifest["expected_sha256"] == expected_sha256
    assert manifest["actual_sha256"] == tooling_downloads.calculate_sha256(destination_path)


def test_bootstrap_archive_retrieval_uses_pooch_processor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processor_marker = object()
    recorded_call: dict[str, object] = {}

    def fake_unzip_processor(
        *,
        members: tuple[str, ...],
        extract_directory: Path,
    ) -> tooling_downloads.DownloadProcessor:
        assert members == ("tool-binary",)
        assert extract_directory == tmp_path / "tool.zip.contents"
        return typing.cast("tooling_downloads.DownloadProcessor", processor_marker)

    def fake_retrieve_file(
        *,
        download_url: str,
        destination_path: Path,
        expected_sha256: str | None,
        processor: tooling_downloads.DownloadProcessor | None = None,
        timeout_seconds: int = tooling_downloads.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        chunk_size_bytes: int = tooling_downloads.DEFAULT_DOWNLOAD_CHUNK_SIZE_BYTES,
    ) -> tooling_downloads.DownloadedFile:
        del timeout_seconds, chunk_size_bytes
        recorded_call.update(
            {
                "download_url": download_url,
                "destination_path": destination_path,
                "expected_sha256": expected_sha256,
                "processor": processor,
            }
        )
        destination_path.write_bytes(b"archive")
        extracted_path = tmp_path / "tool.zip.contents" / "tool-binary"
        extracted_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_path.write_bytes(b"binary")
        return tooling_downloads.DownloadedFile(
            path=destination_path,
            download_url=download_url,
            expected_sha256=expected_sha256,
            actual_sha256=tooling_downloads.calculate_sha256(destination_path),
            size_bytes=destination_path.stat().st_size,
            manifest_path=destination_path.with_suffix(".manifest.json"),
            processed_paths=(extracted_path,),
        )

    monkeypatch.setattr(bootstrap_tools.tooling_downloads, "build_unzip_processor", fake_unzip_processor)
    monkeypatch.setattr(bootstrap_tools.tooling_downloads, "retrieve_file", fake_retrieve_file)

    tool_archive = bootstrap_tools.ToolArchive(
        tool_name="tool",
        download_url="https://example.test/tool.zip",
        archive_filename="tool.zip",
        expected_sha256="b" * 64,
        archive_format=bootstrap_tools.ArchiveFormat.ZIP,
        binary_members=(bootstrap_tools.BinaryMember(archive_member_name="tool-binary", installed_name="tool"),),
    )

    retrieved_archive = bootstrap_tools.retrieve_tool_archive(tool_archive, tmp_path)

    assert recorded_call["processor"] is processor_marker
    assert retrieved_archive.member_paths["tool-binary"].read_bytes() == b"binary"


def test_nsight_package_download_uses_pooch_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded_call: dict[str, object] = {}

    def fake_retrieve_file(
        *,
        download_url: str,
        destination_path: Path,
        expected_sha256: str | None,
        processor: tooling_downloads.DownloadProcessor | None = None,
        timeout_seconds: int = tooling_downloads.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        chunk_size_bytes: int = tooling_downloads.DEFAULT_DOWNLOAD_CHUNK_SIZE_BYTES,
    ) -> tooling_downloads.DownloadedFile:
        del processor, timeout_seconds, chunk_size_bytes
        recorded_call.update(
            {
                "download_url": download_url,
                "destination_path": destination_path,
                "expected_sha256": expected_sha256,
            }
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"deb")
        return tooling_downloads.DownloadedFile(
            path=destination_path,
            download_url=download_url,
            expected_sha256=expected_sha256,
            actual_sha256=tooling_downloads.calculate_sha256(destination_path),
            size_bytes=destination_path.stat().st_size,
            manifest_path=destination_path.with_suffix(".manifest.json"),
            processed_paths=(),
        )

    monkeypatch.setattr(nsight_tools.tooling_downloads, "retrieve_file", fake_retrieve_file)
    nsight_package = nsight_tools.NsightPackage(
        package_name="nsight-systems-2026.1.3",
        package_version="2026.1.3.243-1",
        filename="./nsight-systems-2026.1.3_2026.1.3.243-1_amd64.deb",
        sha256="c" * 64,
        depends=None,
    )

    package_path = nsight_tools.download_package(
        repository_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64",
        nsight_package=nsight_package,
        downloads_directory=tmp_path / "downloads",
    )

    assert package_path == tmp_path / "downloads" / "nsight-systems-2026.1.3_2026.1.3.243-1_amd64.deb"
    assert recorded_call["expected_sha256"] == "c" * 64
    assert recorded_call["download_url"] == (
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
        "nsight-systems-2026.1.3_2026.1.3.243-1_amd64.deb"
    )


def test_data_fetch_uses_dataset_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded_destinations: list[Path] = []

    def fake_retrieve_file(
        *,
        download_url: str,
        destination_path: Path,
        expected_sha256: str | None,
        processor: tooling_downloads.DownloadProcessor | None = None,
        timeout_seconds: int = tooling_downloads.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        chunk_size_bytes: int = tooling_downloads.DEFAULT_DOWNLOAD_CHUNK_SIZE_BYTES,
    ) -> tooling_downloads.DownloadedFile:
        del download_url, expected_sha256, processor, timeout_seconds, chunk_size_bytes
        recorded_destinations.append(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"source")
        return tooling_downloads.DownloadedFile(
            path=destination_path,
            download_url="https://example.test/source",
            expected_sha256=None,
            actual_sha256=tooling_downloads.calculate_sha256(destination_path),
            size_bytes=destination_path.stat().st_size,
            manifest_path=destination_path.with_suffix(".manifest.json"),
            processed_paths=(),
        )

    monkeypatch.setattr(data_fetch.tooling_downloads, "retrieve_file", fake_retrieve_file)
    dataset_paths = data_fetch.DatasetPaths(
        data_directory=tmp_path,
        full_dataset_prefix=tmp_path / "1kg_chr22_full",
        toy_dataset_prefix=tmp_path / "1kg_chr22_toy",
    )

    data_fetch.download_source_files(dataset_paths)

    assert recorded_destinations == [
        tmp_path / "1kg_chr22_full.pgen.zst",
        tmp_path / "1kg_chr22_full.pvar.zst",
        tmp_path / "1kg_chr22_full.psam",
    ]
    assert set(data_registry.BENCHMARK_DATASET_REGISTRY) == {
        "1kg_chr22_full_pgen_zst",
        "1kg_chr22_full_psam",
        "1kg_chr22_full_pvar_zst",
    }
    assert data_registry.TEST_FIXTURE_REGISTRY == {}
    assert data_registry.PREDICTION_LIST_FIXTURE_REGISTRY == {}
    assert data_registry.EXTERNAL_BASELINE_DATA_REGISTRY == {}
