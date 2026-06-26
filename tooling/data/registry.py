"""Download registries for benchmark datasets and fixture inputs."""

from __future__ import annotations

from tooling.common import downloads as tooling_downloads

CHROMOSOME_22_SOURCE_REGISTRY: dict[str, tooling_downloads.DownloadRegistryEntry] = {
    "pgen.zst": tooling_downloads.DownloadRegistryEntry(
        name="1kg_chr22_full_pgen_zst",
        download_url="https://www.dropbox.com/s/w9wwua4pe9em280/chr22_phase3.pgen.zst?dl=1",
        file_name="1kg_chr22_full.pgen.zst",
        expected_sha256=None,
        kind="benchmark_dataset",
        description="1000 Genomes chromosome 22 PGEN source compressed with Zstandard.",
    ),
    "pvar.zst": tooling_downloads.DownloadRegistryEntry(
        name="1kg_chr22_full_pvar_zst",
        download_url="https://www.dropbox.com/s/3acsdd1sqlj2pa8/chr22_phase3_noannot.pvar.zst?dl=1",
        file_name="1kg_chr22_full.pvar.zst",
        expected_sha256=None,
        kind="benchmark_dataset",
        description="1000 Genomes chromosome 22 PVAR source compressed with Zstandard.",
    ),
    "psam": tooling_downloads.DownloadRegistryEntry(
        name="1kg_chr22_full_psam",
        download_url="https://www.dropbox.com/s/6ppo144ikdzery5/phase3_corrected.psam?dl=1",
        file_name="1kg_chr22_full.psam",
        expected_sha256=None,
        kind="benchmark_dataset",
        description="1000 Genomes corrected sample metadata for chromosome 22 fixtures.",
    ),
}

BENCHMARK_DATASET_REGISTRY: dict[str, tooling_downloads.DownloadRegistryEntry] = {
    registry_entry.name: registry_entry for registry_entry in CHROMOSOME_22_SOURCE_REGISTRY.values()
}
TEST_FIXTURE_REGISTRY: dict[str, tooling_downloads.DownloadRegistryEntry] = {}
PREDICTION_LIST_FIXTURE_REGISTRY: dict[str, tooling_downloads.DownloadRegistryEntry] = {}
EXTERNAL_BASELINE_DATA_REGISTRY: dict[str, tooling_downloads.DownloadRegistryEntry] = {}
