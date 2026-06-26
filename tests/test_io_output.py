"""Tests for Rust-backed output persistence."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import typing
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq
import pytest

from g import _core, types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.io import output
from g.types import AssociationMode

EXPECTED_FINAL_COLUMNS = [
    "CHROM",
    "GENPOS",
    "ID",
    "ALLELE0",
    "ALLELE1",
    "A1FREQ",
    "INFO",
    "N",
    "TEST",
    "BETA",
    "SE",
    "CHISQ",
    "LOG10P",
    "EXTRA",
    "CORRECTION_METHOD",
    "CORRECTION_STATUS",
]
EXPECTED_CHUNK_COLUMNS = [
    *EXPECTED_FINAL_COLUMNS,
]


def build_step2_output_schema_fields(
    statistic_dtype: pa.DataType = pa.float32(),
) -> tuple[tuple[str, pa.DataType], ...]:
    """Build expected Step 2 output fields for the configured public statistic dtype."""
    return (
        ("CHROM", pa.string()),
        ("GENPOS", pa.int64()),
        ("ID", pa.string()),
        ("ALLELE0", pa.string()),
        ("ALLELE1", pa.string()),
        ("A1FREQ", pa.float32()),
        ("INFO", pa.float32()),
        ("N", pa.int32()),
        ("TEST", pa.string()),
        ("BETA", statistic_dtype),
        ("SE", statistic_dtype),
        ("CHISQ", statistic_dtype),
        ("LOG10P", statistic_dtype),
        ("EXTRA", pa.string()),
        ("CORRECTION_METHOD", pa.string()),
        ("CORRECTION_STATUS", pa.string()),
    )


STEP2_SCHEMA_COLUMN_NAMES = tuple(column_name for column_name, _ in build_step2_output_schema_fields())
TEST_DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "bgen"
HAPLOTYPES_BGEN_PATH = TEST_DATA_DIRECTORY / "haplotypes.bgen"
DEFAULT_TEST_INPUT_PATH: typing.Final[object] = object()


def resolve_test_output_run_paths(
    output_root: Path,
    association_mode: AssociationMode,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
) -> output.OutputRunPaths:
    """Resolve output paths with the test fixture's default output format."""
    return output.resolve_output_run_paths(
        output_root,
        association_mode,
        output_format,
    )


def prepare_test_output_run(
    *,
    output_root: Path,
    association_mode: AssociationMode,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    resume: bool,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
) -> output.PreparedOutputRun:
    """Prepare output paths with explicit production arguments."""
    return output.prepare_output_run(
        output_root=output_root,
        association_mode=association_mode,
        output_format=output_format,
        resume=resume,
        resume_mode=resume_mode,
    )


def create_test_output_writer_session(
    output_run_paths: output.OutputRunPaths,
    association_mode: AssociationMode,
    *,
    writer_thread_count: int,
    writer_queue_depth: int,
    finalize_parquet: bool,
    output_format: types.OutputFormat,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    parquet_compression: types.ParquetCompression,
    output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    collect_stage_timings: bool = False,
) -> typing.Any:
    """Create a writer session with explicit production arguments."""
    return output.create_output_writer_session(
        output_run_paths,
        association_mode,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
        output_format=output_format,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        parquet_compression=parquet_compression,
        output_statistic_dtype=output_statistic_dtype,
        collect_stage_timings=collect_stage_timings,
    )


def finalize_test_chunks_to_parquet(
    output_run_paths: output.OutputRunPaths,
    association_mode: AssociationMode,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
) -> Path:
    """Finalize chunks with the test fixture's default output format."""
    return output.finalize_chunks_to_parquet(
        output_run_paths,
        association_mode,
        output_format,
    )


class NativeChunkWritingCallback:
    """Callback that writes deterministic association values for native chunks."""

    def __init__(
        self,
        writer_session: typing.Any,
        extra_code_value: int | None = None,
        output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    ) -> None:
        self.writer_session = writer_session
        self.extra_code_value = extra_code_value
        self.output_statistic_dtype = output_statistic_dtype
        self.free_buffers: list[np.ndarray] = []

    def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> np.ndarray:
        if self.free_buffers:
            dosage_buffer = self.free_buffers.pop()
            if dosage_buffer.shape == (variant_count, sample_count):
                return dosage_buffer
        return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: np.ndarray,
        chunk_stats: _core.ChunkStats,
    ) -> None:
        variant_count = metadata.variant_stop_index - metadata.variant_start_index
        extra_code = (
            np.full(variant_count, self.extra_code_value, dtype=np.int32) if self.extra_code_value is not None else None
        )
        statistic_numpy_dtype = (
            np.float64 if self.output_statistic_dtype == types.FloatingPointDtype.FLOAT64 else np.float32
        )
        write_chunk_method = (
            self.writer_session.write_regenie2_native_chunk_f64
            if self.output_statistic_dtype == types.FloatingPointDtype.FLOAT64
            else self.writer_session.write_regenie2_native_chunk
        )
        write_chunk_method(
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=np.full(variant_count, 0.1, dtype=statistic_numpy_dtype),
            standard_error=np.full(variant_count, 0.01, dtype=statistic_numpy_dtype),
            chi_squared=np.full(variant_count, 10.0, dtype=statistic_numpy_dtype),
            log10_p_value=np.full(variant_count, 5.0, dtype=statistic_numpy_dtype),
            extra_code=extra_code,
        )
        self.free_buffers.append(genotype_matrix)


class NativeChunkCaptureCallback:
    """Callback that captures the last native chunk handles for writer tests."""

    def __init__(self) -> None:
        self.metadata: _core.VariantMetadata | None = None
        self.chunk_stats: _core.ChunkStats | None = None
        self.free_buffers: list[np.ndarray] = []

    def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> np.ndarray:
        if self.free_buffers:
            dosage_buffer = self.free_buffers.pop()
            if dosage_buffer.shape == (variant_count, sample_count):
                return dosage_buffer
        return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: np.ndarray,
        chunk_stats: _core.ChunkStats,
    ) -> None:
        self.metadata = metadata
        self.chunk_stats = chunk_stats
        self.free_buffers.append(genotype_matrix)

    def require_metadata(self) -> _core.VariantMetadata:
        assert self.metadata is not None
        return self.metadata

    def require_chunk_stats(self) -> _core.ChunkStats:
        assert self.chunk_stats is not None
        return self.chunk_stats


def assert_step2_output_schema_contract(schema: pa.Schema, statistic_dtype: pa.DataType = pa.float32()) -> None:
    """Assert the public Step 2 output schema contract for association outputs."""
    assert schema.names == list(STEP2_SCHEMA_COLUMN_NAMES)
    for column_name, expected_data_type in build_step2_output_schema_fields(statistic_dtype):
        actual_field = schema.field(column_name)
        assert actual_field.type == expected_data_type
        assert actual_field.nullable is True


def write_native_chunks(
    output_run_paths: output.OutputRunPaths,
    association_mode: AssociationMode,
    *,
    output_format: types.OutputFormat = types.OutputFormat.ARROW,
    extra_code_value: int | None = None,
    output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
) -> None:
    if output_format == types.OutputFormat.REGENIE and not output.get_run_manifest_path(output_run_paths).exists():
        output.write_run_manifest(output_run_paths, {"committed_chunks": []})
    writer_session = create_test_output_writer_session(
        output_run_paths,
        association_mode,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=False,
        output_format=output_format,
        chunks_per_arrow_file=16,
        arrow_compression=types.ArrowCompression.ZSTD,
        parquet_compression=types.ParquetCompression.NONE,
        output_statistic_dtype=output_statistic_dtype,
    )
    callback = NativeChunkWritingCallback(writer_session, extra_code_value, output_statistic_dtype)
    try:
        engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
        engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), callback)
        writer_session.finish()
    except Exception:
        writer_session.abort()
        raise


def build_test_header_object(
    tmp_path: Path,
    *,
    association_mode: AssociationMode = AssociationMode.REGENIE2_LINEAR,
    association_backend_kind: types.AssociationBackendKind = types.AssociationBackendKind.JAX_DOSAGE,
    sample_path: Path | None | object = DEFAULT_TEST_INPUT_PATH,
    covariate_path: Path | None | object = DEFAULT_TEST_INPUT_PATH,
    binary_kernel_config: typing.Any | None = None,
    requested_gpu_genotype_format: types.GpuGenotypeFormat | None = None,
    gpu_genotype_format: types.GpuGenotypeFormat = types.GpuGenotypeFormat.DOSAGE,
    score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    firth_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT64,
    output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    multi_phenotype_sample_mode: output.MultiPhenotypeSampleMode = output.MultiPhenotypeSampleMode.SINGLE_PHENOTYPE,
    phenotype_compute_group_id: str | None = None,
    sample_set_fingerprint: str | None = None,
    covariate_design_fingerprint: str | None = None,
    prediction_alignment_fingerprint: str | None = None,
    write_input_files: bool = True,
) -> output.CurrentRunManifestHeader:
    bgen_path = tmp_path / "study.bgen"
    selected_sample_path = (
        tmp_path / "study.sample" if sample_path is DEFAULT_TEST_INPUT_PATH else typing.cast("Path | None", sample_path)
    )
    phenotype_path = tmp_path / "phenotypes.tsv"
    resolved_covariate_path = (
        tmp_path / "covariates.tsv"
        if covariate_path is DEFAULT_TEST_INPUT_PATH
        else typing.cast("Path | None", covariate_path)
    )
    prediction_list_path = tmp_path / "predictions.list"
    loco_path = tmp_path / "trait.loco"
    if write_input_files:
        input_paths = [
            input_path
            for input_path in (
                bgen_path,
                selected_sample_path,
                phenotype_path,
                resolved_covariate_path,
            )
            if input_path is not None
        ]
        for input_path in input_paths:
            input_path.write_text(input_path.name, encoding="utf-8")
        loco_path.write_text("FID_IID F1_I1\n22 0.1\n", encoding="utf-8")
        prediction_list_path.write_text("trait  trait.loco\n", encoding="utf-8")
    fingerprint_cache = output.ManifestFileFingerprintCache()
    effective_requested_gpu_genotype_format = (
        gpu_genotype_format if requested_gpu_genotype_format is None else requested_gpu_genotype_format
    )
    return output.build_current_run_manifest_header(
        association_mode=association_mode,
        association_backend_kind=association_backend_kind,
        bgen_path=bgen_path,
        sample_path=selected_sample_path,
        phenotype_path=phenotype_path,
        phenotype_name="trait",
        covariate_path=resolved_covariate_path,
        covariate_names=("intercept", "age", "sex"),
        prediction_list_path=prediction_list_path,
        prediction_input_phenotype_names=("trait",),
        fingerprint_cache=fingerprint_cache,
        sample_count=4,
        variant_count=10,
        chunk_size=2,
        variant_limit=None,
        binary_correction_plan=types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.SCORE_ONLY,
            p_threshold=0.05,
            firth_se=False,
        ),
        trusted_no_missing_diploid=False,
        sample_key_mode=types.SampleKeyMode.IID,
        binary_kernel_config=binary_kernel_config,
        gpu_genotype_format=gpu_genotype_format,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        phenotype_compute_group_id=phenotype_compute_group_id,
        sample_set_fingerprint=sample_set_fingerprint,
        covariate_design_fingerprint=covariate_design_fingerprint,
        prediction_alignment_fingerprint=prediction_alignment_fingerprint,
        output_format=output_format,
        bgen_decode_tile_variant_count=64,
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode.CACHE_ON_MISS,
        jax_device=types.Device.CPU,
        jax_matmul_precision=None,
        requested_gpu_genotype_format=effective_requested_gpu_genotype_format,
        finalize_parquet=False,
        writer_thread_count=1,
        writer_queue_depth=1,
        chunks_per_arrow_file=16,
        arrow_compression=types.ArrowCompression.ZSTD,
        parquet_compression=types.ParquetCompression.NONE,
        output_statistic_dtype=output_statistic_dtype,
    )


def build_test_header(tmp_path: Path, **keyword_arguments: typing.Any) -> dict[str, typing.Any]:
    return output.current_run_manifest_header_to_mapping(build_test_header_object(tmp_path, **keyword_arguments))


def test_current_run_manifest_records_configured_x64_policy(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)

    assert current_header["jax_policy"]["enable_x64"] is True
    assert current_header["execution_plan"]["jax_policy"]["enable_x64"] is True


def test_current_run_manifest_records_dtype_policy(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path, score_dtype=types.FloatingPointDtype.FLOAT64)

    assert current_header["score_dtype"] == "float64"
    assert current_header["firth_dtype"] == "float64"
    assert current_header["execution_plan"]["score_dtype"] == "float64"
    assert current_header["execution_plan"]["firth_dtype"] == "float64"


def test_current_run_manifest_records_result_statistic_output_dtype(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    float64_header = build_test_header(tmp_path, output_statistic_dtype=types.FloatingPointDtype.FLOAT64)

    assert current_header["output_writer"]["result_statistic_dtype"] == "float32"
    assert current_header["execution_plan"]["output_writer"]["result_statistic_dtype"] == "float32"
    assert current_header["output_writer"]["parquet_compression"] == "none"
    assert current_header["execution_plan"]["output_writer"]["parquet_compression"] == "none"
    assert float64_header["output_writer"]["result_statistic_dtype"] == "float64"
    assert float64_header["execution_plan"]["output_writer"]["result_statistic_dtype"] == "float64"
    assert float64_header["execution_plan_hash"] != current_header["execution_plan_hash"]


def test_current_run_manifest_records_gpu_genotype_format(tmp_path: Path) -> None:
    current_header = build_test_header(
        tmp_path,
        association_backend_kind=types.AssociationBackendKind.JAX_PACKED8,
        gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
    )

    assert current_header["gpu_genotype_format"] == "packed8"
    assert current_header["execution_plan"]["gpu_genotype_format"] == "packed8"
    assert current_header["association_backend"] == {
        "kind": "jax_packed8",
        "association_mode": "regenie2_linear",
        "device": "cpu",
        "genotype_format": "packed8",
    }
    assert current_header["execution_plan"]["association_backend"] == {
        "kind": "jax_packed8",
        "association_mode": "regenie2_linear",
        "device": "cpu",
        "genotype_format": "packed8",
    }


def test_prepared_run_plan_payload_preserves_requested_and_resolved_gpu_formats(tmp_path: Path) -> None:
    current_header = build_test_header_object(
        tmp_path,
        association_backend_kind=types.AssociationBackendKind.JAX_PACKED8,
        requested_gpu_genotype_format=types.GpuGenotypeFormat.AUTO,
        gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
    )
    prepared_payload = output.build_prepared_run_plan_manifest_payload(current_header)

    compute_payload = typing.cast("dict[str, typing.Any]", prepared_payload["compute"])
    assert compute_payload["requested_gpu_genotype_format"] == "auto"
    assert compute_payload["resolved_gpu_genotype_format"] == "packed8"


def test_current_run_manifest_hashes_small_control_files(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)

    assert current_header["bgen"]["content_hash_algorithm"] == "metadata-only"
    assert current_header["bgen"]["content_sha256"] is None
    assert current_header["execution_plan"]["bgen"]["content_hash_algorithm"] == "metadata-only"
    assert current_header["execution_plan"]["bgen"]["content_sha256"] is None
    for manifest_field_name in ("sample", "phenotype_file", "covariate_file", "prediction_list"):
        input_path = Path(current_header[manifest_field_name]["path"])
        expected_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        assert current_header[manifest_field_name]["content_hash_algorithm"] == "sha256"
        assert current_header[manifest_field_name]["content_sha256"] == expected_hash
        assert current_header["execution_plan"][manifest_field_name]["content_sha256"] == expected_hash
    assert current_header["prediction_inputs"]["prediction_list"] == current_header["prediction_list"]
    assert current_header["execution_plan"]["prediction_inputs"] == current_header["prediction_inputs"]
    assert [loco_file["phenotype"] for loco_file in current_header["prediction_inputs"]["loco_files"]] == ["trait"]
    loco_file = current_header["prediction_inputs"]["loco_files"][0]
    loco_path = Path(loco_file["path"])
    assert loco_file["content_hash_algorithm"] == "sha256"
    assert loco_file["content_sha256"] == hashlib.sha256(loco_path.read_bytes()).hexdigest()


def test_current_run_manifest_allows_optional_unhashed_inputs(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path, sample_path=None, covariate_path=None)

    assert current_header["sample"] is None
    assert current_header["covariate_file"] is None
    assert current_header["execution_plan"]["sample"] is None
    assert current_header["execution_plan"]["covariate_file"] is None


def test_current_run_manifest_records_sample_set_contract(tmp_path: Path) -> None:
    current_header = build_test_header(
        tmp_path,
        multi_phenotype_sample_mode=output.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        phenotype_compute_group_id="group-fingerprint",
        sample_set_fingerprint="sample-fingerprint",
        covariate_design_fingerprint="covariate-fingerprint",
        prediction_alignment_fingerprint="prediction-fingerprint",
    )

    assert current_header["multi_phenotype_sample_mode"] == "per-phenotype"
    assert current_header["phenotype_compute_group_id"] == "group-fingerprint"
    assert current_header["sample_set_fingerprint"] == "sample-fingerprint"
    assert current_header["covariate_design_fingerprint"] == "covariate-fingerprint"
    assert current_header["prediction_alignment_fingerprint"] == "prediction-fingerprint"
    assert current_header["execution_plan"]["multi_phenotype_sample_mode"] == "per-phenotype"
    assert current_header["execution_plan"]["phenotype_compute_group_id"] == "group-fingerprint"
    assert current_header["execution_plan"]["sample_set_fingerprint"] == "sample-fingerprint"
    assert current_header["execution_plan"]["covariate_design_fingerprint"] == "covariate-fingerprint"
    assert current_header["execution_plan"]["prediction_alignment_fingerprint"] == "prediction-fingerprint"


def replace_file_text_preserving_size_and_mtime(path: Path, replacement_text: str) -> None:
    """Replace file text while preserving byte length and mtime."""
    original_stat = path.stat()
    original_length = len(path.read_bytes())
    replacement_bytes = replacement_text.encode("utf-8")
    assert len(replacement_bytes) == original_length
    path.write_bytes(replacement_bytes)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))


@pytest.mark.parametrize(
    ("manifest_field_name", "file_name", "replacement_text"),
    [
        ("sample", "study.sample", "STUDY.sample"),
        ("phenotype_file", "phenotypes.tsv", "PHENOTYPES.tsv"),
        ("covariate_file", "covariates.tsv", "COVARIATES.tsv"),
        ("prediction_list", "predictions.list", "trait\t trait.loco\n"),
    ],
)
def test_fast_resume_rejects_control_file_content_change_with_preserved_metadata(
    tmp_path: Path,
    manifest_field_name: str,
    file_name: str,
    replacement_text: str,
) -> None:
    manifest_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / f"output-{manifest_field_name}-content",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})

    replace_file_text_preserving_size_and_mtime(tmp_path / file_name, replacement_text)
    current_header = build_test_header(tmp_path, write_input_files=False)
    assert current_header[manifest_field_name]["size"] == manifest_header[manifest_field_name]["size"]
    assert current_header[manifest_field_name]["mtime_ns"] == manifest_header[manifest_field_name]["mtime_ns"]
    assert (
        current_header[manifest_field_name]["content_sha256"] != manifest_header[manifest_field_name]["content_sha256"]
    )
    assert current_header["execution_plan_hash"] != manifest_header["execution_plan_hash"]

    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / f"output-{manifest_field_name}-content",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )
    with pytest.raises(ValueError, match=rf"{manifest_field_name}\.content_sha256"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def test_fast_resume_rejects_loco_file_content_change_with_preserved_metadata(tmp_path: Path) -> None:
    manifest_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output-loco-content",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})

    replace_file_text_preserving_size_and_mtime(tmp_path / "trait.loco", "FID_IID F1_I1\n22 0.2\n")
    current_header = build_test_header(tmp_path, write_input_files=False)
    original_loco_file = manifest_header["prediction_inputs"]["loco_files"][0]
    current_loco_file = current_header["prediction_inputs"]["loco_files"][0]
    assert current_loco_file["path"] == original_loco_file["path"]
    assert current_loco_file["size"] == original_loco_file["size"]
    assert current_loco_file["mtime_ns"] == original_loco_file["mtime_ns"]
    assert current_loco_file["content_sha256"] != original_loco_file["content_sha256"]
    assert current_header["execution_plan_hash"] != manifest_header["execution_plan_hash"]

    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output-loco-content",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )
    with pytest.raises(ValueError, match=r"prediction_inputs\.loco_files\[0\]\.content_sha256"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def test_prediction_loco_fingerprints_hash_shared_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loco_path = tmp_path / "shared.loco"
    loco_path.write_text("FID_IID F1_I1\n22 0.1\n", encoding="utf-8")
    prediction_list_path = tmp_path / "predictions.list"
    prediction_list_path.write_text("first shared.loco\nsecond shared.loco\n", encoding="utf-8")
    observed_loco_paths: list[Path] = []
    real_build_file_fingerprint = output.build_file_fingerprint

    def record_file_fingerprint(
        path: Path | None,
        *,
        include_content_hash: bool,
    ) -> output.ManifestFileFingerprint | None:
        if path is not None and path.name == "shared.loco":
            observed_loco_paths.append(path)
        return real_build_file_fingerprint(path, include_content_hash=include_content_hash)

    monkeypatch.setattr(output, "build_file_fingerprint", record_file_fingerprint)

    loco_files = output.build_prediction_loco_file_fingerprints(
        prediction_list_path=prediction_list_path,
        phenotype_names=("first", "second"),
        fingerprint_cache=output.ManifestFileFingerprintCache(),
    )

    assert [loco_file.phenotype for loco_file in loco_files] == ["first", "second"]
    assert [loco_file.path for loco_file in loco_files] == [str(loco_path.resolve()), str(loco_path.resolve())]
    assert observed_loco_paths == [loco_path.resolve()]


def test_prediction_loco_fingerprints_are_stable_for_relative_and_absolute_paths(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID F1_I1\n22 0.1\n", encoding="utf-8")
    relative_prediction_list_path = tmp_path / "relative.list"
    absolute_prediction_list_path = tmp_path / "absolute.list"
    relative_prediction_list_path.write_text("trait trait.loco\n", encoding="utf-8")
    absolute_prediction_list_path.write_text(f"trait {loco_path}\n", encoding="utf-8")

    relative_loco_files = output.build_prediction_loco_file_fingerprints(
        prediction_list_path=relative_prediction_list_path,
        phenotype_names=("trait",),
        fingerprint_cache=output.ManifestFileFingerprintCache(),
    )
    absolute_loco_files = output.build_prediction_loco_file_fingerprints(
        prediction_list_path=absolute_prediction_list_path,
        phenotype_names=("trait",),
        fingerprint_cache=output.ManifestFileFingerprintCache(),
    )

    assert relative_loco_files == absolute_loco_files


def test_bgen_content_change_with_preserved_metadata_keeps_metadata_only_fingerprint(tmp_path: Path) -> None:
    manifest_header = build_test_header(tmp_path)

    replace_file_text_preserving_size_and_mtime(tmp_path / "study.bgen", "STUDY.bgen")
    current_header = build_test_header(tmp_path, write_input_files=False)

    assert current_header["bgen"] == manifest_header["bgen"]
    assert current_header["execution_plan"]["bgen"] == manifest_header["execution_plan"]["bgen"]
    assert current_header["execution_plan_hash"] == manifest_header["execution_plan_hash"]


def initialize_test_output_run(
    prepared_output_run: output.PreparedOutputRun,
    current_header: dict[str, typing.Any],
    *,
    resume: bool = False,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
) -> output.InitializedOutputRun:
    return output.initialize_output_run(
        output_run_paths=prepared_output_run.output_run_paths,
        existing_manifest=prepared_output_run.existing_manifest,
        current_header=current_header,
        resume=resume,
        resume_mode=resume_mode,
    )


def test_resolve_output_run_paths_appends_mode_suffix(tmp_path: Path) -> None:
    output_run_paths = resolve_test_output_run_paths(tmp_path / "results/output", AssociationMode.REGENIE2_LINEAR)
    assert output_run_paths.run_directory == tmp_path / "results/output.regenie2_linear.run"
    assert output_run_paths.chunks_directory == tmp_path / "results/output.regenie2_linear.run/parts"

    dotted_output_run_paths = resolve_test_output_run_paths(
        tmp_path / "results/output.v1",
        AssociationMode.REGENIE2_LINEAR,
    )
    assert dotted_output_run_paths.run_directory == tmp_path / "results/output.v1.regenie2_linear.run"

    literal_run_paths = resolve_test_output_run_paths(tmp_path / "results/output.run", AssociationMode.REGENIE2_LINEAR)
    assert literal_run_paths.run_directory == tmp_path / "results/output.run"

    arrow_run_paths = resolve_test_output_run_paths(
        tmp_path / "results/output",
        AssociationMode.REGENIE2_LINEAR,
        types.OutputFormat.ARROW,
    )
    assert arrow_run_paths.chunks_directory == tmp_path / "results/output.regenie2_linear.run/chunks"

    regenie_run_paths = resolve_test_output_run_paths(
        tmp_path / "results/output",
        AssociationMode.REGENIE2_LINEAR,
        types.OutputFormat.REGENIE,
    )
    assert regenie_run_paths.chunks_directory == tmp_path / "results/output.regenie2_linear.run/regenie"


def test_output_manifest_helpers_cover_empty_paths_and_invalid_json(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path / "missing")
    manifest_path = output.get_run_manifest_path(output_run_paths)
    manifest_path.write_text("[]", encoding="utf-8")

    assert output.build_chunk_file_name(7) == "chunk_000000007.arrow"
    assert output.build_file_fingerprint(None, include_content_hash=False) is None
    assert output.normalize_execution_plan_value(Path("relative/path")) == "relative/path"
    assert output.iter_sorted_chunk_file_paths(output_run_paths.chunks_directory) == ()
    with pytest.raises(ValueError, match="must contain a JSON object"):
        output.load_run_manifest(output_run_paths)


def test_native_manifest_compatibility_reports_missing_and_nested_differences() -> None:
    with pytest.raises(ValueError, match=r"root\.a"):
        output.validate_manifest_compatibility({"root": {"a": 1}}, {"root": {"b": 1}})
    with pytest.raises(ValueError, match=r"root\[0\]\.a"):
        output.validate_manifest_compatibility({"root": [{"a": 1}]}, {"root": [{"a": 2}]})
    with pytest.raises(ValueError, match="root"):
        output.validate_manifest_compatibility({"root": [1, 2]}, {"root": [1]})


def test_resume_rejects_older_manifest_schema_after_prediction_input_schema_bump(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    older_manifest = copy.deepcopy(current_header)
    older_manifest["schema_version"] = output.RUN_MANIFEST_SCHEMA_VERSION - 1
    older_manifest["execution_plan"]["manifest_schema_version"] = output.RUN_MANIFEST_SCHEMA_VERSION - 1
    older_manifest.pop("prediction_inputs")
    older_manifest["execution_plan"].pop("prediction_inputs")

    with pytest.raises(ValueError, match="manifest_schema_version"):
        output.validate_manifest_compatibility(older_manifest, current_header)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"committed_chunks": "bad"}, "committed_chunks field must be a list"),
        ({"committed_chunks": ["bad"]}, "committed chunk entries must be objects"),
        ({"committed_chunks": [{}]}, "missing chunk_identifier"),
    ],
)
def test_read_manifest_committed_chunk_identifiers_rejects_invalid_shapes(
    manifest: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        output.read_manifest_committed_chunk_identifiers(manifest)


def test_strict_manifest_core_wrappers_normalize_and_validate_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path / "chunks")
    output_run_paths.chunks_directory.mkdir()
    monkeypatch.setattr(output._core, "validate_strict_manifest_chunks", lambda *_arguments: [0, 2])
    monkeypatch.setattr(
        output._core,
        "repair_strict_manifest_chunk_commits",
        lambda *_arguments: json.dumps({"bad": "shape"}),
    )

    assert output.validate_strict_manifest_chunks(output_run_paths, {"committed_chunks": []}) == frozenset({0, 2})
    with pytest.raises(ValueError, match="repaired committed chunks must be a list"):
        output.repair_strict_manifest_chunk_commits(output_run_paths, {"committed_chunks": []})


def test_initialize_output_run_uses_existing_manifest_when_current_manifest_is_missing(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path / "chunks")
    output_run_paths.chunks_directory.mkdir()
    existing_manifest = {"schema_version": output.RUN_MANIFEST_SCHEMA_VERSION, "committed_chunks": []}
    initialized_output_run = output.initialize_output_run(
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        current_header={"schema_version": output.RUN_MANIFEST_SCHEMA_VERSION},
        resume=False,
        resume_mode=types.ResumeMode.FAST,
    )

    written_manifest = json.loads(output.get_run_manifest_path(output_run_paths).read_text(encoding="utf-8"))
    assert initialized_output_run.committed_chunk_identifiers == frozenset()
    assert written_manifest["schema_version"] == output.RUN_MANIFEST_SCHEMA_VERSION
    assert written_manifest["committed_chunks"] == []


def test_initialize_output_run_uses_prepared_manifest_without_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path / "chunks")
    output_run_paths.chunks_directory.mkdir()
    existing_manifest = {
        "schema_version": output.RUN_MANIFEST_SCHEMA_VERSION,
        "committed_chunks": [],
        "command": {"interface": "g regenie"},
    }

    def fail_manifest_reload(output_run_paths: output.OutputRunPaths) -> dict[str, typing.Any] | None:
        del output_run_paths
        message = "initialize_output_run reloaded the prepared manifest"
        raise AssertionError(message)

    monkeypatch.setattr(output, "load_run_manifest", fail_manifest_reload)

    initialized_output_run = output.initialize_output_run(
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        current_header={"schema_version": output.RUN_MANIFEST_SCHEMA_VERSION},
        resume=False,
        resume_mode=types.ResumeMode.FAST,
    )

    written_manifest = json.loads(output.get_run_manifest_path(output_run_paths).read_text(encoding="utf-8"))
    assert initialized_output_run.committed_chunk_identifiers == frozenset()
    assert written_manifest["command"] == {"interface": "g regenie"}
    assert written_manifest["committed_chunks"] == []


def test_initialize_output_run_rejects_existing_manifest_with_invalid_commits(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path / "chunks")
    output_run_paths.chunks_directory.mkdir()

    with pytest.raises(ValueError, match="committed_chunks field must be a list"):
        output.initialize_output_run(
            output_run_paths=output_run_paths,
            existing_manifest={"schema_version": output.RUN_MANIFEST_SCHEMA_VERSION, "committed_chunks": "bad"},
            current_header={"schema_version": output.RUN_MANIFEST_SCHEMA_VERSION},
            resume=False,
            resume_mode=types.ResumeMode.FAST,
        )


def test_initialize_output_run_rejects_resume_without_manifest(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path / "chunks")
    output_run_paths.chunks_directory.mkdir()

    with pytest.raises(ValueError, match=r"Resume requires run_manifest\.json"):
        output.initialize_output_run(
            output_run_paths=output_run_paths,
            existing_manifest=None,
            current_header={"schema_version": output.RUN_MANIFEST_SCHEMA_VERSION},
            resume=True,
            resume_mode=types.ResumeMode.FAST,
        )


def test_prepare_output_run_rejects_resume_without_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Resume requires run_manifest\.json"):
        prepare_test_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.REGENIE2_LINEAR,
            resume=True,
        )


def test_scan_committed_chunk_identifiers_reads_arrow_metadata(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(output_run_paths, AssociationMode.REGENIE2_LINEAR)

    assert output.scan_committed_chunk_identifiers(tmp_path) == frozenset({0, 2})


def test_prepare_output_run_rejects_non_empty_directory_without_resume(tmp_path: Path) -> None:
    run_directory = tmp_path / "output.regenie2_linear.run"
    run_directory.mkdir(parents=True)
    (run_directory / "stale_file.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists and is not empty"):
        prepare_test_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.REGENIE2_LINEAR,
            resume=False,
        )


def test_native_writer_uses_shared_schema_and_null_placeholders(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(output_run_paths, AssociationMode.REGENIE2_LINEAR)

    frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(tmp_path)[0])
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    chunk_schema = pyarrow.ipc.open_file(output.iter_sorted_chunk_file_paths(tmp_path)[0]).schema
    assert_step2_output_schema_contract(chunk_schema)
    assert b"g.output.chunk_commits" in (chunk_schema.metadata or {})
    assert frame.get_column("TEST").to_list() == ["ADD", "ADD", "ADD", "ADD"]
    assert frame.get_column("INFO").to_list() == [1.0, 1.0, 1.0, 1.0]
    assert frame.get_column("EXTRA").to_list() == [None, None, None, None]
    assert frame.get_column("CORRECTION_METHOD").to_list() == ["score", "score", "score", "score"]
    assert frame.get_column("CORRECTION_STATUS").to_list() == ["success", "success", "success", "success"]


def test_native_writer_writes_parquet_dataset_parts_with_footer_metadata(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(
        output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.PARQUET,
    )

    part_paths = output.iter_sorted_chunk_file_paths(tmp_path)
    assert [part_path.name for part_path in part_paths] == ["part_000000000_000000002.parquet"]
    assert not (tmp_path / "final.parquet").exists()
    frame = pl.read_parquet(part_paths[0])
    assert frame.columns == EXPECTED_FINAL_COLUMNS
    assert_step2_output_schema_contract(pq.ParquetFile(part_paths[0]).schema_arrow)
    assert frame.get_column("TEST").to_list() == ["ADD", "ADD", "ADD", "ADD"]
    assert frame.get_column("CORRECTION_METHOD").to_list() == ["score", "score", "score", "score"]
    assert frame.get_column("CORRECTION_STATUS").to_list() == ["success", "success", "success", "success"]
    parquet_metadata = pq.ParquetFile(part_paths[0]).metadata.metadata
    assert parquet_metadata is not None
    chunk_commits = json.loads(parquet_metadata[b"g.output.chunk_commits"])
    assert chunk_commits == [
        {
            "chunk_file_name": "part_000000000_000000002.parquet",
            "chunk_identifier": 0,
            "compression": "none",
            "output_format": "parquet",
            "row_count": 2,
            "variant_start_index": 0,
            "variant_stop_index": 2,
        },
        {
            "chunk_file_name": "part_000000000_000000002.parquet",
            "chunk_identifier": 2,
            "compression": "none",
            "output_format": "parquet",
            "row_count": 2,
            "variant_start_index": 2,
            "variant_stop_index": 4,
        },
    ]


def test_native_writer_writes_regenie_text_parts_and_final_output(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path, output_format=types.OutputFormat.REGENIE)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.REGENIE,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)

    write_native_chunks(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.REGENIE,
    )

    part_paths = output.iter_sorted_chunk_file_paths(prepared_output_run.output_run_paths.chunks_directory)
    assert [part_path.name for part_path in part_paths] == ["part_000000000_000000002.regenie"]
    part_lines = part_paths[0].read_text(encoding="utf-8").splitlines()
    assert part_lines[0].split("\t") == EXPECTED_FINAL_COLUMNS
    part_rows = [line.split("\t") for line in part_lines[1:]]
    assert len(part_rows) == 4
    assert {row[8] for row in part_rows} == {"ADD"}
    assert {row[9] for row in part_rows} == {"0.1"}
    assert {row[10] for row in part_rows} == {"0.01"}
    assert {row[11] for row in part_rows} == {"10"}
    assert {row[12] for row in part_rows} == {"5"}
    assert {row[13] for row in part_rows} == {"NA"}
    assert {row[14] for row in part_rows} == {"score"}
    assert {row[15] for row in part_rows} == {"success"}

    final_regenie_path = prepared_output_run.output_run_paths.run_directory / "final.regenie"
    assert final_regenie_path.exists()
    assert final_regenie_path.read_text(encoding="utf-8").splitlines() == part_lines
    sidecar = json.loads(part_paths[0].with_suffix(".regenie.json").read_text(encoding="utf-8"))
    assert [chunk["output_format"] for chunk in sidecar] == ["regenie", "regenie"]

    manifest = output.load_run_manifest(prepared_output_run.output_run_paths)
    assert manifest is not None
    assert manifest["finalized"] is True
    assert manifest["final_output_format"] == "regenie"
    assert manifest["final_regenie"] == str(final_regenie_path)
    assert manifest["final_row_count"] == 4

    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.REGENIE,
        resume=True,
        resume_mode=types.ResumeMode.STRICT,
    )
    initialized_output_run = initialize_test_output_run(
        resumed_output_run,
        current_header,
        resume=True,
        resume_mode=types.ResumeMode.STRICT,
    )
    assert initialized_output_run.committed_chunk_identifiers == frozenset({0, 2})


def test_native_binary_writer_writes_regenie_text_extra_labels(tmp_path: Path) -> None:
    current_header = build_test_header(
        tmp_path,
        association_mode=AssociationMode.REGENIE2_BINARY,
        output_format=types.OutputFormat.REGENIE,
    )
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_BINARY,
        output_format=types.OutputFormat.REGENIE,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)

    write_native_chunks(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_BINARY,
        output_format=types.OutputFormat.REGENIE,
        extra_code_value=types.BinaryExtraCode.TEST_FAIL.value,
    )

    final_regenie_path = prepared_output_run.output_run_paths.run_directory / "final.regenie"
    final_rows = [line.split("\t") for line in final_regenie_path.read_text(encoding="utf-8").splitlines()[1:]]
    assert len(final_rows) == 4
    assert {row[13] for row in final_rows} == {"TEST_FAIL"}
    assert {row[14] for row in final_rows} == {"firth_approximate"}
    assert {row[15] for row in final_rows} == {"failed"}


def test_native_writer_records_output_stage_timings_when_requested(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    writer_session = create_test_output_writer_session(
        output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=False,
        output_format=types.OutputFormat.ARROW,
        chunks_per_arrow_file=16,
        arrow_compression=types.ArrowCompression.ZSTD,
        parquet_compression=types.ParquetCompression.NONE,
        collect_stage_timings=True,
    )
    callback = NativeChunkWritingCallback(writer_session)
    try:
        engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
        engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), callback)
        writer_session.finish()
    except Exception:
        writer_session.abort()
        raise

    timing_payload = json.loads((tmp_path / "output_stage_timings.json").read_text(encoding="utf-8"))
    assert timing_payload["stage_counts"]["rust_output_metadata_clone"] == 0
    assert timing_payload["stage_counts"]["rust_output_result_buffer_copy"] == 0
    assert timing_payload["stage_counts"]["rust_output_writer_record_batch_try_new"] == 2
    assert timing_payload["stage_counts"]["rust_output_writer_arrow_file_write"] == 1
    assert "rust_output_writer_metadata_arrays" in timing_payload["stage_totals_seconds"]
    assert "rust_output_writer_arrow_batch_write" in timing_payload["stage_totals_seconds"]
    assert timing_payload["output_metrics"]["writer_chunk_count"] == 2
    assert timing_payload["output_metrics"]["writer_row_count"] == 4
    assert timing_payload["output_metrics"]["writer_arrow_file_bytes"] > 0


def test_native_binary_writer_maps_successful_correction_extra_code_to_null(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(
        output_run_paths, AssociationMode.REGENIE2_BINARY, extra_code_value=types.BinaryExtraCode.FIRTH.value
    )

    frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(tmp_path)[0])
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    assert frame.get_column("EXTRA").to_list() == [None, None, None, None]
    assert frame.get_column("CORRECTION_METHOD").to_list() == [
        "firth_approximate",
        "firth_approximate",
        "firth_approximate",
        "firth_approximate",
    ]
    assert frame.get_column("CORRECTION_STATUS").to_list() == ["success", "success", "success", "success"]


def test_native_binary_writer_maps_test_fail_extra_code_to_label(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(
        output_run_paths, AssociationMode.REGENIE2_BINARY, extra_code_value=types.BinaryExtraCode.TEST_FAIL.value
    )

    frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(tmp_path)[0])
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    assert frame.get_column("EXTRA").to_list() == ["TEST_FAIL", "TEST_FAIL", "TEST_FAIL", "TEST_FAIL"]
    assert frame.get_column("CORRECTION_METHOD").to_list() == [
        "firth_approximate",
        "firth_approximate",
        "firth_approximate",
        "firth_approximate",
    ]
    assert frame.get_column("CORRECTION_STATUS").to_list() == ["failed", "failed", "failed", "failed"]


def test_public_native_writer_copies_numpy_arrays_before_enqueue(tmp_path: Path) -> None:
    capture_callback = NativeChunkCaptureCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
    engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), capture_callback)
    metadata = capture_callback.require_metadata()
    chunk_stats = capture_callback.require_chunk_stats()
    row_count = metadata.variant_stop_index - metadata.variant_start_index
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    writer_session = create_test_output_writer_session(
        output_run_paths,
        AssociationMode.REGENIE2_BINARY,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=False,
        output_format=types.OutputFormat.ARROW,
        chunks_per_arrow_file=16,
        arrow_compression=types.ArrowCompression.ZSTD,
        parquet_compression=types.ParquetCompression.NONE,
    )
    beta = np.full(row_count, 0.125, dtype=np.float32)
    standard_error = np.full(row_count, 0.025, dtype=np.float32)
    chi_squared = np.full(row_count, 8.0, dtype=np.float32)
    log10_p_value = np.full(row_count, 3.0, dtype=np.float32)
    extra_code = np.full(row_count, types.BinaryExtraCode.TEST_FAIL.value, dtype=np.int32)
    try:
        writer_session.write_regenie2_native_chunk(
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=beta,
            standard_error=standard_error,
            chi_squared=chi_squared,
            log10_p_value=log10_p_value,
            extra_code=extra_code,
        )
        beta.fill(99.0)
        standard_error.fill(99.0)
        chi_squared.fill(99.0)
        log10_p_value.fill(99.0)
        extra_code.fill(99)
        writer_session.finish()
    except Exception:
        with contextlib.suppress(Exception):
            writer_session.abort()
        raise

    frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(tmp_path)[0])
    np.testing.assert_allclose(frame.get_column("BETA").to_numpy(), np.full(row_count, 0.125, dtype=np.float32))
    np.testing.assert_allclose(frame.get_column("SE").to_numpy(), np.full(row_count, 0.025, dtype=np.float32))
    np.testing.assert_allclose(frame.get_column("CHISQ").to_numpy(), np.full(row_count, 8.0, dtype=np.float32))
    np.testing.assert_allclose(frame.get_column("LOG10P").to_numpy(), np.full(row_count, 3.0, dtype=np.float32))
    assert frame.get_column("EXTRA").to_list() == ["TEST_FAIL"] * row_count
    assert frame.get_column("CORRECTION_METHOD").to_list() == ["firth_approximate"] * row_count
    assert frame.get_column("CORRECTION_STATUS").to_list() == ["failed"] * row_count


def test_public_native_writer_preserves_float64_output_statistics(tmp_path: Path) -> None:
    capture_callback = NativeChunkCaptureCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
    engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), capture_callback)
    metadata = capture_callback.require_metadata()
    chunk_stats = capture_callback.require_chunk_stats()
    row_count = metadata.variant_stop_index - metadata.variant_start_index
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    writer_session = create_test_output_writer_session(
        output_run_paths,
        AssociationMode.REGENIE2_BINARY,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=False,
        output_format=types.OutputFormat.ARROW,
        chunks_per_arrow_file=16,
        arrow_compression=types.ArrowCompression.ZSTD,
        parquet_compression=types.ParquetCompression.NONE,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT64,
    )
    high_precision_beta = np.nextafter(np.float64(0.125), np.float64(1.0))
    beta = np.full(row_count, high_precision_beta, dtype=np.float64)
    standard_error = np.full(row_count, np.nextafter(np.float64(0.025), np.float64(1.0)), dtype=np.float64)
    chi_squared = np.full(row_count, np.nextafter(np.float64(8.0), np.float64(9.0)), dtype=np.float64)
    log10_p_value = np.full(row_count, np.nextafter(np.float64(3.0), np.float64(4.0)), dtype=np.float64)
    try:
        writer_session.write_regenie2_native_chunk_f64(
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=beta,
            standard_error=standard_error,
            chi_squared=chi_squared,
            log10_p_value=log10_p_value,
            extra_code=None,
        )
        writer_session.finish()
    except Exception:
        with contextlib.suppress(Exception):
            writer_session.abort()
        raise

    chunk_path = output.iter_sorted_chunk_file_paths(tmp_path)[0]
    assert_step2_output_schema_contract(pyarrow.ipc.open_file(chunk_path).schema, pa.float64())
    frame = pl.read_ipc(chunk_path)
    observed_beta = frame.get_column("BETA").to_numpy()
    assert observed_beta.dtype == np.float64
    np.testing.assert_array_equal(observed_beta, beta)
    assert np.float32(observed_beta[0]) != observed_beta[0]


def test_public_multi_native_writer_copies_numpy_rows_before_enqueue(tmp_path: Path) -> None:
    capture_callback = NativeChunkCaptureCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
    engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), capture_callback)
    metadata = capture_callback.require_metadata()
    chunk_stats = capture_callback.require_chunk_stats()
    row_count = metadata.variant_stop_index - metadata.variant_start_index
    writer_sessions = []
    writer_run_paths = [
        output.OutputRunPaths(tmp_path / "trait-zero", tmp_path / "trait-zero"),
        output.OutputRunPaths(tmp_path / "trait-one", tmp_path / "trait-one"),
    ]
    for output_run_paths in writer_run_paths:
        output_run_paths.chunks_directory.mkdir()
        writer_sessions.append(
            create_test_output_writer_session(
                output_run_paths,
                AssociationMode.REGENIE2_BINARY,
                writer_thread_count=1,
                writer_queue_depth=1,
                finalize_parquet=False,
                output_format=types.OutputFormat.ARROW,
                chunks_per_arrow_file=16,
                arrow_compression=types.ArrowCompression.ZSTD,
                parquet_compression=types.ParquetCompression.NONE,
            )
        )

    beta = np.ascontiguousarray(
        np.stack(
            [
                np.full(row_count, 0.25, dtype=np.float32),
                np.full(row_count, 0.5, dtype=np.float32),
            ],
            axis=0,
        )
    )
    standard_error = np.full((2, row_count), 0.05, dtype=np.float32)
    chi_squared = np.full((2, row_count), 6.0, dtype=np.float32)
    log10_p_value = np.full((2, row_count), 2.0, dtype=np.float32)
    extra_code = np.ascontiguousarray(
        np.stack(
            [
                np.full(row_count, types.BinaryExtraCode.TEST_FAIL.value, dtype=np.int32),
                np.full(row_count, types.BinaryExtraCode.FIRTH.value, dtype=np.int32),
            ],
            axis=0,
        )
    )
    try:
        _core.write_regenie2_multi_native_chunk(
            writer_sessions=writer_sessions,
            active_trait_indices=[0, 1],
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=beta,
            standard_error=standard_error,
            chi_squared=chi_squared,
            log10_p_value=log10_p_value,
            extra_code=extra_code,
        )
        beta.fill(99.0)
        standard_error.fill(99.0)
        chi_squared.fill(99.0)
        log10_p_value.fill(99.0)
        extra_code.fill(99)
        for writer_session in writer_sessions:
            writer_session.finish()
    except Exception:
        for writer_session in writer_sessions:
            with contextlib.suppress(Exception):
                writer_session.abort()
        raise

    first_frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(writer_run_paths[0].chunks_directory)[0])
    second_frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(writer_run_paths[1].chunks_directory)[0])
    np.testing.assert_allclose(first_frame.get_column("BETA").to_numpy(), np.full(row_count, 0.25, dtype=np.float32))
    np.testing.assert_allclose(second_frame.get_column("BETA").to_numpy(), np.full(row_count, 0.5, dtype=np.float32))
    np.testing.assert_allclose(first_frame.get_column("SE").to_numpy(), np.full(row_count, 0.05, dtype=np.float32))
    np.testing.assert_allclose(second_frame.get_column("SE").to_numpy(), np.full(row_count, 0.05, dtype=np.float32))
    assert first_frame.get_column("EXTRA").to_list() == ["TEST_FAIL"] * row_count
    assert second_frame.get_column("EXTRA").to_list() == [None] * row_count
    assert first_frame.get_column("CORRECTION_METHOD").to_list() == ["firth_approximate"] * row_count
    assert first_frame.get_column("CORRECTION_STATUS").to_list() == ["failed"] * row_count
    assert second_frame.get_column("CORRECTION_METHOD").to_list() == ["firth_approximate"] * row_count
    assert second_frame.get_column("CORRECTION_STATUS").to_list() == ["success"] * row_count


def test_public_multi_native_writer_preserves_float64_output_statistics(tmp_path: Path) -> None:
    capture_callback = NativeChunkCaptureCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
    engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), capture_callback)
    metadata = capture_callback.require_metadata()
    chunk_stats = capture_callback.require_chunk_stats()
    row_count = metadata.variant_stop_index - metadata.variant_start_index
    writer_sessions = []
    writer_run_paths = [
        output.OutputRunPaths(tmp_path / "trait-zero-f64", tmp_path / "trait-zero-f64"),
        output.OutputRunPaths(tmp_path / "trait-one-f64", tmp_path / "trait-one-f64"),
    ]
    for output_run_paths in writer_run_paths:
        output_run_paths.chunks_directory.mkdir()
        writer_sessions.append(
            create_test_output_writer_session(
                output_run_paths,
                AssociationMode.REGENIE2_BINARY,
                writer_thread_count=1,
                writer_queue_depth=1,
                finalize_parquet=False,
                output_format=types.OutputFormat.ARROW,
                chunks_per_arrow_file=16,
                arrow_compression=types.ArrowCompression.ZSTD,
                parquet_compression=types.ParquetCompression.NONE,
                output_statistic_dtype=types.FloatingPointDtype.FLOAT64,
            )
        )

    beta = np.ascontiguousarray(
        np.stack(
            [
                np.full(row_count, np.nextafter(np.float64(0.25), np.float64(1.0)), dtype=np.float64),
                np.full(row_count, np.nextafter(np.float64(0.5), np.float64(1.0)), dtype=np.float64),
            ],
            axis=0,
        )
    )
    standard_error = np.full((2, row_count), np.nextafter(np.float64(0.05), np.float64(1.0)), dtype=np.float64)
    chi_squared = np.full((2, row_count), np.nextafter(np.float64(6.0), np.float64(7.0)), dtype=np.float64)
    log10_p_value = np.full((2, row_count), np.nextafter(np.float64(2.0), np.float64(3.0)), dtype=np.float64)
    try:
        _core.write_regenie2_multi_native_chunk_f64(
            writer_sessions=writer_sessions,
            active_trait_indices=[0, 1],
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=beta,
            standard_error=standard_error,
            chi_squared=chi_squared,
            log10_p_value=log10_p_value,
            extra_code=None,
        )
        for writer_session in writer_sessions:
            writer_session.finish()
    except Exception:
        for writer_session in writer_sessions:
            with contextlib.suppress(Exception):
                writer_session.abort()
        raise

    first_chunk_path = output.iter_sorted_chunk_file_paths(writer_run_paths[0].chunks_directory)[0]
    second_chunk_path = output.iter_sorted_chunk_file_paths(writer_run_paths[1].chunks_directory)[0]
    assert_step2_output_schema_contract(pyarrow.ipc.open_file(first_chunk_path).schema, pa.float64())
    assert_step2_output_schema_contract(pyarrow.ipc.open_file(second_chunk_path).schema, pa.float64())
    first_beta = pl.read_ipc(first_chunk_path).get_column("BETA").to_numpy()
    second_beta = pl.read_ipc(second_chunk_path).get_column("BETA").to_numpy()
    np.testing.assert_array_equal(first_beta, beta[0])
    np.testing.assert_array_equal(second_beta, beta[1])


def test_initialize_output_run_compatible_resume_preserves_committed_chunks(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    write_native_chunks(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.PARQUET,
    )

    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )
    initialized_output_run = initialize_test_output_run(resumed_output_run, current_header, resume=True)

    assert initialized_output_run.committed_chunk_identifiers == frozenset({0, 2})
    manifest = output.load_run_manifest(prepared_output_run.output_run_paths)
    assert manifest is not None
    assert [chunk["chunk_identifier"] for chunk in manifest["committed_chunks"]] == [0, 2]


def test_initialize_output_run_preserves_preinitialized_metadata(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(
        prepared_output_run.output_run_paths,
        {
            "command": {
                "effective_config": str(prepared_output_run.output_run_paths.run_directory / "effective_config.toml")
            },
            "runtime": {"device": "cpu"},
        },
    )

    initialize_test_output_run(prepared_output_run, current_header)

    manifest = output.load_run_manifest(prepared_output_run.output_run_paths)
    assert manifest is not None
    assert manifest["command"]["effective_config"].endswith("effective_config.toml")
    assert manifest["runtime"] == {"device": "cpu"}
    assert manifest["phenotype_name"] == "trait"
    assert manifest["committed_chunks"] == []


def test_prepare_output_run_strict_resume_validates_manifest_chunks(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    write_native_chunks(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.PARQUET,
    )

    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
        resume_mode=output.types.ResumeMode.STRICT,
    )
    initialized_output_run = initialize_test_output_run(
        resumed_output_run,
        current_header,
        resume=True,
        resume_mode=output.types.ResumeMode.STRICT,
    )

    assert initialized_output_run.committed_chunk_identifiers == frozenset({0, 2})


def test_strict_resume_repairs_manifest_commits_from_arrow_metadata(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path, output_format=types.OutputFormat.ARROW)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.ARROW,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    write_native_chunks(prepared_output_run.output_run_paths, AssociationMode.REGENIE2_LINEAR)
    manifest = output.load_run_manifest(prepared_output_run.output_run_paths)
    assert manifest is not None
    manifest["committed_chunks"] = []
    output.write_run_manifest(prepared_output_run.output_run_paths, manifest)

    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.ARROW,
        resume=True,
        resume_mode=output.types.ResumeMode.STRICT,
    )
    initialized_output_run = initialize_test_output_run(
        resumed_output_run,
        current_header,
        resume=True,
        resume_mode=output.types.ResumeMode.STRICT,
    )

    assert initialized_output_run.committed_chunk_identifiers == frozenset({0, 2})
    repaired_manifest = output.load_run_manifest(prepared_output_run.output_run_paths)
    assert repaired_manifest is not None
    assert repaired_manifest["committed_chunks"] == [
        {
            "chunk_file_name": "chunk_000000000_000000002.arrow",
            "chunk_identifier": 0,
            "compression": "zstd",
            "output_format": "arrow",
            "row_count": 2,
            "variant_start_index": 0,
            "variant_stop_index": 2,
        },
        {
            "chunk_file_name": "chunk_000000000_000000002.arrow",
            "chunk_identifier": 2,
            "compression": "zstd",
            "output_format": "arrow",
            "row_count": 2,
            "variant_start_index": 2,
            "variant_stop_index": 4,
        },
    ]


def test_fast_resume_trusts_only_manifest_committed_chunks(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    (prepared_output_run.output_run_paths.chunks_directory / "chunk_000000000.arrow").write_bytes(b"staged")

    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )
    initialized_output_run = initialize_test_output_run(resumed_output_run, current_header, resume=True)

    assert initialized_output_run.committed_chunk_identifiers == frozenset()
    manifest = output.load_run_manifest(prepared_output_run.output_run_paths)
    assert manifest is not None
    assert manifest["resume_policy"] == output.RESUME_POLICY
    assert manifest["committed_chunks"] == []


def test_initialize_output_run_rejects_incompatible_manifest_even_in_fast_mode(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(
        prepared_output_run.output_run_paths,
        {**current_header, "association_mode": AssociationMode.REGENIE2_BINARY.value, "committed_chunks": []},
    )
    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match="association_mode"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def build_test_binary_kernel_config() -> regenie2_binary_config.BinaryKernelConfig:
    """Build a non-default binary kernel config for manifest tests."""
    return regenie2_binary_config.BinaryKernelConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=1.0e-7,
            minimum_variance=1.0e-9,
            relative_variance_tolerance=2.0e-6,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=13,
            coefficient_tolerance=1.0e-5,
        ),
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(
            batch_size=7,
            candidate_capacity=11,
        ),
        approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
            maximum_iterations=17,
            gradient_tolerance=2.0e-5,
            coefficient_tolerance=3.0e-5,
            likelihood_tolerance=4.0e-5,
            maximum_step_size=6.0,
            pseudo_maximum_iterations=19,
            pseudo_inner_maximum_iterations=23,
            newton_raphson_zero_start_iterations=29,
            line_search_maximum_attempts=31,
            step_halving_maximum_attempts=37,
            initial_response_scale=4.5,
            sparse_carrier_dosage_threshold=1.0e-3,
            step_halving_scale=0.25,
            use_block_math=True,
        ),
        null_firth=regenie2_binary_config.NullFirthConfig(
            maximum_iterations=41,
            gradient_tolerance=5.0e-5,
            maximum_step_size=7.0,
            fallback_iteration_multiplier=43,
            fallback_step_divisor=11.0,
            line_search_maximum_attempts=47,
            step_halving_scale=0.125,
        ),
    )


@pytest.mark.parametrize(
    ("field_name", "replacement_value"),
    [
        ("bgen", {"path": "/different/study.bgen", "size": 1, "mtime_ns": 2}),
        ("sample", {"path": "/different/study.sample", "size": 1, "mtime_ns": 2}),
        ("phenotype_file", {"path": "/different/phenotypes.tsv", "size": 1, "mtime_ns": 2}),
        ("phenotype_name", "other_trait"),
        ("covariate_file", {"path": "/different/covariates.tsv", "size": 1, "mtime_ns": 2}),
        ("covariate_names", ["intercept", "age"]),
        ("prediction_list", {"path": "/different/predictions.list", "size": 1, "mtime_ns": 2}),
        ("sample_count", 3),
        ("variant_count", 11),
        ("chunk_size", 4),
        (
            "binary_correction_plan",
            {"method": "firth_approximate", "p_threshold": 0.01, "firth_se": False},
        ),
        ("binary_kernel_config", output.normalize_execution_plan_value(build_test_binary_kernel_config())),
        ("trusted_no_missing_diploid", True),
        ("trusted_bgen_validation_mode", "assume_validated"),
        ("sample_key_mode", "fid_iid"),
        ("output_schema_version", 1),
        (
            "association_backend",
            {
                "kind": "jax_packed8",
                "association_mode": "regenie2_linear",
                "device": "cpu",
                "genotype_format": "packed8",
            },
        ),
        ("bgen_decode_tile_variant_count", 128),
        ("jax_policy", {"device": "gpu", "enable_x64": True, "matmul_precision": "highest"}),
        ("gpu_genotype_format", "packed8"),
        ("score_dtype", "float64"),
        ("firth_dtype", "float32"),
        ("multi_phenotype_sample_mode", "complete-case"),
        ("phenotype_compute_group_id", "different-group"),
        ("sample_set_fingerprint", "different-sample-set"),
        ("covariate_design_fingerprint", "different-covariate-design"),
        ("prediction_alignment_fingerprint", "different-prediction-alignment"),
        (
            "output_writer",
            {
                "output_format": "arrow",
                "finalize_parquet": False,
                "writer_thread_count": 2,
                "writer_queue_depth": 3,
                "chunks_per_arrow_file": 2,
                "arrow_compression": "none",
            },
        ),
        ("variant_limit", 8),
    ],
)
def test_initialize_output_run_rejects_manifest_header_mismatch(
    tmp_path: Path,
    field_name: str,
    replacement_value: typing.Any,
) -> None:
    current_header = build_test_header(tmp_path)
    manifest_header = dict(current_header)
    manifest_header[field_name] = replacement_value
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / f"output-{field_name}",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(
        prepared_output_run.output_run_paths,
        {**manifest_header, "committed_chunks": []},
    )
    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / f"output-{field_name}",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match=field_name):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def test_initialize_output_run_rejects_output_statistic_dtype_resume(tmp_path: Path) -> None:
    manifest_header = build_test_header(tmp_path, output_statistic_dtype=types.FloatingPointDtype.FLOAT64)
    current_header = build_test_header(
        tmp_path,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
        write_input_files=False,
    )
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output-statistic-dtype",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output-statistic-dtype",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match=r"output_writer\.result_statistic_dtype"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def test_initialize_output_run_rejects_per_phenotype_complete_case_resume(tmp_path: Path) -> None:
    current_header = build_test_header(
        tmp_path,
        multi_phenotype_sample_mode=output.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        phenotype_compute_group_id="per-phenotype-group",
        sample_set_fingerprint="per-phenotype-samples",
    )
    manifest_header = copy.deepcopy(current_header)
    manifest_header["multi_phenotype_sample_mode"] = "complete-case"
    manifest_header["phenotype_compute_group_id"] = "complete-case-group"
    manifest_header["sample_set_fingerprint"] = "complete-case-samples"
    manifest_header["execution_plan"]["multi_phenotype_sample_mode"] = "complete-case"
    manifest_header["execution_plan"]["phenotype_compute_group_id"] = "complete-case-group"
    manifest_header["execution_plan"]["sample_set_fingerprint"] = "complete-case-samples"
    manifest_header["execution_plan_hash"] = output.build_execution_plan_hash(manifest_header["execution_plan"])
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output-sample-mode",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output-sample-mode",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match="multi_phenotype_sample_mode"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


@pytest.mark.parametrize(
    ("nested_field_name", "replacement_value"),
    [
        ("association_mode", "regenie2_binary"),
        ("phenotype_name", "other_trait"),
        ("covariate_names", ["intercept", "age"]),
        ("prediction_list", {"path": "/different/predictions.list", "size": 1, "mtime_ns": 2}),
        (
            "binary_correction_plan",
            {"method": "firth_approximate", "p_threshold": 0.01, "firth_se": False},
        ),
        ("binary_kernel_config", output.normalize_execution_plan_value(build_test_binary_kernel_config())),
        ("sample_key_mode", "fid_iid"),
        ("output_schema_version", 1),
        (
            "association_backend",
            {
                "kind": "jax_packed8",
                "association_mode": "regenie2_linear",
                "device": "cpu",
                "genotype_format": "packed8",
            },
        ),
        ("trusted_no_missing_diploid", True),
        ("bgen_decode_tile_variant_count", 128),
        ("gpu_genotype_format", "packed8"),
        ("score_dtype", "float64"),
        ("firth_dtype", "float32"),
        ("multi_phenotype_sample_mode", "complete-case"),
        ("phenotype_compute_group_id", "different-group"),
        ("sample_set_fingerprint", "different-sample-set"),
        ("covariate_design_fingerprint", "different-covariate-design"),
        ("prediction_alignment_fingerprint", "different-prediction-alignment"),
        ("chunk_size", 4),
    ],
)
def test_initialize_output_run_rejects_execution_plan_hash_mismatch(
    tmp_path: Path,
    nested_field_name: str,
    replacement_value: typing.Any,
) -> None:
    current_header = build_test_header(tmp_path)
    manifest_header = copy.deepcopy(current_header)
    manifest_header["execution_plan"][nested_field_name] = replacement_value
    manifest_header["execution_plan_hash"] = output.build_execution_plan_hash(manifest_header["execution_plan"])
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / f"output-execution-plan-{nested_field_name}",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / f"output-execution-plan-{nested_field_name}",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match=rf"execution_plan\.{nested_field_name}"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def test_initialize_output_run_rejects_execution_plan_hash_only_mismatch(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    manifest_header = dict(current_header)
    manifest_header["execution_plan_hash"] = "0" * 64
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output-execution-plan-hash",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output-execution-plan-hash",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match="execution_plan_hash"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def test_initialize_output_run_rejects_old_schema_manifest(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    manifest_header = dict(current_header)
    manifest_header["schema_version"] = 1
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match="schema_version"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def test_initialize_output_run_rejects_missing_manifest_header_field(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    manifest_header = dict(current_header)
    del manifest_header["prediction_list"]
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match="prediction_list"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def test_initialize_output_run_incompatible_resume_preserves_manifest_bytes(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    manifest_header = dict(current_header)
    manifest_header["chunk_size"] = 4
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    manifest_path = output.get_run_manifest_path(prepared_output_run.output_run_paths)
    original_manifest_bytes = manifest_path.read_bytes()
    resumed_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match="chunk_size"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)

    assert manifest_path.read_bytes() == original_manifest_bytes


def test_prepare_output_run_resume_requires_manifest(tmp_path: Path) -> None:
    run_directory = tmp_path / "output.regenie2_linear.run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)

    with pytest.raises(ValueError, match=r"Resume requires run_manifest\.json"):
        prepare_test_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.REGENIE2_LINEAR,
            resume=True,
        )


def test_prepare_output_run_strict_resume_requires_manifest(tmp_path: Path) -> None:
    run_directory = tmp_path / "output.regenie2_linear.run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)

    with pytest.raises(ValueError, match=r"Resume requires run_manifest\.json"):
        prepare_test_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.REGENIE2_LINEAR,
            resume=True,
            resume_mode=output.types.ResumeMode.STRICT,
        )


def test_chunk_arrow_schema_is_shared_between_linear_and_binary(tmp_path: Path) -> None:
    linear_run_paths = output.OutputRunPaths(tmp_path / "linear", tmp_path / "linear")
    binary_run_paths = output.OutputRunPaths(tmp_path / "binary", tmp_path / "binary")
    linear_run_paths.chunks_directory.mkdir()
    binary_run_paths.chunks_directory.mkdir()
    write_native_chunks(linear_run_paths, AssociationMode.REGENIE2_LINEAR)
    write_native_chunks(
        binary_run_paths, AssociationMode.REGENIE2_BINARY, extra_code_value=types.BinaryExtraCode.TEST_FAIL.value
    )

    linear_schema = pyarrow.ipc.open_file(
        output.iter_sorted_chunk_file_paths(linear_run_paths.chunks_directory)[0],
    ).schema
    binary_schema = pyarrow.ipc.open_file(
        output.iter_sorted_chunk_file_paths(binary_run_paths.chunks_directory)[0],
    ).schema
    assert linear_schema == binary_schema
    assert linear_schema.names == EXPECTED_CHUNK_COLUMNS
    assert linear_schema.field("INFO").nullable
    assert linear_schema.field("EXTRA").nullable
    assert linear_schema.field("CORRECTION_METHOD").nullable
    assert linear_schema.field("CORRECTION_STATUS").nullable


@pytest.mark.parametrize(
    ("association_mode", "extra_code_value"),
    (
        (AssociationMode.REGENIE2_LINEAR, None),
        (AssociationMode.REGENIE2_BINARY, types.BinaryExtraCode.TEST_FAIL.value),
    ),
)
@pytest.mark.parametrize("output_format", (types.OutputFormat.ARROW, types.OutputFormat.PARQUET))
@pytest.mark.parametrize(
    ("output_statistic_dtype", "expected_statistic_schema_dtype"),
    (
        (types.FloatingPointDtype.FLOAT32, pa.float32()),
        (types.FloatingPointDtype.FLOAT64, pa.float64()),
    ),
)
def test_regenie2_step2_output_schema_contract(
    tmp_path: Path,
    association_mode: AssociationMode,
    extra_code_value: int | None,
    output_format: types.OutputFormat,
    output_statistic_dtype: types.FloatingPointDtype,
    expected_statistic_schema_dtype: pa.DataType,
) -> None:
    """Assert stable schema contract for Step 2 final and intermediate outputs."""
    run_directory = tmp_path / f"{association_mode.value}-{output_format.value}"
    current_header = build_test_header(
        tmp_path,
        association_mode=association_mode,
        output_format=output_format,
        output_statistic_dtype=output_statistic_dtype,
    )
    prepared_output_run = prepare_test_output_run(
        output_root=run_directory,
        association_mode=association_mode,
        output_format=output_format,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    output_run_paths = prepared_output_run.output_run_paths

    write_native_chunks(
        output_run_paths,
        association_mode,
        output_format=output_format,
        extra_code_value=extra_code_value,
        output_statistic_dtype=output_statistic_dtype,
    )

    for chunk_path in output.iter_sorted_chunk_file_paths(output_run_paths.chunks_directory):
        if output_format == types.OutputFormat.ARROW:
            chunk_schema = pyarrow.ipc.open_file(chunk_path).schema
        else:
            chunk_schema = pq.ParquetFile(chunk_path).schema_arrow
        assert_step2_output_schema_contract(chunk_schema, expected_statistic_schema_dtype)

    final_parquet_path = finalize_test_chunks_to_parquet(
        output_run_paths,
        association_mode,
        output_format=output_format,
    )
    final_parquet_schema = pq.ParquetFile(final_parquet_path).schema_arrow
    assert_step2_output_schema_contract(final_parquet_schema, expected_statistic_schema_dtype)


def test_finalize_chunks_to_parquet_projects_technical_columns_away(tmp_path: Path) -> None:
    current_header = build_test_header(
        tmp_path,
        association_mode=AssociationMode.REGENIE2_BINARY,
        output_format=types.OutputFormat.ARROW,
    )
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_BINARY,
        output_format=types.OutputFormat.ARROW,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    write_native_chunks(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_BINARY,
        extra_code_value=types.BinaryExtraCode.FIRTH.value,
    )

    parquet_path = finalize_test_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_BINARY,
        output_format=types.OutputFormat.ARROW,
    )

    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.columns == EXPECTED_FINAL_COLUMNS
    assert parquet_frame.get_column("EXTRA").to_list() == [None, None, None, None]
    assert parquet_frame.get_column("CORRECTION_METHOD").to_list() == [
        "firth_approximate",
        "firth_approximate",
        "firth_approximate",
        "firth_approximate",
    ]
    assert parquet_frame.get_column("CORRECTION_STATUS").to_list() == ["success", "success", "success", "success"]
    parquet_schema = pq.ParquetFile(parquet_path).schema_arrow
    assert parquet_schema.names == EXPECTED_FINAL_COLUMNS
    assert parquet_schema.field("INFO").nullable
    assert parquet_schema.field("EXTRA").nullable
    assert parquet_schema.field("CORRECTION_METHOD").nullable
    assert parquet_schema.field("CORRECTION_STATUS").nullable
    parquet_metadata = pq.ParquetFile(parquet_path).metadata.metadata
    assert parquet_metadata is not None
    assert parquet_metadata[b"g.output.schema_version"] == b"2"
    assert parquet_metadata[b"g.output.association_mode"] == b"regenie2_binary"
    assert parquet_metadata[b"g.output.chunk_file_count"] == b"1"
    assert parquet_metadata[b"g.output.row_count"] == b"4"
    assert parquet_metadata[b"g.output.writer"] == b"rust"
    manifest = json.loads(
        output.get_run_manifest_path(prepared_output_run.output_run_paths).read_text(encoding="utf-8")
    )
    assert manifest["finalized"] is True
    assert manifest["final_row_count"] == 4


def test_finalize_parquet_dataset_parts_to_single_parquet(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    write_native_chunks(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.PARQUET,
    )

    parquet_path = finalize_test_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.PARQUET,
    )

    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.height == 4
    assert parquet_frame.columns == EXPECTED_FINAL_COLUMNS


def test_output_writer_finish_interrupted_flushes_commits_without_final_parquet(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    writer_session = create_test_output_writer_session(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=True,
        output_format=types.OutputFormat.PARQUET,
        chunks_per_arrow_file=16,
        arrow_compression=types.ArrowCompression.ZSTD,
        parquet_compression=types.ParquetCompression.NONE,
    )
    callback = NativeChunkWritingCallback(writer_session)
    try:
        engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
        engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), callback)
        writer_session.finish_interrupted("SIGTERM")
    except Exception:
        writer_session.abort()
        raise

    manifest = json.loads(
        output.get_run_manifest_path(prepared_output_run.output_run_paths).read_text(encoding="utf-8")
    )
    assert [chunk["chunk_identifier"] for chunk in manifest["committed_chunks"]] == [0, 2]
    assert manifest["finalized"] is False
    assert manifest["interrupted"] is True
    assert manifest["interrupted_signal"] == "SIGTERM"
    assert "final_parquet" not in manifest
    assert "final_row_count" not in manifest
    assert "final_chunk_file_count" not in manifest
    assert not (prepared_output_run.output_run_paths.run_directory / "final.parquet").exists()


def test_finalize_chunks_to_parquet_writes_empty_schema_when_no_chunks_exist(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = prepare_test_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)

    parquet_path = finalize_test_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        output_format=types.OutputFormat.PARQUET,
    )

    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.height == 0
    assert parquet_frame.columns == EXPECTED_FINAL_COLUMNS
