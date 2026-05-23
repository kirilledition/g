"""Tests for Rust-backed output persistence."""

from __future__ import annotations

import copy
import json
import typing
from pathlib import Path

import numpy as np
import polars as pl
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
]
EXPECTED_CHUNK_COLUMNS = [
    *EXPECTED_FINAL_COLUMNS,
]
TEST_DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "bgen"
HAPLOTYPES_BGEN_PATH = TEST_DATA_DIRECTORY / "haplotypes.bgen"


class NativeChunkWritingCallback:
    """Callback that writes deterministic association values for native chunks."""

    def __init__(self, writer_session: typing.Any, extra_code_value: int | None = None) -> None:
        self.writer_session = writer_session
        self.extra_code_value = extra_code_value
        self.free_buffers: list[np.ndarray] = []

    def acquire_dosage_buffer(self, sample_count: int, variant_count: int) -> np.ndarray:
        if self.free_buffers:
            dosage_buffer = self.free_buffers.pop()
            if dosage_buffer.shape == (sample_count, variant_count):
                return dosage_buffer
        return np.empty((sample_count, variant_count), dtype=np.float32, order="C")

    def compute_preprocessed_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: np.ndarray,
        chunk_stats: _core.ChunkStats,
    ) -> None:
        variant_count = metadata.variant_stop_index - metadata.variant_start_index
        extra_code = (
            np.full(variant_count, self.extra_code_value, dtype=np.int32) if self.extra_code_value is not None else None
        )
        self.writer_session.write_regenie2_native_chunk(
            metadata=metadata,
            chunk_stats=chunk_stats,
            beta=np.full(variant_count, 0.1, dtype=np.float32),
            standard_error=np.full(variant_count, 0.01, dtype=np.float32),
            chi_squared=np.full(variant_count, 10.0, dtype=np.float32),
            log10_p_value=np.full(variant_count, 5.0, dtype=np.float32),
            extra_code=extra_code,
        )
        self.free_buffers.append(genotype_matrix)


def write_native_chunks(
    output_run_paths: output.OutputRunPaths,
    association_mode: AssociationMode,
    *,
    extra_code_value: int | None = None,
) -> None:
    writer_session = output.create_output_writer_session(
        output_run_paths,
        association_mode,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=False,
    )
    callback = NativeChunkWritingCallback(writer_session, extra_code_value)
    try:
        engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
        engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), callback)
        writer_session.finish()
    except Exception:
        writer_session.abort()
        raise


def build_test_header(
    tmp_path: Path,
    *,
    association_mode: AssociationMode = AssociationMode.REGENIE2_LINEAR,
    binary_kernel_config: typing.Any | None = None,
    jax_enable_x64: bool = True,
) -> dict[str, typing.Any]:
    bgen_path = tmp_path / "study.bgen"
    sample_path = tmp_path / "study.sample"
    phenotype_path = tmp_path / "phenotypes.tsv"
    covariate_path = tmp_path / "covariates.tsv"
    prediction_list_path = tmp_path / "predictions.list"
    for input_path in (bgen_path, sample_path, phenotype_path, covariate_path, prediction_list_path):
        input_path.write_text(input_path.name, encoding="utf-8")
    return output.build_current_run_manifest_header(
        association_mode=association_mode,
        bgen_path=bgen_path,
        sample_path=sample_path,
        phenotype_path=phenotype_path,
        phenotype_name="trait",
        covariate_path=covariate_path,
        covariate_names=("intercept", "age", "sex"),
        prediction_list_path=prediction_list_path,
        sample_count=4,
        variant_count=10,
        chunk_size=2,
        variant_limit=None,
        binary_correction_plan=types.BinaryCorrectionPlan(),
        trusted_no_missing_diploid=False,
        sample_key_mode=types.SampleKeyMode.IID,
        binary_kernel_config=binary_kernel_config,
        jax_enable_x64=jax_enable_x64,
    )


def test_current_run_manifest_records_configured_x64_policy(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path, jax_enable_x64=False)

    assert current_header["jax_policy"]["enable_x64"] is False
    assert current_header["execution_plan"]["jax_policy"]["enable_x64"] is False


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
    output_run_paths = output.resolve_output_run_paths(tmp_path / "results/output", AssociationMode.REGENIE2_LINEAR)
    assert output_run_paths.run_directory == tmp_path / "results/output.regenie2_linear.run"
    assert output_run_paths.chunks_directory == tmp_path / "results/output.regenie2_linear.run/chunks"


def test_scan_committed_chunk_identifiers_discovers_single_chunk_files(tmp_path: Path) -> None:
    (tmp_path / "chunk_000000000.arrow").write_bytes(b"")
    (tmp_path / "chunk_000000512.arrow").write_bytes(b"")
    assert output.scan_committed_chunk_identifiers(tmp_path) == frozenset({0, 512})


def test_prepare_output_run_rejects_non_empty_directory_without_resume(tmp_path: Path) -> None:
    run_directory = tmp_path / "output.regenie2_linear.run"
    run_directory.mkdir(parents=True)
    (run_directory / "stale_file.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists and is not empty"):
        output.prepare_output_run(
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
    assert b"g.output.chunk_commits" in (chunk_schema.metadata or {})
    assert frame.get_column("TEST").to_list() == ["ADD", "ADD", "ADD", "ADD"]
    assert frame.get_column("INFO").to_list() == [1.0, 1.0, 1.0, 1.0]
    assert frame.get_column("EXTRA").to_list() == [None, None, None, None]


def test_native_writer_records_output_stage_timings_when_requested(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    writer_session = output.create_output_writer_session(
        output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=False,
        collect_stage_timings=True,
    )
    callback = NativeChunkWritingCallback(writer_session)
    try:
        engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
        engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), callback)
        writer_session.finish()
    except Exception:
        writer_session.abort()
        raise

    timing_payload = json.loads((tmp_path / "output_stage_timings.json").read_text(encoding="utf-8"))
    assert timing_payload["stage_counts"]["rust_output_result_buffer_copy"] == 2
    assert timing_payload["stage_counts"]["rust_output_writer_arrow_file_write"] == 1
    assert timing_payload["output_metrics"]["writer_chunk_count"] == 2
    assert timing_payload["output_metrics"]["writer_row_count"] == 4


def test_native_binary_writer_maps_successful_correction_extra_code_to_null(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(
        output_run_paths, AssociationMode.REGENIE2_BINARY, extra_code_value=types.BinaryExtraCode.FIRTH.value
    )

    frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(tmp_path)[0])
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    assert frame.get_column("EXTRA").to_list() == [None, None, None, None]


def test_native_binary_writer_maps_test_fail_extra_code_to_label(tmp_path: Path) -> None:
    output_run_paths = output.OutputRunPaths(run_directory=tmp_path, chunks_directory=tmp_path)
    write_native_chunks(
        output_run_paths, AssociationMode.REGENIE2_BINARY, extra_code_value=types.BinaryExtraCode.TEST_FAIL.value
    )

    frame = pl.read_ipc(output.iter_sorted_chunk_file_paths(tmp_path)[0])
    assert frame.columns == EXPECTED_CHUNK_COLUMNS
    assert frame.get_column("EXTRA").to_list() == ["TEST_FAIL", "TEST_FAIL", "TEST_FAIL", "TEST_FAIL"]


def test_initialize_output_run_compatible_resume_preserves_committed_chunks(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    write_native_chunks(prepared_output_run.output_run_paths, AssociationMode.REGENIE2_LINEAR)

    resumed_output_run = output.prepare_output_run(
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
    prepared_output_run = output.prepare_output_run(
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
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    write_native_chunks(prepared_output_run.output_run_paths, AssociationMode.REGENIE2_LINEAR)

    resumed_output_run = output.prepare_output_run(
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


def test_fast_resume_trusts_only_manifest_committed_chunks(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    (prepared_output_run.output_run_paths.chunks_directory / "chunk_000000000.arrow").write_bytes(b"staged")

    resumed_output_run = output.prepare_output_run(
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
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(
        prepared_output_run.output_run_paths,
        {**current_header, "association_mode": AssociationMode.REGENIE2_BINARY.value, "committed_chunks": []},
    )
    resumed_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match="association_mode"):
        initialize_test_output_run(resumed_output_run, current_header, resume=True)


def build_test_binary_kernel_config() -> regenie2_binary_config.BinaryKernelConfig:
    """Build a non-default binary kernel config for manifest tests."""
    return regenie2_binary_config.BinaryKernelConfig(
        maximum_null_iterations=13,
        null_logistic_coefficient_tolerance=1.0e-5,
        minimum_probability=1.0e-7,
        minimum_variance=1.0e-9,
        relative_variance_tolerance=2.0e-6,
        firth_batch_size=7,
        firth_candidate_capacity=11,
        firth_maximum_iterations=17,
        firth_gradient_tolerance=2.0e-5,
        firth_coefficient_tolerance=3.0e-5,
        firth_likelihood_tolerance=4.0e-5,
        firth_maximum_step_size=6.0,
        firth_pseudo_maximum_iterations=19,
        firth_pseudo_inner_maximum_iterations=23,
        firth_newton_raphson_zero_start_iterations=29,
        firth_line_search_maximum_attempts=31,
        firth_step_halving_maximum_attempts=37,
        firth_initial_response_scale=4.5,
        firth_sparse_carrier_dosage_threshold=1.0e-3,
        firth_step_halving_scale=0.25,
        null_firth_maximum_iterations=41,
        null_firth_gradient_tolerance=5.0e-5,
        null_firth_maximum_step_size=7.0,
        null_firth_fallback_iteration_multiplier=43,
        null_firth_fallback_step_divisor=11.0,
        null_firth_line_search_maximum_attempts=47,
        null_firth_step_halving_scale=0.125,
        use_block_firth_math=True,
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
        ("output_schema_version", 2),
        ("bgen_decode_tile_variant_count", 128),
        ("jax_policy", {"device": "gpu", "enable_x64": True, "matmul_precision": "highest"}),
        ("multi_phenotype_sample_mode", "complete_case_intersection"),
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
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / f"output-{field_name}",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(
        prepared_output_run.output_run_paths,
        {**manifest_header, "committed_chunks": []},
    )
    resumed_output_run = output.prepare_output_run(
        output_root=tmp_path / f"output-{field_name}",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=True,
    )

    with pytest.raises(ValueError, match=field_name):
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
        ("output_schema_version", 2),
        ("trusted_no_missing_diploid", True),
        ("bgen_decode_tile_variant_count", 128),
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
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / f"output-execution-plan-{nested_field_name}",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = output.prepare_output_run(
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
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output-execution-plan-hash",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = output.prepare_output_run(
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
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = output.prepare_output_run(
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
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    resumed_output_run = output.prepare_output_run(
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
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    output.write_run_manifest(prepared_output_run.output_run_paths, {**manifest_header, "committed_chunks": []})
    manifest_path = output.get_run_manifest_path(prepared_output_run.output_run_paths)
    original_manifest_bytes = manifest_path.read_bytes()
    resumed_output_run = output.prepare_output_run(
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
        output.prepare_output_run(
            output_root=tmp_path / "output",
            association_mode=AssociationMode.REGENIE2_LINEAR,
            resume=True,
        )


def test_prepare_output_run_strict_resume_requires_manifest(tmp_path: Path) -> None:
    run_directory = tmp_path / "output.regenie2_linear.run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)

    with pytest.raises(ValueError, match=r"Resume requires run_manifest\.json"):
        output.prepare_output_run(
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


def test_finalize_chunks_to_parquet_projects_technical_columns_away(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path, association_mode=AssociationMode.REGENIE2_BINARY)
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_BINARY,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    write_native_chunks(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_BINARY,
        extra_code_value=types.BinaryExtraCode.FIRTH.value,
    )

    parquet_path = output.finalize_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_BINARY,
    )

    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.columns == EXPECTED_FINAL_COLUMNS
    assert parquet_frame.get_column("EXTRA").to_list() == [None, None, None, None]
    parquet_schema = pq.ParquetFile(parquet_path).schema_arrow
    assert parquet_schema.names == EXPECTED_FINAL_COLUMNS
    assert parquet_schema.field("INFO").nullable
    assert parquet_schema.field("EXTRA").nullable
    parquet_metadata = pq.ParquetFile(parquet_path).metadata.metadata
    assert parquet_metadata is not None
    assert parquet_metadata[b"g.output.schema_version"] == b"1"
    assert parquet_metadata[b"g.output.association_mode"] == b"regenie2_binary"
    assert parquet_metadata[b"g.output.chunk_file_count"] == b"1"
    assert parquet_metadata[b"g.output.row_count"] == b"4"
    assert parquet_metadata[b"g.output.writer"] == b"rust"
    manifest = json.loads(
        output.get_run_manifest_path(prepared_output_run.output_run_paths).read_text(encoding="utf-8")
    )
    assert manifest["finalized"] is True
    assert manifest["final_row_count"] == 4


def test_output_writer_finish_interrupted_flushes_commits_without_final_parquet(tmp_path: Path) -> None:
    current_header = build_test_header(tmp_path)
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)
    writer_session = output.create_output_writer_session(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=True,
    )
    callback = NativeChunkWritingCallback(writer_session)
    try:
        engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)
        engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), callback)
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
    prepared_output_run = output.prepare_output_run(
        output_root=tmp_path / "output",
        association_mode=AssociationMode.REGENIE2_LINEAR,
        resume=False,
    )
    initialize_test_output_run(prepared_output_run, current_header)

    parquet_path = output.finalize_chunks_to_parquet(
        prepared_output_run.output_run_paths,
        AssociationMode.REGENIE2_LINEAR,
    )

    parquet_frame = pl.read_parquet(parquet_path)
    assert parquet_frame.height == 0
    assert parquet_frame.columns == EXPECTED_FINAL_COLUMNS
