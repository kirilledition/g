from __future__ import annotations

import dataclasses
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import omegaconf
import pytest

from tooling.data import all_of_us
from tooling.data import all_of_us_inputs as cohort_inputs


def make_arguments(directory: Path) -> all_of_us.PreparationArguments:
    directory.mkdir(parents=True, exist_ok=True)
    mapping_path = directory / "mapping.tsv"
    mapping_path.write_text(
        "source_FID\tsource_IID\tFID\tIID\n0\t003\tfamily\tperson_003\n0\t001\tfamily\tperson_001\n0\t002\tfamily\tperson_002\n",
        encoding="utf-8",
    )
    phenotype_path = directory / "phenotypes.tsv"
    phenotype_path.write_text(
        "FID\tIID\ttrait\nfamily\tperson_001\t2\nfamily\tperson_002\tNA\nfamily\tperson_003\t1\n",
        encoding="utf-8",
    )
    return all_of_us.PreparationArguments(
        mode=all_of_us.PreparationMode.PLAN,
        source_format=all_of_us.SourceFormat.TABLES,
        input_prefix=None,
        input_bgen=None,
        input_sample=None,
        pvar_zstd=False,
        bgen_reference=None,
        confirm_diploid=False,
        identity_map=mapping_path,
        phenotype_file=phenotype_path,
        phenotype_columns=("trait",),
        phenotype_trait=all_of_us.PhenotypeTrait.QUANTITATIVE,
        covariate_file=None,
        covariate_columns=(),
        keep_file=None,
        remove_file=None,
        release_exclusion_files=(),
        variant_extract_file=None,
        minimum_allele_count=None,
        minimum_variant_call_rate=None,
        release_label="synthetic-release",
        genome_build="GRCh38",
        output_directory=directory / "prepared",
        plink_executable="plink2",
        threads=1,
        memory_megabytes=640,
        hash_genotype_inputs=False,
    )


def make_pgen_arguments(directory: Path) -> all_of_us.PreparationArguments:
    arguments = make_arguments(directory)
    prefix = directory / "source.v1"
    Path(f"{prefix}.pgen").write_bytes(b"test-genotypes")
    Path(f"{prefix}.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n22\t100\tvariant\tA\tC\n", encoding="utf-8")
    Path(f"{prefix}.psam").write_text("#IID\tSEX\n001\t1\n002\t2\n003\t0\n", encoding="utf-8")
    return dataclasses.replace(
        arguments,
        source_format=all_of_us.SourceFormat.PGEN,
        input_prefix=prefix,
        confirm_diploid=True,
    )


def test_plan_is_read_only_and_preserves_string_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arguments = make_arguments(tmp_path)
    before = set(tmp_path.iterdir())
    all_of_us.run_tool(arguments)
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_counts"]["selected"] == 3
    assert payload["phenotype_missing_counts"] == {"trait": 1}
    assert payload["analysis_complete_case_counts"] == {"trait": 2}
    assert payload["commands"] == []
    assert set(tmp_path.iterdir()) == before
    assert not arguments.output_directory.exists()
    prepared = all_of_us.prepare_cohort(arguments)
    assert prepared.mappings[0].source.individual_id == "003"
    assert prepared.phenotypes.values == (("1",), ("2",), ("NA",))
    assert all(fingerprint.sha256 for fingerprint in prepared.plan.input_files)


def test_source_order_wins_over_map_and_table_order(tmp_path: Path) -> None:
    prepared = all_of_us.prepare_cohort(make_pgen_arguments(tmp_path))
    assert tuple(mapping.source.individual_id for mapping in prepared.mappings) == ("001", "002", "003")
    assert prepared.phenotypes.values == (("2",), ("NA",), ("1",))


def test_keep_remove_and_release_flags_apply_explicitly(tmp_path: Path) -> None:
    arguments = make_arguments(tmp_path)
    keep_path = tmp_path / "keep.tsv"
    keep_path.write_text("FID\tIID\nfamily\tperson_003\nfamily\tperson_001\n", encoding="utf-8")
    release_path = tmp_path / "release.tsv"
    release_path.write_text("FID\tIID\nfamily\tperson_003\noutside\tcohort\n", encoding="utf-8")
    remove_path = tmp_path / "remove.tsv"
    remove_path.write_text("FID\tIID\nfamily\tperson_002\n", encoding="utf-8")
    prepared = all_of_us.prepare_cohort(
        dataclasses.replace(
            arguments, keep_file=keep_path, remove_file=remove_path, release_exclusion_files=(release_path,)
        ),
    )
    assert tuple(mapping.target.individual_id for mapping in prepared.mappings) == ("person_001",)
    assert prepared.plan.sample_counts["release_exclusions_matching_source"] == 1
    assert prepared.plan.sample_counts["remove_matching_source"] == 1
    assert prepared.plan.sample_counts["exclusions_absent_from_source"] == 1
    assert prepared.plan.sample_selection_files["release_exclusions"] == (str(release_path),)


@pytest.mark.parametrize(
    ("mapping_body", "message"),
    [
        ("0\ta\tf\tone\n0\ta\tf\ttwo\n", "duplicate"),
        ("0\ta\tf\tone\n0\tb\tf\tone\n", "duplicate"),
        ("0\ta\ta_b\tc\n0\tb\ta\tb_c\n", "ambiguous"),
        ("0\ta\t\tone\n", "nonempty"),
        ("0\ta\tf\tbad id\n", "whitespace"),
        ('0\ta\tf\tbad"id\n', "double quotes"),
    ],
)
def test_identity_mapping_rejects_ambiguity(tmp_path: Path, mapping_body: str, message: str) -> None:
    path = tmp_path / "mapping.tsv"
    path.write_text("source_FID\tsource_IID\tFID\tIID\n" + mapping_body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        cohort_inputs.read_identity_mapping(path)


def test_unknown_keep_and_incomplete_mapping_are_rejected(tmp_path: Path) -> None:
    arguments = make_pgen_arguments(tmp_path)
    keep_path = tmp_path / "keep.tsv"
    keep_path.write_text("FID\tIID\n0\tunknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="known"):
        all_of_us.prepare_cohort(dataclasses.replace(arguments, keep_file=keep_path))
    Path(f"{arguments.input_prefix}.psam").write_text("#IID\n001\n002\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cover every"):
        all_of_us.prepare_cohort(arguments)


def test_tables_preserve_missing_values_and_do_not_impute(tmp_path: Path) -> None:
    arguments = make_arguments(tmp_path)
    arguments.phenotype_file.write_text(
        "FID\tIID\ttrait\nfamily\tperson_001\t\nfamily\tperson_002\t-9\nfamily\tperson_003\tnan\n",
        encoding="utf-8",
    )
    covariate_path = tmp_path / "covariates.tsv"
    covariate_path.write_text(
        "FID\tIID\tage\nfamily\tperson_003\t40\nfamily\tperson_001\tNA\nfamily\tperson_002\t52\n",
        encoding="utf-8",
    )
    arguments = dataclasses.replace(
        arguments,
        mode=all_of_us.PreparationMode.TABLES,
        covariate_file=covariate_path,
        covariate_columns=("age",),
        plink_executable="unavailable-plink-is-not-needed",
    )
    prepared = all_of_us.prepare_cohort(arguments)
    manifest = all_of_us.execute_preparation(arguments, prepared)
    assert manifest.status == all_of_us.PreparationStatus.TABLES_PREPARED
    assert prepared.plan.analysis_complete_case_counts == {"trait": 0}
    assert (arguments.output_directory / "phenotypes.tsv").read_text() == (
        "FID\tIID\ttrait\nfamily\tperson_003\tNA\nfamily\tperson_001\tNA\nfamily\tperson_002\tNA\n"
    )
    assert cohort_inputs.read_cohort_list(arguments.output_directory / "cohort.keep.tsv") == frozenset(
        mapping.target for mapping in prepared.mappings
    )
    assert all(Path(fingerprint.path).name != "preparation.json" for fingerprint in manifest.output_files)
    assert all(fingerprint.sha256 for fingerprint in manifest.output_files)


@pytest.mark.parametrize("value", ["inf", "1e100", "1_000", "0", "3", "1 "])
def test_binary_table_requires_finite_regenie_coding(tmp_path: Path, value: str) -> None:
    arguments = make_arguments(tmp_path)
    text = arguments.phenotype_file.read_text().replace("person_003\t1", f"person_003\t{value}")
    arguments.phenotype_file.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        all_of_us.prepare_cohort(dataclasses.replace(arguments, phenotype_trait=all_of_us.PhenotypeTrait.BINARY))


@pytest.mark.parametrize("table_error", ["duplicate", "blank", "short", "absent", "duplicate_header"])
def test_table_structure_and_identity_fail_closed(tmp_path: Path, table_error: str) -> None:
    arguments = make_arguments(tmp_path)
    text = arguments.phenotype_file.read_text()
    if table_error == "duplicate":
        text += "family\tperson_001\t2\n"
    elif table_error == "blank":
        text = text.replace("family\tperson_001", "\tperson_001")
    elif table_error == "short":
        text = text.replace("person_001\t2", "person_001")
    elif table_error == "absent":
        text = text.replace("family\tperson_001\t2\n", "")
    else:
        text = text.replace("FID\tIID\ttrait", "FID\tIID\tIID")
    arguments.phenotype_file.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        all_of_us.prepare_cohort(arguments)


def test_commands_are_explicit_argument_vectors_and_preserve_missingness(tmp_path: Path) -> None:
    arguments = make_pgen_arguments(tmp_path / "paths with spaces;literal$(text)")
    extract_path = tmp_path / "extract.txt"
    extract_path.write_text("variant\n", encoding="utf-8")
    prepared = all_of_us.prepare_cohort(
        dataclasses.replace(
            arguments, minimum_allele_count=5.0, minimum_variant_call_rate=0.98, variant_extract_file=extract_path
        ),
    )
    select_command, export_command = prepared.plan.commands
    assert select_command.arguments[2] == str(arguments.input_prefix)
    assert "--nonfounders" in select_command.arguments
    assert "--mac" in select_command.arguments
    assert "dosage" in select_command.arguments
    assert "--autosome" in select_command.arguments
    assert export_command.arguments[export_command.arguments.index("--export") + 1 :][:4] == (
        "bgen-1.2",
        "ref-first",
        "bits=8",
        "id-paste=fid,iid",
    )
    assert not any("fill-missing" in argument for command in prepared.plan.commands for argument in command.arguments)
    assert prepared.plan.variant_extract_file == str(extract_path)


def test_bgen_input_requires_explicit_reference_and_diploid_contract(tmp_path: Path) -> None:
    arguments = dataclasses.replace(make_arguments(tmp_path), source_format=all_of_us.SourceFormat.BGEN)
    with pytest.raises(ValueError, match="diploid"):
        all_of_us.prepare_cohort(arguments)
    with pytest.raises(ValueError, match="allele-order"):
        all_of_us.prepare_cohort(dataclasses.replace(arguments, confirm_diploid=True))


def test_bgen_oxford_columns_must_be_positioned_for_plink(tmp_path: Path) -> None:
    arguments = make_arguments(tmp_path)
    bgen_path = tmp_path / "source.bgen"
    bgen_path.write_bytes(b"nonempty-stub")
    sample_path = tmp_path / "source.sample"
    sample_path.write_text("sex ID_1 ID_2 missing\nD 0 0 0\n2 0 001 0\n2 0 002 0\n2 0 003 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="first two"):
        all_of_us.prepare_cohort(
            dataclasses.replace(
                arguments,
                source_format=all_of_us.SourceFormat.BGEN,
                input_bgen=bgen_path,
                input_sample=sample_path,
                bgen_reference=all_of_us.ReferenceOrder.FIRST,
                confirm_diploid=True,
            ),
        )


def test_output_overwrite_and_changed_inputs_are_rejected(tmp_path: Path) -> None:
    arguments = make_arguments(tmp_path)
    prepared = all_of_us.prepare_cohort(arguments)
    arguments.phenotype_file.write_text(
        arguments.phenotype_file.read_text() + "other\tparticipant\t1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed"):
        all_of_us.execute_preparation(arguments, prepared)
    assert not arguments.output_directory.exists()
    arguments.output_directory.mkdir()
    with pytest.raises(FileExistsError, match="Refusing"):
        all_of_us.prepare_cohort(arguments)


def test_partial_plink_failure_is_retained_and_never_retried_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = dataclasses.replace(make_pgen_arguments(tmp_path), mode=all_of_us.PreparationMode.EXECUTE)
    prepared = all_of_us.prepare_cohort(arguments)
    monkeypatch.setattr(shutil, "which", lambda _: "/synthetic/plink2")

    def failed_run(*_arguments: object, **_keywords: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(2, ["synthetic-plink2"])

    monkeypatch.setattr(subprocess, "run", failed_run)
    with pytest.raises(subprocess.CalledProcessError):
        all_of_us.execute_preparation(arguments, prepared)
    payload = json.loads((arguments.output_directory / "preparation.json").read_text())
    assert payload["status"] == "failed"
    assert (arguments.output_directory / "phenotypes.tsv").exists()
    with pytest.raises(FileExistsError):
        all_of_us.prepare_cohort(arguments)


def test_export_rejects_empty_compressed_payload_after_successful_plink(tmp_path: Path) -> None:
    arguments = make_arguments(tmp_path)
    prepared = all_of_us.prepare_cohort(arguments)
    arguments.output_directory.mkdir()
    prefix = arguments.output_directory / "genotypes"
    Path(f"{prefix}.pgen").write_bytes(b"placeholder-only")
    Path(f"{prefix}.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n22\t100\tvariant\tA\tC\n", encoding="utf-8")
    Path(f"{prefix}.psam").write_text(
        "#FID\tIID\n"
        + "".join(f"{mapping.target.family_id}\t{mapping.target.individual_id}\n" for mapping in prepared.mappings),
        encoding="utf-8",
    )
    Path(f"{prefix}.sample").write_text(
        "ID_1 ID_2 missing\n0 0 0\n"
        + "".join(f"{mapping.target.family_id} {mapping.target.individual_id} 0\n" for mapping in prepared.mappings),
        encoding="utf-8",
    )
    header = struct.pack("<IIII4sI", 20, 20, 1, 3, b"bgen", 9)
    identifying_data = (
        b"\x00\x00\x00\x00\x02\x0022"
        + struct.pack("<IH", 100, 2)
        + struct.pack("<I", 1)
        + b"A"
        + struct.pack("<I", 1)
        + b"C"
    )
    # Matches the installed exporter's observed tiny-cohort defect: C=4, D=19, no zlib bytes.
    Path(f"{prefix}.bgen").write_bytes(header + identifying_data + struct.pack("<II", 4, 19))
    with pytest.raises(ValueError, match=r"compressed|payload|block"):
        all_of_us.validate_export(prepared, 1)


def test_hydra_config_has_no_plan_artifact_side_effects(tmp_path: Path) -> None:
    arguments = make_arguments(tmp_path / "inputs")
    config_path = Path(__file__).resolve().parents[1] / "tooling" / "configs" / "data_all_of_us.yaml"
    config = omegaconf.OmegaConf.load(config_path)
    assert config.hydra.run.dir == "."
    assert config.hydra.output_subdir is None
    working_directory = tmp_path / "working"
    working_directory.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tooling.cli.data",
            "--config-name",
            "data_all_of_us",
            f"tool.identity_map={arguments.identity_map}",
            f"tool.phenotype_file={arguments.phenotype_file}",
            "tool.phenotype_columns=[trait]",
            "tool.release_label=synthetic-release",
            f"tool.output_directory={arguments.output_directory}",
        ],
        cwd=working_directory,
        env={**os.environ, "PYTHONPATH": str(config_path.parents[2])},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["sample_counts"]["selected"] == 3
    assert list(working_directory.iterdir()) == []


@pytest.mark.parametrize("help_requested", [False, True])
def test_preparation_cli_does_not_import_development_dependencies(tmp_path: Path, *, help_requested: bool) -> None:
    arguments = make_arguments(tmp_path / "inputs")
    repository_root = Path(__file__).resolve().parents[1]
    script = (
        "import importlib.abc, runpy, sys\n"
        "class DenyDevelopmentModules(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in {'polars', 'pooch', 'psutil', 'numpy', 'jax'}:\n"
        "            raise ImportError('Development dependency denied: ' + fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, DenyDevelopmentModules())\n"
        "sys.argv = ['tooling.cli.data', *sys.argv[1:]]\n"
        "runpy.run_module('tooling.cli.data', run_name='__main__')\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--config-name",
            "data_all_of_us",
            f"tool.identity_map={arguments.identity_map}",
            f"tool.phenotype_file={arguments.phenotype_file}",
            "tool.phenotype_columns=[trait]",
            "tool.release_label=synthetic-release",
            f"tool.output_directory={arguments.output_directory}",
            *(["--help"] if help_requested else []),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not arguments.output_directory.exists()


@pytest.mark.phase0_data
@pytest.mark.parametrize(
    "source_format", [all_of_us.SourceFormat.PGEN, all_of_us.SourceFormat.BED, all_of_us.SourceFormat.BGEN]
)
def test_real_plink_roundtrip_preserves_missing_calls(tmp_path: Path, source_format: all_of_us.SourceFormat) -> None:
    executable = os.environ.get("GWAS_ENGINE_PLINK2_INTEGRATION")
    if not executable:
        pytest.skip("Set GWAS_ENGINE_PLINK2_INTEGRATION to a pinned PLINK2 executable on a compute node.")
    arguments = make_arguments(tmp_path)
    additional_ids = tuple(f"{number:03d}" for number in range(4, 33))
    with arguments.identity_map.open("a", encoding="utf-8") as mapping_file:
        mapping_file.writelines(f"0\t{identity}\tfamily\tperson_{identity}\n" for identity in additional_ids)
    with arguments.phenotype_file.open("a", encoding="utf-8") as phenotype_file:
        phenotype_file.writelines(f"family\tperson_{identity}\t1\n" for identity in additional_ids)
    extra_genotypes = "\t0/1" * len(additional_ids)
    vcf_path = tmp_path / "synthetic.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=22>\n##contig=<ID=X>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t001\t002\t003\t"
        + "\t".join(additional_ids)
        + "\n"
        + f"22\t100\tcomplete\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1{extra_genotypes}\n"
        + f"22\t200\tmissing\tG\tT\t.\tPASS\t.\tGT\t./.\t1/1\t0/1{extra_genotypes}\n"
        + f"22\t300\tmulti\tA\tC,G\t.\tPASS\t.\tGT\t0/0\t0/1\t0/2{extra_genotypes}\n"
        + f"X\t100\tsex_chr\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1{extra_genotypes}\n",
        encoding="utf-8",
    )
    prefix = tmp_path / "source.v1"
    sex_path = tmp_path / "source-sex.psam"
    sex_path.write_text(
        "#IID\tSEX\n001\t2\n002\t2\n003\t2\n" + "".join(f"{identity}\t2\n" for identity in additional_ids),
        encoding="utf-8",
    )
    import_result = subprocess.run(
        [
            executable,
            "--vcf",
            str(vcf_path),
            "--split-par",
            "b38",
            "--psam",
            str(sex_path),
            "--make-pgen",
            "--memory",
            "640",
            "--threads",
            "1",
            "--out",
            str(prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr
    if source_format != all_of_us.SourceFormat.PGEN:
        conversion_arguments = (
            ["--make-bed"]
            if source_format == all_of_us.SourceFormat.BED
            else ["--export", "bgen-1.2", "bits=8", "ref-first"]
        )
        subprocess.run(
            [
                executable,
                "--pfile",
                str(prefix),
                *conversion_arguments,
                "--autosome",
                "--max-alleles",
                "2",
                "--memory",
                "640",
                "--threads",
                "1",
                "--out",
                str(prefix),
            ],
            check=True,
            capture_output=True,
        )
    arguments = dataclasses.replace(
        arguments,
        mode=all_of_us.PreparationMode.EXECUTE,
        source_format=source_format,
        input_prefix=None if source_format == all_of_us.SourceFormat.BGEN else prefix,
        input_bgen=Path(f"{prefix}.bgen") if source_format == all_of_us.SourceFormat.BGEN else None,
        input_sample=Path(f"{prefix}.sample") if source_format == all_of_us.SourceFormat.BGEN else None,
        bgen_reference=all_of_us.ReferenceOrder.FIRST if source_format == all_of_us.SourceFormat.BGEN else None,
        confirm_diploid=True,
        plink_executable=executable,
    )
    prepared = all_of_us.prepare_cohort(arguments)
    manifest = all_of_us.execute_preparation(arguments, prepared)
    assert manifest.status == all_of_us.PreparationStatus.COMPLETE
    assert manifest.variant_count == 2
    output_prefix = tmp_path / "roundtrip"
    roundtrip_result = subprocess.run(
        [
            executable,
            "--bgen",
            prepared.plan.outputs["bgen"],
            "ref-first",
            "--sample",
            prepared.plan.outputs["sample"],
            "--export",
            "A",
            "include-alt",
            "--memory",
            "640",
            "--threads",
            "1",
            "--out",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert roundtrip_result.returncode == 0, roundtrip_result.stderr
    raw_lines = Path(f"{output_prefix}.raw").read_text().splitlines()
    raw_table = cohort_inputs.TextTable(
        columns=tuple(raw_lines[0].split()),
        rows=tuple(tuple(line.split()) for line in raw_lines[1:]),
    )
    complete_column = next(index for index, column in enumerate(raw_table.columns) if column.startswith("complete_"))
    missing_column = next(index for index, column in enumerate(raw_table.columns) if column.startswith("missing_"))
    assert tuple(row[complete_column] for row in raw_table.rows) == ("2", "1", "0", *("1" for _ in additional_ids))
    assert tuple(row[missing_column] for row in raw_table.rows) == ("NA", "0", "1", *("1" for _ in additional_ids))
    assert tuple(row[1] for row in raw_table.rows) == (
        "person_001",
        "person_002",
        "person_003",
        *(f"person_{identity}" for identity in additional_ids),
    )
    assert all(fingerprint.sha256 for fingerprint in manifest.output_files)
