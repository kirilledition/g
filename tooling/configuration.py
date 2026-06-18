"""Hydra configuration support for development tooling."""

from __future__ import annotations

import dataclasses
import typing

import hydra
import hydra.core.config_store
import omegaconf

CONFIG_MODULE = "tooling.configs"
CONFIG_STORE_REGISTERED = False


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Dataset paths for repository-local benchmark tooling.

    Attributes:
        data_directory: Repository-relative or absolute data directory.
        bgen_file: BGEN path relative to the data directory.
        sample_file: Sample path relative to the data directory.
        phenotype_file: Phenotype path relative to the data directory.
        prediction_list: REGENIE step 1 prediction list relative to the data directory.
        phenotype_columns: Phenotype columns available to benchmark workloads.

    """

    data_directory: str = "data"
    bgen_file: str = "1kg_chr22_full.bgen"
    sample_file: str = "1kg_chr22_full.sample"
    phenotype_file: str = "pheno_bin.txt"
    prediction_list: str = "baselines/regenie_step1_pred.list"
    phenotype_columns: list[str] = dataclasses.field(default_factory=lambda: ["phenotype_binary"])


@dataclasses.dataclass(frozen=True)
class MachineConfig:
    """Machine profile for development tooling.

    Attributes:
        name: Stable profile name.
        device: Runtime device passed to g benchmark entrypoints.
        slurm_node: Optional SLURM node name for deferred smoke validation.
        cpus_per_task: Optional CPU allocation hint.
        memory: Optional memory allocation hint.

    """

    name: str = "local"
    device: str = "cpu"
    slurm_node: str | None = None
    cpus_per_task: int | None = None
    memory: str | None = None


@dataclasses.dataclass(frozen=True)
class WorkloadConfig:
    """Reusable workload defaults for benchmark tooling.

    Attributes:
        name: Stable workload name.
        chunk_size: Variant chunk size.
        variant_limit: Optional variant cap.
        repeat_count: BGEN reader repeat count.
        staging_depth: Native callback staging depth.
        native_callback_batch_size: Native-to-Python callback chunk batch size.
        output_writer_thread_count: Background output writer thread count.
        output_writer_queue_depth: Background output writer queue depth.

    """

    name: str = "bgen_reader"
    chunk_size: int = 8192
    variant_limit: int | None = 16384
    repeat_count: int = 5
    staging_depth: int = 1
    native_callback_batch_size: int = 1
    output_writer_thread_count: int = 8
    output_writer_queue_depth: int = 8


@dataclasses.dataclass(frozen=True)
class TelemetryConfig:
    """Report and timing-output defaults for development tooling.

    Attributes:
        output_parent: Repository-relative or absolute parent directory for outputs.
        json_summary_path: Optional explicit JSON summary path.
        markdown_summary_path: Optional explicit Markdown summary path.
        stage_timing_mode: Stage timing collection mode for benchmark workloads.

    """

    output_parent: str = "data/profiles"
    json_summary_path: str | None = None
    markdown_summary_path: str | None = None
    stage_timing_mode: str = "exact"


@dataclasses.dataclass(frozen=True)
class SweepConfig:
    """Sweep defaults for benchmark tooling.

    Attributes:
        chunk_sizes: BGEN reader chunk-size sweep values.
        path_modes: Native BGEN path modes.
        sample_selection_modes: Sample-selection benchmark modes.
        decode_tile_variant_counts: Optional BGEN decode tile sizes.
        rayon_thread_counts: Optional Rayon thread counts.
        trusted_no_missing_diploid_modes: Trusted decode path modes.
        storage_modes: Binary-hot genotype storage modes.
        fallback_density_scenarios: Binary-hot fallback-density scenarios.

    """

    chunk_sizes: list[int] = dataclasses.field(default_factory=lambda: [8192])
    path_modes: list[str] = dataclasses.field(default_factory=lambda: ["variant_major_buffered"])
    sample_selection_modes: list[str] = dataclasses.field(default_factory=lambda: ["full"])
    decode_tile_variant_counts: list[int | None] = dataclasses.field(default_factory=list)
    rayon_thread_counts: list[int | None] = dataclasses.field(default_factory=list)
    trusted_no_missing_diploid_modes: list[bool] = dataclasses.field(default_factory=lambda: [False])
    storage_modes: list[str] = dataclasses.field(default_factory=lambda: ["variant_major"])
    fallback_density_scenarios: list[str] = dataclasses.field(default_factory=lambda: ["default"])


@dataclasses.dataclass(frozen=True)
class ToolingConfig:
    """Top-level development tooling configuration.

    Attributes:
        dataset: Dataset path configuration.
        machine: Machine profile.
        workload: Workload defaults.
        telemetry: Report and timing-output defaults.
        sweep: Sweep defaults.

    """

    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    machine: MachineConfig = dataclasses.field(default_factory=MachineConfig)
    workload: WorkloadConfig = dataclasses.field(default_factory=WorkloadConfig)
    telemetry: TelemetryConfig = dataclasses.field(default_factory=TelemetryConfig)
    sweep: SweepConfig = dataclasses.field(default_factory=SweepConfig)


def register_config_store() -> None:
    """Register structured Hydra config schemas once."""
    global CONFIG_STORE_REGISTERED
    if CONFIG_STORE_REGISTERED:
        return
    config_store = hydra.core.config_store.ConfigStore.instance()
    config_store.store(name="tooling_schema", node=ToolingConfig)
    config_store.store(group="dataset", name="dataset_schema", node=DatasetConfig)
    config_store.store(group="machine", name="machine_schema", node=MachineConfig)
    config_store.store(group="workload", name="workload_schema", node=WorkloadConfig)
    config_store.store(group="telemetry", name="telemetry_schema", node=TelemetryConfig)
    config_store.store(group="sweep", name="sweep_schema", node=SweepConfig)
    CONFIG_STORE_REGISTERED = True


def compose_config(
    *,
    config_name: str = "config",
    overrides: typing.Sequence[str] | None = None,
    include_hydra_config: bool = False,
) -> omegaconf.DictConfig:
    """Compose a tooling config with Hydra.

    Args:
        config_name: Config name in the tooling config package.
        overrides: Optional Hydra overrides.
        include_hydra_config: Whether to include Hydra's own config node.

    Returns:
        Composed Hydra configuration.

    """
    register_config_store()
    override_list = list(overrides) if overrides is not None else []
    with hydra.initialize_config_module(config_module=CONFIG_MODULE, version_base=None):
        return hydra.compose(
            config_name=config_name,
            overrides=override_list,
            return_hydra_config=include_hydra_config,
        )


def instantiate_config(config: omegaconf.DictConfig) -> ToolingConfig:
    """Convert a composed config into the typed tooling dataclass.

    Args:
        config: Composed config without Hydra's own config node.

    Returns:
        Typed tooling configuration.

    """
    config_object = typing.cast(
        "dict[str, typing.Any]",
        omegaconf.OmegaConf.to_container(config, resolve=True),
    )
    dataset_values = typing.cast("dict[str, typing.Any]", config_object["dataset"])
    machine_values = typing.cast("dict[str, typing.Any]", config_object["machine"])
    workload_values = typing.cast("dict[str, typing.Any]", config_object["workload"])
    telemetry_values = typing.cast("dict[str, typing.Any]", config_object["telemetry"])
    sweep_values = typing.cast("dict[str, typing.Any]", config_object["sweep"])
    return ToolingConfig(
        dataset=DatasetConfig(
            data_directory=str(dataset_values["data_directory"]),
            bgen_file=str(dataset_values["bgen_file"]),
            sample_file=str(dataset_values["sample_file"]),
            phenotype_file=str(dataset_values["phenotype_file"]),
            prediction_list=str(dataset_values["prediction_list"]),
            phenotype_columns=[
                str(value) for value in typing.cast("list[typing.Any]", dataset_values["phenotype_columns"])
            ],
        ),
        machine=MachineConfig(
            name=str(machine_values["name"]),
            device=str(machine_values["device"]),
            slurm_node=str(machine_values["slurm_node"]) if machine_values["slurm_node"] is not None else None,
            cpus_per_task=int(machine_values["cpus_per_task"]) if machine_values["cpus_per_task"] is not None else None,
            memory=str(machine_values["memory"]) if machine_values["memory"] is not None else None,
        ),
        workload=WorkloadConfig(
            name=str(workload_values["name"]),
            chunk_size=int(workload_values["chunk_size"]),
            variant_limit=int(workload_values["variant_limit"])
            if workload_values["variant_limit"] is not None
            else None,
            repeat_count=int(workload_values["repeat_count"]),
            staging_depth=int(workload_values["staging_depth"]),
            native_callback_batch_size=int(workload_values.get("native_callback_batch_size", 1)),
            output_writer_thread_count=int(workload_values["output_writer_thread_count"]),
            output_writer_queue_depth=int(workload_values["output_writer_queue_depth"]),
        ),
        telemetry=TelemetryConfig(
            output_parent=str(telemetry_values["output_parent"]),
            json_summary_path=(
                str(telemetry_values["json_summary_path"])
                if telemetry_values["json_summary_path"] is not None
                else None
            ),
            markdown_summary_path=(
                str(telemetry_values["markdown_summary_path"])
                if telemetry_values["markdown_summary_path"] is not None
                else None
            ),
            stage_timing_mode=str(telemetry_values["stage_timing_mode"]),
        ),
        sweep=SweepConfig(
            chunk_sizes=[int(value) for value in typing.cast("list[typing.Any]", sweep_values["chunk_sizes"])],
            path_modes=[str(value) for value in typing.cast("list[typing.Any]", sweep_values["path_modes"])],
            sample_selection_modes=[
                str(value) for value in typing.cast("list[typing.Any]", sweep_values["sample_selection_modes"])
            ],
            decode_tile_variant_counts=[
                int(value) if value is not None else None
                for value in typing.cast("list[typing.Any]", sweep_values["decode_tile_variant_counts"])
            ],
            rayon_thread_counts=[
                int(value) if value is not None else None
                for value in typing.cast("list[typing.Any]", sweep_values["rayon_thread_counts"])
            ],
            trusted_no_missing_diploid_modes=[
                bool(value)
                for value in typing.cast("list[typing.Any]", sweep_values["trusted_no_missing_diploid_modes"])
            ],
            storage_modes=[str(value) for value in typing.cast("list[typing.Any]", sweep_values["storage_modes"])],
            fallback_density_scenarios=[
                str(value) for value in typing.cast("list[typing.Any]", sweep_values["fallback_density_scenarios"])
            ],
        ),
    )
