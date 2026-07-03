from __future__ import annotations

import enum

import pytest

from g import _core, types
from g.engine.regenie2_pipeline import backend as backend_planner


class ExpectedPacked8Delivery(enum.Enum):
    DISABLED = False
    ENABLED = True


@pytest.mark.parametrize(
    ("association_mode", "jax_device", "gpu_genotype_format", "expected_backend_kind", "expected_packed8_delivery"),
    [
        (
            types.AssociationMode.REGENIE2_LINEAR,
            types.Device.CPU,
            types.GpuGenotypeFormat.DOSAGE,
            types.AssociationBackendKind.JAX_DOSAGE,
            ExpectedPacked8Delivery.DISABLED,
        ),
        (
            types.AssociationMode.REGENIE2_LINEAR,
            types.Device.GPU,
            types.GpuGenotypeFormat.DOSAGE,
            types.AssociationBackendKind.JAX_DOSAGE,
            ExpectedPacked8Delivery.DISABLED,
        ),
        (
            types.AssociationMode.REGENIE2_LINEAR,
            types.Device.GPU,
            types.GpuGenotypeFormat.PACKED8,
            types.AssociationBackendKind.JAX_PACKED8,
            ExpectedPacked8Delivery.ENABLED,
        ),
        (
            types.AssociationMode.REGENIE2_BINARY,
            types.Device.CPU,
            types.GpuGenotypeFormat.DOSAGE,
            types.AssociationBackendKind.JAX_DOSAGE,
            ExpectedPacked8Delivery.DISABLED,
        ),
        (
            types.AssociationMode.REGENIE2_BINARY,
            types.Device.GPU,
            types.GpuGenotypeFormat.DOSAGE,
            types.AssociationBackendKind.JAX_DOSAGE,
            ExpectedPacked8Delivery.DISABLED,
        ),
        (
            types.AssociationMode.REGENIE2_BINARY,
            types.Device.GPU,
            types.GpuGenotypeFormat.PACKED8,
            types.AssociationBackendKind.JAX_PACKED8,
            ExpectedPacked8Delivery.ENABLED,
        ),
    ],
)
def test_plan_association_backend_resolves_current_jax_paths(
    association_mode: types.AssociationMode,
    jax_device: types.Device,
    gpu_genotype_format: types.GpuGenotypeFormat,
    expected_backend_kind: types.AssociationBackendKind,
    expected_packed8_delivery: ExpectedPacked8Delivery,
) -> None:
    plan = backend_planner.plan_association_backend(
        association_mode=association_mode,
        jax_device=jax_device,
        gpu_genotype_format=gpu_genotype_format,
    )

    assert plan.backend_kind == expected_backend_kind
    assert plan.association_mode == association_mode
    assert plan.jax_device == jax_device
    assert plan.genotype_format == gpu_genotype_format
    assert plan.uses_variant_major_packed8_delivery is expected_packed8_delivery.value
    assert plan.manifest_metadata() == backend_planner.AssociationBackendMetadata(
        kind=expected_backend_kind.value,
        association_mode=association_mode.value,
        device=jax_device.value,
        genotype_format=gpu_genotype_format.value,
    )
    assert not hasattr(_core, "plan_association_backend_payload")
    assert not hasattr(_core, "resolve_association_mode_value")
    assert not hasattr(_core, "normalize_binary_correction_payload")
    assert not hasattr(_core, "build_phenotype_compute_groups_payload")
    assert not hasattr(_core, "build_phenotype_compute_group_id_value")
    assert not hasattr(_core, "build_phenotype_output_directory_name")


def test_plan_association_backend_rejects_unresolved_auto_format() -> None:
    with pytest.raises(ValueError, match="must be resolved"):
        backend_planner.plan_association_backend(
            association_mode=types.AssociationMode.REGENIE2_BINARY,
            jax_device=types.Device.GPU,
            gpu_genotype_format=types.GpuGenotypeFormat.AUTO,
        )


def test_native_host_planning_returns_typed_handles() -> None:
    native_policy = _core.NativeHostPlanningPolicy()

    backend_plan = native_policy.plan_association_backend(
        types.AssociationMode.REGENIE2_LINEAR.value,
        types.Device.GPU.value,
        types.GpuGenotypeFormat.PACKED8.value,
    )
    compute_groups = native_policy.build_phenotype_compute_groups(
        ["one", "two"],
        types.MultiPhenotypeSampleMode.COMPLETE_CASE.value,
    )

    assert isinstance(backend_plan, _core.NativeHostAssociationBackendPlan)
    assert backend_plan.backend_kind == types.AssociationBackendKind.JAX_PACKED8.value
    assert backend_plan.uses_variant_major_packed8_delivery is True
    assert len(compute_groups) == 1
    assert isinstance(compute_groups[0], _core.NativeHostPhenotypeComputeGroupPlan)
    assert compute_groups[0].group_mode == types.PhenotypeComputeGroupMode.COMPLETE_CASE.value
    assert compute_groups[0].phenotype_indices == [0, 1]
