from __future__ import annotations

import enum

import pytest

from g import types
from g.engine import backend_planner


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
    assert plan.manifest_metadata() == {
        "kind": expected_backend_kind.value,
        "association_mode": association_mode.value,
        "device": jax_device.value,
        "genotype_format": gpu_genotype_format.value,
    }
