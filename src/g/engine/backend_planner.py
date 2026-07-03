"""Association backend planning before native engine dispatch."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from g import _core, types


@dataclass(frozen=True)
class AssociationBackendMetadata:
    """Stable metadata for one selected association backend.

    Attributes:
        kind: Concrete backend implementation.
        association_mode: Statistical association mode.
        device: JAX device requested for the backend.
        genotype_format: Native genotype delivery format.

    """

    kind: str
    association_mode: str
    device: str
    genotype_format: str


@dataclass(frozen=True)
class AssociationBackendPlan:
    """Resolved backend choice for one association run.

    Attributes:
        backend_kind: Concrete backend implementation to execute.
        association_mode: Statistical association mode using the backend.
        jax_device: Requested JAX device for the backend.
        genotype_format: Native genotype delivery format for JAX inputs.
        uses_variant_major_packed8_delivery: Whether native dispatch should
            deliver packed8 probability-pair chunks.

    """

    backend_kind: types.AssociationBackendKind
    association_mode: types.AssociationMode
    jax_device: types.Device
    genotype_format: types.GpuGenotypeFormat
    uses_variant_major_packed8_delivery: bool

    def manifest_metadata(self) -> AssociationBackendMetadata:
        """Return stable manifest metadata for this backend selection."""
        return AssociationBackendMetadata(
            kind=self.backend_kind.value,
            association_mode=self.association_mode.value,
            device=self.jax_device.value,
            genotype_format=self.genotype_format.value,
        )


def plan_association_backend(
    *,
    association_mode: types.AssociationMode,
    jax_device: types.Device,
    gpu_genotype_format: types.GpuGenotypeFormat,
) -> AssociationBackendPlan:
    """Select the concrete backend used by association execution."""
    native_host_planning_policy = _core.NativeHostPlanningPolicy()
    backend_payload = native_host_planning_policy.plan_association_backend_payload(
        association_mode.value,
        jax_device.value,
        gpu_genotype_format.value,
    )
    return AssociationBackendPlan(
        backend_kind=types.AssociationBackendKind(typing.cast("str", backend_payload["backend_kind"])),
        association_mode=types.AssociationMode(typing.cast("str", backend_payload["association_mode"])),
        jax_device=types.Device(typing.cast("str", backend_payload["jax_device"])),
        genotype_format=types.GpuGenotypeFormat(typing.cast("str", backend_payload["genotype_format"])),
        uses_variant_major_packed8_delivery=typing.cast(
            "bool",
            backend_payload["uses_variant_major_packed8_delivery"],
        ),
    )
