"""Association backend planning before native engine dispatch."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from g import types


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

    def manifest_metadata(self) -> dict[str, typing.Any]:
        """Return stable manifest metadata for this backend selection."""
        return {
            "kind": self.backend_kind.value,
            "association_mode": self.association_mode.value,
            "device": self.jax_device.value,
            "genotype_format": self.genotype_format.value,
        }


def plan_association_backend(
    *,
    association_mode: types.AssociationMode,
    jax_device: types.Device,
    gpu_genotype_format: types.GpuGenotypeFormat,
) -> AssociationBackendPlan:
    """Select the concrete backend used by association execution."""
    if gpu_genotype_format == types.GpuGenotypeFormat.PACKED8:
        return AssociationBackendPlan(
            backend_kind=types.AssociationBackendKind.JAX_PACKED8,
            association_mode=association_mode,
            jax_device=jax_device,
            genotype_format=gpu_genotype_format,
            uses_variant_major_packed8_delivery=True,
        )
    return AssociationBackendPlan(
        backend_kind=types.AssociationBackendKind.JAX_DOSAGE,
        association_mode=association_mode,
        jax_device=jax_device,
        genotype_format=gpu_genotype_format,
        uses_variant_major_packed8_delivery=False,
    )
