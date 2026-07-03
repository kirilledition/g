"""Backend planning helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g.engine import backend_planner

if typing.TYPE_CHECKING:
    from g import types

type AssociationBackendPlan = backend_planner.AssociationBackendPlan


def plan_association_backend(
    *,
    association_mode: types.AssociationMode,
    jax_device: types.Device,
    gpu_genotype_format: types.GpuGenotypeFormat,
) -> AssociationBackendPlan:
    """Select the concrete backend used by association execution."""
    return backend_planner.plan_association_backend(
        association_mode=association_mode,
        jax_device=jax_device,
        gpu_genotype_format=gpu_genotype_format,
    )
