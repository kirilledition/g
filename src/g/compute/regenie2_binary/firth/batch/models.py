"""Firth candidate batch data models."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import candidates as regenie2_binary_candidate_planning

SPARSE_FIRTH_CARRIER_CAPACITY = 64


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedFirthCandidateBatch:
    """Prepared fixed-capacity Firth candidate lanes.

    Attributes:
        batch_plan: Fixed-shape candidate index and active-lane plan.
        candidate_inputs: Ordered candidate lane inputs.
        initial_coefficients: Initial full-model coefficients for each candidate lane.
        full_null_deviance: Full-sample null deviance reused by compact sparse scalar lanes.

    """

    batch_plan: regenie2_binary_candidate_planning.FirthBatchPlan
    candidate_inputs: regenie2_binary_candidate_planning.FirthCandidateBatchInputs
    initial_coefficients: jax.Array
    full_null_deviance: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedMultiFirthCandidateBatch:
    """Prepared fixed-capacity multi-trait Firth candidate lanes.

    Attributes:
        batch_plan: Fixed-shape candidate index and active-lane plan.
        candidate_inputs: Ordered candidate lane inputs with trait and variant indices.
        initial_coefficients: Initial full-model coefficients for each candidate lane.
        full_null_deviance: Lane-specific full-sample null deviance.

    """

    batch_plan: regenie2_binary_candidate_planning.FirthBatchPlan
    candidate_inputs: regenie2_binary_candidate_planning.MultiFirthCandidateBatchInputs
    initial_coefficients: jax.Array
    full_null_deviance: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthLaneStreamPlan:
    """Fixed-shape lane stream selected from a candidate batch.

    Attributes:
        lane_indices: Candidate-batch positions packed into stream order.
        active_mask: Active mask for the packed stream.
        active_count: Number of active lanes in the stream.

    """

    lane_indices: jax.Array
    active_mask: jax.Array
    active_count: jax.Array
