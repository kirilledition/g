"""Select logical outputs and materialize one combined device payload."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from g.backend import contracts
from g.compute.common import compressed_genotype
from g.compute.common import result as association_result

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def select_trait_variant_values(
    values: jax.Array,
    active_trait_indices: jax.Array | None,
    logical_variant_count: int,
) -> jax.Array:
    """Select traits in caller order and remove the padded variant tail."""
    selected_values = values if active_trait_indices is None else jnp.take(values, active_trait_indices, axis=0)
    return selected_values[:, :logical_variant_count]


def select_association(
    association: contracts.DeviceAssociationResult,
    active_trait_indices: npt.NDArray[np.int32] | None,
    logical_variant_count: int,
) -> contracts.DeviceAssociationResult:
    """Retain full-batch buffers or prepare selected logical association arrays."""
    if active_trait_indices is None and logical_variant_count == association.beta.shape[1]:
        return association
    active_trait_index_array = (
        None if active_trait_indices is None else jnp.asarray(active_trait_indices, dtype=jnp.int32)
    )
    beta = jnp.asarray(
        select_trait_variant_values(association.beta, active_trait_index_array, logical_variant_count),
        dtype=jnp.float32,
    )
    standard_error = jnp.asarray(
        select_trait_variant_values(association.standard_error, active_trait_index_array, logical_variant_count),
        dtype=jnp.float32,
    )
    chi_squared = jnp.asarray(
        select_trait_variant_values(association.chi_squared, active_trait_index_array, logical_variant_count),
        dtype=jnp.float32,
    )
    log10_p_value = jnp.asarray(
        select_trait_variant_values(association.log10_p_value, active_trait_index_array, logical_variant_count),
        dtype=jnp.float32,
    )
    if association.correction_code is None:
        return association_result.AssociationResult(
            beta=beta,
            standard_error=standard_error,
            chi_squared=chi_squared,
            log10_p_value=log10_p_value,
            correction_code=None,
        )
    return association_result.AssociationResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        correction_code=jnp.asarray(
            select_trait_variant_values(association.correction_code, active_trait_index_array, logical_variant_count),
            dtype=jnp.uint8,
        ),
    )


def materialize_batch(
    device_result: contracts.DeviceAssociationBatch,
    active_trait_indices: npt.NDArray[np.int32] | None,
    logical_variant_count: int,
) -> contracts.HostMaterializedAssociationBatch:
    """Materialize selected association and packed8 arrays in one transfer."""
    association = device_result.association
    reuse_full_batch_arrays = active_trait_indices is None and logical_variant_count == association.beta.shape[1]
    selected_association = select_association(association, active_trait_indices, logical_variant_count)

    raw_packed8_statistics = device_result.raw_packed8_statistics
    if raw_packed8_statistics is None or reuse_full_batch_arrays:
        materializable_raw_statistics = raw_packed8_statistics
    else:
        materializable_raw_statistics = compressed_genotype.Packed8RawStatistics(
            dosage_sums=raw_packed8_statistics.dosage_sums[:logical_variant_count],
            dosage_square_sums=raw_packed8_statistics.dosage_square_sums[:logical_variant_count],
            statuses=raw_packed8_statistics.statuses[:logical_variant_count],
            selected_sample_count=raw_packed8_statistics.selected_sample_count,
        )
    materialized_batch = jax.device_get(
        contracts.AssociationBatch(
            association=selected_association,
            raw_packed8_statistics=materializable_raw_statistics,
            firth_candidate_count=device_result.firth_candidate_count,
            firth_candidate_capacity=device_result.firth_candidate_capacity,
        )
    )
    materialized_firth_candidate_count = materialized_batch.firth_candidate_count
    materialized_firth_candidate_capacity = materialized_batch.firth_candidate_capacity
    if (materialized_firth_candidate_count is None) != (materialized_firth_candidate_capacity is None):
        raise ValueError("Firth candidate count and capacity must be materialized together.")
    if materialized_firth_candidate_count is not None and materialized_firth_candidate_capacity is not None:
        host_firth_candidate_count = int(materialized_firth_candidate_count)
        if host_firth_candidate_count > materialized_firth_candidate_capacity:
            message = (
                f"Aggregate Firth candidate count {host_firth_candidate_count} exceeded the static aggregate "
                f"capacity of {materialized_firth_candidate_capacity}. Increase [compute] firth_candidate_capacity "
                "(the per-trait capacity scaling value) and rerun."
            )
            raise ValueError(message)
    return materialized_batch
