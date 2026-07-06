"""Native sample-group fingerprints and compute-group resolution."""

from __future__ import annotations

import typing

from g import execution_plan, types

if typing.TYPE_CHECKING:
    from g.engine.native_dispatch import models


def resolve_sample_key_mode(alignment_config: models.SampleAlignmentConfigProtocol | None) -> types.SampleKeyMode:
    """Resolve the sample key mode for native calls."""
    if alignment_config is None:
        return types.SampleKeyMode.IID
    return alignment_config.sample_key_mode


def adapt_native_phenotype_compute_group(native_compute_group: typing.Any) -> execution_plan.PhenotypeComputeGroup:
    """Convert a native resolved compute-group DTO to the public Python dataclass."""
    return execution_plan.PhenotypeComputeGroup(
        group_mode=types.PhenotypeComputeGroupMode(native_compute_group.group_mode),
        phenotype_indices=tuple(int(phenotype_index) for phenotype_index in native_compute_group.phenotype_indices),
        phenotype_names=tuple(native_compute_group.phenotype_names),
        sample_mode=types.MultiPhenotypeSampleMode(native_compute_group.sample_mode),
        sample_set_fingerprint=native_compute_group.sample_set_fingerprint,
        covariate_design_fingerprint=native_compute_group.covariate_design_fingerprint,
        prediction_alignment_fingerprint=native_compute_group.prediction_alignment_fingerprint,
    )
