"""Native scheduling policy helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

from g import _core


def native_schedule_policy() -> _core.NativeSchedulePolicy:
    """Build the native schedule policy handle."""
    return _core.NativeSchedulePolicy()


def resolve_effective_trusted_no_missing_diploid(
    *,
    trusted_no_missing_diploid: bool,
    uses_packed8_genotypes: bool,
) -> bool:
    """Return trusted BGEN mode after packed8 requirements are applied."""
    return bool(
        native_schedule_policy().resolve_effective_trusted_no_missing_diploid(
            trusted_no_missing_diploid,
            uses_packed8_genotypes,
        )
    )


def intersect_committed_chunk_identifier_sets(
    committed_chunk_identifier_sets: tuple[set[int], ...],
) -> set[int]:
    """Return chunk identifiers already committed by every output in a delivery."""
    native_committed_chunk_identifier_sets = tuple(
        tuple(committed_chunk_identifier_set) for committed_chunk_identifier_set in committed_chunk_identifier_sets
    )
    return set(
        native_schedule_policy().intersect_committed_chunk_identifier_sets(native_committed_chunk_identifier_sets)
    )


def resolve_grouped_union_callback_batch_size(*, native_callback_batch_size: int) -> int:
    """Return the validated callback batch size for grouped union delivery."""
    return int(
        native_schedule_policy().resolve_grouped_union_callback_batch_size(
            native_callback_batch_size=native_callback_batch_size,
        )
    )


def plan_gpu_genotype_format_auto_to_dosage(
    *,
    requested_gpu_genotype_format: str,
    resolution_reason: str,
) -> _core.NativeGpuGenotypeFormatResolutionPlan:
    """Plan non-profiled auto GPU genotype format resolution to dosage."""
    return native_schedule_policy().plan_gpu_genotype_format_auto_to_dosage(
        requested_gpu_genotype_format=requested_gpu_genotype_format,
        resolution_reason=resolution_reason,
    )


def resolve_manifest_gpu_genotype_format(
    *,
    resume: bool,
    manifest_gpu_genotype_format: str | None,
    association_backend_genotype_format: str | None,
) -> str | None:
    """Resolve a concrete GPU genotype format from existing manifest fields."""
    native_gpu_genotype_format = native_schedule_policy().resolve_manifest_gpu_genotype_format(
        resume=resume,
        manifest_gpu_genotype_format=manifest_gpu_genotype_format,
        association_backend_genotype_format=association_backend_genotype_format,
    )
    if native_gpu_genotype_format is None:
        return None
    return native_gpu_genotype_format


def plan_single_trait_binary_gpu_genotype_format_resolution(
    *,
    requested_gpu_genotype_format: str,
    manifest_gpu_genotype_format: str | None,
    association_backend_genotype_format: str | None,
    resume: bool,
    jax_device: str,
) -> _core.NativeGpuGenotypeFormatResolutionPlan:
    """Plan single-trait binary GPU genotype format resolution."""
    return native_schedule_policy().plan_single_trait_binary_gpu_genotype_format_resolution(
        requested_gpu_genotype_format=requested_gpu_genotype_format,
        manifest_gpu_genotype_format=manifest_gpu_genotype_format,
        association_backend_genotype_format=association_backend_genotype_format,
        resume=resume,
        jax_device=jax_device,
    )


def plan_auto_gpu_genotype_format_after_trusted_validation(
    *,
    fallback_error: str | None,
) -> _core.NativeGpuGenotypeFormatResolutionPlan:
    """Plan auto GPU genotype format after trusted BGEN validation."""
    return native_schedule_policy().plan_auto_gpu_genotype_format_after_trusted_validation(
        fallback_error=fallback_error,
    )
