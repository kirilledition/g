"""Pure-Triton experiments for the linear association path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    import numpy as np

TRITON_VARIANT_TILE_SIZE = 128


class TritonLinearAssociationStatistics(NamedTuple):
    """Linear association statistics computed on the GPU."""

    beta: torch.Tensor
    standard_error: torch.Tensor
    test_statistic: torch.Tensor
    valid_mask: torch.Tensor


@dataclass
class TritonLinearAssociationState:
    """Precomputed covariate-only state for Triton linear experiments."""

    covariate_matrix: torch.Tensor
    covariate_matrix_transpose: torch.Tensor
    padded_covariate_matrix: torch.Tensor
    covariate_crossproduct_cholesky_factor: torch.Tensor
    phenotype_residual: torch.Tensor
    phenotype_residual_sum_squares: torch.Tensor
    covariate_count: int
    padded_covariate_count: int


class TritonReductionStatistics(NamedTuple):
    """Fused reduction outputs for one genotype chunk."""

    covariate_genotype_crossproduct: torch.Tensor
    genotype_sum_squares: torch.Tensor
    covariance_with_phenotype: torch.Tensor


def round_up_to_power_of_two(value: int) -> int:
    """Round a positive integer up to the next power of two."""
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


def prepare_triton_linear_association_state(
    covariate_matrix: np.ndarray,
    phenotype_vector: np.ndarray,
    device: torch.device,
) -> TritonLinearAssociationState:
    """Precompute covariate-only state for the Triton linear experiment."""
    covariate_matrix_tensor = torch.as_tensor(covariate_matrix, device=device, dtype=torch.float32)
    phenotype_vector_tensor = torch.as_tensor(phenotype_vector, device=device, dtype=torch.float32)
    covariate_count = int(covariate_matrix_tensor.shape[1])
    padded_covariate_count = round_up_to_power_of_two(covariate_count)
    padded_covariate_matrix = torch.zeros(
        (covariate_matrix_tensor.shape[0], padded_covariate_count),
        device=device,
        dtype=torch.float32,
    )
    padded_covariate_matrix[:, :covariate_count] = covariate_matrix_tensor
    covariate_matrix_transpose = covariate_matrix_tensor.transpose(0, 1).contiguous()
    covariate_crossproduct = covariate_matrix_transpose @ covariate_matrix_tensor
    covariate_crossproduct_cholesky_factor = torch.linalg.cholesky(covariate_crossproduct)
    phenotype_projection = torch.cholesky_solve(
        (covariate_matrix_transpose @ phenotype_vector_tensor).unsqueeze(1),
        covariate_crossproduct_cholesky_factor,
    ).squeeze(1)
    phenotype_residual = phenotype_vector_tensor - covariate_matrix_tensor @ phenotype_projection
    phenotype_residual_sum_squares = torch.dot(phenotype_residual, phenotype_residual)
    return TritonLinearAssociationState(
        covariate_matrix=covariate_matrix_tensor,
        covariate_matrix_transpose=covariate_matrix_transpose,
        padded_covariate_matrix=padded_covariate_matrix.contiguous(),
        covariate_crossproduct_cholesky_factor=covariate_crossproduct_cholesky_factor,
        phenotype_residual=phenotype_residual.contiguous(),
        phenotype_residual_sum_squares=phenotype_residual_sum_squares,
        covariate_count=covariate_count,
        padded_covariate_count=padded_covariate_count,
    )


@triton.autotune(
    configs=[
        triton.Config({"sample_tile_size": 64}, num_warps=4),
        triton.Config({"sample_tile_size": 128}, num_warps=4),
        triton.Config({"sample_tile_size": 128}, num_warps=8),
        triton.Config({"sample_tile_size": 256}, num_warps=8),
    ],
    key=["sample_count", "variant_count", "padded_covariate_count"],
)
@triton.jit
def fused_linear_statistics_kernel(  # noqa: D103
    genotype_matrix_pointer,
    padded_covariate_matrix_pointer,
    phenotype_residual_pointer,
    covariate_genotype_crossproduct_pointer,
    genotype_sum_squares_pointer,
    covariance_with_phenotype_pointer,
    sample_count,
    variant_count,
    padded_covariate_count,
    genotype_sample_stride,
    genotype_variant_stride,
    covariate_sample_stride,
    covariate_stride,
    crossproduct_covariate_stride,
    crossproduct_variant_stride,
    sample_tile_size: tl.constexpr,
    variant_tile_size: tl.constexpr,
    covariate_tile_size: tl.constexpr,
) -> None:
    variant_program_index = tl.program_id(axis=0)
    variant_offsets = variant_program_index * variant_tile_size + tl.arange(0, variant_tile_size)
    variant_mask = variant_offsets < variant_count
    covariate_offsets = tl.arange(0, covariate_tile_size)
    genotype_sum_squares_accumulator = tl.zeros((variant_tile_size,), dtype=tl.float32)
    covariance_with_phenotype_accumulator = tl.zeros((variant_tile_size,), dtype=tl.float32)
    covariate_genotype_crossproduct_accumulator = tl.zeros(
        (covariate_tile_size, variant_tile_size),
        dtype=tl.float32,
    )

    for sample_start in range(0, sample_count, sample_tile_size):
        sample_offsets = sample_start + tl.arange(0, sample_tile_size)
        sample_mask = sample_offsets < sample_count
        genotype_tile_pointer = (
            genotype_matrix_pointer
            + sample_offsets[:, None] * genotype_sample_stride
            + variant_offsets[None, :] * genotype_variant_stride
        )
        phenotype_residual_tile_pointer = phenotype_residual_pointer + sample_offsets
        covariate_tile_pointer = (
            padded_covariate_matrix_pointer
            + sample_offsets[:, None] * covariate_sample_stride
            + covariate_offsets[None, :] * covariate_stride
        )
        genotype_tile = tl.load(
            genotype_tile_pointer,
            mask=sample_mask[:, None] & variant_mask[None, :],
            other=0.0,
        )
        phenotype_residual_tile = tl.load(
            phenotype_residual_tile_pointer,
            mask=sample_mask,
            other=0.0,
        )
        covariate_tile = tl.load(
            covariate_tile_pointer,
            mask=sample_mask[:, None] & (covariate_offsets[None, :] < padded_covariate_count),
            other=0.0,
        )
        genotype_sum_squares_accumulator += tl.sum(genotype_tile * genotype_tile, axis=0)
        covariance_with_phenotype_accumulator += tl.sum(
            genotype_tile * phenotype_residual_tile[:, None],
            axis=0,
        )
        covariate_genotype_crossproduct_accumulator += tl.dot(
            tl.trans(covariate_tile),
            genotype_tile,
            input_precision="ieee",
        )

    genotype_sum_squares_pointer = genotype_sum_squares_pointer + variant_offsets
    covariance_with_phenotype_pointer = covariance_with_phenotype_pointer + variant_offsets
    tl.store(genotype_sum_squares_pointer, genotype_sum_squares_accumulator, mask=variant_mask)
    tl.store(
        covariance_with_phenotype_pointer,
        covariance_with_phenotype_accumulator,
        mask=variant_mask,
    )

    crossproduct_pointer = (
        covariate_genotype_crossproduct_pointer
        + covariate_offsets[:, None] * crossproduct_covariate_stride
        + variant_offsets[None, :] * crossproduct_variant_stride
    )
    tl.store(
        crossproduct_pointer,
        covariate_genotype_crossproduct_accumulator,
        mask=(covariate_offsets[:, None] < padded_covariate_count) & variant_mask[None, :],
    )


def compute_triton_reduction_statistics(
    linear_association_state: TritonLinearAssociationState,
    genotype_matrix: torch.Tensor,
) -> TritonReductionStatistics:
    """Compute fused genotype statistics with a Triton kernel."""
    sample_count, variant_count = genotype_matrix.shape
    padded_covariate_genotype_crossproduct = torch.empty(
        (linear_association_state.padded_covariate_count, variant_count),
        device=genotype_matrix.device,
        dtype=torch.float32,
    )
    genotype_sum_squares = torch.empty((variant_count,), device=genotype_matrix.device, dtype=torch.float32)
    covariance_with_phenotype = torch.empty((variant_count,), device=genotype_matrix.device, dtype=torch.float32)
    grid = (triton.cdiv(variant_count, TRITON_VARIANT_TILE_SIZE),)
    fused_linear_statistics_kernel[grid](
        genotype_matrix,
        linear_association_state.padded_covariate_matrix,
        linear_association_state.phenotype_residual,
        padded_covariate_genotype_crossproduct,
        genotype_sum_squares,
        covariance_with_phenotype,
        sample_count,
        variant_count,
        linear_association_state.padded_covariate_count,
        genotype_matrix.stride(0),
        genotype_matrix.stride(1),
        linear_association_state.padded_covariate_matrix.stride(0),
        linear_association_state.padded_covariate_matrix.stride(1),
        padded_covariate_genotype_crossproduct.stride(0),
        padded_covariate_genotype_crossproduct.stride(1),
        variant_tile_size=TRITON_VARIANT_TILE_SIZE,
        covariate_tile_size=linear_association_state.padded_covariate_count,
    )
    return TritonReductionStatistics(
        covariate_genotype_crossproduct=padded_covariate_genotype_crossproduct[
            : linear_association_state.covariate_count
        ],
        genotype_sum_squares=genotype_sum_squares,
        covariance_with_phenotype=covariance_with_phenotype,
    )


def compute_linear_association_statistics_with_triton(
    linear_association_state: TritonLinearAssociationState,
    genotype_matrix: np.ndarray,
) -> TritonLinearAssociationStatistics:
    """Compute linear association statistics using Triton plus Torch linear algebra."""
    genotype_matrix_tensor = torch.as_tensor(genotype_matrix, device=linear_association_state.covariate_matrix.device)
    genotype_matrix_tensor = genotype_matrix_tensor.to(dtype=torch.float32).contiguous()
    reduction_statistics = compute_triton_reduction_statistics(linear_association_state, genotype_matrix_tensor)
    genotype_projection = torch.cholesky_solve(
        reduction_statistics.covariate_genotype_crossproduct,
        linear_association_state.covariate_crossproduct_cholesky_factor,
    )
    projection_sum_squares = torch.sum(
        reduction_statistics.covariate_genotype_crossproduct * genotype_projection,
        dim=0,
    )
    genotype_residual_sum_squares = torch.clamp(
        reduction_statistics.genotype_sum_squares - projection_sum_squares,
        min=0.0,
    )
    safe_denominator = torch.where(
        genotype_residual_sum_squares > 0.0,
        genotype_residual_sum_squares,
        torch.full_like(genotype_residual_sum_squares, torch.nan),
    )
    beta = reduction_statistics.covariance_with_phenotype / safe_denominator
    sample_count = int(linear_association_state.covariate_matrix.shape[0])
    degrees_of_freedom = sample_count - linear_association_state.covariate_count - 1
    residual_sum_squares = linear_association_state.phenotype_residual_sum_squares - (
        beta * reduction_statistics.covariance_with_phenotype
    )
    residual_sum_squares = torch.clamp(residual_sum_squares, min=0.0)
    residual_variance = residual_sum_squares / degrees_of_freedom
    standard_error = torch.sqrt(residual_variance / safe_denominator)
    test_statistic = beta / standard_error
    valid_mask = torch.isfinite(beta) & torch.isfinite(standard_error) & (standard_error > 0.0)
    return TritonLinearAssociationStatistics(
        beta=beta,
        standard_error=standard_error,
        test_statistic=test_statistic,
        valid_mask=valid_mask,
    )
