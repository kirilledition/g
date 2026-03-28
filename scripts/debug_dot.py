"""Debug the tl.dot operation for X^T G."""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def debug_dot_kernel(
    covariate_ptr,
    genotype_ptr,
    output_ptr,
    num_samples,
    num_covariates,
    num_variants,
    covariate_row_stride,
    covariate_col_stride,
    genotype_row_stride,
    genotype_col_stride,
    output_cov_stride,
    output_var_stride,
    sample_block_size: tl.constexpr,
    variant_tile_size: tl.constexpr,
) -> None:
    """Compute X^T G using tl.dot."""
    # One program per variant tile
    pid = tl.program_id(axis=0)
    variant_start = pid * variant_tile_size
    variant_offsets = variant_start + tl.arange(0, variant_tile_size)
    variant_mask = variant_offsets < num_variants
    
    # Initialize accumulator for this variant tile
    # Shape: (num_covariates, variant_tile_size)
    accumulator = tl.zeros((num_covariates, variant_tile_size), dtype=tl.float32)
    
    # Iterate over samples in blocks
    for sample_start in range(0, num_samples, sample_block_size):
        sample_offsets = sample_start + tl.arange(0, sample_block_size)
        sample_mask = sample_offsets < num_samples
        
        # Load covariate tile: (sample_block_size, num_covariates)
        # X[row, col] = ptr + row * row_stride + col * col_stride
        covariate_ptrs = (
            covariate_ptr 
            + sample_offsets[:, None] * covariate_row_stride 
            + tl.arange(0, num_covariates)[None, :] * covariate_col_stride
        )
        covariate_tile = tl.load(
            covariate_ptrs,
            mask=sample_mask[:, None],
            other=0.0,
        )
        
        # Load genotype tile: (sample_block_size, variant_tile_size)
        genotype_ptrs = (
            genotype_ptr 
            + sample_offsets[:, None] * genotype_row_stride 
            + variant_offsets[None, :] * genotype_col_stride
        )
        genotype_tile = tl.load(
            genotype_ptrs,
            mask=sample_mask[:, None] & variant_mask[None, :],
            other=0.0,
        )
        
        # Compute X^T @ G for this block: (num_covariates, sample_block_size) @ (sample_block_size, variant_tile_size)
        # = (num_covariates, variant_tile_size)
        accumulator += tl.dot(tl.trans(covariate_tile), genotype_tile, input_precision="ieee")
    
    # Store result
    # Output[cov, var] = ptr + cov * cov_stride + var * var_stride
    output_ptrs = (
        output_ptr 
        + tl.arange(0, num_covariates)[:, None] * output_cov_stride 
        + variant_offsets[None, :] * output_var_stride
    )
    tl.store(output_ptrs, accumulator, mask=variant_mask[None, :])


def test_dot():
    """Test X^T G using tl.dot."""
    print("\n=== Test X^T G (tl.dot) ===")
    num_samples = 8
    num_covariates = 3  # Will be padded to 4
    num_variants = 4
    
    # Create test matrices
    covariates = jnp.arange(num_samples * num_covariates, dtype=jnp.float32).reshape(num_samples, num_covariates)
    genotype = jnp.arange(num_samples * num_variants, dtype=jnp.float32).reshape(num_samples, num_variants)
    
    # Pad covariates to power of 2
    padded_covariates = jnp.zeros((num_samples, 4), dtype=jnp.float32)
    padded_covariates = padded_covariates.at[:, :num_covariates].set(covariates)
    
    # Expected: X^T @ G
    expected = covariates.T @ genotype
    
    print(f"X shape: {covariates.shape}, G shape: {genotype.shape}")
    print(f"Expected X^T G shape: {expected.shape}")
    
    # Strides (element strides, assuming row-major)
    cov_row_stride = 4  # padded_covariates.shape[1]
    cov_col_stride = 1
    gen_row_stride = num_variants
    gen_col_stride = 1
    out_cov_stride = num_variants
    out_var_stride = 1
    
    variant_tile_size = 2
    grid = (triton.cdiv(num_variants, variant_tile_size),)
    
    result = jt.triton_call(
        padded_covariates,
        genotype,
        kernel=debug_dot_kernel,
        out_shape=jax.ShapeDtypeStruct((4, num_variants), jnp.float32),
        grid=grid,
        num_samples=num_samples,
        num_covariates=4,  # padded
        num_variants=num_variants,
        covariate_row_stride=cov_row_stride,
        covariate_col_stride=cov_col_stride,
        genotype_row_stride=gen_row_stride,
        genotype_col_stride=gen_col_stride,
        output_cov_stride=out_cov_stride,
        output_var_stride=out_var_stride,
        sample_block_size=4,
        variant_tile_size=variant_tile_size,
    )
    
    # Slice to actual size
    result_sliced = result[:num_covariates, :]
    
    print(f"Expected:\n{expected}")
    print(f"Triton:\n{result_sliced}")
    print(f"Match: {jnp.allclose(result_sliced, expected)}")


if __name__ == "__main__":
    test_dot()
