"""Debug the three operations in the linear kernel separately."""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def debug_sum_squares_kernel(
    genotype_ptr,
    output_ptr,
    num_rows,
    num_cols,
    row_stride,
    col_stride,
    block_size: tl.constexpr,
) -> None:
    """Compute sum of squares per column."""
    pid = tl.program_id(axis=0)
    col_idx = pid
    
    accumulator = 0.0
    
    for row_start in range(0, num_rows, block_size):
        row_offsets = row_start + tl.arange(0, block_size)
        mask = row_offsets < num_rows
        
        ptrs = genotype_ptr + row_offsets * row_stride + col_idx * col_stride
        x = tl.load(ptrs, mask=mask, other=0.0)
        accumulator += tl.sum(x * x, axis=0)
    
    tl.store(output_ptr + col_idx, accumulator)


@triton.jit
def debug_covariance_kernel(
    genotype_ptr,
    phenotype_ptr,
    output_ptr,
    num_rows,
    num_cols,
    row_stride,
    col_stride,
    block_size: tl.constexpr,
) -> None:
    """Compute G^T r (covariance per column)."""
    pid = tl.program_id(axis=0)
    col_idx = pid
    
    accumulator = 0.0
    
    for row_start in range(0, num_rows, block_size):
        row_offsets = row_start + tl.arange(0, block_size)
        mask = row_offsets < num_rows
        
        genotype_ptrs = genotype_ptr + row_offsets * row_stride + col_idx * col_stride
        phenotype_ptrs = phenotype_ptr + row_offsets
        
        g = tl.load(genotype_ptrs, mask=mask, other=0.0)
        p = tl.load(phenotype_ptrs, mask=mask, other=0.0)
        
        accumulator += tl.sum(g * p, axis=0)
    
    tl.store(output_ptr + col_idx, accumulator)


def test_sum_squares():
    """Test sum of squares."""
    print("\n=== Test Sum of Squares ===")
    rows, cols = 8, 4
    genotype = jnp.arange(rows * cols, dtype=jnp.float32).reshape(rows, cols)
    
    expected = jnp.sum(genotype * genotype, axis=0)
    
    row_stride = cols
    col_stride = 1
    block_size = 8
    grid = (cols,)
    
    result = jt.triton_call(
        genotype,
        kernel=debug_sum_squares_kernel,
        out_shape=jax.ShapeDtypeStruct((cols,), jnp.float32),
        grid=grid,
        num_rows=rows,
        num_cols=cols,
        row_stride=row_stride,
        col_stride=col_stride,
        block_size=block_size,
    )
    
    print(f"Expected: {expected}")
    print(f"Triton:   {result}")
    print(f"Match: {jnp.allclose(result, expected)}")


def test_covariance():
    """Test G^T r."""
    print("\n=== Test Covariance (G^T r) ===")
    rows, cols = 8, 4
    genotype = jnp.arange(rows * cols, dtype=jnp.float32).reshape(rows, cols)
    phenotype = jnp.arange(rows, dtype=jnp.float32)
    
    expected = jnp.sum(genotype * phenotype[:, None], axis=0)
    
    row_stride = cols
    col_stride = 1
    block_size = 8
    grid = (cols,)
    
    result = jt.triton_call(
        genotype,
        phenotype,
        kernel=debug_covariance_kernel,
        out_shape=jax.ShapeDtypeStruct((cols,), jnp.float32),
        grid=grid,
        num_rows=rows,
        num_cols=cols,
        row_stride=row_stride,
        col_stride=col_stride,
        block_size=block_size,
    )
    
    print(f"Expected: {expected}")
    print(f"Triton:   {result}")
    print(f"Match: {jnp.allclose(result, expected)}")


if __name__ == "__main__":
    test_sum_squares()
    test_covariance()
