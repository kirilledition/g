"""Debug Triton kernel with 2D arrays and strides - FIXED v2."""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def debug_column_sum_kernel(
    matrix_ptr,
    output_ptr,
    num_rows,
    num_cols,
    row_stride,
    col_stride,
    block_size: tl.constexpr,
) -> None:
    """Sum each column of a 2D matrix."""
    pid = tl.program_id(axis=0)
    col_idx = pid
    
    # Each thread handles one column, accumulating within its block
    accumulator = 0.0
    
    for row_start in range(0, num_rows, block_size):
        row_offsets = row_start + tl.arange(0, block_size)
        mask = row_offsets < num_rows
        
        # Compute pointer to this column's elements
        ptrs = matrix_ptr + row_offsets * row_stride + col_idx * col_stride
        
        x = tl.load(ptrs, mask=mask, other=0.0)
        # Sum across the block
        block_sum = tl.sum(x, axis=0)
        accumulator += block_sum
    
    # Store scalar result
    tl.store(output_ptr + col_idx, accumulator)


def test_column_sum():
    """Test column-wise sum with strides."""
    rows, cols = 4, 4
    matrix = jnp.arange(rows * cols, dtype=jnp.float32).reshape(rows, cols)
    
    expected = jnp.sum(matrix, axis=0)
    
    row_stride = cols
    col_stride = 1
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix:\n{matrix}")
    print(f"Row stride: {row_stride}, Col stride: {col_stride}")
    print(f"Expected: {expected}")
    
    block_size = 8
    grid = (cols,)
    
    result = jt.triton_call(
        matrix,
        kernel=debug_column_sum_kernel,
        out_shape=jax.ShapeDtypeStruct((cols,), jnp.float32),
        grid=grid,
        num_rows=rows,
        num_cols=cols,
        row_stride=row_stride,
        col_stride=col_stride,
        block_size=block_size,
    )
    
    print(f"Triton result: {result}")
    print(f"Match: {jnp.allclose(result, expected)}")


if __name__ == "__main__":
    test_column_sum()
