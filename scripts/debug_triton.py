"""Debug Triton kernel with jax-triton."""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def debug_sum_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    block_size: tl.constexpr,
) -> None:
    """Simple sum reduction for debugging."""
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < num_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Simple sum
    block_sum = tl.sum(x, axis=0)
    
    # Store result (one per block)
    tl.store(output_ptr + pid, block_sum)


def test_simple_sum():
    """Test a simple Triton sum kernel."""
    # Create simple test data
    x = jnp.arange(16, dtype=jnp.float32)
    
    block_size = 8
    grid = (triton.cdiv(x.shape[0], block_size),)
    
    # Call Triton kernel
    result = jt.triton_call(
        x,
        kernel=debug_sum_kernel,
        out_shape=jax.ShapeDtypeStruct((grid[0],), jnp.float32),
        grid=grid,
        num_elements=x.shape[0],
        block_size=block_size,
    )
    
    # Check result
    expected = jnp.sum(x)
    actual = jnp.sum(result)
    
    print(f"Input: {x}")
    print(f"Triton result (per block): {result}")
    print(f"Sum of Triton results: {actual}")
    print(f"Expected sum: {expected}")
    print(f"Match: {jnp.allclose(actual, expected)}")


if __name__ == "__main__":
    test_simple_sum()
