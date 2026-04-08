
## 2026-04-08 - [Avoid On-Device Concatenation with `jax.device_get`]
**Learning:** Concatenating lists of JAX arrays directly on the device using `jnp.concatenate` prior to transfer via `jax.device_get()` introduces significant tracing/JIT overhead and increases peak device memory usage. Passing a list (or dict of lists) directly to `jax.device_get()` and then concatenating on the CPU with `numpy.concatenate` is nearly 10x faster for results accumulation because it bypasses JIT compilation for variable sizes on the device.
**Action:** When accumulating output chunks from JAX operations, always transfer the list of JAX arrays back to the host first using `jax.device_get()` and then concatenate using `numpy.concatenate`. Avoid `jnp.concatenate` for variable-length accumulator lists.
