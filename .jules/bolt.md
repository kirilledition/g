## 2024-05-18 - Optimized JAX array transfers
**Learning:** In JAX, concatenating a list of variably-sized arrays on the device (`jnp.concatenate`) before transferring them to the host via `jax.device_get` causes JAX tracing/JIT overhead and increased peak device memory usage.
**Action:** Always move multiple JAX arrays from device to host by passing the list (or dictionary of lists) directly to `jax.device_get()`, and then use `numpy.concatenate` to concatenate the host NumPy arrays.
