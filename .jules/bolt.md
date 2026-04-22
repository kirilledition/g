## 2026-04-22 - [JAX Array Transfer Optimization]
**Learning:** When moving multiple JAX arrays from device to host, it is significantly faster to pass a dictionary of lists of arrays to `jax.device_get()` and concatenate them on the CPU using `numpy.concatenate`. Using `jnp.concatenate` on the device prior to transfer introduces unnecessary JAX tracing/JIT overhead for variable-sized array lists and increases peak device memory usage.
**Action:** Use `np.concatenate` after `jax.device_get` for batching arrays instead of `jnp.concatenate` before transfer.
