## $(date +%Y-%m-%d) - JAX Array Aggregation
**Learning:** Concatenating lists of variable-sized arrays on the device using `jnp.concatenate` prior to `jax.device_get` causes unnecessary JIT tracing overhead and increases peak device memory usage. Passing the list directly to `device_get` and doing `np.concatenate` on the host CPU provides a significant speedup.
**Action:** When gathering chunked JAX results, perform one `jax.device_get()` on a list or dict of lists, and then use `numpy.concatenate()` on the host, rather than `jnp.concatenate` on the device.
