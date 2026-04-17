## 2024-04-17 - JAX Device-to-Host Transfer Optimization
**Learning:** When moving multiple JAX arrays from device to host, concatenating variable-sized lists of arrays on the device using `jnp.concatenate` prior to transfer introduces unnecessary JAX tracing/JIT overhead and increases peak device memory usage.
**Action:** Always pass the list (or dictionary of lists) of JAX arrays directly to `jax.device_get()` and then concatenate the returned lists of NumPy arrays on the host CPU using `numpy.concatenate`.
