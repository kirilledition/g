## 2024-04-17 - Avoid jnp.concatenate before device_get
**Learning:** Concatenating variable-sized array lists on the device using `jnp.concatenate` prior to `jax.device_get()` transfer introduces unnecessary JAX tracing/JIT overhead and increases peak device memory usage.
**Action:** Pass the list (or dictionary of lists) of arrays directly to `jax.device_get()` and concatenate them on the CPU using `numpy.concatenate` instead.
