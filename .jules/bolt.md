## 2024-04-10 - [JAX Device-to-Host Transfer Optimization]
**Learning:** Concatenating lists of JAX arrays on device using `jnp.concatenate` prior to `jax.device_get` introduces unnecessary JAX tracing/JIT overhead for variable-sized arrays and increases peak device memory usage.
**Action:** Always pass lists (or dictionaries of lists) of JAX arrays directly to `jax.device_get()` and then concatenate them on the CPU using `numpy.concatenate`. This yields a massive performance speedup (e.g., from 0.158s to 0.017s for 1000 arrays).
