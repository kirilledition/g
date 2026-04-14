## 2026-04-14 - Optimize JAX array transfer memory and execution

**Learning:** When moving multiple JAX arrays from device to host, it is significantly faster to pass the list (or dictionary of lists) of arrays directly to `jax.device_get()` and concatenate them on the CPU using `numpy.concatenate`. Concatenating them on the device using `jnp.concatenate` prior to transfer introduces unnecessary JAX tracing/JIT overhead for variable-sized array lists and increases peak device memory usage.
**Action:** Replace `jnp.concatenate([arr for arr in arrays])` followed by `jax.device_get()` with a simple list comprehension `jax.device_get([arr for arr in arrays])` and then run `np.concatenate` on the host result.
