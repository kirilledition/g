## 2024-05-17 - Host-side JAX Concatenation
**Learning:** When moving multiple JAX arrays from device to host, concatenating them on the device using `jnp.concatenate` before transfer introduces unnecessary JAX tracing/JIT overhead for variable-sized array lists, and increases peak device memory usage.
**Action:** Always pass the list (or dictionary of lists) of arrays directly to `jax.device_get()` and then concatenate them on the CPU using `numpy.concatenate` instead.
