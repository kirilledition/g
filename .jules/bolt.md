
## 2026-04-11 - Fast JAX Device to Host Concatenation
**Learning:** When moving multiple JAX arrays from device to host, concatenating them using `jnp.concatenate` on the device and then retrieving the single array with `jax.device_get()` introduces high JAX tracing/JIT overhead and can be very slow. It is significantly faster to pass the list of arrays directly to `jax.device_get()` and then concatenate them on the CPU using `numpy.concatenate`.
**Action:** When merging chunked JAX results from device arrays to host DataFrame columns, fetch a dict/list of the raw JAX arrays using `jax.device_get` directly, and perform the concatenation using `np.concatenate`.
