
## 2024-04-17 - JAX device_get lists before concatenating

**Learning:** When moving multiple JAX arrays from device to host, especially lists of variable-size arrays (like accumulators across chunks), it is significantly faster (around ~30-40% speedup in benchmarks) to pass the list of arrays directly to `jax.device_get()` and concatenate them on the CPU using `numpy.concatenate`. Doing `jnp.concatenate` on the device first introduces unnecessary JAX tracing/JIT overhead and temporary device allocations for these variable-sized lists.

**Action:** Whenever a list of JAX arrays needs to be aggregated and brought to the host, avoid `jnp.concatenate` followed by `jax.device_get`. Instead, use `jax.device_get` directly on the list and then perform `np.concatenate` on the resulting NumPy arrays.
