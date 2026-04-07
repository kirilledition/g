## 2025-04-07 - [jax.device_get Optimization]
 **Learning:** Concatenating lists of JAX arrays on the device using `jnp.concatenate` before transferring them with `jax.device_get` is slower and increases peak device memory.
 **Action:** Always pass lists (or dicts of lists) of JAX arrays directly to `jax.device_get` and then concatenate them on the CPU using `np.concatenate`.
