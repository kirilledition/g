## 2026-04-16 - Host-Side Array Concatenation Optimization
**Learning:** Concatenating lists of variable-sized arrays on the device using `jnp.concatenate` before transferring to the host via `jax.device_get` causes unnecessary JAX tracing/JIT overhead and increases peak device memory.
**Action:** Always transfer the list (or dict of lists) of arrays to the host directly using `jax.device_get`, and then concatenate them on the CPU using `np.concatenate`.
