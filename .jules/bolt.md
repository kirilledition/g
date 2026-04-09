## 2024-04-09 - JAX device transfer and concatenation overhead
**Learning:** Concatenating multiple variable-sized JAX arrays on device using `jnp.concatenate` prior to moving them to the host via `jax.device_get` causes unnecessary JAX tracing/JIT overhead and increases peak device memory.
**Action:** Instead of concatenating on device first, pass the list of JAX arrays directly to `jax.device_get` to transfer them efficiently, and perform the concatenation on CPU using `numpy.concatenate`.
