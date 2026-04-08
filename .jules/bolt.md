## 2026-04-08 - Fast host transfer via JAX dict and numpy concat

**Learning:** When moving multiple JAX arrays from device to host, it is significantly faster to pass the list (or dictionary of lists) of arrays directly to `jax.device_get()` and then concatenate them on the CPU using `numpy.concatenate`. Avoid concatenating them on the device using `jnp.concatenate` prior to transfer, as this introduces unnecessary JAX tracing/JIT overhead for variable-sized array lists and increases peak device memory usage.
**Action:** Instead of `host_vals = jax.device_get({'a': jnp.concatenate([c.a for c in chunks])})`, do `host_vals = jax.device_get({'a': [c.a for c in chunks]}); final_a = np.concatenate(host_vals['a'])`.
