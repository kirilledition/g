## 2026-04-13 - [JAX Device Transfer Optimization]
**Learning:** Moving a list of JAX arrays from device to host using `jax.device_get()` and then concatenating them on CPU (`numpy.concatenate`) is significantly faster than concatenating them on device (`jnp.concatenate`) prior to transfer, as it reduces JAX tracing/JIT overhead and peak device memory usage for variable-sized array lists.
**Action:** Replace `jnp.concatenate` before `jax.device_get` with list transfers followed by `np.concatenate` on the host.
