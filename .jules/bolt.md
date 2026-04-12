## 2024-05-19 - [Host Transfer Optimization]
**Learning:** The application was concatenating device arrays in JAX (`jnp.concatenate`) before transferring them to the host (`jax.device_get()`). This introduces JIT and tracing overhead for arrays whose sizes may not be fixed due to variable chunk sizes.
**Action:** Transfer lists of JAX arrays directly to the host using `jax.device_get()`, and then use `numpy.concatenate` on the host side to concatenate them. This reduces JAX compilation/execution overhead.
