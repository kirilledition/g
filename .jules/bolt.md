## 2026-04-19 - Host-side Array Concatenation
**Learning:** Concatenating lists of JAX arrays eagerly on the device (`jnp.concatenate`) before transferring them to the host (`jax.device_get`) causes unnecessary JAX tracing/XLA execution overhead and memory allocation spikes.
**Action:** When transferring multiple small/variable-sized arrays from device to host, always pass the list (or dict of lists) directly to `jax.device_get()`, and perform the concatenation on the CPU using `numpy.concatenate`.
