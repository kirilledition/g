
## 2024-04-06 - [JAX Device to Host Transfer Bottleneck]
**Learning:** Concatenating multiple JAX arrays on the device using `jnp.concatenate` before transferring to the host via `jax.device_get` incurs a massive performance penalty. This overhead originates from eager dispatch and potential JIT recompilation costs since lists of variant chunks vary in size. Furthermore, it redundantly spikes peak device memory by allocating a large contiguous buffer.
**Action:** When gathering chunked results, pass the list (or structured dictionary of lists) of JAX arrays directly to `jax.device_get()` and perform the concatenation entirely on the CPU using `np.concatenate`.
