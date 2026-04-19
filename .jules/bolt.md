## 2026-04-19 - [JAX Device Array List Concatenation]
**Learning:** Using `jnp.concatenate` on a variable-sized list of JAX arrays (e.g., accumulated results chunks) prior to `jax.device_get` causes massive tracing and JIT overhead (nearly a 10x slowdown for this specific workload). It also spikes peak memory usage on the device.
**Action:** Always fetch the list/dictionary of arrays directly via `jax.device_get` to move them to the host first, and then concatenate them on the CPU using `numpy.concatenate`.
