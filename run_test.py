import jax.numpy as jnp

batch_heuristic_mask = jnp.array([False, False])
batch_standard_coefficients = jnp.zeros((64, 3))
print("batch_heuristic_mask.shape:", batch_heuristic_mask.shape)
print("batch_standard_coefficients.shape:", batch_standard_coefficients.shape)
device_use_heuristic_mask = batch_heuristic_mask[:, None]
print("device_use_heuristic_mask.shape:", device_use_heuristic_mask.shape)
