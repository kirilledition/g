## 2024-05-24 - explicit broadcast over einsum for matrix products
**Learning:** For batched element-wise multiplication of a matrix with itself, explicit broadcasting (`A[:, :, None] * A[:, None, :]`) yields faster execution times (both eager overhead and execution time after JIT) than using `jnp.einsum("np,nq->npq", A, A)`.
**Action:** Use explicit broadcasting for computing pairwise component matrices when possible, while keeping string-parsed `jnp.einsum` strictly for dimension-reducing operations.

## 2024-05-24 - Python list comprehension over DataFrame columns
**Learning:** `name not in df.columns` does a linear search when `df.columns` is a list, causing O(N^2) complexity in comprehensions. Converting `df.columns` to a set first drops this to O(N).
**Action:** Always convert list-like objects to a `set` before repeatedly checking membership inside a loop.
## 2024-05-24 - einsum vs sum(A*B)
**Learning:** `jnp.einsum("ij,ij->j", A, B)` is used as an optimization in the codebase, but surprisingly, benchmarking shows `jnp.sum(A * B, axis=0)` is sometimes faster depending on shapes and layout. However, the codebase explicitly tells us: "In this codebase, `jnp.einsum("ij,ij->j", A, B)` is used as a memory and performance optimization over `jnp.sum(A * B, axis=0)`. The `einsum` operation avoids the materialization of the intermediate full-sized N x M product matrix, yielding roughly a ~25% speedup. Do not replace these existing `einsum` reduction calls with `jnp.sum`." I must respect the existing design choice.
**Action:** Do not replace `einsum` with `sum(A * B)`.
