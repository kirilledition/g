
## 2026-03-27 - [Orthogonal Projection Simplification in Linear Kernel]
**Learning:** Instantiating the $N \times M$ `genotype_residual` matrix dominates the memory and compute overhead of the linear JAX kernel on GPU. Because `phenotype_residual` is constructed to be orthogonal to the `covariate_matrix`, we can leverage geometric properties of projections to bypass this entirely. For covariance, `genotype_residual.T @ phenotype_residual` mathematically reduces to just `genotype_matrix.T @ phenotype_residual`.
**Action:** Always inspect linear algebra kernels for orthogonal properties. Substituting huge matrix subtractions and point-wise multiplications with their associative, orthogonal projection equivalents avoids instantiating large temporary tensors in JAX.
