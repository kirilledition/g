# VISION

## 1. The Core Objective (The "What")

We are building a next-generation, hardware-accelerated Genome-Wide Association Study (GWAS) engine.

The current State of the Art (SOTA) tools—like plink2 and regenie—are phenomenal, but they are heavily CPU-bound and written in legacy C++. We are building a tool designed for the modern accelerator era (Multi-GPU, TPUs, Cerebras) capable of chewing through biobank-scale datasets (like the UK Biobank) orders of magnitude faster than current tools.

## 2. The Ultimate End-State (The "Where we are headed")

When this project is "done," it will exist in two forms, powered by the exact same core engine:

- **For Data Scientists (The Python Package):** A highly polished Python library (`pip install g`) where users can write simple scripts, interact with their data in Polars/Arrow, and let the library handle the complex GPU memory management and math under the hood.
- **For Pipelines (CLI):** CLI tool that can be dropped into bash scripts or cloud pipelines

## 3. The Strategy: The Strangler Fig Pattern (The "How")

We are not going to spend 2 years building a massive C++/Rust/CUDA engine in stealth, only to release it and find out our API is terrible or our math is slightly off.

We will build pragmatically:

- **Start High-Level:** We will build the MVP entirely in Python, using JAX (for XLA-compiled math) and Polars (for I/O). It will be correct, it will have a great API, but it might not be the fastest tool in the world yet.
- **Strangulate the Bottlenecks:** Once milestone-level parity is achieved (starting with plink2 and later regenie), we will profile it. We will ruthlessly rip out the slowest Python components module-by-module and replace them with zero-copy Rust FFI, custom Triton kernels, or raw CUDA.
- **Continuous Delivery:** At every step, we have a working, testable product that is constantly getting faster.

## 4. The "Boss Fights" (Major Technical Milestones)

- **Boss (regenie parity):** Implement Mixed Linear Models (MLMs) for massive biobanks. This requires Ridge Regression, Leave-One-Chromosome-Out (LOCO) predictions, and Firth penalized logistic regression for rare variants—all batched across massive GPU matrix multiplications.

# DREAMS AND IDEAS

Personal kirill dreams backlog for this app

## API translator

I want to have some kind of quick translator layer for famous genomic tools like regenie and plink. like we can do

g api-plink and then plik flags
or
g api-regenie and then regenie flags

also can have python api like

from g import api_plink as g

g(
    and here arguments with the same names as plink
)

also from g import api_hail, api_regenie

## UX documentation

Also i want my app to have great ui and ux, in case of bioinformatics ux is partly documentation. i think neat place for documentation is github wiki

i want to have every flag explained. if flag is actually responsible for some complex math or algorithm - i want it mentioned and linked

Also would be nice to have examples or case studies for what a parameter could mean

When thinking of this i realized that to have this in place, we have to have documentation on computational genetics itself in a repo, like what the fuck is even firth, how our app decides to use it. what are dosages. what are different file formats, what are they. and then i can have parameters documentation linked to that. 

Probably we should write documentation for app and if you need to know algorithm to use it correctly, we will write explanation in wiki

Also want to have something like no stupid question policy. If person asks question on something, it means that we either did not write it in documentation, or were bad at explaining what he needs, or documentation was in non obvious place. we will try to address each question and keep library of those in github issues to make them searchable.

## CI to check reliability

I want to have some kind of ci that will run every month and check for if app still does its functionality, so basically tests, but also some integration tests to run on actual files. This should prevent software from rotting

## Versioning

For software like this there can actually be breaking changes. breaking change can be change in api so it no longer works in a pipeline. it can be change in flag name, or change in output format, or change in output column names. new minor version will probably indicate new features, or something considerably new, need to think about it.

## Negative log p-value

I want app to compute p values in negative log10, i believe that only this makes sense. it will be easier for plotting and will allow to change dtype to bfloat16 for even faster code.

## Parallelization 

Examples and easy code on how to parallelize across aws machines, slurm nodes, or maybe multiple gpus on the same machine.

## 3D Batching

In this repository, there are three different batching ideas that can easily get conflated:

1. **Chunk batching across variants using a 2D genotype matrix**
   - Shape is typically `(sample_count, variant_count)`.
   - One matrix multiply computes statistics for many variants at once.
   - This is already heavily used by the linear kernel and by parts of the logistic kernel.

2. **True batched small-matrix algebra using a leading batch axis**
   - Shape is typically `(variant_count, parameter_count, parameter_count)` or `(variant_count, sample_count, parameter_count)`.
   - Each variant has its own Fisher information matrix, score vector, or masked design matrix.
   - This is the real "3D batching" pattern.

3. **Vectorized repeated solves or updates over many variants**
   - Often expressed through `jax.vmap`, `jnp.einsum`, batched `jnp.linalg.solve`, or explicit leading-batch dimensions.
   - This is the practical form of 3D batching in JAX.

For ordinary least squares, the first pattern often captures most of the available gain because the problem has a closed form and the covariate matrix is shared across variants.

For logistic regression, the second and third patterns matter much more because each IRLS iteration creates variant-specific weights, scores, and information matrices.

## Dedicated precompilation

Reduce or eliminate expensive JAX compilation on rented GPU machines by letting users pre-populate the persistent compilation cache on a cheaper but compatible machine, then transfer that cache to the production environment.
- configure JAX to use the requested cache directory
- run warmup calls for hot jitted kernels with synthetic arrays
- force compilation for all expected shape families
- write metadata describing the cache environment

## Ability to pass custom function

User can define a function and pass it as an object to the library. there will be functions running preprocessing on features, on genotype, generating additional covariates, etc.

## Full verbose output file

Output file will contain pvalues, coefficients and etc on every covariate, not only on genotype