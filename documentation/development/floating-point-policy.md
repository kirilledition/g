# Floating-Point Policy

| Status | Applies to | Owner |
| --- | --- | --- |
| Active pre-release contract | Rust host arrays and JAX association kernels | Compute maintainers |

Floating-point width is selected by ownership boundary and numerical role. A
single global dtype would either double host bandwidth or weaken Firth solver
stability.

## Contract

1. Phenotypes, covariates, LOCO predictions, dosages, and genotype summary
   buffers are stored as `f32` on the Rust host.
2. JAX score-test arithmetic and public result statistics use `float32`.
3. Null Firth, approximate-Firth outer components, convergence checks,
   likelihood, information, corrected statistics, and fallback Newton-Raphson
   use JAX `float64`.
4. The approximate-Firth inner pseudo-logistic proposal evaluates sigmoid,
   score products, and information products in `float32`, widens products
   before `float64` reductions, and only proposes coefficients for subsequent
   `float64` validation.
5. Corrected Firth values are narrowed once when merged into float32 score results.
6. `firth_dtype` does not exist. Firth width is an algorithm invariant, not a
   user option.
7. Epsilon is derived from the active array dtype with `jnp.finfo(dtype).eps`.
   A float64 epsilon must never be cast down and used as a float32 tolerance.

## Configuration Values

Rust configuration thresholds are finite validated `f64` newtypes:

- positive tolerances and variance floors are greater than zero;
- probabilities are in `(0, 1)` and probability floors in `(0, 0.5)`;
- step-halving scales are in `(0, 1)`;
- sparse carrier dosage thresholds are in `(0, 2]`.

The PyO3 boundary extracts the validated scalar with `.get()` and passes a
Python `float` into the typed JAX configuration. Kernels cast the scalar to the
array dtype at the point of use.

## Review Checklist

- New host arrays remain `f32` unless a measured host-side algorithm requires
  wider storage.
- New Firth state and objective operands are explicitly converted to
  `jnp.float64`; the documented inner proposal is the only elementwise
  `float32` exception.
- New convergence constants derive from the operand dtype.
- No configuration or manifest field reintroduces `firth_dtype`.
- Output narrowing occurs once during materialization. Inner-proposal products
  widen before every `float64` reduction.
