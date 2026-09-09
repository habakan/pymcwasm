# What is missing

Not a roadmap. Things that turned out to be needed and are not there.

## Predictive quantities

A fit gives the posterior and stops. Anything downstream of it —
`pm.sample_posterior_predictive`, a linear predictor at new data, a
deterministic recorded with `pm.Deterministic` — has nowhere to run.

**Evaluating a deterministic quantity per draw** is the smaller half and needs
no new machinery: the quantity is a PyTensor graph over the same parameters, so
it lowers to a tape like any other. What is missing is somewhere to put the
answer. A module exports `log_prob_grad` and nothing else, so today the only
way through is to compile the quantity *as* the log density and read the
returned value — which works and is a trick, not an interface. A second entry
point, or an `evaluate(tape, draws)` beside `sample`, is the honest version.

**Drawing from the predictive** is the larger half. The tape has 31 operations
and none of them is random; the emitted module is a pure function. stanwasm
runs Stan's `generated quantities` on its interpreter for the same reason,
never in the emitted module. So either the tape grows a source of randomness
and the emitter learns to thread a seed through it, or the draws come back to
JavaScript and the sampling of `y_rep` happens there, per distribution. The
second is less work and less general.

## Using a posterior as a prior

Feasible now for a normal approximation: the mean and Cholesky factor are
constants, which fold during lowering, and `Cholesky` and `SolveTriangular` are
lowered. Nobody has tried it.

Not feasible for an empirical prior — `pm.Interpolated` and anything else that
looks a value up in a table indexes on a parameter, and the lowering refuses a
parameter-dependent index rather than freezing it at the trace point.

## Elsewhere

- `Scan`, so no state-space models.
- A `Switch` on a parameter, so no truncated or censored likelihoods.
- Discrete parameters.
- A Gaussian process is untried; its `Cholesky` is of a parameter-dependent
  covariance, so the tape grows with the cube of the number of points.
