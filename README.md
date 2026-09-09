# A PyMC model, compiled ahead of time, sampled in a browser

Six PyMC models are compiled into self-contained wasm modules and sampled in the
page with nuts-rs. No server does the sampling, and no Python runs in the
browser.

```
npm install
npx playwright install chromium firefox webkit   # only for `npm test`
npm start        # the page, at http://127.0.0.1:8140/
npm test         # the same thing in three engines, checked against nutpie
```

Nothing is built. `artifacts/` holds the modules; the page loads them and the
published `stanwasm` package from npm, and that is the whole runtime.

## What it does and does not show

Each model's posterior is compared against `nutpie.compile_pymc_model` on the
same model, in the unconstrained space so every parameter is checked rather than
only the untransformed ones. As of this writing the furthest any parameter sits
from nutpie's mean is 0.25 sd, which is Monte Carlo noise between two samplers
that took different trajectories — the gradients themselves agree to 1e-15.

It is not a speed claim. A wasm module driven from CPython has no reason to beat
numba, and this does not try to. What it shows is that the sampling can happen
somewhere Python cannot go.

## How it works

`build/lower_pytensor.py` walks the PyTensor graph of `model.logp()` and writes
it as instructions for the autodiff tape in
[stanwasm](https://github.com/habakan/stanwasm), whose emitter turns a tape into
a standalone wasm module. The page hands that module to `AotSampler`, which is
nuts-rs with no Stan front end behind it.

PyMC does not need the autodiff — PyTensor differentiates its own graph. What
gets used is the emitter and the sampler.

## What it cannot do

- **The data is compiled in.** A module is specific to one model *and one
  dataset*; changing the data means compiling again. That bounds this to fixed
  data and a posterior worth exploring, not to an interactive fit.
- **`Cholesky` lowers but gives the wrong gradient**, so anything with an
  `LKJCholeskyCov` or a GP is out. The decomposition is exact on its own; the
  fault is in the plumbing around the transform.
- **`Scan` is not lowered**, so state-space models are out. It did not appear in
  any of the nine models surveyed, but it will.
- **A `Switch` on a parameter is refused** rather than resolved while tracing,
  which rules out truncated and censored likelihoods.
- **The starting point has to be searched for.** nuts-rs refuses a start whose
  gradient has a zero component, and PyMC's `initial_point()` is zeros — at
  which a centred hierarchical model has an exactly zero gradient in its
  population mean.

## Models

`linear_regression`, `logistic`, `eight_schools`, `varying_intercepts`,
`matrix_regression`, `student_t`. Their definitions are in
`build/lower_pytensor.py`.

## Status

A spike. It exists to find out whether the thing is possible before asking
anyone whether it is wanted.
