# PyMC in a browser, with no Python in the browser

Seven PyMC models, each compiled ahead of time into a self-contained wasm module
and sampled in the page by nuts-rs. Nothing Python-shaped is shipped: no
Pyodide, no Xeus, no Numba, no runtime compilation. A page loads a module, a
sampler, and starts drawing.

```
npm install
npm start        # http://127.0.0.1:8140/
```

That is the whole thing to look at — the page samples all seven and prints each
posterior beside what nutpie got for the same model.

```
npx playwright install chromium firefox webkit
npm test         # the same, in three engines, as a pass/fail
```

## The tradeoff

Everything is decided before the page loads. The log density *and its data* are
compiled into the module, so:

- there is no Python and no compiler in the browser, and nothing to wait for.
  The modules here are 4.7-34 KB; the sampler beside them is the published
  `stanwasm` package, 735 KB, of which about a fifth is what this uses — the
  rest is a Stan front end the page never reaches, and building the crate
  without it gives 164 KB. That build is not published, so the demo loads the
  whole thing;
- and the module answers for exactly one model and one dataset. Changing either
  means compiling again, elsewhere.

That is the opposite end of the trade from running PyMC itself in the browser,
which keeps every model and every dataset available at the cost of carrying a
Python runtime and compiling on arrival. Neither is better; they are for
different pages. This repository exists because the first end had not been
tried.

Prior art on the other end, from PyMC's own developers:
[nutpie#345](https://github.com/pymc-devs/nutpie/pull/345),
[Running PyMC in the browser with PyScript](https://discourse.pymc.io/t/running-pymc-in-the-browser-with-pyscript/9432).

## Is it right?

Each model is checked twice.

**The gradient**, against `model.compile_dlogp()`, at a point other than the one
it was lowered at — so a subgraph wrongly frozen into a constant shows up
instead of cancelling. The worst of the seven is 2.5e-15 relative.

**The posterior**, against `nutpie.compile_pymc_model` on the same model, in the
unconstrained space so transformed parameters are checked too rather than
skipped. The furthest any parameter's mean sits from nutpie's is 0.25 sd, which
is Monte Carlo noise between two samplers that took different trajectories.

Not a speed claim. A wasm module has no business beating numba, and none of
these numbers are a comparison of anything but correctness.

## How it works

`build/lower_pytensor.py` walks the PyTensor graph of `model.logp()` and writes
it as instructions for the autodiff tape in
[stanwasm](https://github.com/habakan/stanwasm) — a Stan implementation, whose
emitter turns a recorded tape into a standalone wasm module and whose
`AotSampler` runs nuts-rs against one. Neither half knows any Stan is involved;
the tape and the module ABI are all they share.

PyMC does not need the autodiff. PyTensor differentiates its own graph, and what
gets used here is the emitter and the sampler.

## What it cannot do

- **`Scan` is not lowered**, so state-space models are out. It did not appear in
  any of the nine logp graphs surveyed, but it will.
- **A `Switch` on a parameter is refused** rather than resolved while tracing,
  which rules out truncated and censored likelihoods.
- **A Gaussian process is untried.** Its `Cholesky` is of a covariance built
  from the parameters, so it lowers to a cubic number of tape nodes in the
  number of points. Nothing is known to be wrong with it.
- **The starting point has to be searched for.** nuts-rs refuses a start whose
  gradient has a zero component, and PyMC's `initial_point()` is zeros — at
  which a centred hierarchical model has an exactly zero gradient in its
  population mean, and so does a logit regression on balanced data.

## Models

`linear_regression`, `logistic`, `eight_schools` (centred),
`varying_intercepts`, `matrix_regression`, `student_t`, `lkj_mvnormal`. Defined
in `build/lower_pytensor.py`; `build/README.md` says how to regenerate an
artifact.

## Status

A spike, three days old, by someone who does not work on PyMC. It answers
whether the thing runs, not whether it should exist. Take it or leave it.

Apache-2.0.
