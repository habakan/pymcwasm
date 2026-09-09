# pymcwasm

Sample a PyMC model in a browser. The log density is compiled to a WebAssembly
module and drawn from by [nuts-rs](https://github.com/pymc-devs/nuts-rs); no
server does the sampling.

There are two ways in, and they are different trades rather than one being the
real one.

## Write the model in the page

PyMC runs under Pyodide, builds the graph, and everything after that happens in
the page too: the log density is lowered to a tape, stanwasm's emitter turns
that into a wasm module, and nuts-rs samples it. The model and its data are
whatever you typed.

```python
import pymcwasm

fit = await pymcwasm.sample(model, draws=1000, warmup=1000, seed=42)
fit["beta"]      # one parameter's draws
fit.summary()    # mean and sd per parameter
```

```
npm install
npm start        # then open /examples/pyodide/
```

Costs a Pyodide runtime and a PyMC install on first load — tens of megabytes,
tens of seconds. Compiling a small model then takes about a second, and drawing
1000 times takes about ten milliseconds.

## Compile it beforehand and ship the module

Nothing Python-shaped reaches the browser. A build step turns a model into a
5–35 KB wasm module, and the page loads that and a sampler.

```
npm start        # then open /
npm test         # the same seven models in three engines, checked against nutpie
```

What a page does with one, in full:

```js
import init, { AotSampler, setAotExports, sharedMemory } from "stanwasm";

await init();
const meta = await (await fetch("/artifacts/eight_schools/meta.json")).json();
const bytes = await (await fetch("/artifacts/eight_schools/model.wasm")).arrayBuffer();

// The module imports its own arithmetic. `examples/browser/app.js` carries the
// series for lgamma, digamma and Phi; a model that reaches none of them can
// pass anything for those three.
const aot = await WebAssembly.instantiate(bytes, {
  stan: { memory: sharedMemory() },
  Math: { exp: Math.exp, log: Math.log, pow: Math.pow, sin: Math.sin, cos: Math.cos,
          tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
          lgamma, digamma, phi },
});
setAotExports(aot.instance.exports);

const sampler = new AotSampler(
  meta.nParams, new Float64Array(meta.scratchInit), meta.layoutId, meta.paramNames,
);
const flat = sampler.sample(new Float64Array(meta.initialPoint), 1000, 1000, 42n);
```

`flat` is draws-major and `meta.nParams` wide, warmup first. `meta` is what the
build step recorded beside the module; `setAotExports` binds one module per
page, and sampling with the wrong one bound is refused rather than mixed, by the
layout id both sides carry.

The module answers for exactly one model and one dataset — the data is compiled
in — so changing either means building again. For a page whose model was decided
when the page was written, that is the whole cost, and nothing has to load a
Python runtime to read it.

## Is it right?

Every model is checked twice.

**The gradient**, against `model.compile_dlogp()`, at a point other than the one
it was lowered at, so a subgraph wrongly frozen into a constant shows up instead
of cancelling. The worst of the seven is 2.5e-15 relative.

**The posterior**, against `nutpie.compile_pymc_model` on the same model, in the
unconstrained space so transformed parameters are checked rather than skipped.
The furthest any parameter's mean sits from nutpie's is about 0.25 sd, which is
Monte Carlo noise between two samplers that took different trajectories.

Not a speed claim, in either direction.

## How it works

`src/pymcwasm/lowering.py` walks the PyTensor graph of `model.logp()` and writes
it as instructions for the autodiff tape in
[stanwasm](https://github.com/habakan/stanwasm) — a Stan implementation, whose
emitter turns a tape into a standalone wasm module and whose `AotSampler` runs
nuts-rs against one. Neither half of stanwasm knows Stan is not involved here:
the tape and the module ABI are all that is shared.

PyMC does not need the autodiff. PyTensor differentiates its own graph; what
gets used is the emitter and the sampler.

```
src/pymcwasm/      the package a Pyodide page imports, and the lowering
build/             the offline artifact builder (needs a stanwasm checkout)
artifacts/         seven models, compiled
examples/browser/  the precompiled path
examples/pyodide/  the in-page path
```

## What it cannot do

- **`Scan` is not lowered**, so state-space models are out. It did not appear in
  any of the nine logp graphs surveyed, but it will.
- **A `Switch` on a parameter is refused** rather than resolved while tracing,
  which rules out truncated and censored likelihoods.
- **A Gaussian process is untried.** Its `Cholesky` is of a covariance built from
  the parameters, so it lowers to a cubic number of tape nodes in the number of
  points. Nothing is known to be wrong with it.
- **The starting point has to be searched for.** nuts-rs refuses a start whose
  gradient has a zero component, and PyMC's `initial_point()` is zeros — at which
  a centred hierarchical model has an exactly zero gradient in its population
  mean, and so does a logit regression on balanced data. `pymcwasm.sample` looks
  for one; a caller supplying its own has to as well.
- **Continuous parameters only**, and no prior or posterior predictive, and no
  `InferenceData`.

## Status

An experiment. It answers whether the thing runs, not whether it should exist.
Not affiliated with PyMC.

Apache-2.0.
