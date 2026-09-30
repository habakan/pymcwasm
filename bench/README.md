# Benchmark

Nine posteriordb posteriors, from 2 to 3,075 parameters, each under five samplers:
PyMC's NUTS on PyTensor's Python linker, on `mode="WASM"`, and `pymcwasm.sample` under
Pyodide in Chromium; PyMC's NUTS and nutpie under CPython. The results are in
[`RESULTS.md`](RESULTS.md).

Every sampler draws 2 chains of 500 warmup and 500 draws, one chain after the other, and
is scored by the smallest bulk ESS over the model's free variables per second of
sampling. Compiling is timed apart, since it is paid once per model. The last column
is the worst distance of a sampler's posterior means from nutpie's, in nutpie's sd, so
that a fast row can be seen to have drawn the same posterior.

The Python linker is only run when 40 iterations of one chain project the whole run
under `BUDGET_S` (300 s); otherwise the table gives the projection. PyMC 6's `pm.sample`
picks nutpie when it is installed, so its own NUTS is asked for by name.

```
POSTERIORDB=/path/to/posteriordb python bench/prepare.py
uv run --python 3.13 --no-project --with pymc --with nutpie --with arviz python bench/native.py
STANWASM=../tapewasm node bench/browser.mjs    # a tapewasm checkout; radon and kidiq need 0.3.4
python bench/table.py
```

One machine, one run each: the numbers show the shape, not a ranking to three digits.

## nuts-rs-wasm's demo MMM

`bench/mmm.py` and `bench/mmm.mjs` sample the PyMC-Marketing MMM from
[nuts-rs-wasm](https://github.com/pymc-labs/nuts-rs-wasm)'s `examples/mmm/`, unchanged, with
nutpie under CPython and with tapewasm in V8 (Node), 2 chains of 750 warmup and 500 draws at
target_accept 0.9. The results are in [`RESULTS-mmm.md`](RESULTS-mmm.md).

```
uv run --python 3.12 --no-project --with pymc-marketing --with nutpie --with -e . \
  python bench/mmm.py native /path/to/nuts-rs-wasm
node bench/mmm.mjs
python bench/mmm.py table
```

Two settings decide most of the gap. tapewasm leaves nuts-rs's gradient-based metric
estimate off by default, to match the reference posteriors elsewhere in this repository; on
this model that halves the step size and doubles the gradient evaluations, and turning it on
(`setGradBasedEstimate(true)`) brings both to nutpie's. And `reroll "auto"` suits
SpiderMonkey and JavaScriptCore: V8 runs this model's straight-line module (`"never"`,
309 KB, 134 KB gzipped, against 14 KB) about twice as fast per evaluation.
