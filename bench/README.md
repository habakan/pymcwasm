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
uv run --python 3.12 --no-project --with pymc-marketing --with nutpie --with-editable . \
  python bench/mmm.py native /path/to/nuts-rs-wasm
node bench/mmm.mjs
node bench/mmm-browser.mjs    # nuts-rs-wasm's hosted demo and this repository's two pages
uv run --python 3.12 --no-project --with pymc-marketing --with nutpie --with-editable . \
  python bench/mmm.py table
```

What decides the gap is the time per evaluation. `reroll "auto"` suits SpiderMonkey and
JavaScriptCore, and V8 runs this model's straight-line module (`"never"`, 309 KB, 134 KB
gzipped, against 14 KB) about twice as fast per evaluation, which brings it within about
1.4x of nutpie's.
The gradient-based metric estimate, which tapewasm leaves off by default and nutpie turns
on, moves the step size from 0.083 to 0.224 and halves the evaluations, but it halves the
ESS as well, so ESS per second barely moves; nutpie's own `draw_diag` row shows the same.

In a browser, `bench/mmm-browser.mjs` puts nuts-rs-wasm's hosted demo (Numba, compiled in the
page) beside `bench/mmm-inpage.html` (PyMC-Marketing and PyMC under Pyodide, lowered and
compiled in the page) and `bench/mmm-precompiled.html` (the module `bench/mmm.py native`
built). PyMC-Marketing's `preliz` imports numba, which Pyodide does not have; the model never
calls what it imports, so the page stands a module in for it.
