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
