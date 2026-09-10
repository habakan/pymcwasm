# The nuts-rs-wasm demo MMM, precompiled

The marketing-mix model from
[pymc-labs/nuts-rs-wasm](https://github.com/pymc-labs/nuts-rs-wasm) (179 weeks,
15 parameters, geometric adstock with 8 lags, logistic saturation, controls,
yearly seasonality) is lowered here and sampled in a browser. The schedule and
settings match that repository's
[8 September benchmark](https://github.com/pymc-labs/nuts-rs-wasm/tree/main/benchmarks/2026-09-08),
so the two can be read side by side. This came out of
[nuts-rs-wasm#6](https://github.com/pymc-labs/nuts-rs-wasm/issues/6).

Apple M3 · two sequential chains · 750 warmup + 500 draws · seeds 42, 142, 242,
342, 442 · target_accept 0.9 · gradient-based metric estimate · start jittered
by U(-1, 1). Every figure is a median over the five seeds, each seed in a fresh
browser context.

## Results

Chromium, the module as tapewasm emits it by default (straight-line):

| | |
| --- | ---: |
| transfer, gzip | 298 KB in 7 requests |
| compile in the page | none — lowering 0.5 s + emit 0.4 s, offline |
| sampling | 1.41 s (1.25 s with the module already loaded) |
| navigation to posterior | 1.45 s |
| leapfrog steps | 75,022 |
| module calls | 77,559 (steps, starts, and one lp per draw for the statistics) |
| µs per call | 18.0 |
| divergences | 0 |
| min bulk ESS / min tail ESS / max R-hat | 188 / 140 / 1.010 |
| furthest parameter mean from nutpie | 0.08 sd |
| adstock_alpha posterior mean | [0.397, 0.193] |

nutpie on the same machine and settings samples in 0.72 s at 9.7 µs per step.

By engine, and with the statements re-rolled into loops (`REROLL=always`):

| sampling (µs per call) | Chromium | Firefox | WebKit | module, gzip |
| --- | ---: | ---: | ---: | ---: |
| straight-line | **1.41 s** (18.0) | 3.36 s (42.9) | 4.34 s (52.7) | 139 KB |
| re-rolled | 2.08 s (27.1) | **2.55 s** (32.9) | **2.52 s** (30.6) | 3.3 KB |

The draws are the same either way; only how fast the engine runs the module
changes.

## What this does not measure

- **Only the precompiled path.** The model and data are compiled in, and the
  page carries no Python. Editing the model in the page would need Pyodide and
  PyMC-Marketing, which has not been tried.
- **No expansion.** The timing ends at the unconstrained draws. There are no
  deterministics such as channel contributions on this path, and no Arrow
  output; nuts-rs-wasm's sampling time includes both.
- **One model.** The lowering is written by hand, and this model needed three
  fixes to it before it lowered correctly.

## Reproduce

Needs a tapewasm checkout at `133f84b` or later (`setTargetAccept`,
`sampleWithStats` and `REROLL` are not in a release yet), built with
`make wasm`.

```
cd bench/mmm
mkdir -p upstream && for f in model.py mmm_example.csv; do
  curl -fsSL -o upstream/$f https://raw.githubusercontent.com/pymc-labs/nuts-rs-wasm/1124d88fb246411572fb5d5a256ac42b9dc2d2f6/examples/mmm/$f
done
cd ../..

export TAPEWASM=/path/to/tapewasm
PY="uv run --python 3.12 --with pymc==6.2.0 --with pytensor==3.2.4 \
  --with pymc-marketing==1.1.0 --with nutpie --with scipy python"
$PY bench/mmm/build.py bench/mmm/artifact
REROLL=always $PY bench/mmm/build.py bench/mmm/artifact-always

for a in artifact artifact-always; do for e in chromium firefox webkit; do
  VENDOR=$TAPEWASM/ts ENGINE=$e ART=$a node bench/mmm/bench.mjs
done; done
$PY bench/mmm/analyze.py        # writes results.json
$PY bench/mmm/native_nutpie.py
```

`results.json` is the run the tables above come from.
