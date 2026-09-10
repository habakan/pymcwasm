"""nutpie on the same model and machine, for what the density costs natively."""

import os
import time

import nutpie

here = os.path.dirname(os.path.abspath(__file__))
g = {"DATA_PATH": os.path.join(here, "upstream", "mmm_example.csv"), "SEASONALITY": True}
exec(open(os.path.join(here, "upstream", "model.py")).read(), g)
compiled = nutpie.compile_pymc_model(g["model"])

for seed in (42, 142, 242, 342, 442):
    t = time.perf_counter()
    tr = nutpie.sample(compiled, draws=500, tune=750, chains=2, cores=1, seed=seed,
                       target_accept=0.9, progress_bar=False)
    s = time.perf_counter() - t
    steps = int(tr.sample_stats.n_steps.sum() + tr.warmup_sample_stats.n_steps.sum())
    print(f"seed {seed}: {s:.3f} s, {steps} steps, {s / steps * 1e6:.1f} us/step, "
          f"{int(tr.sample_stats.diverging.sum())} divergences")
