"""pymcwasm.fit's means against PyMC's own ADVI and nutpie's posterior, on the seven models.

    OUT=advi-pyodide.json node examples/pyodide/advi-check.mjs    # the pymcwasm fits, in Chromium
    uv run --python 3.13 --no-project --with pymc --with scipy python scripts/advi_compare.py advi-pyodide.json

Distances are in the reference posterior's sd, in the unconstrained space, worst over each
model's scalars. PyMC's ADVI is run with three seeds and the same optimizer (Adam at 0.01), so
its own seed-to-seed spread sits beside.
"""

import json
import os
import sys

import numpy as np
import pymc as pm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pymcwasm import param_names  # noqa: E402
from pymcwasm.lowering import MODELS  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def pymc_means(model, n, seed):
    """PyMC's mean-field ADVI mean, flattened in the order param_names uses."""
    # Adam at pymcwasm's rate: PyMC's default adagrad_window at 1e-3 is still moving at 10k.
    with model:
        approx = pm.fit(n=n, method="advi", random_seed=seed, progressbar=False,
                        obj_optimizer=pm.adam(learning_rate=0.01))
    # ordering places each value variable (sigma_log__, not sigma) in the flat mean;
    # mean_data would be the direct way, but it fails to align variables of different shapes.
    flat = approx.mean.eval()
    return np.concatenate([np.ravel(flat[approx.ordering[v.name][1]]) for v in model.value_vars])


if __name__ == "__main__":
    rows = {r["model"]: r for r in json.load(open(sys.argv[1]))}
    n = int(os.environ.get("N_ITERS", 10000))
    for name, build in MODELS.items():
        if name not in rows:
            continue
        model = build()
        names = param_names(model)
        ref = json.load(open(os.path.join(ROOT, "artifacts", name, "reference.json")))
        ref_mean = np.array([ref[k]["mean"] for k in names])
        ref_sd = np.array([ref[k]["sd"] for k in names])
        ours = np.asarray(rows[name]["mean"])
        assert rows[name]["names"] == names, name
        theirs = [pymc_means(model, n, seed) for seed in (1, 2, 3)]
        worst = lambda a, b: float(np.max(np.abs(a - b) / ref_sd))
        spread = max(worst(a, b) for i, a in enumerate(theirs) for b in theirs[i + 1:])
        print(f"{name:20} pymcwasm vs PyMC ADVI {min(worst(ours, t) for t in theirs):5.2f} sd"
              f" (PyMC seed spread {spread:5.2f})  pymcwasm vs nutpie {worst(ours, ref_mean):5.2f}"
              f"  PyMC ADVI vs nutpie {min(worst(t, ref_mean) for t in theirs):5.2f}")
