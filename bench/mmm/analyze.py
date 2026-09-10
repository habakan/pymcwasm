"""Medians over the five seeds of every `raw/*.json`, written to `results.json`.

ESS and R-hat are computed on the unconstrained draws. Both are rank-based, so a
monotone transform to the constrained space leaves them unchanged.
"""

import glob
import json
import os

import numpy as np
from arviz_stats.base import array_stats

here = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(os.path.join(here, "artifact", "reference.json")))


def one(run):
    c = run["cold"]
    names = c["names"]
    x = np.array([np.reshape(ch, (-1, len(names))) for ch in c["draws"]])
    flat = x.reshape(-1, len(names))
    ess = array_stats.ess(x, chain_axis=0, draw_axis=1, method="bulk")
    tail = array_stats.ess(x, chain_axis=0, draw_axis=1, method="tail", prob=(0.05, 0.95))
    rhat = array_stats.rhat(x, chain_axis=0, draw_axis=1)
    gap = max(abs(flat[:, k].mean() - ref[n]["mean"]) / ref[n]["sd"] for k, n in enumerate(names))
    alpha = [k for k, n in enumerate(names) if n.startswith("adstock_alpha")]
    return {
        "sampling_s": c["sampleMs"] / 1e3,
        "reused_sampling_s": run["reusedMs"] / 1e3,
        "navigation_to_posterior_s": c["sinceNavMs"] / 1e3,
        # `sampleWithStats` evaluates the density once more per draw for its lp column.
        "module_calls": run["calls"],
        "us_per_call": c["sampleMs"] * 1e3 / run["calls"],
        "leapfrog_steps": c["leapfrog"],
        "divergences": c["divergences"],
        "transfer_kb": c["transfer"] / 1024,
        "min_ess_bulk": float(ess.min()),
        "min_ess_tail": float(tail.min()),
        "max_rhat": float(rhat.max()),
        "worst_gap_to_nutpie_sd": float(gap),
        "adstock_alpha_mean": (1 / (1 + np.exp(-flat[:, alpha]))).mean(0).tolist(),
    }


summary = {}
for path in sorted(glob.glob(os.path.join(here, "raw", "*.json"))):
    d = json.load(open(path))
    rows = [one(r) for r in d["runs"]]
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != "adstock_alpha_mean"}
    med["adstock_alpha_mean"] = np.median([r["adstock_alpha_mean"] for r in rows], axis=0).round(3).tolist()
    summary[f"{d['engine']}/{d['art']}"] = {"settings": d["settings"], "median": med}
    print(f"{d['engine']:9} {d['art']:16} " + "  ".join(
        f"{k} {v:.3g}" for k, v in med.items() if k in
        ("sampling_s", "us_per_call", "leapfrog_steps", "divergences", "min_ess_bulk", "max_rhat")))
json.dump(summary, open(os.path.join(here, "results.json"), "w"), indent=1)
