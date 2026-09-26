"""Compile a PyMC model into the files a page would serve, and a reference to check them against.

Maintainer notes, not published. See PLAN.md.

Writes into <out>/:
    model.wasm      the emitted module
    meta.json       what `new AotSampler(...)` takes, plus the point to start from
    reference.json  nutpie's posterior mean and sd per parameter, for comparison

    npm install && uv run --with pymc --with scipy --with nutpie \\
        python build/build_artifact.py <model> artifacts/<model>
"""

import json
import os
import sys

import numpy as np
import nutpie
import pymc as pm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pymcwasm.build import build as build_module
from pymcwasm.lowering import MODELS


def reference_posterior(model, names, draws=20000, chains=4, seed=7):
    """nutpie's posterior, pushed back into the space the sampler works in.

    The draws come out constrained and the module's do not, so comparing them
    directly would leave every transformed parameter unchecked.

    Long runs, and `mcse` beside each mean: at 1,000 draws eight_schools'
    reference carried an error of its own comparable to the gap a page is
    checked against, so a seed put the stored `mu` a third of an sd off its
    own long-run value. `versions` records what produced the file.
    """
    import pytensor

    trace = nutpie.sample(
        nutpie.compile_pymc_model(model), draws=draws, chains=chains, seed=seed,
        progress_bar=False,
    )

    columns = {}
    for rv in model.free_RVs:
        value = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms.get(rv)
        a = trace.posterior[rv.name].values
        flat = a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])
        if transform is None:
            unconstrained = flat
        else:
            sym = rv.type()
            fwd = pytensor.function(
                [sym], transform.forward(sym, *rv.owner.inputs),
                on_unused_input="ignore",
            )
            unconstrained = np.stack([np.asarray(fwd(d), dtype=float) for d in flat])
        columns[value.name] = unconstrained.reshape(a.shape[0], a.shape[1], -1)

    # Keyed by the same names the artifact carries, so the two line up by name.
    import arviz as az

    out, at = {}, 0
    for value_name, col in columns.items():
        for j in range(col.shape[2]):
            draws_ij = col[:, :, j]
            out[names[at]] = {
                "mean": float(draws_ij.mean()),
                "sd": float(draws_ij.std()),
                "mcse": float(np.asarray(az.mcse(draws_ij)).ravel()[0]),
            }
            at += 1
    assert at == len(names), (at, len(names))
    return out


def build(name, out_dir):
    model = MODELS[name]()
    meta = build_module(model, out_dir)
    meta["model"] = name
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    with open(os.path.join(out_dir, "reference.json"), "w") as f:
        import nutpie as _nutpie
        import pytensor as _pytensor

        ref = reference_posterior(model, meta["paramNames"])
        ref["versions"] = {
            "nutpie": _nutpie.__version__,
            "pymc": pm.__version__,
            "pytensor": _pytensor.__version__,
        }
        json.dump(ref, f)

    size = os.path.getsize(os.path.join(out_dir, "model.wasm"))
    print(f"{name}: {meta['nParams']} params, {size} bytes of wasm, "
          f"{len(meta['scratchInit'])} scratch slots")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: build_artifact.py <{'|'.join(MODELS)}> <out dir>")
        raise SystemExit(2)
    build(sys.argv[1], os.path.abspath(sys.argv[2]))
