"""Sample a PyMC model in a browser, without a Python sampler.

`pymcwasm` lowers a PyMC model's log density to the autodiff tape that
[stanwasm](https://github.com/habakan/stanwasm) compiles, hands the tape to
stanwasm's emitter, and draws from the resulting wasm module with nuts-rs.
Inside Pyodide, all of that happens in the page.

    import pymcwasm
    fit = await pymcwasm.sample(model, draws=1000, warmup=1000, seed=42)
    fit["beta"]      # one parameter's draws
    fit.summary()    # mean and sd per parameter

Draws are in the space the sampler works in, so a transformed parameter comes
back under its value-variable name (`sigma_log__`, not `sigma`).

`import pymcwasm` is safe anywhere; `sample` needs the `js`/`pyodide` modules a
Pyodide runtime provides.

Not affiliated with PyMC.
"""

import numpy as np

from . import lowering
from ._bridge import DEFAULT_STANWASM_PATH, compile_and_sample

__all__ = ["Fit", "sample", "starting_point", "tape_for"]


def starting_point(model, seed=0, tries=50):
    """A point nuts-rs will accept.

    It refuses a start whose gradient has a zero component, because the mass
    matrix it adapts is scaled by that gradient. PyMC's `initial_point()` is
    zeros, and at the origin a centred hierarchical model has an exactly zero
    gradient in its population mean — so the default point is refused for shapes
    people actually write.
    """
    dlogp = model.compile_dlogp()
    point = model.initial_point()
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        if np.all(np.abs(np.asarray(dlogp(point), dtype=float)) > 1e-12):
            return point
        point = {
            k: np.asarray(v, dtype=float) + rng.uniform(-2, 2, np.shape(v))
            for k, v in model.initial_point().items()
        }
    raise RuntimeError("no starting point with a non-zero gradient in every component")


def param_names(model):
    """One name per unconstrained scalar, in the order the sampler reads them."""
    point = model.initial_point()
    names = []
    for v in model.value_vars:
        shape = np.shape(np.asarray(point[v.name]))
        if not shape:
            names.append(v.name)
        else:
            names.extend(f"{v.name}[{','.join(map(str, i))}]" for i in np.ndindex(shape))
    return names


def tape_for(model, point=None):
    """The model's log density as a tape, and the point it was traced at."""
    import tempfile
    import os

    point = point or starting_point(model)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.tape")
        lowering.lower(model, path, point, point)
        with open(path) as f:
            return f.read(), point


class Fit:
    """Post-warmup draws, one column per unconstrained scalar."""

    def __init__(self, names, draws, ms, module_bytes):
        self.names = list(names)
        self.draws = np.asarray(draws, dtype=float).reshape(-1, len(self.names))
        self.ms = ms
        self.module_bytes = module_bytes

    def __getitem__(self, name):
        return self.draws[:, self.names.index(name)]

    def __repr__(self):
        return (f"<Fit {self.draws.shape[0]} draws x {len(self.names)} parameters, "
                f"{self.ms:.0f} ms, {self.module_bytes} byte module>")

    def summary(self):
        import pandas as pd

        return pd.DataFrame(
            {"mean": self.draws.mean(axis=0), "sd": self.draws.std(axis=0)},
            index=self.names,
        )


async def sample(model, draws=1000, warmup=1000, seed=42, point=None,
                 stanwasm_path=DEFAULT_STANWASM_PATH):
    """Compile `model` in the page and draw from it.

    Every call compiles: the data is part of the tape, so a module answers for
    one model and one dataset.
    """
    tape, point = tape_for(model, point)
    names = param_names(model)
    init = np.concatenate(
        [np.asarray(point[v.name], dtype=float).ravel() for v in model.value_vars]
    )
    got = await compile_and_sample(
        tape, init, warmup, draws, seed, names, stanwasm_path,
    )
    n = got["nParams"]
    flat = np.asarray(got["draws"], dtype=float)
    return Fit(names, flat[warmup * n:], got["ms"], got["moduleBytes"])
