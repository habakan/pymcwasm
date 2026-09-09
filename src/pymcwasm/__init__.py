"""Sample a PyMC model in a browser, without a Python sampler.

`pymcwasm` lowers a PyMC model's log density to the autodiff tape that
[tapewasm](https://github.com/habakan/tapewasm) compiles, hands the tape to
tapewasm's emitter, and draws from the resulting wasm module with nuts-rs.
Inside Pyodide, all of that happens in the page.

    import pymcwasm
    fit = await pymcwasm.sample(model, draws=1000, warmup=1000, seed=42)
    fit["beta"]      # one parameter's draws
    fit.summary()    # mean and sd per parameter

Compiling is the slow half and does not depend on the draws, so it is worth
keeping when a page samples the same model more than once:

    compiled = await pymcwasm.compile(model)
    fit = await compiled.sample(draws=2000, seed=7)

Draws are in the space the sampler works in, so a transformed parameter comes
back under its value-variable name (`sigma_log__`, not `sigma`).

`import pymcwasm` is safe anywhere; `sample` needs the `js`/`pyodide` modules a
Pyodide runtime provides.

Not affiliated with PyMC.
"""

import numpy as np

from . import lowering
from . import _bridge
from ._bridge import DEFAULT_TAPEWASM_PATH

__all__ = ["Compiled", "Fit", "compile", "sample", "starting_point", "tape_for"]


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

    def __init__(self, names, draws, ms, module_bytes, compile_ms, lower_ms):
        self.names = list(names)
        self.draws = np.asarray(draws, dtype=float).reshape(-1, len(self.names))
        self.ms = ms
        self.module_bytes = module_bytes
        self.compile_ms = compile_ms
        self.lower_ms = lower_ms

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


class Compiled:
    """A model already lowered and emitted, ready to sample as often as wanted.

    Compiling is the expensive half — the graph walk especially — and it does
    not depend on the draws, so it is worth keeping.
    """

    def __init__(self, handle, tw, names, init, lower_ms):
        self._handle = handle
        self._tw = tw
        self.names = names
        self.init = init
        self.lower_ms = lower_ms
        self.compile_ms = handle.ms
        self.module_bytes = handle.bytes

    async def sample(self, draws=1000, warmup=1000, seed=42):
        got = await _bridge.draw(
            self._handle, self._tw, self.init, warmup, draws, seed, self.names,
        )
        n = got["nParams"]
        flat = np.asarray(got["draws"], dtype=float)
        return Fit(self.names, flat[warmup * n:], got["ms"], self.module_bytes,
                   self.compile_ms, self.lower_ms)


async def compile(model, point=None, tapewasm_path=DEFAULT_TAPEWASM_PATH):
    """Lower `model` and emit its module.

    The data is part of the tape, so the result answers for one model and one
    dataset — but for as many draws as asked for.
    """
    import time

    t0 = time.perf_counter()
    tape, point = tape_for(model, point)
    names = param_names(model)
    init = np.concatenate(
        [np.asarray(point[v.name], dtype=float).ravel() for v in model.value_vars]
    )
    lower_ms = (time.perf_counter() - t0) * 1000
    handle, tw = await _bridge.compile(tape, tapewasm_path)
    return Compiled(handle, tw, names, init, lower_ms)


async def sample(model, draws=1000, warmup=1000, seed=42, point=None,
                 tapewasm_path=DEFAULT_TAPEWASM_PATH):
    """Compile `model` and draw from it, in one go."""
    compiled = await compile(model, point, tapewasm_path)
    return await compiled.sample(draws=draws, warmup=warmup, seed=seed)
