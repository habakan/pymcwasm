"""Sample a PyMC model in a browser, without a Python sampler.

`pymcwasm` lowers a PyMC model's log density to the autodiff tape that
[tapewasm](https://github.com/habakan/tapewasm) compiles, hands the tape to
tapewasm's emitter, and draws from the resulting wasm module with nuts-rs.
Inside Pyodide, all of that happens in the page.

    import pymcwasm
    fit = await pymcwasm.sample(model, draws=1000, warmup=1000, seed=42)
    fit["beta"]      # one parameter's draws
    fit.summary()    # mean and sd per parameter

With several chains, the fit becomes an ArviZ `InferenceData` in the model's
own space — posterior, and sampler statistics when tapewasm returns them:

    fit = await pymcwasm.sample(model, chains=4)
    idata = fit.to_inference_data()

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
    """Draws, one column per unconstrained scalar, kept per chain.

    `draws` stacks the chains' post-warmup draws, so `fit["beta"]` and
    `summary()` pool them; `to_inference_data()` keeps them apart.
    """

    def __init__(self, names, chain_draws, warmup_draws, stats, ms, module_bytes,
                 compile_ms, lower_ms, model=None):
        self.names = list(names)
        n = len(self.names)
        self.chain_draws = np.asarray(chain_draws, dtype=float).reshape(len(chain_draws), -1, n)
        self.warmup_draws = np.asarray(warmup_draws, dtype=float).reshape(len(chain_draws), -1, n)
        self.draws = self.chain_draws.reshape(-1, n)
        self.stats = stats
        self.ms = ms
        self.module_bytes = module_bytes
        self.compile_ms = compile_ms
        self.lower_ms = lower_ms
        self._model = model

    def to_inference_data(self, save_warmup=True):
        """The fit as an ArviZ `InferenceData`, in the model's own space.

        Needs ArviZ, and the model `compile` was given: each draw goes back
        through the model's transforms, so `sigma` rather than `sigma_log__`,
        with deterministics alongside and the model's dims and coords.
        """
        import arviz as az
        from pymc.util import get_default_varnames

        model = self._model
        outs = get_default_varnames(model.unobserved_value_vars, include_transformed=False)
        fn = model.compile_fn(outs, inputs=model.value_vars, on_unused_input="ignore",
                              point_fn=False)
        point = model.initial_point()
        shapes = [np.shape(np.asarray(point[v.name])) for v in model.value_vars]
        cuts = np.cumsum([int(np.prod(s)) for s in shapes])[:-1]

        def constrained(block):
            """(chain, draw, scalar) -> {name: (chain, draw, *shape)}."""
            per_chain = []
            for chain in block:
                rows = [fn(*[p.reshape(s) for p, s in zip(np.split(row, cuts), shapes)])
                        for row in chain]
                per_chain.append([np.stack([np.asarray(r[i]) for r in rows])
                                  for i in range(len(outs))])
            return {v.name: np.stack([c[i] for c in per_chain]) for i, v in enumerate(outs)}

        groups = {"posterior": constrained(self.chain_draws)}
        warmup = self.warmup_draws.shape[1]
        if self.stats is not None:
            stats = {k: np.asarray(v) for k, v in self.stats.items() if k != "tuning"}
            stats["diverging"] = stats["diverging"].astype(bool)
            groups["sample_stats"] = {k: v[:, warmup:] for k, v in stats.items()}
            if save_warmup and warmup:
                groups["warmup_sample_stats"] = {k: v[:, :warmup] for k, v in stats.items()}
        if save_warmup and warmup:
            groups["warmup_posterior"] = constrained(self.warmup_draws)
        dims = {v.name: list(model.named_vars_to_dims[v.name])
                for v in outs if v.name in model.named_vars_to_dims}
        coords = {k: list(c) for k, c in model.coords.items() if c is not None}
        # ArviZ 1.0 takes the groups as one mapping; 0.x took one keyword per group.
        import inspect

        if "posterior" in inspect.signature(az.from_dict).parameters:
            return az.from_dict(**groups, coords=coords, dims=dims, save_warmup=save_warmup)
        return az.from_dict(groups, coords=coords, dims=dims, save_warmup=save_warmup)

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

    def __init__(self, handle, tw, names, init, lower_ms, model=None):
        self._handle = handle
        self._tw = tw
        self.names = names
        self.init = init
        self.lower_ms = lower_ms
        self.compile_ms = handle.ms
        self.module_bytes = handle.bytes
        self._model = model

    async def sample(self, draws=1000, warmup=1000, seed=42, chains=1):
        """`chains` run one after another from the same start, chain c seeded `seed + c`."""
        n = len(self.names)
        runs = [
            await _bridge.draw(self._handle, self._tw, self.init, warmup, draws, seed + c,
                               self.names, chain=c)
            for c in range(chains)
        ]
        flat = np.stack([np.asarray(r["draws"], dtype=float).reshape(-1, n) for r in runs])
        stats = None
        if all(r.get("stats") for r in runs):
            stats = {k: np.stack([np.asarray(r["stats"][k]) for r in runs])
                     for k in runs[0]["stats"]}
        return Fit(self.names, flat[:, warmup:], flat[:, :warmup], stats,
                   sum(r["ms"] for r in runs), self.module_bytes, self.compile_ms,
                   self.lower_ms, self._model)


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
    return Compiled(handle, tw, names, init, lower_ms, model)


async def sample(model, draws=1000, warmup=1000, seed=42, chains=1, point=None,
                 tapewasm_path=DEFAULT_TAPEWASM_PATH):
    """Compile `model` and draw from it, in one go."""
    compiled = await compile(model, point, tapewasm_path)
    return await compiled.sample(draws=draws, warmup=warmup, seed=seed, chains=chains)
