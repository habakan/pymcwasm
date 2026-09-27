"""The benchmark's shared half, run as is by CPython (bench/native.py) and Pyodide (bench/worker.js).

Every sampler draws the same 2 chains of 500 warmup and 500 draws, one chain after the
other, and is scored by the smallest bulk ESS over the free variables per second of
sampling. Compiling is timed apart, since it is paid once per model and not per draw.
"""

import time

import numpy as np

CHAINS, TUNE, DRAWS, SEED = 2, 500, 500, 1


def load(source, data):
    """A posteriordb PyMC model: `make_model(data)`, or `model(data)` in the older files."""
    ns = {}
    exec(compile(source, "model.py", "exec"), ns)
    return ns.get("make_model", ns.get("model"))(data)


def summarise(model, idata, sample_s, compile_s=0.0):
    """Smallest bulk ESS over the free variables, and each scalar's mean and sd."""
    import arviz as az

    post = idata["posterior"]
    ess, means, sds = [], [], []
    for rv in model.free_RVs:
        a = np.asarray(post[rv.name].values, dtype=float)
        a = a.reshape(a.shape[0], a.shape[1], -1)
        for j in range(a.shape[2]):
            ess.append(float(np.asarray(az.ess(a[:, :, j], method="bulk"))))
        means.extend(a.mean(axis=(0, 1)).tolist())
        sds.extend(a.std(axis=(0, 1)).tolist())
    min_ess = float(np.nanmin(ess))
    return {"sample_s": sample_s, "compile_s": compile_s, "min_ess": min_ess,
            "ess_per_s": min_ess / sample_s, "means": means, "sds": sds}


def pymc_nuts(model, **kw):
    """PyMC's own NUTS. `sampling_time` is what PyMC records, without its compile."""
    import pymc as pm

    with model:
        t = time.perf_counter()
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=1, random_seed=SEED,
                          nuts_sampler="pymc", progressbar=False,
                          compute_convergence_checks=False, **kw)
        total = time.perf_counter() - t
    sample_s = float(idata["posterior"].attrs.get("sampling_time", total))
    out = summarise(model, idata, sample_s, compile_s=total - sample_s)
    # Gradient evaluations, warmup included: what a slower linker would have to repeat.
    steps = float(np.asarray(idata["sample_stats"]["n_steps"].values).sum())
    out["grad_evals"] = steps * (TUNE + DRAWS) / DRAWS
    return out


def seconds_per_gradient(model, repeats=5):
    """One gradient on whatever PyTensor's configured linker is, at the initial point."""
    f, point = model.compile_dlogp(), model.initial_point()
    f(point)
    t = time.perf_counter()
    for _ in range(repeats):
        f(point)
    return (time.perf_counter() - t) / repeats


def nutpie(model):
    import nutpie as np_

    t = time.perf_counter()
    compiled = np_.compile_pymc_model(model)
    compile_s = time.perf_counter() - t
    t = time.perf_counter()
    idata = np_.sample(compiled, draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, seed=SEED,
                       progress_bar=False)
    return summarise(model, idata, time.perf_counter() - t, compile_s)


async def pymcwasm_sample(model, tapewasm_path):
    import pymcwasm

    t = time.perf_counter()
    compiled = await pymcwasm.compile(model, tapewasm_path=tapewasm_path)
    compile_s = time.perf_counter() - t
    t = time.perf_counter()
    fit = await compiled.sample(draws=DRAWS, warmup=TUNE, chains=CHAINS, seed=SEED)
    return summarise(model, fit.to_inference_data(), time.perf_counter() - t, compile_s)
