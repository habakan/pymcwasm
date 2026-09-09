"""Compile a PyMC model into the files a page would serve, and a reference to check them against.

Maintainer notes, not published. See PLAN.md.

Writes into <out>/:
    model.wasm      the emitted module
    meta.json       what `new AotSampler(...)` takes, plus the point to start from
    reference.json  nutpie's posterior mean and sd per parameter, for comparison

    uv run --with pymc --with scipy --with nutpie \\
        python docs/internal/pymc-front-end/build_artifact.py <model> <out dir>
"""

import json
import os
import subprocess
import sys

import numpy as np
import nutpie
import pymc as pm

from lower_pytensor import MODELS, REPO, lower


def compile_tape(tape_path, wasm_path):
    """Run the emitter over a tape file, and read back what a host needs."""
    out = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "stanwasm-codegen",
         "--example", "tape_from_text", "--", tape_path, wasm_path],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    meta, consts = {}, []
    for line in out.stdout.splitlines():
        f = line.split()
        if f[0] in ("n_params", "scratch_len", "layout_id"):
            meta[f[0]] = int(f[1])
        elif f[0] == "const":
            consts.append(float(f[1]))
    return meta, consts


def param_names(model):
    """One name per unconstrained scalar, in the order the sampler reads them."""
    ip = model.initial_point()
    names = []
    for v in model.value_vars:
        shape = np.shape(np.asarray(ip[v.name]))
        if not shape:
            names.append(v.name)
        else:
            for idx in np.ndindex(shape):
                names.append(f"{v.name}[{','.join(str(i) for i in idx)}]")
    return names


def starting_point(model, seed=0, tries=50):
    """A point nuts-rs will accept.

    It refuses a start whose gradient has a zero component, because the mass
    matrix it adapts is scaled by that gradient. PyMC's `initial_point()` is
    zeros, and at the origin a centred hierarchical model has an exactly zero
    gradient in its population mean, as does a logit regression with balanced
    data — so the default point is refused for shapes people actually write.
    """
    dlogp = model.compile_dlogp()
    ip = model.initial_point()
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        if np.all(np.abs(np.asarray(dlogp(ip), dtype=float)) > 1e-12):
            return ip
        ip = {k: np.asarray(v, dtype=float) + rng.uniform(-2, 2, np.shape(v))
              for k, v in model.initial_point().items()}
    raise RuntimeError("no starting point with a non-zero gradient in every component")


def reference_posterior(model, names, draws=1000, chains=4, seed=7):
    """nutpie's posterior, pushed back into the space the sampler works in.

    The draws come out constrained and the module's do not, so comparing them
    directly would leave every transformed parameter unchecked.
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
        columns[value.name] = unconstrained.reshape(unconstrained.shape[0], -1)

    # Keyed by the same names the artifact carries, so the two line up by name.
    out, at = {}, 0
    for value_name, col in columns.items():
        for j in range(col.shape[1]):
            out[names[at]] = {
                "mean": float(col[:, j].mean()),
                "sd": float(col[:, j].std()),
            }
            at += 1
    assert at == len(names), (at, len(names))
    return out


def build(name, out_dir):
    model = MODELS[name]()
    os.makedirs(out_dir, exist_ok=True)

    ip = starting_point(model)
    tape_path = os.path.join(out_dir, "model.tape")
    lower(model, tape_path, ip, ip)

    wasm_path = os.path.join(out_dir, "model.wasm")
    meta, consts = compile_tape(tape_path, wasm_path)

    # The buffer the module works in: zeroed primals and adjoints, then the
    # re-rolled loops' constants at the tail.
    scratch_init = [0.0] * meta["scratch_len"]
    if consts:
        scratch_init[meta["scratch_len"] - len(consts):] = consts

    names = param_names(model)
    assert len(names) == meta["n_params"], (len(names), meta["n_params"])
    start = np.concatenate(
        [np.asarray(ip[v.name], dtype=float).ravel() for v in model.value_vars]
    ).tolist()

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "model": name,
                "nParams": meta["n_params"],
                "layoutId": meta["layout_id"],
                "paramNames": names,
                "scratchInit": scratch_init,
                "initialPoint": start,
            },
            f,
        )

    with open(os.path.join(out_dir, "reference.json"), "w") as f:
        json.dump(reference_posterior(model, names), f)

    print(f"{name}: {meta['n_params']} params, "
          f"{os.path.getsize(wasm_path)} bytes of wasm, "
          f"{meta['scratch_len']} scratch slots")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: build_artifact.py <{'|'.join(MODELS)}> <out dir>")
        raise SystemExit(2)
    build(sys.argv[1], os.path.abspath(sys.argv[2]))
