"""Compile a PyMC model into the files a page serves: `model.wasm` and `meta.json`.

    pymcwasm-build model.py out/ [--data data.json] [--reroll always]

`model.py` defines either `model`, a `pm.Model`, or `make_model(data)` / `model(data)`
returning one. The emitter is npm's `tapewasm`, resolved from the working directory
(`npm i tapewasm`), so building needs Node and no Rust.
"""

import argparse
import json
import os
import runpy
import subprocess

import numpy as np

from . import param_names, starting_point, tape_for

COMPILER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_compile.mjs")


def compile_tape(tape, wasm_path, reroll="auto"):
    """Emit `tape` to `wasm_path` with npm's tapewasm; returns what a host needs beside it."""
    out = subprocess.run(["node", COMPILER, wasm_path, reroll], input=tape,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"tapewasm could not compile the tape:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)


def build(model, out_dir, log_lik=True, reroll="auto", point=None):
    """Write `model.tape`, `model.wasm` and `meta.json` into `out_dir`, and return the meta."""
    os.makedirs(out_dir, exist_ok=True)
    tape, point, groups = tape_for(model, point, log_lik=log_lik)
    with open(os.path.join(out_dir, "model.tape"), "w") as f:
        f.write(tape)
    built = compile_tape(tape, os.path.join(out_dir, "model.wasm"), reroll)

    names = param_names(model)
    terms = sum(int(np.prod(g["shape"])) for g in groups)
    assert len(names) == built["nParams"], (len(names), built["nParams"])
    assert terms == built["nOutputs"], (terms, built["nOutputs"])
    meta = {
        "nParams": built["nParams"],
        "layoutId": built["layoutId"],
        "paramNames": names,
        "scratchInit": built["scratchInit"],
        "initialPoint": np.concatenate(
            [np.asarray(point[v.name], dtype=float).ravel() for v in model.value_vars]
        ).tolist(),
        # What the module's `evaluate` reports, in order, so a page can shape the terms back.
        "logLik": groups,
        "tapewasm": built["tapewasm"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def load_model(path, data=None):
    import pymc as pm

    ns = runpy.run_path(path)
    found = ns.get("make_model", ns.get("model"))
    if isinstance(found, pm.Model):
        return found
    if found is None:
        raise SystemExit(f"{path} defines neither `model` nor `make_model(data)`")
    return found(data)


def main(argv=None):
    p = argparse.ArgumentParser(prog="pymcwasm-build", description=__doc__.split("\n")[0])
    p.add_argument("model", help="a Python file defining `model` or `make_model(data)`")
    p.add_argument("out_dir")
    p.add_argument("--data", help="a JSON file passed to `make_model`")
    p.add_argument("--reroll", choices=["auto", "always", "never"], default="auto",
                   help="`always` trades gradient speed for a smaller module")
    p.add_argument("--no-log-lik", action="store_true",
                   help="leave out the per-observation terms `evaluate` reports")
    args = p.parse_args(argv)

    data = None
    if args.data:
        with open(args.data) as f:
            data = json.load(f)
    model = load_model(args.model, data)
    meta = build(model, args.out_dir, log_lik=not args.no_log_lik, reroll=args.reroll)
    size = os.path.getsize(os.path.join(args.out_dir, "model.wasm"))
    print(f"{args.out_dir}: {meta['nParams']} params, {size} bytes of wasm")


if __name__ == "__main__":
    main()
