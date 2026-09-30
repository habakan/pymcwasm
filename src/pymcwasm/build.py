"""Compile a PyMC model into the files a page serves: `model.wasm`, `expand.wasm`, `meta.json`.

    pymcwasm-build model.py out/ [--data data.json] [--reroll always] [--var-names a,b]

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
from .lowering import lower_expansion

COMPILER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_compile.mjs")


def compile_tape(tape, wasm_path, reroll="auto"):
    """Emit `tape` to `wasm_path` with npm's tapewasm; returns what a host needs beside it."""
    out = subprocess.run(["node", COMPILER, wasm_path, reroll], input=tape,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"tapewasm could not compile the tape:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)


def build(model, out_dir, log_lik=True, reroll="auto", point=None, expand=True, var_names=None):
    """Write `model.tape`, `model.wasm` and `meta.json` into `out_dir`, and return the meta.

    With `expand`, also `expand.tape` and `expand.wasm`, whose `evaluate` returns the
    selected variables in their own space and the deterministics for one draw.
    """
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
    if expand:
        etape, layout = lower_expansion(model, point, var_names)
        with open(os.path.join(out_dir, "expand.tape"), "w") as f:
            f.write(etape)
        ebuilt = compile_tape(etape, os.path.join(out_dir, "expand.wasm"), reroll)
        assert ebuilt["nOutputs"] == sum(v["size"] for v in layout)
        meta["expand"] = {
            "layoutId": ebuilt["layoutId"],
            "scratchInit": ebuilt["scratchInit"],
            "nOutputs": ebuilt["nOutputs"],
            # nuts-rs-wasm's expanded_layout and coords, so a host reads either the same way.
            "layout": layout,
            "coords": json.loads(json.dumps(
                {str(k): np.asarray(v).tolist() for k, v in model.coords.items() if v is not None},
                default=lambda value: value.isoformat())),
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
    p.add_argument("--no-expand", action="store_true",
                   help="leave out `expand.wasm`, the constrained values and deterministics")
    p.add_argument("--var-names", help="comma-separated variables `expand.wasm` returns "
                   "(default: free variables, then deterministics)")
    args = p.parse_args(argv)

    data = None
    if args.data:
        with open(args.data) as f:
            data = json.load(f)
    model = load_model(args.model, data)
    meta = build(model, args.out_dir, log_lik=not args.no_log_lik, reroll=args.reroll,
                 expand=not args.no_expand,
                 var_names=args.var_names.split(",") if args.var_names else None)
    size = os.path.getsize(os.path.join(args.out_dir, "model.wasm"))
    print(f"{args.out_dir}: {meta['nParams']} params, {size} bytes of wasm")


if __name__ == "__main__":
    main()
