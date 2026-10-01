"""Lower every posteriordb posterior that has a PyMC implementation, and check the module.

    npm install && POSTERIORDB=/path/to/posteriordb \\
        uv run --no-project --with pymc --with scipy --with wasmtime \\
        python scripts/posteriordb.py [--expect scripts/posteriordb-passing.txt] [name ...]

Each is lowered at one point, emitted by npm's tapewasm as `pymcwasm-build` does, and its
density, gradient and log-likelihood terms compared with PyMC's at another. With
`--expect`, it exits 1 if any posterior named in that file no longer agrees.
"""

import argparse
import json
import os
import sys
import tempfile
import traceback
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pymcwasm.build import compile_tape  # noqa: E402
from pymcwasm.linker import Module  # noqa: E402
from pymcwasm.lowering import jitter, lower  # noqa: E402

DB = os.path.join(os.environ["POSTERIORDB"], "posterior_database")
TOLERANCE = 1e-8


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    with zipfile.ZipFile(path + ".zip") as z:
        return json.loads(z.read(z.namelist()[0]))


def posteriors():
    """(posterior name, model builder, has reference draws), for each PyMC implementation."""
    for fname in sorted(os.listdir(os.path.join(DB, "posteriors"))):
        post = load_json(os.path.join(DB, "posteriors", fname))
        info = load_json(os.path.join(DB, "models", "info", post["model_name"] + ".info.json"))
        impl = info["model_implementations"].get("pymc")
        if impl is None:
            continue
        ns = {}
        with open(os.path.join(DB, impl["model_code"])) as f:
            exec(compile(f.read(), impl["model_code"], "exec"), ns)
        data = load_json(os.path.join(DB, "data", "data", post["data_name"] + ".json"))
        yield post["name"], lambda ns=ns, data=data: ns.get("make_model", ns.get("model"))(data), bool(
            post.get("reference_posterior_name"))


def rel(got, want):
    got, want = np.atleast_1d(got), np.atleast_1d(want)
    if got.shape != want.shape:
        return np.inf
    return float(np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want)), initial=0.0))


def check(build):
    """Worst relative error of the module's density, gradient and log-likelihood terms."""
    model = build()
    rng = np.random.default_rng(11)
    # An ordered transform of equal initial values is log(0); both sides are -inf there.
    ip = {k: np.where(np.isfinite(v), v, 0.0) for k, v in model.initial_point().items()}
    trace_at, test_at = jitter(ip, rng, 0.3), jitter(ip, rng, 0.7)
    with tempfile.TemporaryDirectory() as d:
        tape = os.path.join(d, "model.tape")
        n_params, _ = lower(model, tape, trace_at, test_at)
        with open(tape) as f:
            built = compile_tape(f.read(), os.path.join(d, "model.wasm"))
        module = Module(os.path.join(d, "model.wasm"), n_params, built["scratchInit"], built["nOutputs"])
    x = np.concatenate([np.ravel(test_at[v.name]) for v in model.value_vars])
    lp, grad = module.log_prob_grad(x)
    terms = model.logp(vars=model.observed_RVs, sum=False)
    want_ll = np.concatenate([np.ravel(t) for t in model.compile_fn(
        terms, inputs=model.value_vars, on_unused_input="ignore", point_fn=True)(test_at)]) if terms else []
    got_ll = module.evaluate(x) if built["nOutputs"] else []
    logp = model.compile_logp()
    want_g = np.asarray(model.compile_dlogp()(test_at), dtype=float)
    # PyMC's own gradient can be NaN where its density is finite (a saturated sigmoid in
    # irt_2pl); there a central difference of that density is the reference instead.
    unknown = np.flatnonzero(~np.isfinite(want_g))
    if unknown.size:
        sizes = np.cumsum([0] + [np.size(test_at[v.name]) for v in model.value_vars])
        def at(v):
            return {var.name: v[a:b].reshape(np.shape(test_at[var.name]))
                    for var, a, b in zip(model.value_vars, sizes[:-1], sizes[1:])}
        for i in unknown:
            h = 1e-6 * max(1.0, abs(x[i]))
            up, down = x.copy(), x.copy()
            up[i] += h
            down[i] -= h
            want_g[i] = (float(logp(at(up))) - float(logp(at(down)))) / (2 * h)
    errs = {"lp": rel(lp, float(logp(test_at))), "grad": rel(grad, want_g), "log_lik": rel(got_ll, want_ll)}
    return n_params, errs, unknown.size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--expect", help="a file of posterior names that must agree")
    args = p.parse_args()
    expected = set()
    if args.expect:
        with open(args.expect) as f:
            expected = {line.strip() for line in f if line.strip() and not line.startswith("#")}

    passing = []
    for name, build, has_ref in posteriors():
        if args.names and name not in args.names:
            continue
        try:
            n_params, errs, by_difference = check(build)
            # A central difference is good to about 1e-6, so a gradient checked that way is too.
            tol = {"grad": 1e-5 if by_difference else TOLERANCE}
            ok = all(e < tol.get(k, TOLERANCE) for k, e in errs.items())
            status = "ok" if ok else "WRONG"
            detail = f"params {n_params:>5}  " + "  ".join(f"{k} {v:.1e}" for k, v in errs.items())
            if by_difference:
                detail += f"  ({by_difference} gradient terms by central difference: PyMC's are NaN)"
            if ok:
                passing.append(name)
        except Exception as e:
            # The lowering refuses what it cannot represent; the reason is the result.
            last = traceback.extract_tb(e.__traceback__)[-1]
            status, detail = "REFUSED", f"{type(e).__name__}: {str(e).splitlines()[0][:120]} ({last.name})"
        print(f"{status:<8} {name}{' [ref]' if has_ref else ''}  {detail}", flush=True)

    print(f"\n{len(passing)} agree")
    lost = sorted(expected - set(passing))
    gained = sorted(set(passing) - expected) if args.expect else []
    for name in gained:
        print(f"newly agrees: {name}")
    for name in lost:
        print(f"NO LONGER AGREES: {name}")
    sys.exit(1 if lost else 0)


if __name__ == "__main__":
    main()
