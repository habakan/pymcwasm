"""The lowering against PyMC's own logp, replayed in Python at a point it was not traced at.

    uv run --no-project --with pymc --with scipy --with pytest pytest tests
"""

import json
import os
import shutil
import sys

import numpy as np
import pymc as pm
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pymcwasm.lowering import _apply, lower  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def replay(path, params):
    vals, root = [], None
    for line in open(path):
        f = line.split()
        if not f or f[0] in ("n_params", "test_params", "outputs"):
            continue
        if f[0] == "root":
            root = int(f[1])
        elif f[0] == "new_var" and len(vals) < len(params):
            vals.append(float(params[len(vals)]))
        else:
            vals.append(_apply(f[0], f[1:], vals))
    return vals[root]


def lp_at(model, tmp_path, trace_at, test_at):
    path = str(tmp_path / "m.tape")
    lower(model, path, trace_at, test_at)
    flat = np.concatenate([np.ravel(test_at[v.name]) for v in model.value_vars])
    return replay(path, flat), float(model.compile_logp()(test_at))


def test_a_bernoulli_logit_far_from_zero_stays_finite(tmp_path):
    # sigmoid(40) rounds to 1, so a log1p(-sigmoid) written out is log(0).
    with pm.Model() as m:
        z = pm.Normal("z", shape=2)
        pm.Bernoulli("y", logit_p=z * 40, observed=[0, 1])
    got, want = lp_at(m, tmp_path, {"z": np.array([0.3, -0.2])}, {"z": np.array([1.0, -1.0])})
    assert np.isfinite(got) and got == pytest.approx(want, rel=1e-12)


def test_a_bounds_check_that_is_true_when_out_of_bounds_folds_to_in_bounds(tmp_path):
    # HalfFlat is switch(value < 0, -inf, 0), the opposite way round to most checks.
    with pm.Model() as m:
        s = pm.HalfFlat("s")
        pm.Normal("y", 0, s, observed=[0.5, -1.0])
    got, want = lp_at(m, tmp_path, {"s_log__": np.array(0.2)}, {"s_log__": np.array(-0.4)})
    assert got == pytest.approx(want, rel=1e-12)


def test_integer_constants_do_not_wrap(tmp_path):
    # nu * sigma**2 is int8 here: 300 on PyMC's compiled path, 44 evaluated as written.
    with pm.Model() as m:
        pm.HalfStudentT("s", nu=3, sigma=10)
    got, want = lp_at(m, tmp_path, {"s_log__": np.array(0.5)}, {"s_log__": np.array(2.0)})
    assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.skipif(shutil.which("node") is None
                    or not os.path.isdir(os.path.join(ROOT, "node_modules", "tapewasm")),
                    reason="needs Node and `npm install`")
def test_build_writes_what_a_precompiled_host_reads(tmp_path, monkeypatch):
    from pymcwasm.build import build

    monkeypatch.chdir(ROOT)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=[0.1, -0.3, 0.4])
    build(m, str(tmp_path))
    meta = json.load(open(tmp_path / "meta.json"))
    assert meta["nParams"] == 2 and meta["paramNames"] == ["mu", "sigma_log__"]
    assert len(meta["initialPoint"]) == 2 and 0 <= meta["layoutId"] < 2**32
    assert len(meta["scratchInit"]) >= 4 and meta["logLik"][0]["shape"] == [3]
    assert (tmp_path / "model.wasm").read_bytes()[:4] == b"\0asm"
