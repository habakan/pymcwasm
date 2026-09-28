"""`mode="WASM"` against PyTensor's own backend, graph by graph.

    npm install && uv run --no-project --with pymc --with scipy --with wasmtime --with pytest pytest tests
"""

import os
import shutil
import sys

import numpy as np
import pytest

pytest.importorskip("wasmtime")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if shutil.which("node") is None or not os.path.isdir(os.path.join(ROOT, "node_modules", "tapewasm")):
    pytest.skip("needs Node and `npm install`", allow_module_level=True)

import pytensor  # noqa: E402
import pytensor.tensor as pt  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "src"))
import pymcwasm.linker  # noqa: E402, F401
from pymcwasm.lowering import MODELS, jitter  # noqa: E402


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)  # where `tapewasm` resolves from


def compare_wasm_and_py(inputs, outputs, values, rtol=1e-12):
    got = pytensor.function(inputs, outputs, mode="WASM")(*values)
    want = pytensor.function(inputs, outputs)(*values)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=rtol, atol=1e-14)
    return got


RNG = np.random.default_rng(0)
X, A = pt.vector("x"), pt.matrix("A")
x4, a34 = RNG.normal(size=4), RNG.normal(size=(3, 4))


@pytest.mark.parametrize("ins, outputs", [
    ([X], lambda: [pt.exp(X).sum()]),
    ([X], lambda: [pt.log1p(X ** 2), X.mean()]),
    ([X], lambda: [pt.softplus(X * 50), pt.sigmoid(-X * 50)]),
    ([A, X], lambda: [A @ X, A.T.sum(axis=0)]),
    ([A], lambda: [A.dimshuffle(1, "x", 0)[:, 0, 1:]]),
    ([A], lambda: [pt.concatenate([A, 2 * A], axis=1)]),
], ids=["exp-sum", "two-outputs", "softplus-far", "matvec", "dimshuffle", "join-axis-1"])
def test_graphs_agree(ins, outputs):
    compare_wasm_and_py(ins, outputs(), [a34 if v is A else x4 for v in ins])


def test_a_gradient_through_repeated_indices_adds_each_one():
    idx = np.array([0, 2, 2, 1, 2])
    (g,) = compare_wasm_and_py([X], [pt.grad((X[idx] ** 2).sum(), X)], [x4])
    assert g[2] == pytest.approx(6 * x4[2])


def test_a_branch_taken_the_other_way_traces_again():
    f = pytensor.function([X], [pt.switch(X > 0, X, -2 * X)], mode="WASM")
    for x in (np.array([1.0, -1.0]), np.array([-3.0, 2.0]), np.array([1.0, -1.0])):
        np.testing.assert_allclose(f(x)[0], np.where(x > 0, x, -2 * x))


def test_overflow_and_domain_errors_are_inf_and_nan_as_pytensor_gives():
    # math.exp raises past 709 and (-8) ** 0.5 is complex in Python; the host imports must not.
    compare_wasm_and_py([X], [pt.exp(X * 1000), pt.log(X), X ** 0.5], [np.array([1.0, -1.0])])


def test_an_integer_input_folds_in_and_a_new_value_traces_again():
    i = pt.lvector("i")
    f = pytensor.function([X, i], [X[i] * 2, i + 1], mode="WASM")
    x = np.array([1.0, 2.0, 3.0])
    for idx in (np.array([2, 0]), np.array([1, 1])):
        np.testing.assert_allclose(f(x, idx)[0], x[idx] * 2)
        np.testing.assert_array_equal(f(x, idx)[1], idx + 1)


def test_pymc_with_an_integer_index_in_data_agrees():
    import pymc as pm

    g = np.array([0, 1, 1, 2, 0])
    with pm.Model() as m:
        b = pm.Normal("b", 0, 1, shape=3)
        idx = pm.Data("g", g)
        pm.Normal("y", b[idx], 1, observed=np.arange(5.0))
    point = jitter(m.initial_point(), np.random.default_rng(3), 0.5)
    for compile_ in (m.compile_logp, m.compile_dlogp):
        np.testing.assert_allclose(compile_(mode="WASM")(point), compile_()(point), rtol=1e-12)


@pytest.mark.parametrize("name", list(MODELS))
def test_pymc_logp_and_dlogp_agree(name):
    m = MODELS[name]()
    point = jitter(m.initial_point(), np.random.default_rng(3), 0.5)
    for compile_ in (m.compile_logp, m.compile_dlogp):
        np.testing.assert_allclose(compile_(mode="WASM")(point), compile_()(point), rtol=1e-12)
