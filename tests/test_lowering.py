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
from pymcwasm.lowering import _apply, lower, lower_expansion  # noqa: E402

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
    assert meta["expand"]["layout"] == [
        {"name": "mu", "shape": [], "size": 1, "dims": []},
        {"name": "sigma", "shape": [], "size": 1, "dims": []},
    ]
    assert (tmp_path / "expand.wasm").read_bytes()[:4] == b"\0asm"


def test_a_start_where_pymc_gradient_is_nan_is_not_refused():
    # A logit of 400 makes PyMC's own d/dz of the Bernoulli term NaN; the start is fine.
    from pymcwasm import starting_point

    with pm.Model() as m:
        z = pm.Normal("z", 3.0, 0.1)
        pm.Bernoulli("y", logit_p=z * 130, observed=[1, 0])
    grad = np.asarray(m.compile_dlogp()(m.initial_point()), dtype=float)
    if np.all(np.isfinite(grad)):
        pytest.skip("this PyMC's gradient is finite here")
    starting_point(m)


def test_the_expansion_is_each_variable_in_its_own_space_then_the_deterministics():
    with pm.Model(coords={"g": ["a", "b"]}) as m:
        mu = pm.Normal("mu", 0, 1, dims="g")
        sigma = pm.HalfNormal("sigma", 1)
        pm.Deterministic("scaled", mu * sigma, dims="g")
        pm.Normal("y", mu.sum(), sigma, observed=[0.1, -0.3])
    tape, layout = lower_expansion(m, m.initial_point())
    assert [(v["name"], v["shape"], v["dims"]) for v in layout] == [
        ("mu", [2], ["g"]), ("sigma", [], []), ("scaled", [2], ["g"])]

    # Replayed at a point the tape was not traced at.
    params = [0.3, -0.7, 0.4]
    vals, outputs = [], []
    for line in tape.splitlines():
        f = line.split()
        if f[0] == "outputs":
            outputs = [int(i) for i in f[1:]]
        elif f[0] == "new_var" and len(vals) < len(params):
            vals.append(params[len(vals)])
        elif f[0] not in ("n_params", "root"):
            vals.append(_apply(f[0], f[1:], vals))
    s = np.exp(0.4)
    np.testing.assert_allclose([vals[i] for i in outputs], [0.3, -0.7, s, 0.3 * s, -0.7 * s])


def _scan_models():
    import pytensor
    import pytensor.tensor as pt

    y = np.sin(np.arange(12) / 3.0)
    with pm.Model() as ar1:  # one state, its previous value
        rho = pm.Normal("rho", 0, 0.5)
        s = pm.HalfNormal("s", 1)
        mu, _ = pytensor.scan(lambda prev, r: r * prev, outputs_info=[pt.as_tensor(y[0])],
                              non_sequences=[rho], n_steps=len(y) - 1)
        pm.Normal("y", mu, s, observed=y[1:])
    with pm.Model() as ar2:  # one state, two taps
        a = pm.Normal("a", 0, 0.5, shape=2)
        s = pm.HalfNormal("s", 1)
        mu, _ = pytensor.scan(lambda p2, p1, a: a[0] * p2 + a[1] * p1,
                              outputs_info=[{"initial": pt.as_tensor(y[:2]), "taps": [-2, -1]}],
                              non_sequences=[a], n_steps=len(y) - 2)
        pm.Normal("y", mu, s, observed=y[2:])
    with pm.Model() as seq:  # a sequence in, one output per step
        b = pm.Normal("b", 0, 1)
        s = pm.HalfNormal("s", 1)
        mu, _ = pytensor.scan(lambda x, b: pt.exp(b * x), sequences=[pt.as_tensor(y)], non_sequences=[b])
        pm.Normal("y", mu, s, observed=np.cos(y))
    return {"ar1": ar1, "ar2": ar2, "seq": seq}


@pytest.mark.parametrize("name", ["ar1", "ar2", "seq"])
def test_a_scan_unrolls(tmp_path, name):
    m = _scan_models()[name]
    rng = np.random.default_rng(3)
    ip = m.initial_point()
    trace_at = {k: v + rng.normal(0, 0.3, np.shape(v)) for k, v in ip.items()}
    test_at = {k: v + rng.normal(0, 0.3, np.shape(v)) for k, v in ip.items()}
    got, want = lp_at(m, tmp_path, trace_at, test_at)
    assert got == pytest.approx(want, rel=1e-12)


def _switch_models():
    t = np.arange(20.0)
    with pm.Model() as changepoint:  # the branch moves with tau
        tau = pm.Uniform("tau", 0, 20)
        mu1, mu2 = pm.Normal("mu1", 0, 5), pm.Normal("mu2", 0, 5)
        pm.Normal("y", pm.math.switch(t < tau, mu1, mu2), 1, observed=np.where(t < 8, 1.0, 3.0))
    with pm.Model() as maximum:
        a, b = pm.Normal("a", 0, 1), pm.Normal("b", 0, 1)
        pm.Normal("y", pm.math.maximum(a, b), 1, observed=[0.3, 0.8])
    with pm.Model() as nan_branch:  # log x is NaN on the side not taken
        x = pm.Normal("x", 0, 2)
        pm.Normal("y", pm.math.switch(x > 0, pm.math.log(x), x), 1, observed=[0.1, -0.4])
    return {"changepoint": changepoint, "maximum": maximum, "nan_branch": nan_branch}


@pytest.mark.parametrize("name,trace,test", [
    ("changepoint", {"tau_interval__": -1.2, "mu1": 1.0, "mu2": 3.0},
     {"tau_interval__": 0.6, "mu1": 0.7, "mu2": 2.5}),
    ("maximum", {"a": 0.9, "b": -0.2}, {"a": -0.5, "b": 0.4}),
    ("nan_branch", {"x": 1.3}, {"x": -0.7}),
])
def test_a_switch_on_a_parameter_branches_where_it_is_evaluated(tmp_path, name, trace, test):
    m = _switch_models()[name]
    as_arrays = lambda p: {k: np.asarray(v, dtype=float) for k, v in p.items()}
    got, want = lp_at(m, tmp_path, as_arrays(trace), as_arrays(test))
    assert np.isfinite(got) and got == pytest.approx(want, rel=1e-12)


def test_a_bounds_check_still_folds(tmp_path):
    # Uniform's switch(lower <= x <= upper, logp, -inf) holds wherever the transform reaches.
    with pm.Model() as m:
        pm.Uniform("u", 0, 5)
        pm.Normal("y", 0, 1, observed=[0.2])
    lower(m, str(tmp_path / "m.tape"), m.initial_point(), m.initial_point())
    ops = {line.split()[0] for line in open(tmp_path / "m.tape")}
    assert not ops & {"pick", "gt", "ge", "lt", "le", "eq", "ne"}


def test_a_max_reduction_follows_the_largest_element(tmp_path):
    # NormalMixture's logsumexp subtracts a max over the components.
    with pm.Model() as m:
        w = pm.Dirichlet("w", a=np.ones(3))
        mu = pm.Normal("mu", 0, 3, shape=3)
        pm.NormalMixture("y", w=w, mu=mu, sigma=1.0, observed=[0.1, 1.2, -0.3])
    trace_at = {"w_simplex__": np.array([0.2, -0.1]), "mu": np.array([2.0, -1.0, 0.5])}
    test_at = {"w_simplex__": np.array([-0.3, 0.4]), "mu": np.array([-1.5, 1.0, 2.5])}
    got, want = lp_at(m, tmp_path, trace_at, test_at)
    assert got == pytest.approx(want, rel=1e-12)


def test_an_ordered_transform_lowers(tmp_path):
    # Its backward pass fills an AllocEmpty with set_subtensor.
    with pm.Model() as m:
        pm.Normal("mu", 0, 1, shape=3, transform=pm.distributions.transforms.ordered)
    trace_at = {"mu_ordered__": np.array([-0.4, 0.2, -0.1])}
    test_at = {"mu_ordered__": np.array([0.3, -0.5, 0.7])}
    got, want = lp_at(m, tmp_path, trace_at, test_at)
    assert got == pytest.approx(want, rel=1e-12)


def test_a_discrete_parameter_is_refused_by_name(tmp_path):
    with pm.Model() as m:
        pm.Bernoulli("z", 0.5, shape=2)
    with pytest.raises(NotImplementedError, match=r"discrete parameters \(z\)"):
        lower(m, str(tmp_path / "m.tape"))


def test_a_constant_needed_before_any_op_lowers(tmp_path):
    # set_subtensor of a raw parameter into an empty buffer: the buffer's other rows are
    # constants, and nothing has been computed before them. Only the row set is read, as
    # PyMC's own logp would read uninitialized memory in the others.
    import pytensor.tensor as pt

    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        buf = pt.set_subtensor(pt.empty((3,))[:1], x)
        pm.Normal("y", buf[0], 1, observed=[0.4])
    got, want = lp_at(m, tmp_path, {"x": np.array(0.3)}, {"x": np.array(-0.8)})
    assert got == pytest.approx(want, rel=1e-12)


def test_a_switch_first_in_the_graph_lowers(tmp_path):
    # Its comparison against 0 needs a constant before anything else is on the tape.
    with pm.Model() as m:
        a, b = pm.Normal("a", 0, 1), pm.Normal("b", 0, 1)
        pm.Normal("y", pm.math.switch(a > 0, a, b), 1, observed=[0.2])
    got, want = lp_at(m, tmp_path, {"a": np.array(0.7), "b": np.array(0.1)},
                      {"a": np.array(-0.4), "b": np.array(0.9)})
    assert got == pytest.approx(want, rel=1e-12)


def test_a_comparison_stacked_before_its_switch_still_branches(tmp_path):
    import pytensor.tensor as pt

    with pm.Model() as m:
        a, b = pm.Normal("a", 0, 1), pm.Normal("b", 0, 1)
        mu = pm.math.switch(pt.stack([a > 0, b > 0]), pt.stack([a, b]), 0.0)
        pm.Normal("y", mu, 1, observed=[0.2, 0.1])
    got, want = lp_at(m, tmp_path, {"a": np.array(0.7), "b": np.array(-0.3)},
                      {"a": np.array(-0.6), "b": np.array(0.4)})
    assert got == pytest.approx(want, rel=1e-12)


def test_a_switch_on_a_reduced_comparison_is_refused(tmp_path):
    # `all` folds the comparisons into a plain constant, which would bake in the branch.
    import pytensor.tensor as pt

    with pm.Model() as m:
        a, b = pm.Normal("a", 0, 1), pm.Normal("b", 0, 1)
        mu = pm.math.switch(pt.all(pt.stack([a, b]) > 0), a, b)
        pm.Normal("y", mu, 1, observed=[0.2])
    with pytest.raises(NotImplementedError, match="keeps no record"):
        lower(m, str(tmp_path / "m.tape"))


def test_a_branch_infinite_in_some_elements_is_not_a_bounds_check(tmp_path):
    import pytensor.tensor as pt

    with pm.Model() as m:
        a = pm.Normal("a", 0, 1)
        mu = pm.math.switch(a > 0, a * pt.ones(2), pt.as_tensor([-np.inf, 3.0]))
        pm.Potential("p", -0.5 * (mu[1] - 1.0) ** 2)
    got, want = lp_at(m, tmp_path, {"a": np.array(0.6)}, {"a": np.array(-0.5)})
    assert got == pytest.approx(want, rel=1e-12)


def test_a_switch_on_a_scan_state_branches_where_it_is_evaluated(tmp_path):
    import pytensor
    import pytensor.tensor as pt

    with pm.Model() as m:
        a, x = pm.Normal("a", 0, 1), pm.Normal("x", 0, 1)
        out, _ = pytensor.scan(lambda p, a, x: pt.switch(p > a, p * 0.5, p + x),
                               outputs_info=[pt.as_tensor(np.float64(1.0))], non_sequences=[a, x], n_steps=6)
        pm.Normal("y", out, 1, observed=np.linspace(0, 1, 6))
    got, want = lp_at(m, tmp_path, {"a": np.array(0.2), "x": np.array(0.5)},
                      {"a": np.array(0.9), "x": np.array(-0.3)})
    assert got == pytest.approx(want, rel=1e-12)


def _gp_models():
    X = np.linspace(0, 10, 12)[:, None]
    with pm.Model() as marginal:
        ls, eta, s = pm.Gamma("ls", 2, 1), pm.HalfNormal("eta", 1), pm.HalfNormal("s", 1)
        gp = pm.gp.Marginal(cov_func=eta**2 * pm.gp.cov.ExpQuad(1, ls))
        gp.marginal_likelihood("y", X, np.sin(X).ravel(), sigma=s)
    with pm.Model() as latent:
        ls, eta = pm.Gamma("ls", 2, 1), pm.HalfNormal("eta", 1)
        f = pm.gp.Latent(cov_func=eta**2 * pm.gp.cov.Matern52(1, ls)).prior("f", X)
        pm.Poisson("c", pm.math.exp(f), observed=np.arange(12) % 4)
    return {"marginal": marginal, "latent": latent}


@pytest.mark.parametrize("name", ["marginal", "latent"])
def test_a_gaussian_process_lowers(tmp_path, name):
    # The covariance clips its squared distance at 0, and the Cholesky is of parameters.
    m = _gp_models()[name]
    trace_at = m.initial_point()
    rng = np.random.default_rng(0)
    test_at = {k: v + 0.3 * rng.normal(size=np.shape(v)) for k, v in trace_at.items()}
    got, want = lp_at(m, tmp_path, trace_at, test_at)
    assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("trace,test", [
    ({"mu": 0.4, "s_interval__": 0.1}, {"mu": -1.3, "s_interval__": 0.5}),  # across the switch
    ({"mu": 0.4, "s_interval__": 0.1}, {"mu": 2.0, "s_interval__": -0.2}),
])
def test_a_truncated_student_t_lowers_its_incomplete_beta(tmp_path, trace, test):
    # Its normalizer is StudentT's cdf at the bound, an incomplete beta in mu.
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        s = pm.Truncated("s", pm.StudentT.dist(nu=3, mu=mu, sigma=2.5), lower=0, upper=10)
        pm.Normal("y", s, 1, observed=[0.5])
    as_arrays = lambda p: {k: np.asarray(v, dtype=float) for k, v in p.items()}
    got, want = lp_at(m, tmp_path, as_arrays(trace), as_arrays(test))
    assert got == pytest.approx(want, rel=1e-12)
