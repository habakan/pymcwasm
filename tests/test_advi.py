"""`Compiled.fit` and `Approximation`, natively: a fake of tapewasm's advi, no browser.

    uv run --no-project --with pymc --with arviz --with pytest pytest tests
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pymcwasm  # noqa: E402
from pymcwasm import Approximation, Compiled, param_names  # noqa: E402


def model():
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1, shape=2)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu[0], sigma, observed=[0.1, -0.3])
    return m


def compiled(m):
    names = param_names(m)  # mu[0], mu[1], sigma_log__
    return Compiled(SimpleNamespace(ms=1.0, bytes=100), None, names, np.zeros(len(names)), 1.0, m)


def fake_advi(calls, mu, sigma, elbo):
    async def advi(handle, tw, init, n, mc_samples, learning_rate, seed, names,
                   snapshot_every=0, callback=None):
        calls.append(dict(init=list(init), n=n, mc_samples=mc_samples, learning_rate=learning_rate,
                          seed=seed, snapshot_every=snapshot_every, callback=callback))
        if callback is not None:
            callback(n, list(mu))
        return {"mu": list(mu), "sigma": list(sigma), "elbo": list(elbo), "ms": 3.0}
    return advi


def test_fit_hands_its_settings_to_tapewasm_and_keeps_what_comes_back(monkeypatch):
    m = model()
    calls = []
    monkeypatch.setattr(pymcwasm._bridge, "advi",
                        fake_advi(calls, [0.5, -1.0, 0.2], [0.1, 0.2, 0.3], [-5.0, -4.0, -3.5]))
    approx = asyncio.run(compiled(m).fit(n=300, learning_rate=0.05, mc_samples=2, seed=9))

    assert isinstance(approx, Approximation)
    assert calls[0]["n"] == 300 and calls[0]["mc_samples"] == 2 and calls[0]["seed"] == 9
    assert calls[0]["learning_rate"] == 0.05
    assert approx.names == ["mu[0]", "mu[1]", "sigma_log__"]
    np.testing.assert_allclose(approx.mean, [0.5, -1.0, 0.2])
    np.testing.assert_allclose(approx.std, [0.1, 0.2, 0.3])
    # PyMC's approx.hist is the loss, the negative ELBO, iteration by iteration.
    np.testing.assert_allclose(approx.hist, [5.0, 4.0, 3.5])


def test_sample_draws_the_mean_field_gaussian_back_into_the_models_space(monkeypatch):
    m = model()
    mu = np.array([0.5, -1.0, np.log(2.0)])
    monkeypatch.setattr(pymcwasm._bridge, "advi", fake_advi([], mu, [1e-9, 1e-9, 1e-9], [-1.0]))
    approx = asyncio.run(compiled(m).fit(n=10))

    post = approx.sample(draws=50, seed=1)["posterior"]
    assert post["mu"].shape == (1, 50, 2)
    # sigma comes back through its log transform, not as sigma_log__.
    np.testing.assert_allclose(post["sigma"].values, 2.0, rtol=1e-6)
    np.testing.assert_allclose(post["mu"].values[0, :, 1], -1.0, atol=1e-6)


def test_sample_spreads_as_the_fitted_sd(monkeypatch):
    m = model()
    monkeypatch.setattr(pymcwasm._bridge, "advi",
                        fake_advi([], [0.0, 0.0, 0.0], [1.0, 3.0, 0.5], [-1.0]))
    draws = asyncio.run(compiled(m).fit(n=10)).sample(draws=20000, seed=2)["posterior"]["mu"].values
    np.testing.assert_allclose(draws[0].std(axis=0), [1.0, 3.0], rtol=0.03)


def test_a_callback_sees_the_iterate_as_the_fit_goes(monkeypatch):
    m = model()
    calls, seen = [], []
    monkeypatch.setattr(pymcwasm._bridge, "advi", fake_advi(calls, [1.0, 2.0, 3.0], [1, 1, 1], [0.0]))
    asyncio.run(compiled(m).fit(n=40, callback=lambda it, mean: seen.append((it, list(mean))),
                                snapshot_every=20))
    assert calls[0]["snapshot_every"] == 20
    assert seen == [(40, [1.0, 2.0, 3.0])]


def test_only_mean_field_advi_is_offered(monkeypatch):
    m = model()
    monkeypatch.setattr(pymcwasm._bridge, "advi", fake_advi([], [0, 0, 0], [1, 1, 1], [0.0]))
    with pytest.raises(ValueError, match="mean-field"):
        asyncio.run(compiled(m).fit(method="fullrank_advi"))
