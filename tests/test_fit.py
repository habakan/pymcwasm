"""`Fit` and `Compiled.sample`, natively: synthetic draws, no browser.

    uv run --no-project --with pymc --with arviz --with pytest pytest tests
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import numpy as np
import pymc as pm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pymcwasm  # noqa: E402
from pymcwasm import Compiled, Fit, param_names  # noqa: E402

STATS = ("diverging", "tuning", "step_size", "n_steps", "lp")


def model():
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1, shape=2)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Deterministic("twice", 2 * mu)
        pm.Normal("y", mu[0], sigma, observed=[0.1, -0.3])
    return m


def fit(m, chains=2, warmup=3, draws=5, stats=True):
    rng = np.random.default_rng(0)
    names = param_names(m)  # mu[0], mu[1], sigma_log__
    flat = rng.normal(size=(chains, warmup + draws, len(names)))
    st = None
    if stats:
        shape = (chains, warmup + draws)
        st = {k: rng.integers(0, 2, shape) if k in ("diverging", "tuning") else rng.normal(size=shape)
              for k in STATS}
    return Fit(names, flat[:, warmup:], flat[:, :warmup], st, 1.0, 100, 1.0, 1.0, m), flat


def test_the_posterior_is_in_the_models_own_space():
    m = model()
    f, flat = fit(m)
    post = f.to_inference_data().posterior
    assert (post.sizes["chain"], post.sizes["draw"]) == (2, 5)
    np.testing.assert_allclose(post["mu"].values, flat[:, 3:, :2])
    np.testing.assert_allclose(post["sigma"].values, np.exp(flat[:, 3:, 2]))
    np.testing.assert_allclose(post["twice"].values, 2 * flat[:, 3:, :2])
    assert "sigma_log__" not in post


def test_stats_and_warmup_split_at_the_warmup():
    f, flat = fit(model())
    idata = f.to_inference_data()
    assert idata.sample_stats["diverging"].dtype == bool
    assert idata.sample_stats.sizes["draw"] == 5
    assert idata.warmup_sample_stats.sizes["draw"] == 3
    assert "tuning" not in idata.sample_stats
    np.testing.assert_allclose(idata.warmup_posterior["sigma"].values, np.exp(flat[:, :3, 2]))
    np.testing.assert_allclose(idata.sample_stats["lp"].values, f.stats["lp"][:, 3:])


def test_named_dims_and_coords_carry_over():
    with pm.Model(coords={"group": ["a", "b"]}) as m:
        mu = pm.Normal("mu", 0, 1, dims="group")
        pm.Normal("y", mu, 1, observed=[0.1, -0.3], dims="group")
    f, flat = fit(m)
    post = f.to_inference_data().posterior
    assert post["mu"].dims == ("chain", "draw", "group")
    assert list(post["group"].values) == ["a", "b"]
    np.testing.assert_allclose(post["mu"].values, flat[:, 3:, :])


def test_without_stats_there_is_no_sample_stats_group():
    f, _ = fit(model(), stats=False)
    assert "sample_stats" not in f.to_inference_data().groups()


def test_pooled_draws_keep_their_old_shape():
    f, _ = fit(model())
    assert f.draws.shape == (10, 3)
    assert f["sigma_log__"].shape == (10,)


def test_chains_are_seeded_apart_and_stacked(monkeypatch):
    m = model()
    names = param_names(m)
    calls = []

    async def fake_draw(handle, tw, init, warmup, draws, seed, param_names, chain=0):
        calls.append((seed, chain))
        total = warmup + draws
        return {"draws": [float(chain)] * (total * len(param_names)),
                "stats": {k: [chain] * total for k in STATS}, "ms": 2.0,
                "nParams": len(param_names)}

    monkeypatch.setattr(pymcwasm._bridge, "draw", fake_draw)
    c = Compiled(SimpleNamespace(ms=1.0, bytes=100), None, names, np.zeros(len(names)), 1.0, m)
    f = asyncio.run(c.sample(draws=4, warmup=2, seed=10, chains=3))
    assert calls == [(10, 0), (11, 1), (12, 2)]
    assert f.chain_draws.shape == (3, 4, len(names))
    assert f.warmup_draws.shape == (3, 2, len(names))
    assert f.chain_draws[2].min() == 2.0
    assert f.stats["lp"].shape == (3, 6)
    assert f.ms == 6.0
