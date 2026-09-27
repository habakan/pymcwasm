"""The decoder this page fits, written in PyMC and built ahead of time:

    npm install && pymcwasm-build examples/live-decoder/model.py artifacts/live-decoder --no-log-lik

z (4D) -> Dense(4->20) -> sigmoid -> Dense(20->196) -> sigmoid, N(0, 1) priors, and the 32
posteriordb MNIST digits pooled to 14x14 as a Normal likelihood at a fixed sd of 0.05.
"""

import json
import os

import numpy as np
import pymc as pm

raw = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "digits.json")))
# 2x2 average pooling, as app.js's downsample14: at 28x28 the same budget fits worse.
x = np.asarray(raw["pixels"], dtype=float).reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196)

with pm.Model() as model:
    z = pm.Normal("z", 0, 1, shape=(len(x), 4))
    w1 = pm.Normal("w1", 0, 1, shape=(4, 20))
    b1 = pm.Normal("b1", 0, 1, shape=20)
    w2 = pm.Normal("w2", 0, 1, shape=(20, 196))
    b2 = pm.Normal("b2", 0, 1, shape=196)
    mu = pm.math.sigmoid(pm.math.sigmoid(z @ w1 + b1) @ w2 + b2)
    # At sd 0.15 the N(0, 1) priors outweigh 32 images and the fit blurs.
    pm.Normal("x", mu, 0.05, observed=x)
