// Starting points, not a catalogue. Each is ordinary PyMC — the only rule is
// that the model ends up in a variable called `model`.
export const PRESETS = {
  "linear regression": `import numpy as np, pymc as pm

rng = np.random.default_rng(0)
x = rng.normal(size=40)
y = 1.0 + 1.8 * x + rng.normal(scale=0.4, size=40)

with pm.Model() as model:
    alpha = pm.Normal("alpha", 0, 10)
    beta = pm.Normal("beta", 0, 10)
    sigma = pm.Exponential("sigma", 1)
    pm.Normal("y", mu=alpha + beta * x, sigma=sigma, observed=y)
`,
  "eight schools (centred)": `import numpy as np, pymc as pm

y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
s = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 5)
    tau = pm.HalfCauchy("tau", 5)
    theta = pm.Normal("theta", mu, tau, shape=8)
    pm.Normal("obs", theta, s, observed=y)
`,
  "logistic regression": `import numpy as np, pymc as pm

rng = np.random.default_rng(1)
x = rng.normal(size=60)
p = 1 / (1 + np.exp(-(0.5 + 1.2 * x)))
y = (rng.uniform(size=60) < p).astype(int)

with pm.Model() as model:
    a = pm.Normal("a", 0, 5)
    b = pm.Normal("b", 0, 5)
    pm.Bernoulli("y", logit_p=a + b * x, observed=y)
`,
  "matrix regression": `import numpy as np, pymc as pm

rng = np.random.default_rng(9)
X = rng.normal(size=(80, 4))
y = X @ np.array([1.0, 2.0, 3.0, 4.0]) + rng.normal(scale=0.4, size=80)

with pm.Model() as model:
    beta = pm.Normal("beta", 0, 5, shape=4)
    sigma = pm.Exponential("sigma", 1)
    pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
`,
};
