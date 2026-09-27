"""Copy the benchmark's posteriordb models into bench/models/<name>/, where both halves read them.

    POSTERIORDB=/path/to/posteriordb python bench/prepare.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from posteriordb import DB, load_json  # noqa: E402

# Spread over size and shape; each agrees with PyMC in scripts/posteriordb.py.
MODELS = [
    "kidiq-kidscore_momiq",
    "wells_data-wells_dist100_model",
    "arK-arK",
    "eight_schools-eight_schools_noncentered",
    "diamonds-diamonds",
    "radon_mn-radon_variable_intercept_noncentered",
    "irt_2pl-irt_2pl",
    "radon_all-radon_hierarchical_intercept_noncentered",
    "ovarian-logistic_regression_rhs",
]

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    for name in MODELS:
        post = load_json(os.path.join(DB, "posteriors", name + ".json"))
        info = load_json(os.path.join(DB, "models", "info", post["model_name"] + ".info.json"))
        out = os.path.join(HERE, "models", name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(DB, info["model_implementations"]["pymc"]["model_code"])) as f, \
                open(os.path.join(out, "model.py"), "w") as g:
            g.write(f.read())
        with open(os.path.join(out, "data.json"), "w") as g:
            json.dump(load_json(os.path.join(DB, "data", "data", post["data_name"] + ".json")), g)
    with open(os.path.join(HERE, "models", "index.json"), "w") as f:
        json.dump(MODELS, f)
    print(len(MODELS), "models into bench/models/")
