"""Lower every posteriordb posterior that has a PyMC implementation, and check its gradient.

    POSTERIORDB=/path/to/posteriordb TAPEWASM=/path/to/tapewasm \\
        uv run --with pymc --with scipy python scripts/posteriordb.py [name ...]
"""

import json
import os
import sys
import traceback
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pymcwasm.lowering import check

DB = os.path.join(os.environ["POSTERIORDB"], "posterior_database")


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


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for name, build, has_ref in posteriors():
        if wanted and name not in wanted:
            continue
        try:
            check(name + (" [ref]" if has_ref else ""), build)
        except Exception as e:
            # The lowering refuses what it cannot represent; the reason is the result.
            last = traceback.extract_tb(e.__traceback__)[-1]
            print(f"{name}: REFUSED {type(e).__name__}: {str(e)[:160]} ({last.name}:{last.lineno})")
        sys.stdout.flush()
