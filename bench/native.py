"""The CPython half: PyMC's own NUTS and nutpie on each model, into bench/results-native.json.

    uv run --python 3.13 --no-project --with pymc --with nutpie --with arviz python bench/native.py
"""

import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402

if __name__ == "__main__":
    import nutpie
    import pymc as pm
    import pytensor

    names = json.load(open(os.path.join(HERE, "models", "index.json")))
    rows = []
    for name in sys.argv[1:] or names:
        d = os.path.join(HERE, "models", name)
        source, data = open(os.path.join(d, "model.py")).read(), json.load(open(os.path.join(d, "data.json")))
        for sampler, run in (("pymc", bench.pymc_nuts), ("nutpie", bench.nutpie)):
            row = {"model": name, "env": "cpython", "sampler": sampler}
            try:
                row.update(run(bench.load(source, data)))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            print(name, sampler, {k: v for k, v in row.items() if k not in ("means", "sds")}, flush=True)
            rows.append(row)
    versions = {"python": platform.python_version(), "machine": platform.machine(),
                "pymc": pm.__version__, "pytensor": pytensor.__version__, "nutpie": nutpie.__version__}
    json.dump({"versions": versions, "rows": rows}, open(os.path.join(HERE, "results-native.json"), "w"))
