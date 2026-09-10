"""Build nuts-rs-wasm's demo MMM into the files `bench.html` loads, and time each step.

    TAPEWASM=/path/to/tapewasm [REROLL=always] uv run --python 3.12 \\
        --with pymc==6.2.0 --with pytensor==3.2.4 --with pymc-marketing==1.1.0 \\
        --with nutpie --with scipy python bench/mmm/build.py <out dir>

`upstream/` holds the model and data fetched as the README says.
"""

import json
import os
import sys
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "build"))
sys.path.insert(0, os.path.join(here, "..", "..", "src"))
import build_artifact as ba  # noqa: E402

g = {"DATA_PATH": os.path.join(here, "upstream", "mmm_example.csv"), "SEASONALITY": True}
exec(open(os.path.join(here, "upstream", "model.py")).read(), g)
model = g["model"]

out = os.path.abspath(sys.argv[1])
os.makedirs(out, exist_ok=True)
ip = model.initial_point()

t = time.perf_counter()
ba.lower(model, os.path.join(out, "model.tape"), ip, ip)
lower_s = time.perf_counter() - t
t = time.perf_counter()
meta, consts = ba.compile_tape(os.path.join(out, "model.tape"), os.path.join(out, "model.wasm"))
emit_s = time.perf_counter() - t

scratch = [0.0] * meta["scratch_len"]
if consts:
    scratch[meta["scratch_len"] - len(consts):] = consts
names = ba.param_names(model)
start = np.concatenate([np.asarray(ip[v.name], float).ravel() for v in model.value_vars]).tolist()
with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump({"model": "mmm", "nParams": meta["n_params"], "layoutId": meta["layout_id"],
               "paramNames": names, "scratchInit": scratch, "initialPoint": start}, f)
with open(os.path.join(out, "reference.json"), "w") as f:
    json.dump(ba.reference_posterior(model, names), f)

print(json.dumps({"lower_s": round(lower_s, 3), "emit_s": round(emit_s, 3),
                  "wasm_bytes": os.path.getsize(os.path.join(out, "model.wasm")),
                  "scratch_len": meta["scratch_len"], "consts": len(consts)}))
