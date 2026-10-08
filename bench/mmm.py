"""nuts-rs-wasm's demo MMM under nutpie and under tapewasm in V8.

    uv run --python 3.12 --no-project --with pymc-marketing --with nutpie --with-editable . \\
      python bench/mmm.py native /path/to/nuts-rs-wasm
    node bench/mmm.mjs
    node bench/mmm-browser.mjs
    uv run --python 3.12 --no-project --with pymc-marketing --with nutpie --with-editable . \\
      python bench/mmm.py table

The model is nuts-rs-wasm's `examples/mmm/model.py` on its CSV, unchanged, sampled as
its tapewasm backend PR was: 2 chains of 750 warmup and 500 draws, target_accept 0.9.
Both samplers run their chains one after the other. nutpie's time includes building
its InferenceData; tapewasm's is `sample()` alone, with the statistics read from a
second, untimed `sampleWithStats()` on the same seed. nutpie jitters each chain's
start; tapewasm starts both chains at the build's initial point.
"""
import json, os, platform, runpy, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "models", "mmm")
FITS, TUNE, DRAWS, CHAINS = 10, 750, 500, 2


def min_bulk_ess(draws):  # (chain, draw, param), unconstrained, in the sampler's order
    from arviz_stats.base import array_stats
    return float(np.min(array_stats.ess(np.asarray(draws), chain_axis=0, draw_axis=1, method="bulk")))


def native(nrw):
    import shutil

    import arviz_stats, nutpie, pymc, pymc_marketing
    from pymcwasm.build import build

    ns = runpy.run_path(os.path.join(nrw, "examples/mmm/model.py"),
                        init_globals={"DATA_PATH": os.path.join(nrw, "examples/mmm/mmm_example.csv")})
    model = ns["model"]
    for reroll in ("auto", "never"):
        build(model, os.path.join(OUT, reroll), log_lik=False, expand=False, reroll=reroll)
    # The precompiled page's copy. Its expansion needs tapewasm 0.3.4; without one the page
    # times sampling alone.
    try:
        build(model, os.path.join(OUT, "browser"), log_lik=False, expand=True, reroll="never")
    except RuntimeError as e:
        if "tapewasm could not compile" not in str(e):
            raise
        print("expand.wasm not built:", str(e).splitlines()[-1])
        build(model, os.path.join(OUT, "browser"), log_lik=False, expand=False, reroll="never")
    for f in ("model.py", "mmm_example.csv"):
        shutil.copy(os.path.join(nrw, "examples/mmm", f), OUT)
    compiled = nutpie.compile_pymc_model(model)
    runs = []
    for adaptation in ("diag", "draw_diag"):
        fits = []
        for s in range(FITS):
            t = time.perf_counter()
            idata = nutpie.sample(compiled, draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, seed=s + 1,
                                  target_accept=0.9, adaptation=adaptation, save_warmup=True,
                                  store_unconstrained=True, progress_bar=False)
            secs = time.perf_counter() - t
            post = np.asarray(idata.sample_stats["unconstrained_draw"])
            fits.append({"seconds": secs, "evals": int(idata.warmup_sample_stats["n_steps"].sum()
                                                       + idata.sample_stats["n_steps"].sum()),
                         "step_size": float(idata.sample_stats["step_size"].mean()),
                         "divergences": int(idata.sample_stats["diverging"].sum()),
                         "draws": post.ravel().tolist()})
        runs.append({"adaptation": adaptation, "fits": fits})
    versions = {"python": platform.python_version(), "machine": platform.machine(), "pymc": pymc.__version__,
                "pymc-marketing": pymc_marketing.__version__, "nutpie": nutpie.__version__,
                "arviz-stats": arviz_stats.__version__}
    json.dump({"versions": versions, "nParams": int(post.shape[-1]), "tune": TUNE, "draws": DRAWS,
               "chains": CHAINS, "runs": runs}, open(os.path.join(OUT, "nutpie.json"), "w"))


def table():
    import statistics as st

    nat = json.load(open(os.path.join(OUT, "nutpie.json")))
    tw = json.load(open(os.path.join(OUT, "tapewasm.json")))
    rows = []
    for src, label in ((nat, lambda r: f"CPython · nutpie · adaptation {r['adaptation']}"),
                       (tw, lambda r: f"V8 · tapewasm · reroll {r['reroll']} · grad-based metric "
                                      f"{'on' if r['gradBased'] else 'off'}")):
        shape = (src["chains"], src["draws"], src["nParams"])
        for run in src["runs"]:
            for f in run["fits"]:
                f["draws"] = np.asarray(f["draws"]).reshape(shape)
                f["min_ess"] = min_bulk_ess(f["draws"])
            rows.append((label(run), run["fits"]))

    # Every fit of nutpie's default, pooled, is what the other rows' means are measured against.
    ref = np.concatenate([f["draws"].reshape(-1, nat["nParams"]) for f in rows[0][1]])
    mean, sd = ref.mean(0), ref.std(0)

    def gap(fits):
        pooled = np.concatenate([f["draws"].reshape(-1, nat["nParams"]) for f in fits])
        return float(np.max(np.abs(pooled.mean(0) - mean) / sd))

    med = lambda fits, f: st.median(f(x) for x in fits)
    lines = ["# nuts-rs-wasm's demo MMM", "", "Generated by `bench/mmm.py table`; see its docstring.", "",
             f"- CPython: {nat['versions']}", f"- V8: {tw['versions']}", "",
             f"Median of {FITS} fits of {CHAINS} chains × ({TUNE} warmup + {DRAWS} draws), target_accept 0.9. "
             "The last column is the worst distance of a row's pooled posterior means from nutpie's default, "
             "in nutpie's sd, over the unconstrained parameters.",
             "", "| sampler | sample s | gradient evals | step size | min bulk ESS | ESS/s | µs per eval "
             "| divergences | means vs nutpie |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, fits in rows:
        lines.append(
            f"| {name} | {med(fits, lambda x: x['seconds']):.2f} | {med(fits, lambda x: x['evals']):,.0f} "
            f"| {med(fits, lambda x: x['step_size']):.3f} | {med(fits, lambda x: x['min_ess']):.0f} "
            f"| {med(fits, lambda x: x['min_ess'] / x['seconds']):.1f} "
            f"| {med(fits, lambda x: x['seconds'] / x['evals'] * 1e6):.1f} "
            f"| {sum(x['divergences'] for x in fits)} | {gap(fits):.2f} sd |")
    browser = os.path.join(OUT, "browser.json")
    if os.path.exists(browser):
        lines += browser_table(json.load(open(browser)), tw["nParams"], mean, sd)
    open(os.path.join(HERE, "RESULTS-mmm.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def browser_table(b, n, mean, sd):
    """The three browser runs of bench/mmm-browser.mjs, beside each other."""
    import statistics as st

    def ours(r):
        fits = r["fits"]
        for f in fits:
            f["min_ess"] = min_bulk_ess(np.asarray(f["draws"]).reshape(CHAINS, DRAWS, n))
        pooled = np.concatenate([np.asarray(f["draws"]).reshape(-1, n) for f in fits])
        return {"secs": [f["sampling_seconds"] for f in fits], "ess_s": [f["min_ess"] / f["sampling_seconds"] for f in fits],
                "us": [f["sampling_seconds"] / f["evals"] * 1e6 for f in fits], "div": sum(f["divergences"] for f in fits),
                "gap": f"{float(np.max(np.abs(pooled.mean(0) - mean) / sd)):.2f} sd"}

    runs = b["numba"]["runs"]
    numba = {"secs": [r["sampling_seconds"] for r in runs], "ess_s": [r["min_ess_per_second"] for r in runs],
             "us": [r["sampling_seconds"] / r["logp_evaluations"] * 1e6 for r in runs],
             "div": sum(r["divergences"] for r in runs), "gap": "—"}
    ip, pc, cold = b["inpage"], b["precompiled"], b["numbaCold"]
    rows = [
        ("nuts-rs-wasm · Numba, compiled in the page (hosted demo, reused)", b["numba"]["network"], numba,
         f"{cold['seconds_to_prepare']:.1f} + {b['numba']['preparation_wall_seconds']['reused']:.1f}"),
        ("pymcwasm · tapewasm, compiled in the page (Pyodide)", ip["network"], ours(ip),
         f"{ip['loaded_seconds']:.1f} + {ip['model_seconds'] + ip['compile_seconds']:.1f}"),
        ("pymcwasm · tapewasm, compiled beforehand", pc["network"], ours(pc), f"{pc['readySeconds']:.2f}"),
    ]
    med = st.median
    out = ["", "## In a browser", "",
           f"{b['browser']}, `node bench/mmm-browser.mjs`, one page each from a fresh context; the hosted "
           f"page's source had sha256 {b['hostedSha256'][:12]}, checked for the fits described here. Seeds 42 "
           "to 442, 2 chains × (750 warmup + 500 draws), target_accept 0.9, the gradient-based metric "
           "estimate on (nuts-rs's default there, set here); tapewasm with `reroll never`. The hosted "
           "page jitters each chain's start, the two tapewasm pages do not.",
           "",
           "Ready is loading (runtime, packages and imports), then preparing the model (building it and "
           "compiling its density). The hosted page's two parts come from two loads of it, the first stopped "
           "once preparing starts; compiled beforehand, ready is only instantiating the modules.",
           "",
           "Sample s is what each path times: the Numba adapter's `sampling_seconds` (warmup, sampling, "
           "expansion and Arrow), `pymcwasm`'s `Compiled.sample` in the page (whose `sampleWithStats` "
           "evaluates the density once more per draw, about 3%, and converts the draws and statistics), "
           "and `sample()` compiled beforehand, with one `expand.wasm` evaluate per draw "
           f"{'included' if pc.get('expansion') else 'left out: its build needs tapewasm 0.3.4'}. ESS is "
           "min bulk ESS over the seven free variables, ArviZ's on the hosted page and arviz-stats' here; "
           "being rank-based it is the same in either space. Downloaded is bytes as sent: the CDNs "
           "compress, the local server that serves this repository's files does not.",
           "",
           "The two tapewasm rows do not run one module: the yearly Fourier features, `sin` and `cos` "
           "of the dates, differ in the last bit between Pyodide's libm and the build machine's, so "
           "30 of the tape's constants do and so does the layout id. Each module then follows its own "
           "trajectory, which is as much of their ESS gap as the table can explain.",
           "", "| sampler | downloaded | requests | ready s | sample s | ESS/s | µs per eval | divergences "
           "| means vs nutpie |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, net, r, ready in rows:
        out.append(f"| {name} | {net['MB']:.1f} MB | {net['requests']} | {ready} | {med(r['secs']):.2f} "
                   f"| {med(r['ess_s']):.1f} | {med(r['us']):.1f} | {r['div']} | {r['gap']} |")
    return out


if __name__ == "__main__":
    {"native": lambda: native(sys.argv[2]), "table": table}[sys.argv[1]]()
