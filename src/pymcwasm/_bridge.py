"""JS interop between Pyodide and the tapewasm wasm module.

Nothing here computes anything. The PyTensor graph is lowered to a tape in
Python (`pymcwasm.lowering`), tapewasm's own wasm module compiles that tape and
runs nuts-rs over the result, and this only carries values across.

`js` and `pyodide.code` are modules a Pyodide runtime provides rather than pip
packages, so they are imported inside the functions that need them — that keeps
`import pymcwasm` working in a plain interpreter and fails only when something
actually reaches for the browser.

The site-root detection in `pystanwasm` is fuller than this: it recognises
JupyterLite, marimo and Quarto layouts. Here the path is given, or defaults to
the shape those layouts serve a package from.
"""

DEFAULT_TAPEWASM_PATH = "/files/tapewasm/pkg/tapewasm.js"

_loaded = {}


async def load(tapewasm_path=DEFAULT_TAPEWASM_PATH):
    """Import tapewasm's JS module once per path and initialise its wasm."""
    if tapewasm_path in _loaded:
        return _loaded[tapewasm_path]

    from pyodide.code import run_js

    # `import()` inside run_js resolves against Pyodide's own module URL, which
    # is the CDN — so a page-relative path has to be made absolute first.
    tw = await run_js(
        f"""
        (async () => {{
            const base = globalThis.location ? globalThis.location.href : undefined;
            const url = base ? new URL("{tapewasm_path}", base).href : "{tapewasm_path}";
            const m = await import(url);
            await m.default();
            return m;
        }})()
        """
    )
    _loaded[tapewasm_path] = tw
    return tw


# The emitted module imports these; only the ones a tape reaches are asked for,
# but a module asks for what it asks for. lgamma/digamma/phi are the series
# tapewasm's own tests use.
_MATH_JS = """
({
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
  tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  lgamma: (x) => {
    let z = x, r = 0;
    while (z < 10) { r -= Math.log(z); z += 1; }
    const i = 1 / z, i2 = i * i;
    return r + (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI)
      + i * (1 / 12 + i2 * (-1 / 360 + i2 / 1260));
  },
  digamma: (x) => {
    let z = x, r = 0;
    while (z < 6) { r -= 1 / z; z += 1; }
    const i = 1 / z, i2 = i * i;
    return r + Math.log(z) - 0.5 * i - i2 * (1 / 12 - i2 * (1 / 120 - i2 / 252));
  },
  phi: (x) => {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
      + t * (-1.821255978 + t * 1.330274429))));
    const c = 1 - (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x) * p;
    return x >= 0 ? c : 1 - c;
  },
})
"""


async def compile(tape, tapewasm_path=DEFAULT_TAPEWASM_PATH):
    """Turn a tape into a bound module. Returns a JS handle and what it took."""
    from pyodide.code import run_js

    tw = await load(tapewasm_path)
    build = run_js(
        """
        (async (tw, tape, mathSrc) => {
            const t0 = performance.now();
            const built = tw.compileTape(tape);
            const aot = await WebAssembly.instantiate(built.wasm, {
                tapewasm: { memory: tw.sharedMemory() },
                Math: eval(mathSrc),
            });
            return { built, exports: aot.instance.exports,
                     ms: performance.now() - t0, bytes: built.wasm.length };
        })
        """
    )
    return await build(tw, tape, _MATH_JS), tw


async def draw(handle, tw, init, warmup, draws, seed, param_names):
    """Sample a module compiled earlier.

    `setAotExports` binds one module per page, so this re-binds before every
    run — two models compiled in one page would otherwise take each other's
    buffers, which the layout id refuses rather than silently mixing.
    """
    from pyodide.code import run_js

    sampler = run_js(
        """
        ((tw, h, init, warmup, draws, seed, names) => {
            tw.setAotExports(h.exports);
            const s = new tw.AotSampler(
                h.built.nParams, h.built.scratchInit, h.built.layoutId, names,
            );
            const t0 = performance.now();
            const flat = s.sample(new Float64Array(init), warmup, draws, BigInt(seed));
            return { draws: Array.from(flat), ms: performance.now() - t0,
                     nParams: h.built.nParams };
        })
        """
    )
    return sampler(
        tw, handle, list(init), int(warmup), int(draws), int(seed), list(param_names),
    ).to_py()
