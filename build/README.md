# Regenerating the artifacts

The demo does not need this. `artifacts/` is committed because the compile step
needs Rust and a checkout of stanwasm, and asking that of someone who wants to
see the thing run would defeat the point.

```
export STANWASM=/path/to/stanwasm   # a checkout of github.com/habakan/stanwasm
uv run --with pymc --with scipy --with nutpie \
  python build_artifact.py <model> ../artifacts/<model>
```

`lower_pytensor.py` walks the PyTensor graph of `model.logp()` and writes an
instruction file; `build_artifact.py` runs stanwasm's emitter over it and records
the buffer sizes a host needs, plus nutpie's posterior for the same model pushed
back into the unconstrained space.

`python lower_pytensor.py` on its own checks every model's gradient against
`model.compile_dlogp()`, lowering at one point and evaluating at another so that
a subgraph wrongly folded into a constant shows up rather than cancelling.
