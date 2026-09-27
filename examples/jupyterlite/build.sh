#!/usr/bin/env bash
# The wheel and npm's tapewasm go into files/, which JupyterLite serves as /files/.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$here/../.."
rm -rf "$here/files/tapewasm" "$here/files/"pymcwasm-*.whl
# Built beside, not into, files/: uv writes a `*` .gitignore into its output directory.
wheels="$(mktemp -d)"
uv build -q --wheel -o "$wheels" "$repo"
cp "$wheels"/pymcwasm-*.whl "$here/files/"
cp -R "$repo/node_modules/tapewasm" "$here/files/tapewasm"
rm -f "$here/files/tapewasm/pkg/.gitignore"   # wasm-pack's `*` would hide the wasm
cd "$here"
uv run -q --no-project --with jupyterlite-core --with jupyterlite-pyodide-kernel --with jupyter-server \
  jupyter lite build --output-dir _output
