#!/usr/bin/env bash
# Builds bartrs for Pyodide 0.29.2 into wheels/, from BARTRS=<checkout> or else the fork's
# commit whose PGBART takes compile_kwargs["mode"]. pyodide-build fetches Emscripten.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${BARTRS:-}" ]; then
  bartrs="$(cd "$BARTRS" && pwd)"
else
  bartrs="$(mktemp -d)"
  git -C "$bartrs" init -q
  git -C "$bartrs" fetch -q --depth 1 https://github.com/habakan/bartrs a8959fd0e3f1a4a07031605e66d689a29a9858a8
  git -C "$bartrs" checkout -q FETCH_HEAD
fi

tools="$(mktemp -d)"
uv venv -q -p 3.13 "$tools"
VIRTUAL_ENV="$tools" uv pip install -q pyodide-build
pyodide="$tools/bin/pyodide"
"$pyodide" xbuildenv install 0.29.2
"$pyodide" xbuildenv install-emscripten

# Pyodide's Rust unwinds with wasm exceptions, so its sysroot replaces rustup's for this nightly.
toolchain="$("$pyodide" config get rust_toolchain)"
url="$("$pyodide" config get rust_emscripten_target_url)"
rustup toolchain install "$toolchain" --profile minimal
rustlib="$(dirname "$(dirname "$(rustup which --toolchain "$toolchain" rustc)")")/lib/rustlib"
if [ "$(cat "$rustlib/wasm32-unknown-emscripten_install-url.txt" 2>/dev/null)" != "$url" ]; then
  rm -rf "$rustlib/wasm32-unknown-emscripten"
  curl -sL "$url" | tar xj -C "$rustlib"
  printf %s "$url" > "$rustlib/wasm32-unknown-emscripten_install-url.txt"
fi
source "$("$pyodide" config get emscripten_dir)/../../emsdk_env.sh" > /dev/null 2>&1

out="$(mktemp -d)"
(cd "$bartrs" && RUSTUP_TOOLCHAIN="$toolchain" CARGO_PROFILE_RELEASE_DEBUG=false "$pyodide" build -o "$out")
# pyodide-build tags it pyemscripten_2025_0 (PEP 783); Pyodide 0.29's micropip wants pyodide_2025_0.
mkdir -p "$here/wheels"
for whl in "$out"/*pyemscripten_2025_0_wasm32.whl; do
  cp "$whl" "$here/wheels/$(basename "${whl/pyemscripten_2025_0/pyodide_2025_0}")"
done
ls -l "$here/wheels"
